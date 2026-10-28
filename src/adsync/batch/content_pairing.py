"""Pair season recordings by audio content when filename episode numbers are wrong.

A filename pair is trusted once its audio passes the distributed identity
check. Files whose pair fails are re-paired only by a unique, mutual content
match; duplicates and ties are reported instead of guessed. When a re-paired
video and AD carry different episode numbers, "previously on" recaps reveal the
story order and decide whose numbering is correct.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass, field
import logging
import os
from pathlib import Path

import numpy as np

from adsync.batch.discovery import _TAG, DiscoveryIssue, EpisodeKey, EpisodePair
from adsync.config import SyncConfig
from adsync.quality import FingerprintFeatures, assess_content_identity

LABEL_SOURCES = ("auto", "video", "ad")
_RECAP_WINDOW_SEC = 300.0
_RECAP_WINDOW_FRACTION = 0.25
_CLIP_OFFSET_BIN_SEC = 0.1
_DENSE_MATCHES_PER_SEC = 12   # identical audio gives ~35; reused score gives a few
_MIN_CLIP_SEC = 2
_MIN_RECAP_CLIPS = 2          # a montage of scenes, not one continuous shared sequence
_SHARED_EPISODES = 3          # opening audio matching this many episodes is titles, not a recap

log = logging.getLogger("adsync")


@dataclass(frozen=True)
class Recording:
    path: Path
    key: EpisodeKey
    features: FingerprintFeatures


@dataclass(frozen=True)
class ContentMatch:
    key: EpisodeKey
    video: Recording
    ad: Recording
    coverage: float
    confirmed: bool
    changed: bool
    label_source: str


@dataclass
class ContentPairing:
    pairs: list[ContentMatch] = field(default_factory=list)
    issues: list[DiscoveryIssue] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@contextmanager
def _quiet_log():
    # Reading and cross-matching every file would repeat per-file probe and fingerprint chatter.
    previous = log.level
    log.setLevel(max(previous, logging.WARNING))
    try:
        yield
    finally:
        log.setLevel(previous)


def _identity(video: Recording, ad: Recording) -> dict:
    from adsync.align.fingerprint import _match, _offset_spans

    t_ad, t_vid = _match(video.features.hashes, video.features.times, ad.features.hashes, ad.features.times)
    evidence = _offset_spans(t_ad, t_vid, ad.features.duration)
    return assess_content_identity(evidence, ad.features.duration, video_duration=video.features.duration)


def relabel_stem(stem: str, key: EpisodeKey) -> str:
    """Replace the episode tag in a filename stem, or prefix one when absent."""
    tag = f"S{key.season:02d}E{key.episode:02d}"
    match = _TAG.search(stem)
    if match is None:
        return f"{tag} - {stem}"
    return stem[:match.start()] + tag + stem[match.end():]


def _replayed_clips(later: Recording, earlier: Recording) -> list[tuple[int, int, int]]:
    """Dense clips of ``earlier``'s story in ``later``'s opening, as (start s, end s, offset bin)."""
    from adsync.align.fingerprint import _match

    lf, ef = later.features, earlier.features
    opening = lf.times < min(_RECAP_WINDOW_SEC, _RECAP_WINDOW_FRACTION * lf.duration)
    t_later, t_earlier = _match(ef.hashes, ef.times, lf.hashes[opening], lf.times[opening])
    # Recaps replay story scenes; another episode's own opening holds the shared titles.
    story = t_earlier >= min(_RECAP_WINDOW_SEC, _RECAP_WINDOW_FRACTION * ef.duration)
    if not story.any():
        return []
    bins = np.round((t_earlier[story] - t_later[story]) / _CLIP_OFFSET_BIN_SEC).astype(np.int64)
    seconds = np.floor(t_later[story]).astype(np.int64)
    # Each match also counts for neighbouring offset bins, so quantization edges agree.
    cells, counts = np.unique(np.stack([np.concatenate([bins - 1, bins, bins + 1]), np.tile(seconds, 3)]),
                              axis=1, return_counts=True)
    dense = cells[:, counts >= _DENSE_MATCHES_PER_SEC]
    clips = []
    for offset in np.unique(dense[0]):
        secs = np.sort(dense[1, dense[0] == offset])
        for run in np.split(secs, np.flatnonzero(np.diff(secs) != 1) + 1):
            if len(run) >= _MIN_CLIP_SEC:
                clips.append((int(run[0]), int(run[-1]) + 1, int(offset)))
    merged: list[tuple[int, int, int]] = []
    for start, end, offset in sorted(clips, key=lambda clip: (clip[2], clip[0])):
        if merged and offset - merged[-1][2] <= 2 and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]), merged[-1][2])
        else:
            merged.append((start, end, offset))
    return merged


def _story_order_votes(matches: Sequence[ContentMatch]) -> tuple[int, int]:
    """Count recaps whose story order agrees only with the video or only with the AD numbering."""
    n = len(matches)
    clips = {(i, j): _replayed_clips(matches[i].video, matches[j].video)
             for i in range(n) for j in range(n) if i != j}
    video_votes = ad_votes = 0
    for i, later in enumerate(matches):
        owners = Counter(second for j in range(n) if j != i
                         for second in {s for start, end, _ in clips[i, j] for s in range(start, end)})
        shared = {second for second, count in owners.items() if count >= _SHARED_EPISODES}
        for j, earlier in enumerate(matches):
            if i == j or not (later.changed or earlier.changed):
                continue
            recap = [clip for clip in clips[i, j]
                     if sum(s not in shared for s in range(clip[0], clip[1])) >= _MIN_CLIP_SEC]
            if len(recap) < _MIN_RECAP_CLIPS:
                continue
            video_order = earlier.video.key < later.video.key
            ad_order = earlier.ad.key < later.ad.key
            if video_order != ad_order:
                if video_order:
                    video_votes += 1
                else:
                    ad_votes += 1
    return video_votes, ad_votes


def resolve_content_pairs(
    pairs: Sequence[tuple[Recording, Recording]], *, label_from: str = "auto",
) -> ContentPairing:
    """Confirm filename pairs by audio, repair unique swaps, and report the rest.

    ``pairs`` are filename-matched (video, AD) recordings. A failing pair whose
    files match nothing else is kept unconfirmed: speed-changed AD audio cannot
    be fingerprinted before correction, so the per-episode check decides.
    """
    if label_from not in LABEL_SOURCES:
        raise ValueError(f"label_from must be one of {', '.join(LABEL_SOURCES)}")
    videos = [video for video, _ in pairs]
    ads = [ad for _, ad in pairs]
    n = len(pairs)
    result = ContentPairing()
    coverage: dict[tuple[int, int], float] = {}
    passed: dict[tuple[int, int], bool] = {}

    def passes(i: int, j: int) -> bool:
        if (i, j) not in passed:
            identity = _identity(videos[i], ads[j])
            coverage[i, j], passed[i, j] = identity["coverage"], identity["status"] == "pass"
        return passed[i, j]

    with _quiet_log():
        confirmed = {i for i in range(n) if passes(i, i)}
        unresolved = [i for i in range(n) if i not in confirmed]
        video_hits = {i: [j for j in range(n) if j != i and passes(i, j)] for i in unresolved}
        ad_hits = {j: [i for i in range(n) if i != j and passes(i, j)] for j in unresolved}

    repaired = {i: video_hits[i][0] for i in unresolved
                if len(video_hits[i]) == 1 and video_hits[i][0] in ad_hits and ad_hits[video_hits[i][0]] == [i]}
    used_videos, used_ads = set(confirmed) | set(repaired), set(confirmed) | set(repaired.values())
    kept = {i for i in unresolved if i not in used_videos and i not in used_ads
            and not video_hits[i] and not ad_hits[i]}

    matches: list[ContentMatch] = []
    for i in sorted(confirmed | kept):
        matches.append(ContentMatch(videos[i].key, videos[i], ads[i], coverage[i, i],
                                    i in confirmed, False, "filenames"))
        if i in kept:
            result.notes.append(
                f"{videos[i].key.label}: the audio did not confirm this filename pairing, and neither file matches "
                "another episode. It may differ in speed or mix; the per-episode identity check will decide."
            )
    swapped = [ContentMatch(videos[i].key, videos[i], ads[j], coverage[i, j], True, True, "video")
               for i, j in sorted(repaired.items())]
    source = label_from
    if swapped and label_from == "auto":
        source = _auto_label_source(matches + swapped, result.notes)
    for match in sorted(swapped, key=lambda m: m.ad.key if source == "ad" else m.video.key):
        key = match.ad.key if source == "ad" else match.video.key
        matches.append(ContentMatch(key, match.video, match.ad, match.coverage, True, True, source))
        result.notes.append(
            f"{key.label}: {match.video.path.name} plays the episode in {match.ad.path.name} "
            f"({match.coverage:.0%} of the episode matched); numbered from the {'AD' if source == 'ad' else 'video'} filename."
        )
    result.pairs = sorted(matches, key=lambda m: m.key)
    result.issues = _unresolved_issues(videos, ads, confirmed, repaired, kept, video_hits, ad_hits)
    return result


def _auto_label_source(matches: Sequence[ContentMatch], notes: list[str]) -> str:
    with _quiet_log():
        video_votes, ad_votes = _story_order_votes(matches)
    if ad_votes and not video_votes:
        notes.append(f"Recaps replay earlier episodes in story order: {ad_votes} recap(s) fit only the AD "
                     "numbering, so the swapped episodes are numbered from the AD filenames.")
        return "ad"
    if video_votes and not ad_votes:
        notes.append(f"Recaps replay earlier episodes in story order: {video_votes} recap(s) fit only the "
                     "video numbering, so the swapped episodes keep their video filenames' numbers.")
        return "video"
    notes.append("Recap evidence does not settle which filenames are numbered correctly; the swapped episodes keep "
                 "their video filenames' numbers. Use --label-from ad to number them from the AD filenames instead.")
    return "video"


def _unresolved_issues(videos, ads, confirmed, repaired, kept, video_hits, ad_hits) -> list[DiscoveryIssue]:
    used_videos = set(confirmed) | set(repaired) | kept
    used_ads = set(confirmed) | set(repaired.values()) | kept
    issues = []
    for i in range(len(videos)):
        reasons, paths = [], []
        if i not in used_videos:
            paths.append(videos[i].path)
            reasons.append(_video_reason(videos[i], video_hits[i], ads, used_ads))
        if i not in used_ads:
            paths.append(ads[i].path)
            reasons.append(_ad_reason(ads[i], ad_hits[i], videos, used_videos, video_hits))
        if reasons:
            issues.append(DiscoveryIssue(videos[i].key, "; ".join(reasons), tuple(paths)))
    return issues


def _video_reason(video: Recording, hits: list[int], ads, used_ads) -> str:
    if not hits:
        return f"No AD track in this selection contains the audio of {video.path.name}"
    if len(hits) > 1:
        names = ", ".join(ads[j].path.name for j in hits)
        return f"{video.path.name} matches more than one AD track ({names})"
    other = ads[hits[0]]
    if hits[0] in used_ads:
        return f"{video.path.name} holds the same episode as {other.key.label}, already paired (duplicate video)"
    return f"{video.path.name} matches {other.path.name}, which also matches another video"


def _ad_reason(ad: Recording, hits: list[int], videos, used_videos, video_hits) -> str:
    if not hits:
        return f"{ad.path.name} matches no video in this selection"
    if len(hits) > 1:
        names = ", ".join(videos[i].path.name for i in hits)
        return f"{ad.path.name} matches more than one video ({names})"
    other = videos[hits[0]]
    if hits[0] in used_videos:
        return f"{ad.path.name} holds the same episode as {other.key.label}, already paired (duplicate recording)"
    return f"{ad.path.name} matches {other.path.name}, which matches more than one AD track"


def load_recording_features(path: Path, config: SyncConfig, *, video: bool, threads: int = 1) -> FingerprintFeatures:
    """Fingerprint the same analysis audio alignment uses, sharing its cache entries."""
    from adsync.features.analysis_audio import ad_analysis_stream, analysis_stream, load_analysis_audio
    from adsync.media.probe import probe
    from adsync.quality import fingerprint_features

    info = probe(path)
    stream = analysis_stream(info, config) if video else ad_analysis_stream(info, config)
    audio = load_analysis_audio(path, info, stream, config, threads=threads)
    return fingerprint_features(audio, config.analysis_sr)


def _parallel_reads(files: int) -> int:
    import psutil

    # Landmark extraction holds roughly 1 GiB of spectrogram per hour of audio.
    by_memory = int(psutil.virtual_memory().available // (2 * 1024 ** 3))
    return max(1, min(8, files, (os.cpu_count() or 2) // 2, by_memory))


def check_season_content(
    pairs: Sequence[EpisodePair], config: SyncConfig, *, label_from: str = "auto",
    echo: Callable[[str], None] = print,
) -> tuple[list[EpisodePair], list[DiscoveryIssue], list[str], list[dict]]:
    """Confirm or repair filename pairs by audio; return pairs, issues, notes and records.

    A pair with a file that cannot be decoded keeps its filename pairing.
    """
    files = [(index, video, path) for index, pair in enumerate(pairs)
             for video, path in ((True, pair.video_path), (False, pair.ad_path))]
    workers = _parallel_reads(len(files))
    echo(f"Content check: comparing the audio of {len(pairs)} episode pair(s), {workers} file(s) at a time.")
    features: dict[tuple[int, bool], FingerprintFeatures] = {}
    notes: list[str] = []
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        with _quiet_log():
            futures = {pool.submit(load_recording_features, path, config, video=video,
                                   threads=max(1, (os.cpu_count() or 2) // workers)): (index, video, path)
                       for index, video, path in files}
            pending, done = set(futures), 0
            while pending:
                # A bounded wait keeps Ctrl+C responsive on Windows.
                finished, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                for future in finished:
                    index, video, path = futures[future]
                    try:
                        features[index, video] = future.result()
                    except Exception as exc:
                        notes.append(f"Content check could not read {path.name} ({exc}); its pair relies on filenames.")
                    done += 1
                    echo(f"Content check: read {done} of {len(files)} files.")
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown()

    readable = [i for i in range(len(pairs)) if (i, True) in features and (i, False) in features]
    resolved = resolve_content_pairs(
        [(Recording(pairs[i].video_path, pairs[i].key, features[i, True]),
          Recording(pairs[i].ad_path, pairs[i].key, features[i, False])) for i in readable],
        label_from=label_from,
    )
    output_roots = {pair.video_path: pair.output_path.parent.parent for pair in pairs}
    corrected = [pairs[i] for i in range(len(pairs)) if i not in set(readable)]
    records = []
    for match in resolved.pairs:
        stem = match.video.path.stem
        if match.key != match.video.key:
            stem = relabel_stem(stem, match.key)
        output = output_roots[match.video.path] / f"Season {match.key.season:02d}" / f"{stem}.AD.mkv"
        corrected.append(EpisodePair(match.key, match.video.path, match.ad.path, output))
        records.append({"episode": match.key.label, "video": str(match.video.path), "ad": str(match.ad.path),
                        "coverage": round(match.coverage, 4), "confirmed": match.confirmed,
                        "changed": match.changed, "label_source": match.label_source})
    confirmed = sum(match.confirmed and not match.changed for match in resolved.pairs)
    repaired = sum(match.changed for match in resolved.pairs)
    echo(f"Content check: {confirmed} of {len(readable)} readable pair(s) confirmed by their audio, {repaired} re-paired.")
    return sorted(corrected, key=lambda pair: pair.key), resolved.issues, notes + resolved.notes, records
