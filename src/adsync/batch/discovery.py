"""Read-only, deterministic pairing of numbered videos and AD tracks."""

from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

_VIDEO_EXTENSIONS = {".mkv", ".mp4", ".m4v", ".avi", ".mov", ".webm", ".ts", ".m2ts", ".mpg", ".mpeg", ".wmv"}
_AD_EXTENSIONS = _VIDEO_EXTENSIONS | {".m4a", ".mka", ".mp3", ".wav", ".flac", ".opus", ".ac3", ".eac3", ".aac", ".ogg", ".oga", ".aiff", ".aif", ".wma", ".dts", ".mp2"}
_TAG = re.compile(
    r"(?<![a-z0-9])(?:s(?P<s>\d{1,3})[ ._-]*e(?P<e>\d{1,3})"
    r"|(?P<xs>\d{1,2})x(?P<xe>\d{2,3}))(?=$|[^a-z0-9]|[ex]\d)", re.I,
)
_DOTTED_TAG = re.compile(r"^(\d{1,2})\.(\d{2})[ \t]+(?:-[ \t]+)?[^\W\d_]", re.I)
_CONTINUATION = re.compile(
    r"^(?P<sep>[ ._-]*)(?:[ex](?P<label>\d{1,3})"
    r"|(?P<range>[-+&])[ ._-]*(?:e|x)?(?P<bare>\d{1,3}))"
    r"(?=$|[^a-z0-9]|[ex]\d)", re.I,
)
_EPISODE = re.compile(
    r"(?<![a-z0-9])(?:episode|ep|ad|e)[ ._-]*(\d{1,3})(?=$|[^a-z0-9]|e\d)", re.I,
)
_FOLDER_SEASON = re.compile(r"(?<![a-z0-9])(?:season[ ._-]*|s)(\d{1,3})(?![a-z0-9])", re.I)
_INTEGER = re.compile(r"(?<![a-z0-9])\d+(?![a-z0-9])", re.I)
_VERSION = re.compile(r"(?:^|[ ._-])(?:v|ver|version)[ ._-]*\d+", re.I)


@dataclass(frozen=True, order=True)
class EpisodeKey:
    season: int
    episode: int

    @property
    def label(self) -> str:
        return f"S{self.season:02d}E{self.episode:02d}"

    def __str__(self) -> str:
        return self.label


@dataclass(frozen=True)
class EpisodePair:
    key: EpisodeKey
    video_path: Path
    ad_path: Path
    output_path: Path


@dataclass(frozen=True)
class DiscoveryIssue:
    key: EpisodeKey | None
    reason: str
    paths: tuple[Path, ...]


@dataclass
class DiscoveryResult:
    pairs: list[EpisodePair] = field(default_factory=list)
    issues: list[DiscoveryIssue] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class _Evidence:
    path: Path
    keys: set[EpisodeKey]
    numbers: set[int]
    pattern: str | None
    season_tags: set[int]
    reason: str | None = None


def _sort_path(path: Path) -> tuple[str, str]:
    return str(path).casefold(), str(path)


def _absolute(path: str | Path) -> Path:
    return Path(os.path.expandvars(path)).expanduser().resolve()


def _scan(root: Path, extensions: set[str], exclusions: tuple[Path, ...], issues: list[DiscoveryIssue], *, skip_ad_outputs: bool) -> list[Path]:
    found: list[Path] = []

    def excluded(path: Path) -> bool:
        return any(path.is_relative_to(folder) for folder in exclusions)

    def unreadable(exc: OSError) -> None:
        issues.append(DiscoveryIssue(None, f"Cannot read directory: {exc}", (Path(exc.filename) if exc.filename else root,)))

    if excluded(root):
        return found
    for current, directories, filenames in os.walk(root, topdown=True, followlinks=False, onerror=unreadable):
        folder = Path(current)
        directories[:] = sorted(
            (name for name in directories
             if not (folder / name).is_symlink()
             and not getattr(folder / name, "is_junction", lambda: False)()
             and not excluded(folder / name)), key=str.casefold,
        )
        for name in sorted(filenames, key=str.casefold):
            path = folder / name
            if excluded(path) or path.suffix.casefold() not in extensions:
                continue
            if path.stem.casefold().endswith((".synced", ".prepped")):
                continue
            if skip_ad_outputs and path.stem.casefold().endswith(".ad"):
                continue
            if path.is_symlink():
                continue
            found.append(path)
    return sorted(found, key=_sort_path)


def _continuations(stem: str, end: int, initial: int) -> set[int]:
    numbers = {initial}
    previous = initial
    while match := _CONTINUATION.match(stem[end:]):
        number = int(match.group("label") or match.group("bare"))
        if match.group("range") == "-" or "-" in match.group("sep"):
            numbers.update(range(min(previous, number), max(previous, number) + 1))
        else:
            numbers.add(number)
        previous = number
        end += match.end()
    return numbers


def _numeric(stem: str) -> tuple[set[int], str | None]:
    # S.E labels are interpreted separately, with matching season-folder
    # evidence. Never fall back to reading just the integer before a decimal.
    if re.match(r"^\d+\.\d", stem):
        return set(), None
    matches = list(_EPISODE.finditer(stem))
    if matches:
        numbers: set[int] = set()
        for match in matches:
            numbers.update(_continuations(stem, match.end(), int(match.group(1))))
        return numbers, "numbered"

    # A bare year, codec/version or resolution is not an episode identifier.
    if _VERSION.search(stem):
        return set(), None
    integers = list(_INTEGER.finditer(stem))
    leading = re.match(r"^(\d{1,2})(?=$|[ ._-])", stem)
    if leading:
        numbers = _continuations(stem, leading.end(), int(leading.group(1)))
        if len(integers) > 1 and len(numbers) == 1:
            return set(), None
        return numbers, "numbered"
    trailing = re.fullmatch(r"([a-z][a-z ._-]*?)[ ._-]*(\d{1,2})", stem, re.I)
    if trailing:
        prefix = re.sub(r"[ ._-]+", " ", trailing.group(1)).strip().casefold()
        if prefix in {"h", "x", "v", "ver", "version"}:
            return set(), None
        return {int(trailing.group(2))}, f"trailing:{prefix}"
    return set(), None


def _evidence(path: Path, root: Path, hint: int | None) -> _Evidence:
    seasons: set[int] = set()
    folder = path.parent
    while True:
        seasons.update(int(match.group(1)) for match in _FOLDER_SEASON.finditer(folder.name))
        if folder == root:
            break
        folder = folder.parent

    explicit: set[EpisodeKey] = set()
    for match in _TAG.finditer(path.stem):
        season = int(match.group("s") or match.group("xs"))
        episode = int(match.group("e") or match.group("xe"))
        explicit.update(EpisodeKey(season, number) for number in _continuations(path.stem, match.end(), episode))
    dotted = _DOTTED_TAG.match(path.stem)
    if dotted and seasons and not _VERSION.search(path.stem):
        # Leading "1.01 Pilot" is common in AD packs. The folder must identify
        # the same season; the existing conflict check below rejects mismatch.
        explicit.add(EpisodeKey(int(dotted.group(1)), int(dotted.group(2))))
    if explicit:
        # Some packs separate the second episode tag with title text rather
        # than attaching E02 directly to S01E01. Never silently select only E01.
        tagged_seasons = {key.season for key in explicit}
        for match in _EPISODE.finditer(path.stem):
            for episode in _continuations(path.stem, match.end(), int(match.group(1))):
                explicit.update(EpisodeKey(season, episode) for season in tagged_seasons)
        tags = seasons | {key.season for key in explicit}
        reason = None
        if len(seasons) > 1 or (seasons and any(key.season not in seasons for key in explicit)):
            reason = "Filename episode tag conflicts with season folder context"
            explicit.update(EpisodeKey(season, key.episode) for season in seasons for key in tuple(explicit))
        elif len(explicit) > 1:
            reason = "Multiple episodes identified in one media file"
        return _Evidence(path, explicit, {key.episode for key in explicit}, None, tags, reason)

    numbers, pattern = _numeric(path.stem)
    context = seasons or ({hint} if hint is not None else set())
    keys = {EpisodeKey(season, number) for season in context for number in numbers}
    reason = None
    if len(seasons) > 1:
        reason = "Conflicting season folders; episode season is ambiguous"
    elif len(numbers) > 1:
        reason = "Multiple episodes identified in one media file"
    return _Evidence(path, keys, numbers, pattern, seasons, reason)


def _can_infer_season(video: list[_Evidence], audio: list[_Evidence]) -> bool:
    if any(item.season_tags for item in video + audio):
        return False
    sets = []
    for items in (video, audio):
        candidates = [item for item in items if len(item.numbers) == 1 and item.pattern and not item.reason]
        if len({item.pattern for item in candidates}) != 1:
            return False
        numbers = {number for item in candidates for number in item.numbers}
        if len(numbers) < 2:
            return False
        sets.append(numbers)
    return len(sets[0] & sets[1]) >= 2


def _known_season_hints(
    video: list[_Evidence], audio: list[_Evidence], selected: set[int],
) -> tuple[int | None, int | None]:
    """Transfer one evidenced season only across two unambiguous number matches."""
    def relevant(items: list[_Evidence]) -> list[_Evidence]:
        return [item for item in items
                if not selected or not item.season_tags or item.season_tags & selected]

    video, audio = relevant(video), relevant(audio)
    known = {season for item in video + audio for season in item.season_tags}
    if len(known) != 1:
        return None, None
    season = next(iter(known))

    def hint_for(unseasoned: list[_Evidence], counterpart: list[_Evidence]) -> int | None:
        candidates = [item for item in unseasoned
                      if not item.season_tags and not item.keys and not item.reason
                      and len(item.numbers) == 1 and item.pattern]
        if len({item.pattern for item in candidates}) != 1:
            return None
        candidate_counts = Counter(number for item in candidates for number in item.numbers)
        tagged_counts = Counter(
            key.episode for item in counterpart if not item.reason and len(item.keys) == 1
            for key in item.keys if key.season == season
        )
        blocked_numbers = {key.episode for item in counterpart if item.reason
                           for key in item.keys if key.season == season}
        matching = {number for number, count in candidate_counts.items()
                    if count == 1 and tagged_counts[number] == 1 and number not in blocked_numbers}
        return season if len(matching) >= 2 else None

    return hint_for(video, audio), hint_for(audio, video)


def discover_season(
    video_root: str | Path,
    ad_root: str | Path,
    output_dir: str | Path,
    seasons: Iterable[int] | None = None,
    excluded_dirs: Iterable[str | Path] = (),
) -> DiscoveryResult:
    """Pair one show's episodes recursively without probing or modifying media.

    Numbered names without season tags need an explicit season folder, a single
    selected season, or corroborating numeric sequences in both roots. A single
    known season can transfer across two exact episode matches. With no known
    season, Season 01 is a disclosed placeholder, never a claimed identity.
    Missing, conflicting and ambiguous files remain issues for caller policy.
    """
    video_root, ad_root, output_dir = map(_absolute, (video_root, ad_root, output_dir))
    for root in (video_root, ad_root):
        if not root.exists():
            raise FileNotFoundError(f"Input folder does not exist: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"Input path is not a folder: {root}")
    selected = set(seasons or ())
    if any(not isinstance(season, int) or isinstance(season, bool) or season < 0 for season in selected):
        raise ValueError("Season numbers must be nonnegative integers")
    hint = next(iter(selected)) if len(selected) == 1 else None
    exclusions = tuple(_absolute(folder) for folder in excluded_dirs)
    result = DiscoveryResult()

    def scan(root: Path, extensions: set[str], *, skip_ad_outputs: bool) -> list[Path]:
        # When output equals/contains an input root, excluding the entire root
        # would hide the originals. Generated .AD media is still filtered above.
        excluded_output = (output_dir,) if output_dir != root and output_dir.is_relative_to(root) else ()
        return _scan(root, extensions, exclusions + excluded_output, result.issues, skip_ad_outputs=skip_ad_outputs)

    video_paths = scan(video_root, _VIDEO_EXTENSIONS, skip_ad_outputs=True)
    ad_paths = scan(ad_root, _AD_EXTENSIONS, skip_ad_outputs=False)
    if video_root == ad_root:
        primary_videos = set(video_paths)
        ad_paths = [path for path in ad_paths if path not in primary_videos]
    videos = [_evidence(path, video_root, hint) for path in video_paths]
    audio = [_evidence(path, ad_root, hint) for path in ad_paths]
    if not selected and _can_infer_season(videos, audio):
        videos = [_evidence(path, video_root, 1) for path in video_paths]
        audio = [_evidence(path, ad_root, 1) for path in ad_paths]
        result.notes.append(
            "No season identifier; matched numeric sequence using Season 01. "
            "Use --season N to supply the actual season."
        )
    elif hint is None:
        video_hint, audio_hint = _known_season_hints(videos, audio, selected)
        if video_hint is not None:
            videos = [_evidence(path, video_root, video_hint) for path in video_paths]
        if audio_hint is not None:
            audio = [_evidence(path, ad_root, audio_hint) for path in ad_paths]
        if video_hint is not None or audio_hint is not None:
            inferred = video_hint if video_hint is not None else audio_hint
            result.notes.append(
                f"Inferred Season {inferred:02d} for unseasoned filenames from at least "
                "two exact episode-number matches with explicitly tagged counterparts. "
                "Use --season N to supply season context explicitly."
            )

    blocked: set[EpisodeKey] = set()

    def index(items: list[_Evidence], role: str) -> dict[EpisodeKey, list[Path]]:
        grouped: dict[EpisodeKey, list[Path]] = defaultdict(list)
        for item in items:
            known_seasons = item.season_tags | {key.season for key in item.keys}
            if selected and known_seasons and not (known_seasons & selected):
                continue
            if item.reason:
                blocked.update(item.keys)
                key = next(iter(item.keys)) if len(item.keys) == 1 else None
                result.issues.append(DiscoveryIssue(key, item.reason, (item.path,)))
            elif len(item.keys) == 1:
                grouped[next(iter(item.keys))].append(item.path)
            else:
                reason = f"Cannot determine episode number and season for {role}; use numbered names or --season N"
                result.issues.append(DiscoveryIssue(None, reason, (item.path,)))
        return grouped

    by_video, by_audio = index(videos, "video"), index(audio, "AD track")
    for key in sorted(by_video.keys() | by_audio.keys()):
        if key in blocked:
            continue
        video_matches, ad_matches = by_video.get(key, []), by_audio.get(key, [])
        paths = tuple(sorted(video_matches + ad_matches, key=_sort_path))
        problems = []
        if len(video_matches) > 1:
            problems.append("Multiple video files match this episode")
        if len(ad_matches) > 1:
            problems.append("Multiple AD tracks match this episode")
        if not video_matches:
            problems.append("No matching video file")
        if not ad_matches:
            problems.append("No matching AD track")
        if problems:
            result.issues.append(DiscoveryIssue(key, "; ".join(problems), paths))
            continue
        video, ad = video_matches[0], ad_matches[0]
        if video.samefile(ad):
            result.issues.append(DiscoveryIssue(key, "Video and AD input are the same file", paths))
            continue
        result.pairs.append(EpisodePair(key, video, ad, output_dir / f"Season {key.season:02d}" / f"{video.stem}.AD.mkv"))
    result.issues.sort(key=lambda issue: (
        issue.key is None, issue.key or EpisodeKey(0, 0),
        tuple(_sort_path(path) for path in issue.paths), issue.reason,
    ))
    return result
