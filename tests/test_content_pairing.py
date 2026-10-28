"""Season pairing by audio content when episode numbers in filenames are wrong."""

from pathlib import Path

import numpy as np
import pytest

from adsync.batch.discovery import EpisodeKey

SR = 8000
RECAP_CLIPS = ((40.0, 43.0), (70.0, 73.0))


@pytest.fixture(autouse=True)
def no_user_cache(monkeypatch):
    monkeypatch.setenv("ADSYNC_CACHE_DIR", "off")


def _noise(seed: int, seconds: float) -> np.ndarray:
    return np.random.default_rng(seed).normal(scale=0.1, size=int(SR * seconds)).astype(np.float32)


TITLES, CREDITS = _noise(9001, 10), _noise(9002, 8)


def _body(episode: int) -> np.ndarray:
    return _noise(episode, 100)


def _episode(episode: int, *, recaps: bool = False, late_titles: bool = False) -> np.ndarray:
    """Recap montage (or cold open), shared titles, unique story, shared credits."""
    if recaps and episode > 1:
        opening = np.concatenate([_body(episode - 1)[int(SR * lo):int(SR * hi)] for lo, hi in RECAP_CLIPS])
    else:
        opening = _noise(500 + episode, sum(hi - lo for lo, hi in RECAP_CLIPS))
    body = _body(episode)
    if late_titles:
        split = int(SR * 30)
        return np.concatenate([opening, body[:split], TITLES, body[split:], CREDITS])
    return np.concatenate([opening, TITLES, body, CREDITS])


def _described(audio: np.ndarray) -> np.ndarray:
    """An AD mix: delayed, quieter, with its own narration-only lead-in."""
    return np.concatenate([_noise(77, 1.5), 0.8 * audio])


def _recording(name: str, key: EpisodeKey, audio: np.ndarray):
    from adsync.batch.content_pairing import Recording
    from adsync.quality import fingerprint_features

    return Recording(Path(name), key, fingerprint_features(audio, SR))


def _season(video_content: dict[int, int], ad_content: dict[int, int], *, recaps: bool = False,
            late_titles: tuple[int, ...] = ()):
    """Name pairs for episode labels; values say which episode each file really holds."""
    def audio(episode):
        return _episode(episode, recaps=recaps, late_titles=episode in late_titles)

    return [
        (_recording(f"Show.S01E{n:02d}.mkv", EpisodeKey(1, n), audio(video_content[n])),
         _recording(f"Show.S01E{n:02d}.mp3", EpisodeKey(1, n), _described(audio(ad_content[n]))))
        for n in sorted(video_content)
    ]


def _summary(result):
    return [(m.key.episode, m.video.path.name, m.ad.path.name) for m in result.pairs]


def test_pairs_whose_audio_agrees_with_their_names_are_kept():
    from adsync.batch.content_pairing import resolve_content_pairs

    result = resolve_content_pairs(_season({1: 1, 2: 2, 3: 3}, {1: 1, 2: 2, 3: 3}))
    assert _summary(result) == [(1, "Show.S01E01.mkv", "Show.S01E01.mp3"),
                                (2, "Show.S01E02.mkv", "Show.S01E02.mp3"),
                                (3, "Show.S01E03.mkv", "Show.S01E03.mp3")]
    assert all(m.confirmed and not m.changed for m in result.pairs)
    assert not result.issues


def test_swapped_video_files_are_repaired_and_keep_video_numbering_without_story_evidence():
    from adsync.batch.content_pairing import resolve_content_pairs

    result = resolve_content_pairs(_season({1: 2, 2: 1, 3: 3}, {1: 1, 2: 2, 3: 3}))
    assert _summary(result) == [(1, "Show.S01E01.mkv", "Show.S01E02.mp3"),
                                (2, "Show.S01E02.mkv", "Show.S01E01.mp3"),
                                (3, "Show.S01E03.mkv", "Show.S01E03.mp3")]
    assert [m.changed for m in result.pairs] == [True, True, False]
    assert [m.label_source for m in result.pairs[:2]] == ["video", "video"]
    assert all(m.confirmed for m in result.pairs)
    assert not result.issues
    assert any("Show.S01E01.mkv" in note and "Show.S01E02.mp3" in note for note in result.notes)


def test_label_from_ad_names_repaired_pairs_by_the_ad_numbering():
    from adsync.batch.content_pairing import resolve_content_pairs

    result = resolve_content_pairs(_season({1: 2, 2: 1, 3: 3}, {1: 1, 2: 2, 3: 3}), label_from="ad")
    assert _summary(result) == [(1, "Show.S01E02.mkv", "Show.S01E01.mp3"),
                                (2, "Show.S01E01.mkv", "Show.S01E02.mp3"),
                                (3, "Show.S01E03.mkv", "Show.S01E03.mp3")]
    assert [m.label_source for m in result.pairs[:2]] == ["ad", "ad"]


def test_shared_titles_and_credits_alone_never_count_as_a_match():
    from adsync.batch.content_pairing import resolve_content_pairs

    result = resolve_content_pairs(_season({1: 1, 2: 2}, {1: 9, 2: 2}))
    assert _summary(result) == [(1, "Show.S01E01.mkv", "Show.S01E01.mp3"),
                                (2, "Show.S01E02.mkv", "Show.S01E02.mp3")]
    assert [m.confirmed for m in result.pairs] == [False, True]
    assert not any(m.changed for m in result.pairs)


def test_duplicate_ad_recording_is_reported_instead_of_guessed():
    from adsync.batch.content_pairing import resolve_content_pairs

    result = resolve_content_pairs(_season({1: 1, 2: 2}, {1: 2, 2: 2}))
    assert _summary(result) == [(2, "Show.S01E02.mkv", "Show.S01E02.mp3")]
    assert [issue.key for issue in result.issues] == [EpisodeKey(1, 1)]
    assert "S01E02" in result.issues[0].reason
    assert Path("Show.S01E01.mp3") in result.issues[0].paths


def test_ambiguous_content_match_is_reported_not_guessed():
    from adsync.batch.content_pairing import resolve_content_pairs

    result = resolve_content_pairs(_season({1: 1, 2: 2, 3: 3}, {1: 2, 2: 3, 3: 2}))
    assert _summary(result) == [(3, "Show.S01E03.mkv", "Show.S01E02.mp3")]
    assert sorted(issue.key.episode for issue in result.issues) == [1, 2, 3]
    ambiguous = next(issue for issue in result.issues if issue.key == EpisodeKey(1, 2))
    assert "more than one" in ambiguous.reason
    leftover = next(issue for issue in result.issues if issue.key == EpisodeKey(1, 3))
    assert leftover.paths == (Path("Show.S01E03.mp3"),)


def test_recaps_show_the_video_numbering_is_wrong_so_ad_numbering_is_used():
    from adsync.batch.content_pairing import resolve_content_pairs

    result = resolve_content_pairs(_season({1: 1, 2: 3, 3: 2, 4: 4}, {1: 1, 2: 2, 3: 3, 4: 4}, recaps=True))
    assert _summary(result) == [(1, "Show.S01E01.mkv", "Show.S01E01.mp3"),
                                (2, "Show.S01E03.mkv", "Show.S01E02.mp3"),
                                (3, "Show.S01E02.mkv", "Show.S01E03.mp3"),
                                (4, "Show.S01E04.mkv", "Show.S01E04.mp3")]
    assert [m.label_source for m in result.pairs if m.changed] == ["ad", "ad"]
    assert any("recap" in note.lower() for note in result.notes)


def test_recaps_show_the_ad_numbering_is_wrong_so_video_numbering_is_kept():
    from adsync.batch.content_pairing import resolve_content_pairs

    result = resolve_content_pairs(_season({1: 1, 2: 2, 3: 3, 4: 4}, {1: 1, 2: 3, 3: 2, 4: 4}, recaps=True))
    assert _summary(result) == [(1, "Show.S01E01.mkv", "Show.S01E01.mp3"),
                                (2, "Show.S01E02.mkv", "Show.S01E03.mp3"),
                                (3, "Show.S01E03.mkv", "Show.S01E02.mp3"),
                                (4, "Show.S01E04.mkv", "Show.S01E04.mp3")]
    assert [m.label_source for m in result.pairs if m.changed] == ["video", "video"]


def test_a_title_sequence_placed_late_is_not_mistaken_for_a_recap():
    from adsync.batch.content_pairing import resolve_content_pairs

    # Episode 2 runs its titles after the opening window, so episode 1's early
    # titles match "story" audio in episode 2; one continuous run is not a montage.
    result = resolve_content_pairs(_season({1: 2, 2: 1, 3: 3, 4: 4}, {1: 1, 2: 2, 3: 3, 4: 4},
                                           recaps=True, late_titles=(2,)))
    assert _summary(result)[:2] == [(1, "Show.S01E02.mkv", "Show.S01E01.mp3"),
                                    (2, "Show.S01E01.mkv", "Show.S01E02.mp3")]
    assert [m.label_source for m in result.pairs if m.changed] == ["ad", "ad"]


@pytest.mark.parametrize(("stem", "expected"), [
    ("Dexter.S02E07.1080p.English", "Dexter.S02E05.1080p.English"),
    ("Show 2x07 Title", "Show S02E05 Title"),
    ("Pilot", "S02E05 - Pilot"),
])
def test_relabeled_output_name_changes_only_the_episode_tag(stem, expected):
    from adsync.batch.content_pairing import relabel_stem

    assert relabel_stem(stem, EpisodeKey(2, 5)) == expected
