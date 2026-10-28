"""Episode matching uses filename evidence and never pairs by sorted position."""

from pathlib import Path

import pytest


def _files(root, *names):
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"unchanged source")
    return root


@pytest.mark.parametrize("video_name,ad_name", [
    ("Show.S01E02.1080p.mkv", "[S01.E02] Episode.mp3"),
    ("Show.1x02.mp4", "show.s1e2.m4a"),
    ("Show.S00E03.mkv", "[S00.E03] Special.flac"),
])
def test_explicit_tags_pair_across_formats_without_touching_media(tmp_path, video_name, ad_name):
    from adsync.batch.discovery import discover_season

    video = _files(tmp_path / "video", video_name)
    audio = _files(tmp_path / "audio", ad_name)
    result = discover_season(video, audio, tmp_path / "output")
    assert not result.issues
    assert len(result.pairs) == 1
    pair = result.pairs[0]
    assert pair.video_path == video / video_name
    assert pair.ad_path == audio / ad_name
    assert pair.output_path == tmp_path / "output" / f"Season {pair.key.season:02d}" / f"{Path(video_name).stem}.AD.mkv"
    assert pair.key.label == ("S00E03" if "S00" in video_name else "S01E02")
    assert pair.video_path.read_bytes() == pair.ad_path.read_bytes() == b"unchanged source"
    assert not (tmp_path / "output").exists()


def test_season_folders_supply_context_for_episode_only_names(tmp_path):
    from adsync.batch.discovery import discover_season

    video = _files(tmp_path / "video", "Season 1/02 - Pilot.mkv", "Season 2/E02 - Return.mp4")
    audio = _files(tmp_path / "audio", "Season 01/episode 2.mp3", "Season 02/ad 2.mka")
    result = discover_season(video, audio, tmp_path / "output")
    assert not result.issues
    assert [pair.key.label for pair in result.pairs] == ["S01E02", "S02E02"]


def test_missing_and_duplicate_matches_are_reported_without_arbitrary_selection(tmp_path):
    from adsync.batch.discovery import discover_season

    video = _files(tmp_path / "video", "Show.S01E01.mkv", "Show.S01E02.mkv", "A/Show.S01E03.mkv", "B/Show.S01E03.mp4", "Show.S01E05.mkv")
    audio = _files(tmp_path / "audio", "S01E01.mp3", "S01E03.mp3", "S01E04.mp3", "A/S01E05.mp3", "B/S01E05.flac")
    result = discover_season(video, audio, tmp_path / "output")
    assert [pair.key.label for pair in result.pairs] == ["S01E01"]
    by_key = {issue.key.label: issue for issue in result.issues}
    assert set(by_key) == {"S01E02", "S01E03", "S01E04", "S01E05"}
    assert "AD" in by_key["S01E02"].reason
    assert "video" in by_key["S01E04"].reason.lower()
    assert video / "A/Show.S01E03.mkv" in by_key["S01E03"].paths
    assert video / "B/Show.S01E03.mp4" in by_key["S01E03"].paths
    assert audio / "A/S01E05.mp3" in by_key["S01E05"].paths
    assert audio / "B/S01E05.flac" in by_key["S01E05"].paths


@pytest.mark.parametrize("multi_name", ["Show.S01E01E02.mkv", "Show.S01E01-E02.mkv", "Show.1x01-02.mkv", "Show.S01E01.S01E02.mkv"])
def test_multi_episode_video_blocks_every_identified_episode(tmp_path, multi_name):
    from adsync.batch.discovery import discover_season

    video = _files(tmp_path / "video", multi_name, "Show.S01E01.mkv")
    audio = _files(tmp_path / "audio", "S01E01.mp3", "S01E02.mp3")
    result = discover_season(video, audio, tmp_path / "output")
    assert not result.pairs
    assert any(video / multi_name in issue.paths and "multiple" in issue.reason.lower() for issue in result.issues)


def test_filter_excludes_other_seasons_and_their_missing_matches(tmp_path):
    from adsync.batch.discovery import discover_season

    video = _files(tmp_path / "video", "S01E01.mkv", "S02E01.mkv", "S03E01.mkv")
    audio = _files(tmp_path / "audio", "S02E01.mp3")
    result = discover_season(video, audio, tmp_path / "output", seasons=[2])
    assert not result.issues
    assert [pair.key.label for pair in result.pairs] == ["S02E01"]


def test_generated_files_destination_and_state_directories_are_excluded(tmp_path):
    from adsync.batch.discovery import discover_season

    video = _files(tmp_path / "video", "S01E01.mkv", "S01E01.synced.mkv", "S01E01.prepped.mkv", "S01E01.AD.mkv", "output/S01E02.mkv", "state/S01E03.mkv")
    audio = _files(tmp_path / "audio", "S01E01.mp3")
    result = discover_season(video, audio, video / "output", excluded_dirs=[video / "state"])
    assert not result.issues
    assert [pair.key.label for pair in result.pairs] == ["S01E01"]


def test_output_equal_to_input_root_keeps_originals_discoverable(tmp_path):
    from adsync.batch.discovery import discover_season

    video = _files(tmp_path / "video", "S01E01.mkv", "Season 01/S01E01.AD.mkv")
    audio = _files(tmp_path / "audio", "S01E01.mp3")
    result = discover_season(video, audio, video)
    assert len(result.pairs) == 1
    assert not result.issues


def test_symlink_directory_loop_is_not_followed(tmp_path):
    from adsync.batch.discovery import discover_season

    video = _files(tmp_path / "video", "S01E01.mkv")
    audio = _files(tmp_path / "audio", "S01E01.mp3")
    try:
        (video / "loop").symlink_to(video, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Directory symlinks unavailable: {exc}")
    result = discover_season(video, audio, tmp_path / "output")
    assert len(result.pairs) == 1
    assert not result.issues


def test_single_season_filter_provides_numbered_filename_context(tmp_path):
    from adsync.batch.discovery import discover_season

    video = _files(tmp_path / "video", "02 - Pilot.mkv")
    audio = _files(tmp_path / "audio", "ad 2.mp3")
    result = discover_season(video, audio, tmp_path / "output", seasons=[4])
    assert not result.issues
    assert [pair.key.label for pair in result.pairs] == ["S04E02"]


@pytest.mark.parametrize("video_names,ad_names", [
    (["Episode 1.mkv", "Episode 2.mkv", "Episode 4.mkv"], ["ad 1.mp3", "ad 2.mp3", "ad 3.mp3"]),
    (["Show 01.mkv", "Show 02.mkv", "Show 04.mkv"], ["Track 01.mp3", "Track 02.mp3", "Track 03.mp3"]),
    (["01 - Pilot.mkv", "02 - Return.mkv", "04 - Finale.mkv"], ["1.mp3", "2.mp3", "3.mp3"]),
])
def test_consistent_numeric_evidence_infers_season_one_and_preserves_gaps(tmp_path, video_names, ad_names):
    from adsync.batch.discovery import discover_season

    video = _files(tmp_path / "video", *video_names)
    audio = _files(tmp_path / "audio", *ad_names)
    result = discover_season(video, audio, tmp_path / "output")
    assert [pair.key.label for pair in result.pairs] == ["S01E01", "S01E02"]
    assert {issue.key.label for issue in result.issues} == {"S01E03", "S01E04"}
    assert len(result.notes) == 1
    assert "Season 01" in result.notes[0] and "--season" in result.notes[0]


@pytest.mark.parametrize("video_names,ad_names", [
    (["1917.mkv", "2001 A Space Odyssey.mkv"], ["1917.mp3", "2001 A Space Odyssey.mp3"]),
    (["720.mkv", "1080.mkv"], ["720.mp3", "1080.mp3"]),
    (["Episode 1.mkv"], ["ad 1.mp3"]),
    (["Episode 1.mkv", "Episode 2.mkv"], ["ad 2.mp3", "ad 3.mp3"]),
    (["Film One 1.mkv", "Other Film 2.mkv"], ["Track 1.mp3", "Track 2.mp3"]),
    (["Show x264.mkv", "Show h265.mkv"], ["Track 1.mp3", "Track 2.mp3"]),
    (["Show v1.mkv", "Show v2.mkv"], ["Track 1.mp3", "Track 2.mp3"]),
])
def test_ambiguous_numeric_evidence_never_pairs_by_file_order(tmp_path, video_names, ad_names):
    from adsync.batch.discovery import discover_season

    video = _files(tmp_path / "video", *video_names)
    audio = _files(tmp_path / "audio", *ad_names)
    result = discover_season(video, audio, tmp_path / "output")
    assert not result.pairs
    assert result.issues


def test_explicit_season_tags_disable_virtual_season_guessing(tmp_path):
    from adsync.batch.discovery import discover_season

    video = _files(tmp_path / "video", "Show.S02E01.mkv", "Episode 2.mkv", "Episode 3.mkv")
    audio = _files(tmp_path / "audio", "S02E01.mp3", "ad 2.mp3", "ad 3.mp3")
    result = discover_season(video, audio, tmp_path / "output")
    assert [pair.key.label for pair in result.pairs] == ["S02E01"]
    assert len(result.issues) == 4


def test_same_file_cannot_be_both_video_and_ad_input(tmp_path):
    from adsync.batch.discovery import discover_season

    folder = _files(tmp_path / "input", "S01E01.mkv")
    result = discover_season(folder, folder, tmp_path / "output")
    assert not result.pairs
    assert len(result.issues) == 1
    assert folder / "S01E01.mkv" in result.issues[0].paths


def test_missing_root_raises_before_processing(tmp_path):
    from adsync.batch.discovery import discover_season

    with pytest.raises(FileNotFoundError):
        discover_season(tmp_path / "missing", tmp_path / "audio", tmp_path / "output")


@pytest.mark.parametrize("folder", ["Show - Season 3 (2024)", "Season.03", "S03"])
def test_root_folder_name_is_a_season_identifier(tmp_path, folder):
    from adsync.batch.discovery import discover_season

    video = _files(tmp_path / "videos" / folder, "ep 2.mkv")
    audio = _files(tmp_path / "audio", "S03E02.mp3")
    result = discover_season(video, audio, tmp_path / "output")
    assert not result.issues and not result.notes
    assert [pair.key.label for pair in result.pairs] == ["S03E02"]


def test_filename_and_folder_season_conflict_never_pairs(tmp_path):
    from adsync.batch.discovery import discover_season

    video = _files(tmp_path / "video", "Season 1/Show.S02E02.mkv")
    audio = _files(tmp_path / "audio", "S02E02.mp3")
    result = discover_season(video, audio, tmp_path / "output")
    assert not result.pairs
    assert any("conflict" in issue.reason.lower() for issue in result.issues)


def test_normal_ad_suffix_in_audio_name_is_not_treated_as_generated_video(tmp_path):
    from adsync.batch.discovery import discover_season

    video = _files(tmp_path / "video", "Show.S01E02.mkv")
    audio = _files(tmp_path / "audio", "Show.S01E02.AD.mp3")
    result = discover_season(video, audio, tmp_path / "output")
    assert not result.issues
    assert len(result.pairs) == 1
    assert result.pairs[0].ad_path == audio / "Show.S01E02.AD.mp3"


def test_multi_episode_tag_after_title_is_still_ambiguous(tmp_path):
    from adsync.batch.discovery import discover_season

    video = _files(tmp_path / "video", "Show.S01E01.Pilot.and.E02.Return.mkv")
    audio = _files(tmp_path / "audio", "S01E01.mp3", "S01E02.mp3")
    result = discover_season(video, audio, tmp_path / "output")
    assert not result.pairs
    assert any("multiple" in issue.reason.lower() for issue in result.issues)


def test_multi_episode_range_blocks_middle_episode_too(tmp_path):
    from adsync.batch.discovery import discover_season

    video = _files(tmp_path / "video", "Show.S01E01-E03.mkv", "Show.S01E02.mkv")
    audio = _files(tmp_path / "audio", "S01E02.mp3")
    result = discover_season(video, audio, tmp_path / "output")
    assert not result.pairs


def test_numeric_duplicates_remain_ambiguous_after_inference(tmp_path):
    from adsync.batch.discovery import discover_season

    video = _files(tmp_path / "video", "Episode 1.mkv", "Episode 1 version 2.mkv", "Episode 2.mkv")
    audio = _files(tmp_path / "audio", "ad 1.mp3", "ad 2.mp3")
    result = discover_season(video, audio, tmp_path / "output")
    assert [pair.key.label for pair in result.pairs] == ["S01E02"]
    assert len(result.issues) == 1 and result.issues[0].key.label == "S01E01"


def test_unreadable_subdirectory_becomes_issue_without_hiding_readable_pairs(tmp_path, monkeypatch):
    import os
    from adsync.batch.discovery import discover_season

    video = _files(tmp_path / "video", "S01E01.mkv", "unreadable/S01E02.mkv")
    audio = _files(tmp_path / "audio", "S01E01.mp3")
    real_scandir = os.scandir

    def scandir(path):
        if Path(path) == video / "unreadable":
            raise PermissionError(13, "Access denied", str(path))
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", scandir)
    result = discover_season(video, audio, tmp_path / "output")
    assert [pair.key.label for pair in result.pairs] == ["S01E01"]
    assert len(result.issues) == 1
    assert result.issues[0].paths == (video / "unreadable",)


def test_one_shared_folder_pairs_video_with_audio_without_counting_video_as_ad(tmp_path):
    from adsync.batch.discovery import discover_season

    folder = _files(tmp_path / "input", "Show.S01E01.mkv", "S01E01.mp3")
    result = discover_season(folder, folder, tmp_path / "output")
    assert len(result.pairs) == 1 and not result.issues
    assert result.pairs[0].ad_path == folder / "S01E01.mp3"


def test_shared_folder_keeps_ambiguous_ad_containers_visible(tmp_path):
    from adsync.batch.discovery import discover_season

    folder = _files(tmp_path / "input", "Show.S01E01.mkv", "S01E01.mp3", "S01E01.AD.mkv")
    result = discover_season(folder, folder, tmp_path / "output")
    assert not result.pairs
    assert any(folder / "S01E01.AD.mkv" in issue.paths for issue in result.issues)


@pytest.mark.parametrize("tagged_role", ["video", "audio"])
def test_one_known_season_supplies_cross_folder_context_with_two_exact_matches(tmp_path, tagged_role):
    from adsync.batch.discovery import discover_season

    video_names = ["Show.S03E01.mkv", "Show.S03E02.mkv", "Show.S03E04.mkv"]
    audio_names = ["ad 1.mp3", "ad 2.mp3", "ad 3.mp3"]
    if tagged_role == "audio":
        video_names = ["Show 01.mkv", "Show 02.mkv", "Show 04.mkv"]
        audio_names = ["S03E01.mp3", "S03E02.mp3", "S03E03.mp3"]
    video = _files(tmp_path / "video", *video_names)
    audio = _files(tmp_path / "audio", *audio_names)
    result = discover_season(video, audio, tmp_path / "output")
    assert [pair.key.label for pair in result.pairs] == ["S03E01", "S03E02"]
    assert {issue.key.label for issue in result.issues} == {"S03E03", "S03E04"}
    assert len(result.notes) == 1 and "Season 03" in result.notes[0]
    assert all(pair.output_path.parent.name == "Season 03" for pair in result.pairs)


def test_multiple_known_seasons_do_not_supply_cross_folder_numeric_context(tmp_path):
    from adsync.batch.discovery import discover_season

    video = _files(tmp_path / "video", "S02E01.mkv", "S02E02.mkv", "S03E01.mkv", "S03E02.mkv")
    audio = _files(tmp_path / "audio", "ad 1.mp3", "ad 2.mp3")
    result = discover_season(video, audio, tmp_path / "output")
    assert not result.pairs and not result.notes
    assert result.issues
    explicit = discover_season(video, audio, tmp_path / "output", seasons=[3])
    assert [pair.key.label for pair in explicit.pairs] == ["S03E01", "S03E02"]
    assert not explicit.issues


@pytest.mark.parametrize("video_names,audio_names", [
    (["S03E01.mkv", "S03E02.mkv"], ["ad 2.mp3", "ad 3.mp3"]),
    (["S03E01.mkv", "S03E02.mkv"], ["Film One 1.mp3", "Other Film 2.mp3"]),
    (["A/S03E01.mkv", "B/S03E01.mkv", "S03E02.mkv"], ["ad 1.mp3", "ad 2.mp3"]),
    (["S03E01.mkv", "S03E02.mkv"], ["ad 1.mp3", "ad 1 version 2.mp3", "ad 2.mp3"]),
])
def test_cross_folder_season_inference_requires_two_unambiguous_number_matches(tmp_path, video_names, audio_names):
    from adsync.batch.discovery import discover_season

    video = _files(tmp_path / "video", *video_names)
    audio = _files(tmp_path / "audio", *audio_names)
    result = discover_season(video, audio, tmp_path / "output")
    assert not result.pairs and not result.notes
    assert result.issues


def test_cross_folder_inference_does_not_supply_unproven_context_to_other_root(tmp_path):
    from adsync.batch.discovery import discover_season

    video = _files(tmp_path / "video", "S03E01.mkv", "S03E02.mkv", "Other Film 3.mkv")
    audio = _files(tmp_path / "audio", "ad 1.mp3", "ad 2.mp3", "ad 3.mp3")
    result = discover_season(video, audio, tmp_path / "output")
    assert [pair.key.label for pair in result.pairs] == ["S03E01", "S03E02"]
    assert any(issue.paths == (video / "Other Film 3.mkv",) and issue.key is None for issue in result.issues)


def test_cross_folder_inference_ignores_explicit_seasons_outside_selection(tmp_path):
    from adsync.batch.discovery import discover_season

    video = _files(tmp_path / "video", "S02E01.mkv", "S03E01.mkv", "S03E02.mkv")
    audio = _files(tmp_path / "audio", "ad 1.mp3", "ad 2.mp3")
    result = discover_season(video, audio, tmp_path / "output", seasons=[3, 4])
    assert [pair.key.label for pair in result.pairs] == ["S03E01", "S03E02"]
    assert not result.issues


def test_leading_dotted_season_episode_labels_use_matching_folder_context(tmp_path):
    from adsync.batch.discovery import discover_season

    video = _files(tmp_path / "video", "Show.S01E01.mkv", "Show.S01E13.mkv", "Show.S01E23.mkv")
    audio = _files(tmp_path / "Show - Season 1 (2008)", "1.01 Pilot.mp3", "1.13 Episode Title.mp3", "1.23 Finale.mp3")
    result = discover_season(video, audio, tmp_path / "output")
    assert [pair.key.label for pair in result.pairs] == ["S01E01", "S01E13", "S01E23"]
    assert not result.issues and not result.notes


def test_leading_dotted_video_label_is_supported_with_season_folder(tmp_path):
    from adsync.batch.discovery import discover_season

    video = _files(tmp_path / "Season 2", "2.01 - Pilot.mkv")
    audio = _files(tmp_path / "audio", "S02E01.mp3")
    result = discover_season(video, audio, tmp_path / "output")
    assert [pair.key.label for pair in result.pairs] == ["S02E01"]
    assert not result.issues


def test_dotted_filename_season_conflict_does_not_guess_from_folder(tmp_path):
    from adsync.batch.discovery import discover_season

    video = _files(tmp_path / "video", "S01E01.mkv")
    audio = _files(tmp_path / "Season 1", "2.01 Wrong Season.mp3")
    result = discover_season(video, audio, tmp_path / "output")
    assert not result.pairs
    assert any("conflict" in issue.reason.lower() for issue in result.issues)


@pytest.mark.parametrize("selected", [None, [1]])
def test_dotted_label_without_a_season_folder_is_not_interpreted_as_an_episode(tmp_path, selected):
    from adsync.batch.discovery import discover_season

    video = _files(tmp_path / "video", "S01E01.mkv")
    audio = _files(tmp_path / "audio", "1.01 Pilot.mp3")
    result = discover_season(video, audio, tmp_path / "output", seasons=selected)
    assert not result.pairs


@pytest.mark.parametrize("name", [
    "1.5 Decimal.mp3", "1.01.mp3", "1.01.2 Version.mp3", "1.01 v2.mp3",
    "2024.01 January.mp3", "1.01Beta.mp3", "1.01 1080p.mp3",
])
def test_decimals_versions_years_are_not_dotted_episode_tags(tmp_path, name):
    from adsync.batch.discovery import discover_season

    video = _files(tmp_path / "video", "S01E01.mkv")
    audio = _files(tmp_path / "Season 1", name)
    result = discover_season(video, audio, tmp_path / "output")
    assert not result.pairs
