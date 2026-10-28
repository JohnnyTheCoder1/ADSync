"""Real-world release names pair by their episode tag; numbers in titles and codecs are not episodes."""

import pytest

from test_season_discovery import _files


@pytest.mark.parametrize("video_name, label", [
    ("Lost - S05E06 - 316.mkv", "S05E06"),
    ("Battlestar Galactica - S01E01 - 33.mkv", "S01E01"),
    ("Chernobyl - S01E01 - 1-23-45.mkv", "S01E01"),
    ("24 - S01E01 - 12-00 A.M.-1-00 A.M..mkv", "S01E01"),
    ("Show.S01E01.x264-GRP.mkv", "S01E01"),
    ("Show.S02E07.720p.HDTV.x265-GRP.mkv", "S02E07"),
])
def test_numbers_after_the_episode_tag_are_title_or_codec_text(tmp_path, video_name, label):
    from adsync.batch.discovery import discover_season

    video = _files(tmp_path / "video", video_name)
    audio = _files(tmp_path / "audio", f"[{label[:3]}.{label[3:]}] Track.mp3")
    result = discover_season(video, audio, tmp_path / "output")
    assert not result.issues
    assert [pair.key.label for pair in result.pairs] == [label]


@pytest.mark.parametrize("video_name", [
    "Show.S01E01E02.mkv", "Show - S01E01-E02 - Pilot.mkv", "Show S01E01-02.mkv", "Show.S01E01 & E02.mkv",
])
def test_genuine_multi_episode_files_are_still_flagged(tmp_path, video_name):
    from adsync.batch.discovery import discover_season

    video = _files(tmp_path / "video", video_name)
    audio = _files(tmp_path / "audio", "Show.S01E01.mp3", "Show.S01E02.mp3")
    result = discover_season(video, audio, tmp_path / "output")
    assert not result.pairs
    assert any("Multiple episodes" in issue.reason for issue in result.issues)


def test_episodes_blocked_by_a_multi_episode_file_are_reported(tmp_path):
    from adsync.batch.discovery import discover_season

    video = _files(tmp_path / "video", "Show.S01E01E02.mkv", "Show.S01E03.mkv")
    audio = _files(tmp_path / "audio", "Show.S01E01.mp3", "Show.S01E02.mp3", "Show.S01E03.mp3")
    result = discover_season(video, audio, tmp_path / "output")
    assert [pair.key.label for pair in result.pairs] == ["S01E03"]
    reported = {issue.key.label for issue in result.issues if issue.key is not None}
    assert {"S01E01", "S01E02"} <= reported
