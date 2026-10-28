"""Public season command: preview, filtering and unresolved-pair exit status."""

from pathlib import Path

from typer.testing import CliRunner

from adsync.cli import app


def _folders(tmp_path):
    video, ad = tmp_path / "video", tmp_path / "ad"
    video.mkdir()
    ad.mkdir()
    for label in ("S01E01", "S01E02", "S02E01"):
        (video / f"Show.{label}.mkv").touch()
        (ad / f"Show.{label}.mp3").touch()
    return video, ad


def test_season_dry_run_lists_pairs_without_writing_output(tmp_path):
    video, ad = _folders(tmp_path)
    output = tmp_path / "output"
    result = CliRunner().invoke(app, ["season", str(video), str(ad),
                                     "--output-dir", str(output), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "S01E01" in result.output and "S02E01" in result.output
    assert "3" in result.output
    assert not output.exists()
    assert sorted(p.name for p in video.iterdir()) == [
        "Show.S01E01.mkv", "Show.S01E02.mkv", "Show.S02E01.mkv",
    ]


def test_season_filter_does_not_process_other_seasons(tmp_path):
    video, ad = _folders(tmp_path)
    result = CliRunner().invoke(app, ["season", str(video), str(ad), "--season", "2", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "S02E01" in result.output
    assert "S01E01" not in result.output


def test_strict_season_with_missing_ad_stops_without_output(tmp_path):
    video, ad = _folders(tmp_path)
    (ad / "Show.S01E02.mp3").unlink()
    output = tmp_path / "output"
    result = CliRunner().invoke(app, ["season", str(video), str(ad),
                                     "--output-dir", str(output), "--strict"])
    assert result.exit_code == 2, result.output
    assert "S01E02" in result.output
    assert not output.exists()


def test_dry_run_missing_match_returns_review_status(tmp_path):
    video, ad = _folders(tmp_path)
    (ad / "Show.S01E02.mp3").unlink()
    result = CliRunner().invoke(app, ["season", str(video), str(ad), "--dry-run"])
    assert result.exit_code == 1, result.output
    assert "S01E02" in result.output


def test_season_rejects_invalid_device_before_processing(tmp_path):
    video, ad = _folders(tmp_path)
    result = CliRunner().invoke(app, ["season", str(video), str(ad), "--device", "bad"])
    assert result.exit_code == 2
    assert "device" in result.output.lower()


def test_explicit_missing_episode_is_not_silently_ignored(tmp_path):
    video, ad = _folders(tmp_path)
    result = CliRunner().invoke(app, ["season", str(video), str(ad), "--season", "1",
                                     "--episode", "1", "--episode", "99", "--strict", "--dry-run"])
    assert result.exit_code == 2, result.output
    assert "S01E99" in result.output


def _fake_recordings(monkeypatch, contents: dict[str, int]):
    """Stand in for audio decoding: each file name maps to the episode it really holds."""
    import numpy as np
    from adsync.batch import content_pairing
    from adsync.quality import fingerprint_features

    monkeypatch.setenv("ADSYNC_CACHE_DIR", "off")
    rate = 8000

    def load(path, config, *, video, threads=1):
        audio = np.random.default_rng(contents[Path(path).name]).normal(scale=0.1, size=rate * 60)
        return fingerprint_features(audio.astype(np.float32), rate)

    monkeypatch.setattr(content_pairing, "load_recording_features", load)


def _swapped_video_folders(tmp_path, monkeypatch):
    video, ad = tmp_path / "video", tmp_path / "ad"
    video.mkdir()
    ad.mkdir()
    for n in (1, 2, 3):
        (video / f"Show.S01E0{n}.mkv").touch()
        (ad / f"Show.S01E0{n}.mp3").touch()
    _fake_recordings(monkeypatch, {"Show.S01E01.mkv": 2, "Show.S01E02.mkv": 1, "Show.S01E03.mkv": 3,
                                   "Show.S01E01.mp3": 1, "Show.S01E02.mp3": 2, "Show.S01E03.mp3": 3})
    return video, ad


def test_season_repairs_swapped_files_by_audio_before_processing(tmp_path, monkeypatch):
    video, ad = _swapped_video_folders(tmp_path, monkeypatch)
    result = CliRunner().invoke(app, ["season", str(video), str(ad), "--output-dir", str(tmp_path / "out"),
                                     "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "S01E01: Show.S01E01.mkv + Show.S01E02.mp3" in result.output
    assert "S01E02: Show.S01E02.mkv + Show.S01E01.mp3" in result.output
    assert "S01E03: Show.S01E03.mkv + Show.S01E03.mp3" in result.output
    assert "2 re-paired" in result.output


def test_label_from_ad_numbers_repaired_outputs_by_the_ad_files(tmp_path, monkeypatch):
    video, ad = _swapped_video_folders(tmp_path, monkeypatch)
    result = CliRunner().invoke(app, ["season", str(video), str(ad), "--output-dir", str(tmp_path / "out"),
                                     "--label-from", "ad", "--dry-run"])
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    pair = lines.index("S01E01: Show.S01E02.mkv + Show.S01E01.mp3")
    assert lines[pair + 1].endswith(str(Path("Season 01") / "Show.S01E01.AD.mkv"))


def test_no_content_check_pairs_by_filenames_without_reading_audio(tmp_path, monkeypatch):
    from adsync.batch import content_pairing

    video, ad = _swapped_video_folders(tmp_path, monkeypatch)

    def forbidden(*args, **kwargs):
        raise AssertionError("audio was read")

    monkeypatch.setattr(content_pairing, "load_recording_features", forbidden)
    result = CliRunner().invoke(app, ["season", str(video), str(ad), "--no-content-check", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "S01E01: Show.S01E01.mkv + Show.S01E01.mp3" in result.output


def test_unknown_label_source_is_rejected_before_reading_files(tmp_path):
    video, ad = _folders(tmp_path)
    result = CliRunner().invoke(app, ["season", str(video), str(ad), "--label-from", "sideways", "--dry-run"])
    assert result.exit_code == 2
    assert "No such option" not in result.output
    assert "auto, video, ad" in result.output


def test_explicit_missing_season_is_not_silently_ignored(tmp_path):
    video, ad = _folders(tmp_path)
    result = CliRunner().invoke(app, ["season", str(video), str(ad), "--season", "1",
                                     "--season", "99", "--strict", "--dry-run"])
    assert result.exit_code == 2, result.output
    assert "99" in result.output


def test_unreachable_output_share_stops_before_any_work(tmp_path):
    video, ad = _folders(tmp_path)
    share = "\\\\127.0.0.1\\adsync-test-missing-share\\Shows"
    result = CliRunner().invoke(app, ["season", str(video), str(ad), "--output-dir", share, "--no-content-check"])
    assert result.exit_code == 2, result.output
    assert "not reachable" in result.output
