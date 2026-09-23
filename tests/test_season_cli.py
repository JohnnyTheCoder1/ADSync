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


def test_explicit_missing_season_is_not_silently_ignored(tmp_path):
    video, ad = _folders(tmp_path)
    result = CliRunner().invoke(app, ["season", str(video), str(ad), "--season", "1",
                                     "--season", "99", "--strict", "--dry-run"])
    assert result.exit_code == 2, result.output
    assert "99" in result.output
