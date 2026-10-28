"""Personal batch entry points follow the same safe publication contract."""

import importlib.util
import subprocess
from pathlib import Path

import pytest


def _load_batch(name):
    script = Path(__file__).parents[1] / f"{name}.py"
    if not script.is_file():
        pytest.skip("Personal batch scripts are optional and excluded from the public repository")
    spec = importlib.util.spec_from_file_location(name, script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["batch_from", "batch_warp_redo"])
def test_batch_environment_selects_final_directory(name, monkeypatch, tmp_path):
    monkeypatch.setenv("ADSYNC_OUTPUT_DIR", str(tmp_path / "shared library"))
    module = _load_batch(name)
    assert module.FINAL_DIR == tmp_path / "shared library"


def test_redo_failure_preserves_previously_published_movie(tmp_path, monkeypatch):
    module = _load_batch("batch_warp_redo")
    monkeypatch.setattr(module, "FINAL_DIR", tmp_path)
    source, ad, final = (tmp_path / name for name in ["prepped.mkv", "ad.wav", "movie.mkv"])
    source.write_bytes(b"source")
    ad.write_bytes(b"ad")
    final.write_bytes(b"old complete media")
    monkeypatch.setattr(module.subprocess, "run", lambda args: subprocess.CompletedProcess(args, 2))

    assert not module.sync_episode("S01E01", source, ad, final.name)
    assert final.read_bytes() == b"old complete media"


def test_batch_without_ad_publishes_complete_media_and_preserves_prep(tmp_path, monkeypatch):
    module = _load_batch("batch_from")
    prepped = tmp_path / "prepped.mkv"
    prepped.write_bytes(b"complete prepped media")
    monkeypatch.setattr(module, "FINAL_DIR", tmp_path / "final")
    monkeypatch.setattr(module, "PREP_DIR", tmp_path / "work")
    monkeypatch.setattr(module.sys, "argv", ["batch_from.py", "--season", "2"])
    monkeypatch.setattr(module, "_get_episodes", lambda season: ({"S02E04": (prepped.name, None)}, tmp_path, tmp_path))
    monkeypatch.setattr(module, "step1_prep", lambda source, tag: prepped)
    module.main()
    assert (tmp_path / "final" / "From.S02E04.mkv").read_bytes() == b"complete prepped media"
    assert prepped.read_bytes() == b"complete prepped media"


def test_failed_batch_prep_does_not_leave_a_file_next_run_would_skip(tmp_path, monkeypatch):
    module = _load_batch("batch_from")
    monkeypatch.setattr(module, "PREP_DIR", tmp_path)
    source = tmp_path / "source.mkv"
    source.write_bytes(b"source media")

    def fail_encode(args):
        Path(args[-1]).write_bytes(b"unfinished media")
        raise RuntimeError("encoder failed")

    monkeypatch.setattr(module, "run_ffmpeg", fail_encode)
    with pytest.raises(RuntimeError, match="encoder failed"):
        module.step1_prep(source, "S02E04")
    assert not (tmp_path / "S02E04.prepped.mkv").exists()
    assert source.read_bytes() == b"source media"


def test_season_three_pairs_literal_episode_prefix_not_glob_characters(tmp_path, monkeypatch):
    module = _load_batch("batch_from")
    video_dir, ad_dir = tmp_path / "video", tmp_path / "ad"
    video_dir.mkdir()
    ad_dir.mkdir()
    (video_dir / "From.S03E03.episode.mkv").touch()
    for name in [
        "S03 unrelated.mp3", "[S03.E02] Wrong.mp3", "[S03.E030] Wrong.mp3",
        "[s03.e03] Correct.mp3",
    ]:
        (ad_dir / name).touch()
    (ad_dir / "[S03.E03] Metadata folder").mkdir()
    monkeypatch.setattr(module, "S03_MKV_DIR", video_dir)
    monkeypatch.setattr(module, "S03_AD_DIR", ad_dir)
    assert module._discover_s03_episodes() == {
        "S03E03": ("From.S03E03.episode.mkv", "[s03.e03] Correct.mp3"),
    }


def test_season_three_rejects_ambiguous_ad_versions(tmp_path, monkeypatch):
    module = _load_batch("batch_from")
    video_dir, ad_dir = tmp_path / "video", tmp_path / "ad"
    video_dir.mkdir()
    ad_dir.mkdir()
    (video_dir / "From.S03E03.episode.mkv").touch()
    (ad_dir / "[S03.E03] Version A.mp3").touch()
    (ad_dir / "[S03.E03] Version B.mp3").touch()
    monkeypatch.setattr(module, "S03_MKV_DIR", video_dir)
    monkeypatch.setattr(module, "S03_AD_DIR", ad_dir)
    with pytest.raises(ValueError, match="S03E03"):
        module._discover_s03_episodes()
