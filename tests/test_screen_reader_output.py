"""Errors and results reach a screen reader as plain lines with their reasons."""

import subprocess

import numpy as np
import pytest
from typer.testing import CliRunner

from adsync.cli import app
from adsync.media.mux import mux_ad_track
from adsync.media.probe import probe
from adsync.models import SyncReport
from adsync.report import json_report
from adsync.utils.subprocesses import FFmpegError, _find_binary

BOX_DRAWING = set("─│┌┐└┘├┤┬┴┼╭╮╰╯━┃")


def _ffmpeg(*args):
    subprocess.run(
        [_find_binary("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *map(str, args)],
        check=True, capture_output=True, stdin=subprocess.DEVNULL,
    )


@pytest.fixture(scope="module")
def tiny_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("cli") / "episode.mkv"
    _ffmpeg("-f", "lavfi", "-i", "color=c=blue:s=32x32:r=10:d=1",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1",
            "-map", "0:v", "-map", "1:a", "-c:v", "ffv1", "-c:a", "flac", path)
    return path


@pytest.fixture
def junk_file(tmp_path):
    path = tmp_path / "broken track.mkv"
    path.write_bytes(b"not audio " * 100)
    return path


def test_mux_failure_carries_ffmpegs_reason(tiny_video, tmp_path):
    with pytest.raises(FFmpegError, match="(?i)encoder not found"):
        mux_ad_track(tiny_video, np.zeros(48000, dtype=np.float32), 48000, tmp_path / "out.mkv",
                     codec="no_such_encoder")


def test_unreadable_media_is_named_with_its_reason(junk_file):
    with pytest.raises(ValueError, match=r"broken track\.mkv: Invalid data found"):
        probe(junk_file)


def test_help_is_plain_text():
    out = CliRunner().invoke(app, ["sync", "--help"]).output
    assert "--ad-stream" in out
    assert not BOX_DRAWING & set(out)


def test_cli_failure_is_a_plain_error_line(tiny_video, junk_file, tmp_path):
    result = CliRunner().invoke(app, ["sync", str(tiny_video), str(junk_file), "-o", str(tmp_path / "out.mkv")])
    assert result.exit_code == 2
    assert "Error:" in result.output and "Invalid data found" in result.output
    assert not BOX_DRAWING & set(result.output)


def test_withheld_output_exits_nonzero_and_says_so(tiny_video, tmp_path, monkeypatch):
    report = SyncReport(mode="warp", confidence=0.95,
                        identity_check={"status": "fail", "reasons": ["Shared intro only"]})
    monkeypatch.setattr("adsync._pipeline.run_pipeline", lambda **kwargs: report)
    ad = tmp_path / "ad.wav"
    ad.write_bytes(b"placeholder")
    result = CliRunner().invoke(app, ["sync", str(tiny_video), str(ad), "-o", str(tmp_path / "out.mkv")])
    assert result.exit_code == 1
    assert "No output written" in result.output


def test_bracketed_path_parts_survive_printing(capsys):
    report = SyncReport(mode="warp", confidence=0.95, output_path=r"D:\Movies\Alien [extended]\Alien [extended].AD.mkv",
                        warnings=["Check the [x265] release"])
    json_report.print_summary(report)
    out = capsys.readouterr().out
    assert "Alien [extended].AD.mkv" in out and "[x265]" in out
    assert not BOX_DRAWING & set(out)


@pytest.mark.parametrize("confidence, verdict", [(0.95, "High"), (0.72, "Medium"), (0.65, "below")])
def test_verdict_follows_the_confidence_threshold(capsys, confidence, verdict):
    json_report.print_summary(SyncReport(mode="warp", confidence=confidence), threshold=0.70)
    assert verdict in capsys.readouterr().out.strip().splitlines()[-1]
