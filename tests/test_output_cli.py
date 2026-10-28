"""CLI destination selection and real FFmpeg integration on tiny generated media."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from adsync.cli import app
from adsync.models import SyncReport
from adsync.utils.subprocesses import _find_binary


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    folder = tmp_path_factory.mktemp("output-media")
    video, audio = folder / "episode.mkv", folder / "ad.wav"
    binary = _find_binary("ffmpeg")
    subprocess.run([
        binary, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=32x32:d=0.3",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=0.3",
        "-c:v", "mpeg4", "-c:a", "pcm_s16le", "-metadata:s:a:0", "language=eng",
        "-shortest", str(video),
    ], check=True, capture_output=True)
    subprocess.run([
        binary, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "sine=frequency=880:sample_rate=48000:duration=0.3",
        str(audio),
    ], check=True, capture_output=True)
    return video, audio


@pytest.mark.parametrize("command", ["sync", "mux", "prep"])
def test_cli_output_dir_env_and_explicit_filename(command, media, tmp_path, monkeypatch):
    from adsync import _pipeline
    from adsync.media.probe import probe

    video, audio = media
    if command == "sync":
        # Alignment is intentionally outside this filesystem/CLI test. mux/prep
        # below exercise real FFmpeg against the same destination options.
        def run_pipeline(**kwargs):
            destination = kwargs["output_path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"synced test media")
            return SyncReport(mode="offset", confidence=1.0, output_path=str(destination))

        monkeypatch.setattr(_pipeline, "run_pipeline", run_pipeline)

    runner = CliRunner()
    inputs = [str(video)] + ([str(audio)] if command != "prep" else [])
    suffix = ".prepped.mkv" if command == "prep" else ".synced.mkv"
    monkeypatch.setenv("ADSYNC_OUTPUT_DIR", str(tmp_path / "environment"))

    result = runner.invoke(app, [command, *inputs])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "environment" / f"episode{suffix}").is_file()

    result = runner.invoke(app, [command, *inputs, "--output-dir", str(tmp_path / "override")])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "override" / f"episode{suffix}").is_file()

    explicit = tmp_path / "explicit" / "chosen.mkv"
    result = runner.invoke(app, [command, *inputs, "--output-dir", str(tmp_path / "ignored"), "-o", str(explicit)])
    assert result.exit_code == 0, result.output
    assert explicit.is_file()
    assert not (tmp_path / "ignored").exists()
    if command != "sync":
        info = probe(explicit)
        assert len(info.video_streams) == 1
        assert len(info.audio_streams) == (2 if command == "mux" else 1)


@pytest.mark.parametrize("command", ["mux", "prep"])
def test_cli_publication_failure_reports_recoverable_file(command, media, tmp_path, monkeypatch):
    from adsync.media import output

    video, audio = media
    blocker = tmp_path / "share-unavailable"
    blocker.write_text("not a directory")
    inputs = [str(video)] + ([str(audio)] if command == "mux" else [])
    captured = []
    real_publish = output._publish

    def capture_publish(local, destination):
        captured.append(local)
        return real_publish(local, destination)

    monkeypatch.setattr(output, "_publish", capture_publish)
    result = CliRunner().invoke(app, [command, *inputs, "-o", str(blocker / "movie.mkv")])
    assert result.exit_code == 2, result.output
    assert len(captured) == 1
    local = captured[0]
    try:
        assert "Completed local output retained" in result.output
        assert local.name in result.output
        assert local.stat().st_size > 0
    finally:
        local.unlink()
        local.parent.rmdir()


def test_mux_track_failure_preserves_prior_destination(media, tmp_path):
    import numpy as np
    from adsync.media.mux import mux_ad_track

    video, _ = media
    destination = tmp_path / "existing.mkv"
    destination.write_bytes(b"previous complete media")
    with pytest.raises(Exception):
        mux_ad_track(video, np.ones(4800), 48000, destination, codec="no_such_encoder")
    assert destination.read_bytes() == b"previous complete media"


def test_mux_track_renders_valid_media(media, tmp_path):
    import numpy as np
    from adsync.media.mux import mux_ad_track
    from adsync.media.probe import probe

    video, _ = media
    destination = tmp_path / "nested" / "movie.mkv"
    result = mux_ad_track(video, np.zeros(14400), 48000, destination)
    assert result == destination
    assert len(probe(result).audio_streams) == 2


@pytest.mark.parametrize("command", ["mux", "prep"])
def test_cli_rejects_output_equal_to_input(command, media):
    video, audio = media
    before = video.read_bytes()
    inputs = [str(video)] + ([str(audio)] if command == "mux" else [])
    result = CliRunner().invoke(app, [command, *inputs, "-o", str(video)])
    assert result.exit_code == 2, result.output
    assert "input file" in result.output
    assert video.read_bytes() == before


def _cleanup_child(process, stage):
    """Keep a failing lifecycle regression test from leaking its test child."""
    if process.poll() is None:
        process.kill()
    process.wait(timeout=5)
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            stream.close()
    if stage.exists():
        stage.unlink()
    if stage.parent.exists():
        stage.parent.rmdir()


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_prep_callback_failure_reaps_child_before_cleaning_staging(media, tmp_path, monkeypatch, failure):
    from adsync.media import prep

    video, _ = media
    destination = tmp_path / "previous.mkv"
    destination.write_bytes(b"previous complete media")
    real_popen = subprocess.Popen
    started = []

    def popen(command, **kwargs):
        if "-progress" not in command:
            return real_popen(command, **kwargs)
        # A real process holds the output file open and waits. This reproduces
        # Windows' staging-cleanup failure when cancellation leaves FFmpeg alive.
        code = (
            "import sys,time\n"
            "f=open(sys.argv[1],'wb'); f.write(b'partial'); f.flush()\n"
            "print('out_time_us=100000', flush=True)\n"
            "print('progress=continue', flush=True)\n"
            "time.sleep(30)\n"
        )
        process = real_popen([sys.executable, "-c", code, command[-1]], **kwargs)
        started.append((process, Path(command[-1])))
        return process

    def interrupted_progress(*args):
        raise failure("callback interrupted")

    monkeypatch.setattr(prep.subprocess, "Popen", popen)
    try:
        with pytest.raises(failure, match="callback interrupted"):
            prep.prep_video(video, destination, on_progress=interrupted_progress)
        process, stage = started[0]
        assert process.poll() is not None, "FFmpeg child survived callback failure"
        assert process.stdout.closed and process.stderr.closed
        assert not stage.parent.exists()
        assert destination.read_bytes() == b"previous complete media"
    finally:
        for process, stage in started:
            _cleanup_child(process, stage)


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_mux_conversion_failure_reaps_child_before_cleaning_staging(media, tmp_path, monkeypatch, failure):
    import numpy as np
    from adsync.media import mux

    video, _ = media
    destination = tmp_path / "previous.mkv"
    destination.write_bytes(b"previous complete media")
    started = []

    def start_encoder(args, *, threads=None):
        code = (
            "import sys,time\n"
            "f=open(sys.argv[1],'wb'); f.write(b'partial'); f.flush()\n"
            "print('ready', flush=True)\n"
            "time.sleep(30)\n"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", code, args[-1]],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        started.append((process, Path(args[-1])))
        assert process.stdout.readline().rstrip(b"\r\n") == b"ready"
        process.stdout.close()
        return process

    def failed_conversion(*args, **kwargs):
        raise failure("conversion interrupted")

    monkeypatch.setattr(mux, "run_ffmpeg_streamed", start_encoder)
    monkeypatch.setattr(mux.np, "clip", failed_conversion)
    try:
        with pytest.raises(failure, match="conversion interrupted"):
            mux.mux_ad_track(video, np.zeros(100), 48000, destination, n_existing_audio=1)
        process, stage = started[0]
        assert process.poll() is not None, "FFmpeg child survived conversion failure"
        assert process.stdin.closed and process.stderr.closed
        assert not stage.parent.exists()
        assert destination.read_bytes() == b"previous complete media"
    finally:
        for process, stage in started:
            _cleanup_child(process, stage)
