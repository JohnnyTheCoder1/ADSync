"""Output QC must compare the streams' playback timelines, including gaps."""

import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from adsync.quality import verify_media_sync


pytestmark = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="Real container timestamp checks require FFmpeg and ffprobe",
)


@pytest.fixture(scope="module")
def shared_soundtrack(tmp_path_factory):
    path = tmp_path_factory.mktemp("quality-timestamps") / "soundtrack.wav"
    samples = np.random.default_rng(904).normal(scale=0.1, size=60 * 16000).astype("float32")
    sf.write(path, samples, 16000, subtype="FLOAT")
    return path


def make_container(source: Path, output: Path, scenario: str) -> list[float]:
    command = ["ffmpeg", "-hide_banner", "-v", "error", "-y"]
    if scenario == "common_start":
        command += ["-itsoffset", "1.0"]
    command += ["-i", str(source)]
    if scenario in {"delayed_start", "common_start"}:
        command += ["-itsoffset", "1.0"]
    command += ["-i", str(source)]
    if scenario == "internal_gap":
        command += ["-filter_complex", r"[1:a]asetpts=PTS+gte(T\,30)/TB[delayed]",
                    "-map", "0:a", "-map", "[delayed]"]
    else:
        command += ["-map", "0:a", "-map", "1:a"]
    command += ["-c:a", "libopus" if scenario == "common_preroll" else "flac",
                "-metadata:s:a:0", "language=eng",
                "-metadata:s:a:1", "language=eng",
                "-metadata:s:a:1", "title=Audio Description", str(output)]
    subprocess.run(command, check=True, capture_output=True, timeout=30)
    info = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(output)], timeout=30,
    ))
    return [float(stream["start_time"]) for stream in info["streams"]]


@pytest.mark.parametrize("scenario", ["delayed_start", "internal_gap"])
def test_container_timestamp_desync_cannot_pass_qc(shared_soundtrack, tmp_path, monkeypatch, scenario):
    monkeypatch.setenv("ADSYNC_CACHE_DIR", "off")
    output = tmp_path / f"{scenario}.mkv"
    starts = make_container(shared_soundtrack, output, scenario)
    if scenario == "delayed_start":
        assert starts[1] - starts[0] == pytest.approx(1.0)
    else:
        assert starts == [0.0, 0.0]
    # Both streams contain the identical samples. Only their packet timestamps
    # differ, so decoding them independently from sample zero would falsely pass.
    result = verify_media_sync(output)
    assert result["status"] == "fail", result
    assert result["failures"]


@pytest.mark.parametrize("scenario", ["common_start", "common_preroll"])
def test_common_timestamps_preserve_correct_audio_sync(shared_soundtrack, tmp_path, monkeypatch, scenario):
    monkeypatch.setenv("ADSYNC_CACHE_DIR", "off")
    output = tmp_path / f"{scenario}.mkv"
    starts = make_container(shared_soundtrack, output, scenario)
    assert starts[0] == starts[1]
    assert starts[0] < 0 if scenario == "common_preroll" else starts[0] > 0
    result = verify_media_sync(output)
    assert result["status"] == "pass", result
