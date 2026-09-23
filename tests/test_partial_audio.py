from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import correlate, correlation_lags

from adsync._pipeline import run_pipeline
from adsync.config import SyncConfig
from adsync.quality import verify_video_preservation
from test_partial_harness import harness


pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
                                reason="Container and audio checks require FFmpeg")


def _case(tmp_path, name):
    case = next(c for c in harness().make_scenarios() if c.name == name)
    target, source = tmp_path / "target.wav", tmp_path / "source.wav"
    sf.write(target, case.target, case.sample_rate, subtype="FLOAT")
    sf.write(source, case.source, case.sample_rate, subtype="FLOAT")
    return case, target, source


def test_fingerprint_hint_does_not_hide_a_repeated_scene_alternative(tmp_path, monkeypatch):
    monkeypatch.setenv("ADSYNC_CACHE_DIR", "off")
    _, target, source = _case(tmp_path, "repeated_scene")
    report = run_pipeline(video_path=target, ad_path=source, output_path=None, mux=False,
                          config=SyncConfig(mode="partial", device="cpu", threads=2, speed_detect=False))
    detail = report.timing_debug["partial_alignment"]
    assert detail["ambiguous_ranges"], detail
    assert report.alignment_review_required
    assert sum(r["end_sec"] - r["start_sec"] for r in detail["ambiguous_ranges"]) >= 20


def test_partial_render_preserves_short_island_and_original_video(tmp_path, monkeypatch):
    monkeypatch.setenv("ADSYNC_CACHE_DIR", "off")
    case, target, source = _case(tmp_path, "paired_edit")
    video, output, decoded = tmp_path / "video.mkv", tmp_path / "output.mkv", tmp_path / "decoded.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=64x48:r=10:d=100",
                    "-i", str(target), "-map", "0:v", "-map", "1:a", "-c:v", "ffv1", "-c:a", "flac",
                    str(video)], check=True, capture_output=True)
    report = run_pipeline(video_path=video, ad_path=source, output_path=output, mux=True,
                          config=SyncConfig(mode="partial", device="cpu", threads=2, speed_detect=False,
                                            fingerprint=False))
    assert report.mode == "partial" and report.alignment_review_required
    assert verify_video_preservation(video, output)["status"] == "pass"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(output), "-map", "0:a:1", "-ac", "1",
                    "-ar", "16000", "-c:a", "pcm_f32le", str(decoded)], check=True, capture_output=True)
    rendered, sr = sf.read(decoded)
    for start, end in [(10, 14), (45, 49), (80, 84)]:
        expected = case.target[start * sr:end * sr]
        actual = rendered[start * sr:end * sr]
        lags = correlation_lags(len(actual), len(expected))
        correlation = correlate(actual, expected, method="fft")
        selected = np.abs(lags) <= .1 * sr
        best = np.argmax(correlation[selected])
        assert abs(lags[selected][best] / sr) < .05
        assert correlation[selected][best] / (np.linalg.norm(expected) * np.linalg.norm(actual)) > .6
    assert np.sqrt(np.mean(rendered[41 * sr:43 * sr] ** 2)) < .001
