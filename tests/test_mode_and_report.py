"""Modes do what they say, reports use one orientation, QC honours the user's nudge."""

import subprocess
import weakref

import numpy as np
import pytest
import soundfile as sf

from adsync.models import Anchor
from adsync.utils.subprocesses import _find_binary
from test_pipeline_regressions import pipeline_case


def _drift_estimate(ppm, anchors, intercept):
    return lambda *args, **kwargs: (ppm, 0.9, anchors, intercept)


def test_offset_mode_never_applies_drift(pipeline_case, monkeypatch):
    anchors = [Anchor(source_time=t, target_time=t, score=0.9, window=5.0) for t in (2.0, 6.0, 10.0)]
    monkeypatch.setattr("adsync.align.drift.estimate_drift", _drift_estimate(500.0, anchors, 0.0))
    report = pipeline_case.run(mode="offset")
    assert report.mode == "offset"
    assert report.segments[0].stretch == 1.0


def test_drift_anchors_are_reported_as_ad_to_video_with_the_nudge(pipeline_case, monkeypatch):
    # The drift estimator measures at video times, returning (video, AD) pairs.
    measured = [Anchor(source_time=t, target_time=t - 5.0, score=0.9, window=5.0) for t in (7.0, 9.0, 11.0)]
    monkeypatch.setattr("adsync.align.drift.estimate_drift", _drift_estimate(0.0, measured, 5.0))
    report = pipeline_case.run(mode="drift", offset_adjust=0.25)
    assert [(a.source_time, a.target_time) for a in report.anchors] == [(2.0, 7.25), (4.0, 9.25), (6.0, 11.25)]


def test_analysis_audio_is_released_before_the_full_quality_decode(pipeline_case, monkeypatch):
    from adsync.media import extract

    fake = extract.decode_audio
    analysis, alive_at_render = [], []

    def tracking_decode(info, **kwargs):
        if kwargs.get("sample_fmt") == "f32":
            alive_at_render.append([ref() is not None for ref in analysis])
        audio = fake(info, **kwargs)
        if kwargs.get("sample_fmt") != "f32":
            analysis.append(weakref.ref(audio))
        return audio

    monkeypatch.setattr(extract, "decode_audio", tracking_decode)
    pipeline_case.run(mux=True)
    assert alive_at_render == [[False, False]]


def test_qc_judges_timing_relative_to_the_users_nudge(tmp_path):
    from adsync.quality import verify_media_sync

    sr, nudge = 48000, 0.5
    rng = np.random.default_rng(4)
    bursts = np.repeat(rng.uniform(0.02, 0.3, size=240), sr // 4)
    soundtrack = (rng.normal(size=bursts.size) * bursts).astype(np.float32)
    delayed = np.concatenate([np.zeros(int(nudge * sr), np.float32), soundtrack[:-int(nudge * sr)]])
    sf.write(tmp_path / "main.wav", soundtrack, sr)
    sf.write(tmp_path / "ad.wav", delayed, sr)
    output = tmp_path / "rendered.mkv"
    subprocess.run([_find_binary("ffmpeg"), "-nostdin", "-v", "error", "-y",
                    "-i", tmp_path / "main.wav", "-i", tmp_path / "ad.wav", "-map", "0:a", "-map", "1:a",
                    "-c:a", "flac", "-metadata:s:a:0", "language=eng", "-metadata:s:a:1", "language=eng",
                    "-metadata:s:a:1", "title=Audio Description", output],
                   check=True, capture_output=True, stdin=subprocess.DEVNULL)
    result = verify_media_sync(output, work_dir=tmp_path / "qc", report={"offset_adjust": nudge, "segments": []})
    assert not any("Coherent nonzero offset" in failure for failure in result["failures"]), result["failures"]
