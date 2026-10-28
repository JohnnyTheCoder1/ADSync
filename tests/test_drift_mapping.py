"""Drift coefficients must map AD time forward into video time."""

from __future__ import annotations

import numpy as np
import pytest

from adsync.align.drift import estimate_drift
from adsync.compute import CorrelationBackend
from adsync.models import Anchor, FeatureBundle


@pytest.mark.parametrize("stretch,offset", [(1.01, 5.0), (0.99, -7.0), (1.0, 3.0)])
def test_fitted_drift_reconstructs_measured_video_positions(monkeypatch, stretch, offset):
    sr, duration = 100, 600.0
    audio = np.zeros(int(sr * duration))
    features = FeatureBundle(
        sr=sr, hop_length=1, duration=duration,
        rms=np.zeros(10), onset=np.zeros(10),
        mel=np.zeros((2, 10)), mfcc=np.zeros((2, 10)),
    )

    def measured_match(vid, ad, video_time, *args, **kwargs):
        # Existing local matcher semantics: it locates each video window in
        # the AD. A measurement at V corresponds to A=(V-offset)/stretch.
        return Anchor(
            source_time=video_time, target_time=(video_time - offset) / stretch,
            score=1.0, window=30.0,
        )

    monkeypatch.setattr("adsync.align.drift._local_offset", measured_match)
    with CorrelationBackend("cpu") as compute:
        ppm, confidence, anchors, intercept = estimate_drift(
            features, features, y_vid=audio, y_ad=audio, audio_sr=sr, compute=compute,
        )

    # This is exactly how the pipeline rebuilds the affine AD-to-video map.
    rebuilt_video_times = [intercept + (1.0 + ppm / 1e6) * a.target_time for a in anchors]
    np.testing.assert_allclose(rebuilt_video_times, [a.source_time for a in anchors], atol=1e-9)
    assert ppm == pytest.approx((stretch - 1.0) * 1e6, abs=1e-8)
    assert intercept == pytest.approx(offset, abs=1e-9)
    assert confidence > 0.99
