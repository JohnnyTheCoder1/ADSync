"""Tests for drift estimation."""

from __future__ import annotations

import numpy as np
import pytest

from adsync.align.drift import estimate_drift
from adsync.models import FeatureBundle


def _make_features(onset: list[float], sr: int = 16000, hop: int = 512) -> FeatureBundle:
    n = len(onset)
    return FeatureBundle(
        sr=sr,
        hop_length=hop,
        duration=n * hop / sr,
        rms=[0.0] * n,
        onset=onset,
        mel=[[0.0] * n],
        mfcc=[[0.0] * n],
    )


def test_no_drift() -> None:
    """Identical signals should yield near-zero drift."""
    np.random.seed(42)
    sig = np.random.randn(5000).tolist()
    vid = _make_features(sig)
    ad = _make_features(sig)

    drift, conf, anchors, intercept = estimate_drift(vid, ad, window_sec=5.0, search_sec=2.0)
    assert abs(drift) < 500  # < 500 ppm
    assert len(anchors) > 0


def test_drift_returns_anchors() -> None:
    """Drift estimation should produce anchors."""
    np.random.seed(42)
    sig = np.random.randn(6000).tolist()
    vid = _make_features(sig)
    ad = _make_features(sig)

    _, _, anchors, _ = estimate_drift(vid, ad, window_sec=5.0, search_sec=2.0)
    assert len(anchors) >= 3
    # Anchors should be roughly on the diagonal
    for a in anchors:
        assert abs(a.source_time - a.target_time) < 2.0
