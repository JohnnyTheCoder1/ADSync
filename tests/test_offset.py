"""Tests for global offset detection."""

from __future__ import annotations

import numpy as np
import pytest

from adsync.align.global_offset import estimate_global_offset
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


def test_zero_offset() -> None:
    """Identical signals should yield ~0 offset with high confidence."""
    np.random.seed(42)
    sig = np.random.randn(2000).tolist()
    vid = _make_features(sig)
    ad = _make_features(sig)

    offset, conf = estimate_global_offset(vid, ad, max_offset_sec=5.0)
    assert abs(offset) < 0.1
    assert conf > 0.5


def test_known_offset() -> None:
    """A shifted signal should be detected as an offset."""
    np.random.seed(42)
    n = 3000
    sig = np.random.randn(n)

    hop = 512
    sr = 16000
    shift_frames = 30  # shift AD by 30 frames later
    shift_sec = shift_frames * hop / sr

    vid_onset = sig.tolist()
    ad_onset = np.zeros(n).tolist()
    # AD is the same signal but shifted earlier (AD starts later)
    ad_onset[shift_frames:] = sig[: n - shift_frames].tolist()

    vid = _make_features(vid_onset, sr=sr, hop=hop)
    ad = _make_features(ad_onset, sr=sr, hop=hop)

    offset, conf = estimate_global_offset(vid, ad, max_offset_sec=5.0)
    # The offset should be approximately shift_sec
    assert abs(abs(offset) - shift_sec) < 0.5


def test_uncorrelated_low_confidence() -> None:
    """Uncorrelated noise should yield low confidence."""
    np.random.seed(42)
    vid = _make_features(np.random.randn(2000).tolist())
    np.random.seed(99)
    ad = _make_features(np.random.randn(2000).tolist())

    _, conf = estimate_global_offset(vid, ad, max_offset_sec=5.0)
    assert conf < 0.6
