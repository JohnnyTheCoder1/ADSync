"""Tests for math utilities."""

from __future__ import annotations

import numpy as np
import pytest

from adsync.utils.mathx import crossfade, linear_fit_offset, rms_normalize


def test_rms_normalize() -> None:
    y = np.ones(1000) * 0.1
    result = rms_normalize(y, target_db=-20.0)
    rms = np.sqrt(np.mean(result ** 2))
    target = 10 ** (-20.0 / 20.0)
    assert abs(rms - target) < 0.01


def test_rms_normalize_silent() -> None:
    y = np.zeros(1000)
    result = rms_normalize(y)
    assert np.allclose(result, 0.0)


def test_crossfade_basic() -> None:
    a = np.ones(100)
    b = np.ones(100) * 2
    result = crossfade(a, b, 20)
    assert len(result) == 180
    # Start should be 1.0, end should be 2.0
    assert result[0] == pytest.approx(1.0)
    assert result[-1] == pytest.approx(2.0)
    # Middle of crossfade should be ~1.5
    assert abs(result[90] - 1.5) < 0.2


def test_crossfade_zero_overlap() -> None:
    a = np.ones(50)
    b = np.ones(50) * 3
    result = crossfade(a, b, 0)
    assert len(result) == 100


def test_linear_fit() -> None:
    times = np.array([0.0, 50.0, 100.0])
    offsets = np.array([1.0, 1.5, 2.0])
    intercept, slope = linear_fit_offset(times, offsets)
    assert abs(intercept - 1.0) < 0.01
    assert abs(slope - 0.01) < 0.001
