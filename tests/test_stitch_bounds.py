"""Bounds and overlapping ownership when rebuilding an audio timeline."""

from __future__ import annotations

import numpy as np
import pytest

from adsync.models import SegmentMap
from adsync.rebuild.stitch import stitch_segments


SR = 1000


def _signal(seconds=10, channels=1):
    y = np.tile(np.array([0.3, -0.3], dtype=np.float32), seconds * SR // 2)
    return y if channels == 1 else np.stack([y, -y])


def _segment(start, duration, src_start=0.0):
    return SegmentMap(
        src_start=src_start, src_end=src_start + duration,
        dst_start=start, dst_end=start + duration,
        offset=start - src_start, stretch=1.0, confidence=0.9,
    )


@pytest.mark.parametrize("channels", [1, 2])
def test_empty_map_returns_silence_for_exact_video_duration(channels):
    y = _signal(5, channels)
    out = stitch_segments(y, SR, [], 10.0)
    assert out.shape == (*y.shape[:-1], 10 * SR)
    assert out.dtype == np.float32
    assert np.count_nonzero(out) == 0
    assert not np.shares_memory(y, out)


@pytest.mark.parametrize("start", [-12.0, 10.0, 12.0, 30.0])
def test_segment_wholly_outside_video_is_ignored(start):
    out = stitch_segments(_signal(5), SR, [_segment(start, 5.0)], 10.0)
    assert out.shape == (10 * SR,)
    assert np.count_nonzero(out) == 0


@pytest.mark.parametrize("channels", [1, 2])
def test_negative_destination_crops_source_and_preserves_input(channels):
    y = _signal(5, channels)
    original = y.copy()
    out = stitch_segments(y, SR, [_segment(-2.0, 5.0)], 10.0)
    np.testing.assert_array_equal(out[..., :3 * SR], original[..., 2 * SR:])
    assert np.count_nonzero(out[..., 3 * SR:]) == 0
    np.testing.assert_array_equal(y, original)


def test_nested_overlaps_do_not_reintroduce_doubled_audio():
    y = _signal(10)
    out = stitch_segments(
        y, SR, [_segment(0.0, 10.0), _segment(3.0, 2.0), _segment(7.0, 3.0)],
        10.0, crossfade_ms=80,
    )
    np.testing.assert_allclose(out[8 * SR:9 * SR], y[8 * SR:9 * SR], atol=1e-7)


def test_zero_duration_output_is_empty():
    out = stitch_segments(_signal(5), SR, [_segment(0.0, 5.0)], 0.0)
    assert out.shape == (0,)
