"""Segment edges never click: holes fade, contiguous segments leave no silent samples."""

import numpy as np
from scipy.interpolate import PchipInterpolator

from adsync.models import SegmentMap
from adsync.rebuild.stitch import stitch_segments
from adsync.rebuild.warp_render import render_from_warp

SR = 8000


def _tone(seconds, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * 233.37 * t)).astype(np.float32)


def _largest_step(y):
    return float(np.max(np.abs(np.diff(y, axis=-1))))


def _segment(src_start, src_end, dst_start, stretch=1.0):
    return SegmentMap(src_start=src_start, src_end=src_end, dst_start=dst_start,
                      dst_end=dst_start + (src_end - src_start) * stretch,
                      offset=dst_start - src_start, stretch=stretch, confidence=0.9)


def test_warp_render_fades_both_sides_of_a_hole():
    y = _tone(20)
    # AD 0-10 s plays at video 0-10 s and AD 10-20 s at 15-25 s: a 5 s scene the AD lacks.
    fns = [PchipInterpolator([0.0, 10.0], [0.0, 10.0]), PchipInterpolator([10.0, 20.0], [15.0, 25.0])]
    out = render_from_warp(y, SR, fns, [(0.0, 10.0), (10.0, 20.0)], 25.0)
    assert _largest_step(out) <= 1.25 * _largest_step(y)


def test_stitch_fades_both_sides_of_a_hole():
    y = _tone(20)
    out = stitch_segments(y, SR, [_segment(0.0, 10.0, 0.0), _segment(10.0, 20.0, 15.0)], 25.0)
    assert _largest_step(out) <= 1.25 * _largest_step(y)


def test_contiguous_stitched_segments_leave_no_silent_samples():
    sr = 48000
    y = _tone(30, sr)
    # Piecewise segments meet at fractional-sample destinations with small stretch changes.
    edges = [0.0, 4.123457, 9.876541, 15.000013, 22.222229, 29.0]
    segments, dst = [], 1.2345678
    for a, b, stretch in zip(edges, edges[1:], [1.003, 0.998, 1.001, 0.9995, 1.002]):
        segments.append(_segment(a, b, dst, stretch))
        dst = segments[-1].dst_end
    out = stitch_segments(y, sr, segments, 32.0)
    body = out[int(1.3 * sr):int(29.9 * sr)]
    assert np.count_nonzero(body == 0.0) == 0
    assert _largest_step(body) <= 1.25 * _largest_step(y)
