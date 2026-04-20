"""Retime AD audio segments according to the segment map."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from adsync.models import SegmentMap

log = logging.getLogger("adsync")


def retime_segment(
    y: NDArray[np.floating],
    sr: int,
    seg: SegmentMap,
) -> NDArray[np.floating]:
    """Extract and retime one segment of the AD audio.

    Applies:
    1. Cut source range
    2. Linear-interpolation resample to target length if stretch != 1.0

    Uses numpy linear interpolation (O(n)) — fast even on very long segments.
    At typical stretch values (0.99–1.01) the resulting pitch shift (~17 cents)
    is imperceptible, and there are zero phase artifacts.
    """
    start_sample = int(seg.src_start * sr)
    end_sample = int(seg.src_end * sr)
    segment = y[start_sample:end_sample]

    if len(segment) == 0:
        return segment

    if abs(seg.stretch - 1.0) > 1e-6:
        src_len = len(segment)
        target_len = int(src_len * seg.stretch)
        if target_len == 0:
            return np.zeros(0, dtype=segment.dtype)
        log.debug(
            "Resampling [%.1f–%.1f] stretch=%.6f (%d → %d samples)",
            seg.src_start, seg.src_end, seg.stretch,
            src_len, target_len,
        )
        src_indices = np.linspace(0, src_len - 1, target_len)
        segment = np.interp(src_indices, np.arange(src_len), segment).astype(segment.dtype)

    return segment
