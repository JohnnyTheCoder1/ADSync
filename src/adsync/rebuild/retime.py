"""Retime AD audio segments according to the segment map."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from adsync.models import SegmentMap
from adsync.utils.mathx import catmull_rom_interp

log = logging.getLogger("adsync")

_BLOCK_SECONDS = 30


def retime_segment(
    y: NDArray[np.floating],
    sr: int,
    seg: SegmentMap,
    *,
    out_start: int,
    length: int,
) -> NDArray[np.float32]:
    """Render output samples ``[out_start, out_start + length)`` of *seg* from *y*.

    Source positions come from the segment's absolute mapping, so touching
    segments meet without a gap or a jump. Stretch 1 copies samples verbatim;
    other stretches use cubic interpolation in 30-second blocks to bound
    memory. Accepts 1-D (mono) or 2-D (channels, samples) input and returns an
    owned float32 array of the same layout.
    """
    out = _empty_like(y, length)
    if length == 0:
        return out
    if abs(seg.stretch - 1.0) <= 1e-6:
        first = out_start - round((seg.dst_start - seg.src_start) * sr)
        lo, hi = max(first, 0), min(first + length, y.shape[-1])
        out[...] = 0.0
        if hi > lo:
            out[..., lo - first:hi - first] = y[..., lo:hi]
        return out

    log.debug("Resampling [%.1f–%.1f] stretch=%.6f (%d samples)", seg.src_start, seg.src_end, seg.stretch, length)
    block = sr * _BLOCK_SECONDS
    for start in range(0, length, block):
        end = min(start + block, length)
        seconds = (out_start + np.arange(start, end, dtype=np.float64)) / sr
        positions = (seg.src_start + (seconds - seg.dst_start) / seg.stretch) * sr
        catmull_rom_interp(y, positions, out[..., start:end])
    return out


def _empty_like(y: NDArray[np.floating], n: int) -> NDArray[np.float32]:
    if y.ndim == 1:
        return np.zeros(n, dtype=np.float32) if n == 0 else np.empty(n, dtype=np.float32)
    shape = (y.shape[0], n)
    return np.zeros(shape, dtype=np.float32) if n == 0 else np.empty(shape, dtype=np.float32)
