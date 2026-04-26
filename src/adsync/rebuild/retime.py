"""Retime AD audio segments according to the segment map."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from adsync.models import SegmentMap

log = logging.getLogger("adsync")

_BLOCK_SECONDS = 30


def retime_segment(
    y: NDArray[np.floating],
    sr: int,
    seg: SegmentMap,
) -> NDArray[np.float32]:
    """Cut [src_start, src_end] from *y* and resample to seg.stretch ratio.

    Linear interpolation in 30-second blocks so memory stays bounded on
    full-length films.  At typical stretch values (0.99–1.01) the resulting
    pitch shift (~17 cents) is imperceptible.

    Returns an owned float32 array — the caller is free to mutate it.
    """
    start_sample = int(seg.src_start * sr)
    end_sample = int(seg.src_end * sr)
    src = y[start_sample:end_sample]
    src_len = len(src)

    if src_len == 0:
        return np.zeros(0, dtype=np.float32)

    if abs(seg.stretch - 1.0) <= 1e-6:
        return src.astype(np.float32, copy=True)

    target_len = int(src_len * seg.stretch)
    if target_len == 0:
        return np.zeros(0, dtype=np.float32)

    log.debug(
        "Resampling [%.1f–%.1f] stretch=%.6f (%d → %d samples)",
        seg.src_start, seg.src_end, seg.stretch, src_len, target_len,
    )

    src = np.ascontiguousarray(src, dtype=np.float32)
    out = np.empty(target_len, dtype=np.float32)
    scale = (src_len - 1) / max(target_len - 1, 1)
    block = sr * _BLOCK_SECONDS

    for start in range(0, target_len, block):
        end = min(start + block, target_len)
        pos = np.arange(start, end, dtype=np.float64)
        pos *= scale
        np.clip(pos, 0.0, src_len - 1.0, out=pos)
        idx = pos.astype(np.int64)
        np.clip(idx, 0, src_len - 2, out=idx)
        np.subtract(pos, idx, out=pos)
        frac = pos.astype(np.float32)
        chunk = out[start:end]
        np.multiply(src[idx], 1.0 - frac, out=chunk)
        idx += 1
        chunk += src[idx] * frac

    return out
