"""Stitch retimed segments with crossfades and silence fills."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from adsync.models import SegmentMap
from adsync.rebuild.retime import retime_segment
from adsync.utils.mathx import crossfade

log = logging.getLogger("adsync")


def stitch_segments(
    y: NDArray[np.floating],
    sr: int,
    segments: list[SegmentMap],
    video_duration: float,
    *,
    crossfade_ms: int = 80,
) -> NDArray[np.floating]:
    """Build the final synced AD waveform from segments with crossfades.

    This is the careful version that:
    - Inserts silence for gaps
    - Crossfades joins
    - Verifies no clipping or DC offset
    """
    if not segments:
        return y

    segments = sorted(segments, key=lambda s: s.dst_start)
    crossfade_samples = int(crossfade_ms * sr / 1000)

    output_len = int(video_duration * sr)

    # Build list of (dst_start, audio_chunk) pairs
    chunks: list[tuple[int, NDArray[np.floating]]] = []

    for seg in segments:
        retimed = retime_segment(y, sr, seg)
        if len(retimed) == 0:
            continue

        # Owned copy so in-place crossfade ops below are safe.
        retimed = np.array(retimed, dtype=np.float64)
        retimed = retimed - np.mean(retimed)

        dst_start_sample = int(seg.dst_start * sr)
        chunks.append((dst_start_sample, retimed))

    if not chunks:
        return np.zeros(output_len, dtype=np.float64)

    output = np.zeros(output_len, dtype=np.float64)

    for i, (dst_start, chunk) in enumerate(chunks):
        end = dst_start + len(chunk)

        if dst_start < 0:
            chunk = chunk[-dst_start:]
            dst_start = 0
        if end > output_len:
            chunk = chunk[: output_len - dst_start]
            end = dst_start + len(chunk)

        if len(chunk) == 0:
            continue

        if i > 0 and crossfade_samples > 0:
            prev_end = chunks[i - 1][0] + len(chunks[i - 1][1])
            overlap = prev_end - dst_start
            if overlap > 0:
                xf_len = min(crossfade_samples, overlap, len(chunk))
                fade_in = np.linspace(0.0, 1.0, xf_len)
                fade_out = np.linspace(1.0, 0.0, xf_len)
                output[dst_start : dst_start + xf_len] *= fade_out
                chunk[:xf_len] *= fade_in

        output[dst_start:end] += chunk

    peak = np.max(np.abs(output))
    if peak > 0.99:
        output = output * (0.99 / peak)
        log.debug("Normalized output peak from %.3f to 0.99", peak)

    return output
