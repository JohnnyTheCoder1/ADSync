"""Stitch retimed segments with crossfades and silence fills."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from adsync.models import SegmentMap
from adsync.rebuild.retime import retime_segment

log = logging.getLogger("adsync")


def stitch_segments(
    y: NDArray[np.floating],
    sr: int,
    segments: list[SegmentMap],
    video_duration: float,
    *,
    crossfade_ms: int = 80,
) -> NDArray[np.float32]:
    """Build the final synced AD waveform from segments with crossfades.

    Inserts silence for gaps, crossfades joins, and clamps the peak.  Accepts
    1-D (mono) or 2-D (channels, samples) input and returns the same layout.
    An empty map yields silence for the video duration. Each retimed segment
    is released after placement, without retaining a second full track.
    """
    crossfade_samples = int(crossfade_ms * sr / 1000)
    output_len = int(video_duration * sr)
    output_shape = (y.shape[0], output_len) if y.ndim == 2 else (output_len,)
    output = np.zeros(output_shape, dtype=np.float32)
    if not segments or output_len == 0:
        return output

    # Keep the furthest occupied endpoint, including the tail of a longer
    # segment that surrounds a shorter one. Tracking only the last segment
    # can otherwise mix an older tail at full volume beneath a later segment.
    placed_end = 0
    for seg in sorted(segments, key=lambda s: s.dst_start):
        dst_start = int(seg.dst_start * sr)
        mapped_end = seg.dst_start + (seg.src_end - seg.src_start) * seg.stretch
        if dst_start >= output_len or mapped_end <= 0.0:
            continue

        chunk = retime_segment(y, sr, seg)
        chunk_len = chunk.shape[-1]
        lo = max(0, -dst_start)
        hi = min(chunk_len, output_len - dst_start)
        if hi <= lo:
            del chunk
            continue

        chunk -= chunk.mean(axis=-1, keepdims=True).astype(np.float32)
        chunk = chunk[..., lo:hi]
        dst_start += lo
        end = dst_start + chunk.shape[-1]

        overlap = placed_end - dst_start
        if overlap > 0:
            xf_len = min(crossfade_samples, overlap, chunk.shape[-1])
            if xf_len > 0:
                fade_in = np.linspace(0.0, 1.0, xf_len, dtype=np.float32)
                fade_out = np.linspace(1.0, 0.0, xf_len, dtype=np.float32)
                output[..., dst_start: dst_start + xf_len] *= fade_out
                chunk[..., :xf_len] *= fade_in
            # The new segment owns its destination range after the fade.
            stale_end = min(placed_end, end)
            if stale_end > dst_start + xf_len:
                output[..., dst_start + xf_len: stale_end] = 0.0

        output[..., dst_start:end] += chunk
        placed_end = max(placed_end, end)
        del chunk

    # Avoid allocating another full HQ-track array just to inspect its peak.
    block = 30 * sr
    peak = max(
        float(np.max(np.abs(output[..., start:start + block])))
        for start in range(0, output_len, block)
    )
    if peak > 0.99:
        output *= np.float32(0.99 / peak)
        log.debug("Normalized output peak from %.3f to 0.99", peak)

    return output
