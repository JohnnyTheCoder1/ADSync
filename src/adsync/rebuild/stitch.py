"""Stitch retimed segments with crossfades and silence fills."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from adsync.models import SegmentMap
from adsync.rebuild.retime import retime_segment
from adsync.utils.mathx import EDGE_FADE_MS, fade_edges

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
        # Round both ends from absolute times so touching segments share a sample boundary.
        mapped_end = seg.dst_start + (seg.src_end - seg.src_start) * seg.stretch
        dst_start = max(0, round(seg.dst_start * sr))
        end = min(output_len, round(mapped_end * sr))
        if end <= dst_start:
            continue

        chunk = retime_segment(y, sr, seg, out_start=dst_start, length=end - dst_start)
        chunk -= chunk.mean(axis=-1, keepdims=True).astype(np.float32)

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

        elif overlap < 0:
            # Silence before this segment: fade out what precedes it and fade this one in.
            fade = int(EDGE_FADE_MS * sr / 1000)
            fade_edges(output, max(0, placed_end - fade), placed_end, sr, head=False, tail=True)
            fade_edges(chunk, 0, chunk.shape[-1], sr, head=True, tail=False)

        output[..., dst_start:end] += chunk
        placed_end = max(placed_end, end)
        del chunk

    if 0 < placed_end < output_len:
        fade = int(EDGE_FADE_MS * sr / 1000)
        fade_edges(output, max(0, placed_end - fade), placed_end, sr, head=False, tail=True)

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
