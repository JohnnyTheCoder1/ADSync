"""Render the synced AD track from a continuous warp function.

For each output sample the renderer maps t_video → t_ad through the inverse
of the warp and interpolates the source.  Done in 30-second blocks so peak
memory stays bounded regardless of film length.  Mono and multi-channel AD
sources are both supported — the same warp is applied to every channel.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator

log = logging.getLogger("adsync")

_BLOCK_SECONDS = 30


def render_from_warp(
    y_ad: NDArray[np.floating],
    sr: int,
    warp_fns: list[PchipInterpolator],
    segment_ranges: list[tuple[float, float]],
    video_duration: float,
    *,
    crossfade_ms: int = 200,
) -> NDArray[np.floating]:
    """Render the synced AD waveform using continuous warp function(s)."""
    output_len = int(video_duration * sr)
    y_ad = np.ascontiguousarray(y_ad, dtype=np.float32)
    ad_len = y_ad.shape[-1]

    if y_ad.ndim == 1:
        output = np.zeros(output_len, dtype=np.float32)
    else:
        output = np.zeros((y_ad.shape[0], output_len), dtype=np.float32)

    if len(warp_fns) == 1:
        _render_single_segment(
            y_ad, sr, warp_fns[0], segment_ranges[0],
            video_duration, output, ad_len,
        )
    else:
        _render_multi_segment(
            y_ad, sr, warp_fns, segment_ranges,
            video_duration, output, ad_len, crossfade_ms,
        )

    dc = output.mean(axis=-1, keepdims=True)
    if np.any(np.abs(dc) > 1e-6):
        output -= dc.astype(np.float32)

    peak = float(np.max(np.abs(output)))
    if peak > 0.99:
        output *= np.float32(0.99 / peak)
        log.debug("Normalized output peak from %.3f to 0.99", peak)

    return output


def _render_single_segment(
    y_ad: NDArray[np.float32],
    sr: int,
    warp_fn: PchipInterpolator,
    segment_range: tuple[float, float],
    video_duration: float,
    output: NDArray[np.float32],
    ad_len: int,
) -> None:
    inverse_fn = _invert_pchip(warp_fn, segment_range, video_duration)

    vid_start = max(0.0, float(warp_fn(segment_range[0])))
    vid_end = min(video_duration, float(warp_fn(segment_range[1])))
    valid_lo = max(0, math.ceil(vid_start * sr))
    valid_hi = min(output.shape[-1], math.floor(vid_end * sr) + 1)

    block = sr * _BLOCK_SECONDS
    for start in range(valid_lo, valid_hi, block):
        end = min(start + block, valid_hi)
        _interp_into(y_ad, sr, inverse_fn, ad_len, output[..., start:end], start, end)


def _render_multi_segment(
    y_ad: NDArray[np.float32],
    sr: int,
    warp_fns: list[PchipInterpolator],
    segment_ranges: list[tuple[float, float]],
    video_duration: float,
    output: NDArray[np.float32],
    ad_len: int,
    crossfade_ms: int,
) -> None:
    crossfade_samples = int(crossfade_ms * sr / 1000)
    output_len = output.shape[-1]
    block = sr * _BLOCK_SECONDS
    buf_shape = (y_ad.shape[0], block) if y_ad.ndim == 2 else (block,)
    buf = np.empty(buf_shape, dtype=np.float32)
    placed: list[tuple[int, int]] = []

    for warp_fn, seg_range in zip(warp_fns, segment_ranges):
        src_start, src_end = seg_range
        vid_start = max(0.0, min(video_duration, float(warp_fn(src_start))))
        vid_end = max(0.0, min(video_duration, float(warp_fn(src_end))))
        if vid_end <= vid_start:
            continue

        out_start = max(0, min(output_len, math.ceil(vid_start * sr)))
        out_end = max(0, min(output_len, math.floor(vid_end * sr) + 1))
        if out_end <= out_start:
            continue

        inverse_fn = _invert_pchip(warp_fn, seg_range, video_duration)

        xf_len = 0
        fade_in: NDArray[np.float32] | None = None
        if placed and crossfade_samples > 0:
            overlap = placed[-1][1] - out_start
            if overlap > 0:
                xf_len = min(crossfade_samples, overlap, out_end - out_start)
                t = np.linspace(0.0, np.pi, xf_len, dtype=np.float32)
                output[..., out_start:out_start + xf_len] *= 0.5 * (1.0 + np.cos(t))
                fade_in = 0.5 * (1.0 - np.cos(t))

        for start in range(out_start, out_end, block):
            end = min(start + block, out_end)
            chunk = buf[..., : end - start]
            _interp_into(y_ad, sr, inverse_fn, ad_len, chunk, start, end)
            if fade_in is not None and start == out_start:
                chunk[..., :xf_len] *= fade_in
            output[..., start:end] += chunk

        placed.append((out_start, out_end))


def _interp_into(
    y_ad: NDArray[np.float32],
    sr: int,
    inverse_fn: PchipInterpolator,
    ad_len: int,
    dest: NDArray[np.float32],
    out_start: int,
    out_end: int,
) -> None:
    t = np.arange(out_start, out_end, dtype=np.float64)
    t /= sr
    src = inverse_fn(t)
    src *= sr
    np.clip(src, 0.0, ad_len - 1.0, out=src)
    idx = src.astype(np.int64)
    np.clip(idx, 0, ad_len - 2, out=idx)
    np.subtract(src, idx, out=src)
    frac = src.astype(np.float32)
    np.multiply(y_ad[..., idx], 1.0 - frac, out=dest)
    idx += 1
    dest += y_ad[..., idx] * frac


def _invert_pchip(
    forward_fn: PchipInterpolator,
    segment_range: tuple[float, float],
    video_duration: float,
    grid_step: float = 0.01,
) -> PchipInterpolator:
    """Invert a forward warp (AD→video) to get the inverse (video→AD)."""
    src_start, src_end = segment_range
    n_points = max(10, int((src_end - src_start) / grid_step))
    ad_times = np.linspace(src_start, src_end, n_points)
    vid_times = forward_fn(ad_times)

    # Guard against tiny numerical noise that can break strict monotonicity.
    for i in range(1, len(vid_times)):
        if vid_times[i] <= vid_times[i - 1]:
            vid_times[i] = vid_times[i - 1] + 1e-9

    return PchipInterpolator(vid_times, ad_times)
