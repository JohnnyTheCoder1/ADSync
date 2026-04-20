"""Render the synced AD track from a continuous warp function.

Instead of cutting, stretching, and placing independent chunks (which causes
seam artifacts), this renderer computes every output sample from a continuous
monotone warp function: for each output time t_video, look up the
corresponding AD source position t_ad = g(t_video) and interpolate.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator

log = logging.getLogger("adsync")


def render_from_warp(
    y_ad: NDArray[np.floating],
    sr: int,
    warp_fns: list[PchipInterpolator],
    segment_ranges: list[tuple[float, float]],
    video_duration: float,
    *,
    crossfade_ms: int = 200,
) -> NDArray[np.floating]:
    """Render the synced AD waveform using continuous warp function(s).

    Parameters
    ----------
    y_ad : NDArray
        Source AD audio at native sample rate.
    sr : int
        Sample rate.
    warp_fns : list[PchipInterpolator]
        Forward warp functions: AD time → video time.  One per segment.
    segment_ranges : list[tuple[float, float]]
        (start, end) in AD-time for each warp function.
    video_duration : float
        Duration of the output timeline in seconds.
    crossfade_ms : int
        Crossfade duration at segment boundaries (milliseconds).

    Returns
    -------
    NDArray[np.floating]
        The rendered output waveform aligned to the video timeline.
    """
    output_len = int(video_duration * sr)
    ad_len = len(y_ad)

    if len(warp_fns) == 1:
        output = _render_single_segment(
            y_ad, sr, warp_fns[0], segment_ranges[0],
            video_duration, output_len, ad_len,
        )
    else:
        output = _render_multi_segment(
            y_ad, sr, warp_fns, segment_ranges,
            video_duration, output_len, ad_len, crossfade_ms,
        )

    dc = np.mean(output)
    if abs(dc) > 1e-6:
        output = output - dc

    peak = np.max(np.abs(output))
    if peak > 0.99:
        output = output * (0.99 / peak)
        log.debug("Normalized output peak from %.3f to 0.99", peak)

    return output


def _render_single_segment(
    y_ad: NDArray,
    sr: int,
    warp_fn: PchipInterpolator,
    segment_range: tuple[float, float],
    video_duration: float,
    output_len: int,
    ad_len: int,
) -> NDArray[np.floating]:
    """Render from a single PCHIP warp function."""
    # Invert the warp: we have t_video = f(t_ad), need t_ad = g(t_video)
    inverse_fn = _invert_pchip(warp_fn, segment_range, video_duration)

    # Determine the valid output time range (where the inverse is defined)
    vid_start = float(warp_fn(segment_range[0]))
    vid_end = float(warp_fn(segment_range[1]))
    vid_start = max(0.0, vid_start)
    vid_end = min(video_duration, vid_end)

    # Generate all output sample times
    t_output = np.arange(output_len, dtype=np.float64) / sr

    # Map to source AD positions (only within valid domain)
    t_source = inverse_fn(t_output)

    # Zero out samples outside the valid warp domain instead of
    # letting PCHIP extrapolate (which can produce wild values)
    ad_duration = ad_len / sr
    valid_mask = (t_output >= vid_start) & (t_output <= vid_end)
    t_source = np.clip(t_source, 0.0, ad_duration - 1.0 / sr)

    # Convert to sample positions and interpolate
    source_samples = t_source * sr
    source_samples = np.clip(source_samples, 0, ad_len - 2)

    output = np.interp(source_samples, np.arange(ad_len), y_ad)
    # Silence regions outside the warp function's valid domain
    output[~valid_mask] = 0.0
    return output.astype(np.float64)


def _render_multi_segment(
    y_ad: NDArray,
    sr: int,
    warp_fns: list[PchipInterpolator],
    segment_ranges: list[tuple[float, float]],
    video_duration: float,
    output_len: int,
    ad_len: int,
    crossfade_ms: int,
) -> NDArray[np.floating]:
    """Render multiple warp segments with crossfades at boundaries."""
    crossfade_samples = int(crossfade_ms * sr / 1000)
    output = np.zeros(output_len, dtype=np.float64)

    # Render each segment
    rendered_segments: list[tuple[int, int, NDArray]] = []

    for i, (warp_fn, seg_range) in enumerate(zip(warp_fns, segment_ranges)):
        # Determine the output time range this segment covers
        src_start, src_end = seg_range
        vid_start = float(warp_fn(src_start))
        vid_end = float(warp_fn(src_end))

        vid_start = max(0.0, min(video_duration, vid_start))
        vid_end = max(0.0, min(video_duration, vid_end))

        if vid_end <= vid_start:
            continue

        out_start = int(vid_start * sr)
        out_end = int(vid_end * sr)
        out_start = max(0, min(output_len, out_start))
        out_end = max(0, min(output_len, out_end))

        if out_end <= out_start:
            continue

        # Invert the warp for this segment
        inverse_fn = _invert_pchip(warp_fn, seg_range, video_duration)

        # Render this segment's output samples
        t_output = np.arange(out_start, out_end, dtype=np.float64) / sr
        t_source = inverse_fn(t_output)
        source_samples = t_source * sr
        source_samples = np.clip(source_samples, 0, ad_len - 1)
        seg_audio = np.interp(source_samples, np.arange(ad_len), y_ad)

        rendered_segments.append((out_start, out_end, seg_audio.astype(np.float64)))

    # Place segments with crossfades
    for i, (out_start, out_end, seg_audio) in enumerate(rendered_segments):
        seg_len = len(seg_audio)

        if i > 0 and crossfade_samples > 0:
            # Check overlap with previous segment
            prev_start, prev_end, _ = rendered_segments[i - 1]
            overlap = prev_end - out_start
            if overlap > 0:
                xf_len = min(crossfade_samples, overlap, seg_len)
                # Raised-cosine (Hann) crossfade for perceptual smoothness
                t = np.linspace(0, np.pi, xf_len)
                fade_in = 0.5 * (1.0 - np.cos(t))
                fade_out = 0.5 * (1.0 + np.cos(t))
                output[out_start: out_start + xf_len] *= fade_out
                seg_audio[:xf_len] *= fade_in

        output[out_start:out_end] += seg_audio[:out_end - out_start]

    return output


def _invert_pchip(
    forward_fn: PchipInterpolator,
    segment_range: tuple[float, float],
    video_duration: float,
    grid_step: float = 0.01,
) -> PchipInterpolator:
    """Invert a forward warp (AD→video) to get the inverse (video→AD).

    Since the forward PCHIP is monotone, the inverse exists.  We evaluate
    the forward function on a dense grid and build a new PCHIP from the
    swapped (video_time, ad_time) pairs.
    """
    src_start, src_end = segment_range
    n_points = max(10, int((src_end - src_start) / grid_step))
    ad_times = np.linspace(src_start, src_end, n_points)
    vid_times = forward_fn(ad_times)

    # Guard against tiny numerical noise that can break strict monotonicity.
    for i in range(1, len(vid_times)):
        if vid_times[i] <= vid_times[i - 1]:
            vid_times[i] = vid_times[i - 1] + 1e-9

    return PchipInterpolator(vid_times, ad_times)
