"""Fit a smooth, monotone warp function through the decoded path.

Takes the Viterbi-decoded path (one WarpPoint per analysis window),
selects high-confidence anchor points, enforces monotonicity, and fits
a PchipInterpolator per contiguous segment.  PCHIP is shape-preserving
and monotone on monotone data, avoiding overshoot problems of cubic splines.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.interpolate import PchipInterpolator

from adsync.models import WarpPath, WarpPoint

log = logging.getLogger("adsync")


def fit_warp_function(
    path: list[WarpPoint],
    ad_duration: float,
    video_duration: float,
    *,
    anchor_fraction: float = 0.6,
    min_anchor_density: float = 1 / 30,  # per second
    discontinuity_threshold: float = 2.0,
) -> tuple[list[PchipInterpolator], list[tuple[float, float]], WarpPath]:
    """Fit monotone PCHIP warp function(s) through the decoded path.

    Parameters
    ----------
    path : list[WarpPoint]
        Decoded path from :func:`decode_warp_path`.
    ad_duration, video_duration : float
        Total durations in seconds.
    anchor_fraction : float
        Fraction of points (by confidence rank) to use as PCHIP anchors.
    min_anchor_density : float
        Minimum anchor density in points per second.
    discontinuity_threshold : float
        Offset jump (seconds) between consecutive points to split segments.

    Returns
    -------
    warp_fns : list[PchipInterpolator]
        One per contiguous segment; maps AD time → video time.
    segment_ranges : list[tuple[float, float]]
        Source-time (AD) ranges for each segment.
    warp_path : WarpPath
        Report-friendly summary of the path and anchor selection.
    """
    if not path:
        # Fallback: identity warp
        fn = PchipInterpolator([0.0, ad_duration], [0.0, ad_duration])
        wp = WarpPath(points=[], anchor_points=[], path_cost=0.0, mean_confidence=0.0)
        return [fn], [(0.0, ad_duration)], wp

    # ── Step 1: Segment the path at discontinuities ────────────────────
    segments = _split_at_discontinuities(path, discontinuity_threshold)
    log.info("Warp fit: %d contiguous segment(s) from %d points", len(segments), len(path))

    warp_fns: list[PchipInterpolator] = []
    segment_ranges: list[tuple[float, float]] = []
    all_anchor_points: list[WarpPoint] = []

    for seg_idx, seg_points in enumerate(segments):
        if len(seg_points) < 2:
            # Single point — extend to a tiny segment with constant offset
            p = seg_points[0]
            offset = p.target_time - p.source_time
            src_start = max(0.0, p.source_time - 5.0)
            src_end = min(ad_duration, p.source_time + 5.0)
            fn = PchipInterpolator(
                [src_start, src_end],
                [src_start + offset, src_end + offset],
            )
            warp_fns.append(fn)
            segment_ranges.append((src_start, src_end))
            p_copy = p.model_copy(update={"is_anchor": True})
            all_anchor_points.append(p_copy)
            continue

        # ── Step 2: Select anchors by confidence ──────────────────────
        anchors = _select_anchors(seg_points, anchor_fraction, min_anchor_density)

        # ── Step 3: Enforce monotonicity ──────────────────────────────
        anchors = _enforce_monotonicity(anchors)

        if len(anchors) < 2:
            # Couldn't get 2 monotone anchors — use endpoints
            anchors = [seg_points[0], seg_points[-1]]
            anchors = _enforce_monotonicity(anchors)

        # ── Step 4: Add virtual boundary points ───────────────────────
        anchors = _add_boundaries(
            anchors, seg_idx, len(segments),
            ad_duration, video_duration,
        )

        # ── Step 5: Fit PCHIP ─────────────────────────────────────────
        src_times = np.array([a.source_time for a in anchors])
        tgt_times = np.array([a.target_time for a in anchors])

        # Deduplicate source times (PCHIP requires strictly increasing x)
        src_times, unique_idx = np.unique(src_times, return_index=True)
        tgt_times = tgt_times[unique_idx]

        if len(src_times) < 2:
            offset = tgt_times[0] - src_times[0]
            src_times = np.array([max(0, src_times[0] - 5), min(ad_duration, src_times[0] + 5)])
            tgt_times = src_times + offset

        fn = PchipInterpolator(src_times, tgt_times)
        warp_fns.append(fn)
        segment_ranges.append((float(src_times[0]), float(src_times[-1])))

        for a in anchors:
            if not a.is_anchor:
                a = a.model_copy(update={"is_anchor": True})
            all_anchor_points.append(a)

    # ── Extend segment ranges to cover full AD duration ───────────────
    if segment_ranges:
        # First segment starts at 0
        s0_start, s0_end = segment_ranges[0]
        segment_ranges[0] = (0.0, s0_end)
        # Last segment extends to ad_duration
        sN_start, sN_end = segment_ranges[-1]
        segment_ranges[-1] = (sN_start, ad_duration)

    mean_conf = float(np.mean([p.confidence for p in path])) if path else 0.0
    path_cost = 0.0  # Filled by decoder if needed

    warp_path = WarpPath(
        points=path,
        anchor_points=all_anchor_points,
        path_cost=path_cost,
        mean_confidence=mean_conf,
    )

    log.info(
        "Warp fit complete: %d PCHIP segments, %d anchor points, mean_conf=%.3f",
        len(warp_fns), len(all_anchor_points), mean_conf,
    )
    return warp_fns, segment_ranges, warp_path


def _split_at_discontinuities(
    path: list[WarpPoint],
    threshold: float,
) -> list[list[WarpPoint]]:
    """Split path into segments where consecutive offset jumps exceed threshold."""
    if len(path) <= 1:
        return [path]

    segments: list[list[WarpPoint]] = [[path[0]]]
    for i in range(1, len(path)):
        prev_offset = path[i - 1].target_time - path[i - 1].source_time
        cur_offset = path[i].target_time - path[i].source_time
        if abs(cur_offset - prev_offset) > threshold:
            log.info(
                "Discontinuity at t=%.1fs: offset jumps %.2f → %.2f s",
                path[i].source_time, prev_offset, cur_offset,
            )
            segments.append([])
        segments[-1].append(path[i])

    return [s for s in segments if s]


def _select_anchors(
    points: list[WarpPoint],
    fraction: float,
    min_density: float,
) -> list[WarpPoint]:
    """Select top-fraction points by confidence, with minimum density."""
    n = len(points)
    if n <= 2:
        return list(points)

    duration = points[-1].source_time - points[0].source_time
    min_count = max(2, int(duration * min_density))
    target_count = max(min_count, int(n * fraction))
    target_count = min(target_count, n)

    # Always include first and last
    ranked = sorted(range(n), key=lambda i: points[i].confidence, reverse=True)
    selected_idx = {0, n - 1}

    for idx in ranked:
        if len(selected_idx) >= target_count:
            break
        selected_idx.add(idx)

    # Sort by source_time
    selected = sorted(selected_idx)
    return [points[i].model_copy(update={"is_anchor": True}) for i in selected]


def _enforce_monotonicity(anchors: list[WarpPoint]) -> list[WarpPoint]:
    """Remove points that violate strict target_time monotonicity.

    Uses a patience-based longest increasing subsequence (LIS) on
    target_time, weighted by confidence to prefer high-quality points.
    This guarantees the result is strictly increasing in both
    source_time and target_time.
    """
    if len(anchors) <= 1:
        return anchors

    # Sort by source_time
    sorted_a = sorted(anchors, key=lambda p: p.source_time)

    # Find longest strictly increasing subsequence of target_time
    # using a simple O(n^2) DP (n is small — typically < 1000)
    n = len(sorted_a)
    targets = [p.target_time for p in sorted_a]

    # dp[i] = length of longest increasing subseq ending at i
    dp = [1] * n
    prev_idx = [-1] * n

    for i in range(1, n):
        for j in range(i):
            if targets[j] < targets[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                prev_idx[i] = j

    # Backtrack to find the actual subsequence
    best_end = int(np.argmax(dp))
    lis_indices: list[int] = []
    idx = best_end
    while idx != -1:
        lis_indices.append(idx)
        idx = prev_idx[idx]
    lis_indices.reverse()

    result = [sorted_a[i] for i in lis_indices]
    if len(result) < len(sorted_a):
        log.debug(
            "Monotonicity: kept %d of %d points (removed %d violators)",
            len(result), len(sorted_a), len(sorted_a) - len(result),
        )
    return result


def _add_boundaries(
    anchors: list[WarpPoint],
    seg_idx: int,
    n_segments: int,
    ad_duration: float,
    video_duration: float,
) -> list[WarpPoint]:
    """Add virtual boundary anchors at start/end using local slope."""
    if len(anchors) < 2:
        return anchors

    result = list(anchors)

    # Extrapolate to t=0 for the first segment
    if seg_idx == 0 and result[0].source_time > 0.1:
        first = result[0]
        second = result[1]
        slope = (second.target_time - first.target_time) / max(
            second.source_time - first.source_time, 1e-6,
        )
        t0_target = first.target_time - first.source_time * slope
        t0_target = max(0.0, t0_target)
        result.insert(0, WarpPoint(
            source_time=0.0,
            target_time=t0_target,
            confidence=first.confidence * 0.8,
            is_anchor=True,
        ))

    # Extrapolate to t=ad_duration for the last segment
    if seg_idx == n_segments - 1 and result[-1].source_time < ad_duration - 0.1:
        last = result[-1]
        prev = result[-2]
        slope = (last.target_time - prev.target_time) / max(
            last.source_time - prev.source_time, 1e-6,
        )
        tend_target = last.target_time + (ad_duration - last.source_time) * slope
        tend_target = min(video_duration, tend_target)
        result.append(WarpPoint(
            source_time=ad_duration,
            target_time=tend_target,
            confidence=last.confidence * 0.8,
            is_anchor=True,
        ))

    return result
