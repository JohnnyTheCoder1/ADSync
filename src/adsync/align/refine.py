"""Refine uncertain edit boundaries with independent short waveform windows."""

from __future__ import annotations

import logging

import numpy as np
from scipy.interpolate import PchipInterpolator

from adsync.align.fingerprint import _terminal_waveform_support

log = logging.getLogger("adsync")


def refine_warp_boundaries(warp_fns, segment_ranges, warp_path, y_vid, y_ad, sr) -> list[dict]:
    """Measure both sides of each cut; mutate ranges and report points together.

    Windowed correlation and ten-second fingerprint votes can put a jump too
    early inside a quiet gap. A bounded local search compares both fitted
    hypotheses using two disjoint waveform subwindows. The midpoint between
    last-left and first-right support is explicit uncertainty, not a claimed
    sample-exact cut. Unsupported boundaries remain unchanged and reviewable.
    """
    diagnostics = []
    for index in range(len(segment_ranges) - 1):
        left_fn, right_fn = warp_fns[index:index + 2]
        left_start, boundary = segment_ranges[index]
        right_start, right_end = segment_ranges[index + 1]
        if abs(right_start - boundary) > 1e-5:
            continue
        jump = float(right_fn(boundary) - left_fn(boundary))
        if abs(jump) < .25:
            continue
        low = max(left_start, boundary - 30)
        high = min(right_end - 1, boundary + 30)
        left_support, right_support = [], []
        for start in np.arange(np.ceil(low), np.floor(high) + .1, 1):
            scores = []
            for fn in (left_fn, right_fn):
                offset = float(fn(start + .5) - start - .5)
                scores.append(_terminal_waveform_support(y_vid, y_ad, sr, start, start + 1, offset))
            a, b = scores
            # Ambiguous repeated music cannot pick a boundary. One hypothesis
            # must be clearly supported and beat the other at the same time.
            if a is not None and (b is None or a > b + .15):
                left_support.append((float(start), float(a)))
            if b is not None and (a is None or b > a + .15):
                right_support.append((float(start), float(b)))
        record = {"original_boundary_sec": boundary, "jump_sec": jump,
                  "left_support": left_support, "right_support": right_support,
                  "status": "unresolved"}
        diagnostics.append(record)
        if len(left_support) < 2 or len(right_support) < 2:
            continue
        last_left = max(t for t, _ in left_support) + 1
        first_right = min(t for t, _ in right_support)
        if first_right < last_left - .01 or first_right - last_left > 30:
            continue
        refined = (last_left + first_right) / 2
        if not left_start < refined < right_end:
            continue
        # PCHIP extrapolation is not trustworthy far outside its last knot.
        # Extend with the measured local slope, preserving every existing knot.
        updated_functions = {}
        for slot, endpoint, after in ((index, refined, True), (index + 1, refined, False)):
            fn = warp_fns[slot]
            if (after and endpoint > fn.x[-1]) or (not after and endpoint < fn.x[0]):
                edge = fn.x[-1] if after else fn.x[0]
                slope = float(fn.derivative()(edge))
                if not .5 <= slope <= 2:
                    break
                target = float(fn(edge)) + (endpoint - edge) * slope
                xs, ys = fn.x.copy(), fn(fn.x)
                xs = np.r_[xs, endpoint] if after else np.r_[endpoint, xs]
                ys = np.r_[ys, target] if after else np.r_[target, ys]
                updated_functions[slot] = PchipInterpolator(xs, ys)
        else:
            for slot, fn in updated_functions.items():
                warp_fns[slot] = fn
            segment_ranges[index] = (left_start, refined)
            segment_ranges[index + 1] = (refined, right_end)
            for attr in ("points", "anchor_points"):
                revised = []
                for point in getattr(warp_path, attr):
                    if min(boundary, refined) <= point.source_time <= max(boundary, refined):
                        fn = warp_fns[index if point.source_time < refined else index + 1]
                        point = point.model_copy(update={"target_time": float(fn(point.source_time))})
                    revised.append(point)
                setattr(warp_path, attr, revised)
            record.update(status="refined", boundary_sec=refined,
                          bracket_sec=[last_left, first_right], uncertainty_sec=first_right - last_left)
            log.info("Refined edit boundary %.2f -> %.2f s (independent bracket %.2f–%.2f s)",
                     boundary, refined, last_left, first_right)
    return diagnostics
