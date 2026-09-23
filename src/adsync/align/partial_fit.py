"""Fit measured partial runs without extrapolating across unsupported content."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PPoly

from adsync.align.partial_decode import ALGORITHM, MAX_CANDIDATES, TIMING_EQUIVALENCE_SEC, decode_partial_path
from adsync.models import CandidateWindow, WarpPath, WarpPoint


def _gaps(intervals, duration):
    end = 0.
    gaps = []
    for lo, hi in intervals:
        if lo > end + 1e-8:
            gaps.append({"start_sec": float(end), "end_sec": float(lo), "reason": "unmeasured"})
        end = max(end, hi)
    if end < duration - 1e-8:
        gaps.append({"start_sec": float(end), "end_sec": float(duration), "reason": "unmeasured"})
    return gaps


def fit_partial_alignment(
    lattice: list[CandidateWindow],
    ad_duration: float,
    video_duration: float,
    *,
    window_sec: float = 2.,
    step_sec: float = 1.,
    max_stretch: float = .01,
):
    """Return partial linear functions, source ranges, path and JSON diagnostics.

    Every interval's measured rate is bounded by ``max_stretch``. Window support limits
    the map; adjacent runs are clipped in both clocks so their render regions
    cannot overlap. A gap records absent alignment evidence, not proven deletion.
    """
    decoded = decode_partial_path(lattice, ad_duration, video_duration,
        window_sec=window_sec, step_sec=step_sec, max_stretch=max_stretch)
    half = window_sec / 2
    threshold = .1 * step_sec
    fits = []
    for indices in decoded.runs:
        nodes = [decoded.nodes[i] for i in indices]
        xs = np.array([n.source_time for n in nodes])
        ys = np.array([n.target_time for n in nodes])
        slopes = np.diff(ys) / np.diff(xs)
        lo, hi = max(0., float(xs[0] - half)), min(ad_duration, float(xs[-1] + half))
        if lo < xs[0]:
            ys = np.r_[ys[0] + (lo - xs[0]) * slopes[0], ys]
            xs = np.r_[lo, xs]
        if hi > xs[-1]:
            ys = np.r_[ys, ys[-1] + (hi - xs[-1]) * slopes[-1]]
            xs = np.r_[xs, hi]
        lo = max(lo, float(np.interp(0, ys, xs)))
        hi = min(hi, float(np.interp(video_duration, ys, xs)))
        fits.append({"nodes": nodes, "xs": xs, "ys": ys, "lo": lo, "hi": hi})

    for left, right in zip(fits, fits[1:]):
        if left["hi"] > right["lo"]:
            boundary = .5 * (left["nodes"][-1].source_time + right["nodes"][0].source_time)
            left["hi"] = min(left["hi"], boundary)
            right["lo"] = max(right["lo"], boundary)
        lend = np.interp(left["hi"], left["xs"], left["ys"])
        rstart = np.interp(right["lo"], right["xs"], right["ys"])
        if lend > rstart:
            target_boundary = .5 * (left["nodes"][-1].target_time + right["nodes"][0].target_time)
            left["hi"] = min(left["hi"], float(np.interp(target_boundary, left["ys"], left["xs"])))
            right["lo"] = max(right["lo"], float(np.interp(target_boundary, right["ys"], right["xs"])))

    fns, ranges, points, matched, ambiguous = [], [], [], [], []
    chosen = {}
    for fit in fits:
        lo, hi = fit["lo"], fit["hi"]
        if hi <= lo:
            continue
        xs = np.r_[lo, fit["xs"][(fit["xs"] > lo) & (fit["xs"] < hi)], hi]
        ys = np.interp(xs, fit["xs"], fit["ys"])
        slopes = np.diff(ys) / np.diff(xs)
        target_lo, target_hi = ys[0], ys[-1]
        run_index = len(fns)
        # Linear pieces preserve measured coordinates and rate bounds; cubic
        # slopes can overshoot the permitted rates near a drift change.
        fns.append(PPoly(np.vstack([slopes, ys[:-1]]), xs))
        ranges.append((float(lo), float(hi)))
        uncertain = False
        confidences = []
        for node in fit["nodes"]:
            margin = decoded.window_margins[node.window_index]
            is_ambiguous = margin is not None and margin < threshold
            uncertain |= is_ambiguous
            confidence = node.score * (.25 if is_ambiguous else 1.)
            confidences.append(confidence)
            points.append(WarpPoint(source_time=node.source_time, target_time=node.target_time,
                                    confidence=confidence, is_anchor=True))
            chosen[node.window_index] = {"candidate_index": node.candidate_index,
                "target_time": node.target_time, "score_margin": margin,
                "ambiguous": is_ambiguous, "run_index": run_index, "provenance": node.provenance}
            if is_ambiguous:
                start, end = max(lo, node.source_time - half), min(hi, node.source_time + half)
                if ambiguous and start <= ambiguous[-1]["end_sec"] + 1e-8:
                    ambiguous[-1]["end_sec"] = float(end)
                    ambiguous[-1]["score_margin"] = min(ambiguous[-1]["score_margin"], margin)
                else:
                    ambiguous.append({"start_sec": float(start), "end_sec": float(end),
                                      "reason": "competing_path", "score_margin": float(margin)})
        matched.append({"source_start": float(lo), "source_end": float(hi),
            "target_start": float(target_lo), "target_end": float(target_hi),
            "point_count": len(fit["nodes"]), "confidence": float(np.mean(confidences)),
            "ambiguous": bool(uncertain), "rate": float((target_hi - target_lo) / (hi - lo)),
            "min_rate": float(np.min(slopes)), "max_rate": float(np.max(slopes)),
            "max_residual_sec": float(max(abs(float(fns[-1](n.source_time)) - n.target_time) for n in fit["nodes"]))})
    windows = [{"source_time": float(window.source_center), **chosen.get(i, {
        "candidate_index": None, "target_time": None, "score_margin": None,
        "ambiguous": False, "run_index": None, "provenance": None})} for i, window in enumerate(lattice)]
    path = WarpPath(points=points, anchor_points=points, path_cost=-decoded.path_score,
        mean_confidence=float(np.mean([p.confidence for p in points])) if points else 0., n_segments=len(fns))
    diagnostics = {"algorithm": ALGORITHM,
        "status": "unmeasured" if not matched else "ambiguous" if ambiguous else "matched",
        "matched_intervals": matched, "source_gaps": _gaps(ranges, ad_duration),
        "target_gaps": _gaps([(m["target_start"], m["target_end"]) for m in matched], video_duration),
        "ambiguous_ranges": ambiguous, "path_score": float(decoded.path_score),
        "candidates": {"input": decoded.input_candidates, "eligible": len(decoded.nodes),
                       "selected": len(decoded.selected), "limit": MAX_CANDIDATES},
        "windows": windows, "ambiguity_margin_threshold": threshold,
        "timing_equivalence_sec": TIMING_EQUIVALENCE_SEC,
        "boundary_uncertainty_sec": half}
    return fns, ranges, path, diagnostics
