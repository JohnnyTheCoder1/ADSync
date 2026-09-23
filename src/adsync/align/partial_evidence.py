"""Recover local landmark evidence without extending it into adjacent gaps."""

from __future__ import annotations

import numpy as np

from adsync.models import CandidateWindow, OffsetCandidate


def augment_partial_evidence(lattice: list[CandidateWindow], fp, *, window_sec: float = 2.,
                             max_candidates: int = 5) -> int:
    source = getattr(fp, "raw_match_t_ad", None)
    target = getattr(fp, "raw_match_t_vid", None)
    if source is None or target is None or not len(source):
        return 0
    source, target = np.asarray(source), np.asarray(target)
    finite = np.isfinite(source) & np.isfinite(target)
    order = np.argsort(source[finite], kind="stable")
    source, offsets = source[finite][order], (target[finite] - source[finite])[order]
    half = window_sec / 2
    augmented = 0
    for window in lattice:
        if window.candidates:
            continue
        center = window.source_center
        lo, hi = np.searchsorted(source, [center - half, center + half])
        times, shifts = source[lo:hi], offsets[lo:hi]
        remaining = np.ones(len(times), dtype=bool)
        for _ in range(max_candidates):
            if np.count_nonzero(remaining) < 6:
                break
            bins, counts = np.unique(np.rint(shifts[remaining] / .032), return_counts=True)
            peak = bins[np.argmax(counts)] * .032
            cluster = remaining & (np.abs(shifts - peak) <= .048)
            remaining[cluster] = False
            local_times = times[cluster]
            distinct = np.unique(np.floor(local_times / .05))
            if (len(distinct) < 6 or np.ptp(local_times) < half
                    or np.count_nonzero(distinct < center / .05) < 2
                    or np.count_nonzero(distinct >= center / .05) < 2):
                continue
            offset = float(np.median(shifts[cluster]))
            if any(abs(c.offset_sec - offset) <= .05 for c in window.candidates):
                continue
            window.candidates.append(OffsetCandidate(
                offset_sec=offset, score=min(.75, .45 + .02 * len(distinct)),
                peak_sharpness=1., peak_ratio=1., source="fingerprint",
            ))
        if window.candidates:
            augmented += 1
    return augmented
