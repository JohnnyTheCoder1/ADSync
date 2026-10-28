"""Narrow verification targets from coherent raw fingerprint evidence.

Alignment's broad offset inlier filtering is useful for fitting the main map,
but it can hide a short rendered error beyond that filter's radius. Target
planning therefore consumes the raw matched pairs before those votes disappear.
These are probe requests, never publication verdicts or inferred corrections.
"""

from __future__ import annotations

import math

import numpy as np


QC_TARGETS_REVISION = "raw-local-offset-targets-v1"


def raw_offset_windows(
    t_ad: np.ndarray | None,
    t_main: np.ndarray | None,
    duration: float,
    *,
    tolerance_sec: float = .150,
    window_sec: float = 2.0,
    hop_sec: float = .5,
    max_probe_sec: float = 4.0,
) -> list[dict]:
    """Locate short, locally dominant nonzero raw-offset clusters.

    At least eight matching votes at four distinct anchor times must span half
    a second and form 65% of the local raw matches. Duplicate hashes at one
    instant, ordinary sparse support and scattered coincidences cannot satisfy
    this temporal evidence rule. The thresholds request independent audio
    verification; they do not label a timing error by themselves.

    Overlapping detections at the same offset form one supported run. A long
    run gets at most three short probes (beginning, middle and end), avoiding an
    exhaustive waveform scan or a broad template that dilutes a short error.
    ``t_main - t_ad`` uses the same sign convention as the fingerprint aligner.
    """
    if not all(math.isfinite(value) and value > 0 for value in
               (duration, tolerance_sec, window_sec, hop_sec, max_probe_sec)):
        raise ValueError("Target duration and time scales must be finite and positive")
    if hop_sec > window_sec or max_probe_sec < window_sec or window_sec < .5:
        raise ValueError("Target hop must fit its window and probes must fit the window scale")
    if t_ad is None and t_main is None:
        return []
    if t_ad is None or t_main is None:
        raise ValueError("Both raw matched-pair arrays are required")
    ad, main = np.asarray(t_ad, dtype=np.float64), np.asarray(t_main, dtype=np.float64)
    if ad.ndim != 1 or main.ndim != 1 or ad.shape != main.shape:
        raise ValueError("Raw matched-pair arrays must be equally sized one-dimensional arrays")
    keep = np.isfinite(ad) & np.isfinite(main) & (ad >= 0) & (ad < duration)
    ad, main = ad[keep], main[keep]
    if not len(ad):
        return []
    order = np.argsort(ad, kind="stable")
    ad, offsets = ad[order], (main - ad)[order]
    radius = .08  # Fingerprint quantization spread, not a timing acceptance limit.
    groups: list[dict] = []
    # Searchsorted avoids repeatedly scanning the whole episode's raw-pair set.
    for start in np.arange(0, duration, hop_sec):
        first, last = np.searchsorted(ad, [start, min(duration, start + window_sec)])
        if last - first < 8:
            continue
        local = offsets[first:last]
        bins, counts = np.unique(np.round(local / .05), return_counts=True)
        center = float(bins[int(np.argmax(counts))] * .05)
        cluster = np.abs(local - center) <= radius
        if not cluster.any():
            continue
        center = float(np.median(local[cluster]))
        cluster = np.abs(local - center) <= radius
        indices = np.flatnonzero(cluster) + first
        times = ad[indices]
        fraction = len(indices) / (last - first)
        if (abs(center) <= tolerance_sec or len(indices) < 8 or fraction < .65
                or len(np.unique(times)) < 4 or float(np.ptp(times)) < .5):
            continue
        support_start, support_end = float(times[0]), float(times[-1])
        previous = next((row for row in reversed(groups)
                         if support_start <= row["end"] + hop_sec
                         and abs(center - row["offset"]) <= radius), None)
        if previous is not None:
            previous["end"] = max(previous["end"], support_end)
            previous["indices"].update(indices.tolist())
            previous["min_fraction"] = min(previous["min_fraction"], fraction)
        else:
            groups.append({"start": support_start, "end": support_end, "offset": center,
                           "indices": set(indices.tolist()), "min_fraction": fraction})

    targets: list[dict] = []
    for group in groups:
        indices = np.asarray(sorted(group["indices"]), dtype=np.int64)
        lag = float(np.median(offsets[indices]))
        low = max(0.0, math.floor(group["start"] / hop_sec) * hop_sec)
        high = min(duration, math.ceil(group["end"] / hop_sec) * hop_sec)
        if high - low <= max_probe_sec:
            probes = [(low, high)]
        else:
            middle = float(np.median(ad[indices]))
            centers = [low + max_probe_sec / 2, middle, high - max_probe_sec / 2]
            starts = sorted({max(low, min(high - max_probe_sec, center - max_probe_sec / 2))
                             for center in centers})
            probes = [(start, start + max_probe_sec) for start in starts]
        for start, end in probes:
            selected = indices[(ad[indices] >= start) & (ad[indices] <= end)]
            if not len(selected):
                continue
            targets.append({"start_sec": float(start), "end_sec": float(end),
                            "center_sec": float((start + end) / 2),
                            "reasons": ["raw_nonzero_fingerprint_cluster"],
                            "suspected_offset_sec": lag,
                            "raw_support": {"matches": len(selected),
                                            "distinct_anchor_times": len(np.unique(ad[selected])),
                                            "span_sec": float(np.ptp(ad[selected])),
                                            "min_local_fraction": float(group["min_fraction"]),
                                            "run_start_sec": group["start"], "run_end_sec": group["end"],
                                            "requires_independent_verification": True}})
    return sorted(targets, key=lambda row: (row["start_sec"], row["end_sec"], row["suspected_offset_sec"]))
