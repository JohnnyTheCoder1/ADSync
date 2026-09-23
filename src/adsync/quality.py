"""Content identity and calibrated, local verification of rendered audio.

Fingerprint steering confidence is deliberately not an identity or publication
decision. These checks retain their measurements so an inconclusive region can
be investigated without repeating an entire episode's analysis.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Iterable

import numpy as np
from scipy.signal import butter, fftconvolve, sosfiltfilt, resample_poly
from scipy.ndimage import uniform_filter1d

from adsync.align.fingerprint import FingerprintResult, fingerprint_align
from adsync.models import MediaInfo, StreamInfo
from adsync.quality_policy import QC_DECISION_POLICY_REVISION, evaluate_quality_policy
from adsync.quality_targets import QC_TARGETS_REVISION, raw_offset_windows


QC_POLICY_REVISION = "distributed-identity-local-signal-v6+" + QC_DECISION_POLICY_REVISION + "+" + QC_TARGETS_REVISION
FEATURE_REVISION = "landmarks-1024-256-v1"


def cached_landmarks(y: np.ndarray, sr: int, *, cache: Any = None) -> tuple[np.ndarray, np.ndarray]:
    """Reuse content-bound source features across preflight, alignment and QC."""
    from adsync.align.fingerprint import _landmarks
    from adsync.cache import default_cache, derived_key

    audio = np.ascontiguousarray(y, dtype=np.float32)
    cache = default_cache() if cache is None else cache
    if cache is None:
        return _landmarks(audio, sr)
    digest = hashlib.sha256(memoryview(audio).cast("B")).hexdigest()
    key = derived_key(digest, {"kind": "landmarks", "sr": sr, "samples": len(audio)}, revision=FEATURE_REVISION)
    existing = cache.get(key)
    if (existing is not None and {"hashes", "times"} <= existing.keys()
            and existing["hashes"].ndim == existing["times"].ndim == 1
            and len(existing["hashes"]) == len(existing["times"])
            and existing["hashes"].dtype == np.uint32):
        return existing["hashes"], existing["times"]
    hashes, times = _landmarks(audio, sr)
    cache.put(key, {"hashes": hashes, "times": times})
    return hashes, times


@dataclass
class FingerprintFeatures:
    hashes: np.ndarray
    times: np.ndarray
    duration: float
    sample_rate: int


def fingerprint_features(y: np.ndarray, sr: int, *, cache: Any = None) -> FingerprintFeatures:
    """Extract once for comparison with multiple season recordings."""
    hashes, times = cached_landmarks(y, sr, cache=cache)
    return FingerprintFeatures(hashes, times, len(y) / sr, sr)


def compare_recording_content(first: FingerprintFeatures, second: FingerprintFeatures) -> dict[str, Any]:
    """Detect duplicate episode recordings despite gain/encode/edit differences.

    Matching hashes alone are not enough: the same broadly distributed identity
    gate is applied in both directions, so a common intro is not a duplicate.
    """
    from adsync.align.fingerprint import _match, _offset_spans

    if first.sample_rate != second.sample_rate:
        raise ValueError("Fingerprint content comparison requires matching sample rates")
    second_times, first_times = _match(first.hashes, first.times, second.hashes, second.times)
    forward = assess_content_identity(_offset_spans(second_times, first_times, second.duration), second.duration,
                                      video_duration=first.duration)
    reverse = assess_content_identity(_offset_spans(first_times, second_times, first.duration), first.duration,
                                      video_duration=second.duration)
    duplicate = forward["status"] == reverse["status"] == "pass"
    return {"duplicate": duplicate, "policy_revision": QC_POLICY_REVISION,
            "first_coverage": reverse["coverage"], "second_coverage": forward["coverage"],
            "forward": forward, "reverse": reverse}


def _pairs(fp: FingerprintResult) -> tuple[np.ndarray, np.ndarray]:
    ta = np.asarray(fp.match_t_ad if fp.match_t_ad is not None else [], dtype=np.float64)
    tv = np.asarray(fp.match_t_vid if fp.match_t_vid is not None else [], dtype=np.float64)
    if ta.shape != tv.shape:
        raise ValueError("Fingerprint time arrays must have the same shape")
    keep = np.isfinite(ta) & np.isfinite(tv)
    return ta[keep], tv[keep] - ta[keep]


def _union(intervals: Iterable[tuple[float, float]], lo: float, hi: float) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for start, end in sorted((max(lo, a), min(hi, b)) for a, b in intervals if b > a):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def _coverage(intervals: Iterable[tuple[float, float]], lo: float, hi: float) -> float:
    return sum(b - a for a, b in _union(intervals, lo, hi)) / max(hi - lo, 1e-9)


def assess_content_identity(
    fp: FingerprintResult, ad_duration: float, *, video_duration: float | None = None,
) -> dict[str, Any]:
    """Require distributed soundtrack agreement before treating a pair as one episode.

    A fail means identity is not established sufficiently to safely render the
    supplied pairing; it does not claim to know which episode the recording is.
    Offsets and edits are allowed. A matching title sequence alone never passes.
    """
    if not math.isfinite(ad_duration) or ad_duration <= 0:
        raise ValueError("AD duration must be positive and finite")
    times, offsets = _pairs(fp)
    supported: list[tuple[float, float]] = []
    buckets: list[dict[str, Any]] = []
    width = min(10.0, max(2.0, ad_duration / 12))
    for bucket in np.unique(np.floor(times / width).astype(np.int64)):
        lo, hi = float(bucket * width), min(ad_duration, float((bucket + 1) * width))
        mask = (times >= lo) & (times < hi)
        cluster = dominant_offset_cluster(times[mask], offsets[mask], min_matches=4, min_span_sec=0.5)
        # Density and temporal distribution both matter. One repeated hash at a
        # single instant cannot turn a whole ten-second bucket into agreement.
        valid = cluster["strong"] and cluster["matches"] / max(hi - lo, 1e-9) >= 0.8
        if valid and video_duration is not None:
            target = times[mask] + offsets[mask]
            valid = bool(np.mean((target >= 0) & (target <= video_duration)) >= 0.9)
        if valid:
            supported.append((lo, hi))
        buckets.append({"start_sec": lo, "end_sec": hi, "supported": bool(valid), "cluster": cluster})
    coverage = _coverage(supported, 0, ad_duration)
    thirds = [_coverage(supported, i * ad_duration / 3, (i + 1) * ad_duration / 3) for i in range(3)]
    reasons: list[str] = []
    if coverage >= 0.50 and min(thirds) >= 0.25:
        status = "pass"
    elif ad_duration >= 60 and (coverage < 0.15 or (coverage < 0.35 and sum(v >= 0.10 for v in thirds) <= 1)):
        status = "fail"
        reasons.append("Shared audio is too sparse or concentrated to establish episode identity")
    else:
        status = "review"
        reasons.append("Distributed content identity is inconclusive")
    return {"status": status, "policy_revision": QC_POLICY_REVISION,
            "coverage": coverage, "third_coverage": thirds, "supported_ranges": _union(supported, 0, ad_duration),
            "match_count": len(times), "reported_strong": bool(fp.strong), "reasons": reasons, "buckets": buckets}


def dominant_offset_cluster(
    times: np.ndarray, offsets: np.ndarray, *, radius_sec: float = 0.08,
    min_matches: int = 12, min_span_sec: float = 3.0,
) -> dict[str, Any]:
    """Measure the dominant offset separately from unrelated retained matches."""
    times, offsets = np.asarray(times, dtype=np.float64), np.asarray(offsets, dtype=np.float64)
    valid = np.isfinite(times) & np.isfinite(offsets)
    times, offsets = times[valid], offsets[valid]
    result: dict[str, Any] = {"strong": False, "matches": 0, "total_matches": len(times),
                              "fraction": 0.0, "lag_sec": None, "p95_abs_sec": None,
                              "all_p95_abs_sec": None, "span_sec": 0.0}
    if not len(times):
        return result
    values, counts = np.unique(np.round(offsets / 0.05), return_counts=True)
    center = float(values[int(np.argmax(counts))] * 0.05)
    cluster = np.abs(offsets - center) <= radius_sec
    center = float(np.median(offsets[cluster]))
    cluster = np.abs(offsets - center) <= radius_sec
    count, fraction = int(cluster.sum()), float(cluster.mean())
    span = float(np.ptp(times[cluster]))
    result.update(strong=bool(count >= min_matches and fraction >= 0.65 and span >= min_span_sec),
                  matches=count, fraction=fraction, lag_sec=float(np.median(offsets[cluster])),
                  p95_abs_sec=float(np.percentile(np.abs(offsets[cluster]), 95)),
                  all_p95_abs_sec=float(np.percentile(np.abs(offsets), 95)), span_sec=span)
    return result


def waveform_correlation(
    main: np.ndarray, ad: np.ndarray, sr: int, *, guard_sec: float = 2.0,
    absolute_rms_floor: float = 1e-6, relative_energy_floor: float = 1e-6,
    template_bounds: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Independent Pearson lag; positive lag means main time minus AD time.

    The central main template searches the AD guard. Numerically suspect and
    near-silent positions are rejected before normalization, not score-clipped
    into spurious matches. The relative floor is based on local signal energy.
    """
    x, y = np.asarray(main, dtype=np.float64), np.asarray(ad, dtype=np.float64)
    guard = int(round(guard_sec * sr))
    weak: dict[str, Any] = {"status": "weak", "reason": "insufficient samples", "lag_sec": None, "score": 0.0}
    start, end = template_bounds if template_bounds is not None else (guard, len(x) - guard)
    if not (0 <= start < end <= len(x)) or end - start < max(round(sr * 0.5), 2) or not np.isfinite(x).all() or not np.isfinite(y).all():
        return weak
    template = x[start:end].copy()
    template -= template.mean()
    # Removing a large DC component before cumulative sums avoids cancellation
    # of almost equal squared totals for a quiet template in an offset signal.
    y = y - np.mean(y)
    n = len(template)
    energy = float(np.dot(template, template))
    reference_power = max(float(np.mean((x - x.mean()) ** 2)), float(np.mean(y * y)))
    floor_power = max(absolute_rms_floor ** 2, reference_power * relative_energy_floor)
    weak.update(template_rms=float(np.sqrt(energy / n)), energy_floor_rms=float(np.sqrt(floor_power)))
    if energy / n <= floor_power or len(y) < n:
        return {**weak, "reason": "template energy below absolute or relative floor"}
    prefix = np.r_[0.0, np.cumsum(y, dtype=np.float64)]
    square = np.r_[0.0, np.cumsum(y * y, dtype=np.float64)]
    sums = prefix[n:] - prefix[:-n]
    energies = square[n:] - square[:-n] - sums * sums / n
    usable = energies > n * floor_power
    numerators = fftconvolve(y, template[::-1], mode="valid")
    scores = np.full(len(numerators), -np.inf)
    scores[usable] = numerators[usable] / np.sqrt(energy * energies[usable])
    # Rounding excursions of a few ULPs are benign; a physically impossible
    # correlation signals bad normalization and is excluded, never accepted.
    scores[np.abs(scores) > 1.0 + 1e-6] = -np.inf
    if not np.isfinite(scores).any():
        return {**weak, "reason": "search energy below floor or normalization unstable"}
    peak = int(np.argmax(scores))
    score = min(1.0, float(scores[peak]))
    fractional = 0.0
    if 0 < peak < len(scores) - 1 and np.isfinite(scores[peak - 1:peak + 2]).all():
        before, at, after = scores[peak - 1:peak + 2]
        denominator = before - 2 * at + after
        if abs(denominator) > 1e-12:
            fractional = float(np.clip(0.5 * (before - after) / denominator, -0.5, 0.5))
    outside = scores.copy()
    exclude = round(0.15 * sr)
    outside[max(0, peak - exclude):min(len(scores), peak + exclude + 1)] = -np.inf
    competing = max(0.0, float(np.max(outside)))
    ratio = score / max(competing, 1e-12)
    # Diagnostic only: alignment callers retain the original score/ratio gate.
    # QC can combine weak but statistically distinctive peaks across separate
    # frequency bands when loud narration suppresses normalized amplitude.
    background = outside[np.isfinite(outside)]
    median = float(np.median(background)) if len(background) else 0.0
    noise = float(1.4826 * np.median(np.abs(background - median))) if len(background) else 0.0
    peak_z = (score - median) / noise if noise > 1e-12 else 0.0
    bounded_zero = abs(peak - start) <= 1
    strong = score >= 0.30 and ratio >= 1.08 and (0 < peak < len(scores) - 1 or bounded_zero)
    return {"status": "strong" if strong else "weak", "reason": "" if strong else "ambiguous waveform peak",
            "lag_sec": -(peak + fractional - start) / sr, "score": score, "peak_ratio": ratio,
            "template_rms": weak["template_rms"], "search_rms": float(np.sqrt(max(energies[peak], 0) / n)),
            "energy_floor_rms": weak["energy_floor_rms"], "competing_score": competing,
            "peak_z": float(peak_z), "background_median": median, "background_mad_sigma": noise,
            "usable_peak": bool(0 < peak < len(scores) - 1 or bounded_zero)}


def targeted_windows(
    duration: float, *, cut_times: Iterable[float] = (),
    sparse_intervals: Iterable[tuple[float, float]] = (),
    terminal_intervals: Iterable[tuple[float, float]] = (),
    support_gap_intervals: Iterable[tuple[float, float]] = (),
) -> list[dict[str, Any]]:
    """Plan checks around edits, all sparse regions, and short ending pieces."""
    if duration <= 0:
        return []
    windows: dict[tuple[float, float], dict[str, Any]] = {}
    def add(start: float, end: float, reason: str) -> None:
        start, end = max(0.0, float(start)), min(float(duration), float(end))
        if end - start < min(0.5, duration):
            return
        key = (round(start, 4), round(end, 4))
        if key in windows:
            if reason not in windows[key]["reasons"]:
                windows[key]["reasons"].append(reason)
        else:
            windows[key] = {"start_sec": start, "end_sec": end, "center_sec": (start + end) / 2, "reasons": [reason]}
    for center in np.linspace(duration * 0.05, duration * 0.95, 9):
        add(max(0, center - 6), min(duration, center + 6), "distributed")
    for cut in cut_times:
        add(cut - 8, cut, "before_edit")
        add(cut, cut + 8, "after_edit")
    for start, end in _union(sparse_intervals, 0, duration):
        for left in np.arange(max(0, start), min(duration, end), 8.0):
            add(float(left), min(end, left + 12), "sparse_anchors")
    for start, end in support_gap_intervals:
        add(start, end, "short_support_gap")
    for start, end in [(max(0, duration - 30), duration), *terminal_intervals]:
        for left in np.arange(max(0, start), min(duration, end), 8.0):
            add(float(left), min(end, left + 10), "terminal")
        add(max(start, end - 1), end, "terminal_last_second")
    return sorted(windows.values(), key=lambda w: (w["start_sec"], w["end_sec"]))


def _agreeing_group(rows: list[dict[str, Any]], *, min_count: int = 2,
                    max_disagreement_sec: float = .035) -> tuple[list[dict[str, Any]], bool]:
    groups = [tuple(i for i, other in enumerate(rows)
                    if row["lag_sec"] <= other["lag_sec"] <= row["lag_sec"] + max_disagreement_sec)
              for row in rows]
    # Every member must agree with every other member. Center-radius grouping
    # could otherwise chain distant modes through a middle band.
    groups = sorted(set(groups), key=lambda group: (-len(group), -sum(rows[i].get("score", 0) for i in group), group))
    ambiguous = bool(len(groups) > 1 and len(groups[0]) == len(groups[1]) and set(groups[0]).isdisjoint(groups[1]))
    return ([rows[i] for i in groups[0]] if groups and len(groups[0]) >= min_count and not ambiguous else []), ambiguous


def _waveform_window(main: np.ndarray, ad: np.ndarray, sr: int, start: float, end: float, *, search_radius_sec: float = 2.0) -> dict[str, Any]:
    # Shift the bounded search at the ends so the template itself still covers
    # the requested short tail; guards never require extrapolated audio.
    size = min(len(main), len(ad))
    from adsync.quality_signal import signal_levels, raw_peak_supported, reassess_envelope_conflict
    signal = signal_levels(main, ad, sr, start, end)
    left, right = max(0, round((start - search_radius_sec) * sr)), min(size, round((end + search_radius_sec) * sr))
    x, y = main[left:right], ad[left:right]
    guard = min(2.0, max(0.0, (len(x) / sr - 2.0) / 4))
    bounds = (max(0, round(start * sr) - left), min(len(x), round(end * sr) - left))
    bands: dict[str, dict[str, Any]] = {"full": waveform_correlation(x, y, sr, guard_sec=guard, template_bounds=bounds)}
    filtered = [("full", x, y)]
    for name, lo, hi in (("low", 80.0, 250.0), ("mid", 250.0, 1200.0), ("high", 1200.0, min(7000.0, sr * 0.45))):
        if hi <= lo or hi >= sr / 2 or len(x) < sr:
            continue
        sos = butter(4, (lo, hi), btype="bandpass", fs=sr, output="sos")
        xx, yy = sosfiltfilt(sos, x), sosfiltfilt(sos, y)
        filtered.append((name, xx, yy))
        bands[name] = waveform_correlation(xx, yy, sr, guard_sec=guard, template_bounds=bounds)
    # Full-band and low-band correlation can reflect the very same decaying
    # rumble. Require two disjoint frequency bands rather than counting that
    # single physical feature twice.
    strong = [row for name, row in bands.items() if name != "full" and row["status"] == "strong"]
    agreeing, competing_groups = _agreeing_group(strong)
    # One extremely clear full-band result is sufficient independent evidence;
    # ambiguous narration normally needs agreement between two filtered bands.
    full = bands["full"]
    if not agreeing and not competing_groups and full["status"] == "strong" and full["score"] >= 0.85 and full["peak_ratio"] >= 1.5:
        agreeing = [full]
    method = "waveform" if agreeing else None
    statistical_consensus = []
    if not agreeing and not competing_groups and signal["state"] != "near_silent":
        statistical = [{"name": name, **row} for name, row in bands.items() if name != "full"
                       and raw_peak_supported(row)]
        # These are carrier waveform peaks after identical zero-phase filters,
        # not coarse energy-envelope events. Require sample-timing agreement
        # within five milliseconds; a broad fifteen-ms coincidence admitted
        # unrelated narration peaks in different frequency bands.
        statistical_consensus, ambiguous = _agreeing_group(statistical, max_disagreement_sec=.005)
        # One specific band identifies the lag; the other disjoint band's
        # statistically distinctive peak independently corroborates it, even
        # when that band's repeated musical notes admit several alternatives.
        if statistical_consensus and not ambiguous and any(raw_peak_supported(r, require_unique=True) for r in statistical_consensus):
            agreeing = statistical_consensus
            method = "cross_band_waveform"
        else:
            statistical_consensus = []
    envelopes = {}
    envelope_audio = {}
    chunks: list[dict[str, Any]] = []
    chunk_consensus: list[dict[str, Any]] = []
    if not agreeing and not competing_groups and end - start >= 3:
        # Remixed/pitch-processed tracks can share timed sound events while
        # individual carrier samples decorrelate. Envelopes provide a separate
        # estimator with stricter multiband agreement and peak specificity.
        for name, xx, yy in filtered:
            a = np.sqrt(np.maximum(uniform_filter1d(np.asarray(xx, dtype=np.float64) ** 2, max(1, round(.02 * sr))), 0))
            b = np.sqrt(np.maximum(uniform_filter1d(np.asarray(yy, dtype=np.float64) ** 2, max(1, round(.02 * sr))), 0))
            a, b = resample_poly(a, 100, sr), resample_poly(b, 100, sr)
            envelope_audio[name] = (a, b)
            envelope_bounds = (round(bounds[0] / sr * 100), min(len(a), round(bounds[1] / sr * 100)))
            envelopes[name] = waveform_correlation(a, b, 100, template_bounds=envelope_bounds)
        clear = [r for r in envelopes.values() if r["status"] == "strong" and r["score"] >= 0.55 and r["peak_ratio"] >= 1.30]
        agreeing, _ = _agreeing_group(clear)
        if not agreeing:
            clear = [r for r in envelopes.values() if r["status"] == "strong" and r["score"] >= 0.40 and r["peak_ratio"] >= 1.15]
            agreeing, _ = _agreeing_group(clear, min_count=3)
        if agreeing:
            method = "band_energy_envelope"
        elif end - start >= 8:
            # Intermittent narration can dominate a long energy template.
            # Search disjoint shorter pieces without any expected-zero bias;
            # require three separate pieces, each backed by two bands, to
            # agree. A single lucky short peak cannot approve the region.
            count = max(3, int((end - start) // 3))
            for chunk_start in np.linspace(bounds[0] / sr, bounds[1] / sr - 2.0, count):
                a0, b0 = round(chunk_start * 100), round((chunk_start + 2.0) * 100)
                chunk_bands = {}
                for name, (a, b) in envelope_audio.items():
                    lo, hi = max(0, a0 - 200), min(len(a), b0 + 200)
                    row = waveform_correlation(a[lo:hi], b[lo:hi], 100, template_bounds=(a0 - lo, b0 - lo))
                    chunk_bands[name] = row
                clear = [r for r in chunk_bands.values() if r["status"] == "strong" and r["score"] >= 0.55 and r["peak_ratio"] >= 1.30]
                agreed, ambiguous = _agreeing_group(clear)
                chunks.append({"start_sec": left / sr + chunk_start, "end_sec": left / sr + chunk_start + 2.0,
                               "status": "strong" if agreed and not ambiguous else "weak",
                               "lag_sec": float(np.median([r["lag_sec"] for r in agreed])) if agreed else None,
                               "bands": chunk_bands})
            strong_chunks = [r for r in chunks if r["status"] == "strong"]
            chunk_consensus, ambiguous = _agreeing_group(strong_chunks, min_count=3)
            # Any other independently strong chunk outside the consensus is a
            # local inconsistency, even if the majority of the window agrees.
            if chunk_consensus and (ambiguous or any(abs(r["lag_sec"] - np.median([c["lag_sec"] for c in chunk_consensus])) > .10 for r in strong_chunks)):
                chunk_consensus = []
            if chunk_consensus:
                agreeing = chunk_consensus
                method = "separated_envelope_chunks"
    conflict = None
    if agreeing and method in {"band_energy_envelope", "separated_envelope_chunks"}:
        envelope_lag = float(np.median([r["lag_sec"] for r in agreeing]))
        reassessment = reassess_envelope_conflict({"status": "strong", "lag_sec": envelope_lag,
                                                  "method": method, "bands": bands})
        if reassessment["decision"] == "conflict":
            # Narration can share another scene's amplitude pattern without
            # sharing its waveform. Such an envelope peak does not establish
            # a rendered delay when surviving, distinctive bed audio disagrees.
            conflict = reassessment["waveform"]["conflict"]
            agreeing, method = [], None
        elif reassessment["decision"] == "remeasure":
            conflict = {"reason": "Raw carrier statistics are incomplete", "missing_statistics": reassessment["missing_statistics"]}
            agreeing, method = [], None
    if agreeing and method in {"waveform", "cross_band_waveform"}:
        consensus_lag = float(np.median([r["lag_sec"] for r in agreeing]))
        contrary_raw = [{"name": name, **row} for name, row in bands.items() if name != "full"
                        and raw_peak_supported(row, require_unique=True)
                        and abs(row["lag_sec"] - consensus_lag) > .10]
        if contrary_raw:
            conflict = {"kind": "mixed_raw_lags", "reason": "Disjoint frequency bands support incompatible local lags",
                        "consensus_lag_sec": consensus_lag, "consensus_method": method,
                        "raw_bands": contrary_raw,
                        "suggested_windows": [{"start_sec": float(t), "end_sec": min(end, float(t + 2))}
                                              for t in np.arange(start, max(start, end - 1), 1.0)]}
            agreeing, method = [], None
    return {"status": "strong" if agreeing else "weak",
            "lag_sec": float(np.median([r["lag_sec"] for r in agreeing])) if agreeing else None,
            "method": method, "bands": bands, "envelopes": envelopes,
            "chunks": chunks, "chunk_consensus": chunk_consensus,
            "conflict": conflict,
            "statistical_consensus": statistical_consensus, "signal": signal,
            "template_start_sec": start, "template_end_sec": end,
            "sample_start_sec": left / sr, "sample_end_sec": right / sr}


def verify_audio_sync(
    main: np.ndarray, ad: np.ndarray, sr: int, *, fingerprint: FingerprintResult | None = None,
    cut_times: Iterable[float] = (), sparse_intervals: Iterable[tuple[float, float]] = (),
    terminal_intervals: Iterable[tuple[float, float]] = (), tolerance_sec: float = 0.150,
) -> dict[str, Any]:
    """Verify rendered shared audio at local windows, retaining all evidence."""
    main, ad = np.asarray(main), np.asarray(ad)
    if main.ndim != 1 or ad.ndim != 1 or sr <= 0:
        raise ValueError("QC expects mono arrays and a positive sample rate")
    duration = min(len(main), len(ad)) / sr
    if duration <= 0:
        return {"status": "fail", "failures": ["Empty rendered audio"], "review_reasons": [], "windows": []}
    fp = fingerprint if fingerprint is not None else fingerprint_align(main, ad, sr)
    identity = assess_content_identity(fp, duration, video_duration=len(main) / sr)
    times, offsets = _pairs(fp)
    supplied_cuts = list(cut_times)
    rendered_cuts = [value for span in fp.spans if abs(span.offset) > tolerance_sec
                     for value in (span.ad_start, span.ad_end) if 0 < value < duration]
    suspect_ranges = [(s.ad_start, s.ad_end) for s in fp.spans if abs(s.offset) > tolerance_sec]
    ordered_times = np.unique(times)
    support_gaps = []
    # The landmark aligner retains offsets only within its broad inlier radius.
    # A short larger error can therefore disappear *inside* a healthy merged
    # span. Target missing raw support, not just the span-level unmatched list.
    for index in np.flatnonzero(np.diff(ordered_times) >= 1.5):
        lo, hi = float(ordered_times[index]), float(ordered_times[index + 1])
        before = ordered_times[(ordered_times >= lo - 10) & (ordered_times <= lo)]
        after = ordered_times[(ordered_times >= hi) & (ordered_times <= hi + 10)]
        expected_missing = (hi - lo) * min(len(before), len(after)) / 10
        normal_gaps = np.r_[np.diff(before), np.diff(after)]
        # Many real narrated/remixed tracks naturally produce sparse hashes.
        # Only a surprising loss against *both* local flanks merits an extra
        # tiny-gap probe; broad sparse areas are already handled separately.
        if (expected_missing < 12 or len(normal_gaps) < 8
                or hi - lo < 2 * float(np.percentile(normal_gaps, 95))):
            continue
        if hi - lo <= 5.2:
            support_gaps.append((lo + 0.1, hi - 0.1))
        else:
            suspect_ranges.append((lo + 0.1, hi - 0.1))
    for lo in np.arange(0.0, duration, 2.0):
        selected = (times >= lo) & (times < lo + 2)
        cluster = dominant_offset_cluster(times[selected], offsets[selected], min_matches=8, min_span_sec=0.75)
        if cluster["strong"] and abs(cluster["lag_sec"]) > tolerance_sec:
            suspect_ranges.append((float(lo), min(duration, float(lo + 2))))
    sparse = [*sparse_intervals, *fp.unmatched, *suspect_ranges]
    support_gaps = [(a, b) for a, b in support_gaps if not any(lo <= a and hi >= b for lo, hi in sparse)]
    windows = targeted_windows(duration, cut_times=[*supplied_cuts, *rendered_cuts],
                               sparse_intervals=sparse, terminal_intervals=terminal_intervals, support_gap_intervals=support_gaps)
    raw_ad = fp.raw_match_t_ad if fp.raw_match_t_ad is not None else fp.match_t_ad
    raw_main = fp.raw_match_t_vid if fp.raw_match_t_vid is not None else fp.match_t_vid
    raw_targets = raw_offset_windows(raw_ad, raw_main, duration, tolerance_sec=tolerance_sec)
    # Keep independent narrow probes: merging them into broad passing windows
    # would hide the very short error that produced the minority raw votes.
    windows.extend(raw_targets)
    windows.sort(key=lambda window: (window["start_sec"], window["end_sec"]))
    failures: list[str] = []
    reviews: list[str] = []
    passed_thirds: set[int] = set()
    independent_thirds: set[int] = set()
    if abs(len(main) - len(ad)) / sr > 1.0:
        failures.append("Rendered audio stream durations differ by more than one second")
    if identity["status"] != "pass":
        reviews.extend(identity["reasons"])
    for window in windows:
        start, end = window["start_sec"], window["end_sec"]
        mask = (times >= start) & (times <= end)
        cluster = dominant_offset_cluster(times[mask], offsets[mask], min_span_sec=min(3.0, (end - start) * 0.5))
        waveform = _waveform_window(main, ad, sr, start, end)
        if waveform["status"] != "strong" and "short_support_gap" in window["reasons"]:
            # If the AD briefly contains earlier/later audio, the correct main
            # template may be absent from that AD interval. Reverse the query
            # over a wider bounded search before allowing flanking context.
            reverse = _waveform_window(ad, main, sr, start, end, search_radius_sec=8.0)
            window["reverse_search"] = reverse
            if reverse["status"] == "strong":
                waveform = {**reverse, "lag_sec": -reverse["lag_sec"], "orientation": "AD template in main"}
                waveform["bands"] = {name: {**row, "lag_sec": -row["lag_sec"] if row["lag_sec"] is not None else None}
                                     for name, row in reverse["bands"].items()}
        fp_good = cluster["strong"] and abs(cluster["lag_sec"]) <= tolerance_sec and cluster["p95_abs_sec"] <= tolerance_sec
        fp_bad = cluster["strong"] and abs(cluster["lag_sec"]) > tolerance_sec
        wave_good = waveform["status"] == "strong" and abs(waveform["lag_sec"]) <= tolerance_sec
        wave_bad = waveform["status"] == "strong" and abs(waveform["lag_sec"]) > tolerance_sec
        local_bands = [r for r in waveform["bands"].values()
                       if r["status"] == "strong" and abs(r["lag_sec"]) <= tolerance_sec]
        if (fp_bad and wave_good and waveform["method"] == "waveform" and len(local_bands) >= 3
                and max(r["score"] for r in local_bands) >= .50):
            window["adjudication"] = {
                "reason": "Global fingerprint location conflicts with independently aligned local waveform in at least three bands",
                "fingerprint_lag_sec": cluster["lag_sec"], "local_lag_sec": waveform["lag_sec"],
                "limitation": "Repeated soundtrack locations do not identify narration semantics",
            }
            fp_bad = False
        local_main, local_ad = main[round(start * sr):round(end * sr)], ad[round(start * sr):round(end * sr)]
        silent = (len(local_main) > 0 and len(local_ad) > 0
                  and float(np.sqrt(np.mean(np.asarray(local_main, dtype=np.float64) ** 2))) <= 1e-6
                  and float(np.sqrt(np.mean(np.asarray(local_ad, dtype=np.float64) ** 2))) <= 1e-6)
        if fp_bad and wave_good:
            status = "review"
            reviews.append(f"Conflicting fingerprint and local audio evidence at {start:.3f}-{end:.3f} s")
        elif wave_bad or fp_bad:
            status = "fail"
            failures.append(f"Coherent nonzero offset at {start:.3f}-{end:.3f} s")
        elif wave_good or (fp_good and "terminal_last_second" not in window["reasons"]):
            # Dominant statistics remove the old all-retained-p95 false alarm.
            # We still demand independent evidence somewhere in every third.
            status = "pass"
            third = min(2, int(window["center_sec"] / duration * 3))
            passed_thirds.add(third)
            if wave_good:
                independent_thirds.add(third)
        elif silent:
            # No shared audible content exists to time here. This cannot count
            # as a passing measurement, but also is not a timing uncertainty.
            status = "silent"
        else:
            status = "review"
            if ("short_support_gap" in window["reasons"] and end - start <= 5
                    and start >= 8 and end <= duration - 8
                    and not any(start - 4 <= cut <= end + 4 for cut in supplied_cuts)):
                before = _waveform_window(main, ad, sr, start - 6, start - 2)
                after = _waveform_window(main, ad, sr, end + 2, end + 6)
                context_good = (before["status"] == after["status"] == "strong"
                                and max(abs(before["lag_sec"]), abs(after["lag_sec"])) <= tolerance_sec
                                and abs(before["lag_sec"] - after["lag_sec"]) <= .035)
                window["context"] = {"before": before, "after": after,
                                     "limitation": "Timing inside this short narration/sparse gap is bounded by measured flanks, not directly measured"}
                if context_good:
                    status = "supported_context"
            # Weak terminal/sparse soundtrack evidence is retained explicitly;
            # no guessed offset is treated as a measured timing failure.
            if status == "review":
                reviews.append(f"Insufficient local shared-audio evidence at {start:.3f}-{end:.3f} s")
        window.update(status=status, fingerprint=cluster, waveform=waveform)
    if len(passed_thirds) < 3:
        reviews.append("Missing reliable alignment evidence in one or more thirds")
    if len(independent_thirds) < 3:
        reviews.append("Independent waveform corroboration missing in one or more thirds")
    context_coverage = _coverage([(w["start_sec"], w["end_sec"]) for w in windows
                                  if w["status"] == "supported_context"], 0, duration)
    evidence = {"status": "fail" if failures else "review" if reviews else "pass", "policy_revision": QC_POLICY_REVISION,
            "duration_sec": duration, "sample_rate": sr, "tolerance_sec": tolerance_sec,
            "supported_context_coverage": context_coverage,
            "identity": identity, "failures": list(dict.fromkeys(failures)), "review_reasons": list(dict.fromkeys(reviews)),
            "windows": windows, "fingerprint_spans": [vars(s).copy() for s in fp.spans],
            "raw_fingerprint_probe_count": len(raw_targets),
            "fingerprint_unmatched": fp.unmatched, "fingerprint_dropped_ranges": fp.dropped_ranges,
            "limitation": "Checks shared soundtrack timing; does not semantically verify each narration sentence."}
    return evaluate_quality_policy(evidence, cut_times=supplied_cuts)


def report_targets(report: Any) -> dict[str, Any]:
    """Map edit/sparse/terminal evidence from a sync report onto rendered time."""
    if report is None:
        return {}
    data = report.model_dump() if hasattr(report, "model_dump") else report
    segments = data.get("segments") or []
    cuts = [float(s["dst_start"]) for s in segments[1:]]
    cuts.extend(float(s["dst_end"]) for s in segments[:-1])
    terminal = [(max(float(s["dst_start"]), float(s["dst_end"]) - 30), float(s["dst_end"]))
                for s in segments[-1:] if s["dst_end"] > s["dst_start"]]
    # Unmatched intervals live on source time. Map through measured segment
    # endpoints instead of treating source coordinates as output coordinates.
    sparse = []
    for lo, hi in data.get("fingerprint_unmatched") or []:
        for s in segments:
            a, b = max(lo, s["src_start"]), min(hi, s["src_end"])
            if b > a and s["src_end"] > s["src_start"]:
                scale = (s["dst_end"] - s["dst_start"]) / (s["src_end"] - s["src_start"])
                sparse.append((s["dst_start"] + (a - s["src_start"]) * scale,
                               s["dst_start"] + (b - s["src_start"]) * scale))
    debug = data.get("timing_debug") or {}
    ranges, coefficients = debug.get("segment_ranges") or [], debug.get("fitted_pchip") or []
    if ranges and len(ranges) == len(coefficients):
        from scipy.interpolate import PPoly

        adjust = float(data.get("offset_adjust") or 0)
        functions = [PPoly(np.asarray(piece["c"]), np.asarray(piece["x"])) for piece in coefficients]
        for index, ((start, end), fn) in enumerate(zip(ranges, functions)):
            if index:
                cuts.append(float(fn(start)))
            if index < len(ranges) - 1:
                cuts.append(float(fn(end)))
            for lo, hi in data.get("fingerprint_unmatched") or []:
                a, b = max(lo, start), min(hi, end)
                if b > a:
                    sparse.append((float(fn(a)), float(fn(b))))
        last_start, last_end = ranges[-1]
        terminal = [(float(functions[-1](max(last_start, last_end - 30))),
                     float(functions[-1](last_end)))]
        for support in (debug.get("fingerprint") or {}).get("terminal_support") or []:
            terminal.append((support["boundary"] + support["offset"] + adjust,
                             support["last_time"] + support["offset"] + adjust))
    return {"cut_times": sorted(set(cuts)), "sparse_intervals": sparse, "terminal_intervals": terminal}


def select_verification_streams(info: MediaInfo) -> tuple[StreamInfo, StreamInfo]:
    """Select the mux-appended description and its corresponding main mix."""
    from adsync.media.prep import pick_audio_stream

    streams = info.audio_streams
    if len(streams) < 2:
        raise ValueError("Output QC needs both original and description audio streams")
    # ADSync appends its output after all original audio when preparation is
    # disabled, and custom titles are permitted. A source's pre-existing track
    # named Audio Description must never displace the newly appended output.
    ad = streams[-1]
    remaining = info.model_copy(update={"audio_streams": [s for s in streams if s.index != ad.index]})
    main = pick_audio_stream(remaining, language=ad.language or "eng", strict=False)
    return main, ad


def verify_media_sync(path: Path, *, work_dir: Path | None = None, report: Any = None) -> dict[str, Any]:
    """Decode both muxed tracks in one input read and verify them before publication."""
    import soundfile as sf
    from adsync.media.probe import probe

    path = Path(path)
    before = (path.stat().st_size, path.stat().st_mtime_ns)
    main_stream, ad_stream = select_verification_streams(probe(path))
    if work_dir is not None:
        Path(work_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="adsync-qc-", dir=work_dir) as folder:
        targets = [Path(folder) / name for name in ("main.wav", "description.wav")]
        command = ["ffmpeg", "-hide_banner", "-v", "error", "-xerror", "-y", "-threads", "2",
                   "-filter_threads", "2", "-err_detect", "explode", "-copyts", "-start_at_zero", "-i", str(path)]
        for stream, target in zip((main_stream, ad_stream), targets):
            command += ["-map", f"0:{stream.index}", "-vn", "-sn", "-dn",
                        "-af", "aresample=async=1:first_pts=0", "-ac", "1", "-ar", "16000",
                        "-c:a", "pcm_f32le", "-threads", "2", str(target)]
        options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3600, **options)
        if process.returncode or process.stderr.strip():
            raise RuntimeError(f"Rendered audio decode failed: {process.stderr.decode(errors='replace')[-6000:]}")
        main, rate = sf.read(targets[0], dtype="float32")
        ad, other_rate = sf.read(targets[1], dtype="float32")
        if rate != other_rate:
            raise RuntimeError("QC decode sample rates disagree")
        result = verify_audio_sync(main, ad, rate, **report_targets(report))
    if before != (path.stat().st_size, path.stat().st_mtime_ns):
        result["status"] = "fail"
        result["failures"].append("Output changed during verification")
    result["output_signature"] = {"path": str(path.resolve()), "size": before[0], "mtime_ns": before[1]}
    result["audio_streams"] = {"main": main_stream.model_dump(), "description": ad_stream.model_dump()}
    return result


def verify_video_preservation(source: Path, output: Path) -> dict[str, Any]:
    """Compare every compressed video stream without decoder timestamp rounding.

    Full packet hashes establish that muxing retained the source video payload;
    this is separate from audio synchronization and whole-file transfer hashes.
    """
    import re

    def packet_hashes(path: Path) -> list[str]:
        options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        result = subprocess.run([
            "ffmpeg", "-hide_banner", "-v", "error", "-xerror", "-threads", "2", "-i", str(path),
            "-map", "0:v", "-c:v", "copy", "-f", "streamhash", "-hash", "sha256", "-",
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3600, **options)
        if result.returncode or result.stderr.strip():
            raise RuntimeError(f"Video preservation hashing failed: {result.stderr.decode(errors='replace')[-6000:]}")
        hashes = re.findall(r"\d+,v,SHA256=([0-9a-fA-F]{64})", result.stdout.decode())
        if not hashes:
            raise RuntimeError("No video packet hashes returned")
        return [value.lower() for value in hashes]

    source, output = Path(source), Path(output)
    before = [(p.stat().st_size, p.stat().st_mtime_ns) for p in (source, output)]
    source_hashes, output_hashes = packet_hashes(source), packet_hashes(output)
    unchanged = before == [(p.stat().st_size, p.stat().st_mtime_ns) for p in (source, output)]
    matches = source_hashes == output_hashes and unchanged
    return {"status": "pass" if matches else "fail", "method": "sha256-compressed-video-packets",
            "source_stream_hashes": source_hashes, "output_stream_hashes": output_hashes,
            "inputs_unchanged": unchanged,
            "failures": [] if matches else ["Source/output compressed video differs or changed during verification"]}
