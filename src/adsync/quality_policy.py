"""Local QC decisions without treating narration frequency as timing error.

Signal estimators establish whether an offset was actually measured. This
module never upgrades weak signal evidence into a measurement. It may accept
a short, explicitly unmeasured interval bracketed by independent consistent
measurements, while preserving the uncertainty in the returned evidence.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Iterable


QC_DECISION_POLICY_REVISION = "bounded-local-uncertainty-v1"
_OLD_CONTEXT_CAP = "More than three percent of the rendered track relies on bounded context instead of direct timing evidence"


def _merge(ranges: Iterable[tuple[float, float]]) -> list[list[float]]:
    merged: list[list[float]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([float(start), float(end)])
    return merged


def _strong_aligned(measurement: dict, tolerance: float) -> bool:
    lag = measurement.get("lag_sec")
    return (measurement.get("status") == "strong" and isinstance(lag, (int, float))
            and math.isfinite(lag) and abs(lag) <= tolerance and not measurement.get("conflict"))


def _quiet_both_tracks(waveform: dict) -> bool:
    signal = waveform.get("signal") or {}
    if signal.get("state") != "near_silent" or waveform.get("conflict"):
        return False
    # The signal stage owns audibility calibration and transient rejection.
    # Require its explicit two-track verdict and diagnostic evidence, rather
    # than treating a missing/weak waveform estimate as proof of quiet audio.
    for name in ("main", "ad"):
        diagnostic = signal.get(name) or {}
        values = [diagnostic.get(key) for key in ("ac_rms", "peak_ac", "noise_floor_rms")]
        if not all(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 for value in values):
            return False
        if diagnostic["ac_rms"] > diagnostic["noise_floor_rms"]:
            return False
    return True


def _flank_bounds(measurement: dict, start: float, end: float, side: str) -> tuple[float, float] | None:
    if "template_start_sec" in measurement and "template_end_sec" in measurement:
        return float(measurement["template_start_sec"]), float(measurement["template_end_sec"])
    # Compatibility with the existing documented context probes: each queries
    # a four-second template separated from the uncertain region by two seconds.
    # Their sample bounds also include a two-second correlation search margin.
    expected_sample = (start - 8, start) if side == "before" else (end, end + 8)
    observed = (measurement.get("sample_start_sec"), measurement.get("sample_end_sec"))
    if not all(isinstance(value, (int, float)) for value in observed):
        return None
    if any(abs(a - b) > .002 for a, b in zip(observed, expected_sample)):
        return None
    return (start - 6, start - 2) if side == "before" else (end + 2, end + 6)


def _bounded_context(window: dict, *, duration: float, tolerance: float,
                     uncertainty: list[list[float]], cuts: list[float],
                     max_gap: float, max_disagreement: float) -> tuple[bool, str]:
    start, end = float(window["start_sec"]), float(window["end_sec"])
    component = next((part for part in uncertainty if part[0] <= start and part[1] >= end), [start, end])
    if component[1] - component[0] > max_gap:
        return False, "The contiguous unmeasured interval exceeds the bounded-gap limit"
    if any(start - 4 <= cut <= end + 4 for cut in cuts):
        return False, "An edit is too close to infer continuity across this interval"
    if (window.get("waveform") or {}).get("conflict") or (window.get("reverse_search") or {}).get("conflict"):
        return False, "Conflicting local signal evidence cannot be replaced by neighboring context"
    context = window.get("context") or {}
    before, after = context.get("before") or {}, context.get("after") or {}
    if not all(_strong_aligned(row, tolerance) for row in (before, after)):
        return False, "Both local flanks need independent measured alignment"
    if abs(before["lag_sec"] - after["lag_sec"]) > max_disagreement:
        return False, "The measured flanks disagree about the local offset"
    for side, flank in (("before", before), ("after", after)):
        bounds = _flank_bounds(flank, start, end, side)
        if bounds is None:
            return False, "The measured flank template bounds are unavailable"
        low, high = bounds
        if not (math.isfinite(low) and math.isfinite(high) and 0 <= low < high <= duration and high - low >= 2):
            return False, "The unmeasured interval is not bracketed by valid local templates"
        distance = start - high if side == "before" else low - end
        if distance < -1e-6 or distance > 6:
            return False, "A measured flank overlaps or is too far from the unmeasured interval"
        if any(low < other_end - 1e-6 and high > other_start + 1e-6 for other_start, other_end in uncertainty):
            return False, "A supposed measured flank overlaps another unmeasured interval"
    return True, "Short unmeasured interval bracketed by independent consistent local measurements"


def evaluate_quality_policy(
    evidence: dict,
    *,
    cut_times: Iterable[float] = (),
    max_context_gap_sec: float = 5.0,
    max_flank_disagreement_sec: float = .035,
) -> dict:
    """Return a copied QC record with an explicit local publication decision.

    There is no global quota for narration or bounded uncertainty. Every such
    interval must independently pass the same finite local continuity test.
    Overlapping gaps are joined, and unmeasured intervals cannot be recycled
    as one another's flanks. Coherent failures, wrong content, unresolved edges,
    large unsupported regions and missing distributed corroboration still gate
    publication. No narration sentence is semantically validated by this policy.
    """
    result = deepcopy(evidence)
    duration = float(result.get("duration_sec", 0))
    tolerance = float(result.get("tolerance_sec", .150))
    if not all(math.isfinite(value) and value > 0 for value in
               (duration, tolerance, max_context_gap_sec, max_flank_disagreement_sec)):
        raise ValueError("QC duration and policy tolerances must be finite and positive")
    cuts = [float(value) for value in cut_times]
    if not all(math.isfinite(value) for value in cuts):
        raise ValueError("Edit locations must be finite")
    windows = result.get("windows") or []
    for window in windows:
        start, end = float(window["start_sec"]), float(window["end_sec"])
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration):
            raise ValueError("QC window lies outside the rendered timeline")
    failures = list(result.get("failures") or [])
    # Replace only the old aggregator's derived messages. Preserve all other
    # review reasons, including conflicting evidence and decoder diagnostics.
    derived = {"Missing reliable alignment evidence in one or more thirds",
               "Independent waveform corroboration missing in one or more thirds", _OLD_CONTEXT_CAP}
    reviews = [reason for reason in result.get("review_reasons", [])
               if reason not in derived and not reason.startswith("Insufficient local shared-audio evidence at ")]
    identity = result.get("identity") or {}
    if identity.get("status") == "fail":
        failures.extend(identity.get("reasons") or ["Distributed content identity failed"])
    elif identity.get("status") != "pass":
        reviews.extend(identity.get("reasons") or ["Distributed content identity is unresolved"])

    uncertain_windows = [row for row in windows if row.get("status") not in {"pass", "silent", "fail"}]
    uncertainty = _merge((float(row["start_sec"]), float(row["end_sec"])) for row in uncertain_windows)
    independent_thirds: set[int] = set()
    for window in windows:
        start, end = float(window["start_sec"]), float(window["end_sec"])
        status = window.get("status")
        waveform = window.get("waveform") or {}
        lag = waveform.get("lag_sec")
        measured_error = (waveform.get("status") == "strong" and not waveform.get("conflict")
                          and isinstance(lag, (int, float)) and math.isfinite(lag) and abs(lag) > tolerance)
        if status == "fail" or measured_error:
            window["status"] = "fail"
            failures.append(f"Coherent local timing error at {start:.3f}-{end:.3f} s")
        elif status == "pass":
            if _strong_aligned(waveform, tolerance):
                independent_thirds.add(min(2, int(((start + end) / 2) / duration * 3)))
        elif status == "silent":
            window["policy_assessment"] = {"accepted_bounded_uncertainty": False, "directly_measured": False,
                                            "reason": "Both tracks contain negligible local audio energy"}
        elif _quiet_both_tracks(waveform):
            window["status"] = "near_silent"
            window["policy_assessment"] = {"accepted_bounded_uncertainty": False, "directly_measured": False,
                                            "reason": "Both tracks have independently characterized negligible local audio energy"}
        elif status != "silent":
            accepted, reason = _bounded_context(window, duration=duration, tolerance=tolerance,
                                                uncertainty=uncertainty, cuts=cuts,
                                                max_gap=max_context_gap_sec,
                                                max_disagreement=max_flank_disagreement_sec)
            window["status"] = "supported_context" if accepted else "review"
            window["policy_assessment"] = {"accepted_bounded_uncertainty": accepted,
                                            "directly_measured": False, "reason": reason}
            if not accepted:
                reviews.append(f"Unresolved local timing at {start:.3f}-{end:.3f} s: {reason}")
    if result.get("status") == "fail" and not failures:
        failures.append("The signal verifier reported a failure without a localized explanation")
    if len(independent_thirds) < 3:
        reviews.append("Independent waveform corroboration missing in one or more thirds")
    bounded = _merge((row["start_sec"], row["end_sec"]) for row in windows if row.get("status") == "supported_context")
    unresolved = _merge((row["start_sec"], row["end_sec"]) for row in windows if row.get("status") == "review")
    measured = _merge((row["start_sec"], row["end_sec"]) for row in windows if row.get("status") == "pass")
    quiet = _merge((row["start_sec"], row["end_sec"]) for row in windows if row.get("status") in {"silent", "near_silent"})
    bounded_seconds = sum(end - start for start, end in bounded)
    result.update(status="fail" if failures else "review" if reviews else "pass",
                  failures=list(dict.fromkeys(failures)), review_reasons=list(dict.fromkeys(reviews)),
                  decision_policy_revision=QC_DECISION_POLICY_REVISION,
                  supported_context_coverage=bounded_seconds / duration,
                  policy_summary={"bounded_unmeasured_ranges": bounded, "bounded_unmeasured_sec": bounded_seconds,
                                  "unresolved_ranges": unresolved,
                                  "directly_supported_check_ranges": measured,
                                  "quiet_unmeasured_ranges": quiet,
                                  "independent_thirds": sorted(independent_thirds),
                                  "max_context_gap_sec": max_context_gap_sec,
                                  "max_flank_disagreement_sec": max_flank_disagreement_sec,
                                  "limitation": "Bounded context is not directly measured audio or semantic narration validation."})
    return result
