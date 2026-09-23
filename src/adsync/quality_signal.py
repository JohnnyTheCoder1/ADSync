"""Signal-level diagnostics for local QC; these do not make publication decisions."""

from __future__ import annotations

import math
import numpy as np


ENVELOPE_CONFLICT_REVISION = "statistical-raw-envelope-consistency-1"


def raw_peak_supported(row: dict, *, require_unique: bool = False) -> bool:
    """One calibrated carrier-evidence rule for agreement and disagreement.

    Weak absolute correlation can still identify shared sound under narration.
    Requiring a distinctive competing-peak margin makes a single band's
    contradiction meaningful; corroborating bands may have repeated notes.
    """
    if not row.get("usable_peak"):
        return False
    try:
        score, peak_z, ratio, lag = (float(row.get(key, 0)) for key in ("score", "peak_z", "peak_ratio", "lag_sec"))
    except (TypeError, ValueError):
        return False
    return (all(math.isfinite(value) for value in (score, peak_z, ratio, lag))
            and score >= .015 and peak_z >= 8 and (not require_unique or ratio >= 1.4))


def reassess_envelope_conflict(waveform: dict, *, measurement_revision: str | None = None) -> dict:
    """Apply the production conflict rule to fresh or saved signal measurements.

    This pure decision helper never creates a passing verdict. It retains the
    original measurement revision, signal statistics, and caller-supplied hash
    bindings. Old records without the necessary raw statistics explicitly need
    remeasurement instead of being assumed safe. Callers retain their original
    QC report and record this reassessment separately.
    """
    revised = dict(waveform)
    result = {"decision": "unchanged", "measurement_revision": measurement_revision,
              "decision_rule_revision": ENVELOPE_CONFLICT_REVISION,
              "original_measurement": {key: waveform.get(key) for key in ("status", "lag_sec", "method")},
              "waveform": revised}
    if waveform.get("status") != "strong" or waveform.get("method") not in {"band_energy_envelope", "separated_envelope_chunks"}:
        return result
    missing = []
    bands = waveform.get("bands") or {}
    if not bands:
        missing.append("bands")
    try:
        lag = float(waveform.get("lag_sec"))
        if not math.isfinite(lag):
            raise ValueError("Nonfinite lag")
    except (TypeError, ValueError):
        missing.append("lag_sec")
        lag = 0.0
    for name, row in bands.items():
        if name == "full" or row.get("lag_sec") is None:
            continue
        for key in ("score", "peak_z", "peak_ratio", "usable_peak"):
            if key not in row:
                missing.append(f"bands.{name}.{key}")
    if missing:
        result.update(decision="remeasure", missing_statistics=missing)
        return result
    contrary = [{"name": name, **row} for name, row in bands.items() if name != "full"
                and raw_peak_supported(row, require_unique=True) and abs(row["lag_sec"] - lag) > .10]
    if contrary:
        conflict = {"reason": "Envelope timing conflicts with a uniquely supported raw frequency-band peak",
                    "envelope_lag_sec": lag, "envelope_method": waveform["method"], "raw_bands": contrary}
        revised.update(status="weak", lag_sec=None, method=None, conflict=conflict)
        result["decision"] = "conflict"
    return result


def signal_levels(main: np.ndarray, ad: np.ndarray, sr: int, start: float, end: float) -> dict:
    """Distinguish negligible residual noise from an intrinsically quiet programme.

    DC is removed from both the requested window and reference blocks. The
    reference is the 90th percentile of nearby half-second AC RMS values, up to
    thirty seconds either side. A peak bound prevents a short genuine transient
    from being hidden by its low average energy. 'near_silent' is unmeasured,
    not evidence of zero lag or a general human-audibility claim.
    """
    def measure(audio):
        audio = np.asarray(audio)
        lo, hi = max(0, round(start * sr)), min(len(audio), round(end * sr))
        window = np.asarray(audio[lo:hi], dtype=np.float64)
        if not len(window):
            return {"ac_rms": 0.0, "peak_ac": 0.0, "reference_rms": 0.0,
                    "relative_db": None, "noise_floor_rms": 1e-7, "quiet": True}
        centered = window - window.mean()
        rms = float(np.sqrt(np.mean(centered ** 2)))
        peak = float(np.max(np.abs(centered)))
        reference = np.asarray(audio[max(0, lo - 30 * sr):min(len(audio), hi + 30 * sr)], dtype=np.float64)
        block = max(1, round(.5 * sr))
        blocks = reference[:len(reference) // block * block].reshape(-1, block)
        if len(blocks):
            powers = np.mean(blocks * blocks, axis=1) - np.mean(blocks, axis=1) ** 2
            reference_rms = float(np.percentile(np.sqrt(np.maximum(powers, 0)), 90))
        else:
            reference_rms = rms
        floor = min(2e-4, max(1e-7, reference_rms * .003))
        quiet = rms <= floor and peak <= max(2e-6, min(.001, 5 * floor))
        return {"ac_rms": rms, "peak_ac": peak, "reference_rms": reference_rms,
                "relative_db": 20 * math.log10(max(rms, 1e-15) / max(reference_rms, 1e-15)),
                "noise_floor_rms": floor, "quiet": bool(quiet)}

    first, second = measure(main), measure(ad)
    state = ("near_silent" if first["quiet"] and second["quiet"] else "main_quiet" if first["quiet"]
             else "ad_quiet" if second["quiet"] else "active")
    return {"state": state, "main": first, "ad": second,
            "template_start_sec": start, "template_end_sec": end}
