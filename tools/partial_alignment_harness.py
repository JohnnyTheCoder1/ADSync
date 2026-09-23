"""Compare warp and partial alignment on seeded audio with known edit maps.

Generated audio and results must be written outside a Git checkout. This is an
analysis benchmark; dense map scores do not establish rendered-audio quality.
Exit zero means all runs completed; individual fixture verdicts are in JSON.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf
from scipy.interpolate import PPoly
from scipy.signal import butter, sosfilt


SCENARIOS = ("clean", "paired_edit", "short_fragments", "repeated_scene", "unrelated")


@dataclass
class Scenario:
    name: str
    target: np.ndarray
    source: np.ndarray
    fragments: list[dict]
    source_gaps: list[list[float]]
    sample_rate: int = 16000
    ambiguity_expected: bool = False

    def manifest(self) -> dict:
        return {
            "name": self.name,
            "sample_rate": self.sample_rate,
            "source_duration_sec": len(self.source) / self.sample_rate,
            "target_duration_sec": len(self.target) / self.sample_rate,
            "fragments": self.fragments,
            "source_gaps": self.source_gaps,
            "ambiguity_expected": self.ambiguity_expected,
        }


def _soundtrack(rng: np.random.Generator, duration: float, sr: int) -> np.ndarray:
    t = np.arange(round(duration * sr)) / sr
    control_t = np.arange(0, duration + .2, .12)
    result = np.zeros(len(t))
    for lo, hi in ((100, 450), (500, 1800), (2100, 6500)):
        band = sosfilt(butter(3, [lo, hi], btype="bandpass", fs=sr, output="sos"),
                       rng.standard_normal(len(t)))
        envelope = np.interp(t, control_t, rng.uniform(.05, 1, len(control_t))) ** 2
        result += .09 * band * envelope
    pitch = np.interp(t, control_t, rng.uniform(85, 210, len(control_t)))
    phase = 2 * np.pi * np.cumsum(pitch) / sr
    syllables = np.interp(t, control_t, rng.uniform(0, 1, len(control_t))) ** 3
    for harmonic in (1, 2, 3, 5, 8):
        result += .04 / harmonic * np.sin(harmonic * phase) * syllables
    result += .025 * rng.standard_normal(len(t))
    return (result / max(1.0, float(np.max(np.abs(result))) / .8)).astype(np.float32)


def make_scenarios(seed: int = 20260923) -> list[Scenario]:
    rng = np.random.default_rng(seed)
    sr = 16000
    target = _soundtrack(rng, 100, sr)
    unrelated = _soundtrack(rng, 40, sr)

    def clip(a, start, end):
        return a[round(start * sr):round(end * sr)]

    def fragment(name, start, end, *target_starts):
        return {"name": name, "source_start": float(start), "source_end": float(end),
                "target_starts": [float(t) for t in target_starts], "rate": 1.0}

    return [
        Scenario("clean", target[:60 * sr], target[:60 * sr].copy(),
                 [fragment("whole", 0, 60, 0)], []),
        Scenario("paired_edit", target,
                 np.concatenate([clip(target, 0, 40), clip(target, 44, 50),
                                 clip(unrelated, 0, 4), clip(target, 50, 100)]),
                 [fragment("prefix", 0, 40, 0), fragment("short_island", 40, 46, 44),
                  fragment("suffix", 50, 100, 50)], [[46.0, 50.0]]),
        Scenario("short_fragments", target,
                 np.concatenate([clip(target, 0, 25), clip(unrelated, 4, 8),
                                 clip(target, 35, 40), clip(unrelated, 8, 12),
                                 clip(target, 60, 100)]),
                 [fragment("prefix", 0, 25, 0), fragment("five_second_island", 29, 34, 35),
                  fragment("suffix", 38, 78, 60)], [[25.0, 29.0], [34.0, 38.0]]),
        Scenario("repeated_scene",
                 np.concatenate([clip(target, 0, 24), clip(unrelated, 12, 20),
                                 clip(target, 0, 24)]), clip(target, 0, 24).copy(),
                 [fragment("duplicate_scene", 0, 24, 0, 32)], [], ambiguity_expected=True),
        Scenario("unrelated", target[:60 * sr], unrelated, [], [[0.0, 40.0]]),
    ]


def _restore_maps(debug: dict) -> list[tuple[float, float, PPoly]]:
    ranges = debug.get("segment_ranges", [])
    coefficients = debug.get("fitted_pchip", [])
    if len(ranges) != len(coefficients):
        raise ValueError("Fitted map and source range counts differ")
    maps = []
    for bounds, item in zip(ranges, coefficients):
        if len(bounds) != 2 or not np.all(np.isfinite(bounds)) or bounds[1] <= bounds[0]:
            raise ValueError("Invalid fitted source range")
        c, x = np.asarray(item["c"], dtype=float), np.asarray(item["x"], dtype=float)
        if not np.all(np.isfinite(c)) or not np.all(np.isfinite(x)):
            raise ValueError("Nonfinite fitted polynomial")
        if x.ndim != 1 or len(x) < 2 or np.any(np.diff(x) <= 0):
            raise ValueError("Invalid fitted polynomial knots")
        maps.append((float(bounds[0]), float(bounds[1]), PPoly(c, x)))
    return maps


def timing_for_scoring(report: dict) -> dict:
    """Retain real fallback placements when a warp run returns a piecewise map."""
    debug = dict(report.get("timing_debug") or {})
    if "segment_ranges" in debug or "fitted_pchip" in debug:
        debug["scorer_map_source"] = "fitted_pchip"
        return debug
    segments = report.get("segments") or []
    debug["scorer_map_source"] = "segment_map_fallback" if segments else "empty"
    debug["segment_ranges"] = [[s["src_start"], s["src_end"]] for s in segments]
    debug["fitted_pchip"] = [
        {"x": [s["src_start"], s["src_end"]], "c": [[s["stretch"]], [s["dst_start"]]]}
        for s in segments
    ]
    return debug


def _probes(start: float, end: float, margin: float, step: float) -> np.ndarray:
    return np.arange(start + margin + step / 2, end - margin, step)


def _predictions(maps, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = np.zeros(len(times), dtype=int)
    values = np.full(len(times), np.nan)
    for lo, hi, fn in maps:
        mask = (times >= lo) & (times < hi)
        count[mask] += 1
        values[mask] = fn(times[mask])
    return count, values


def _summary(classification: np.ndarray, errors: np.ndarray, conflicts: np.ndarray,
             ambiguous: np.ndarray) -> dict:
    n = len(classification)
    return {
        "scoring_status": "measured" if n else "no_interior_samples",
        "sample_count": n,
        "correct_placement_coverage": float(np.mean(classification == 1)) if n else None,
        "wrong_but_placed_coverage": float(np.mean(classification == 2)) if n else None,
        "unmeasured_coverage": float(np.mean(classification == 0)) if n else None,
        "conflicting_placement_coverage": float(np.mean(conflicts)) if n else None,
        "reported_ambiguous_coverage": float(np.mean(ambiguous)) if n else None,
        "error_median_ms": float(np.median(errors) * 1000) if len(errors) else None,
        "error_p95_ms": float(np.percentile(errors, 95) * 1000) if len(errors) else None,
        "error_max_ms": float(np.max(errors) * 1000) if len(errors) else None,
        "error_sample_count": len(errors),
    }


def score_alignment(debug: dict, fragments: list[dict], source_gaps: list[list[float]], *,
                    boundary_margin_sec: float = 1.0, sample_step_sec: float = .1,
                    tolerance_sec: float = .05) -> dict:
    """Score every known interior; missing predictions never disappear from coverage."""
    if not 0 <= boundary_margin_sec <= 1:
        raise ValueError("Boundary margin must be between zero and one second")
    if not math.isfinite(sample_step_sec) or sample_step_sec <= 0:
        raise ValueError("Sample step must be positive and finite")
    if not math.isfinite(tolerance_sec) or tolerance_sec < 0:
        raise ValueError("Tolerance must be nonnegative and finite")
    maps = _restore_maps(debug)
    summaries, classifications, error_sets, conflicts, ambiguities = [], [], [], [], []
    ambiguous_ranges = debug.get("partial_alignment", {}).get("ambiguous_ranges", [])
    for fragment in fragments:
        start, end = fragment["source_start"], fragment["source_end"]
        times = _probes(start, end, boundary_margin_sec, sample_step_sec)
        count, values = _predictions(maps, times)
        targets = np.asarray(fragment["target_starts"], dtype=float)
        truth = targets[:, None] + (times - start) * fragment.get("rate", 1.0)
        error = np.min(np.abs(truth - values), axis=0)
        single = (count == 1) & np.isfinite(values)
        classification = np.where(count > 0, 2, 0)
        classification[single & (error <= tolerance_sec)] = 1
        measured_error = error[single]
        ambiguous = np.zeros(len(times), dtype=bool)
        for bounds in ambiguous_ranges:
            ambiguous |= (times >= bounds["start_sec"]) & (times < bounds["end_sec"])
        summaries.append({**fragment, **_summary(classification, measured_error, count > 1, ambiguous)})
        classifications.append(classification)
        error_sets.append(measured_error)
        conflicts.append(count > 1)
        ambiguities.append(ambiguous)
    combine = lambda values: np.concatenate(values) if values else np.array([])
    gaps, gap_counts = [], []
    for start, end in source_gaps:
        times = _probes(start, end, boundary_margin_sec, sample_step_sec)
        count, _ = _predictions(maps, times)
        placed = count > 0
        gaps.append({"source_start": start, "source_end": end, "sample_count": len(times),
                     "false_placement_coverage": float(np.mean(placed)) if len(times) else None})
        gap_counts.append(placed)
    placed = combine(gap_counts)
    return {
        "sample_step_sec": sample_step_sec,
        "boundary_margin_sec": boundary_margin_sec,
        "placement_tolerance_sec": tolerance_sec,
        "error_population": "finite, singly placed samples only; use coverage alongside errors",
        "fitted_segment_count": len(maps),
        "fragments": summaries,
        "overall": _summary(combine(classifications), combine(error_sets), combine(conflicts),
                            combine(ambiguities)),
        "source_gaps": gaps,
        "gap_overall": {"sample_count": len(placed),
                        "false_placement_coverage": float(np.mean(placed)) if len(placed) else None},
    }


def evaluate_scores(scores: dict, *, ambiguity_expected: bool) -> dict:
    """Apply explicit fixture checks, including each individual matched island."""
    reasons = []
    for fragment in scores["fragments"]:
        name = fragment["name"]
        if not fragment["sample_count"]:
            reasons.append(f"{name}: no interior samples")
            continue
        if ambiguity_expected:
            if fragment["reported_ambiguous_coverage"] < .9:
                reasons.append(f"{name}: ambiguity reported for less than 90% of its interior")
        elif fragment["correct_placement_coverage"] < .9:
            reasons.append(f"{name}: correct placement below 90%")
        if fragment["wrong_but_placed_coverage"] > .01:
            reasons.append(f"{name}: wrong placement above 1%")
    for gap in scores["source_gaps"]:
        if gap["false_placement_coverage"] is None:
            reasons.append("source gap: no interior samples")
        elif gap["false_placement_coverage"] > .01:
            reasons.append("source gap: false placement above 1%")
    return {"status": "fail" if reasons else "pass", "reasons": reasons}


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _outside_checkout(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if any((ancestor / ".git").exists() for ancestor in (resolved, *resolved.parents)):
        raise ValueError("Choose a workdir outside any Git checkout for generated media")
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=("warp", "partial"), default=["warp", "partial"])
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--threads", type=int, choices=range(1, 9), default=2)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--fingerprint", action="store_true", help="Enable fingerprinting (off by default)")
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        workdir = _outside_checkout(args.workdir)
    except ValueError as exc:
        parser.error(str(exc))
    workdir.mkdir(parents=True, exist_ok=True)
    assets = workdir / "audio"
    assets.mkdir(exist_ok=True)
    cases = [case for case in make_scenarios(args.seed) if case.name in args.scenarios]
    manifest = {"schema_version": 1, "seed": args.seed, "scenarios": [c.manifest() for c in cases]}
    for case in cases:
        sf.write(assets / f"{case.name}_target.wav", case.target, case.sample_rate, subtype="FLOAT")
        sf.write(assets / f"{case.name}_source.wav", case.source, case.sample_rate, subtype="FLOAT")
    _write_json(workdir / "manifest.json", manifest)
    if args.generate_only:
        return 0

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from adsync._pipeline import run_pipeline
    from adsync.config import SyncConfig

    settings = {"device": "cpu", "threads": args.threads, "analysis_sr": 16000,
                "speed_detect": False, "fingerprint": args.fingerprint}
    results = {**manifest, "configuration": settings, "cache": "disabled",
               "acceptance_criteria": {"min_fragment_correct_coverage": .9,
                                       "max_fragment_wrong_coverage": .01,
                                       "max_gap_false_placement_coverage": .01,
                                       "min_repeat_reported_ambiguous_coverage": .9},
               "runs": []}
    cache_setting = os.environ.get("ADSYNC_CACHE_DIR")
    os.environ["ADSYNC_CACHE_DIR"] = "off"
    logger = logging.getLogger("adsync")
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    try:
        for case in cases:
            for mode in dict.fromkeys(args.modes):
                stem = f"{case.name}_{mode}"
                print(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} Starting {stem}", flush=True)
                result = {"scenario": case.name, "requested_mode": mode, "run_status": "error"}
                started = time.perf_counter()
                with (workdir / f"{stem}.log").open("w", encoding="utf-8") as stream:
                    handler = logging.StreamHandler(stream)
                    logger.addHandler(handler)
                    try:
                        with redirect_stdout(stream), redirect_stderr(stream):
                            report = run_pipeline(
                                video_path=assets / f"{case.name}_target.wav",
                                ad_path=assets / f"{case.name}_source.wav", output_path=None,
                                config=SyncConfig(mode=mode, **settings), mux=False,
                            )
                        report_data = report.model_dump()
                        _write_json(workdir / f"{stem}_report.json", report_data)
                        timing = timing_for_scoring(report_data)
                        scores = score_alignment(timing, case.fragments, case.source_gaps)
                        result.update({"run_status": "completed", "reported_mode": report.mode,
                                       "confidence": report.confidence,
                                       "identity_status": report.identity_check.get("status"),
                                       "quality_status": report.quality_check.get("status"),
                                       "warnings": report.warnings, "scores": scores,
                                       "scorer_map_source": timing["scorer_map_source"],
                                       "evaluation": evaluate_scores(scores, ambiguity_expected=case.ambiguity_expected),
                                       "timing_debug": report.timing_debug,
                                       "alignment_review_required": report_data.get("alignment_review_required"),
                                       "ambiguity_expected": case.ambiguity_expected})
                    except Exception as exc:
                        result["error"] = f"{type(exc).__name__}: {exc}"
                    finally:
                        logger.removeHandler(handler)
                result["wall_time_sec"] = time.perf_counter() - started
                results["runs"].append(result)
                _write_json(workdir / "partial-results.json", results)
                print(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} "
                      f"{stem}: {result['run_status']} ({result['wall_time_sec']:.2f}s)", flush=True)
    finally:
        logger.setLevel(previous_level)
        if cache_setting is None:
            os.environ.pop("ADSYNC_CACHE_DIR", None)
        else:
            os.environ["ADSYNC_CACHE_DIR"] = cache_setting
    return 0 if all(r["run_status"] == "completed" for r in results["runs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
