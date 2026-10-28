"""Calibrate local QC with real aligned audio, known shifts and staged outputs.

The two source files must already contain the same aligned content. A review
is recorded separately from a measured failure; no thresholds are changed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def load_excerpt(path: Path, start: float, duration: float | None, sr: int = 16000) -> np.ndarray:
    with sf.SoundFile(path) as handle:
        handle.seek(round(start * handle.samplerate))
        y = handle.read(-1 if duration is None else round(duration * handle.samplerate),
                        dtype="float32", always_2d=True).mean(axis=1)
        original_sr = handle.samplerate
    if original_sr != sr:
        ratio = Fraction(sr, original_sr)
        y = resample_poly(y, ratio.numerator, ratio.denominator).astype(np.float32)
    return y


def shifted(y: np.ndarray, seconds: float, sr: int) -> np.ndarray:
    count = round(abs(seconds) * sr)
    if not count:
        return y.copy()
    output = np.zeros_like(y)
    if count < len(y):
        if seconds > 0:
            output[count:] = y[:-count]
        else:
            output[:-count] = y[count:]
    return output


def assess(main: np.ndarray, ad: np.ndarray, sr: int) -> dict:
    from adsync.quality import verify_audio_sync
    from adsync.hardware import thread_budget
    with thread_budget():
        return verify_audio_sync(main, ad, sr)


def summarize(name: str, expected: str, result: dict, elapsed: float) -> dict:
    accepted = result["status"] == expected if expected != "no_false_failure" else result["status"] != "fail"
    return {"name": name, "expected": expected, "observed": result["status"],
            "control_satisfied": accepted, "elapsed_seconds": elapsed,
            "window_counts": dict(Counter(w.get("status", "unknown") for w in result.get("windows", []))),
            "failures": result.get("failures", []), "review_reasons": result.get("review_reasons", []),
            "policy_summary": result.get("policy_summary")}


def progress(name: str, result: dict, expected: str | None = None) -> None:
    expectation = f", expected {expected}" if expected else ""
    print(f"{datetime.now(timezone.utc).isoformat()} {name}: {result['status']}{expectation}; "
          f"{len(result.get('failures', []))} failures, {len(result.get('review_reasons', []))} review reasons",
          flush=True)


def run_calibration(main_path: Path, ad_path: Path, output: Path, *, sr: int = 16000,
                    start: float = 0, duration: float = 60, include_full: bool = True) -> list[dict]:
    output.mkdir(parents=True, exist_ok=True)
    main = load_excerpt(main_path, start, duration, sr)
    ad = load_excerpt(ad_path, start, duration, sr)
    size = min(len(main), len(ad))
    main, ad = main[:size], ad[:size]
    local = ad.copy()
    lo, hi, shift = round(42 * sr), round(44 * sr), round(.8 * sr)
    if hi > len(local):
        raise ValueError("Calibration excerpt must exceed 44 seconds")
    local[lo:hi] = ad[lo - shift:hi - shift]
    cases = [
        ("real_waveform_identity", "pass", main, main.copy()),
        ("real_aligned_description", "pass", main, ad),
        ("real_delay_200ms", "fail", main, shifted(ad, .2, sr)),
        ("real_advance_200ms", "fail", main, shifted(ad, -.2, sr)),
        ("real_delay_800ms", "fail", main, shifted(ad, .8, sr)),
        ("real_delay_1500ms", "fail", main, shifted(ad, 1.5, sr)),
        ("real_hidden_two_second_delay", "fail", main, local),
    ]
    summaries = []
    for name, expected, x, y in cases:
        started = time.perf_counter()
        result = assess(x, y, sr)
        summary = summarize(name, expected, result, time.perf_counter() - started)
        summaries.append(summary)
        (output / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        progress(name, result, expected)
    if include_full:
        x, y = load_excerpt(main_path, 0, None, sr), load_excerpt(ad_path, 0, None, sr)
        size = min(len(x), len(y))
        started = time.perf_counter()
        result = assess(x[:size], y[:size], sr)
        summary = summarize("real_aligned_full", "no_false_failure", result, time.perf_counter() - started)
        summaries.append(summary)
        (output / "real_aligned_full.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        progress("real_aligned_full", result, "no false failure")
    receipt = {"main": str(main_path.resolve()), "description": str(ad_path.resolve()), "sample_rate": sr,
               "excerpt_start_sec": start, "excerpt_duration_sec": duration,
               "positive_control": "Real independently mixed description paired to its known-aligned original soundtrack",
               "known_local_error": {"excerpt_interval_sec": [42, 44], "description_delay_sec": .8},
               "results": summaries,
               "all_controls_satisfied": all(row["control_satisfied"] for row in summaries)}
    (output / "calibration-summary.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return summaries


def run_staged(progress_dir: Path, output: Path, labels: list[str] | None = None) -> list[dict]:
    from adsync.quality import verify_media_sync
    from adsync.hardware import thread_budget
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for progress_path in sorted(progress_dir.glob("S*.progress.json")):
        label = progress_path.name.removesuffix(".progress.json")
        if labels and label not in labels:
            continue
        record = json.loads(progress_path.read_text(encoding="utf-8"))
        media = Path(record["local_path"])
        expected = record["local_signature"]
        current = media.stat()
        if (current.st_size, current.st_mtime_ns) != (expected["size"], expected["mtime_ns"]):
            raise RuntimeError(f"Staged media differs from its signature: {label}")
        report_path = progress_dir / f"{label}.report.json"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
        with thread_budget():
            result = verify_media_sync(media, report=report)
        result["staged_signature"] = expected
        (output / f"{label}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        summary = {"label": label, "status": result["status"], "failures": result.get("failures", []),
                   "review_reasons": result.get("review_reasons", [])}
        results.append(summary)
        (output / "staged-summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        progress(label, result)
    if not results:
        raise ValueError("No staged progress records matched the requested folder and labels")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=Path)
    parser.add_argument("--ad", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--skip-full", action="store_true")
    parser.add_argument("--staged-progress-dir", type=Path)
    parser.add_argument("--labels", nargs="*")
    args = parser.parse_args()
    if bool(args.main) != bool(args.ad):
        parser.error("--main and --ad must be provided together")
    satisfied = True
    if args.main and args.ad:
        results = run_calibration(args.main, args.ad, args.output_dir, start=args.start,
                                  duration=args.duration, include_full=not args.skip_full)
        satisfied = all(row["control_satisfied"] for row in results)
    elif not args.staged_progress_dir:
        parser.error("Provide --main and --ad, or --staged-progress-dir")
    if args.staged_progress_dir:
        staged = run_staged(args.staged_progress_dir, args.output_dir / "staged", args.labels)
        satisfied = satisfied and all(row["status"] != "fail" for row in staged)
    return 0 if satisfied else 1


if __name__ == "__main__":
    raise SystemExit(main())
