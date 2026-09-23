"""Compare complete season throughput with different episode worker counts.

Uses separate output/state directories for each measurement. Run against a
small representative season before choosing an explicit --jobs override.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--ad-dir", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if any(n < 1 for n in args.jobs):
        parser.error("All worker counts must be positive")
    root = args.workdir.resolve() / time.strftime("%Y%m%d-%H%M%S")
    root.mkdir(parents=True, exist_ok=False)
    results = []
    for count in args.jobs:
        case = root / f"jobs-{count}"
        case.mkdir()
        command = [sys.executable, "-m", "adsync", "season", str(args.video_dir.resolve()),
                   str(args.ad_dir.resolve()), "--output-dir", str(case / "output"),
                   "--state-dir", str(case / "state"), "--jobs", str(count), "--device", args.device]
        print(f"{time.strftime('%H:%M:%S')} Starting {count} worker(s)", flush=True)
        started = time.perf_counter()
        with (case / "run.log").open("w", encoding="utf-8") as handle:
            process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
        elapsed = time.perf_counter() - started
        if process.returncode not in (0, 1):
            raise RuntimeError(f"Benchmark failed with exit {process.returncode}; see {case / 'run.log'}")
        summary = json.loads((case / "state" / "season-summary.json").read_text(encoding="utf-8"))
        completed = sum(summary["counts"].get(s, 0) for s in ("completed", "needs_review"))
        if completed != len(summary["batch"]["results"]) or not completed:
            raise RuntimeError("Every benchmark episode must finish a new sync")
        gpu_calls = 0
        scores = []
        for episode in summary["batch"]["results"]:
            report = json.loads(Path(episode["report_path"]).read_text(encoding="utf-8"))
            gpu_calls += report["gpu_correlations"]
            scores.append(report["confidence"])
        result = {"requested_jobs": count, "actual_jobs": summary["resources"]["jobs"],
                  "threads_per_job": summary["resources"]["threads_per_job"],
                  "episodes": completed, "elapsed_seconds": elapsed, "gpu_correlations": gpu_calls,
                  "confidences": scores, "summary": str(case / "state" / "season-summary.json")}
        results.append(result)
        (root / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"{time.strftime('%H:%M:%S')} {count} workers: {elapsed:.2f} s, "
              f"{completed} episodes, {gpu_calls} GPU correlations", flush=True)
    best = min(results, key=lambda r: r["elapsed_seconds"])
    print(f"{time.strftime('%H:%M:%S')} Fastest measured: {best['actual_jobs']} workers; results: {root / 'results.json'}")


if __name__ == "__main__":
    main()
