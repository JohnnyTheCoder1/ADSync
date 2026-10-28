"""Measure CPU/CUDA correlation latency including host/device transfers.

Run with the project's Python: python tools/benchmark_compute.py --json-out results.json
These are warmed correlation timings, not full movie processing times.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adsync.compute import CorrelationBackend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")

    started = time.perf_counter()
    gpu = CorrelationBackend("cuda")
    startup_seconds = time.perf_counter() - started
    cpu = CorrelationBackend("cpu")
    results = []
    rng = np.random.default_rng(15)
    try:
        for name, region_size, window_size in (
            ("small", 4096, 512),
            ("warp_30s_radius", 272_000, 32_000),
            ("warp_90s_radius", 752_000, 32_000),
            ("global_120s_radius", 4_320_000, 480_000),
        ):
            region = rng.normal(size=region_size)
            template = region[region_size // 3:region_size // 3 + window_size].copy()
            expected = cpu.normalized_correlation(region, template)
            actual = gpu.normalized_correlation(region, template)
            np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=2e-10)
            if np.argmax(actual) != region_size // 3:
                raise AssertionError("CUDA chose the wrong matching position")
            elapsed = {}
            for label, backend in (("cpu", cpu), ("cuda", gpu)):
                measurements = []
                for _ in range(args.runs):
                    started = time.perf_counter()
                    backend.normalized_correlation(region, template)
                    measurements.append(time.perf_counter() - started)
                elapsed[label] = statistics.median(measurements)
            result = {
                "workload": name, "region_samples": region_size,
                "template_samples": window_size, "runs": args.runs,
                "cpu_seconds": elapsed["cpu"], "cuda_seconds": elapsed["cuda"],
                "speedup": elapsed["cpu"] / elapsed["cuda"],
                "max_score_error": float(np.max(np.abs(actual - expected))),
            }
            results.append(result)
            print(f"{time.strftime('%H:%M:%S')} {name}: CPU {elapsed['cpu'] * 1000:.2f} ms, "
                  f"CUDA {elapsed['cuda'] * 1000:.2f} ms, {result['speedup']:.2f}x", flush=True)
        report = {"gpu": gpu.device_name, "cuda_startup_seconds": startup_seconds,
                  "includes_transfers": True, "warmed": True, "results": results}
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    finally:
        gpu.close()
        cpu.close()


if __name__ == "__main__":
    main()
