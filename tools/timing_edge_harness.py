"""Bounded, reproducible audio regressions for small and short terminal cuts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.interpolate import PPoly

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from adsync._pipeline import run_pipeline
from adsync.config import SyncConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    sr, duration = 16000, 120
    rng = np.random.default_rng(20260922)
    time = np.arange(sr * duration) / sr
    # Broadband, time-varying shared soundtrack with no repeating motif.
    video = (rng.normal(0, .08, len(time)) * (.65 + .25 * np.sin(.7 * time))
             + .035 * np.sin(2 * np.pi * (120 * time + 2 * time * time))).astype(np.float32)
    video_path = args.workdir / "source.wav"
    sf.write(video_path, video, sr, subtype="FLOAT")
    results = []
    for name, boundary, removed in (("smallcut075", 60, .75), ("terminal10", 100, 10)):
        ad = np.r_[video[:round(boundary * sr)], video[round((boundary + removed) * sr):]]
        ad_path = args.workdir / f"{name}.wav"
        sf.write(ad_path, ad, sr, subtype="FLOAT")
        report = run_pipeline(video_path=video_path, ad_path=ad_path, output_path=None,
                              config=SyncConfig(device=args.device, threads=4, speed_detect=False),
                              report_path=args.workdir / f"report_{name}.json", mux=False)
        debug = report.timing_debug
        ranges = debug.get("segment_ranges", [])
        fns = [PPoly(np.array(p["c"]), np.array(p["x"])) for p in debug.get("fitted_pchip", [])]
        probes = np.arange(.5, len(ad) / sr - .5, .25)
        # A fixed one-second margin around the known splice reports the
        # uncertainty explicitly; far tighter than the standard harness's 6s.
        probes = probes[np.abs(probes - boundary) > 1]
        errors = []
        for t in probes:
            fn = next((fn for fn, (lo, hi) in zip(fns, ranges) if lo <= t <= hi), None)
            if fn is None:
                raise RuntimeError(f"Missing fitted source range for {name} at {t}")
            errors.append(abs(float(fn(t)) - t - (removed if t >= boundary else 0)) * 1000)
        terminal = debug.get("fingerprint", {}).get("terminal_support", [])
        result = {"name": name, "status": "pass" if max(errors) <= 50 else "fail",
                  "known_cut_sec": boundary, "removed_sec": removed,
                  "excluded_cut_margin_sec": 1, "p95_ms": float(np.percentile(errors, 95)),
                  "max_ms": max(errors), "within_50_ms_percent": 100 * np.mean(np.array(errors) <= 50),
                  "fitted_ranges": ranges, "terminal_support": terminal,
                  "boundary_refinement": debug.get("boundary_refinement", []),
                  "content_identity": report.identity_check.get("status")}
        if name == "terminal10" and not terminal:
            result["status"] = "fail"
            result["reason"] = "Ten-second terminal span was not independently confirmed"
        results.append(result)
        print(f"{name}: {result['status']}, p95={result['p95_ms']:.6f} ms, "
              f"max={result['max_ms']:.6f} ms, ranges={ranges}, "
              f"confirmed terminal pieces={len(terminal)}")
    (args.workdir / "edge-results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0 if all(r["status"] == "pass" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
