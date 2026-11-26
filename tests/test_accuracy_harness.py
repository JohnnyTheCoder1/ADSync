"""Regression coverage for scoring anchors in ADSync analysis reports."""
from __future__ import annotations

import json
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

HARNESS = Path(__file__).resolve().parents[1] / "tools" / "accuracy_harness.py"


class AccuracyHarnessTests(unittest.TestCase):
    def test_scores_report_anchors_from_ad_time_to_video_time(self) -> None:
        harness = runpy.run_path(str(HARNESS))
        make_scenarios = harness["make_scenarios"]
        run_scenario = harness["run_scenario"]
        scenario = next(sc for sc in make_scenarios(185.0) if sc.name == "offset5")
        # Known five-second AD delay: the sample at AD 35 s belongs at video 30 s.
        report = {
            "confidence": 0.99,
            "global_offset": -5.0,
            "anchors": [{"source_time": 35.0, "target_time": 30.0}],
        }

        for mode in ("offset", "drift", "piecewise", "warp"):
            with self.subTest(mode=mode), TemporaryDirectory() as work:
                workdir = Path(work)

                def write_report(*, report_path, **kwargs):
                    report_path.write_text(json.dumps({**report, "mode": mode}), encoding="utf-8")

                with (patch("subprocess.run"),
                      patch("adsync._pipeline.run_pipeline", side_effect=write_report)):
                    result = run_scenario(scenario, Path("video.flac"), Path("ad.wav"), workdir, 185.0)

                self.assertIsNone(result.error)
                self.assertEqual(result.n_anchors_scored, 1)
                self.assertAlmostEqual(result.p95_ms, 0.0)
                self.assertEqual(result.pct_within_50ms, 100.0)
