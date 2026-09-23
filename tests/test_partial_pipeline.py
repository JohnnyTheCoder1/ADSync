import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator
from typer.testing import CliRunner

from adsync.cli import app
from adsync.config import SyncConfig
from adsync.models import CandidateWindow, OffsetCandidate, SyncReport, WarpPath, WarpPoint
from test_pipeline_regressions import pipeline_case


@pytest.fixture
def partial_case(pipeline_case, monkeypatch):
    state, run = pipeline_case, pipeline_case.run
    state.lattice = [CandidateWindow(
        source_center=float(t), speech_score=1, energy=1,
        candidates=[OffsetCandidate(offset_sec=0, score=.95, peak_sharpness=10, peak_ratio=2)],
    ) for t in range(1, 12)]
    state.partial_diagnostics = {
        "algorithm": "partial-affine-gap-v1", "status": "matched", "path_score": 8.0,
        "matched_intervals": [], "source_gaps": [], "target_gaps": [],
        "ambiguous_ranges": [], "windows": [], "candidates": {},
    }
    state.partial_ranges = [(0., 12.)]
    state.partial_functions = [PchipInterpolator([0., 12.], [0., 12.])]

    def fit(lattice, ad_duration, video_duration, **kwargs):
        state.partial_options = kwargs
        points = [WarpPoint(source_time=t, target_time=float(fn(t)), confidence=.95, is_anchor=True)
                  for fn, (lo, hi) in zip(state.partial_functions, state.partial_ranges)
                  for t in (lo, (lo + hi) / 2, hi)]
        path = WarpPath(points=points, anchor_points=points, n_segments=len(state.partial_ranges),
                        mean_confidence=.95 if points else 0)
        return state.partial_functions, state.partial_ranges, path, state.partial_diagnostics

    monkeypatch.setitem(sys.modules, "adsync.align.partial_fit", SimpleNamespace(fit_partial_alignment=fit))
    return state, run


def test_partial_mode_is_an_explicit_opt_in():
    assert SyncConfig(mode="partial").mode == "partial"
    assert SyncConfig().mode == "auto"


def test_partial_pipeline_renders_supported_fragment_without_filling_gaps(partial_case):
    state, run = partial_case
    state.partial_ranges = [(2., 5.)]
    state.partial_functions = [PchipInterpolator([2., 5.], [4., 7.])]
    state.partial_diagnostics["source_gaps"] = [{"start_sec": 5., "end_sec": 12., "reason": "unmeasured"}]
    report = run("partial", mux=True)
    assert report.mode == "partial"
    assert np.argmax(np.abs(state.rendered)) == 5 * state.sr
    assert np.all(state.rendered[:4 * state.sr] == 0)
    assert np.all(state.rendered[8 * state.sr:] == 0)
    assert report.confidence < .7
    assert report.timing_debug["partial_alignment"]["source_gaps"]
    assert state.partial_options["window_sec"] == 2.
    assert state.partial_options["step_sec"] == 1.


def test_partial_without_accepted_matches_does_not_mux_or_load_hq(partial_case):
    state, run = partial_case
    state.partial_ranges = []
    state.partial_functions = []
    state.partial_diagnostics["status"] = "unmeasured"
    report = run("partial", mux=True)
    assert report.confidence == 0
    assert report.output_path is None
    assert state.rendered is None
    assert not any(name == "ad_audio_hq.wav" for name, _ in state.extractions)
    assert any("no supported" in warning.lower() for warning in report.warnings)


def test_partial_ambiguity_cannot_report_success(partial_case):
    state, run = partial_case
    state.partial_diagnostics.update(status="ambiguous", ambiguous_ranges=[
        {"start_sec": 1., "end_sec": 11., "reason": "competing_path", "score_margin": 0.},
    ])
    report = run("partial")
    assert report.confidence < .7
    assert any("competing" in warning.lower() for warning in report.warnings)


def test_partial_short_internal_gap_requires_review(partial_case):
    state, run = partial_case
    state.partial_ranges = [(0., 5.), (5.5, 12.)]
    state.partial_functions = [PchipInterpolator([lo, hi], [lo, hi]) for lo, hi in state.partial_ranges]
    state.partial_diagnostics["source_gaps"] = [{"start_sec": 5., "end_sec": 5.5, "reason": "unmeasured"}]
    report = run("partial")
    assert report.alignment_review_required
    assert report.confidence < .7


def test_partial_manual_offset_is_applied_once_and_reported(partial_case, tmp_path):
    state, run = partial_case
    report = run("partial", mux=True, offset_adjust=1.25, report_path=tmp_path / "report.json")
    assert np.argmax(np.abs(state.rendered)) == round(4.25 * state.sr)
    assert report.anchors[0].target_time == pytest.approx(report.anchors[0].source_time + 1.25)
    assert report.warp_path.points[0].target_time == pytest.approx(1.25)
    assert report.timing_debug["partial_alignment"]["playback_offset_adjust_sec"] == 1.25
    assert SyncReport.model_validate_json((tmp_path / "report.json").read_text()).mode == "partial"


@pytest.mark.parametrize("command", ["sync", "analyze", "debug"])
def test_partial_is_forwarded_by_single_file_commands(tmp_path, monkeypatch, command):
    video, ad = tmp_path / "video.mkv", tmp_path / "ad.wav"
    video.touch()
    ad.touch()
    seen = []
    def pipeline(**kwargs):
        seen.append(kwargs["config"])
        return SyncReport(mode="partial", confidence=.99)
    monkeypatch.setattr("adsync._pipeline.run_pipeline", pipeline)
    args = [command, str(video), str(ad), "--mode", "partial", "--device", "cpu"]
    if command == "debug":
        args += ["--workdir", str(tmp_path / "debug")]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert seen[0].mode == "partial"


def test_review_status_cannot_be_bypassed_with_zero_confidence_threshold(tmp_path, monkeypatch):
    video, ad = tmp_path / "video.mkv", tmp_path / "ad.wav"
    video.touch()
    ad.touch()
    monkeypatch.setattr("adsync._pipeline.run_pipeline", lambda **kwargs:
                        SyncReport(mode="partial", confidence=.99, alignment_review_required=True))
    result = CliRunner().invoke(app, ["sync", str(video), str(ad), "--mode", "partial",
                                     "--confidence-threshold", "0"])
    assert result.exit_code == 1, result.output
