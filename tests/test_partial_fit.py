"""Partial fits preserve measured coordinates without manufacturing coverage."""

import json

import numpy as np
import pytest

from adsync.models import CandidateWindow, OffsetCandidate


def lattice(points):
    return [CandidateWindow(source_center=t, speech_score=.5, energy=1,
        candidates=[OffsetCandidate(offset_sec=o, score=.95, peak_sharpness=10, peak_ratio=3)
                    for o in offsets]) for t, offsets in points]


def fit(points, ad_duration=30, video_duration=40, **kwargs):
    from adsync.align.partial_fit import fit_partial_alignment
    return fit_partial_alignment(lattice(points), ad_duration, video_duration, **kwargs)


def test_fit_only_covers_observed_window_support_and_keeps_gaps():
    fns, ranges, path, diag = fit([(t, [2]) for t in (5, 6, 7, 20, 21, 22)])
    assert ranges == [(4, 8), (19, 23)]
    assert len(path.points) == 6
    assert [float(fn(np.mean(r))) - np.mean(r) for fn, r in zip(fns, ranges)] == pytest.approx([2, 2])
    assert [(g["start_sec"], g["end_sec"]) for g in diag["source_gaps"]] == [(0, 4), (8, 19), (23, 30)]
    assert all(g["reason"] == "unmeasured" for g in diag["source_gaps"] + diag["target_gaps"])
    json.dumps(diag, allow_nan=False)


def test_paired_edit_fitting_does_not_delete_three_point_island():
    points = [(t, [] if 46 <= t < 50 else [4 if 40 <= t < 46 else 0]) for t in range(2, 98, 2)]
    fns, ranges, _, diag = fit(points, 100, 100, step_sec=2)
    assert len(fns) == 3
    island = next(i for i, (lo, hi) in enumerate(ranges) if lo <= 42 <= hi)
    assert float(fns[island](42)) == pytest.approx(46)
    assert diag["matched_intervals"][island]["point_count"] == 3
    for a, b in zip(diag["matched_intervals"], diag["matched_intervals"][1:]):
        assert a["source_end"] <= b["source_start"]
        assert a["target_end"] <= b["target_start"]


def test_empty_fit_has_no_identity_fallback():
    fns, ranges, path, diag = fit([])
    assert not fns and not ranges and not path.points
    assert diag["status"] == "unmeasured"
    assert diag["source_gaps"] == [{"start_sec": 0., "end_sec": 30., "reason": "unmeasured"}]


def test_drift_keeps_physical_slope_and_observed_times():
    points = [(t, [2 + .008 * t]) for t in np.arange(2, 20, .9)]
    fns, ranges, _, _ = fit(points)
    assert len(fns) == 1
    probes = np.linspace(*ranges[0], 30)
    assert np.max(abs(fns[0](probes) - (1.008 * probes + 2))) < 1e-9
    assert np.all(abs(fns[0].derivative()(probes) - 1) <= .01 + 1e-9)


def test_changing_local_drift_preserves_measured_positions_and_rate_bounds():
    times = np.arange(1., 101.)
    offsets = 2 + .008 * np.abs(times - 50)
    fns, ranges, path, diag = fit(list(zip(times, [[o] for o in offsets])), 102, 106)
    assert len(fns) == 1
    assert np.max(np.abs(fns[0](times) - times - offsets)) < 1e-8
    dense = np.linspace(*ranges[0], 2000)
    assert np.max(np.abs(fns[0].derivative()(dense) - 1)) <= .01 + 1e-8
    assert all(float(fns[0](p.source_time)) == pytest.approx(p.target_time, abs=1e-8) for p in path.points)
    assert diag["matched_intervals"][0]["max_residual_sec"] < 1e-8


@pytest.mark.parametrize("offset", [-2.5, 8.5])
def test_window_support_is_clipped_in_source_and_target_coordinates(offset):
    fns, ranges, _, diag = fit([(t, [offset]) for t in range(1, 19)], 20, 20)
    for fn, (lo, hi), item in zip(fns, ranges, diag["matched_intervals"]):
        assert 0 <= lo < hi <= 20
        assert float(fn(lo)) >= -1e-8
        assert float(fn(hi)) <= 20 + 1e-8
        assert item["target_end"] - item["target_start"] == pytest.approx(hi - lo)


def test_overlapping_support_is_split_without_target_overlap():
    points = [(t, [2 if t < 5 else 1]) for t in (1, 2, 3, 5, 6, 7)]
    _, ranges, _, diag = fit(points, window_sec=4)
    assert len(ranges) == 2
    first, second = diag["matched_intervals"]
    assert first["source_end"] <= second["source_start"]
    assert first["target_end"] <= second["target_start"]


def test_ambiguous_repeat_is_reported_without_probability_claim():
    _, _, _, diag = fit([(t, [0, 30]) for t in range(1, 9)], 10, 40)
    assert diag["status"] == "ambiguous"
    assert diag["ambiguous_ranges"]
    assert all(w["score_margin"] == pytest.approx(0) for w in diag["windows"])
    assert "probability" not in json.dumps(diag)
