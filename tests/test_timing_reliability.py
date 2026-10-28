"""Regressions for measured cuts, sparse local evidence and closing edits."""

import numpy as np
import pytest

from adsync.align.candidate_lattice import augment_lattice_with_fingerprint
from adsync.align.fingerprint import FingerprintResult, FingerprintSpan, _offset_spans
from adsync.align.warp_fit import fit_warp_function
from adsync.models import CandidateWindow, OffsetCandidate, WarpPoint


def _point(t, offset, confidence=0.8):
    return WarpPoint(source_time=t, target_time=t + offset, confidence=confidence)


def _at(fns, ranges, t):
    i = next(i for i, (lo, hi) in enumerate(ranges) if lo <= t <= hi)
    return float(fns[i](t))


@pytest.mark.parametrize("jump", [0.4, 0.8, -0.8, 1.2])
def test_small_measured_cut_does_not_bend_flat_regions(jump):
    path = [_point(float(t), 3 + (jump if t >= 60 else 0), 0.9 if t < 30 else 0.5)
            for t in range(0, 120, 2)]
    fns, ranges, _ = fit_warp_function(path, 120, 130)
    for t in (35, 45, 53, 57, 61, 67, 79, 99):
        assert _at(fns, ranges, t) - t == pytest.approx(3 + (jump if t >= 60 else 0), abs=0.025)
    assert len(fns) == 2


def test_anchor_selection_preserves_local_measured_drift():
    times = np.arange(0, 100, 2, dtype=float)
    offsets = 2 + 0.3 * np.sin(times / 12)
    path = [_point(t, off, 0.9 if t < 50 else 0.3) for t, off in zip(times, offsets)]
    fns, ranges, _ = fit_warp_function(path, 100, 110)
    errors = [abs(_at(fns, ranges, t) - t - off) for t, off in zip(times, offsets)]
    assert max(errors) < 0.04


def test_terminal_extrapolation_does_not_compress_measured_offset():
    path = [_point(float(t), 2) for t in range(0, 98, 2)]
    fns, ranges, _ = fit_warp_function(path, 100, 100)
    assert _at(fns, ranges, 99) == pytest.approx(101)


def test_short_dense_terminal_is_preserved_as_unconfirmed_evidence():
    body = np.arange(0.1, 80, 0.1, dtype=np.float32)
    tail = np.arange(91, 93, 0.025, dtype=np.float32)
    times = np.r_[body, tail]
    targets = np.r_[body + 2, tail + 4]
    fp = _offset_spans(times, targets, 100)
    assert len(fp.short_terminal_spans) == 1
    assert fp.short_terminal_spans[0].offset == pytest.approx(4)
    assert len(fp.spans) == 1, "Unconfirmed terminal hypotheses must not steer alignment"


def test_local_fingerprint_fallback_leaves_good_correlation_untouched():
    times = np.arange(0.1, 100, 0.05, dtype=np.float32)
    fp = FingerprintResult(spans=[FingerprintSpan(0, 100, 3, len(times))],
                           match_t_ad=times, match_t_vid=times + 3)
    lattice = [CandidateWindow(source_center=float(t), speech_score=0.5, energy=1,
               candidates=[] if t == 50 else [OffsetCandidate(offset_sec=3, score=.95,
                   peak_sharpness=4, peak_ratio=4)]) for t in range(4, 96, 2)]
    n = augment_lattice_with_fingerprint(lattice, fp, only_weak=True)
    assert n == 1
    assert [w for w in lattice if w.source_center == 50][0].candidates[0].source == "fingerprint"
    assert all(len(w.candidates) == 1 for w in lattice)


def test_confirmed_short_negative_terminal_survives_decoder_vetting_and_render():
    from adsync.align.fingerprint import TerminalSupport
    from adsync.align.warp_decode import decode_warp_path
    from adsync.rebuild.warp_render import render_from_warp

    support = TerminalSupport(boundary=92, body_time=88, body_offset=4,
                              first_time=96, last_time=97, offset=2,
                              matches=40, waveform_score=.8)
    lattice = [CandidateWindow(source_center=float(t), speech_score=.5, energy=1,
               candidates=[OffsetCandidate(offset_sec=4, score=.9, peak_sharpness=4, peak_ratio=4)])
               for t in range(4, 98, 2)]
    points, _ = decode_warp_path(lattice, terminal_support=[support])
    assert next(p for p in points if p.source_time == 96).target_time == pytest.approx(98)
    fns, ranges, wp = fit_warp_function(points, 100, 106, terminal_support=[support])
    assert len(fns) == 2
    assert ranges == [(0, 92), (92, 100)]
    assert _at(fns, ranges, 89) == pytest.approx(93)
    assert _at(fns, ranges, 97) == pytest.approx(99)
    assert all(np.all(fn.derivative()(np.linspace(*r, 30)) > 0) for fn, r in zip(fns, ranges))
    signal = np.zeros(10000, dtype=np.float32)
    signal[9700] = .5
    signal[9701] = -.5
    rendered = render_from_warp(signal, 100, fns, ranges, 106)
    assert np.argmax(rendered) == 9900


def test_waveform_corroboration_rejects_silence_and_wrong_content():
    from adsync.align.fingerprint import _terminal_waveform_support

    rng = np.random.default_rng(24)
    sr = 8000
    source = rng.normal(0, .1, sr * 6).astype(np.float32)
    video = np.r_[np.zeros(sr), source, np.zeros(sr)].astype(np.float32)
    assert _terminal_waveform_support(video, source, sr, 2, 4, 1) > .8
    assert _terminal_waveform_support(video, source * 1e-9, sr, 2, 4, 1) is None
    assert _terminal_waveform_support(video, source, sr, 2, 4, -1) is None


@pytest.mark.parametrize("matching_audio", [True, False])
def test_terminal_promotion_requires_independent_audio(matching_audio):
    from adsync.align.fingerprint import _recover_terminal_support

    rng = np.random.default_rng(37)
    sr = 8000
    ad = rng.normal(0, .1, sr * 100).astype(np.float32)
    video = rng.normal(0, .1, sr * 104).astype(np.float32)
    if matching_audio:
        video[92 * sr:102 * sr] = ad[90 * sr:100 * sr]
    body = np.arange(.1, 80, .1, dtype=np.float32)
    tail = np.arange(96, 98, .025, dtype=np.float32)
    times, targets = np.r_[body, tail], np.r_[body + 4, tail + 2]
    fp = _offset_spans(times, targets, 100)
    _recover_terminal_support(fp, times, targets, video, ad, sr)
    assert bool(fp.terminal_support) == matching_audio
    assert len(fp.spans) == (2 if matching_audio else 1)
    if matching_audio:
        assert fp.terminal_support[0].offset == pytest.approx(2)
        assert fp.match_t_ad[-1] > 97


@pytest.mark.parametrize("bad_region", [False, True])
def test_local_residual_clusters_ignore_scattered_coincidences_but_keep_bad_spans(bad_region):
    from adsync.align.confidence import local_fingerprint_residuals
    from scipy.interpolate import PchipInterpolator

    times = np.arange(.1, 200, .1)
    residual = np.zeros(len(times))
    residual[::7] = .7  # harmless sparse hash coincidences under the inlier gate
    if bad_region:
        residual[(times >= 100) & (times < 120)] = 1.5
    result = local_fingerprint_residuals(
        [PchipInterpolator([0, 200], [2, 202])], [(0, 200)], times, times + 2 + residual,
    )
    assert result["raw_p95_ms"] > 600
    assert result["p95_ms"] == pytest.approx(1500 if bad_region else 0)
    assert result["status"] == ("review" if bad_region else "pass")


def test_waveform_refinement_moves_early_synthetic_cut_to_measured_boundary():
    from adsync.align.refine import refine_warp_boundaries
    from scipy.interpolate import PchipInterpolator
    from adsync.models import WarpPath

    sr = 8000
    video = np.random.default_rng(29).normal(0, .1, 110 * sr).astype(np.float32)
    ad = np.r_[video[:60 * sr], video[68 * sr:108 * sr]]
    path = [_point(t, 0 if t < 48 else 8, .05 if 40 <= t < 70 else .8)
            for t in range(0, 100, 2)]
    wp = WarpPath(points=path, anchor_points=path, n_segments=2)
    fns = [PchipInterpolator([0, 48], [0, 48]), PchipInterpolator([48, 100], [56, 108])]
    ranges = [(0, 48), (48, 100)]
    diagnostics = refine_warp_boundaries(fns, ranges, wp, video, ad, sr)
    assert ranges[0][1] == pytest.approx(60, abs=.6)
    assert ranges[1][0] == ranges[0][1]
    assert diagnostics[0]["status"] == "refined"
    assert _at(fns, ranges, 56) == pytest.approx(56)
    assert _at(fns, ranges, 65) == pytest.approx(73)
    assert next(p for p in wp.points if p.source_time == 56).target_time == pytest.approx(56)
