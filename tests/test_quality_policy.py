"""Decision policy separates measured timing, bounded gaps, and real uncertainty."""

from copy import deepcopy

import numpy as np
import pytest


def _measured(start, end, lag=0.01):
    return {"start_sec": start, "end_sec": end, "center_sec": (start + end) / 2,
            "status": "pass", "reasons": ["regular"],
            "waveform": {"status": "strong", "lag_sec": lag, "method": "waveform"}}


def _gap(start, end, *, before_lag=0.01, after_lag=0.012):
    return {"start_sec": start, "end_sec": end, "center_sec": (start + end) / 2,
            "status": "supported_context", "reasons": ["short_support_gap"],
            "waveform": {"status": "weak", "lag_sec": None},
            "context": {
                "before": {"status": "strong", "lag_sec": before_lag, "method": "waveform",
                           "sample_start_sec": start - 8, "sample_end_sec": start},
                "after": {"status": "strong", "lag_sec": after_lag, "method": "waveform",
                          "sample_start_sec": end, "sample_end_sec": end + 8},
            }}


def _evidence(*windows):
    return {"duration_sec": 180, "tolerance_sec": 0.15, "status": "review",
            "identity": {"status": "pass", "coverage": 0.90}, "failures": [],
            "review_reasons": ["More than three percent of the rendered track relies on bounded context instead of direct timing evidence"],
            "windows": [_measured(2, 8), _measured(85, 91), _measured(166, 174), *windows]}


def test_many_separated_short_gaps_are_not_rejected_for_narration_frequency():
    from adsync.quality_policy import evaluate_quality_policy

    evidence = _evidence(*[_gap(start, start + 2) for start in (20, 50, 80, 110, 140)])
    original = deepcopy(evidence)
    result = evaluate_quality_policy(evidence)
    assert result["status"] == "pass", result["review_reasons"]
    assert result["supported_context_coverage"] == pytest.approx(10 / 180)
    assert result["policy_summary"]["bounded_unmeasured_sec"] == pytest.approx(10)
    assert len(result["policy_summary"]["bounded_unmeasured_ranges"]) == 5
    assert evidence == original


def test_correctly_timed_audio_with_intermittent_narration_is_a_positive_fixture():
    from adsync.align.fingerprint import FingerprintResult, FingerprintSpan
    from adsync.quality import verify_audio_sync
    from adsync.quality_policy import evaluate_quality_policy

    sr = 1000
    rng = np.random.default_rng(342)
    main = rng.normal(scale=.1, size=180 * sr)
    ad = main.copy()
    times = np.arange(.1, 180, .05)
    for start in (20, 50, 80, 110, 140):
        ad[start * sr:(start + 2) * sr] += rng.normal(scale=2, size=2 * sr)
        times = times[(times < start) | (times >= start + 2)]
    fp = FingerprintResult(spans=[FingerprintSpan(0, 180, 0, len(times))], n_matches=len(times),
                           match_t_ad=times, match_t_vid=times.copy())
    evidence = verify_audio_sync(main, ad, sr, fingerprint=fp)
    result = evaluate_quality_policy(evidence)
    assert result["status"] == "pass", result["review_reasons"]
    assert result["policy_summary"]["bounded_unmeasured_sec"] > 5


@pytest.mark.parametrize("lag", [4.0, 20.0])
def test_coherent_local_shift_is_a_hard_veto_despite_healthy_global_evidence(lag):
    from adsync.quality_policy import evaluate_quality_policy

    wrong = _measured(101, 105, lag)
    wrong["status"] = "fail"
    evidence = _evidence(wrong, _gap(40, 42))
    result = evaluate_quality_policy(evidence)
    assert result["status"] == "fail"
    assert result["failures"]


def test_wrong_episode_identity_cannot_pass_with_an_aligned_intro():
    from adsync.quality_policy import evaluate_quality_policy

    evidence = _evidence()
    evidence["identity"] = {"status": "fail", "coverage": .03, "reasons": ["Sparse intro-only identity"]}
    assert evaluate_quality_policy(evidence)["status"] == "fail"


@pytest.mark.parametrize("start,end", [(50, 62), (0, 4), (177, 180)])
def test_long_or_unbracketed_boundary_uncertainty_stays_for_review(start, end):
    from adsync.quality_policy import evaluate_quality_policy

    result = evaluate_quality_policy(_evidence(_gap(start, end)))
    assert result["status"] == "review"
    assert [start, end] in result["policy_summary"]["unresolved_ranges"]


def test_unknown_opening_is_not_approved_by_episode_average_confidence():
    from adsync.quality_policy import evaluate_quality_policy

    opening = {"start_sec": 0, "end_sec": 12, "status": "review", "reasons": ["sparse_anchors"],
               "waveform": {"status": "weak", "lag_sec": None}}
    evidence = _evidence(opening)
    evidence["confidence"] = .99
    result = evaluate_quality_policy(evidence)
    assert result["status"] == "review"
    assert [0, 12] in result["policy_summary"]["unresolved_ranges"]


def test_overlapping_short_windows_cannot_chain_across_a_long_unmeasured_span():
    from adsync.quality_policy import evaluate_quality_policy

    result = evaluate_quality_policy(_evidence(_gap(50, 54), _gap(53, 57), _gap(56, 60)))
    assert result["status"] == "review"
    assert result["policy_summary"]["unresolved_ranges"] == [[50.0, 60.0]]


def test_nearby_unmeasured_gaps_cannot_be_reused_as_measured_flanks():
    from adsync.quality_policy import evaluate_quality_policy

    result = evaluate_quality_policy(_evidence(_gap(50, 52), _gap(55, 57)))
    assert result["status"] == "review"


@pytest.mark.parametrize("change", ["disagreement", "weak", "conflict", "cut"])
def test_context_requires_consistent_independent_flanks_and_no_nearby_edit(change):
    from adsync.quality_policy import evaluate_quality_policy

    gap = _gap(50, 52)
    cuts = []
    if change == "disagreement":
        gap["context"]["after"]["lag_sec"] = .10
    elif change == "weak":
        gap["context"]["before"]["status"] = "weak"
    elif change == "conflict":
        gap["waveform"]["conflict"] = {"reason": "Raw and envelope evidence disagree"}
    else:
        cuts = [51]
    assert evaluate_quality_policy(_evidence(gap), cut_times=cuts)["status"] == "review"


def test_generic_decoder_failure_and_missing_independent_third_are_not_dropped():
    from adsync.quality_policy import evaluate_quality_policy

    evidence = _evidence()
    evidence["failures"] = ["Rendered audio stream durations differ by more than one second"]
    assert evaluate_quality_policy(evidence)["status"] == "fail"
    evidence["failures"] = []
    evidence["windows"][-1]["waveform"]["status"] = "weak"
    assert evaluate_quality_policy(evidence)["status"] == "review"


def _quiet_opening(state="near_silent"):
    return {"start_sec": 0, "end_sec": 4, "status": "review", "reasons": ["sparse_anchors"],
            "waveform": {"status": "weak", "lag_sec": None,
                         "signal": {"state": state,
                                    "main": {"ac_rms": 1e-7, "peak_ac": 5e-7, "noise_floor_rms": 1e-6},
                                    "ad": {"ac_rms": 2e-7, "peak_ac": 8e-7, "noise_floor_rms": 1e-6}}}}


def test_credible_two_track_near_silence_is_recorded_without_inventing_a_measurement():
    from adsync.quality_policy import evaluate_quality_policy

    result = evaluate_quality_policy(_evidence(_quiet_opening()))
    assert result["status"] == "pass"
    assert result["windows"][-1]["status"] == "near_silent"
    assert result["windows"][-1]["policy_assessment"]["directly_measured"] is False
    assert result["policy_summary"]["quiet_unmeasured_ranges"] == [[0.0, 4.0]]


@pytest.mark.parametrize("state", ["main_quiet", "ad_quiet", "active"])
def test_one_sided_quiet_or_audible_opening_remains_unresolved(state):
    from adsync.quality_policy import evaluate_quality_policy

    result = evaluate_quality_policy(_evidence(_quiet_opening(state)))
    assert result["status"] == "review"
    assert [0.0, 4.0] in result["policy_summary"]["unresolved_ranges"]


def test_a_measured_nonzero_offset_cannot_be_hidden_by_a_quiet_label():
    from adsync.quality_policy import evaluate_quality_policy

    window = _quiet_opening()
    window["waveform"].update(status="strong", lag_sec=.36, method="waveform")
    result = evaluate_quality_policy(_evidence(window))
    assert result["status"] == "fail"
    assert result["windows"][-1]["status"] == "fail"
