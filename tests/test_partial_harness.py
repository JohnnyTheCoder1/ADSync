"""Independent ground-truth checks for the public alignment scorer."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


TOOL = Path(__file__).resolve().parents[1] / "tools" / "partial_alignment_harness.py"


def harness():
    assert TOOL.is_file(), "The public partial alignment harness is missing"
    spec = importlib.util.spec_from_file_location("partial_alignment_harness", TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def linear_map(start, end, offset=0.0):
    return {"x": [start, end], "c": [[1.0], [start + offset]]}


def paired_truth():
    return [
        {"name": "prefix", "source_start": 0, "source_end": 40, "target_starts": [0]},
        {"name": "short_island", "source_start": 40, "source_end": 46, "target_starts": [44]},
        {"name": "suffix", "source_start": 50, "source_end": 100, "target_starts": [50]},
    ]


def test_missing_short_island_is_visible_even_when_aggregate_error_is_zero():
    result = harness().score_alignment(
        {"segment_ranges": [[0, 40], [50, 100]],
         "fitted_pchip": [linear_map(0, 40), linear_map(50, 100)]},
        paired_truth(), [[46, 50]],
    )
    island = result["fragments"][1]
    assert island["sample_count"] == 40
    assert island["correct_placement_coverage"] == 0
    assert island["unmeasured_coverage"] == 1
    assert island["error_p95_ms"] is None
    assert result["overall"]["error_p95_ms"] == 0
    assert result["overall"]["correct_placement_coverage"] == pytest.approx(86 / 90)


def test_wrong_island_is_not_hidden_by_perfect_long_fragments():
    result = harness().score_alignment(
        {"segment_ranges": [[0, 40], [40, 46], [50, 100]],
         "fitted_pchip": [linear_map(0, 40), linear_map(40, 46), linear_map(50, 100)]},
        paired_truth(), [[46, 50]],
    )
    assert result["fragments"][1]["wrong_but_placed_coverage"] == 1
    assert result["fragments"][1]["error_p95_ms"] == pytest.approx(4000)
    assert result["overall"]["error_p95_ms"] == 0


def test_forced_identity_counts_source_gap_as_false_placement():
    result = harness().score_alignment(
        {"segment_ranges": [[0, 100]], "fitted_pchip": [linear_map(0, 100)]},
        paired_truth(), [[46, 50]],
    )
    assert result["source_gaps"][0]["sample_count"] == 20
    assert result["source_gaps"][0]["false_placement_coverage"] == 1
    assert result["gap_overall"]["false_placement_coverage"] == 1


def test_replay_uses_polynomial_local_origin_and_all_valid_repeat_locations():
    result = harness().score_alignment(
        {"segment_ranges": [[10, 20]], "fitted_pchip": [linear_map(10, 20, 30)]},
        [{"name": "repeat", "source_start": 10, "source_end": 20,
          "target_starts": [10, 40]}], [],
    )
    assert result["overall"]["correct_placement_coverage"] == 1
    assert result["overall"]["error_max_ms"] == 0


def test_overlapping_maps_are_conflicting_not_cherry_picked():
    result = harness().score_alignment(
        {"segment_ranges": [[0, 10], [0, 10]],
         "fitted_pchip": [linear_map(0, 10), linear_map(0, 10, 5)]},
        [{"name": "clean", "source_start": 0, "source_end": 10, "target_starts": [0]}], [],
    )
    assert result["overall"]["conflicting_placement_coverage"] == 1
    assert result["overall"]["wrong_but_placed_coverage"] == 1
    assert result["overall"]["correct_placement_coverage"] == 0


def test_too_short_fragment_is_explicitly_unscored_and_margin_is_bounded():
    module = harness()
    result = module.score_alignment({}, [
        {"name": "tiny", "source_start": 0, "source_end": 1, "target_starts": [0]},
        {"name": "brief", "source_start": 2, "source_end": 5, "target_starts": [2]},
    ], [])
    assert result["fragments"][0]["scoring_status"] == "no_interior_samples"
    assert result["fragments"][0]["unmeasured_coverage"] is None
    assert result["fragments"][1]["sample_count"] == 10
    with pytest.raises(ValueError, match="margin"):
        module.score_alignment({}, [], [], boundary_margin_sec=1.1)


def test_incomplete_map_metadata_is_not_silently_truncated():
    with pytest.raises(ValueError, match="range"):
        harness().score_alignment({"segment_ranges": [[0, 10]], "fitted_pchip": []}, [], [])


def test_json_normalization_never_emits_nonfinite_numbers():
    normalized = harness().json_safe({"value": np.float64(np.nan),
                                      "array": np.array([np.inf, -np.inf, .5])})
    assert json.loads(json.dumps(normalized, allow_nan=False)) == {
        "value": None, "array": [None, None, .5],
    }


def test_fixture_provenance_and_seed_are_reproducible():
    module = harness()
    first = module.make_scenarios(seed=7)
    second = module.make_scenarios(seed=7)
    paired = next(case for case in first if case.name == "paired_edit")
    assert paired.sample_rate == 16000
    assert len(paired.source) == 1_600_000
    np.testing.assert_array_equal(paired.source[40 * 16000:46 * 16000],
                                  paired.target[44 * 16000:50 * 16000])
    np.testing.assert_array_equal(paired.source[50 * 16000:], paired.target[50 * 16000:])
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a.source, b.source)
        np.testing.assert_array_equal(a.target, b.target)
    assert paired.fragments[1]["target_starts"] == [44.0]
    assert paired.source_gaps == [[46.0, 50.0]]


def test_ambiguity_coverage_uses_union_of_reported_source_ranges():
    result = harness().score_alignment(
        {"partial_alignment": {"ambiguous_ranges": [
            {"start_sec": 0, "end_sec": 6}, {"start_sec": 4, "end_sec": 10},
        ]}},
        [{"name": "repeat", "source_start": 0, "source_end": 10, "target_starts": [0, 20]}], [],
    )
    assert result["overall"]["reported_ambiguous_coverage"] == 1
    assert result["fragments"][0]["reported_ambiguous_coverage"] == 1


def test_benchmark_verdict_requires_every_short_fragment_and_explicit_ambiguity():
    module = harness()
    scores = module.score_alignment(
        {"segment_ranges": [[0, 40], [50, 100]],
         "fitted_pchip": [linear_map(0, 40), linear_map(50, 100)]},
        paired_truth(), [[46, 50]],
    )
    assert module.evaluate_scores(scores, ambiguity_expected=False)["status"] == "fail"
    repeat = module.score_alignment(
        {"segment_ranges": [[0, 10]], "fitted_pchip": [linear_map(0, 10)]},
        [{"name": "repeat", "source_start": 0, "source_end": 10, "target_starts": [0, 20]}], [],
    )
    assert module.evaluate_scores(repeat, ambiguity_expected=True)["status"] == "fail"
    assert module.evaluate_scores(repeat, ambiguity_expected=False)["status"] == "pass"


def test_generate_only_writes_strict_manifest_and_refuses_checkout_media(tmp_path):
    module = harness()
    output = tmp_path / "fixture"
    assert module.main(["--workdir", str(output), "--scenarios", "paired_edit", "--generate-only"]) == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["scenarios"][0]["source_duration_sec"] == 100
    assert (output / "audio" / "paired_edit_source.wav").is_file()
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".git").write_text("gitdir: elsewhere", encoding="utf-8")
    with pytest.raises(SystemExit):
        module.main(["--workdir", str(checkout / "artifacts"), "--generate-only"])
    assert not (checkout / "artifacts").exists()


def test_piecewise_fallback_is_scored_instead_of_becoming_an_empty_map():
    module = harness()
    debug = module.timing_for_scoring({
        "mode": "piecewise", "timing_debug": {},
        "segments": [{"src_start": 2, "src_end": 8, "dst_start": 12,
                      "dst_end": 24, "stretch": 2}],
    })
    scores = module.score_alignment(debug, [
        {"name": "double_speed", "source_start": 2, "source_end": 8,
         "target_starts": [12], "rate": 2},
    ], [[2, 8]])
    assert scores["overall"]["correct_placement_coverage"] == 1
    assert scores["gap_overall"]["false_placement_coverage"] == 1
    assert debug["scorer_map_source"] == "segment_map_fallback"
