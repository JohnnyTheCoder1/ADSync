"""Sparse partial paths retain measured islands and expose competing paths."""

import itertools
import math

import numpy as np
import pytest

from adsync.models import CandidateWindow, OffsetCandidate


def window(time, offsets, score=.9):
    return CandidateWindow(source_center=time, speech_score=.5, energy=1,
        candidates=[OffsetCandidate(offset_sec=offset, score=value,
            peak_sharpness=10, peak_ratio=1) for offset, value in
            ([(o, score) for o in offsets] if offsets and not isinstance(offsets[0], tuple) else offsets)])


def decode(windows, duration=100, video_duration=110, **kwargs):
    from adsync.align.partial_decode import decode_partial_path
    return decode_partial_path(windows, duration, video_duration, **kwargs)


def test_paired_edits_retain_short_real_island_and_skip_source_only_content():
    windows = [window(t, [] if 46 <= t < 50 else [4 if 40 <= t < 46 else 0], .99)
               for t in range(2, 98, 2)]
    result = decode(windows, window_sec=2, step_sec=2)
    selected = [result.nodes[i] for i in result.selected]
    island = [n for n in selected if 40 <= n.source_time < 46]
    assert [n.source_time for n in island] == [40, 42, 44]
    assert [n.target_time - n.source_time for n in island] == [4, 4, 4]
    assert not any(46 <= n.source_time < 50 for n in selected)
    assert len(result.runs) == 3


def test_single_peak_cannot_be_a_match_run():
    result = decode([window(10, [0]), window(11, [0]), window(12, [0]),
                     window(20, [50]), window(30, [0]), window(31, [0]), window(32, [0])])
    assert 20 not in [result.nodes[i].source_time for i in result.selected]


def test_equal_complete_alternatives_have_zero_margin():
    result = decode([window(t, [0, 30]) for t in range(1, 9)], duration=10, video_duration=40)
    assert result.selected
    assert all(result.window_margins[i] == pytest.approx(0, abs=1e-10) for i in range(8))


@pytest.mark.parametrize("separation", [0., .025])
def test_equivalent_measurements_do_not_manufacture_placement_ambiguity(separation):
    result = decode([window(t, [2, 2 + separation]) for t in range(1, 9)], duration=10, video_duration=15)
    assert all(margin > .1 for margin in result.window_margins)


def test_target_time_cannot_reverse_to_take_a_stronger_repeat():
    windows = [window(t, [30], .95) for t in range(1, 10)]
    windows += [window(t, [(0, 1), (30, .8)]) for t in range(10, 16)]
    result = decode(windows)
    selected = [result.nodes[i] for i in result.selected]
    assert np.all(np.diff([n.target_time for n in selected]) > 0)
    assert all(n.target_time - n.source_time == 30 for n in selected)


def _enumerated_paths(windows, duration, video_duration):
    """Independent enumeration of the documented three-observation objective."""
    paths = []
    for choices in itertools.product(*[range(-1, len(w.candidates)) for w in windows]):
        points = [(i, windows[i].source_center, windows[i].source_center + windows[i].candidates[c].offset_sec,
                   windows[i].candidates[c].score) for i, c in enumerate(choices) if c >= 0]
        if not points:
            paths.append((-.01 * (duration + video_duration), choices))
            continue
        score = sum(p[3] for p in points)
        score -= .01 * (max(0, points[0][1] - 1) + max(0, points[0][2] - 1))
        score -= .01 * (max(0, duration - points[-1][1] - 1) + max(0, video_duration - points[-1][2] - 1))
        run = 1
        valid = True
        for a, b in zip(points, points[1:]):
            dt, dv = b[1] - a[1], b[2] - a[2]
            if dv <= 0:
                valid = False
                break
            if dt <= 1.5 and abs(dv - dt) <= .01 * dt + 1e-9:
                run += 1
            else:
                if run < 3:
                    valid = False
                    break
                score -= .75 + .01 * (max(0, dt - 2) + max(0, dv - 2))
                run = 1
        if valid and run >= 3:
            paths.append((score, choices))
    return paths


@pytest.mark.parametrize("seed", range(5))
def test_exact_objective_and_window_exclusion_margins_match_exhaustive_paths(seed):
    rng = np.random.default_rng(seed)
    windows = [window(t, [(0, float(rng.uniform(.4, .95))), (4, float(rng.uniform(.4, .95)))])
               for t in range(1, 8)]
    result = decode(windows, duration=8, video_duration=12)
    paths = _enumerated_paths(windows, 8, 12)
    best = max(score for score, _ in paths)
    assert result.path_score == pytest.approx(best, abs=1e-10)
    selected = {result.nodes[n].window_index: result.nodes[n].candidate_index for n in result.selected}
    for wi, ci in selected.items():
        alternative = max(score for score, path in paths if path[wi] != ci)
        assert result.window_margins[wi] == pytest.approx(best - alternative, abs=1e-10)


def test_empty_low_score_and_out_of_target_candidates_produce_no_map():
    assert not decode([]).selected
    assert not decode([window(t, [0], .1) for t in range(1, 9)]).selected
    assert not decode([window(t, [-100]) for t in range(1, 9)]).selected


@pytest.mark.parametrize("parameter,value", [("ad_duration", 0), ("video_duration", math.inf),
    ("window_sec", 0), ("step_sec", math.nan), ("max_stretch", -.01)])
def test_invalid_numeric_inputs_raise(parameter, value):
    from adsync.align.partial_decode import decode_partial_path
    args = dict(ad_duration=10, video_duration=10, window_sec=2, step_sec=1, max_stretch=.01)
    args[parameter] = value
    with pytest.raises(ValueError):
        decode_partial_path([], **args)


def test_nonfinite_candidates_and_unordered_windows_raise():
    with pytest.raises(ValueError):
        decode([window(1, [float("nan")])])
    with pytest.raises(ValueError):
        decode([window(2, [0]), window(1, [0])])


def test_explicit_candidate_limit_fails_instead_of_silently_pruning():
    with pytest.raises(ValueError, match="candidate"):
        decode([window(t, [0]) for t in range(1, 8)], max_candidates=6)
