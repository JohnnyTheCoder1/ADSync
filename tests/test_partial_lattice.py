"""Real correlation tests for matches at recording boundaries."""

import numpy as np
import pytest

from adsync.align.candidate_lattice import build_candidate_lattice
from adsync.models import FeatureBundle


def features(duration, sr=4000):
    frames = max(1, round(duration * sr / 128))
    return FeatureBundle(sr=sr, hop_length=128, duration=duration,
                         rms=np.zeros(frames), onset=np.zeros(frames),
                         mel=np.zeros((8, frames)), mfcc=np.zeros((2, frames)))


def repeated_lattice(**kwargs):
    rng = np.random.default_rng(17)
    source = rng.normal(0, .1, 4 * 4000)
    target = np.concatenate([source, rng.normal(0, .1, 4000), source])
    return build_candidate_lattice(
        features(9), features(4), y_vid=target, y_ad=source, audio_sr=4000,
        window_sec=1, step_sec=1, search_radius_sec=10, min_score=.8,
        max_candidates=5, multiband=False, threads=1, **kwargs,
    )


def test_boundary_search_preserves_both_complete_repeated_placements():
    lattice = repeated_lattice(include_boundary_matches=True)
    assert [window.source_center for window in lattice] == [.5, 1.5, 2.5, 3.5]
    for window in (lattice[0], lattice[-1]):
        assert len(window.candidates) == 2
        assert sorted(c.offset_sec for c in window.candidates) == pytest.approx([0, 5], abs=1e-6)
        assert min(c.score for c in window.candidates) > .99


def test_boundary_search_measures_exactly_one_window_and_non_grid_tail():
    rng = np.random.default_rng(21)
    for duration, expected in ((1, [.5]), (2.25, [.5, 1.5, 1.75])):
        y = rng.normal(0, .1, round(duration * 4000))
        bundle = features(duration)
        lattice = build_candidate_lattice(
            bundle, bundle, y_vid=y, y_ad=y, audio_sr=4000,
            window_sec=1, step_sec=1, multiband=False, threads=1,
            include_boundary_matches=True,
        )
        assert [window.source_center for window in lattice] == expected
        for window in lattice:
            assert any(abs(c.offset_sec) < 1e-6 and c.score > .99 for c in window.candidates)


def test_existing_window_positions_stay_unchanged_without_opt_in():
    lattice = repeated_lattice()
    assert [window.source_center for window in lattice] == [.5, 1.5, 2.5]
