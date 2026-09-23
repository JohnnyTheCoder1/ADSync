from types import SimpleNamespace

import numpy as np

from adsync.models import CandidateWindow, OffsetCandidate


def _window(time, candidates=()):
    return CandidateWindow(source_center=time, candidates=list(candidates), energy=1., speech_score=.5)


def _fingerprints(source, offsets):
    source = np.asarray(source, dtype=float)
    return SimpleNamespace(raw_match_t_ad=source, raw_match_t_vid=source + offsets)


def test_raw_local_matches_recover_evidence_discarded_by_long_span_vetting():
    from adsync.align.partial_evidence import augment_partial_evidence
    lattice = [_window(43.)]
    fp = _fingerprints(np.arange(42.1, 44., .1), 4.)
    assert augment_partial_evidence(lattice, fp) == 1
    assert lattice[0].candidates[0].offset_sec == 4.
    assert lattice[0].candidates[0].source == "fingerprint"


def test_landmarks_do_not_compete_with_a_measured_short_island():
    from adsync.align.partial_evidence import augment_partial_evidence
    candidate = OffsetCandidate(offset_sec=4., score=.5, peak_sharpness=10., peak_ratio=2.)
    lattice = [_window(43., [candidate])]
    fp = _fingerprints(np.arange(42.1, 44., .1), 0.)
    assert augment_partial_evidence(lattice, fp) == 0
    assert lattice[0].candidates == [candidate]


def test_evidence_outside_a_window_cannot_fill_an_unmatched_insert():
    from adsync.align.partial_evidence import augment_partial_evidence
    lattice = [_window(48.)]
    fp = _fingerprints(np.r_[np.arange(44., 46., .1), np.arange(50., 52., .1)], 0.)
    assert augment_partial_evidence(lattice, fp) == 0


def test_duplicate_hash_votes_are_not_independent_temporal_support():
    from adsync.align.partial_evidence import augment_partial_evidence
    lattice = [_window(5.)]
    fp = _fingerprints(np.full(100, 4.5), 2.)
    assert augment_partial_evidence(lattice, fp) == 0


def test_matches_in_only_one_half_do_not_claim_the_whole_window():
    from adsync.align.partial_evidence import augment_partial_evidence
    lattice = [_window(5.)]
    fp = _fingerprints(np.arange(4.05, 4.95, .05), 2.)
    assert augment_partial_evidence(lattice, fp) == 0


def test_repeated_landmark_alternatives_are_preserved():
    from adsync.align.partial_evidence import augment_partial_evidence
    times = np.arange(4.1, 6., .1)
    fp = _fingerprints(np.r_[times, times], np.r_[np.zeros(len(times)), np.full(len(times), 30.)])
    lattice = [_window(5.)]
    assert augment_partial_evidence(lattice, fp) == 1
    assert sorted(c.offset_sec for c in lattice[0].candidates) == [0., 30.]
