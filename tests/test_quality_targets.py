"""Short off-time matches must survive as QC targets after alignment filtering."""

from pathlib import Path
import sys

import numpy as np
import pytest


@pytest.fixture(scope="module")
def real_pair():
    assets = Path(__file__).resolve().parents[1] / "harness_assets"
    if not all((assets / name).exists() for name in ("excerpt.flac", "ad_base.wav")):
        pytest.skip("Optional real-media accuracy harness assets are not installed")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from qc_calibration import load_excerpt
    from adsync.align.fingerprint import _landmarks

    rate = 16000
    main = load_excerpt(assets / "excerpt.flac", 0, 60, rate)
    ad = load_excerpt(assets / "ad_base.wav", 0, 60, rate)
    return main, ad, rate, _landmarks(main, rate)


def test_real_two_second_shift_produces_a_narrow_independently_verifiable_target(real_pair):
    from adsync.align.fingerprint import _landmarks, _match, _offset_spans
    from adsync.quality import _waveform_window
    from adsync.quality_targets import raw_offset_windows

    main, ad, rate, (main_hashes, main_times) = real_pair
    wrong = ad.copy()
    wrong[42 * rate:44 * rate] = ad[round(41.2 * rate):round(43.2 * rate)]
    hashes, times = _landmarks(wrong, rate)
    raw_ad, raw_main = _match(main_hashes, main_times, hashes, times)
    retained = _offset_spans(raw_ad, raw_main, 60)
    # The old alignment inlier filter removes every real -.8-second vote here.
    assert not np.any((retained.match_t_ad >= 42) & (retained.match_t_ad < 44))
    targets = raw_offset_windows(raw_ad, raw_main, 60)
    local = [w for w in targets if w["start_sec"] < 44 and w["end_sec"] > 42]
    assert local
    assert all(w["end_sec"] - w["start_sec"] <= 4 for w in local)
    measurements = [_waveform_window(main, wrong, rate, w["start_sec"], w["end_sec"]) for w in local]
    assert any(m["status"] == "strong" and m["lag_sec"] == pytest.approx(-.787, abs=.04)
               for m in measurements)


def test_real_aligned_description_does_not_gain_spurious_error_targets(real_pair):
    from adsync.align.fingerprint import _landmarks, _match
    from adsync.quality_targets import raw_offset_windows

    _, ad, rate, (main_hashes, main_times) = real_pair
    hashes, times = _landmarks(ad, rate)
    raw_ad, raw_main = _match(main_hashes, main_times, hashes, times)
    assert raw_offset_windows(raw_ad, raw_main, 60) == []


@pytest.mark.parametrize("delay", [.6, 1.0, 1.5, 3.0])
def test_four_second_errors_still_get_local_targets_and_independent_measurements(delay):
    from adsync.quality import _waveform_window
    from adsync.quality_targets import raw_offset_windows

    rate = 4000
    main = np.random.default_rng(902).normal(scale=.1, size=240 * rate)
    ad = main.copy()
    ad[103 * rate:107 * rate] = main[round((103 - delay) * rate):round((107 - delay) * rate)]
    times = np.arange(.1, 240, .1)
    offsets = np.where((times >= 103) & (times < 107), -delay, 0)
    targets = raw_offset_windows(times, times + offsets, 240)
    assert targets
    measurements = [_waveform_window(main, ad, rate, row["start_sec"], row["end_sec"], search_radius_sec=4)
                    for row in targets]
    assert any(m["status"] == "strong" and abs(m["lag_sec"]) > .15 for m in measurements)


def test_naturally_sparse_zero_offset_support_does_not_expand_the_probe_queue():
    from adsync.quality_targets import raw_offset_windows

    times = np.arange(.1, 120, 1.8)
    assert raw_offset_windows(times, times.copy(), 120) == []


def test_scattered_coincidences_and_repeated_votes_at_one_time_are_not_targets():
    from adsync.quality_targets import raw_offset_windows

    times = np.arange(.1, 120, .1)
    offsets = np.zeros(len(times))
    offsets[::17] = 9.0
    times = np.r_[times, np.full(80, 43.0)]
    offsets = np.r_[offsets, np.full(80, -1.0)]
    assert raw_offset_windows(times, times + offsets, 120) == []


def test_a_long_coherent_error_needs_few_bounded_probes_not_a_full_sliding_scan():
    from adsync.quality_targets import raw_offset_windows

    times = np.arange(.1, 600, .1)
    targets = raw_offset_windows(times, times + 20, 600)
    assert 1 <= len(targets) <= 3
    assert all(w["end_sec"] - w["start_sec"] <= 4 for w in targets)
    assert targets[0]["start_sec"] < 5
    assert targets[-1]["end_sec"] > 595


def test_opposite_offsets_are_not_merged_and_pairs_are_sorted_internally():
    from adsync.quality_targets import raw_offset_windows

    times = np.arange(0, 12, .1)
    offsets = np.where(times < 6, -.8, .8)
    targets = raw_offset_windows(times[::-1], (times + offsets)[::-1], 12)
    assert {round(w["suspected_offset_sec"], 1) for w in targets} == {-.8, .8}
    assert all(w["reasons"] == ["raw_nonzero_fingerprint_cluster"] for w in targets)
