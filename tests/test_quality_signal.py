"""Signal evidence must accept aligned overlays and reject coherent faults."""

import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt


def narration_pair(sr=8000, seconds=24, seed=802):
    rng = np.random.default_rng(seed)
    main = rng.normal(scale=.02, size=sr * seconds)
    voice = sosfiltfilt(butter(4, (200, 3000), btype="bandpass", fs=sr, output="sos"), rng.normal(size=len(main)))
    syllables = np.interp(np.arange(len(main)) / sr, np.arange(0, seconds, .1), rng.uniform(.1, 1, seconds * 10))
    ad = main + voice * syllables * .5
    return main, ad


@pytest.mark.parametrize("delay", [0.0, .6, 1.0, 1.5, 3.0])
def test_weak_shared_waveform_survives_loud_narration_with_correct_lag(delay):
    from adsync.quality import _waveform_window

    sr = 8000
    main, ad = narration_pair(sr)
    if delay:
        offset = round(delay * sr)
        ad = np.r_[np.zeros(offset), ad[:-offset]]
    row = _waveform_window(main, ad, sr, 8, 16, search_radius_sec=4)
    assert row["status"] == "strong", row
    assert row["lag_sec"] == pytest.approx(-delay, abs=.015)
    assert row["template_start_sec"] == 8
    assert row["template_end_sec"] == 16


def test_unrelated_narrated_recordings_cannot_pass_on_low_raw_scores():
    from adsync.quality import _waveform_window

    for seed in range(830, 850):
        main, _ = narration_pair(seed=seed)
        _, ad = narration_pair(seed=seed + 100)
        row = _waveform_window(main, ad, 8000, 8, 16, search_radius_sec=4)
        assert row["status"] == "weak", (seed, row)


def test_level_evidence_distinguishes_program_relative_noise_from_quiet_music():
    from adsync.quality_signal import signal_levels

    sr = 4000
    rng = np.random.default_rng(871)
    programme = rng.normal(scale=.05, size=40 * sr)
    programme[-2 * sr:] = 1e-4 + rng.normal(scale=3e-6, size=2 * sr)
    quiet_music = rng.normal(scale=3e-5, size=40 * sr)
    near = signal_levels(programme, programme.copy(), sr, 39, 40)
    quiet = signal_levels(quiet_music, quiet_music.copy(), sr, 39, 40)
    assert near["state"] == "near_silent"
    assert near["main"]["relative_db"] < -60
    assert quiet["state"] == "active"


def test_dc_offset_does_not_become_timing_evidence_in_silent_tail():
    from adsync.quality import _waveform_window

    sr = 4000
    rng = np.random.default_rng(872)
    main = rng.normal(scale=.05, size=40 * sr)
    ad = main.copy()
    main[-2 * sr:] = .01 + rng.normal(scale=3e-6, size=2 * sr)
    ad[-2 * sr:] = -.025 + rng.normal(scale=4e-6, size=2 * sr)
    row = _waveform_window(main, ad, sr, 39, 40)
    assert row["status"] == "weak"
    assert row["signal"]["state"] == "near_silent"


def test_quiet_shared_audio_is_measured_when_its_own_program_level_is_quiet():
    from adsync.quality import _waveform_window

    sr = 4000
    main = np.random.default_rng(873).normal(scale=3e-5, size=40 * sr)
    ad = np.r_[np.zeros(2400), main[:-2400]]
    row = _waveform_window(main, ad, sr, 20, 28)
    assert row["signal"]["state"] == "active"
    assert row["status"] == "strong"
    assert row["lag_sec"] == pytest.approx(-.6, abs=.002)


def test_band_consensus_cannot_chain_disagreeing_lags_through_a_middle_band():
    from adsync.quality import _agreeing_group

    rows = [{"lag_sec": value, "score": .8} for value in (0, .014, .028)]
    group, _ = _agreeing_group(rows, max_disagreement_sec=.015)
    assert max(row["lag_sec"] for row in group) - min(row["lag_sec"] for row in group) <= .015


def test_real_remixed_bed_corroborates_a_unique_peak_across_frequency_bands():
    from pathlib import Path
    from adsync.features.load import load_wav
    from adsync.quality import _waveform_window

    assets = Path(__file__).resolve().parent.parent / "harness_assets"
    if not (assets / "excerpt.flac").is_file() or not (assets / "ad_base.wav").is_file():
        pytest.skip("Optional real calibration assets are not installed")
    main, sr = load_wav(assets / "excerpt.flac", sr=16000)
    ad, _ = load_wav(assets / "ad_base.wav", sr=16000)
    row = _waveform_window(main, ad, sr, 321.316, 332.428)
    assert row["status"] == "strong"
    assert .005 < row["lag_sec"] < .035
    assert row["method"] == "cross_band_waveform"


def test_short_real_error_is_not_averaged_into_a_good_majority_lag():
    from pathlib import Path
    from adsync.features.load import load_wav
    from adsync.quality import _waveform_window

    assets = Path(__file__).resolve().parent.parent / "harness_assets"
    if not (assets / "excerpt.flac").is_file() or not (assets / "ad_base.wav").is_file():
        pytest.skip("Optional real calibration assets are not installed")
    main, sr = load_wav(assets / "excerpt.flac", sr=16000)
    ad, _ = load_wav(assets / "ad_base.wav", sr=16000)
    original = _waveform_window(main, ad, sr, 42, 46)
    assert original["status"] == "strong"
    damaged = ad.copy()
    damaged[42 * sr:44 * sr] = ad[round(41.2 * sr):round(43.2 * sr)]
    row = _waveform_window(main, damaged, sr, 42, 46)
    assert row["status"] == "weak"
    assert row["conflict"]["kind"] == "mixed_raw_lags"
    assert any(b["lag_sec"] < -.7 for b in row["conflict"]["raw_bands"])


def test_weak_cross_band_peaks_with_fourteen_ms_disagreement_stay_unmeasured():
    from pathlib import Path
    from adsync.features.load import load_wav
    from adsync.quality import _waveform_window

    assets = Path(__file__).resolve().parent.parent / "harness_assets"
    if not (assets / "excerpt.flac").is_file() or not (assets / "ad_base.wav").is_file():
        pytest.skip("Optional real calibration assets are not installed")
    main, sr = load_wav(assets / "excerpt.flac", sr=16000)
    ad, _ = load_wav(assets / "ad_base.wav", sr=16000)
    row = _waveform_window(main, ad, sr, 301.668, 310.524)
    # The programme is aligned, but narration creates unrelated low/high
    # correlation peaks about fourteen milliseconds apart at a false lag.
    assert row["status"] == "weak"
    assert row["lag_sec"] is None


@pytest.mark.parametrize("carrier_delay", [0.0, .6])
def test_statistically_clear_weak_carrier_vetoes_a_conflicting_narration_envelope(carrier_delay):
    from scipy.ndimage import gaussian_filter1d
    from adsync.quality import _waveform_window

    sr = 4000
    rng = np.random.default_rng(931)
    n = 24 * sr
    envelope = np.interp(np.arange(n) / sr, np.arange(0, 24, .08), rng.uniform(.05, 1, 300))
    envelope = gaussian_filter1d(envelope, .015 * sr)
    shift = round(1.2 * sr)
    narration_envelope = np.r_[envelope[shift:], np.zeros(shift)]
    main, ad = np.zeros(n), np.zeros(n)
    for name, edges in [("low", (80, 250)), ("mid", (250, 1200)), ("high", (1200, 1800))]:
        sos = butter(4, edges, btype="bandpass", fs=sr, output="sos")
        x, y = [sosfiltfilt(sos, rng.normal(size=n)) for _ in range(2)]
        x, y = x / x.std(), y / y.std()
        main += x * envelope * .03
        ad += y * narration_envelope * .06
        if name == "high":
            carrier = x * envelope * .012
            delay_samples = round(carrier_delay * sr)
            if delay_samples:
                carrier = np.r_[np.zeros(delay_samples), carrier[:-delay_samples]]
            ad += carrier
    row = _waveform_window(main, ad, sr, 8, 12)
    high = row["bands"]["high"]
    assert high["status"] == "weak"
    assert high["peak_z"] >= 8 and high["peak_ratio"] >= 1.4
    assert high["lag_sec"] == pytest.approx(-carrier_delay, abs=.005)
    assert row["status"] == "weak"
    assert row["conflict"]["envelope_lag_sec"] > 1.1
    assert row["conflict"]["raw_bands"][0]["lag_sec"] == pytest.approx(-carrier_delay, abs=.005)


def test_saved_envelope_reassessment_preserves_measurement_binding_and_only_downgrades():
    from adsync.quality_signal import reassess_envelope_conflict

    original = {"status": "strong", "lag_sec": .9, "method": "band_energy_envelope",
                "bound_output_sha256": "a" * 64,
                "bands": {"high": {"status": "weak", "lag_sec": .001, "score": .12,
                                    "peak_z": 17, "peak_ratio": 1.51, "usable_peak": True}}}
    result = reassess_envelope_conflict(original, measurement_revision="previous-policy")
    assert result["decision"] == "conflict"
    assert result["measurement_revision"] == "previous-policy"
    assert result["waveform"]["status"] == "weak"
    assert result["waveform"]["bound_output_sha256"] == "a" * 64
    assert result["original_measurement"]["lag_sec"] == .9
    assert original["status"] == "strong"


def test_saved_envelope_without_raw_statistics_requests_remeasurement():
    from adsync.quality_signal import reassess_envelope_conflict

    old = {"status": "strong", "lag_sec": .9, "method": "band_energy_envelope",
           "bands": {"high": {"status": "weak", "lag_sec": .001, "score": .12, "peak_ratio": 1.51}}}
    result = reassess_envelope_conflict(old, measurement_revision="legacy")
    assert result["decision"] == "remeasure"
    assert result["missing_statistics"]
    assert result["waveform"]["status"] != "pass"


def test_saved_consistent_nonzero_envelope_is_not_downgraded():
    from adsync.quality_signal import reassess_envelope_conflict

    original = {"status": "strong", "lag_sec": -.6, "method": "band_energy_envelope",
                "bands": {"high": {"status": "weak", "lag_sec": -.601, "score": .12,
                                    "peak_z": 17, "peak_ratio": 1.51, "usable_peak": True}}}
    result = reassess_envelope_conflict(original, measurement_revision="previous-policy")
    assert result["decision"] == "unchanged"
    assert result["waveform"]["status"] == "strong"
    assert result["waveform"]["lag_sec"] == -.6
