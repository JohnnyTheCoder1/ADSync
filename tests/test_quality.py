"""Regression checks for wrong-content acceptance and local QC false alarms."""

import numpy as np
import pytest

from adsync.align.fingerprint import FingerprintResult, FingerprintSpan


def evidence(times, offsets=0.0, *, duration=600.0):
    times = np.asarray(times, dtype=np.float64)
    return FingerprintResult(
        spans=[FingerprintSpan(0.0, duration, float(np.median(offsets)), len(times))],
        n_matches=len(times), match_t_ad=times, match_t_vid=times + offsets,
    )


def test_sparse_intro_match_cannot_establish_episode_identity():
    from adsync.quality import assess_content_identity

    fp = FingerprintResult(spans=[FingerprintSpan(0, 20, 0, 700)], n_matches=700,
                           match_t_ad=np.linspace(0, 20, 700), match_t_vid=np.linspace(0, 20, 700))
    assert fp.strong  # The previous steering flag accepted this false lead.
    result = assess_content_identity(fp, 600)
    assert result["status"] == "fail"
    assert result["coverage"] < 0.05


def test_broad_edited_content_establishes_identity_without_zero_offset():
    from adsync.quality import assess_content_identity

    times = np.arange(0.1, 600, 0.4)
    offsets = np.where(times < 300, 12.0, 29.0)
    fp = evidence(times, offsets)
    fp.spans = [FingerprintSpan(0, 300, 12, 750), FingerprintSpan(300, 600, 29, 750)]
    result = assess_content_identity(fp, 600, video_duration=630)
    assert result["status"] == "pass"
    assert min(result["third_coverage"]) >= 0.95


def test_scattered_offsets_do_not_pollute_dominant_cluster_p95():
    from adsync.quality import dominant_offset_cluster

    times = np.arange(0, 60, 0.1)
    offsets = np.zeros(len(times))
    offsets[::7] = 0.416
    cluster = dominant_offset_cluster(times, offsets)
    assert cluster["strong"]
    assert cluster["lag_sec"] == pytest.approx(0)
    assert cluster["p95_abs_sec"] == pytest.approx(0)
    assert cluster["all_p95_abs_sec"] > 0.4


def test_waveform_offset_sign_and_subsample_precision():
    from adsync.quality import waveform_correlation

    rng = np.random.default_rng(591)
    rate = 4000
    main = rng.normal(size=12 * rate)
    ad = np.r_[np.zeros(640), main[:-640]]
    result = waveform_correlation(main, ad, rate, guard_sec=2)
    assert result["status"] == "strong"
    # Common convention with fingerprints: main time minus AD time.
    assert result["lag_sec"] == pytest.approx(-0.160, abs=0.0001)
    assert 0.9999 <= result["score"] <= 1.0


def test_near_silent_correlated_tail_is_not_evidence():
    from adsync.quality import waveform_correlation

    rng = np.random.default_rng(593)
    main = rng.normal(size=48000) * 1e-12
    result = waveform_correlation(main, main, 4000, guard_sec=2)
    assert result["status"] == "weak"
    assert "energy" in result["reason"]


def test_targeted_windows_cover_both_sides_of_edits_sparse_and_short_tail():
    from adsync.quality import targeted_windows

    windows = targeted_windows(600, cut_times=[213.4], sparse_intervals=[(330, 390)],
                               terminal_intervals=[(590, 600)])
    assert any(w["end_sec"] <= 213.4 and w["end_sec"] >= 211 for w in windows)
    assert any(w["start_sec"] >= 213.4 and w["start_sec"] <= 215 for w in windows)
    for point in (337, 355, 375, 387, 594, 598):
        assert any(w["start_sec"] <= point <= w["end_sec"] for w in windows)
    assert all(0 <= w["start_sec"] < w["end_sec"] <= 600 for w in windows)


def test_rendered_qc_accepts_scattered_false_fingerprints_with_waveform_support():
    from adsync.quality import verify_audio_sync

    rate = 4000
    main = np.random.default_rng(9).normal(scale=0.1, size=60 * rate)
    times = np.arange(0.1, 60, 0.05)
    offsets = np.zeros(len(times))
    offsets[::7] = 0.416
    result = verify_audio_sync(main, main.copy(), rate, fingerprint=evidence(times, offsets, duration=60))
    assert result["status"] == "pass", result


def test_rendered_qc_detects_short_coherent_bad_span_despite_good_global_average():
    from adsync.quality import verify_audio_sync

    rate = 4000
    main = np.random.default_rng(10).normal(scale=0.1, size=120 * rate)
    ad = main.copy()
    ad[55 * rate:65 * rate] = main[55 * rate - 800:65 * rate - 800]
    times = np.arange(0.1, 120, 0.05)
    offsets = np.where((times >= 55) & (times < 65), -0.2, 0.0)
    result = verify_audio_sync(main, ad, rate, fingerprint=evidence(times, offsets, duration=120),
                               cut_times=[55, 65])
    assert result["status"] == "fail"
    assert any(w["status"] == "fail" for w in result["windows"])


def test_qc_targets_use_fitted_output_coordinates_without_scanning_entire_final_segment():
    from adsync.quality import report_targets

    report = {"segments": [], "fingerprint_unmatched": [(190, 210)], "offset_adjust": 0.5,
              "timing_debug": {"segment_ranges": [[0, 200], [200, 600]],
                               "fitted_pchip": [{"x": [0, 200], "c": [[0], [0], [1], [10.5]]},
                                                {"x": [200, 600], "c": [[0], [0], [1], [230.5]]}]}}
    targets = report_targets(report)
    assert targets["cut_times"] == [210.5, 230.5]
    assert targets["sparse_intervals"] == [(200.5, 210.5), (230.5, 240.5)]
    assert targets["terminal_intervals"] == [(600.5, 630.5)]


def test_output_qc_selects_appended_ad_and_correct_main_language():
    from adsync.models import MediaInfo, StreamInfo
    from adsync.quality import select_verification_streams

    info = MediaInfo(path="three_tracks.mkv", audio_streams=[
        StreamInfo(index=1, codec_type="audio", language="spa", channels=6),
        StreamInfo(index=2, codec_type="audio", language="eng", channels=6),
        StreamInfo(index=3, codec_type="audio", language="eng", channels=2, title="Audio Description"),
    ])
    main, ad = select_verification_streams(info)
    assert main.index == 2
    assert ad.index == 3


def test_appended_custom_title_wins_over_old_original_audio_description():
    from adsync.models import MediaInfo, StreamInfo
    from adsync.quality import select_verification_streams

    info = MediaInfo(path="four_tracks.mkv", audio_streams=[
        StreamInfo(index=1, codec_type="audio", language="eng", channels=6),
        StreamInfo(index=2, codec_type="audio", language="eng", channels=2, title="Audio Description"),
        StreamInfo(index=3, codec_type="audio", language="eng", channels=2, title="English AD"),
    ])
    main, ad = select_verification_streams(info)
    assert main.index == 1
    assert ad.index == 3


def test_mutual_terminal_silence_is_not_claimed_as_timing_evidence_or_false_review():
    from adsync.quality import verify_audio_sync

    sr = 4000
    main = np.random.default_rng(43).normal(scale=0.1, size=120 * sr)
    main[-12 * sr:] = 0
    fp = evidence(np.arange(0.1, 108, 0.05), duration=120)
    result = verify_audio_sync(main, main.copy(), sr, fingerprint=fp)
    assert result["status"] == "pass", result["review_reasons"]
    assert any(w["status"] == "silent" for w in result["windows"])


def test_cached_landmarks_reuse_analysis_and_invalidate_changed_audio(tmp_path, monkeypatch):
    from adsync.cache import ArtifactCache
    from adsync.quality import cached_landmarks
    import adsync.align.fingerprint as module

    sr = 8000
    audio = np.random.default_rng(70).normal(scale=0.1, size=sr * 3).astype(np.float32)
    cache = ArtifactCache(tmp_path)
    first = cached_landmarks(audio, sr, cache=cache)
    original = module._landmarks
    def forbid_recompute(*args):
        raise AssertionError("Unchanged landmark extraction ran again")
    monkeypatch.setattr(module, "_landmarks", forbid_recompute)
    again = cached_landmarks(audio.copy(), sr, cache=cache)
    assert np.array_equal(first[0], again[0])
    assert np.array_equal(first[1], again[1])
    with pytest.raises(AssertionError, match="ran again"):
        cached_landmarks(audio * 0.5, sr, cache=cache)
    monkeypatch.setattr(module, "_landmarks", original)


def test_content_comparison_detects_reencoded_duplicate_and_rejects_different_recording(tmp_path):
    from adsync.cache import ArtifactCache
    from adsync.quality import fingerprint_features, compare_recording_content

    sr = 8000
    rng = np.random.default_rng(75)
    first = rng.normal(scale=0.1, size=sr * 60).astype(np.float32)
    reencoded = (np.round(first * 15000) / 30000).astype(np.float32)
    other = rng.normal(scale=0.1, size=len(first)).astype(np.float32)
    cache = ArtifactCache(tmp_path)
    features = [fingerprint_features(y, sr, cache=cache) for y in (first, reencoded, other)]
    assert compare_recording_content(features[0], features[1])["duplicate"]
    assert not compare_recording_content(features[0], features[2])["duplicate"]


def test_video_preservation_rejects_changed_pixels_but_accepts_remux(tmp_path):
    import shutil
    import subprocess
    from adsync.quality import verify_video_preservation

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg unavailable")
    original, copy, changed = [tmp_path / name for name in ("original.mkv", "copy.mkv", "changed.mkv")]
    for target, source in ((original, "testsrc2=size=64x48:rate=5"), (changed, "color=red:size=64x48:rate=5")):
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", source, "-t", "1",
                        "-c:v", "ffv1", str(target)], check=True, capture_output=True)
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(original), "-map", "0", "-c", "copy", str(copy)],
                   check=True, capture_output=True)
    assert verify_video_preservation(original, copy)["status"] == "pass"
    assert verify_video_preservation(original, changed)["status"] == "fail"


def test_last_second_is_measured_instead_of_discarded_as_search_guard():
    from adsync.quality import verify_audio_sync

    sr = 4000
    main = np.random.default_rng(211).normal(scale=0.1, size=60 * sr)
    ad = main.copy()
    ad[-sr:] = np.roll(ad[-sr:], 800)
    result = verify_audio_sync(main, ad, sr, fingerprint=evidence(np.arange(0.1, 60, 0.05), duration=60))
    assert result["status"] != "pass"
    assert any(w["start_sec"] >= 59 and w["status"] != "pass" for w in result["windows"])


def test_rendered_local_shift_is_targeted_when_original_report_has_no_edits():
    from adsync.quality import verify_audio_sync

    sr = 1000
    main = np.random.default_rng(212).normal(scale=0.1, size=600 * sr)
    ad = main.copy()
    ad[273 * sr:293 * sr] = main[273 * sr - 220:293 * sr - 220]
    times = np.arange(0.1, 600, 0.1)
    offsets = np.where((times >= 273) & (times < 293), -0.22, 0)
    fp = evidence(times, offsets)
    fp.spans = [FingerprintSpan(0, 273, 0, 2730), FingerprintSpan(273, 293, -0.22, 200),
                FingerprintSpan(293, 600, 0, 3070)]
    result = verify_audio_sync(main, ad, sr, fingerprint=fp)
    assert result["status"] == "fail"
    assert any(273 <= w["start_sec"] < 293 and w["status"] == "fail" for w in result["windows"])


def test_independent_band_envelopes_verify_remixed_waveforms_without_lowering_raw_thresholds():
    from scipy.ndimage import gaussian_filter1d
    from adsync.quality import _waveform_window

    sr = 4000
    rng = np.random.default_rng(301)
    n = sr * 24
    envelope = np.interp(np.arange(n) / sr, np.arange(0, 24, 0.08), rng.uniform(0.1, 1, 300))
    envelope = gaussian_filter1d(envelope, 0.015 * sr)
    main = rng.normal(scale=0.1, size=n) * envelope
    remix = rng.normal(scale=0.1, size=n) * envelope
    result = _waveform_window(main, remix, sr, 4, 20)
    assert result["bands"]["full"]["status"] == "weak"
    assert result["status"] == "strong"
    assert result["method"] == "band_energy_envelope"
    assert abs(result["lag_sec"]) < 0.035
    shifted = np.r_[np.zeros(1000), remix[:-1000]]
    result = _waveform_window(main, shifted, sr, 4, 20)
    assert result["status"] == "strong"
    assert result["lag_sec"] == pytest.approx(-0.250, abs=0.035)


def test_envelope_fallback_does_not_accept_unrelated_recordings():
    from adsync.quality import _waveform_window

    sr = 4000
    rng = np.random.default_rng(303)
    for _ in range(10):
        n = sr * 20
        first = rng.normal(scale=0.1, size=n)
        second = rng.normal(scale=0.1, size=n)
        for signal in (first, second):
            signal *= np.interp(np.arange(n) / sr, np.arange(20), rng.uniform(0, 1, 20))
        result = _waveform_window(first, second, sr, 4, 16)
        assert result["status"] == "weak", result


@pytest.mark.parametrize("delay", [0.6, 1.0, 1.5, 3.0])
def test_real_fingerprints_expose_four_second_shift_hidden_by_global_zero_span(monkeypatch, delay):
    from adsync.quality import verify_audio_sync

    monkeypatch.setenv("ADSYNC_CACHE_DIR", "off")
    sr = 4000
    main = np.random.default_rng(902).normal(scale=0.1, size=240 * sr).astype(np.float32)
    ad = main.copy()
    ad[103 * sr:107 * sr] = main[round((103 - delay) * sr):round((107 - delay) * sr)]
    result = verify_audio_sync(main, ad, sr)
    assert result["status"] == "fail"
    assert any(w["start_sec"] < 107 and w["end_sec"] > 103 and w["status"] == "fail" for w in result["windows"])


@pytest.mark.parametrize("delay", [0.0, 0.6])
def test_separated_envelope_chunks_resolve_intermittent_narration(delay):
    from scipy.ndimage import gaussian_filter1d
    from adsync.quality import _waveform_window

    sr = 4000
    rng = np.random.default_rng(311)
    n = sr * 24
    envelope = np.interp(np.arange(n) / sr, np.arange(0, 24, 0.08), rng.uniform(0.1, 1, 300))
    envelope = gaussian_filter1d(envelope, 0.015 * sr)
    main = rng.normal(scale=0.1, size=n) * envelope
    remix = rng.normal(scale=0.1, size=n) * envelope
    for lo, hi in [(6.3, 7.3), (12.9, 14.2)]:
        a, b = round(lo * sr), round(hi * sr)
        remix[a:b] += rng.normal(scale=2, size=b - a)
    if delay:
        offset = round(delay * sr)
        remix = np.r_[np.zeros(offset), remix[:-offset]]
    result = _waveform_window(main, remix, sr, 4, 20)
    assert result["status"] == "strong"
    assert result["method"] == "separated_envelope_chunks"
    assert result["lag_sec"] == pytest.approx(-delay, abs=0.035)
    assert len(result["chunk_consensus"]) >= 3


def test_strong_local_waveform_resolves_repeated_content_fingerprint_location():
    from adsync.quality import verify_audio_sync

    sr = 4000
    main = np.random.default_rng(330).normal(scale=.1, size=180 * sr)
    # A repeated musical motif elsewhere can win global landmark votes even
    # though the correct local rendered soundtrack is independently aligned.
    times = np.arange(.1, 180, .05)
    offsets = np.where((times >= 90) & (times < 110), 45.312, 0)
    fp = evidence(times, offsets, duration=180)
    fp.spans = [FingerprintSpan(0, 90, 0, 1800), FingerprintSpan(90, 110, 45.312, 400),
                FingerprintSpan(110, 180, 0, 1400)]
    result = verify_audio_sync(main, main.copy(), sr, fingerprint=fp)
    assert result["status"] == "pass", result["failures"]
    assert any(w.get("adjudication") for w in result["windows"])


def test_short_narration_gap_with_measured_aligned_flanks_is_bounded_context():
    from adsync.quality import verify_audio_sync

    sr = 4000
    rng = np.random.default_rng(340)
    main = rng.normal(scale=.1, size=180 * sr)
    ad = main.copy()
    ad[103 * sr:107 * sr] += rng.normal(scale=2, size=4 * sr)
    times = np.arange(.1, 180, .05)
    times = times[(times < 103) | (times >= 107)]
    result = verify_audio_sync(main, ad, sr, fingerprint=evidence(times, duration=180))
    assert result["status"] == "pass", result["review_reasons"]
    assert any(w["status"] == "supported_context" for w in result["windows"])


def test_quiet_broad_rumble_is_not_counted_as_two_independent_bands():
    from adsync.quality import _waveform_window

    sr = 4000
    t = np.arange(6 * sr) / sr
    main = .00015 * np.cos(2 * np.pi * 100 * t) * np.exp(-((t - 3) / .25) ** 2)
    ad = .001 * np.cos(2 * np.pi * 100 * (t - .36)) * np.exp(-((t - 3.36) / .275) ** 2)
    result = _waveform_window(main, ad, sr, 2.5, 3.5)
    assert result["bands"]["full"]["score"] > .95
    assert result["bands"]["low"]["score"] > .95
    assert result["status"] == "weak"


def test_quiet_distinct_transient_still_exposes_a_real_shift():
    from adsync.quality import _waveform_window

    sr = 4000
    t = np.arange(6 * sr) / sr
    main = .00015 * np.cos(2 * np.pi * 100 * t) * np.exp(-((t - 3) / .1) ** 2)
    ad = .001 * np.cos(2 * np.pi * 100 * (t - .36)) * np.exp(-((t - 3.36) / .11) ** 2)
    result = _waveform_window(main, ad, sr, 2.5, 3.5)
    assert result["status"] == "strong"
    assert result["lag_sec"] == pytest.approx(-.36, abs=.02)


def test_naturally_sparse_landmarks_do_not_create_an_error_window_at_every_gap():
    from adsync.quality import verify_audio_sync

    sr = 1000
    main = np.random.default_rng(341).normal(scale=.1, size=120 * sr)
    result = verify_audio_sync(main, main.copy(), sr,
                               fingerprint=evidence(np.arange(.1, 120, 1.8), duration=120))
    assert not any("short_support_gap" in w["reasons"] for w in result["windows"])


def test_correctly_timed_narration_gaps_pass_without_becoming_fake_measurements():
    from adsync.quality import verify_audio_sync

    sr = 1000
    rng = np.random.default_rng(342)
    main = rng.normal(scale=.1, size=180 * sr)
    ad = main.copy()
    times = np.arange(.1, 180, .05)
    for start in (20, 50, 80, 110, 140):
        ad[start * sr:(start + 2) * sr] += rng.normal(scale=2, size=2 * sr)
        times = times[(times < start) | (times >= start + 2)]
    result = verify_audio_sync(main, ad, sr, fingerprint=evidence(times, duration=180))
    assert result["supported_context_coverage"] > .03
    assert result["status"] == "pass"
    assert result["policy_summary"]["bounded_unmeasured_sec"] > 5
    assert any(w["status"] == "supported_context" for w in result["windows"])


def test_narration_envelope_cannot_override_conflicting_unique_shared_waveform():
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import butter, sosfiltfilt
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
            ad += x * envelope * .027
    result = _waveform_window(main, ad, sr, 8, 12)
    assert result["bands"]["high"]["score"] > .4
    assert abs(result["bands"]["high"]["lag_sec"]) < .01
    assert result["envelopes"]["full"]["lag_sec"] > 1.1
    assert result["status"] == "weak"
    assert result["conflict"]["envelope_lag_sec"] > 1.1
    assert result["conflict"]["raw_bands"][0]["name"] == "high"
