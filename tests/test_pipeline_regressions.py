"""Pipeline regressions with deterministic alignment and real audio rendering."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

from adsync._pipeline import run_pipeline
from adsync.config import SyncConfig
from adsync.models import (
    Anchor,
    CandidateWindow,
    FeatureBundle,
    MediaInfo,
    OffsetCandidate,
    StreamInfo,
    WarpPath,
    WarpPoint,
)


@pytest.fixture
def pipeline_case(monkeypatch, tmp_path):
    """Keep file I/O and matching deterministic; exercise pipeline/map/render."""
    monkeypatch.chdir(tmp_path)
    sr = 8000
    duration = 12.0
    source = np.zeros(int(duration * sr), dtype=np.float32)
    # A balanced pulse gives an exact timestamp without DC-normalization noise.
    source[3 * sr] = 0.6
    source[3 * sr + 1] = -0.6
    state = SimpleNamespace(
        sr=sr, duration=duration, source=source, offset=0.0,
        extractions=[], rendered=None, anchor_calls=0, lattice=[],
    )

    def fake_probe(path):
        return MediaInfo(
            path=str(path), duration=duration,
            audio_streams=[StreamInfo(index=0, codec_type="audio", sample_rate=sr)],
        )

    def fake_extract(info, output_path, **kwargs):
        state.extractions.append((Path(output_path).name, kwargs.copy()))
        return output_path

    def fake_load(path, *, sr=sr, mono=True):
        return source.copy(), sr

    def fake_features(y, sr, **kwargs):
        return FeatureBundle(
            sr=sr, hop_length=512, duration=len(y) / sr,
            rms=np.zeros(8), onset=np.zeros(8),
            mel=np.zeros((2, 8)), mfcc=np.zeros((2, 8)),
        )

    def fake_anchors(*args, **kwargs):
        state.anchor_calls += 1
        return [
            Anchor(source_time=t, target_time=t + state.offset, score=0.9, window=2.0)
            for t in (2.0, 6.0, 10.0)
        ]

    def fake_decode(*args, **kwargs):
        return [
            WarpPoint(source_time=t, target_time=t + state.offset, confidence=0.8)
            for t in (2.0, 4.0, 6.0, 8.0, 10.0)
        ], 0.0

    def fake_fit(points, ad_duration, video_duration, **kwargs):
        fn = PchipInterpolator([0.0, ad_duration], [state.offset, ad_duration + state.offset])
        path = WarpPath(points=points, anchor_points=points, n_segments=1, mean_confidence=0.8)
        return [fn], [(0.0, ad_duration)], path

    def fake_mux(video_path, y, sr, output_path, **kwargs):
        state.rendered = y.copy()
        state.original_audio = kwargs.get("original_audio")

    monkeypatch.setattr("adsync.media.probe.probe", fake_probe)
    monkeypatch.setattr("adsync.media.extract.extract_audio", fake_extract)
    monkeypatch.setattr("adsync.features.load.load_wav", fake_load)
    monkeypatch.setattr("adsync.features.preprocess.preprocess", lambda y, *a, **kw: y)
    monkeypatch.setattr("adsync.features.extract_basic.extract_basic_features", fake_features)
    monkeypatch.setattr("adsync.align.global_offset.estimate_global_offset", lambda *a, **kw: (state.offset, 0.95, [state.offset]))
    monkeypatch.setattr("adsync.align.drift.estimate_drift", lambda *a, **kw: (0.0, 0.0, [], state.offset))
    monkeypatch.setattr("adsync.align.anchors.find_anchors", fake_anchors)
    monkeypatch.setattr("adsync.align.candidate_lattice.build_candidate_lattice", lambda *a, **kw: state.lattice)
    monkeypatch.setattr("adsync.align.warp_decode.decode_warp_path", fake_decode)
    monkeypatch.setattr("adsync.align.warp_fit.fit_warp_function", fake_fit)
    monkeypatch.setattr("adsync.media.mux.mux_ad_track", fake_mux)
    monkeypatch.setattr("adsync.report.json_report.print_summary", lambda report: None)

    def run(mode="offset", *, mux=False, offset_adjust=0.0, report_path=None, prepare_audio=False,
            fingerprint=False):
        return run_pipeline(
            video_path=Path("video.mkv"), ad_path=Path("ad.wav"),
            output_path=Path("synced.mkv") if mux else None,
            report_path=report_path,
            config=SyncConfig(
                mode=mode, analysis_sr=sr, fingerprint=fingerprint,
                speed_detect=False, offset_adjust=offset_adjust,
                device="cpu",
                prepare_audio=prepare_audio,
            ), mux=mux,
        )

    state.run = run
    return state


def test_analyze_skips_unused_hq_decode(pipeline_case):
    case = pipeline_case
    case.run()
    assert len(case.extractions) == 2
    assert all(kwargs.get("sample_fmt", "s16") != "f32" for _, kwargs in case.extractions)


def test_negative_offset_rebuild_trims_source_before_video_start(pipeline_case):
    case = pipeline_case
    case.offset = -2.0
    report = case.run(mux=True)
    assert report.mode == "offset"
    assert case.rendered.shape[-1] == int(case.duration * case.sr)
    assert np.argmax(case.rendered) == case.sr, "AD source 3 s must play at video 1 s"
    assert np.count_nonzero(case.rendered[10 * case.sr:]) == 0


def test_forced_warp_starved_lattice_uses_piecewise_fallback(pipeline_case):
    case = pipeline_case
    report = case.run("warp")
    assert case.anchor_calls == 1
    assert report.mode == "piecewise"
    assert report.segments


def test_warp_report_retains_a_replayable_fit_and_vetting_evidence(pipeline_case):
    case = pipeline_case
    case.lattice = [CandidateWindow(
        source_center=t, speech_score=0, energy=1,
        candidates=[OffsetCandidate(offset_sec=0, score=.9, peak_sharpness=1, peak_ratio=2)],
    ) for t in (2., 6., 10.)]
    report = case.run("warp", offset_adjust=.25)
    debug = report.timing_debug
    assert debug["pre_vetting_path"]
    assert debug["post_vetting_path"]
    assert debug["segment_ranges"] == [[0, case.duration]]
    from scipy.interpolate import PPoly
    restored = PPoly(np.array(debug["fitted_pchip"][0]["c"]),
                     np.array(debug["fitted_pchip"][0]["x"]))
    assert float(restored(3)) == pytest.approx(3.25)


def test_failed_identity_keeps_report_without_rendering_media(pipeline_case, monkeypatch, tmp_path):
    from adsync.align.fingerprint import FingerprintResult

    monkeypatch.setattr("adsync.align.fingerprint.fingerprint_align", lambda *a: FingerprintResult())
    monkeypatch.setattr("adsync.quality.assess_content_identity",
                        lambda *a, **kw: {"status": "fail", "reasons": ["Shared intro only"]})
    report_path = tmp_path / "needs-review.json"
    report = pipeline_case.run(mux=True, fingerprint=True, report_path=report_path)
    assert report_path.exists()
    assert report.identity_check["status"] == "fail"
    assert report.output_path is None
    assert pipeline_case.rendered is None
    assert len(pipeline_case.extractions) == 2


def test_pipeline_fills_one_local_gap_despite_high_global_coverage(pipeline_case, monkeypatch):
    from adsync.align.fingerprint import FingerprintResult, FingerprintSpan

    times = np.arange(.01, 11.99, .01, dtype=np.float32)
    fp = FingerprintResult(spans=[FingerprintSpan(0, 12, 0, len(times))],
                           match_t_ad=times, match_t_vid=times.copy())
    monkeypatch.setattr("adsync.align.fingerprint.fingerprint_align", lambda *a: fp)
    case = pipeline_case
    case.lattice = [CandidateWindow(source_center=float(t), speech_score=.5, energy=1,
        candidates=[] if t == 6 else [OffsetCandidate(offset_sec=0, score=.9,
            peak_sharpness=2, peak_ratio=2)]) for t in range(1, 12)]
    report = case.run("warp", fingerprint=True)
    assert report.fp_anchor_windows == 1
    assert len(case.lattice[5].candidates) == 1


@pytest.mark.parametrize("mode", ["offset", "piecewise", "warp"])
@pytest.mark.parametrize("adjust", [1.25, -1.25])
def test_manual_offset_moves_rendered_audio_in_each_mode(pipeline_case, mode, adjust):
    case = pipeline_case
    case.lattice = [
        CandidateWindow(
            source_center=t, speech_score=0.0, energy=1.0,
            candidates=[OffsetCandidate(offset_sec=0.0, score=0.9, peak_sharpness=1.0, peak_ratio=2.0)],
        )
        for t in (2.0, 6.0, 10.0)
    ]
    case.run(mode, mux=True, offset_adjust=adjust)
    assert np.argmax(case.rendered) == round((3.0 + adjust) * case.sr)


def test_unreliable_drift_cannot_override_reliable_global_offset(pipeline_case, monkeypatch):
    case = pipeline_case
    weak_anchors = [
        Anchor(source_time=t, target_time=t + 4.0, score=0.25, window=2.0)
        for t in (2.0, 6.0, 10.0)
    ]
    monkeypatch.setattr(
        "adsync.align.drift.estimate_drift",
        lambda *args, **kwargs: (10000.0, 0.05, weak_anchors, 5.0),
    )
    report = case.run("auto", mux=True)
    assert report.mode == "offset"
    assert report.segments[0].stretch == 1.0
    assert report.segments[0].offset == 0.0
    assert np.argmax(case.rendered) == 3 * case.sr


def test_report_cannot_overwrite_input_before_decoding(pipeline_case):
    case = pipeline_case
    with pytest.raises(ValueError, match="overwrite"):
        case.run(report_path=Path("ad.wav"))
    assert not case.extractions


def test_failed_publication_keeps_report_with_local_media(pipeline_case, monkeypatch, tmp_path):
    import json
    from adsync.media.output import OutputPublicationError

    local_media = tmp_path / "retained.mkv"
    local_media.write_bytes(b"completed media")

    def cannot_publish(*args, **kwargs):
        raise OutputPublicationError(local_media, Path("unavailable/output.mkv"), OSError("share offline"))

    monkeypatch.setattr("adsync.media.mux.mux_ad_track", cannot_publish)
    with pytest.raises(OutputPublicationError):
        pipeline_case.run(mux=True)
    report = json.loads(local_media.with_suffix(".report.json").read_text(encoding="utf-8"))
    assert report["output_path"] == str(local_media)
    assert "share offline" in " ".join(report["warnings"])
    assert local_media.read_bytes() == b"completed media"
