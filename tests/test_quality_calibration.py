"""Independent end-to-end positive/negative QC controls on real shared audio."""

from pathlib import Path
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from qc_calibration import assess, load_excerpt, run_staged, shifted


@pytest.fixture(scope="module")
def real_pair():
    assets = Path(__file__).resolve().parents[1] / "harness_assets"
    main, ad = assets / "excerpt.flac", assets / "ad_base.wav"
    if not main.exists() or not ad.exists():
        pytest.skip("Optional real-media accuracy harness assets are not installed")
    sr = 16000
    # Same real excerpt in two independently mixed files; no synthetic pairing
    # or mocked fingerprint evidence can manufacture a positive outcome here.
    x, y = load_excerpt(main, 0, 60, sr), load_excerpt(ad, 0, 60, sr)
    size = min(len(x), len(y))
    return x[:size], y[:size], sr


def test_real_aligned_description_has_a_usable_positive_qc_control(real_pair):
    main, ad, sr = real_pair
    result = assess(main, ad, sr)
    assert result["status"] == "pass", result.get("review_reasons", result.get("failures"))
    assert any(w.get("waveform", {}).get("status") == "strong" for w in result["windows"])


@pytest.mark.parametrize("delay", [.2, -.2, .8, 1.5])
def test_real_description_detects_known_signed_timing_errors(real_pair, delay):
    main, ad, sr = real_pair
    result = assess(main, shifted(ad, delay, sr), sr)
    assert result["status"] == "fail", (delay, result.get("review_reasons"))
    assert result["failures"]


def test_real_two_second_error_cannot_hide_inside_good_global_alignment(real_pair):
    main, ad, sr = real_pair
    wrong = ad.copy()
    lo, hi, delta = 42 * sr, 44 * sr, round(.8 * sr)
    wrong[lo:hi] = ad[lo - delta:hi - delta]
    result = assess(main, wrong, sr)
    assert result["status"] == "fail", result.get("review_reasons")
    assert any(w["status"] == "fail" and w["start_sec"] < 44 and w["end_sec"] > 42
               for w in result["windows"])


def test_full_aligned_real_track_does_not_turn_weak_shared_audio_into_false_errors(real_pair):
    assets = Path(__file__).resolve().parents[1] / "harness_assets"
    sr = 16000
    main = load_excerpt(assets / "excerpt.flac", 0, None, sr)
    ad = load_excerpt(assets / "ad_base.wav", 0, None, sr)
    size = min(len(main), len(ad))
    result = assess(main[:size], ad[:size], sr)
    # Sparse narration regions can legitimately require review; an invented
    # coherent nonzero offset is a false measurement on this known good pair.
    assert result["status"] != "fail", result.get("failures")


def test_staged_calibration_records_selected_artifact_and_preserves_its_signature(tmp_path, monkeypatch):
    media = tmp_path / "rendered.mkv"
    media.write_bytes(b"staged test artifact")
    info = media.stat()
    signature = {"size": info.st_size, "mtime_ns": info.st_mtime_ns, "sha256": "bound-existing-receipt"}
    records = tmp_path / "records"
    records.mkdir()
    (records / "S01E01.progress.json").write_text(json.dumps({"local_path": str(media), "local_signature": signature}))
    monkeypatch.setattr("adsync.quality.verify_media_sync", lambda *args, **kwargs:
                        {"status": "pass", "windows": [], "failures": [], "review_reasons": []})
    output = tmp_path / "results"
    result = run_staged(records, output, ["S01E01"])
    assert result[0]["status"] == "pass"
    assert json.loads((output / "S01E01.json").read_text())["staged_signature"] == signature


def test_staged_calibration_does_not_claim_success_for_missing_targets(tmp_path):
    with pytest.raises(ValueError, match="No staged progress"):
        run_staged(tmp_path, tmp_path / "results", ["S01E01"])
