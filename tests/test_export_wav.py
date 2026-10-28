"""WAV exports never exceed what a RIFF header can describe."""

import numpy as np
import soundfile as sf

from adsync.rebuild import export


def _stereo(frames, sr=8000):
    t = np.arange(frames, dtype=np.float64) / sr
    return np.stack([0.25 * np.sin(2 * np.pi * 440 * t), 0.25 * np.sin(2 * np.pi * 660 * t)]).astype(np.float32)


def test_export_too_large_for_riff_is_written_as_rf64(monkeypatch, tmp_path):
    y = _stereo(8000)
    # 8000 frames x 2 channels x 4 bytes exceeds this stand-in for the 4 GiB limit.
    monkeypatch.setattr(export, "_RIFF_DATA_LIMIT", 1000)
    path = export.export_wav(y, 8000, tmp_path / "big.wav", subtype="FLOAT")
    assert sf.info(str(path)).format == "RF64"
    data, _ = sf.read(str(path), dtype="float32")
    assert np.array_equal(data.T, y)


def test_export_within_riff_limit_stays_plain_wav(tmp_path):
    y = _stereo(8000)
    path = export.export_wav(y, 8000, tmp_path / "small.wav", subtype="FLOAT")
    assert sf.info(str(path)).format == "WAV"
    data, _ = sf.read(str(path), dtype="float32")
    assert np.array_equal(data.T, y)
