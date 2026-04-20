"""Audio preprocessing before feature extraction."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfilt

from adsync.utils.mathx import rms_normalize


def preprocess(
    y: NDArray[np.floating],
    sr: int,
    *,
    highpass_hz: float = 80.0,
    normalize_db: float = -20.0,
    trim_silence: bool = True,
    trim_top_db: float = 40.0,
) -> NDArray[np.floating]:
    """Mild preprocessing: high-pass, normalize, trim silence."""
    # high-pass filter to remove rumble
    if highpass_hz > 0:
        sos = butter(4, highpass_hz, btype="high", fs=sr, output="sos")
        y = sosfilt(sos, y).astype(y.dtype)

    # mild RMS normalization
    y = rms_normalize(y, target_db=normalize_db)

    # trim leading/trailing digital silence
    if trim_silence:
        import librosa
        y_trimmed, _ = librosa.effects.trim(y, top_db=trim_top_db)
        if len(y_trimmed) > sr:  # only use trimmed if it keeps a reasonable length
            y = y_trimmed

    return y
