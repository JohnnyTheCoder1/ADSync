"""Tier-1 (coarse / cheap) feature extraction."""

from __future__ import annotations

from collections.abc import Iterator
import logging

import librosa
import numpy as np
from numpy.typing import NDArray

from adsync.models import FeatureBundle

log = logging.getLogger("adsync")
_BLOCK_FRAMES = 4096
_FULL_MEL_BANDS = 128


def _centered_blocks(
    y: NDArray[np.floating], n_fft: int, hop_length: int, n_frames: int,
) -> Iterator[tuple[slice, NDArray[np.floating]]]:
    """Yield contiguous frame ranges with the overlap needed by the FFT."""
    pad = n_fft // 2
    for first in range(0, n_frames, _BLOCK_FRAMES):
        stop = min(n_frames, first + _BLOCK_FRAMES)
        start_sample = first * hop_length - pad
        stop_sample = (stop - 1) * hop_length - pad + n_fft
        samples = y[max(0, start_sample):min(len(y), stop_sample)]
        # Pad only at the true signal boundaries, never at internal seams.
        if start_sample < 0 or stop_sample > len(y):
            samples = np.pad(samples, (max(0, -start_sample), max(0, stop_sample - len(y))))
        yield slice(first, stop), samples


def extract_basic_features(
    y: NDArray[np.floating],
    sr: int,
    *,
    hop_length: int = 512,
    n_fft: int = 2048,
    n_mels: int = 64,
    n_mfcc: int = 13,
) -> FeatureBundle:
    """Compute Tier-1 features: RMS, onset strength, low-res mel, and MFCC."""
    duration = float(len(y) / sr)

    n_frames = 1 + (len(y) - n_fft % 2) // hop_length
    rms = np.empty(n_frames, dtype=np.float32)
    mel_dtype = np.result_type(y.dtype, np.float32)
    mel = np.empty((n_mels, n_frames), dtype=mel_dtype)
    full_mel = np.empty((_FULL_MEL_BANDS, n_frames), dtype=mel_dtype)
    # librosa infers an even FFT size from an already-computed spectrogram.
    mel_fft = 2 * (n_fft // 2)
    filters = [
        librosa.filters.mel(sr=sr, n_fft=mel_fft, n_mels=bands)
        for bands in (n_mels, _FULL_MEL_BANDS)
    ]
    for columns, samples in _centered_blocks(y, n_fft, hop_length, n_frames):
        rms[columns] = librosa.feature.rms(
            y=samples, frame_length=n_fft, hop_length=hop_length, center=False,
        )[0]
        # Reuse one power spectrogram per block for both mel projections.
        power = np.abs(librosa.stft(samples, n_fft=n_fft, hop_length=hop_length, center=False))
        np.square(power, out=power)
        for output, basis in zip((mel, full_mel), filters, strict=True):
            output[:, columns] = np.einsum("ft,mf->mt", power, basis, optimize=True)

    # Preserve whole-track dB normalization and continuous onset detection.
    mel_db = librosa.power_to_db(mel, ref=np.max)
    del mel
    mel128_db = librosa.power_to_db(full_mel)
    del full_mel
    onset = librosa.onset.onset_strength(S=mel128_db, sr=sr, hop_length=hop_length)
    # Match librosa's slice when more coefficients are requested than available.
    n_coefficients = len(range(_FULL_MEL_BANDS)[:n_mfcc])
    mfcc = np.empty((n_coefficients, n_frames), dtype=mel128_db.dtype)
    for first in range(0, n_frames, _BLOCK_FRAMES):
        stop = min(n_frames, first + _BLOCK_FRAMES)
        mfcc[:, first:stop] = librosa.feature.mfcc(S=mel128_db[:, first:stop], sr=sr, n_mfcc=n_mfcc)

    log.info("Features: %.1f s, %d frames (hop=%d)", duration, len(rms), hop_length)

    return FeatureBundle(
        sr=sr,
        hop_length=hop_length,
        duration=duration,
        rms=rms,
        onset=onset,
        mel=mel_db,
        mfcc=mfcc,
    )
