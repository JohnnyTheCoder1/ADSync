"""Tier-1 (coarse / cheap) feature extraction."""

from __future__ import annotations

import logging

import librosa
import numpy as np
from numpy.typing import NDArray

from adsync.models import FeatureBundle

log = logging.getLogger("adsync")


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

    # RMS envelope
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]

    # Onset strength
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # Mel spectrogram (low-res summary)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)

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
