"""Tier-2 (fine / expensive) feature extraction — used around anchor candidates."""

from __future__ import annotations

import logging

import librosa
import numpy as np
from numpy.typing import NDArray

log = logging.getLogger("adsync")


def extract_fine_window(
    y: NDArray[np.floating],
    sr: int,
    start_sec: float,
    end_sec: float,
    *,
    hop_length: int = 256,
    n_fft: int = 2048,
    n_mfcc: int = 20,
    n_mels: int = 128,
) -> dict[str, NDArray]:
    """Extract high-resolution features for a short window of audio.

    Returns a dict with keys: mel, mfcc, onset, spectral_flux.
    """
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    segment = y[start_sample:end_sample]

    if len(segment) < n_fft:
        segment = np.pad(segment, (0, n_fft - len(segment)))

    mel = librosa.feature.melspectrogram(
        y=segment, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    mfcc = librosa.feature.mfcc(
        y=segment, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft,
    )

    onset = librosa.onset.onset_strength(y=segment, sr=sr, hop_length=hop_length)

    # spectral flux
    S = np.abs(librosa.stft(segment, n_fft=n_fft, hop_length=hop_length))
    flux = np.sqrt(np.mean(np.diff(S, axis=1) ** 2, axis=0))

    return {
        "mel": mel_db,
        "mfcc": mfcc,
        "onset": onset,
        "spectral_flux": flux,
    }


def mfcc_similarity(feat_a: dict[str, NDArray], feat_b: dict[str, NDArray]) -> float:
    """Cosine similarity between mean MFCC vectors of two feature dicts."""
    mean_a = feat_a["mfcc"].mean(axis=1)
    mean_b = feat_b["mfcc"].mean(axis=1)
    dot = np.dot(mean_a, mean_b)
    norm = np.linalg.norm(mean_a) * np.linalg.norm(mean_b)
    if norm < 1e-10:
        return 0.0
    return float(dot / norm)
