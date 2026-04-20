"""Build a candidate lattice — top-K offset hypotheses per analysis window.

Uses raw-audio cross-correlation (downsampled to ~4 kHz) for robustness
against repetitive musical content, with speech-likeness scoring per window.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.signal import fftconvolve, find_peaks

from adsync.models import CandidateWindow, FeatureBundle, OffsetCandidate

log = logging.getLogger("adsync")


def build_candidate_lattice(
    video_features: FeatureBundle,
    ad_features: FeatureBundle,
    *,
    y_vid: NDArray | None = None,
    y_ad: NDArray | None = None,
    audio_sr: int = 16000,
    window_sec: float = 8.0,
    step_sec: float = 2.0,
    search_radius_sec: float = 30.0,
    min_score: float = 0.3,
    max_candidates: int = 5,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[CandidateWindow]:
    """Scan the AD track in windows and keep top-K offset candidates per window.

    When *y_vid* and *y_ad* are provided, uses raw-audio cross-correlation
    (downsampled to ~4 kHz) instead of onset features.  This is substantially
    more robust in music-heavy regions where onset patterns repeat.

    Returns a list of :class:`CandidateWindow`, one per analysis position.
    """
    # Decide signal source: raw audio (preferred) or onset features
    if y_vid is not None and y_ad is not None:
        ds_factor = max(1, audio_sr // 4000)
        ds_sr = audio_sr / ds_factor

        def _ds(y: NDArray) -> NDArray:
            n = len(y) - len(y) % ds_factor
            return np.mean(y[:n].reshape(-1, ds_factor), axis=1)

        vid = _ds(y_vid).astype(np.float64)
        ad = _ds(y_ad).astype(np.float64)
        sec_per_sample = 1.0 / ds_sr
    else:
        vid = video_features.onset.astype(np.float64)
        ad = ad_features.onset.astype(np.float64)
        sec_per_sample = video_features.hop_length / video_features.sr

    win_samples = int(window_sec / sec_per_sample)
    step_samples = int(step_sec / sec_per_sample)
    search_samples = int(search_radius_sec / sec_per_sample)
    min_peak_dist = max(1, win_samples // 4)

    # Precompute mel spectrogram info for speech scoring
    mel = ad_features.mel  # (n_mels, n_frames)
    mel_hop_sec = ad_features.hop_length / ad_features.sr
    n_mels = mel.shape[0]
    # Narration band: ~100-4000 Hz mapped to mel bin indices
    # At sr=16000, mel bands 0..n_mels span 0..8000 Hz roughly linearly in mel
    narration_lo = max(0, int(n_mels * 100 / 8000))
    narration_hi = min(n_mels, int(n_mels * 4000 / 8000))

    ad_positions = list(range(0, len(ad) - win_samples, step_samples))
    total_windows = len(ad_positions)
    lattice: list[CandidateWindow] = []

    for step_i, ad_start in enumerate(ad_positions):
        if on_progress is not None:
            on_progress(step_i, total_windows)

        ad_seg = ad[ad_start: ad_start + win_samples].copy()
        ad_seg -= np.mean(ad_seg)
        ad_energy = np.sqrt(np.sum(ad_seg ** 2))
        if ad_energy < 1e-10:
            # Silent window — no candidates
            source_center = (ad_start + win_samples / 2) * sec_per_sample
            lattice.append(CandidateWindow(
                source_center=source_center,
                candidates=[],
                speech_score=0.0,
                energy=0.0,
            ))
            continue

        ad_center_sec = (ad_start + win_samples / 2) * sec_per_sample

        # Search region in video
        nominal_vid_start = ad_start
        search_start = max(0, nominal_vid_start - search_samples)
        search_end = min(len(vid), nominal_vid_start + search_samples + win_samples)

        if search_end - search_start < win_samples:
            lattice.append(CandidateWindow(
                source_center=ad_center_sec,
                candidates=[],
                speech_score=0.0,
                energy=float(ad_energy),
            ))
            continue

        v_region = vid[search_start:search_end].astype(np.float64)
        v_region = v_region - np.mean(v_region)

        # FFT-based sliding cross-correlation
        raw_corr = fftconvolve(v_region, ad_seg[::-1], mode="valid")
        n_pos = len(raw_corr)
        if n_pos == 0:
            lattice.append(CandidateWindow(
                source_center=ad_center_sec,
                candidates=[],
                speech_score=0.0,
                energy=float(ad_energy),
            ))
            continue

        # Per-position energy normalization
        v_sq = v_region ** 2
        cs = np.empty(len(v_sq) + 1, dtype=np.float64)
        cs[0] = 0.0
        np.cumsum(v_sq, out=cs[1:])
        v_norms = np.sqrt(np.maximum(
            cs[win_samples: win_samples + n_pos] - cs[:n_pos], 1e-20,
        ))
        norm_corr = raw_corr / (ad_energy * v_norms)

        mean_corr = float(np.mean(np.abs(norm_corr)))
        peak_indices, _ = find_peaks(norm_corr, distance=min_peak_dist)

        if len(peak_indices) == 0:
            best_idx = int(np.argmax(norm_corr))
            if norm_corr[best_idx] >= min_score:
                peak_indices = np.array([best_idx])
            else:
                lattice.append(CandidateWindow(
                    source_center=ad_center_sec,
                    candidates=[],
                    speech_score=0.0,
                    energy=float(ad_energy),
                ))
                continue

        peak_scores = norm_corr[peak_indices]
        top_order = np.argsort(-peak_scores)[:max_candidates]
        top_indices = peak_indices[top_order]
        top_scores = peak_scores[top_order]

        mask = top_scores >= min_score
        top_indices = top_indices[mask]
        top_scores = top_scores[mask]

        if len(top_indices) == 0:
            lattice.append(CandidateWindow(
                source_center=ad_center_sec,
                candidates=[],
                speech_score=0.0,
                energy=float(ad_energy),
            ))
            continue

        # Build candidates with parabolic refinement and quality metrics
        candidates: list[OffsetCandidate] = []
        best_peak_score = float(top_scores[0])
        second_best = float(top_scores[1]) if len(top_scores) > 1 else 0.0

        for rank, (pidx, pscore) in enumerate(zip(top_indices, top_scores)):
            pidx = int(pidx)
            score = float(pscore)

            # Sub-sample parabolic interpolation
            sub = 0.0
            if 0 < pidx < len(norm_corr) - 1:
                y_prev = float(norm_corr[pidx - 1])
                y_peak = float(norm_corr[pidx])
                y_next = float(norm_corr[pidx + 1])
                denom = y_prev - 2.0 * y_peak + y_next
                if abs(denom) > 1e-12:
                    sub = 0.5 * (y_prev - y_next) / denom
                    score = float(y_peak - 0.25 * (y_prev - y_next) * sub)

            # Convert to offset in seconds
            vid_match_start = search_start + pidx + sub
            vid_center_sec = (vid_match_start + win_samples / 2) * sec_per_sample
            offset_sec = vid_center_sec - ad_center_sec

            # Quality metrics
            sharpness = score / max(mean_corr, 1e-10)
            if rank == 0:
                ratio = score / max(second_best, 1e-10)
                ratio = min(ratio, 10.0)
            else:
                ratio = score / max(best_peak_score, 1e-10)

            candidates.append(OffsetCandidate(
                offset_sec=offset_sec,
                score=score,
                peak_sharpness=sharpness,
                peak_ratio=ratio if rank == 0 else ratio,
            ))

        # Speech-likeness score from mel spectrogram
        speech_score = _compute_speech_score(
            mel, mel_hop_sec, ad_center_sec, window_sec,
            narration_lo, narration_hi,
        )

        lattice.append(CandidateWindow(
            source_center=ad_center_sec,
            candidates=candidates,
            speech_score=speech_score,
            energy=float(ad_energy),
        ))

    log.info(
        "Candidate lattice: %d windows, %d with candidates (avg %.1f candidates/window)",
        len(lattice),
        sum(1 for w in lattice if w.candidates),
        np.mean([len(w.candidates) for w in lattice if w.candidates]) if any(w.candidates for w in lattice) else 0,
    )
    return lattice


def _compute_speech_score(
    mel: NDArray,
    mel_hop_sec: float,
    center_sec: float,
    window_sec: float,
    narration_lo: int,
    narration_hi: int,
) -> float:
    """Estimate speech-likeness of a window from mel spectrogram.

    Combines spectral flatness (low = tonal/speech) with narration-band
    energy ratio (speech concentrates 100-4000 Hz).
    """
    n_frames = mel.shape[1]
    frame_start = max(0, int((center_sec - window_sec / 2) / mel_hop_sec))
    frame_end = min(n_frames, int((center_sec + window_sec / 2) / mel_hop_sec))

    if frame_end <= frame_start:
        return 0.0

    window_mel = mel[:, frame_start:frame_end]

    # Convert from dB back to power for spectral flatness
    # mel might already be in power or dB depending on extract_basic
    # Use the raw values — relative flatness is what matters
    power = np.maximum(window_mel, 1e-10)

    # Spectral flatness per frame: geometric_mean / arithmetic_mean
    log_mean = np.mean(np.log(power + 1e-10), axis=0)
    geo_mean = np.exp(log_mean)
    arith_mean = np.mean(power, axis=0)
    flatness = np.mean(geo_mean / np.maximum(arith_mean, 1e-10))
    flatness = float(np.clip(flatness, 0.0, 1.0))

    # Narration-band energy ratio
    total_energy = np.sum(power)
    if total_energy < 1e-10:
        return 0.0
    narration_energy = np.sum(power[narration_lo:narration_hi, :])
    narration_ratio = float(narration_energy / total_energy)

    # Combine: low flatness = speech-like, high narration ratio = speech-like
    speech_score = 0.5 * (1.0 - flatness) + 0.5 * narration_ratio
    return float(np.clip(speech_score, 0.0, 1.0))
