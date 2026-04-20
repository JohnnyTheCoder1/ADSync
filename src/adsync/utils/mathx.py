"""Small math / DSP utilities that don't belong in a larger module."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def rms_normalize(y: NDArray[np.floating], target_db: float = -20.0) -> NDArray[np.floating]:
    """Normalize *y* so its RMS equals *target_db* dBFS."""
    rms = np.sqrt(np.mean(y ** 2))
    if rms < 1e-10:
        return y
    target_linear = 10 ** (target_db / 20.0)
    return y * (target_linear / rms)


def crossfade(a: NDArray[np.floating], b: NDArray[np.floating], n_samples: int) -> NDArray[np.floating]:
    """Overlap-add crossfade of length *n_samples* between *a* (end) and *b* (start)."""
    if n_samples <= 0:
        return np.concatenate([a, b])
    n_samples = min(n_samples, len(a), len(b))
    fade_out = np.linspace(1.0, 0.0, n_samples)
    fade_in = np.linspace(0.0, 1.0, n_samples)

    result = np.empty(len(a) + len(b) - n_samples, dtype=a.dtype)
    result[: len(a) - n_samples] = a[: -n_samples]
    result[len(a) - n_samples : len(a)] = a[-n_samples:] * fade_out + b[:n_samples] * fade_in
    result[len(a):] = b[n_samples:]
    return result


def linear_fit_offset(times: NDArray[np.floating], offsets: NDArray[np.floating]) -> tuple[float, float]:
    """Return (intercept, slope) for offset-vs-time via least-squares."""
    if len(times) < 2:
        return (float(offsets[0]) if len(offsets) else 0.0, 0.0)
    coeffs = np.polyfit(times, offsets, 1)  # slope, intercept
    return float(coeffs[1]), float(coeffs[0])
