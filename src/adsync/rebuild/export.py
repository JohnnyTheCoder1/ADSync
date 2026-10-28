"""Export synced AD audio to file."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from adsync.utils.subprocesses import run_ffmpeg

log = logging.getLogger("adsync")

# Sample bytes a RIFF header can describe, minus room for the header chunks.
_RIFF_DATA_LIMIT = 0xFFFFFFFF - 65536
_SAMPLE_BYTES = {"PCM_16": 2, "PCM_24": 3, "PCM_32": 4, "FLOAT": 4, "DOUBLE": 8}


def export_wav(y: NDArray[np.floating], sr: int, path: Path, *, subtype: str = "PCM_16") -> Path:
    """Write a waveform to WAV, switching to RF64 when it would not fit a RIFF header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # soundfile expects (frames, channels); we store (channels, frames).
    data = y.T if y.ndim == 2 else y
    container = "RF64" if data.size * _SAMPLE_BYTES[subtype] > _RIFF_DATA_LIMIT else "WAV"
    sf.write(str(path), data, sr, subtype=subtype, format=container)
    log.info("Exported WAV: %s (%.1f s)", path.name, y.shape[-1] / sr)
    return path


def export_aac(wav_path: Path, output_path: Path, *, bitrate: str = "192k") -> Path:
    """Encode a WAV to AAC in an M4A container."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "-i", str(wav_path),
        "-c:a", "aac",
        "-b:a", bitrate,
        str(output_path),
    ]
    run_ffmpeg(args)
    log.info("Exported AAC: %s", output_path.name)
    return output_path
