"""Extract audio from media files to analysis WAVs."""

from __future__ import annotations

import logging
from pathlib import Path

from adsync.models import MediaInfo
from adsync.utils.subprocesses import run_ffmpeg

log = logging.getLogger("adsync")


def extract_audio(
    info: MediaInfo,
    output_path: Path,
    *,
    stream_index: int | None = None,
    sr: int = 16000,
    mono: bool = True,
) -> Path:
    """Extract an audio stream to a standardized WAV file.

    Parameters
    ----------
    info:
        Probed media info.
    output_path:
        Destination WAV path.
    stream_index:
        Audio stream index to extract. If *None* the first audio stream is used.
    sr:
        Target sample rate.
    mono:
        Convert to mono.
    """
    if not info.audio_streams:
        raise ValueError(f"No audio streams found in {info.path}")

    if stream_index is None:
        stream_index = info.audio_streams[0].index

    channels_args = ["-ac", "1"] if mono else []

    args = [
        "-i", info.path,
        "-map", f"0:{stream_index}",
        "-ar", str(sr),
        *channels_args,
        "-c:a", "pcm_s16le",
        "-f", "wav",
        str(output_path),
    ]

    run_ffmpeg(args)
    log.info("Extracted audio → %s (%d Hz, %s)", output_path.name, sr, "mono" if mono else "stereo")
    return output_path
