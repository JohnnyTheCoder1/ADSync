"""Audio segment cut / shift / concat operations via FFmpeg."""

from __future__ import annotations

import logging
from pathlib import Path

from adsync.utils.subprocesses import run_ffmpeg

log = logging.getLogger("adsync")


def cut_segment(
    input_path: Path,
    output_path: Path,
    *,
    start: float,
    end: float,
    sr: int | None = None,
) -> Path:
    """Extract a time range from an audio file."""
    args = [
        "-i", str(input_path),
        "-ss", f"{start:.6f}",
        "-to", f"{end:.6f}",
        "-c:a", "pcm_s16le",
    ]
    if sr:
        args += ["-ar", str(sr)]
    args.append(str(output_path))
    run_ffmpeg(args)
    return output_path


def generate_silence(output_path: Path, *, duration: float, sr: int = 16000, channels: int = 1) -> Path:
    """Create a silent WAV file of the given duration."""
    args = [
        "-f", "lavfi",
        "-i", f"anullsrc=r={sr}:cl={'mono' if channels == 1 else 'stereo'}",
        "-t", f"{duration:.6f}",
        "-c:a", "pcm_s16le",
        str(output_path),
    ]
    run_ffmpeg(args)
    return output_path


def concat_files(file_list: list[Path], output_path: Path) -> Path:
    """Concatenate audio files using the FFmpeg concat demuxer."""
    list_path = output_path.with_suffix(".txt")
    lines = [f"file '{f.as_posix()}'" for f in file_list]
    list_path.write_text("\n".join(lines), encoding="utf-8")

    args = [
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_path),
        "-c:a", "pcm_s16le",
        str(output_path),
    ]
    run_ffmpeg(args)
    list_path.unlink(missing_ok=True)
    return output_path
