"""Decode audio from media files for analysis and rendering."""

from __future__ import annotations

import logging
import struct
import subprocess
import threading
from pathlib import Path
from typing import BinaryIO

import numpy as np
from numpy.typing import NDArray

from adsync.models import MediaInfo, StreamInfo
from adsync.utils.subprocesses import NO_WINDOW, FFmpegError, ffmpeg_command

log = logging.getLogger("adsync")

_SAMPLE_FORMATS = {"s16": ("pcm_s16le", np.dtype("<i2")), "f32": ("pcm_f32le", np.dtype("<f4"))}
_UNKNOWN_DURATION_SEC = 600.0  # initial buffer when the probe reports no duration
_READ_CHUNK = 4 << 20


def decode_audio(
    info: MediaInfo,
    *,
    stream_index: int | None = None,
    sr: int = 16000,
    mono: bool = True,
    speed_ratio: float | None = None,
    sample_fmt: str = "s16",
    threads: int | None = None,
) -> NDArray[np.float32]:
    """Decode one audio stream straight into memory, with no intermediate file.

    The result follows the container's playback timeline: a start delay
    becomes leading silence and timestamp gaps are filled, as a player would.
    Returns ``(frames,)`` for single-channel output, else ``(channels, frames)``.
    ``s16`` keeps the analysis quantization; ``f32`` is for rebuild sources.
    *speed_ratio* stretches by the same exact fraction the analysis uses.
    """
    if sample_fmt not in _SAMPLE_FORMATS:
        raise ValueError(f"Unsupported sample_fmt {sample_fmt!r} (use 's16' or 'f32')")
    stream = _select_stream(info, stream_index)
    codec, dtype = _SAMPLE_FORMATS[sample_fmt]
    args = [
        "-v", "error", "-i", info.path, "-map", f"0:{stream.index}", "-af", "aresample=async=1",
        "-ar", str(sr), *(["-ac", "1"] if mono else []), "-c:a", codec, "-f", "wav", "pipe:1",
    ]
    proc = subprocess.Popen(
        ffmpeg_command(args, threads=threads),
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **NO_WINDOW,
    )
    errors: list[bytes] = []
    drain = threading.Thread(target=lambda: errors.append(proc.stderr.read()), daemon=True)
    drain.start()
    try:
        channels = _read_wav_header(proc.stdout)
        frame_bytes = (channels or 1) * dtype.itemsize
        lead = max(0.0, (stream.start_time or 0.0) - (info.start_time or 0.0))
        seconds = stream.duration or info.duration or _UNKNOWN_DURATION_SEC
        frames_hint = int((seconds + 1.0) * sr)
        raw = _read_to_end(proc.stdout, frames_hint * frame_bytes, lead_bytes=round(lead * sr) * frame_bytes)
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        drain.join()
        proc.stdout.close()
        proc.stderr.close()
    if proc.returncode != 0:
        raise FFmpegError(proc.returncode, b"".join(errors))
    if not channels:
        raise ValueError(f"ffmpeg decoded no audio from {info.path}")

    samples = raw[: raw.size - raw.size % (channels * dtype.itemsize)].view(dtype)
    if sample_fmt == "s16":
        samples = samples.astype(np.float32)
        samples *= np.float32(1 / 32768)  # libsndfile's int16 scaling, so results match a WAV read
    if channels > 1:
        samples = samples.reshape(-1, channels).T
    if speed_ratio is not None and abs(speed_ratio - 1.0) > 1e-9:
        samples = _stretch(samples, speed_ratio)
    log.info(
        "Decoded %s stream %d: %d Hz, %d ch%s",
        Path(info.path).name, stream.index, sr, channels,
        f", speed x{speed_ratio:.6f}" if speed_ratio else "",
    )
    return samples


def _select_stream(info: MediaInfo, stream_index: int | None) -> StreamInfo:
    if not info.audio_streams:
        raise ValueError(f"No audio streams found in {info.path}")
    if stream_index is None:
        return info.audio_streams[0]
    for stream in info.audio_streams:
        if stream.index == stream_index:
            return stream
    raise ValueError(f"No audio stream {stream_index} in {info.path}")


def _stretch(y: NDArray[np.float32], stretch: float) -> NDArray[np.float32]:
    """Resample by exactly the analysis ratio, so playback time scales by *stretch* and pitch follows."""
    import soxr

    from adsync.align.speed import stretch_fraction

    up, down = stretch_fraction(stretch)
    out = soxr.resample(y.T if y.ndim == 2 else y, down, up, quality="VHQ")
    return out.T if y.ndim == 2 else out


def _read_wav_header(stream: BinaryIO) -> int | None:
    """Consume a streamed WAV header up to the sample data; return its channel count."""
    riff = stream.read(12)
    if len(riff) < 12:
        return None
    if riff[:4] not in (b"RIFF", b"RF64") or riff[8:12] != b"WAVE":
        raise ValueError("ffmpeg did not produce a WAV stream")
    channels = None
    while True:
        head = stream.read(8)
        if len(head) < 8:
            return None
        chunk, size = head[:4], struct.unpack("<I", head[4:])[0]
        if chunk == b"data":
            return channels
        body = stream.read(size + (size & 1))
        if chunk == b"fmt ":
            channels = struct.unpack("<H", body[2:4])[0]


def _read_to_end(stream: BinaryIO, size_hint: int, *, lead_bytes: int = 0) -> NDArray[np.uint8]:
    """Read a pipe to EOF into one buffer, after ``lead_bytes`` of zeros."""
    buf = np.empty(max(size_hint + lead_bytes, 1 << 16), dtype=np.uint8)
    buf[:lead_bytes] = 0
    filled = lead_bytes
    while True:
        if filled == buf.size:
            grown = np.empty(buf.size + buf.size // 2, dtype=np.uint8)
            grown[:filled] = buf
            buf = grown
        # Bounded reads: Windows pipe reads cost time in proportion to the requested size, not the bytes returned.
        count = stream.readinto(memoryview(buf)[filled:filled + _READ_CHUNK])
        if not count:
            break
        filled += count
    # Release over-allocation left by a short duration estimate.
    return buf[:filled] if filled >= 0.9 * buf.size else buf[:filled].copy()
