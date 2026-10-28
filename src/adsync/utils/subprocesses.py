"""Subprocess helpers for running FFmpeg / ffprobe."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

from adsync.hardware import effective_thread_count

log = logging.getLogger("adsync")

# Tool processes never get a console window, even when the host has none (pythonw, GUIs).
NO_WINDOW = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def _find_binary(name: str) -> str:
    """Locate *name* on PATH or raise."""
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(
            f"{name} not found on PATH. Install FFmpeg and ensure it is accessible."
        )
    return path


def check_dependencies() -> None:
    """Verify that ffmpeg and ffprobe are available on PATH.

    Raises :class:`FileNotFoundError` with a user-friendly message if either
    binary is missing.
    """
    missing: list[str] = []
    for name in ("ffmpeg", "ffprobe"):
        if shutil.which(name) is None:
            missing.append(name)
    if missing:
        names = " and ".join(missing)
        raise FileNotFoundError(
            f"{names} not found on PATH. "
            "Install FFmpeg (https://ffmpeg.org) and ensure it is accessible."
        )


class FFmpegError(RuntimeError):
    """User-friendly wrapper around FFmpeg failures."""

    def __init__(self, returncode: int, stderr: str | bytes | None) -> None:
        self.returncode = returncode
        self.stderr = decode_output(stderr)
        summary = _parse_ffmpeg_error(self.stderr)
        super().__init__(f"FFmpeg failed (exit {returncode}): {summary}")


def decode_output(data: str | bytes | None) -> str:
    """Tool output as text: ffmpeg and ffprobe write UTF-8 whatever the console code page is."""
    if data is None:
        return ""
    if isinstance(data, str):
        return data
    return data.decode("utf-8", errors="replace")


def ffmpeg_prefix() -> list[str]:
    """The ffmpeg binary plus the flags every call needs; ffmpeg must never read the console."""
    return [_find_binary("ffmpeg"), "-nostdin", "-hide_banner"]


def _parse_ffmpeg_error(stderr: str) -> str:
    """Extract the most relevant error line from FFmpeg's stderr."""
    for line in reversed(stderr.strip().splitlines()):
        lower = line.lower()
        if any(kw in lower for kw in ("error", "invalid", "no such", "unknown", "not found")):
            return line.strip()
    # Fall back to last non-empty line
    lines = [l.strip() for l in stderr.strip().splitlines() if l.strip()]
    return lines[-1] if lines else "(no output)"


def _threaded_args(args: list[str], threads: int | None = None) -> list[str]:
    """Cap each input decoder, the output encoder, and filter thread pools.

    These helpers accept one output, at the final argument. FFmpeg's codec
    options are positional, so an output-only -threads flag cannot cap inputs.
    """
    count = str(effective_thread_count(threads))
    result = ["-filter_threads", count, "-filter_complex_threads", count]
    for arg in args[:-1]:
        if arg == "-i":
            result.extend(["-threads", count])
        result.append(arg)
    if args:
        result.extend(["-threads", count, args[-1]])
    return result


def run_ffmpeg(args: list[str], *, check: bool = True, threads: int | None = None) -> subprocess.CompletedProcess[str]:
    """Run an ffmpeg command and return the result.

    Raises :class:`FFmpegError` with a parsed message on failure.
    """
    cmd = [*ffmpeg_prefix(), "-nostats", "-y", *_threaded_args(args, threads)]
    log.debug("ffmpeg %s", " ".join(args))
    raw = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, check=False, **NO_WINDOW)
    result = subprocess.CompletedProcess(cmd, raw.returncode, decode_output(raw.stdout), decode_output(raw.stderr))
    if check and result.returncode != 0:
        raise FFmpegError(result.returncode, result.stderr)
    return result


def run_ffmpeg_streamed(
    args: list[str],
    *,
    check: bool = True,
    threads: int | None = None,
) -> subprocess.Popen[bytes]:
    """Start an ffmpeg process with stdin open for streaming writes.

    Returns the Popen object so the caller can write chunks to ``proc.stdin``
    and call ``proc.stdin.close()`` + ``proc.wait()`` when done.
    """
    cmd = [*ffmpeg_prefix(), "-nostats", "-y", *_threaded_args(args, threads)]
    log.debug("ffmpeg %s", " ".join(args))
    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        **NO_WINDOW,
    )


def run_ffprobe(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run an ffprobe command; raises CalledProcessError with decoded output on failure."""
    cmd = [_find_binary("ffprobe"), "-hide_banner", *args]
    log.debug("ffprobe %s", " ".join(args))
    raw = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, check=False, **NO_WINDOW)
    stdout, stderr = decode_output(raw.stdout), decode_output(raw.stderr)
    if raw.returncode != 0:
        raise subprocess.CalledProcessError(raw.returncode, cmd, stdout, stderr)
    return subprocess.CompletedProcess(cmd, raw.returncode, stdout, stderr)
