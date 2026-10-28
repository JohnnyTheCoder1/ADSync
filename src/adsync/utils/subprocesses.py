"""Subprocess helpers for running FFmpeg / ffprobe."""

from __future__ import annotations

import ctypes
import logging
import os
import shutil
import subprocess
from pathlib import Path

import psutil

from adsync.hardware import effective_thread_count

log = logging.getLogger("adsync")


def _has_console() -> bool:
    return os.name != "nt" or bool(ctypes.windll.kernel32.GetConsoleWindow())


def no_window() -> dict:
    """Popen options: share an existing console so Ctrl+C reaches ffmpeg; otherwise open no window."""
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" and not _has_console() else {}


def kill_tree(proc: subprocess.Popen) -> None:
    """Kill *proc* and its descendants; package-manager shims run the real ffmpeg as a child."""
    try:
        family = psutil.Process(proc.pid).children(recursive=True)
    except psutil.Error:
        family = []
    proc.kill()
    for member in family:
        try:
            member.kill()
        except psutil.Error:
            pass


def run_tool(cmd: list[str], *, timeout: float | None = None) -> subprocess.CompletedProcess[bytes]:
    """Run a tool with captured output; on any interruption the whole process tree is stopped."""
    with subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, **no_window()) as proc:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except BaseException:
            kill_tree(proc)
            raise
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


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


def ffmpeg_command(args: list[str], *, threads: int | None = None) -> list[str]:
    """A complete single-output ffmpeg command with the run's thread budget applied."""
    return [*ffmpeg_prefix(), "-nostats", "-y", *_threaded_args(args, threads)]


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
    cmd = ffmpeg_command(args, threads=threads)
    log.debug("ffmpeg %s", " ".join(args))
    raw = run_tool(cmd)
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
    cmd = ffmpeg_command(args, threads=threads)
    log.debug("ffmpeg %s", " ".join(args))
    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        **no_window(),
    )


def run_ffprobe(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run an ffprobe command; raises CalledProcessError with decoded output on failure."""
    cmd = [_find_binary("ffprobe"), "-hide_banner", *args]
    log.debug("ffprobe %s", " ".join(args))
    raw = run_tool(cmd)
    stdout, stderr = decode_output(raw.stdout), decode_output(raw.stderr)
    if raw.returncode != 0:
        raise subprocess.CalledProcessError(raw.returncode, cmd, stdout, stderr)
    return subprocess.CompletedProcess(cmd, raw.returncode, stdout, stderr)
