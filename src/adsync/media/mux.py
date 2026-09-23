"""Mux streams into the final container."""

from __future__ import annotations

import logging
import json
import subprocess
import threading
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from adsync.media.output import staged_output
from adsync.models import StreamInfo
from adsync.utils.progress import Progress
from adsync.utils.subprocesses import run_ffmpeg_streamed

# ~2 MB per chunk at 4 bytes / sample → constant memory overhead
_CHUNK_SAMPLES = 512 * 1024

log = logging.getLogger("adsync")


def mux_ad_track(
    video_path: str | Path,
    synced_y: NDArray[np.floating],
    sr: int,
    output_path: str | Path,
    *,
    codec: str = "libopus",
    bitrate: str = "96k",
    language: str = "eng",
    title: str = "Audio Description",
    n_existing_audio: int | None = None,
    threads: int | None = None,
    original_audio: StreamInfo | None = None,
) -> Path:
    """Render an AD mux locally, then atomically publish the completed MKV.

    With *original_audio*, keep that main audio stream only and downmix it when
    multichannel during the same FFmpeg pass that encodes the synced AD.
    """
    output_path = Path(output_path)
    with staged_output(output_path, inputs=[video_path]) as local_path:
        _mux_ad_track(
            video_path, synced_y, sr, local_path,
            codec=codec, bitrate=bitrate, language=language, title=title,
            n_existing_audio=n_existing_audio, threads=threads,
            original_audio=original_audio,
        )
    log.info("Muxed output → %s", output_path)
    return output_path


def _mux_ad_track(
    video_path: str | Path,
    synced_y: NDArray[np.floating],
    sr: int,
    output_path: str | Path,
    *,
    codec: str = "libopus",
    bitrate: str = "96k",
    language: str = "eng",
    title: str = "Audio Description",
    n_existing_audio: int | None = None,
    threads: int | None = None,
    original_audio: StreamInfo | None = None,
) -> Path:
    """Mux original video with AD audio encoded directly into the MKV.

    Streams raw PCM in chunks to FFmpeg via stdin — never materialises the
    full byte array.  Channel count is inferred from *synced_y* (1-D = mono,
    2-D = (channels, samples)).  Uses Opus by default.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if synced_y.ndim == 1:
        n_channels = 1
        total_samples = len(synced_y)
    else:
        n_channels, total_samples = synced_y.shape

    attachment_inputs, attachment_outputs = _attachment_options(video_path, output_path.parent)
    main_audio_args: list[str] = []
    mapping = ["-map", "0"]
    if attachment_inputs:
        mapping += ["-map", "-0:t"]
    if original_audio is not None:
        if original_audio.codec_type != "audio" or original_audio.index < 0:
            raise ValueError("Selected original audio must identify an audio stream")
        # Keep all original non-audio streams, then append the selected main
        # stream as a:0 and the new AD as a:1. Absolute source indexes are used.
        mapping += ["-map", "-0:a", "-map", f"0:{original_audio.index}"]
        n_audio = 1
        main_audio_args = _prepared_main_audio_args(original_audio, language)
    else:
        n_audio = n_existing_audio if n_existing_audio is not None else _count_audio_streams(video_path)

    args = [
        *attachment_inputs,
        "-i", str(video_path),
        "-f", "f32le", "-ar", str(sr), "-ac", str(n_channels), "-i", "pipe:0",
        *mapping,
        "-map", "1:a:0",
        "-c", "copy",
        *main_audio_args,
        f"-c:a:{n_audio}", codec,
        f"-b:a:{n_audio}", bitrate,
        f"-metadata:s:a:{n_audio}", f"language={language}",
        f"-metadata:s:a:{n_audio}", f"title={title}",
        *attachment_outputs,
        str(output_path),
    ]

    proc = run_ffmpeg_streamed(args, threads=threads)
    assert proc.stdin is not None

    stderr_buf: list[bytes] = []
    def _drain_stderr() -> None:
        if proc.stderr:
            stderr_buf.append(proc.stderr.read())
    drain = threading.Thread(target=_drain_stderr, daemon=True)
    drain.start()

    total_chunks = (total_samples + _CHUNK_SAMPLES - 1) // _CHUNK_SAMPLES

    stream_error: OSError | None = None
    try:
        try:
            with Progress() as progress:
                task = progress.add_task("Encoding AD track", total=total_chunks)
                for start in range(0, total_samples, _CHUNK_SAMPLES):
                    chunk = synced_y[..., start : start + _CHUNK_SAMPLES]
                    # Float straight through to the encoder — no 16-bit quantization
                    # stage. The renderers guarantee peak <= 0.99; clip is a belt.
                    pcm = np.clip(chunk, -1.0, 1.0).astype(np.float32, copy=False)
                    if pcm.ndim == 2:
                        pcm = np.ascontiguousarray(pcm.T)
                    proc.stdin.write(pcm.tobytes())
                    progress.advance(task)
        except OSError as exc:
            # The pipe error itself says nothing; surface FFmpeg's stderr.
            stream_error = exc
        except BaseException:
            # Stop before closing buffered stdin so flushing cannot hang after
            # cancellation or a conversion error while FFmpeg has stopped reading.
            if proc.poll() is None:
                proc.kill()
            raise
        finally:
            try:
                proc.stdin.close()
            except OSError:
                pass
        proc.wait()
    finally:
        # Reap the encoder before staged_output attempts to remove its file.
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        drain.join()
        if proc.stderr is not None:
            proc.stderr.close()

    if proc.returncode != 0 or stream_error is not None:
        stderr = b"".join(stderr_buf)
        raise subprocess.CalledProcessError(
            proc.returncode if proc.returncode != 0 else 1, "ffmpeg", stderr=stderr,
        )

    return output_path


def _prepared_main_audio_args(stream: StreamInfo, language: str) -> list[str]:
    """Use prep's exact matrix/limiter, scoped only to output main audio a:0."""
    from adsync.media.prep import _LIMITER, build_downmix_pan

    lang_tag = (stream.language or language or "und")[:3]
    args = ["-metadata:s:a:0", f"language={lang_tag}", "-disposition:a:0", "default"]
    if (stream.channels or 0) <= 2:
        return args  # The surrounding -c copy preserves stereo/mono bytes.
    pan = build_downmix_pan(stream.channel_layout)
    if pan is None:
        log.warning("Unknown channel layout %r; using FFmpeg's default stereo downmix", stream.channel_layout)
        args += ["-ac:a:0", "2"]
    args += [
        "-filter:a:0", ",".join(filter(None, [pan, _LIMITER])),
        "-c:a:0", "libopus", "-b:a:0", "192k", "-ar:a:0", "48000",
        "-metadata:s:a:0", f"title={'English Stereo' if lang_tag.startswith('en') else 'Stereo Downmix'}",
    ]
    return args


def _attachment_options(video_path: Path, stage_dir: Path) -> tuple[list[str], list[str]]:
    """Preserve attachment bytes without mapping attachment packets into mux.

    Some FFmpeg versions submit invalid attachment packets when an MKV and raw
    PCM input are combined. Dumping the extradata while opening the same input,
    then attaching those local bytes, avoids another read/write of the video.
    Paths are generated locally; embedded filenames are used only as metadata.
    """
    from adsync.utils.subprocesses import run_ffprobe

    details = json.loads(run_ffprobe([
        "-v", "error", "-select_streams", "t", "-show_entries",
        "stream=index:stream_tags", "-of", "json", str(video_path),
    ]).stdout)
    inputs: list[str] = []
    outputs: list[str] = []
    for number, stream in enumerate(details.get("streams", [])):
        local_path = stage_dir / f"attachment-{number:04d}.bin"
        inputs += [f"-dump_attachment:{stream['index']}", str(local_path)]
        outputs += ["-attach", str(local_path)]
        tags = dict(stream.get("tags", {}))
        if not any(key.casefold() == "filename" for key in tags):
            tags["filename"] = f"attachment-{number:04d}.bin"
        if not any(key.casefold() == "mimetype" for key in tags):
            tags["mimetype"] = "application/octet-stream"
        for key, value in tags.items():
            outputs += [f"-metadata:s:t:{number}", f"{key}={value}"]
    return inputs, outputs


def mux_ad_file(
    video_path: str | Path,
    ad_path: str | Path,
    output_path: str | Path,
    *,
    codec: str = "libopus",
    bitrate: str = "96k",
    language: str = "eng",
    title: str = "Audio Description",
) -> Path:
    """Mux a ready AD file locally and publish only the completed container."""
    output_path = Path(output_path)
    with staged_output(output_path, inputs=[video_path, ad_path]) as local_path:
        _mux_ad_file(
            video_path, ad_path, local_path,
            codec=codec, bitrate=bitrate, language=language, title=title,
        )
    log.info("Muxed output → %s", output_path)
    return output_path


def _mux_ad_file(
    video_path: str | Path,
    ad_path: str | Path,
    output_path: str | Path,
    *,
    codec: str = "libopus",
    bitrate: str = "96k",
    language: str = "eng",
    title: str = "Audio Description",
) -> Path:
    """Mux a pre-synced AD audio *file* into the video container.

    Unlike :func:`mux_ad_track` (which streams PCM from memory), this takes
    an on-disk audio file as the second input — intended for the ``adsync mux``
    CLI command where the user supplies an already-synced file.
    """
    from adsync.utils.subprocesses import run_ffmpeg

    video_path = Path(video_path)
    ad_path = Path(ad_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_audio = _count_audio_streams(video_path)
    attachment_inputs, attachment_outputs = _attachment_options(video_path, output_path.parent)

    args = [
        *attachment_inputs,
        "-i", str(video_path),
        "-i", str(ad_path),
        "-map", "0",
        *(["-map", "-0:t"] if attachment_inputs else []),
        "-map", "1:a:0",
        "-c", "copy",
        f"-c:a:{n_audio}", codec,
        f"-b:a:{n_audio}", bitrate,
        f"-metadata:s:a:{n_audio}", f"language={language}",
        f"-metadata:s:a:{n_audio}", f"title={title}",
        *attachment_outputs,
        str(output_path),
    ]

    run_ffmpeg(args)
    return output_path


def _count_audio_streams(video_path: Path) -> int:
    """Quick helper to count audio streams in the original file."""
    from adsync.media.probe import probe
    info = probe(video_path)
    return len(info.audio_streams)
