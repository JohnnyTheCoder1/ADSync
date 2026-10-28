"""Content-cached analysis audio, shared by alignment and the season content check."""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np

from adsync.config import SyncConfig
from adsync.models import MediaInfo


def analysis_stream(info: MediaInfo, config: SyncConfig) -> int | None:
    """The video audio stream alignment analyzes, or None for the only stream."""
    if not (config.prepare_audio or len(info.audio_streams) > 1):
        return None
    from adsync.media.prep import pick_audio_stream

    return pick_audio_stream(info, language=config.ad_language, strict=config.prepare_audio).index


def analysis_cache_options(stream: int | None, config: SyncConfig) -> dict:
    return {
        "kind": "preprocessed-audio", "stream": stream,
        "sr": config.analysis_sr, "mono": config.mono,
        "highpass_hz": config.highpass_hz, "trim_silence": False,
    }


def load_analysis_audio(path: Path, info: MediaInfo, stream: int | None, config: SyncConfig,
                        *, threads: int = 1) -> np.ndarray:
    """Return preprocessed analysis audio, decoding only on a cache miss."""
    from adsync.cache import content_key, default_cache
    from adsync.features.load import load_wav
    from adsync.features.preprocess import preprocess
    from adsync.media.extract import extract_audio

    cache = default_cache()
    options = analysis_cache_options(stream, config)
    key = content_key(path, options) if cache is not None else None
    if key is not None:
        artifact = cache.get(key)
        if artifact is not None and set(artifact) == {"audio"} and artifact["audio"].ndim == 1:
            return artifact["audio"]
    with tempfile.TemporaryDirectory(prefix="adsync-analysis-") as work:
        wav = Path(work) / "audio.wav"
        extract_audio(info, wav, sr=config.analysis_sr, mono=config.mono, stream_index=stream, threads=threads)
        y, _ = load_wav(wav, sr=config.analysis_sr)
    audio = preprocess(y, config.analysis_sr, highpass_hz=config.highpass_hz, trim_silence=False)
    if key is not None:
        if content_key(path, options) != key:
            raise RuntimeError(f"Source changed during analysis decode: {path}")
        cache.put(key, {"audio": audio})
    return audio
