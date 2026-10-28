"""Audio decodes straight from ffmpeg into memory, sample-identical to the old WAV route."""

import subprocess

import librosa
import numpy as np
import pytest

from adsync.media.extract import decode_audio
from adsync.media.probe import probe
from adsync.models import MediaInfo, StreamInfo
from adsync.utils.subprocesses import FFmpegError, _find_binary


def _ffmpeg(*args):
    subprocess.run(
        [_find_binary("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *map(str, args)],
        check=True, capture_output=True, stdin=subprocess.DEVNULL,
    )


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    folder = tmp_path_factory.mktemp("decode")
    stereo = folder / "stereo.flac"
    _ffmpeg("-f", "lavfi", "-i",
            "aevalsrc=0.3*sin(2*PI*440*t)+0.1*sin(2*PI*3001*t)|0.25*sin(2*PI*660*t):s=44100:d=3",
            "-c:a", "flac", stereo)
    surround = folder / "surround.flac"
    tones = "|".join(f"0.1*sin(2*PI*{f}*t)" for f in (310, 410, 510, 610, 710, 810))
    _ffmpeg("-f", "lavfi", "-i", f"aevalsrc={tones}:s=48000:d=3:c=5.1", "-c:a", "flac", surround)
    return {"stereo": stereo, "surround": surround}


def _wav_route(path, wav, *, sr, mono, sample_fmt, speed_ratio=None):
    # The pre-change route: ffmpeg writes a WAV file and librosa reads it back.
    filters = []
    if speed_ratio:
        filters = ["-af", f"aresample={sr},asetrate={round(sr / speed_ratio)},aresample={sr}:filter_size=256:cutoff=0.985"]
    codec = {"s16": "pcm_s16le", "f32": "pcm_f32le"}[sample_fmt]
    _ffmpeg("-i", path, "-map", "0:0", *filters, "-ar", sr, *(["-ac", "1"] if mono else []),
            "-c:a", codec, "-f", "wav", wav)
    return librosa.load(str(wav), sr=sr, mono=mono)[0]


@pytest.mark.parametrize("name, sr, mono, sample_fmt, speed_ratio", [
    ("stereo", 16000, True, "s16", None),
    ("surround", 48000, False, "f32", None),
    ("stereo", 44100, False, "f32", 25 / 24),
])
def test_decode_matches_wav_route_sample_for_sample(sources, tmp_path, name, sr, mono, sample_fmt, speed_ratio):
    expected = _wav_route(sources[name], tmp_path / "old.wav", sr=sr, mono=mono,
                          sample_fmt=sample_fmt, speed_ratio=speed_ratio)
    decoded = decode_audio(probe(sources[name]), sr=sr, mono=mono, sample_fmt=sample_fmt, speed_ratio=speed_ratio)
    assert decoded.dtype == np.float32
    assert decoded.shape == expected.shape
    assert np.array_equal(decoded, expected)


def test_decode_reads_past_an_underestimated_duration(sources, tmp_path):
    expected = _wav_route(sources["stereo"], tmp_path / "old.wav", sr=16000, mono=True, sample_fmt="s16")
    stream = StreamInfo(index=0, codec_type="audio", channels=2, sample_rate=44100, duration=0.5)
    info = MediaInfo(path=str(sources["stereo"]), duration=0.5, audio_streams=[stream])
    assert np.array_equal(decode_audio(info, sr=16000, mono=True), expected)


def test_decode_failure_raises_ffmpeg_error(tmp_path):
    broken = tmp_path / "broken.mkv"
    broken.write_bytes(b"not a media file " * 64)
    stream = StreamInfo(index=0, codec_type="audio", channels=2, sample_rate=48000)
    with pytest.raises(FFmpegError):
        decode_audio(MediaInfo(path=str(broken), duration=1.0, audio_streams=[stream]))
