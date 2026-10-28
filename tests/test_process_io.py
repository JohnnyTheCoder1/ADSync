"""FFmpeg and ffprobe calls survive non-ASCII metadata and never read the console."""

import json
import locale
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

from adsync.media.mux import mux_ad_track
from adsync.media.probe import probe
from adsync.models import MediaInfo, StreamInfo
from adsync.utils.subprocesses import FFmpegError, _find_binary, run_ffmpeg

# Each holds UTF-8 bytes (0x81, 0x8D, 0x90) that Windows-1252 cannot decode.
STREAM_TITLE = "Ōkami — ō Łódź сериал バ"
FORMAT_TITLE = "Сезон 1 — エピソード"
ATTACHMENT_NAME = "フォント-ō.ttf"


def _ffmpeg(*args):
    subprocess.run(
        [_find_binary("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *map(str, args)],
        check=True, capture_output=True, stdin=subprocess.DEVNULL,
    )


@pytest.fixture
def default_windows_locale(monkeypatch):
    # Text-mode subprocess pipes decode with locale.getencoding(); most Windows installs use cp1252.
    monkeypatch.setattr(locale, "getencoding", lambda: "cp1252")


@pytest.fixture(scope="module")
def titled_media(tmp_path_factory):
    folder = tmp_path_factory.mktemp("non-ascii")
    font = folder / "font.ttf"
    font.write_bytes(b"fixture font")
    source = folder / "source.mkv"
    _ffmpeg(
        "-f", "lavfi", "-i", "color=c=black:s=32x32:r=10:d=1",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1",
        "-map", "0:v", "-map", "1:a", "-c:v", "ffv1", "-c:a", "flac",
        "-metadata", f"title={FORMAT_TITLE}", "-metadata:s:a:0", f"title={STREAM_TITLE}",
        "-attach", font, "-metadata:s:t:0", "mimetype=application/x-truetype-font",
        "-metadata:s:t:0", f"filename={ATTACHMENT_NAME}",
        source,
    )
    return source


def test_probe_keeps_non_ascii_titles(titled_media, default_windows_locale):
    assert probe(titled_media).audio_streams[0].title == STREAM_TITLE


def test_mux_keeps_non_ascii_attachment_names(titled_media, tmp_path, default_windows_locale):
    output = tmp_path / "out.mkv"
    mux_ad_track(titled_media, np.zeros(48000, dtype=np.float32), 48000, output)
    raw = subprocess.run(
        [_find_binary("ffprobe"), "-v", "error", "-select_streams", "t",
         "-show_entries", "stream_tags", "-of", "json", str(output)],
        check=True, capture_output=True, stdin=subprocess.DEVNULL,
    ).stdout
    assert json.loads(raw.decode("utf-8"))["streams"][0]["tags"]["filename"] == ATTACHMENT_NAME


def test_ffmpeg_failure_with_non_ascii_path_raises_readable_error(tmp_path, default_windows_locale):
    missing = tmp_path / "mangler-ō-сериал.mkv"
    with pytest.raises(FFmpegError, match="No such file or directory"):
        run_ffmpeg(["-i", str(missing), "-f", "null", "-"])


def test_ffmpeg_ignores_keystrokes_on_inherited_stdin(tmp_path):
    # ffmpeg 7 polls for "q" once per stats period, so the job must outlast one.
    output = tmp_path / "tone.wav"
    child = (
        "import sys\n"
        "from adsync.utils.subprocesses import run_ffmpeg\n"
        "run_ffmpeg(['-re', '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=8000:duration=2',"
        " '-c:a', 'pcm_s16le', sys.argv[1]])\n"
    )
    subprocess.run([sys.executable, "-c", child, str(output)], input=b"q", check=True, timeout=60)
    assert sf.info(str(output)).frames == 16000


class _Spawned(Exception):
    pass


def _two_audio_streams(path):
    streams = [StreamInfo(index=i, codec_type="audio", channels=2, sample_rate=48000, language="eng")
               for i in (1, 2)]
    return MediaInfo(path=str(path), duration=1.0, video_streams=[], audio_streams=streams)


def _spawn_prep(tmp_path):
    from adsync.media.prep import _run_ffmpeg_with_progress
    _run_ffmpeg_with_progress(["-i", "in.mkv", "out.mkv"], 1.0, None)


def _spawn_qc_decode(tmp_path):
    from adsync.quality import verify_media_sync
    source = tmp_path / "rendered.mkv"
    source.write_bytes(b"")
    verify_media_sync(source, work_dir=tmp_path / "qc")


def _spawn_qc_hashes(tmp_path):
    from adsync.quality import verify_video_preservation
    for name in ("source.mkv", "rendered.mkv"):
        (tmp_path / name).write_bytes(b"")
    verify_video_preservation(tmp_path / "source.mkv", tmp_path / "rendered.mkv")


def _spawn_decode(tmp_path):
    from adsync.media.extract import decode_audio
    decode_audio(_two_audio_streams(tmp_path / "in.mkv"))


@pytest.mark.parametrize("spawn", [_spawn_prep, _spawn_qc_decode, _spawn_qc_hashes, _spawn_decode])
def test_ffmpeg_children_never_inherit_console_input(monkeypatch, tmp_path, spawn):
    calls = []

    def capture(command, **kwargs):
        calls.append((list(command), kwargs))
        raise _Spawned

    monkeypatch.setattr(subprocess, "run", capture)
    monkeypatch.setattr(subprocess, "Popen", capture)
    monkeypatch.setattr("adsync.media.probe.probe", _two_audio_streams)
    with pytest.raises(_Spawned):
        spawn(tmp_path)
    command, kwargs = calls[0]
    assert "-nostdin" in command
    assert kwargs.get("stdin") == subprocess.DEVNULL
