"""Single-pass main-audio preparation preserves copied media and AD signal."""

import json
from pathlib import Path
import subprocess
import wave

import numpy as np
import pytest

from adsync.media.mux import mux_ad_file, mux_ad_track
from adsync.media.prep import pick_audio_stream, prep_video
from adsync.media.probe import probe
from adsync.utils.subprocesses import _find_binary


def _ffmpeg(*args):
    return subprocess.run(
        [_find_binary("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y", *map(str, args)],
        check=True, capture_output=True,
    ).stdout


@pytest.fixture(scope="module", params=[2, 6])
def source_media(tmp_path_factory, request):
    folder = tmp_path_factory.mktemp(f"mux-prep-{request.param}ch")
    subtitle = folder / "captions.srt"
    subtitle.write_text("1\n00:00:00,000 --> 00:00:00,800\nFixture caption\n", encoding="utf-8")
    metadata = folder / "chapters.txt"
    metadata.write_text(
        ";FFMETADATA1\ntitle=Fixture title\n[CHAPTER]\nTIMEBASE=1/1000\n"
        "START=0\nEND=1000\ntitle=Chapter One\n", encoding="utf-8",
    )
    attachment = folder / "fixture.ttf"
    attachment.write_bytes(b"Fixture font attachment")
    components = [
        "0.06*sin(2*PI*310*t)", "0.05*sin(2*PI*410*t)",
        "0.09*sin(2*PI*510*t)", "0.04*sin(2*PI*610*t)",
        "0.03*sin(2*PI*710*t)", "0.02*sin(2*PI*810*t)",
    ][:request.param]
    layout = "stereo" if request.param == 2 else "5.1"
    main_audio = "aevalsrc=" + "|".join(components) + f":s=48000:d=1:c={layout}"
    source = folder / "source.mkv"
    _ffmpeg(
        "-f", "lavfi", "-i", "color=c=blue:s=32x32:r=10:d=1",
        "-f", "lavfi", "-i", "sine=frequency=200:sample_rate=48000:duration=1",
        "-f", "lavfi", "-i", main_audio,
        "-f", "srt", "-i", subtitle, "-f", "ffmetadata", "-i", metadata,
        "-map", "0:v", "-map", "1:a", "-map", "2:a", "-map", "3:s",
        "-map_metadata", "4", "-map_chapters", "4",
        "-c:v", "ffv1", "-c:a", "flac", "-c:s", "srt",
        "-metadata:s:a:0", "language=dan", "-metadata:s:a:1", "language=eng",
        "-metadata:s:a:1", "title=Main English",
        "-disposition:a:0", "default", "-disposition:a:1", "0",
        "-attach", attachment, "-metadata:s:t:0", "mimetype=application/x-truetype-font",
        "-metadata:s:t:0", "filename=fixture.ttf",
        source,
    )
    return source, request.param


def _structure(path):
    result = subprocess.run([
        _find_binary("ffprobe"), "-v", "quiet", "-show_streams", "-show_chapters",
        "-show_format", "-of", "json", str(path),
    ], check=True, capture_output=True)
    return json.loads(result.stdout)


def _stream_hash(path, stream):
    return _ffmpeg("-i", path, "-map", stream, "-c", "copy", "-f", "hash", "-hash", "sha256", "-")


def _decode(path, stream):
    return np.frombuffer(_ffmpeg("-i", path, "-map", stream, "-c:a", "pcm_f32le", "-f", "f32le", "-"), dtype=np.float32)


def test_single_pass_matches_prep_then_mux_without_changing_video_or_ad(source_media, tmp_path):
    source, original_channels = source_media
    picked = pick_audio_stream(probe(source), language="eng")
    # A non-silent mono AD signal detects accidentally applying the main-audio
    # stereo filter to all output audio streams.
    ad = (0.04 * np.sin(2 * np.pi * 997 * np.arange(48000) / 48000)).astype(np.float32)
    direct = tmp_path / "single.mkv"
    mux_ad_track(source, ad, 48000, direct, original_audio=picked, threads=1)

    prepared = prep_video(source, tmp_path / "prepared.mkv", language="eng")
    reference = tmp_path / "reference.mkv"
    mux_ad_track(prepared.output_path, ad, 48000, reference, threads=1)
    structure = _structure(direct)
    audio = [stream for stream in structure["streams"] if stream["codec_type"] == "audio"]
    assert len(audio) == 2
    assert [stream["channels"] for stream in audio] == [2, 1]
    assert [stream["tags"]["language"] for stream in audio] == ["eng", "eng"]
    assert audio[0]["disposition"]["default"] == 1
    assert audio[1]["tags"]["title"] == "Audio Description"
    assert audio[0]["codec_name"] == ("opus" if original_channels == 6 else "flac")
    assert _stream_hash(direct, "0:v:0") == _stream_hash(source, "0:v:0")
    assert _stream_hash(direct, "0:s:0") == _stream_hash(source, "0:s:0")
    np.testing.assert_array_equal(_decode(direct, "0:a:0"), _decode(reference, "0:a:0"))
    np.testing.assert_array_equal(_decode(direct, "0:a:1"), _decode(reference, "0:a:1"))
    if original_channels == 2:
        assert _stream_hash(direct, "0:a:0") == _stream_hash(source, "0:a:1")
    assert structure["format"]["tags"]["title"] == "Fixture title"
    assert structure["chapters"][0]["tags"]["title"] == "Chapter One"
    attachments = [stream for stream in structure["streams"] if stream["codec_type"] == "attachment"]
    assert len(attachments) == 1
    assert attachments[0]["tags"]["filename"] == "fixture.ttf"
    extracted = tmp_path / "extracted-font.bin"
    _ffmpeg("-dump_attachment:t:0", extracted, "-i", direct, "-map", "0:a:0", "-t", "0", "-f", "null", "-")
    assert extracted.read_bytes() == b"Fixture font attachment"


def test_default_mux_still_keeps_all_original_audio_streams(source_media, tmp_path):
    source, channels = source_media
    result = mux_ad_track(source, np.zeros(48000, np.float32), 48000, tmp_path / "default.mkv", threads=1)
    info = probe(result)
    assert len(info.audio_streams) == 3
    assert [stream.language for stream in info.audio_streams] == ["dan", "eng", "eng"]
    assert info.audio_streams[1].channels == channels
    assert _stream_hash(result, "0:a:0") == _stream_hash(source, "0:a:0")
    assert _stream_hash(result, "0:a:1") == _stream_hash(source, "0:a:1")


def test_selected_main_stream_count_overrides_old_audio_count(source_media, tmp_path):
    source, _ = source_media
    picked = pick_audio_stream(probe(source), language="eng")
    result = mux_ad_track(
        source, np.zeros(48000, np.float32), 48000, tmp_path / "selected.mkv",
        original_audio=picked, n_existing_audio=99, threads=1,
    )
    info = probe(result)
    assert len(info.audio_streams) == 2
    assert info.audio_streams[1].codec_name == "opus"


def test_unknown_layout_still_downmixes_only_main_audio(source_media, tmp_path):
    source, channels = source_media
    if channels != 6:
        pytest.skip("Unknown multichannel layout case")
    picked = pick_audio_stream(probe(source), language="eng").model_copy(update={"channel_layout": None})
    result = mux_ad_track(
        source, np.zeros(48000, np.float32), 48000, tmp_path / "unknown-layout.mkv",
        original_audio=picked, threads=1,
    )
    assert [stream.channels for stream in probe(result).audio_streams] == [2, 1]


def test_file_mux_preserves_attachment_with_second_audio_input(source_media, tmp_path):
    source, _ = source_media
    ad_path = tmp_path / "ad.wav"
    with wave.open(str(ad_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(48000)
        handle.writeframes(np.zeros(48000, np.int16).tobytes())
    result = mux_ad_file(source, ad_path, tmp_path / "file-mux.mkv")
    assert len(probe(result).audio_streams) == 3
    extracted = tmp_path / "extracted-font.bin"
    _ffmpeg("-dump_attachment:t:0", extracted, "-i", result, "-map", "0:a:0", "-t", "0", "-f", "null", "-")
    assert extracted.read_bytes() == b"Fixture font attachment"
