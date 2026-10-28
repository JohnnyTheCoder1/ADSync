"""Every source stream either survives into the MKV in a valid form or is dropped on purpose."""

import json
import subprocess

import numpy as np
import pytest

from adsync.media.mux import mux_ad_track
from adsync.media.prep import prep_video
from adsync.utils.subprocesses import _find_binary


def _ffmpeg(*args):
    subprocess.run(
        [_find_binary("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *map(str, args)],
        check=True, capture_output=True, stdin=subprocess.DEVNULL,
    )


def _streams(path):
    raw = subprocess.run(
        [_find_binary("ffprobe"), "-v", "error", "-show_entries",
         "stream=index,codec_type,codec_name:stream_disposition=attached_pic:stream_tags=filename",
         "-of", "json", str(path)],
        check=True, capture_output=True, stdin=subprocess.DEVNULL,
    ).stdout
    return json.loads(raw.decode("utf-8"))["streams"]


@pytest.fixture(scope="module")
def web_mp4(tmp_path_factory):
    folder = tmp_path_factory.mktemp("mp4")
    subtitle = folder / "captions.srt"
    subtitle.write_text("1\n00:00:00,100 --> 00:00:00,900\nFixture caption\n", encoding="utf-8")
    source = folder / "episode.mp4"
    # A web-download style MP4: mov_text captions and a timecode data track.
    _ffmpeg("-f", "lavfi", "-i", "color=c=blue:s=32x32:r=10:d=1",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1", "-i", subtitle,
            "-map", "0:v", "-map", "1:a", "-map", "2:s", "-c:v", "mpeg4", "-c:a", "aac", "-c:s", "mov_text",
            "-timecode", "00:00:00:00", source)
    assert [s["codec_type"] for s in _streams(source)].count("data") == 1
    return source


@pytest.fixture(scope="module")
def covered_mkv(tmp_path_factory):
    folder = tmp_path_factory.mktemp("cover")
    cover = folder / "cover.jpg"
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=64x64:duration=1", "-frames:v", "1", cover)
    source = folder / "movie.mkv"
    _ffmpeg("-f", "lavfi", "-i", "color=c=blue:s=32x32:r=10:d=1",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1",
            "-map", "0:v", "-map", "1:a", "-c:v", "ffv1", "-c:a", "flac",
            "-attach", cover, "-metadata:s:t:0", "mimetype=image/jpeg", "-metadata:s:t:0", "filename=cover.jpg",
            source)
    return source


def test_mp4_captions_survive_the_mux_as_srt(web_mp4, tmp_path):
    output = tmp_path / "out.mkv"
    mux_ad_track(web_mp4, np.zeros(48000, dtype=np.float32), 48000, output)
    kinds = [(s["codec_type"], s["codec_name"]) for s in _streams(output)]
    assert ("subtitle", "subrip") in kinds
    assert all(kind != "data" for kind, _ in kinds)


def test_cover_art_stays_an_attachment(covered_mkv, tmp_path):
    output = tmp_path / "out.mkv"
    mux_ad_track(covered_mkv, np.zeros(48000, dtype=np.float32), 48000, output)
    streams = _streams(output)
    pictures = [s for s in streams if s.get("disposition", {}).get("attached_pic")]
    assert len([s for s in streams if s["codec_type"] == "video"]) - len(pictures) == 1
    assert [s.get("tags", {}).get("filename") for s in pictures] == ["cover.jpg"]


def test_prep_converts_mp4_captions(web_mp4, tmp_path):
    output = tmp_path / "episode.prepped.mkv"
    prep_video(web_mp4, output, language=None)
    assert ("subtitle", "subrip") in [(s["codec_type"], s["codec_name"]) for s in _streams(output)]
