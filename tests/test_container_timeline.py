"""Decoded audio sits on the container's playback timeline, so the AD lands where the picture is."""

import subprocess

import numpy as np
import pytest

from adsync.media.extract import decode_audio
from adsync.media.probe import probe
from adsync.utils.subprocesses import _find_binary

SR = 16000
CLICK = "aevalsrc='if(between(t,1,1.0004),0.9,0)':s=48000:d=3"
PICTURE = "color=c=black:s=32x32:r=25:d=4"


def _ffmpeg(*args):
    subprocess.run(
        [_find_binary("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *map(str, args)],
        check=True, capture_output=True, stdin=subprocess.DEVNULL,
    )


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    folder = tmp_path_factory.mktemp("timeline")
    delayed = folder / "delayed.mkv"
    # Audio starts 0.479 s after the picture, so its click plays at 1.479 s.
    _ffmpeg("-f", "lavfi", "-i", PICTURE, "-itsoffset", "0.479", "-f", "lavfi", "-i", CLICK,
            "-map", "0:v", "-map", "1:a", "-c:v", "ffv1", "-c:a", "flac", delayed)
    gapped = folder / "gapped.mkv"
    # Audio timestamps jump 1 s after 0.5 s (a broadcast dropout), so the click plays at 2 s.
    _ffmpeg("-f", "lavfi", "-i", PICTURE, "-f", "lavfi", "-i", CLICK,
            "-filter_complex", r"[1:a]asetpts=PTS+gte(T\,0.5)/TB[a]",
            "-map", "0:v", "-map", "[a]", "-c:v", "ffv1", "-c:a", "flac", gapped)
    return {"delayed": delayed, "gapped": gapped}


@pytest.mark.parametrize("name, playback_sec", [("delayed", 1.479), ("gapped", 2.0)])
def test_decoded_click_lands_at_its_playback_time(media, name, playback_sec):
    y = decode_audio(probe(media[name]), sr=SR)
    assert abs(np.argmax(np.abs(y)) / SR - playback_sec) <= 0.001
