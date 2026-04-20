"""Tests for media probing."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from adsync.media.probe import probe
from adsync.models import MediaInfo


SAMPLE_FFPROBE_OUTPUT = json.dumps({
    "format": {
        "format_name": "matroska,webm",
        "duration": "2700.123",
    },
    "streams": [
        {
            "index": 0,
            "codec_type": "video",
            "codec_name": "h264",
        },
        {
            "index": 1,
            "codec_type": "audio",
            "codec_name": "aac",
            "channels": 2,
            "sample_rate": "48000",
            "tags": {"language": "eng"},
            "duration": "2700.123",
        },
        {
            "index": 2,
            "codec_type": "audio",
            "codec_name": "aac",
            "channels": 2,
            "sample_rate": "48000",
            "tags": {"language": "spa"},
            "duration": "2700.123",
        },
    ],
})


@patch("adsync.media.probe.run_ffprobe")
def test_probe_parses_streams(mock_ffprobe: MagicMock, tmp_path: Path) -> None:
    """Probe should parse video and audio streams from ffprobe JSON."""
    mock_ffprobe.return_value = MagicMock(stdout=SAMPLE_FFPROBE_OUTPUT)

    dummy = tmp_path / "test.mkv"
    dummy.write_bytes(b"fake")

    info = probe(dummy)

    assert isinstance(info, MediaInfo)
    assert info.duration == pytest.approx(2700.123)
    assert len(info.video_streams) == 1
    assert len(info.audio_streams) == 2
    assert info.audio_streams[0].codec_name == "aac"
    assert info.audio_streams[0].channels == 2
    assert info.audio_streams[0].sample_rate == 48000
    assert info.audio_streams[0].language == "eng"
    assert info.audio_streams[1].language == "spa"


def test_probe_file_not_found() -> None:
    """Probe should raise FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        probe("/nonexistent/file.mkv")
