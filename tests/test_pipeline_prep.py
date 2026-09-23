"""Season preparation uses the same selected stream for matching and final mux."""

from pathlib import Path

import pytest

from adsync.models import MediaInfo, StreamInfo
from test_pipeline_regressions import pipeline_case


def test_preparation_selects_same_english_stream_for_analysis_and_final_mux(pipeline_case, monkeypatch):
    case = pipeline_case
    def probe(path):
        streams = [StreamInfo(index=0, codec_type="audio", sample_rate=case.sr, channels=2, language="eng")]
        if Path(path).name == "video.mkv":
            streams = [
                StreamInfo(index=1, codec_type="audio", sample_rate=case.sr, channels=2, language="hin"),
                StreamInfo(index=2, codec_type="audio", sample_rate=case.sr, channels=6, language="eng", channel_layout="5.1"),
            ]
        return MediaInfo(path=str(path), duration=case.duration, audio_streams=streams)
    monkeypatch.setattr("adsync.media.probe.probe", probe)
    case.run(mux=True, prepare_audio=True)
    video_decode = [kwargs for name, kwargs in case.extractions if name == "video_audio.wav"]
    assert video_decode[0]["stream_index"] == 2
    assert case.original_audio.index == 2 and case.original_audio.channels == 6


def test_season_preparation_rejects_only_wrong_language_before_decode(pipeline_case, monkeypatch):
    case = pipeline_case
    def probe(path):
        return MediaInfo(path=str(path), duration=case.duration, audio_streams=[
            StreamInfo(index=0, codec_type="audio", sample_rate=case.sr, channels=2, language="hin"),
        ])
    monkeypatch.setattr("adsync.media.probe.probe", probe)
    with pytest.raises(ValueError, match="eng"):
        case.run(mux=True, prepare_audio=True)
    assert not case.extractions
