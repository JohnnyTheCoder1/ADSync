"""Audio streams are chosen by meaning, never by position."""

from pathlib import Path

import pytest

from adsync.media.prep import pick_ad_stream, pick_audio_stream
from adsync.models import MediaInfo, StreamInfo
from test_pipeline_regressions import pipeline_case


def _stream(index, title=None, *, language="eng", visual_impaired=False, channels=2, sample_rate=48000):
    return StreamInfo(index=index, codec_type="audio", channels=channels, sample_rate=sample_rate,
                      language=language, title=title, visual_impaired=visual_impaired)


def _info(*streams):
    return MediaInfo(path="source.mkv", duration=10.0, audio_streams=list(streams))


def test_flagged_description_stream_is_analyzed_and_rendered(pipeline_case, monkeypatch):
    case = pipeline_case

    def probe(path):
        streams = [_stream(0, sample_rate=case.sr)]
        if Path(path).name == "ad.wav":
            streams = [_stream(0, "English", sample_rate=case.sr),
                       _stream(1, "English Audio Description", visual_impaired=True, sample_rate=case.sr)]
        return MediaInfo(path=str(path), duration=case.duration, audio_streams=streams)

    monkeypatch.setattr("adsync.media.probe.probe", probe)
    case.run(mux=True)
    assert [kw.get("stream_index") for name, kw in case.extractions if name == "ad.wav"] == [1, 1]


@pytest.mark.parametrize("title", ["Audio Description", "English [AD]", "Described Video", "Descriptive audio"])
def test_description_title_identifies_the_stream_without_a_flag(title):
    assert pick_ad_stream(_info(_stream(0, "Stereo"), _stream(1, title))).index == 1


def test_description_in_the_wanted_language_wins():
    info = _info(_stream(1, "Audiodescripción", language="spa", visual_impaired=True),
                 _stream(2, "Audio Description", visual_impaired=True))
    assert pick_ad_stream(info, language="eng").index == 2


def test_ambiguous_ad_source_raises_and_names_the_option():
    with pytest.raises(ValueError, match="--ad-stream"):
        pick_ad_stream(_info(_stream(0, "English"), _stream(2, "Commentary")))


def test_explicit_ad_stream_wins():
    info = _info(_stream(0, "English"), _stream(1, "Audio Description", visual_impaired=True))
    assert pick_ad_stream(info, index=0).index == 0


def test_main_audio_skips_description_tracks():
    info = _info(_stream(1, "Audio Description", visual_impaired=True, channels=6), _stream(2, "English", channels=6))
    assert pick_audio_stream(info, language="eng").index == 2
