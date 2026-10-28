"""Damaged or truncated sources are reported instead of silently producing a short track."""

import logging
import subprocess
from pathlib import Path

from adsync.media.extract import decode_audio
from adsync.media.probe import probe
from adsync.models import MediaInfo, StreamInfo
from adsync.utils.subprocesses import _find_binary
from test_pipeline_regressions import pipeline_case


def test_truncated_ad_file_is_flagged_and_held_below_the_threshold(pipeline_case, monkeypatch):
    case = pipeline_case

    def probe_with_long_header(path):
        duration = 60.0 if Path(path).name == "ad.wav" else case.duration
        return MediaInfo(path=str(path), duration=duration,
                         audio_streams=[StreamInfo(index=0, codec_type="audio", sample_rate=case.sr)])

    monkeypatch.setattr("adsync.media.probe.probe", probe_with_long_header)
    report = case.run()
    assert any("truncated" in warning for warning in report.warnings)
    assert report.confidence < 0.70


def test_problems_ffmpeg_reports_on_success_are_logged(tmp_path, caplog):
    source = tmp_path / "damaged.mkv"
    subprocess.run([_find_binary("ffmpeg"), "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "anoisesrc=d=60:c=pink:r=48000:a=0.1", "-c:a", "flac", str(source)],
                   check=True, capture_output=True, stdin=subprocess.DEVNULL)
    data = bytearray(source.read_bytes())
    middle = len(data) // 2
    data[middle:middle + 200_000] = bytes(200_000)
    source.write_bytes(bytes(data))
    with caplog.at_level(logging.WARNING, logger="adsync"):
        decode_audio(probe(source), sr=16000)
    assert any("damaged.mkv" in record.getMessage() and "reported" in record.getMessage() for record in caplog.records)


def test_mkv_track_duration_tags_are_read(tmp_path):
    source = tmp_path / "tagged.mkv"
    # The file lasts 5 s (picture) while its audio track lasts 3 s.
    subprocess.run([_find_binary("ffmpeg"), "-nostdin", "-v", "error", "-y",
                    "-f", "lavfi", "-i", "color=c=black:s=32x32:r=10:d=5",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=3",
                    "-map", "0:v", "-map", "1:a", "-c:v", "ffv1", "-c:a", "flac", str(source)],
                   check=True, capture_output=True, stdin=subprocess.DEVNULL)
    assert abs(probe(source).audio_streams[0].duration - 3.0) < 0.05
