"""Preflight protects media; secondary report errors retain recovery details."""

from pathlib import Path

import pytest

from adsync._pipeline import run_pipeline
from adsync.config import SyncConfig
from adsync.media.output import OutputPublicationError
from test_pipeline_regressions import pipeline_case


def test_debug_artifact_cannot_overwrite_ad_input(tmp_path, monkeypatch):
    source = tmp_path / "video_audio.wav"
    source.write_bytes(b"original AD")
    def no_probe(*args):
        pytest.fail("Alias must be detected before media processing")
    monkeypatch.setattr("adsync.media.probe.probe", no_probe)
    with pytest.raises(ValueError, match="overwrite"):
        run_pipeline(video_path=tmp_path / "video.mkv", ad_path=source,
                     output_path=None, debug_dir=tmp_path,
                     config=SyncConfig(device="cpu"), mux=False)
    assert source.read_bytes() == b"original AD"


def test_recovery_report_failure_preserves_original_publication_error(pipeline_case, monkeypatch, tmp_path):
    local_media = tmp_path / "retained.mkv"
    local_media.write_bytes(b"completed")
    error = OutputPublicationError(local_media, Path("offline/movie.mkv"), OSError("share offline"))
    def fail_publication(*args, **kwargs):
        raise error
    def fail_report(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr("adsync.media.mux.mux_ad_track", fail_publication)
    monkeypatch.setattr("adsync.report.json_report.write_report", fail_report)
    with pytest.raises(OutputPublicationError) as caught:
        pipeline_case.run(mux=True)
    assert caught.value is error
    assert str(local_media) in str(caught.value)
