"""Partial alignments cannot bypass publication review with a passing QC score."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from adsync.batch.runner import atomic_json
from adsync.batch.transaction import content_signature
from test_batch_transaction import transaction


@pytest.fixture
def partial_transaction(transaction, monkeypatch):
    state = transaction
    state.render_media = True
    state.alignment_review_required = True
    state.spec["config"]["confidence_threshold"] = 0.0
    atomic_json(state.job, state.spec)

    def pipeline(**kwargs):
        state.renders += 1
        if state.render_media:
            kwargs["output_path"].write_bytes(b"partial synchronized media")
        atomic_json(kwargs["report_path"], {
            "mode": "partial", "confidence": 0.0,
            "alignment_review_required": state.alignment_review_required,
            "output_path": str(kwargs["output_path"]) if state.render_media else None,
            "timing_debug": {"partial_alignment": {
                "source_gaps": [{"start_sec": 5.0, "end_sec": 12.0}],
            }},
        })
        return SimpleNamespace(confidence=0.0)

    monkeypatch.setattr("adsync._pipeline.run_pipeline", pipeline)
    return state


def test_partial_review_flag_withholds_media_even_when_qc_passes_at_zero_threshold(partial_transaction):
    state = partial_transaction
    assert state.run() == 1
    assert state.payload()["status"] == "needs_review"
    assert not Path(state.spec["output_path"]).exists()
    assert not Path(state.spec["commit_path"]).exists()
    assert Path(state.payload()["retained_local_path"]).read_bytes() == b"partial synchronized media"
    report = json.loads(Path(state.spec["report_path"]).read_text())
    assert report["alignment_review_required"] is True
    assert report["quality_check"]["status"] == "review"
    assert "alignment" in state.payload()["error"].lower()


@pytest.mark.parametrize("stale_media", [False, True])
def test_partial_without_output_preserves_review_evidence_without_media_qc(partial_transaction, stale_media):
    state = partial_transaction
    state.render_media = False
    if stale_media:
        local_output = Path(state.spec["work_path"]) / "rendered" / "episode.mkv"
        local_output.parent.mkdir(parents=True)
        local_output.write_bytes(b"stale media from an older attempt")

    assert state.run() == 1
    payload = state.payload()
    assert payload["status"] == "needs_review"
    assert not payload.get("retained_local_path")
    assert state.checks == 0
    assert not Path(state.spec["output_path"]).exists()
    assert not Path(state.spec["commit_path"]).exists()
    report = json.loads(Path(state.spec["report_path"]).read_text())
    qc = json.loads(Path(state.spec["qc_path"]).read_text())
    assert report["output_path"] is None
    assert report["alignment_review_required"] is True
    assert report["timing_debug"]["partial_alignment"]["source_gaps"] == [
        {"start_sec": 5.0, "end_sec": 12.0},
    ]
    assert report["quality_check"]["status"] == "review"
    assert qc["status"] == "review"
    assert qc["review_reasons"]
    assert "alignment" in payload["error"].lower()


def test_resumed_passing_qc_cannot_override_partial_review_flag(partial_transaction, monkeypatch):
    from adsync.media import output

    state = partial_transaction
    state.alignment_review_required = False
    publish = output.publish_completed_file

    def unavailable(local, destination, **kwargs):
        raise output.OutputPublicationError(Path(local), Path(destination), OSError("Drive unavailable"))

    monkeypatch.setattr(output, "publish_completed_file", unavailable)
    assert state.run() == 2
    progress_path = Path(state.spec["progress_path"])
    progress = json.loads(progress_path.read_text())
    local_report = Path(progress["local_report_signature"]["path"])
    report = json.loads(local_report.read_text())
    report["alignment_review_required"] = True
    atomic_json(local_report, report)
    progress["local_report_signature"] = content_signature(local_report)
    atomic_json(progress_path, progress)
    monkeypatch.setattr(output, "publish_completed_file", publish)

    assert state.run() == 1
    assert state.payload()["status"] == "needs_review"
    assert state.renders == 1
    assert state.checks == 1
    assert not Path(state.spec["output_path"]).exists()
    assert not Path(state.spec["commit_path"]).exists()
    qc = json.loads(Path(state.spec["qc_path"]).read_text())
    assert qc["status"] == "review"
    assert qc["review_reasons"]
