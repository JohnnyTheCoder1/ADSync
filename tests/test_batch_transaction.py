"""Publication must be gated by local QC and recover without re-rendering."""

import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from adsync.batch.runner import atomic_json
from adsync.batch.transaction import content_signature
from adsync.batch.worker import main
from adsync.config import SyncConfig


@pytest.fixture
def transaction(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    video, ad = tmp_path / "video.mkv", tmp_path / "ad.mp3"
    video.write_bytes(b"video")
    ad.write_bytes(b"audio")
    spec = {
        "attempt_id": "attempt1", "threads": 1, "label": "S01E01", "prepare": False,
        "video_path": str(video), "ad_path": str(ad),
        "output_path": str(tmp_path / "library" / "episode.mkv"),
        "report_path": str(tmp_path / "state" / "episode.report.json"),
        "result_path": str(tmp_path / "state" / "episode.result.json"),
        "progress_path": str(tmp_path / "state" / "episode.progress.json"),
        "commit_path": str(tmp_path / "state" / "episode.commit.json"),
        "qc_path": str(tmp_path / "state" / "episode.qc.json"),
        "work_path": str(tmp_path / "work"), "overwrite": False,
        "fingerprint": {"video": content_signature(video), "ad": content_signature(ad),
                        "algorithm_revision": "test", "qc_revision": "test"},
        "config": SyncConfig(device="cpu").model_dump(mode="json"),
    }
    state = SimpleNamespace(spec=spec, renders=0, checks=0, qc_status="pass", video_status="pass", seen_paths=[])

    def pipeline(**kwargs):
        state.renders += 1
        state.seen_paths.append(kwargs["output_path"])
        kwargs["output_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["output_path"].write_bytes(b"finished synchronized media")
        atomic_json(kwargs["report_path"], {"confidence": 0.99, "output_path": str(kwargs["output_path"])})
        if getattr(state, "after_render", None):
            state.after_render()
        return SimpleNamespace(confidence=0.99)

    def verify(path, **kwargs):
        state.checks += 1
        state.qc_saw_published_output = Path(spec["output_path"]).exists()
        return {"status": state.qc_status, "reasons": []}

    def verify_video(source, output):
        return {"status": state.video_status, "source_sha256": "compressed-video-packets",
                "output_sha256": "compressed-video-packets" if state.video_status == "pass" else "changed"}

    monkeypatch.setattr("adsync._pipeline.run_pipeline", pipeline)
    monkeypatch.setitem(sys.modules, "adsync.quality", SimpleNamespace(
        verify_media_sync=verify, verify_video_preservation=verify_video, QC_POLICY_REVISION="test"))
    job = tmp_path / "job.json"
    atomic_json(job, spec)
    state.job = job
    state.run = lambda: main([str(job)])
    state.payload = lambda: json.loads(Path(spec["result_path"]).read_text())
    return state


def test_qc_runs_locally_and_review_does_not_publish(transaction):
    transaction.qc_status = "review"
    assert transaction.run() == 1
    assert not transaction.qc_saw_published_output
    assert not Path(transaction.spec["output_path"]).exists()
    assert not Path(transaction.spec["commit_path"]).exists()
    payload = transaction.payload()
    assert payload["status"] == "needs_review"
    assert Path(payload["retained_local_path"]).read_bytes() == b"finished synchronized media"


def test_content_identity_rejection_keeps_report_and_qc_without_publishing(transaction, monkeypatch):
    identity = {"status": "fail", "reasons": ["Only one isolated matching region"],
                "matched_duration": 12.0, "distributed_coverage": 0.03}

    def unrelated_episode(**kwargs):
        atomic_json(kwargs["report_path"], {"confidence": 0.05, "output_path": None,
                                            "identity_check": identity, "quality_check": {}})
        return SimpleNamespace(confidence=0.05)

    monkeypatch.setattr("adsync._pipeline.run_pipeline", unrelated_episode)
    assert transaction.run() == 2
    payload = transaction.payload()
    assert payload["status"] == "failed"
    assert "content identity" in payload["error"].lower()
    assert not Path(transaction.spec["output_path"]).exists()
    assert not Path(transaction.spec["commit_path"]).exists()
    report = json.loads(Path(transaction.spec["report_path"]).read_text())
    qc = json.loads(Path(transaction.spec["qc_path"]).read_text())
    assert report["identity_check"] == identity
    assert report["output_path"] is None
    assert report["quality_check"]["status"] == "fail"
    assert qc["stage"] == "content_identity"
    assert qc["identity"] == identity
    assert transaction.checks == 0


@pytest.mark.parametrize("status", ["pass", "review"])
def test_final_report_contains_compact_qc_summary_and_valid_resume_signature(transaction, status):
    transaction.qc_status = status
    assert transaction.run() == (0 if status == "pass" else 1)
    report = json.loads(Path(transaction.spec["report_path"]).read_text())
    summary = report["quality_check"]
    assert summary["status"] == status
    assert summary["policy_revision"] == "test"
    assert summary["counts"] == {"windows": 0, "failures": 0, "review_reasons": 0}
    assert summary["video_preservation"]["status"] == "pass"
    assert "windows" not in summary
    progress = json.loads(Path(transaction.spec["progress_path"]).read_text())
    local_report = Path(progress["local_report_signature"]["path"])
    assert progress["local_report_signature"]["sha256"] == hashlib.sha256(local_report.read_bytes()).hexdigest()
    assert transaction.run() == (0 if status == "pass" else 1)
    assert transaction.renders == 1
    assert transaction.checks == 1


def test_video_packet_mismatch_prevents_publication_even_when_audio_qc_passes(transaction):
    transaction.video_status = "fail"
    assert transaction.run() == 1
    assert not Path(transaction.spec["output_path"]).exists()
    qc = json.loads(Path(transaction.spec["qc_path"]).read_text())
    assert qc["status"] == "fail"
    assert qc["video_preservation"]["source_sha256"] != qc["video_preservation"]["output_sha256"]


def test_source_content_change_during_render_cannot_be_bound_to_old_hash(transaction):
    def change_source():
        source = Path(transaction.spec["ad_path"])
        before = source.stat()
        source.write_bytes(b"other")
        os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    transaction.after_render = change_source
    assert transaction.run() == 2
    assert not Path(transaction.spec["output_path"]).exists()
    assert not Path(transaction.spec["commit_path"]).exists()
    assert "source" in transaction.payload()["error"].lower()
    assert Path(transaction.payload()["retained_local_path"]).exists()
    # Restoring the source must not bless an artifact rendered during mutation.
    source = Path(transaction.spec["ad_path"])
    source.write_bytes(b"audio")
    original = transaction.spec["fingerprint"]["ad"]
    os.utime(source, ns=(original["mtime_ns"], original["mtime_ns"]))
    transaction.after_render = None
    assert transaction.run() == 0
    assert transaction.renders == 2
    assert transaction.checks == 2


def test_passing_local_qc_commits_hash_bound_media_report_and_qc(transaction):
    assert transaction.run() == 0
    assert not transaction.qc_saw_published_output
    assert transaction.seen_paths[0] != Path(transaction.spec["output_path"])
    commit = json.loads(Path(transaction.spec["commit_path"]).read_text())
    assert commit["phase"] == "transfer_verified"
    assert commit["fingerprint"] == transaction.spec["fingerprint"]
    for key in ("output", "report", "qc"):
        path = Path(transaction.spec[f"{key}_path"])
        assert commit[f"{key}_signature"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    progress = json.loads(Path(transaction.spec["progress_path"]).read_text())
    assert progress["phase"] == "transfer_verified"


def test_verified_local_artifact_resumes_failed_transfer_without_render_or_qc(transaction, monkeypatch):
    from adsync.media import output

    publish = output.publish_completed_file

    def unavailable(local, destination, **kwargs):
        raise output.OutputPublicationError(Path(local), Path(destination), OSError("Drive unavailable"))

    monkeypatch.setattr(output, "publish_completed_file", unavailable)
    assert transaction.run() == 2
    assert Path(transaction.payload()["retained_local_path"]).exists()
    progress = json.loads(Path(transaction.spec["progress_path"]).read_text())
    assert progress["phase"] == "locally_verified"
    monkeypatch.setattr(output, "publish_completed_file", publish)
    assert transaction.run() == 0
    assert transaction.renders == 1
    assert transaction.checks == 1


def test_changed_processed_artifact_cannot_reuse_passing_qc(transaction, monkeypatch):
    from adsync.media import output

    publish = output.publish_completed_file
    def unavailable(local, destination, **kwargs):
        raise output.OutputPublicationError(Path(local), Path(destination), OSError("Drive unavailable"))
    monkeypatch.setattr(output, "publish_completed_file", unavailable)
    assert transaction.run() == 2
    Path(transaction.payload()["retained_local_path"]).write_bytes(b"tampered media")
    monkeypatch.setattr(output, "publish_completed_file", publish)
    assert transaction.run() == 0
    assert transaction.renders == 2
    assert transaction.checks == 2
    assert Path(transaction.spec["output_path"]).read_bytes() == b"finished synchronized media"


def test_no_resume_reprocesses_retained_local_output(transaction, monkeypatch):
    from adsync.media import output

    publish = output.publish_completed_file
    def unavailable(local, destination, **kwargs):
        raise output.OutputPublicationError(Path(local), Path(destination), OSError("Drive unavailable"))
    monkeypatch.setattr(output, "publish_completed_file", unavailable)
    assert transaction.run() == 2
    monkeypatch.setattr(output, "publish_completed_file", publish)
    transaction.spec["resume"] = False
    atomic_json(transaction.job, transaction.spec)
    assert transaction.run() == 0
    assert transaction.renders == 2
    assert transaction.checks == 2


def test_qc_policy_change_rechecks_existing_render_without_reprocessing(transaction):
    transaction.qc_status = "review"
    assert transaction.run() == 1
    transaction.qc_status = "pass"
    transaction.spec["fingerprint"]["qc_revision"] = "new-independent-window-policy"
    atomic_json(transaction.job, transaction.spec)
    assert transaction.run() == 0
    assert transaction.renders == 1
    assert transaction.checks == 2
    commit = json.loads(Path(transaction.spec["commit_path"]).read_text())
    assert commit["fingerprint"]["qc_revision"] == "new-independent-window-policy"


def test_failed_commit_recovers_already_published_media_without_recopy(transaction, monkeypatch):
    from adsync.batch import runner
    from adsync.media import output

    original = runner.atomic_json
    def fail_commit(path, value):
        if Path(path) == Path(transaction.spec["commit_path"]):
            raise OSError("Interrupted receipt write")
        original(path, value)
    monkeypatch.setattr(runner, "atomic_json", fail_commit)
    assert transaction.run() == 2
    assert Path(transaction.spec["output_path"]).exists()
    assert not Path(transaction.spec["commit_path"]).exists()
    monkeypatch.setattr(runner, "atomic_json", original)
    def unexpected_copy(*args, **kwargs):
        pytest.fail("Already verified identical published media should not be copied again")
    monkeypatch.setattr(output, "publish_completed_file", unexpected_copy)
    assert transaction.run() == 0
    assert transaction.renders == 1
    assert transaction.checks == 1
