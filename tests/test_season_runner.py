"""Durable episode scheduling without decoding user media."""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from adsync.batch import runner
from adsync.batch.worker import main as worker_main
from adsync.config import SyncConfig


@pytest.fixture
def season(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setattr(runner.tempfile, "gettempdir", lambda: str(tmp_path))
    pairs = []
    for n in range(1, 5):
        video, ad = tmp_path / f"video{n}.mkv", tmp_path / f"ad{n}.wav"
        video.write_bytes(b"video")
        ad.write_bytes(b"audio")
        pairs.append(SimpleNamespace(key=SimpleNamespace(label=f"S01E{n:02d}"), video_path=video,
                                     ad_path=ad, output_path=tmp_path / "output" / f"S01E{n:02d}.mkv"))
    state = SimpleNamespace(pairs=pairs, state_dir=tmp_path / "state", processes=[], active=0,
                            peak=0, behaviors={}, duration=0.01, terminated=[], environments=[])
    lock = threading.Lock()
    monkeypatch.setattr(runner, "_output_lock_path", lambda output: tmp_path / "reservations" / (Path(output).name + ".lock"))

    class FakeProcess:
        def __init__(self, command, **kwargs):
            assert command[:3] == [runner.sys.executable, "-m", "adsync.batch.worker"]
            assert kwargs["stderr"] == runner.subprocess.STDOUT
            self.spec = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
            self.spec_path = command[-1]
            self.env = kwargs["env"]
            self.returncode = None
            self.deadline = time.monotonic() + state.duration
            self.pid = 98765
            with lock:
                state.active += 1
                state.peak = max(state.peak, state.active)
                state.processes.append(self)
                state.environments.append(kwargs["env"])

        def finish(self, canceled=False):
            if self.returncode is not None:
                return
            behavior = state.behaviors.get(self.spec["label"], "completed")
            if behavior == "identity_failed" and not canceled:
                self.returncode = worker_main([self.spec_path])
                with lock:
                    state.active -= 1
                return
            status = "failed" if behavior == "failed_local" else behavior
            self.returncode = -15 if canceled else {"completed": 0, "needs_review": 1, "failed": 2}[status]
            if not canceled:
                if status in {"completed", "needs_review"}:
                    from adsync.batch.transaction import content_signature
                    path = (Path(self.spec["output_path"]) if status == "completed" else
                            Path(self.spec["work_path"]) / "rendered" / "review.mkv")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"completed-media")
                    Path(self.spec["report_path"]).write_text('{"confidence": 0.9}', encoding="utf-8")
                    runner.atomic_json(Path(self.spec["qc_path"]), {"status": "pass" if status == "completed" else "review"})
                    if status == "completed":
                        runner.atomic_json(Path(self.spec["commit_path"]), {
                            "phase": "transfer_verified", "fingerprint": self.spec["fingerprint"],
                            **{f"{name}_signature": content_signature(self.spec[f"{name}_path"])
                               for name in ("output", "report", "qc")},
                        })
                payload = {"attempt_id": self.spec["attempt_id"], "status": status}
                if status == "completed":
                    payload["phase"] = "transfer_verified"
                elif status == "needs_review":
                    payload.update(phase="processed", retained_local_path=str(path))
                    runner.atomic_json(Path(self.spec["progress_path"]), {
                        "phase": "processed", "fingerprint": self.spec["fingerprint"],
                        "work_path": self.spec["work_path"], "local_path": str(path),
                    })
                if status == "failed":
                    payload.update(error="Share unavailable", retained_local_path="retained-local.mkv")
                if behavior == "failed_local":
                    from adsync.media.output import OutputPublicationError, staged_output
                    try:
                        with patch.dict(os.environ, {"ADSYNC_STAGING_DIR": self.env["ADSYNC_STAGING_DIR"]}):
                            with staged_output(Path(self.spec["output_path"])) as local_path:
                                local_path.write_bytes(b"completed-local-media")
                    except OutputPublicationError as exc:
                        payload.update(error=str(exc), retained_local_path=str(exc.local_path))
                runner.atomic_json(Path(self.spec["result_path"]), payload)
            with lock:
                state.active -= 1

        def poll(self):
            if time.monotonic() >= self.deadline:
                self.finish()
            return self.returncode

        def wait(self, timeout=None):
            self.finish()
            return self.returncode

    def stop(proc):
        state.terminated.append(proc)
        proc.finish(canceled=True)

    monkeypatch.setattr(runner.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(runner, "_stop_process_tree", stop)
    state.run = lambda selected=None, **kw: runner.run_season(
        pairs if selected is None else selected, config=kw.pop("config", SyncConfig(device="cpu")),
        jobs=kw.pop("jobs", 2), threads_per_job=kw.pop("threads_per_job", 2),
        state_dir=kw.pop("state_dir", state.state_dir), **kw,
    )
    return state


def test_bounded_parallel_workers_write_parent_receipts_and_logs(season):
    report = season.run(jobs=2)
    assert report.counts == {"completed": 4}
    assert season.peak == 2
    assert [r.label for r in report.results] == [p.key.label for p in season.pairs]
    saved = json.loads(Path(report.state_path).read_text(encoding="utf-8"))
    assert all(entry["receipt"]["output_signature"]["size"] > 0 for entry in saved["episodes"].values())
    assert all(entry["receipt"]["output_signature"]["sha256"] for entry in saved["episodes"].values())
    assert all(Path(result.log_path).is_file() for result in report.results)
    assert all(env["ADSYNC_THREADS"] == "2" and env["OPENBLAS_NUM_THREADS"] == "1" for env in season.environments)
    assert all(not Path(p.spec["work_path"]).exists() for p in season.processes)
    assert not (season.state_dir / "run.lock").exists()


def test_resume_skips_success_and_review_with_matching_metadata(season):
    season.behaviors["S01E02"] = "needs_review"
    first = season.run()
    assert first.counts == {"completed": 3, "needs_review": 1}
    count = len(season.processes)
    second = season.run(config=SyncConfig(device="cuda"), threads_per_job=1)
    assert second.counts == {"skipped_completed": 3, "skipped_needs_review": 1}
    assert len(season.processes) == count


@pytest.mark.parametrize("change", ["input", "output", "config", "prepare", "version", "report_missing", "report_changed"])
def test_changed_signature_never_silently_resumes_or_overwrites(season, monkeypatch, change):
    pair = season.pairs[0]
    season.run([pair])
    kwargs = {}
    if change == "input":
        pair.ad_path.write_bytes(b"different-audio")
    elif change == "output":
        stat = pair.output_path.stat()
        os.utime(pair.output_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000))
    elif change == "config":
        kwargs["config"] = SyncConfig(device="cpu", offset_adjust=0.2)
    elif change == "prepare":
        kwargs["prepare"] = False
    elif change == "version":
        monkeypatch.setattr(runner, "__version__", "a-new-application-version")
    elif change == "report_missing":
        (season.state_dir / "episodes" / "S01E01.report.json").unlink()
    else:
        (season.state_dir / "episodes" / "S01E01.report.json").write_text('{"changed": true}', encoding="utf-8")
    report = season.run([pair], **kwargs)
    assert report.counts == {"conflict": 1}
    assert len(season.processes) == 1
    assert pair.output_path.read_bytes() == b"completed-media"


def test_existing_output_needs_overwrite_or_matching_receipt(season):
    pair = season.pairs[0]
    pair.output_path.parent.mkdir()
    pair.output_path.write_bytes(b"existing")
    assert season.run([pair]).counts == {"conflict": 1}
    assert pair.output_path.read_bytes() == b"existing"
    assert season.run([pair], overwrite=True).counts == {"completed": 1}
    assert len(season.processes) == 1


def test_conflict_does_not_erase_previous_success_receipt(season):
    pair = season.pairs[0]
    season.run([pair])
    assert season.run([pair], config=SyncConfig(offset_adjust=0.3)).counts == {"conflict": 1}
    assert season.run([pair]).counts == {"skipped_completed": 1}


def test_same_size_same_timestamp_source_change_invalidates_receipt(season):
    pair = season.pairs[0]
    season.run([pair])
    before = pair.ad_path.stat()
    pair.ad_path.write_bytes(b"other")
    os.utime(pair.ad_path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert season.run([pair]).counts == {"conflict": 1}


def test_algorithm_revision_invalidates_receipt_without_package_version_change(season, monkeypatch):
    pair = season.pairs[0]
    season.run([pair])
    monkeypatch.setattr(runner, "ALGORITHM_REVISION", "different-timing-semantics", raising=False)
    assert season.run([pair]).counts == {"conflict": 1}


def test_qc_revision_reschedules_review_using_same_local_work_folder(season, monkeypatch):
    pair = season.pairs[0]
    season.behaviors[pair.key.label] = "needs_review"
    assert season.run([pair]).counts == {"needs_review": 1}
    previous_work = season.processes[0].spec["work_path"]
    monkeypatch.setattr(runner, "QC_POLICY_REVISION", "changed-qc-policy")
    assert season.run([pair]).counts == {"needs_review": 1}
    assert season.processes[1].spec["work_path"] == previous_work


def test_completion_requires_transaction_commit_marker(season):
    pair = season.pairs[0]
    season.run([pair])
    (season.state_dir / "episodes" / "S01E01.commit.json").unlink(missing_ok=True)
    assert season.run([pair]).counts == {"conflict": 1}


def test_committed_worker_is_resumable_after_parent_checkpoint_interruption(season):
    pair = season.pairs[0]
    first = season.run([pair])
    state_path = Path(first.state_path)
    state = json.loads(state_path.read_text())
    state["episodes"][pair.key.label].pop("receipt")
    state["episodes"][pair.key.label]["status"] = "running"
    runner.atomic_json(state_path, state)
    assert season.run([pair]).counts == {"skipped_completed": 1}
    assert len(season.processes) == 1


def test_publication_failure_keeps_recovery_path_and_continues(season):
    season.behaviors["S01E01"] = "failed"
    report = season.run()
    assert report.counts == {"failed": 1, "completed": 3}
    assert report.results[0].retained_local_path == "retained-local.mkv"
    assert "Share unavailable" in report.results[0].error


def test_identity_failure_evidence_survives_parent_work_cleanup(season, monkeypatch):
    pair = season.pairs[0]
    season.behaviors[pair.key.label] = "identity_failed"

    def pipeline(**kwargs):
        runner.atomic_json(kwargs["report_path"], {
            "confidence": 0.01, "output_path": None,
            "identity_check": {"status": "fail", "reasons": ["Wrong episode content"]},
        })
        return SimpleNamespace(confidence=0.01)

    monkeypatch.setattr("adsync._pipeline.run_pipeline", pipeline)
    batch = season.run([pair])
    assert batch.counts == {"failed": 1}
    result = batch.results[0]
    assert result.phase == "identity_failed"
    assert "Wrong episode content" in result.error
    assert not Path(season.processes[0].spec["work_path"]).exists()
    assert json.loads(Path(result.report_path).read_text())["identity_check"]["status"] == "fail"
    assert json.loads(Path(result.qc_path).read_text())["stage"] == "content_identity"
    assert not pair.output_path.exists()


def test_completed_publication_failure_inside_worker_temp_survives_cleanup(season, monkeypatch):
    season.behaviors["S01E01"] = "failed_local"
    def unavailable(*args):
        raise OSError("Shared destination is unavailable")
    monkeypatch.setattr("adsync.media.output._publish", unavailable)
    report = season.run([season.pairs[0]])
    assert report.counts == {"failed": 1}
    retained = Path(report.results[0].retained_local_path)
    assert retained.read_bytes() == b"completed-local-media"
    assert retained.is_relative_to(Path(season.processes[0].spec["work_path"]))


def test_output_metadata_failure_at_checkpoint_does_not_abort_other_episodes(season, monkeypatch):
    original = runner._signature
    checks = 0
    def signature(path):
        nonlocal checks
        if Path(path) == season.pairs[0].output_path:
            checks += 1
            if checks >= 1:
                raise PermissionError("Share disconnected after publication")
        return original(path)
    monkeypatch.setattr(runner, "_signature", signature)
    report = season.run()
    assert report.counts == {"failed": 1, "completed": 3}
    assert "metadata" in report.results[0].error.lower()


def test_state_lock_rejects_concurrent_run_without_touching_files(season):
    with runner._exclusive_lock(season.state_dir / "run.lock", "test"):
        with pytest.raises(runner.BatchStateError, match="locked.*run.lock"):
            season.run()
    assert not season.processes


def test_output_reservation_coordinates_different_state_directories(season):
    pair = season.pairs[0]
    with runner._exclusive_lock(runner._output_lock_path(pair.output_path), "another state directory"):
        report = season.run([pair], state_dir=season.state_dir / "other")
    assert report.counts == {"conflict": 1}
    assert not season.processes


def test_interrupt_cancels_active_workers_and_preserves_pending(season, monkeypatch):
    season.duration = 30
    def interrupt(*args, **kwargs):
        deadline = time.monotonic() + 2
        while not season.processes and time.monotonic() < deadline:
            time.sleep(0.01)
        raise KeyboardInterrupt
    monkeypatch.setattr(runner, "wait", interrupt)
    report = season.run(jobs=1)
    assert report.interrupted
    assert report.counts == {"canceled": 1, "pending": 3}
    assert len(season.terminated) == 1
    assert season.active == 0
    assert all(not Path(p.spec["work_path"]).exists() for p in season.processes)
    saved = json.loads(Path(report.state_path).read_text(encoding="utf-8"))
    assert all(entry["status"] != "running" for entry in saved["episodes"].values())


def test_state_file_cannot_alias_source(season):
    pair = season.pairs[0]
    season.state_dir.mkdir()
    source = season.state_dir / "state.json"
    source.write_bytes(b"source-media")
    pair.video_path = source
    with pytest.raises(ValueError, match="overwrite an input"):
        season.run([pair])
    assert source.read_bytes() == b"source-media"


def test_output_cannot_replace_another_episodes_input(season):
    season.pairs[0].output_path = season.pairs[1].video_path
    with pytest.raises(ValueError, match="overwrite an input"):
        season.run()


@pytest.mark.parametrize("prepare", [True, False])
def test_worker_requests_single_pass_preparation_on_original_video(tmp_path, monkeypatch, prepare):
    calls = []
    source = tmp_path / "source.mkv"
    spec = {
        "attempt_id": "worker-contract", "threads": 2, "prepare": prepare,
        "video_path": str(source), "ad_path": str(tmp_path / "ad.wav"),
        "output_path": str(tmp_path / "output.mkv"), "report_path": str(tmp_path / "report.json"),
        "result_path": str(tmp_path / "result.json"), "config": SyncConfig(device="cpu").model_dump(mode="json"),
        "work_path": str(tmp_path / "work"), "progress_path": str(tmp_path / "progress.json"),
        "qc_path": str(tmp_path / "qc.json"), "commit_path": str(tmp_path / "commit.json"),
        "fingerprint": {"source": "test"},
    }
    job = tmp_path / "job.json"
    runner.atomic_json(job, spec)
    def pipeline(**kwargs):
        calls.append(kwargs)
        kwargs["output_path"].write_bytes(b"complete media")
        runner.atomic_json(kwargs["report_path"], {"confidence": 0.9})
        return SimpleNamespace(confidence=0.9)
    monkeypatch.setattr("adsync._pipeline.run_pipeline", pipeline)
    monkeypatch.setattr("adsync.quality.verify_media_sync", lambda *a, **k: {"status": "pass"})
    monkeypatch.setattr("adsync.quality.verify_video_preservation", lambda *a, **k: {"status": "pass"}, raising=False)
    assert worker_main([str(job)]) == 0
    assert calls[0]["video_path"] == source
    assert calls[0]["config"].prepare_audio is prepare
