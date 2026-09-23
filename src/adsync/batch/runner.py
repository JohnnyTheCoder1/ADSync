"""Isolated episode workers, durable receipts, and bounded season scheduling."""

from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
from threading import Event
import time
from typing import Callable, Iterable
import uuid

from adsync import __version__
from adsync.config import SyncConfig
from adsync.batch.transaction import (ALGORITHM_REVISION, commit_matches, content_signature,
                                      read_record, same_processing_identity, signature_matches)
from adsync.media.output import validate_output_path
from adsync.quality import QC_POLICY_REVISION

log = logging.getLogger("adsync")
_SUCCESS = {"completed", "needs_review"}


class BatchStateError(RuntimeError):
    """The state directory cannot safely be used for this run."""


@dataclass
class EpisodeResult:
    label: str
    status: str
    video_path: str
    ad_path: str
    output_path: str
    report_path: str
    log_path: str
    elapsed_sec: float = 0.0
    error: str | None = None
    retained_local_path: str | None = None
    phase: str = "pending"
    qc_path: str | None = None
    commit_path: str | None = None


@dataclass
class BatchReport:
    results: list[EpisodeResult]
    state_path: str
    interrupted: bool = False

    @property
    def counts(self) -> dict[str, int]:
        return dict(Counter(result.status for result in self.results))


@dataclass
class _Job:
    result: EpisodeResult
    fingerprint: dict
    spec_path: Path
    result_path: Path
    progress_path: Path
    resume: bool = True
    attempt_id: str = field(default_factory=lambda: uuid.uuid4().hex)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical(path: str | Path) -> str:
    return os.path.normcase(str(Path(path).expanduser().resolve()))


def _signature(path: str | Path) -> dict:
    path = Path(path)
    stat = path.stat()
    if not path.is_file():
        raise ValueError(f"Not a regular file: {path}")
    return {"path": _canonical(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def atomic_json(path: Path, value: dict) -> None:
    """Replace one JSON record only after its complete bytes reach the disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def _exclusive_lock(path: Path, description: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        try:
            owner = path.read_text(encoding="utf-8")[:500]
        except OSError:
            owner = "owner unavailable"
        raise BatchStateError(
            f"{description} is locked: {path}. Owner: {owner}. "
            "If a previous run crashed, confirm its parent and worker processes have stopped before removing this lock."
        ) from exc
    try:
        owner = {"token": token, "pid": os.getpid(), "host": socket.gethostname(),
                 "started": _timestamp(), "description": description}
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(owner, handle)
            handle.flush()
            os.fsync(handle.fileno())
        yield owner
    finally:
        try:
            if json.loads(path.read_text(encoding="utf-8")).get("token") == token:
                path.unlink()
        except (OSError, ValueError):
            log.warning("Could not release run lock: %s", path)


def _output_lock_path(output_path: str | Path) -> Path:
    # Local reservations coordinate different work/state folders on this PC,
    # without requiring an available network destination before rendering.
    local = os.environ.get("LOCALAPPDATA") if os.name == "nt" else None
    base = Path(local) / "ADSync" if local else Path(tempfile.gettempdir()) / "adsync"
    key = hashlib.sha256(_canonical(output_path).encode("utf-8")).hexdigest()
    return base / "output-locks" / f"{key}.lock"


def _stop_process_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW, check=False,
        )
    else:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            proc.kill()
        proc.wait(timeout=5)


def _run_worker(job: _Job, *, config: dict, threads: int, overwrite: bool, prepare: bool, canceled: Event) -> EpisodeResult:
    result = job.result
    started = time.monotonic()
    work_path = None
    try:
        reservation = _output_lock_path(result.output_path)
        with _exclusive_lock(reservation, f"Output {result.output_path}") as owner:
            if canceled.is_set():
                result.status = "canceled"
                return result
            # Check after acquiring the reservation, closing the race between
            # independent state directories scheduling the same destination.
            progress = read_record(job.progress_path)
            recoverable = (job.resume and same_processing_identity(progress.get("fingerprint"), job.fingerprint)
                           and progress.get("phase") in {"locally_verified", "published"})
            if Path(result.output_path).exists() and not overwrite and not recoverable:
                result.status = "conflict"
                result.error = "Output already exists without a matching completion receipt; use --overwrite to replace it."
                return result
            local = os.environ.get("LOCALAPPDATA") if os.name == "nt" else None
            configured_root = os.environ.get("ADSYNC_WORK_DIR")
            work_root = (Path(os.path.expandvars(configured_root)).expanduser() if configured_root else
                         (Path(local) / "ADSync" if local else Path(tempfile.gettempdir()) / "adsync") / "episodes")
            work_root = work_root.resolve()
            work_root.mkdir(parents=True, exist_ok=True)
            previous_work = Path(progress.get("work_path", "")).resolve()
            if (job.resume and same_processing_identity(progress.get("fingerprint"), job.fingerprint) and previous_work.parent == work_root
                    and previous_work.name.startswith("episode-") and previous_work.is_dir()):
                work_path = previous_work
            else:
                work_path = Path(tempfile.mkdtemp(prefix="episode-", dir=work_root))
            spec = {
                "attempt_id": job.attempt_id, "label": result.label,
                "video_path": result.video_path, "ad_path": result.ad_path,
                "output_path": result.output_path, "report_path": result.report_path,
                "result_path": str(job.result_path), "config": config, "threads": threads,
                "prepare": prepare, "work_path": str(work_path),
                "fingerprint": job.fingerprint, "progress_path": str(job.progress_path),
                "qc_path": result.qc_path, "commit_path": result.commit_path, "overwrite": overwrite,
                "resume": job.resume,
            }
            atomic_json(job.spec_path, spec)
            env = os.environ.copy()
            for name in ("TEMP", "TMP", "TMPDIR"):
                env[name] = str(work_path)
            env["ADSYNC_STAGING_DIR"] = str(work_path / "publication")
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                         "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
                env[name] = str(max(1, threads // 2))
            env["ADSYNC_THREADS"] = str(threads)
            source_root = str(Path(__file__).resolve().parents[2])
            env["PYTHONPATH"] = source_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
            with Path(result.log_path).open("wb") as log_file:
                proc = subprocess.Popen(
                    [sys.executable, "-m", "adsync.batch.worker", str(job.spec_path)],
                    stdin=subprocess.DEVNULL, stdout=log_file, stderr=subprocess.STDOUT,
                    env=env, **options,
                )
                try:
                    owner["worker_pid"] = proc.pid
                    atomic_json(reservation, owner)
                    while proc.poll() is None:
                        if canceled.wait(0.1):
                            _stop_process_tree(proc)
                            result.status = "canceled"
                            return result
                    returncode = proc.wait()
                finally:
                    if proc.poll() is None:
                        _stop_process_tree(proc)
            if canceled.is_set():
                result.status = "canceled"
                return result
            try:
                payload = json.loads(job.result_path.read_text(encoding="utf-8"))
                if payload.get("attempt_id") != job.attempt_id:
                    raise ValueError("Worker result belongs to a previous attempt")
                status = payload["status"]
                if status not in {*_SUCCESS, "failed"}:
                    raise ValueError(f"Unknown worker status: {status}")
                if status in _SUCCESS:
                    if returncode != (0 if status == "completed" else 1):
                        raise ValueError(f"Worker exited with code {returncode} after reporting {status}")
                    if _signature(result.report_path)["size"] == 0:
                        raise ValueError("Worker output or report is empty")
                    if status == "completed" and not commit_matches(result.commit_path, job.fingerprint, verify_hash=False):
                        raise ValueError("Worker has no valid hash-bound publication commit")
                    if status == "needs_review" and not Path(payload.get("retained_local_path", "")).is_file():
                        raise ValueError("Review output was not retained locally")
                result.status = status
                result.error = payload.get("error")
                result.retained_local_path = payload.get("retained_local_path")
                result.phase = payload.get("phase", "pending")
            except (OSError, ValueError, KeyError) as exc:
                result.status = "failed"
                result.error = f"Worker exited with code {returncode} without a valid result: {exc}. See {result.log_path}"
    except BatchStateError as exc:
        result.status, result.error = "conflict", str(exc)
    except Exception as exc:
        result.status, result.error = "failed", f"{type(exc).__name__}: {exc}"
    finally:
        result.elapsed_sec = time.monotonic() - started
        if work_path is not None:
            # Only the fresh child directory allocated above is disposable.
            resolved = work_path.resolve()
            if result.status != "completed" and not result.retained_local_path:
                progress = read_record(job.progress_path)
                candidate = Path(progress.get("local_path", "")).resolve()
                if (progress.get("fingerprint") == job.fingerprint and candidate.is_relative_to(resolved)
                        and candidate.is_file()):
                    result.retained_local_path = str(candidate)
                    result.phase = progress.get("phase", "pending")
            retained = Path(result.retained_local_path).resolve() if result.retained_local_path else None
            preserve = retained is not None and retained.is_relative_to(resolved) and retained.is_file()
            if preserve:
                log.warning("[%s] Completed local output retained at: %s", result.label, retained)
            elif resolved.parent == work_root.resolve() and resolved.name.startswith("episode-"):
                shutil.rmtree(resolved, ignore_errors=True)
    return result


def run_season(
    pairs: Iterable,
    *,
    config: SyncConfig,
    jobs: int,
    threads_per_job: int,
    state_dir: Path,
    resume: bool = True,
    overwrite: bool = False,
    prepare: bool = True,
    on_event: Callable[[dict], None] | None = None,
) -> BatchReport:
    """Process independent episode pairs; only verified completion receipts resume.

    Receipts bind SHA256 content, settings, algorithm and QC policy revisions.
    Local QC precedes publication; failed transfer phases can resume without
    repeating decoding or quality checks. Failed episodes do not stop the run.
    """
    if jobs < 1 or threads_per_job < 1:
        raise ValueError("jobs and threads_per_job must be positive")
    pairs = list(pairs)
    state_dir = Path(state_dir).expanduser().resolve()
    state_path = state_dir / "state.json"
    protected = [Path(p) for pair in pairs for p in (pair.video_path, pair.ad_path, pair.output_path)]
    input_paths = [Path(p) for pair in pairs for p in (pair.video_path, pair.ad_path)]
    labels = [pair.key.label for pair in pairs]
    if len(set(labels)) != len(labels) or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", label) or label in {".", ".."} for label in labels):
        raise ValueError("Episode labels must be unique and safe filenames")
    outputs = [_canonical(pair.output_path) for pair in pairs]
    if len(set(outputs)) != len(outputs):
        raise ValueError("Two episodes cannot publish to the same output path")
    for target in (state_path, state_dir / "run.lock"):
        validate_output_path(target, protected)
    for pair in pairs:
        validate_output_path(pair.output_path, input_paths)
        for suffix in (".job.json", ".result.json", ".report.json", ".qc.json", ".commit.json", ".progress.json", ".log"):
            validate_output_path(state_dir / "episodes" / f"{pair.key.label}{suffix}", protected)
    state_dir.mkdir(parents=True, exist_ok=True)

    def emit(event: str, result: EpisodeResult) -> None:
        message = f"[{result.label}] {event}: {result.status}"
        if result.error:
            message += f" ({result.error})"
        payload = {"event": event, "label": result.label, "status": result.status,
                   "message": message, "timestamp": _timestamp()}
        if on_event:
            try:
                on_event(payload)
            except Exception:
                log.exception("Batch progress callback failed")
        else:
            log.info("%s", message)

    with _exclusive_lock(state_dir / "run.lock", "Season state"):
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if state.get("schema_version") != 1 or not isinstance(state.get("episodes"), dict):
                    raise ValueError("Unrecognized state schema")
            except (OSError, ValueError) as exc:
                raise BatchStateError(f"Cannot read season state {state_path}: {exc}") from exc
        else:
            state = {"schema_version": 1, "episodes": {}}
            atomic_json(state_path, state)
        worker_config = config.model_dump(mode="json")
        if "threads" in type(config).model_fields:
            worker_config["threads"] = threads_per_job
        worker_config["prepare_audio"] = prepare
        content_config = {k: v for k, v in worker_config.items() if k not in {"threads", "device"}}
        results: dict[str, EpisodeResult] = {}
        scheduled_jobs: dict[str, _Job] = {}
        pending: deque[_Job] = deque()

        def checkpoint(job: _Job, result: EpisodeResult) -> None:
            previous = state["episodes"].get(result.label, {})
            receipt = previous.get("receipt")
            entry = {**asdict(result), "fingerprint": job.fingerprint, "config": worker_config,
                     "prepare": prepare, "updated": _timestamp()}
            if result.status in _SUCCESS:
                try:
                    if result.status == "completed":
                        commit = read_record(result.commit_path)
                        if not commit_matches(result.commit_path, job.fingerprint, verify_hash=False):
                            raise ValueError("Publication commit is missing or changed")
                        entry["output_signature"] = {**_signature(result.output_path), "sha256": commit["output_signature"]["sha256"]}
                        entry["report_signature"] = commit["report_signature"]
                        entry["qc_signature"] = commit["qc_signature"]
                    else:
                        entry["report_signature"] = content_signature(result.report_path)
                        entry["qc_signature"] = content_signature(result.qc_path)
                        entry["local_signature"] = content_signature(result.retained_local_path)
                except (OSError, ValueError) as exc:
                    result.status = "failed"
                    result.error = f"Cannot record completed output metadata: {exc}"
                    entry.update(asdict(result))
                else:
                    receipt = entry.copy()
            if receipt is not None:
                entry["receipt"] = receipt
            state["episodes"][result.label] = entry
            state["updated"] = _timestamp()
            atomic_json(state_path, state)

        for pair in pairs:
            label = pair.key.label
            episode_dir = state_dir / "episodes"
            episode_dir.mkdir(parents=True, exist_ok=True)
            result = EpisodeResult(
                label, "pending", str(Path(pair.video_path).resolve()), str(Path(pair.ad_path).resolve()),
                str(Path(pair.output_path).resolve()), str(episode_dir / f"{label}.report.json"),
                str(episode_dir / f"{label}.log"),
                qc_path=str(episode_dir / f"{label}.qc.json"),
                commit_path=str(episode_dir / f"{label}.commit.json"),
            )
            results[label] = result
            job = _Job(result, {}, episode_dir / f"{label}.job.json", episode_dir / f"{label}.result.json",
                       episode_dir / f"{label}.progress.json", resume=resume)
            try:
                job.fingerprint = {"video": content_signature(pair.video_path), "ad": content_signature(pair.ad_path),
                                   "config": content_config, "prepare": prepare,
                                   "version": __version__, "algorithm_revision": ALGORITHM_REVISION,
                                   "qc_revision": QC_POLICY_REVISION,
                                   "output_path": _canonical(pair.output_path)}
                saved = state["episodes"].get(label, {})
                receipt = saved.get("receipt", saved)
                # The worker's final commit is authoritative even if the parent
                # died before copying its result into the season checkpoint.
                completed = resume and commit_matches(result.commit_path, job.fingerprint)
                review = (receipt.get("status") == "needs_review"
                          and signature_matches(receipt.get("retained_local_path", ""), receipt.get("local_signature"))
                          and signature_matches(result.report_path, receipt.get("report_signature"))
                          and signature_matches(result.qc_path, receipt.get("qc_signature")))
                if completed or (resume and receipt.get("fingerprint") == job.fingerprint and review):
                    if completed:
                        result.status = "completed"
                        result.phase = "transfer_verified"
                        checkpoint(job, result)
                        result.status = "skipped_completed"
                    else:
                        result.status = "skipped_needs_review"
                        result.phase = receipt.get("phase", "pending")
                        result.retained_local_path = receipt.get("retained_local_path")
                    emit("skip", result)
                    continue
            except (OSError, ValueError) as exc:
                result.status, result.error = "failed", f"Cannot inspect episode inputs/output: {exc}"
                checkpoint(job, result)
                emit("done", result)
                continue
            pending.append(job)
            scheduled_jobs[label] = job
            checkpoint(job, result)

        canceled = Event()
        interrupted = False
        with ThreadPoolExecutor(max_workers=jobs, thread_name_prefix="adsync-episode") as pool:
            active = {}
            try:
                while pending or active:
                    while pending and len(active) < jobs:
                        job = pending.popleft()
                        job.result.status = "running"
                        checkpoint(job, job.result)
                        emit("start", job.result)
                        future = pool.submit(_run_worker, job, config=worker_config, threads=threads_per_job,
                                             overwrite=overwrite, prepare=prepare, canceled=canceled)
                        active[future] = job
                    done, _ = wait(active, timeout=0.2, return_when=FIRST_COMPLETED)
                    for future in done:
                        job = active.pop(future)
                        result = future.result()
                        checkpoint(job, result)
                        emit("done", result)
            except BaseException as exc:
                canceled.set()
                interrupted = isinstance(exc, KeyboardInterrupt)
                for future, job in active.items():
                    try:
                        future.result()
                    except Exception:
                        pass
                    if job.result.status not in _SUCCESS:
                        job.result.status = "canceled"
                    checkpoint(job, job.result)
                    emit("done", job.result)
                # An interruption can land between the running checkpoint and
                # pool submission; no episode may remain recorded as running.
                for job in scheduled_jobs.values():
                    if job.result.status == "running":
                        job.result.status = "canceled"
                        checkpoint(job, job.result)
                        emit("done", job.result)
                if not interrupted:
                    raise
        return BatchReport([results[label] for label in labels], str(state_path), interrupted)
