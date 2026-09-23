"""Local quality gates and durable, hash-bound publication receipts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

from adsync.media.output import destination_publication_lock, sha256_file


# Deliberately separate from the package version: semantic fixes invalidate
# completed runs even while the development application remains at 0.1.0.
ALGORITHM_REVISION = "2026-09-local-evidence-refined-boundaries-v2"


def same_processing_identity(previous: dict | None, current: dict) -> bool:
    """A QC-policy update invalidates the verdict, not identical rendered media."""
    if not previous:
        return False
    return ({key: value for key, value in previous.items() if key != "qc_revision"}
            == {key: value for key, value in current.items() if key != "qc_revision"})


def _qc_summary(qc: dict, policy_revision: str | None) -> dict:
    """Keep the report readable; full per-window evidence remains in QC JSON."""
    return {"status": qc.get("status"), "policy_revision": qc.get("policy_revision", policy_revision),
            "counts": {key: len(qc.get(key) or []) for key in ("windows", "failures", "review_reasons")},
            "video_preservation": qc.get("video_preservation", {})}


def content_signature(path: str | Path) -> dict:
    path = Path(path)
    stat = path.stat()
    if not path.is_file():
        raise ValueError(f"Not a regular file: {path}")
    digest = sha256_file(path)
    after = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise OSError(f"File changed while recording content signature: {path}")
    return {"path": os.path.normcase(str(path.resolve())), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "sha256": digest}


def signature_matches(path: str | Path, signature: dict | None, *, verify_hash: bool = True) -> bool:
    if not isinstance(signature, dict) or not signature.get("sha256"):
        return False
    try:
        path = Path(path)
        stat = path.stat()
        if not path.is_file() or stat.st_size <= 0:
            return False
        metadata = {"path": os.path.normcase(str(path.resolve())), "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns}
        if any(signature.get(key) != value for key, value in metadata.items()):
            return False
        return not verify_hash or sha256_file(path) == signature["sha256"]
    except (OSError, ValueError):
        return False


def read_record(path: str | Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def commit_matches(path: str | Path, fingerprint: dict, *, verify_hash: bool = True) -> bool:
    commit = read_record(path)
    if commit.get("phase") != "transfer_verified" or commit.get("fingerprint") != fingerprint:
        return False
    for artifact in ("output", "report", "qc"):
        signature = commit.get(f"{artifact}_signature", {})
        artifact_path = signature.get("path")
        if not artifact_path:
            return False
        # Avoid competing destination integrity reads across season workers.
        if artifact == "output" and verify_hash:
            with destination_publication_lock(artifact_path):
                valid = signature_matches(artifact_path, signature)
        else:
            valid = signature_matches(artifact_path, signature, verify_hash=verify_hash)
        if not valid:
            return False
    return True


def execute_transaction(spec: dict, config, *, run_pipeline: Callable, verify_media_sync: Callable,
                        verify_video_preservation: Callable) -> dict:
    """Resume processed/QC/published phases; the commit marker is written last.

    A multi-file publication cannot be one filesystem rename. Consumers must
    require the final commit marker, which binds the output, report and QC to
    source content, settings and both policy revisions. An interrupted bundle
    without that marker remains recoverable but is never accepted as complete.
    """
    from adsync.batch.runner import atomic_json
    from adsync.media.output import publish_completed_file

    work = Path(spec["work_path"])
    work.mkdir(parents=True, exist_ok=True)
    destination = Path(spec["output_path"])
    report_path = Path(spec["report_path"])
    qc_path = Path(spec["qc_path"])
    progress_path = Path(spec["progress_path"])
    commit_path = Path(spec["commit_path"])
    fingerprint = spec["fingerprint"]
    progress = read_record(progress_path)
    if not spec.get("resume", True) or not same_processing_identity(progress.get("fingerprint"), fingerprint):
        progress = {}
    elif progress.get("fingerprint") != fingerprint:
        progress["fingerprint"] = fingerprint
        progress["phase"] = "processed"
        progress.pop("local_qc_signature", None)
        progress.pop("qc_status", None)
        atomic_json(progress_path, progress)
    local_output = work / "rendered" / destination.name
    local_report = work / "report.json"
    local_qc = work / "qc.json"
    processed = (signature_matches(local_output, progress.get("local_signature"))
                 and signature_matches(local_report, progress.get("local_report_signature")))
    if not processed:
        # The public report describes the eventual library path; recovery paths
        # belong to the journal, so deleting staging never leaves broken links.
        local_output.parent.mkdir(parents=True, exist_ok=True)
        run_pipeline(video_path=Path(spec["video_path"]), ad_path=Path(spec["ad_path"]),
                     output_path=local_output, report_path=local_report, config=config, mux=True)
        report = read_record(local_report)
        identity = report.get("identity_check") or {}
        if identity.get("status") == "fail":
            # Identity preflight deliberately produces no rendered media. Save
            # its detailed evidence outside disposable staging before failing.
            reasons = identity.get("reasons") or ["Distributed content identity was not established"]
            qc = {"status": "fail", "stage": "content_identity", "identity": identity,
                  "policy_revision": fingerprint.get("qc_revision"), "failures": reasons,
                  "review_reasons": [], "windows": []}
            report["output_path"] = None
            report["quality_check"] = _qc_summary(qc, fingerprint.get("qc_revision"))
            atomic_json(report_path, report)
            atomic_json(qc_path, qc)
            atomic_json(progress_path, {"fingerprint": fingerprint, "phase": "identity_failed",
                                       "work_path": str(work), "report_signature": content_signature(report_path),
                                       "qc_signature": content_signature(qc_path)})
            raise ValueError("Content identity failed; output withheld: " + "; ".join(reasons))
        if report.get("alignment_review_required") and (
            not report.get("output_path") or not local_output.is_file() or local_output.stat().st_size == 0
        ):
            # A rejected partial map is review evidence, not a failed render.
            reason = "Partial alignment has no accepted audio map; output requires review."
            qc = {"status": "review", "stage": "partial_alignment",
                  "policy_revision": fingerprint.get("qc_revision"), "failures": [],
                  "review_reasons": [reason], "windows": []}
            report["output_path"] = None
            report["quality_check"] = _qc_summary(qc, fingerprint.get("qc_revision"))
            atomic_json(local_report, report)
            atomic_json(local_qc, qc)
            atomic_json(report_path, report)
            atomic_json(qc_path, qc)
            progress = {"fingerprint": fingerprint, "phase": "alignment_review_required",
                        "work_path": str(work), "report_signature": content_signature(report_path),
                        "qc_signature": content_signature(qc_path)}
            atomic_json(progress_path, progress)
            return {"status": "needs_review", "confidence": float(report.get("confidence", 0)),
                    "phase": progress["phase"], "error": reason}
        if not report or not local_output.is_file() or local_output.stat().st_size == 0:
            raise ValueError("Pipeline did not produce complete local media and report")
        report["output_path"] = str(destination)
        atomic_json(local_report, report)
        progress = {"fingerprint": fingerprint, "phase": "processed", "work_path": str(work),
                    "local_path": str(local_output), "local_signature": content_signature(local_output),
                    "local_report_signature": content_signature(local_report)}
        atomic_json(progress_path, progress)
    report = read_record(local_report)
    qc_cached = signature_matches(local_qc, progress.get("local_qc_signature"))
    if not qc_cached:
        qc = verify_media_sync(local_output, work_dir=work / "quality", report=report)
        video_proof = verify_video_preservation(Path(spec["video_path"]), local_output)
        qc["video_preservation"] = video_proof
        if video_proof.get("status") != "pass":
            qc["status"] = "fail" if video_proof.get("status") == "fail" else "review"
            qc.setdefault("failures", []).append("Original video stream was not verified as unchanged.")
        atomic_json(local_qc, qc)
        progress["local_qc_signature"] = content_signature(local_qc)
        progress["qc_status"] = qc.get("status")
        progress["phase"] = "locally_verified" if qc.get("status") == "pass" else "processed"
        atomic_json(progress_path, progress)
    # Input hashes were captured before any worker read. Verify again even
    # when reusing QC, so a source replacement during this run cannot attach a
    # passing receipt to the previous source content.
    for source in ("video", "ad"):
        expected_source = fingerprint.get(source)
        if expected_source and not signature_matches(spec[f"{source}_path"], expected_source):
            progress["phase"] = "source_changed"
            for key in ("local_signature", "local_report_signature", "local_qc_signature"):
                progress.pop(key, None)
            atomic_json(progress_path, progress)
            raise ValueError(f"{source.upper()} source changed during processing; publication refused")
    qc = read_record(local_qc)
    alignment_review_required = bool(report.get("alignment_review_required"))
    if alignment_review_required:
        # Apply this even to cached passing QC; signal agreement cannot approve gaps.
        reason = "Partial alignment contains gaps or ambiguity that require review."
        if qc.get("status") == "pass":
            qc["status"] = "review"
        reasons = qc.setdefault("review_reasons", [])
        if reason not in reasons:
            reasons.append(reason)
        atomic_json(local_qc, qc)
        progress["local_qc_signature"] = content_signature(local_qc)
        progress["qc_status"] = qc.get("status")
        progress["phase"] = "alignment_review_required"
        atomic_json(progress_path, progress)
    summary = _qc_summary(qc, fingerprint.get("qc_revision"))
    if report.get("quality_check") != summary:
        report["quality_check"] = summary
        atomic_json(local_report, report)
        progress["local_report_signature"] = content_signature(local_report)
        atomic_json(progress_path, progress)
    confidence = float(report.get("confidence", 0))
    if alignment_review_required or qc.get("status") != "pass" or confidence < config.confidence_threshold:
        # Save review evidence without placing unapproved media in the library.
        atomic_json(report_path, report)
        atomic_json(qc_path, qc)
        return {"status": "needs_review", "confidence": confidence,
                "phase": progress["phase"], "retained_local_path": str(local_output),
                "error": ("Alignment requires review; output has not been published."
                          if alignment_review_required else
                          "Local quality checks require review; output has not been published.")}

    # If publication succeeded just before a crash, recognize exactly those
    # bytes. Never infer permission to replace a different existing output.
    expected = progress["local_signature"]["sha256"]
    existing_matches = False
    if destination.is_file():
        with destination_publication_lock(destination):
            existing_matches = (destination.stat().st_size == progress["local_signature"]["size"]
                                and sha256_file(destination) == expected)
    if not existing_matches:
        publish_completed_file(local_output, destination, expected_sha256=expected,
                               overwrite=spec.get("overwrite", False))
    progress["phase"] = "published"
    # The destination's bytes were checked before atomic rename by publication.
    # Reuse that proof instead of reading the HDD a second time.
    stat = destination.stat()
    output_signature = {"path": os.path.normcase(str(destination.resolve())), "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns, "sha256": expected}
    progress["output_signature"] = output_signature
    atomic_json(progress_path, progress)
    atomic_json(report_path, report)
    atomic_json(qc_path, qc)
    commit = {"schema_version": 1, "phase": "transfer_verified", "status": "completed",
              "fingerprint": fingerprint, "confidence": confidence,
              "output_signature": output_signature, "report_signature": content_signature(report_path),
              "qc_signature": content_signature(qc_path)}
    atomic_json(commit_path, commit)
    progress["phase"] = "transfer_verified"
    atomic_json(progress_path, progress)
    return {"status": "completed", "confidence": confidence, "phase": "transfer_verified",
            "commit_path": str(commit_path)}
