"""Private subprocess entry point for one season episode."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import traceback


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("Usage: python -m adsync.batch.worker JOB.json", file=sys.stderr)
        return 2
    spec = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    threads = int(spec["threads"])
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = str(max(1, threads // 2))
    os.environ["ADSYNC_THREADS"] = str(threads)

    from adsync._pipeline import run_pipeline
    from adsync.batch.runner import atomic_json
    from adsync.batch.transaction import execute_transaction, read_record
    from adsync.config import SyncConfig
    from adsync.logging import setup_logging
    from adsync.media.output import OutputPublicationError

    setup_logging()
    payload = {"attempt_id": spec["attempt_id"], "status": "failed"}
    code = 2
    try:
        config = SyncConfig.model_validate({**spec["config"], "prepare_audio": spec.get("prepare", True)})
        from adsync.quality import verify_media_sync, verify_video_preservation
        payload.update(execute_transaction(spec, config, run_pipeline=run_pipeline,
                                           verify_media_sync=verify_media_sync,
                                           verify_video_preservation=verify_video_preservation))
        code = 0 if payload["status"] == "completed" else 1
    except OutputPublicationError as exc:
        payload["error"] = str(exc)
        payload["retained_local_path"] = str(exc.local_path)
        traceback.print_exc()
    except Exception as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    # A report/receipt write can fail after rendering just like a transfer can.
    # Keep the local artifact and journal for both kinds of failure.
    progress = read_record(spec.get("progress_path", ""))
    if progress.get("phase"):
        payload.setdefault("phase", progress["phase"])
    if payload["status"] != "completed" and progress.get("local_path"):
        payload.setdefault("retained_local_path", progress["local_path"])
        payload.setdefault("phase", progress.get("phase"))
    atomic_json(Path(spec["result_path"]), payload)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
