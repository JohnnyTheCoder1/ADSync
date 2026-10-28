"""Verify rendered AD locally before publication; retain calibrated QC evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from adsync.batch.runner import atomic_json
from adsync.hardware import thread_budget
from adsync.quality import verify_media_sync, verify_video_preservation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("media", type=Path)
    parser.add_argument("--sync-report", type=Path, help="Saved pipeline report; targets edits and sparse fitted intervals")
    parser.add_argument("--report", type=Path, help="Destination for detailed QC JSON (default beside media)")
    parser.add_argument("--source-video", type=Path, help="Also compare every compressed video stream SHA256")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    report = json.loads(args.sync_report.read_text(encoding="utf-8")) if args.sync_report else None
    with thread_budget(args.threads):
        result = verify_media_sync(args.media, work_dir=args.work_dir, report=report)
        if args.source_video:
            result["video_preservation"] = verify_video_preservation(args.source_video, args.media)
            if result["video_preservation"]["status"] != "pass":
                result["status"] = "fail"
    destination = args.report or args.media.with_suffix(".qc.json")
    atomic_json(destination, result)
    print(f"QC {result['status']}: {destination}")
    print(f"{len(result['windows'])} local windows; {len(result['failures'])} timing failures; "
          f"{len(result['review_reasons'])} unresolved checks")
    return {"pass": 0, "review": 1, "fail": 2}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
