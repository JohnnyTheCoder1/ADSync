"""Carry a source's extra streams into Matroska in a form it can hold."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from adsync.utils.subprocesses import run_ffmpeg, run_ffprobe


@dataclass
class MkvStreamPlan:
    """ffmpeg options around one source input and the MKV output.

    Attachments and cover art are re-attached from local copies (mapped cover
    art would become a one-frame video track), MP4 text captions become SRT,
    and data tracks Matroska cannot hold are left out.
    """

    inputs: list[str] = field(default_factory=list)   # before the source's -i
    mapping: list[str] = field(default_factory=list)  # after the source's -map options
    outputs: list[str] = field(default_factory=list)  # after the codec options


def mkv_stream_plan(source: Path, stage_dir: Path) -> MkvStreamPlan:
    """Plan attachments, cover art, captions and data tracks of *source*; files go in *stage_dir*."""
    details = json.loads(run_ffprobe([
        "-v", "error", "-show_entries",
        "stream=index,codec_type,codec_name:stream_disposition=attached_pic:stream_tags",
        "-of", "json", str(source),
    ]).stdout)
    plan = MkvStreamPlan()
    attached = subtitles = 0
    for stream in details.get("streams", []):
        index, kind = stream["index"], stream.get("codec_type")
        if kind == "subtitle":
            if stream.get("codec_name") == "mov_text":
                plan.outputs += [f"-c:s:{subtitles}", "srt"]
            subtitles += 1
            continue
        if kind == "data":
            plan.mapping += ["-map", f"-0:{index}"]
            continue
        picture = kind == "video" and bool(stream.get("disposition", {}).get("attached_pic"))
        if kind != "attachment" and not picture:
            continue

        local = stage_dir / f"attachment-{attached:04d}.bin"
        tags = dict(stream.get("tags", {}))
        if picture:
            extension = "png" if stream.get("codec_name") == "png" else "jpg"
            run_ffmpeg(["-i", str(source), "-map", f"0:{index}", "-c", "copy", "-frames:v", "1",
                        "-update", "1", "-f", "image2", str(local)])
            defaults = {"filename": f"cover.{extension}", "mimetype": f"image/{'png' if extension == 'png' else 'jpeg'}"}
        else:
            plan.inputs += [f"-dump_attachment:{index}", str(local)]
            defaults = {"filename": f"attachment-{attached:04d}.bin", "mimetype": "application/octet-stream"}
        for key, value in defaults.items():
            if not any(existing.casefold() == key for existing in tags):
                tags[key] = value
        plan.mapping += ["-map", f"-0:{index}"]
        plan.outputs += ["-attach", str(local)]
        for key, value in tags.items():
            plan.outputs += [f"-metadata:s:t:{attached}", f"{key}={value}"]
        attached += 1
    return plan
