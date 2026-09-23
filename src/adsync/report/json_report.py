"""JSON and terminal report generation."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from rich.console import Console
from rich.table import Table

from adsync.models import SyncReport
from adsync.utils.io import write_json

log = logging.getLogger("adsync")
console = Console()


def write_report(report: SyncReport, path: Path | str) -> Path:
    """Serialize the sync report to a JSON file."""
    return write_json(path, report.model_dump())


def print_summary(report: SyncReport) -> None:
    """Print a concise summary to the terminal."""
    console.print()
    console.rule("[bold]ADSync Report[/bold]")

    # Key metrics
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column()

    table.add_row("Mode", report.mode)
    table.add_row("Confidence", f"{report.confidence:.2%}")
    if report.identity_check:
        table.add_row("Content identity", report.identity_check.get("status", "unknown"))
    if report.quality_check:
        table.add_row("Local output checks", report.quality_check.get("status", "unknown"))
    table.add_row("Compute", f"{report.compute_backend} ({report.gpu_correlations} GPU, {report.cpu_correlations} CPU correlations)")
    if report.compute_device:
        table.add_row("GPU", report.compute_device)

    if report.speed_stretch is not None:
        table.add_row(
            "Speed correction",
            f"×{report.speed_stretch:.6f} ({(report.speed_stretch - 1) * 100:+.2f}% — pitch restored)",
        )
    if report.fp_anchor_windows:
        table.add_row("Fingerprint anchors", f"{report.fp_anchor_windows} windows")

    if report.global_offset is not None:
        table.add_row("Global offset", f"{report.global_offset:+.3f} s")
    if report.drift_ppm is not None:
        table.add_row("Drift", f"{report.drift_ppm:+.1f} ppm")

    table.add_row("Anchors", str(len(report.anchors)))
    if report.mode == "warp" and report.warp_path is not None:
        table.add_row("Segments", str(report.warp_path.n_segments))
    else:
        table.add_row("Segments", str(len(report.segments)))

    if report.warp_path is not None:
        wp = report.warp_path
        table.add_row("Warp points", str(len(wp.points)))
        table.add_row("Warp anchors", str(len(wp.anchor_points)))
        table.add_row("Warp mean conf", f"{wp.mean_confidence:.3f}")
        if wp.dropped_segments:
            table.add_row("Dropped excursions", str(wp.dropped_segments))
        if wp.bridged_sec > 0:
            table.add_row("Bridged", f"{wp.bridged_sec:.0f} s (coasting, no evidence)")

    if report.fp_residual_p95_ms is not None:
        p50 = f"{report.fp_residual_p50_ms:.0f}" if report.fp_residual_p50_ms is not None else "?"
        table.add_row(
            "Verification",
            f"median {p50} ms, p95 {report.fp_residual_p95_ms:.0f} ms vs fingerprint",
        )

    if report.output_path:
        table.add_row("Output", report.output_path)

    console.print(table)

    if report.mode == "partial":
        _print_partial_map(report)
    else:
        _print_timing_map(report)

    # Warnings
    if report.warnings:
        console.print()
        console.print("[bold yellow]Warnings:[/bold yellow]")
        for w in report.warnings:
            console.print(f"  Warning: {w}")

    # Confidence interpretation
    console.print()
    if report.alignment_review_required:
        console.print("[bold yellow]Review required: the alignment has unresolved evidence.[/bold yellow]")
    elif report.confidence >= 0.90:
        console.print("[bold green]High alignment confidence[/bold green]")
    elif report.confidence >= 0.75:
        console.print("[bold yellow]Medium confidence — review recommended[/bold yellow]")
    else:
        console.print("[bold red]Low confidence — debug review strongly recommended[/bold red]")

    console.print()


def _print_partial_map(report: SyncReport, max_rows: int = 12) -> None:
    detail = report.timing_debug.get("partial_alignment", {})
    adjustment = detail.get("playback_offset_adjust_sec", 0.0)
    rows = []
    for span in detail.get("matched_intervals", []):
        rows.append(f"AD {_mmss(span['source_start'])}-{_mmss(span['source_end'])} -> "
                    f"video {_mmss(span['target_start'])}-{_mmss(span['target_end'])}")
    for key, title in (("source_gaps", "Unmeasured AD"), ("target_gaps", "Unmeasured video"),
                       ("ambiguous_ranges", "Competing matches at AD")):
        for span in detail.get(key, []):
            rows.append(f"{title} {_mmss(span['start_sec'])}-{_mmss(span['end_sec'])}")
    if rows:
        console.print()
        console.print("[bold]Partial alignment:[/bold]")
        if adjustment:
            console.print(f"  Measured coordinates before playback adjustment {adjustment:+.2f} s")
        for row in rows[:max_rows]:
            console.print(f"  {row}")
        if len(rows) > max_rows:
            console.print(f"  {len(rows) - max_rows} more regions in the JSON report")


def _print_timing_map(report: SyncReport, jump_threshold: float = 2.0, max_rows: int = 12) -> None:
    """Plain-language map of where the AD sits relative to the video."""
    if report.warp_path is None or len(report.warp_path.points) < 2:
        return
    pts = report.warp_path.points
    runs: list[tuple[float, float, float]] = []  # (ad_start, ad_end, median_offset)
    start = 0
    offsets = [p.target_time - p.source_time for p in pts]
    for i in range(1, len(pts) + 1):
        if i == len(pts) or abs(offsets[i] - offsets[i - 1]) > jump_threshold:
            seg = offsets[start:i]
            mid = sorted(seg)[len(seg) // 2]
            runs.append((pts[start].source_time, pts[i - 1].source_time, mid))
            start = i

    console.print()
    console.print("[bold]Timing map (AD time, offset into video):[/bold]")
    for ad_start, ad_end, off in runs[:max_rows]:
        console.print(f"  {_mmss(ad_start)}–{_mmss(ad_end)}  sits at {off:+.2f} s")
    if len(runs) > max_rows:
        console.print(f"  … and {len(runs) - max_rows} more segment(s)")


def _mmss(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
