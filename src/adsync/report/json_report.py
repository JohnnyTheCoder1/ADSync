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

    if report.global_offset is not None:
        table.add_row("Global offset", f"{report.global_offset:+.3f} s")
    if report.drift_ppm is not None:
        table.add_row("Drift", f"{report.drift_ppm:+.1f} ppm")

    table.add_row("Anchors", str(len(report.anchors)))
    table.add_row("Segments", str(len(report.segments)))

    if report.warp_path is not None:
        wp = report.warp_path
        table.add_row("Warp points", str(len(wp.points)))
        table.add_row("Warp anchors", str(len(wp.anchor_points)))
        table.add_row("Warp mean conf", f"{wp.mean_confidence:.3f}")

    if report.output_path:
        table.add_row("Output", report.output_path)

    console.print(table)

    # Warnings
    if report.warnings:
        console.print()
        console.print("[bold yellow]Warnings:[/bold yellow]")
        for w in report.warnings:
            console.print(f"  ⚠  {w}")

    # Confidence interpretation
    console.print()
    if report.confidence >= 0.90:
        console.print("[bold green]✓ High confidence — output should be reliable[/bold green]")
    elif report.confidence >= 0.75:
        console.print("[bold yellow]⚠ Medium confidence — review recommended[/bold yellow]")
    else:
        console.print("[bold red]✗ Low confidence — debug review strongly recommended[/bold red]")

    console.print()
