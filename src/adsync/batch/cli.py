"""Public season workflow: preview pairs, size workers, and run resumably."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from typing import Optional

import typer

from adsync.config import SyncConfig

def default_state_dir(video: Path, ad: Path, output: Path) -> Path:
    """A stable local checkpoint directory, independent of a network output."""
    identity = "\n".join(os.path.normcase(str(p.resolve())) for p in (video, ad, output))
    key = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    return base / "ADSync" / "seasons" / key


def _probe_tasks(pairs, analysis_sr: int):
    """Probe each source once, with a small bounded pool of ffprobe processes."""
    from adsync.hardware import TaskResources
    from adsync.media.probe import probe

    def inspect(pair):
        video = probe(pair.video_path)
        ad = probe(pair.ad_path)
        if not video.video_streams or not video.audio_streams or not ad.audio_streams:
            raise ValueError(f"{pair.key.label}: video must contain video/audio, and AD must contain audio")
        video_duration = video.duration or max((s.duration or 0 for s in video.audio_streams), default=0)
        ad_duration = ad.duration or max((s.duration or 0 for s in ad.audio_streams), default=0)
        if min(video_duration, ad_duration) <= 0:
            raise ValueError(f"{pair.key.label}: could not determine positive media durations")
        stream = ad.audio_streams[0]
        return TaskResources(
            video_duration=video_duration, ad_duration=ad_duration,
            channels=stream.channels or 2, sample_rate=stream.sample_rate or 48000,
            analysis_sr=analysis_sr,
        )

    valid, tasks, issues = [], [], []
    with ThreadPoolExecutor(max_workers=min(8, len(pairs))) as pool:
        futures = [pool.submit(inspect, pair) for pair in pairs]
        for pair, future in zip(pairs, futures):
            try:
                task = future.result()
            except Exception as exc:
                issues.append(f"{pair.key.label}: {exc}")
            else:
                valid.append(pair)
                tasks.append(task)
    return valid, tasks, issues


def season(
    video_folder: Path = typer.Argument(..., exists=True, file_okay=False, help="Video folder for one show; scanned recursively"),
    ad_folder: Path = typer.Argument(..., exists=True, file_okay=False, help="AD audio folder for the same show; scanned recursively"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", envvar="ADSYNC_OUTPUT_DIR", help="Local or shared destination; default: VIDEO_FOLDER/ADSync"),
    seasons: Optional[list[int]] = typer.Option(None, "--season", min=0, help="Season to include; repeat for several, omit for all"),
    episodes: Optional[list[int]] = typer.Option(None, "--episode", min=1, help="Episode number to include within selected seasons; repeat as needed"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview file pairing without processing or creating output"),
    strict: bool = typer.Option(False, "--strict", help="Stop before processing if any selected input is unresolved"),
    jobs: Optional[int] = typer.Option(None, "--jobs", min=1, help="Concurrent episodes; default: detected CPU/RAM/GPU capacity"),
    threads: Optional[int] = typer.Option(None, "--threads", min=1, help="CPU threads per episode; default: split between jobs"),
    device: str = typer.Option("auto", "--device", envvar="ADSYNC_DEVICE", help="auto, cpu, or cuda"),
    mode: str = typer.Option("auto", "--mode", help="auto, offset, drift, piecewise, warp, or partial (experimental)"),
    language: str = typer.Option("eng", "--language", help="Original audio language to keep and AD language tag"),
    no_prep: bool = typer.Option(False, "--no-prep", help="Keep all original audio streams without stereo preparation"),
    codec: str = typer.Option("libopus", "--codec", help="Output AD audio codec"),
    bitrate: str = typer.Option("96k", "--bitrate", help="Output AD audio bitrate"),
    confidence_threshold: float = typer.Option(0.70, "--confidence-threshold", min=0.0, max=1.0),
    analysis_sr: int = typer.Option(16000, "--analysis-sr", min=1000),
    offset_adjust: float = typer.Option(0.0, "--offset-adjust", help="Manual playback nudge in seconds for every selected episode"),
    no_fingerprint: bool = typer.Option(False, "--no-fingerprint"),
    no_speed_detect: bool = typer.Option(False, "--no-speed-detect"),
    no_multiband: bool = typer.Option(False, "--no-multiband"),
    state_dir: Optional[Path] = typer.Option(None, "--state-dir", help="Local checkpoint/report/log folder; default: per-show user cache"),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Skip verified unchanged results from previous runs"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Allow replacing existing outputs that do not match a saved completion"),
) -> None:
    """Match, prepare and sync one or more seasons with hardware-sized workers."""
    from adsync.batch.discovery import DiscoveryIssue, EpisodeKey, discover_season

    try:
        config = SyncConfig(
            device=device, mode=mode, analysis_sr=analysis_sr,
            ad_language=language, output_codec=codec, output_bitrate=bitrate,
            confidence_threshold=confidence_threshold, offset_adjust=offset_adjust,
            fingerprint=not no_fingerprint, speed_detect=not no_speed_detect,
            multiband=not no_multiband,
        )
        video_folder, ad_folder = video_folder.resolve(), ad_folder.resolve()
        destination = (output_dir or video_folder / "ADSync").expanduser().resolve()
        work = (state_dir or default_state_dir(video_folder, ad_folder, destination)).expanduser().resolve()
        discovery = discover_season(video_folder, ad_folder, destination,
                                    seasons=seasons, excluded_dirs=[work])
        pairs = discovery.pairs
        issues = list(discovery.issues)
        known_keys = {p.key for p in pairs} | {i.key for i in issues if i.key is not None}
        known_seasons = {key.season for key in known_keys}
        for missing_season in sorted(set(seasons or ()) - known_seasons):
            issues.append(DiscoveryIssue(None, f"No files found for requested season {missing_season:02d}", ()))
        if episodes:
            wanted = set(episodes)
            pairs = [p for p in pairs if p.key.episode in wanted]
            issues = [i for i in issues if i.key is None or i.key.episode in wanted]
            for selected_season in sorted(set(seasons or known_seasons)):
                for episode in sorted(wanted):
                    key = EpisodeKey(selected_season, episode)
                    if key not in known_keys:
                        issues.append(DiscoveryIssue(key, "Requested episode is missing from both input folders", ()))
        typer.echo(f"Matched {len(pairs)} episode(s). Preparation: {'disabled' if no_prep else language + ' / stereo as needed'}.")
        for note in getattr(discovery, "notes", []):
            typer.echo(f"Pairing note: {note}")
        for pair in pairs:
            typer.echo(f"{pair.key.label}: {pair.video_path.name} + {pair.ad_path.name}")
            typer.echo(f"  Output: {pair.output_path}")
        for issue in issues:
            typer.echo(f"Review {issue.key.label if issue.key else 'file'}: {issue.reason}")
            for path in issue.paths:
                typer.echo(f"  {path}")
        if strict and issues:
            typer.echo("Strict mode: resolve the listed inputs before processing.")
            raise typer.Exit(2)
        if not pairs:
            typer.echo("No unambiguous episode pairs to process.")
            raise typer.Exit(2)
        if dry_run:
            typer.echo("Dry run complete. No media or output files were written.")
            raise typer.Exit(1 if issues else 0)

        from adsync.hardware import detect_hardware, plan_resources
        from adsync.batch.runner import atomic_json, run_season

        hardware = detect_hardware(device=device)
        pairs, tasks, probe_issues = _probe_tasks(pairs, analysis_sr)
        for message in probe_issues:
            typer.echo(f"Review: {message}")
        if not pairs or (strict and probe_issues):
            raise typer.Exit(2)
        resources = plan_resources(hardware, tasks, requested_jobs=jobs,
                                   requested_threads=threads, device=device)
        if config.device == "auto" and resources.device == "cpu":
            config = config.model_copy(update={"device": "cpu"})
        gib = 1024 ** 3
        typer.echo(f"Hardware: {hardware.usable_cpus} usable CPU threads, "
                   f"{hardware.available_memory_bytes / gib:.1f} GiB available RAM; "
                   f"GPU: {hardware.gpu_name or 'none'}")
        typer.echo(f"Processing: {resources.jobs} episode job(s), "
                   f"{resources.threads_per_job} CPU threads per job, {resources.device} correlations.")
        for reason in resources.reasons:
            typer.echo(f"  {reason}")
        typer.echo(f"Checkpoints, reports and logs: {work}")
        result = run_season(pairs, config=config, jobs=resources.jobs,
                            threads_per_job=resources.threads_per_job, state_dir=work,
                            resume=resume, overwrite=overwrite, prepare=not no_prep)
        summary_path = work / "season-summary.json"
        from adsync.media.output import validate_output_path
        protected = [path for pair in pairs for path in (pair.video_path, pair.ad_path, pair.output_path)]
        validate_output_path(summary_path, protected)
        summary = {
            "hardware": asdict(hardware), "resources": asdict(resources),
            "pairing_notes": getattr(discovery, "notes", []),
            "unresolved": [asdict(issue) for issue in issues], "probe_errors": probe_issues,
            "counts": result.counts, "batch": asdict(result),
        }
        atomic_json(summary_path, json.loads(json.dumps(summary, default=str)))
        typer.echo("Season summary: " + ", ".join(f"{count} {status.replace('_', ' ')}"
                                                for status, count in sorted(result.counts.items())))
        for item in result.results:
            if item.error:
                typer.echo(f"{item.label}: {item.error}")
            if item.retained_local_path:
                typer.echo(f"  Completed local file: {item.retained_local_path}")
        typer.echo(f"State: {result.state_path}")
        typer.echo(f"Summary: {summary_path}")
        if result.interrupted:
            raise typer.Exit(130)
        if any(result.counts.get(s, 0) for s in ("failed", "conflict", "canceled", "pending")):
            raise typer.Exit(2)
        if issues or probe_issues or any(result.counts.get(s, 0) for s in ("needs_review", "skipped_needs_review")):
            raise typer.Exit(1)
    except typer.Exit:
        raise
    except (OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(2)
