"""Core pipeline — orchestrates the full sync workflow."""

from __future__ import annotations

import logging
from pathlib import Path

from adsync.config import SyncConfig
from adsync.models import Anchor, SegmentMap, SyncReport
from adsync.utils.tempdir import WorkDir

log = logging.getLogger("adsync")


def run_pipeline(
    *,
    video_path: Path,
    ad_path: Path,
    output_path: Path | None,
    config: SyncConfig,
    report_path: Path | None = None,
    debug_dir: Path | None = None,
    keep_temp: bool = False,
    mux: bool = True,
) -> SyncReport:
    """Execute the full ADSync pipeline and return a report."""
    from rich.progress import Progress, BarColumn, TimeRemainingColumn

    from adsync.align.anchors import find_anchors
    from adsync.align.confidence import compute_confidence
    from adsync.align.drift import estimate_drift
    from adsync.align.global_offset import estimate_global_offset
    from adsync.align.piecewise_map import build_piecewise_map
    from adsync.features.extract_basic import extract_basic_features
    from adsync.features.load import load_wav
    from adsync.features.preprocess import preprocess
    from adsync.media.extract import extract_audio
    from adsync.media.mux import mux_ad_track
    from adsync.media.probe import probe
    from adsync.rebuild.export import export_wav
    from adsync.rebuild.stitch import stitch_segments
    from adsync.report.json_report import print_summary, write_report

    progress = Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
        transient=True,
    )

    workdir = WorkDir(root=debug_dir)

    try:
        # ── Step 1: Probe ────────────────────────────────────────────────
        log.info("Step 1/12: Probing inputs")
        video_info = probe(video_path)
        ad_info = probe(ad_path)

        if not video_info.audio_streams:
            raise ValueError(f"No audio streams in video file: {video_path}")
        if not ad_info.audio_streams:
            raise ValueError(f"No audio streams in AD file: {ad_path}")

        # ── Step 2: Extract analysis WAVs ────────────────────────────────
        log.info("Step 2/12: Extracting analysis WAVs")
        vid_wav = workdir.child("video_audio.wav")
        ad_wav = workdir.child("ad_audio.wav")
        ad_hq_wav = workdir.child("ad_audio_hq.wav")
        ad_hq_sr = ad_info.audio_streams[0].sample_rate or config.output_sr

        # Run independent extractions in parallel (each spawns an FFmpeg subprocess)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [
                pool.submit(extract_audio, video_info, vid_wav, sr=config.analysis_sr, mono=config.mono),
                pool.submit(extract_audio, ad_info, ad_wav, sr=config.analysis_sr, mono=config.mono),
                pool.submit(extract_audio, ad_info, ad_hq_wav, sr=ad_hq_sr, mono=False),
            ]
            for f in as_completed(futures):
                f.result()

        # ── Step 3: Load & preprocess ────────────────────────────────────
        log.info("Step 3/12: Loading and preprocessing audio")
        y_vid, sr = load_wav(vid_wav, sr=config.analysis_sr)
        y_ad, _ = load_wav(ad_wav, sr=config.analysis_sr)

        # trim_silence=False: trimming removes different amounts of leading
        # silence from each track, destroying the timing relationship needed
        # for offset/drift estimation.
        y_vid = preprocess(y_vid, sr, highpass_hz=config.highpass_hz, trim_silence=False)
        y_ad = preprocess(y_ad, sr, highpass_hz=config.highpass_hz, trim_silence=False)

        video_duration = float(len(y_vid) / sr)
        ad_duration = float(len(y_ad) / sr)

        # ── Step 4: Compute features ────────────────────────────────────
        log.info("Step 4/12: Computing features")
        vid_feat = extract_basic_features(
            y_vid, sr,
            hop_length=config.hop_length,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            n_mfcc=config.n_mfcc,
        )
        ad_feat = extract_basic_features(
            y_ad, sr,
            hop_length=config.hop_length,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            n_mfcc=config.n_mfcc,
        )

        if debug_dir:
            _save_debug_features(vid_feat, ad_feat, workdir)

        # ── Steps 5-8: Alignment ─────────────────────────────────────────
        mode = config.mode
        global_offset: float | None = None
        drift_ppm: float | None = None
        anchors: list[Anchor] = []
        segments: list[SegmentMap] = []

        # Warp mode also needs offset/drift hints for the decoder.
        if mode in ("auto", "offset", "drift", "warp"):
            log.info("Step 5/12: Estimating global offset")
            offset, offset_conf = estimate_global_offset(
                vid_feat, ad_feat,
                y_vid=y_vid, y_ad=y_ad, sr=sr,
            )
            global_offset = offset

            log.info("Step 6/12: Estimating drift")
            with progress:
                drift_task = progress.add_task("Estimating drift", total=11)
                def _drift_progress(cur: int, tot: int) -> None:
                    progress.update(drift_task, completed=cur)
                drift, drift_conf, drift_anchors, drift_intercept = estimate_drift(
                    vid_feat, ad_feat, offset_hint=global_offset,
                    y_vid=y_vid, y_ad=y_ad, audio_sr=sr,
                    on_progress=_drift_progress,
                )
                progress.update(drift_task, completed=11)
            drift_ppm = drift
            anchors = drift_anchors

            if mode in ("offset", "drift") or (mode == "auto" and offset_conf >= config.confidence_threshold):
                # When drift is significant, use the drift intercept (offset at t=0)
                # rather than the global average offset, because stretching is
                # anchored at t=0 in the source.
                if len(drift_anchors) >= 3 and abs(drift_ppm) > 10:
                    base_offset = drift_intercept + config.offset_adjust
                    stretch = 1.0 + drift_ppm / 1e6
                else:
                    base_offset = offset + config.offset_adjust
                    stretch = 1.0

                # Clamp stretch to ±max_stretch
                stretch = max(1.0 - config.max_stretch, min(1.0 + config.max_stretch, stretch))

                dst_end = base_offset + ad_duration * stretch
                best_conf = max(offset_conf, drift_conf)

                segments = [SegmentMap(
                    src_start=0.0, src_end=ad_duration,
                    dst_start=max(0.0, base_offset),
                    dst_end=min(video_duration, dst_end),
                    offset=base_offset, stretch=stretch,
                    confidence=best_conf,
                )]

                if abs(drift_ppm) > 10 and len(drift_anchors) >= 3:
                    mode = "drift" if mode != "offset" else "offset"
                    log.info(
                        "Mode: %s with drift correction (conf=%.3f, "
                        "base_offset=%.3f s (intercept), drift=%.1f ppm = %.3f s/hr)",
                        mode, best_conf, base_offset, drift_ppm,
                        drift_ppm * 3600 / 1e6,
                    )
                else:
                    mode = "offset"
                    log.info("Mode: offset (conf=%.3f)", best_conf)
            elif mode == "warp":
                log.info(
                    "Offset/drift collected as hints for warp (offset=%.3f s, drift=%.1f ppm)",
                    global_offset, drift_ppm,
                )
            else:
                log.info("Offset mode insufficient (conf=%.3f), trying piecewise", offset_conf)

        warp_fns = None
        warp_segment_ranges = None
        warp_path_result = None

        if mode in ("auto", "warp") and not segments:
            from adsync.align.candidate_lattice import build_candidate_lattice
            from adsync.align.warp_decode import decode_warp_path
            from adsync.align.warp_fit import fit_warp_function

            log.info("Step 7/12: Building candidate lattice")
            with progress:
                lattice_task = progress.add_task("Building lattice", total=1)
                def _lattice_progress(cur: int, tot: int) -> None:
                    progress.update(lattice_task, total=tot, completed=cur)
                lattice = build_candidate_lattice(
                    vid_feat, ad_feat,
                    y_vid=y_vid, y_ad=y_ad, audio_sr=sr,
                    window_sec=config.anchor_window_sec,
                    step_sec=config.anchor_step_sec,
                    max_candidates=config.warp_max_candidates,
                    on_progress=_lattice_progress,
                )
                progress.update(lattice_task, completed=progress.tasks[lattice_task].total)

            windows_with_candidates = sum(1 for w in lattice if w.candidates)
            if windows_with_candidates < 3:
                log.warning(
                    "Only %d windows with candidates — falling back to piecewise",
                    windows_with_candidates,
                )
            else:
                log.info("Step 8/12: Decoding optimal warp path")
                warp_points, warp_cost = decode_warp_path(
                    lattice,
                    lambda_jump=config.warp_lambda_jump,
                    lambda_curve=config.warp_lambda_curve,
                    lambda_speech=config.warp_lambda_speech,
                    drift_hint_ppm=drift_ppm,
                    offset_hint=global_offset,
                )

                log.info("Step 8.5/12: Fitting continuous warp function")
                warp_fns, warp_segment_ranges, warp_path_result = fit_warp_function(
                    warp_points, ad_duration, video_duration,
                    anchor_fraction=config.warp_anchor_fraction,
                    discontinuity_threshold=config.warp_discontinuity_threshold,
                )
                warp_path_result.path_cost = warp_cost

                # Convert warp anchor points to Anchor objects for report compat
                anchors = [Anchor(
                    source_time=p.source_time,
                    target_time=p.target_time,
                    score=p.confidence,
                    window=config.anchor_window_sec,
                ) for p in warp_path_result.anchor_points]

                mode = "warp"

        # Steps 7-8 (legacy): Piecewise anchor search
        if mode in ("auto", "piecewise") and not segments and warp_fns is None:
            log.info("Step 7/12: Anchor search (piecewise)")
            with progress:
                anchor_task = progress.add_task("Searching anchors", total=1)
                def _anchor_progress(cur: int, tot: int) -> None:
                    progress.update(anchor_task, total=tot, completed=cur)
                anchors = find_anchors(
                    vid_feat, ad_feat,
                    window_sec=config.anchor_window_sec,
                    step_sec=config.anchor_step_sec,
                    on_progress=_anchor_progress,
                )
                progress.update(anchor_task, completed=progress.tasks[anchor_task].total)

            log.info("Step 8/12: Building piecewise map")
            segments = build_piecewise_map(anchors, ad_duration, video_duration, config)
            mode = "piecewise"

        # ── Confidence ───────────────────────────────────────────────────
        confidence, warnings = compute_confidence(
            anchors, segments, ad_duration, video_duration, mode=mode,
            warp_path=warp_path_result,
        )

        # Drop the analysis-rate audio (and features, unless we'll plot them)
        # before loading the HQ AD — keeps peak RSS down on long films.
        del y_vid, y_ad
        if not debug_dir:
            del vid_feat, ad_feat

        # ── Step 9: Rebuild ──────────────────────────────────────────────
        synced_y = None
        hq_sr = None

        if mux or debug_dir:
            log.info("Step 9/12: Rebuilding synced AD track (HQ)")
            y_ad_hq, hq_sr = load_wav(ad_hq_wav, sr=ad_hq_sr, mono=False)
            hq_video_duration = video_duration

            if mode == "warp" and warp_fns is not None:
                from adsync.rebuild.warp_render import render_from_warp
                synced_y = render_from_warp(
                    y_ad_hq, hq_sr, warp_fns, warp_segment_ranges,
                    hq_video_duration,
                    crossfade_ms=config.crossfade_ms,
                )
            else:
                synced_y = stitch_segments(
                    y_ad_hq, hq_sr, segments, hq_video_duration,
                    crossfade_ms=config.crossfade_ms,
                )

            if debug_dir:
                log.info("Step 10/12: Exporting debug WAV")
                synced_wav_path = workdir.child("synced_ad.wav")
                export_wav(synced_y, hq_sr, synced_wav_path)

        # ── Step 11: Encode AD and mux into container in one FFmpeg call ─
        final_output: str | None = None
        if mux and output_path and synced_y is not None:
            log.info("Step 10–11/12: Encoding & muxing final MKV")
            mux_ad_track(
                video_path, synced_y, hq_sr, output_path,
                codec=config.output_codec,
                bitrate=config.output_bitrate,
                language=config.ad_language, title=config.ad_title,
                n_existing_audio=len(video_info.audio_streams),
            )
            final_output = str(output_path)

        # ── Step 12: Report ──────────────────────────────────────────────
        log.info("Step 12/12: Generating report")
        report = SyncReport(
            mode=mode,
            confidence=confidence,
            global_offset=global_offset,
            drift_ppm=drift_ppm,
            anchors=anchors,
            segments=segments,
            warnings=warnings,
            output_path=final_output,
            warp_path=warp_path_result,
        )

        if report_path:
            write_report(report, report_path)

        if debug_dir:
            from adsync.report.plots import plot_anchors, plot_features
            plot_features(vid_feat, ad_feat, workdir.subdir("plots"))
            plot_anchors(anchors, segments, workdir.subdir("plots"))

        print_summary(report)
        return report

    finally:
        if not keep_temp:
            workdir.cleanup()



def _save_debug_features(vid_feat, ad_feat, workdir: WorkDir) -> None:
    """Dump feature CSVs for debugging."""
    import csv

    for name, feat in [("video", vid_feat), ("ad", ad_feat)]:
        # RMS CSV
        rms_path = workdir.child(f"{name}_rms.csv")
        with open(rms_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "time", "rms"])
            spf = feat.hop_length / feat.sr
            for i, v in enumerate(feat.rms):
                w.writerow([i, f"{i * spf:.4f}", f"{v:.6f}"])

        # Onset CSV
        onset_path = workdir.child(f"{name}_onset.csv")
        with open(onset_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "time", "onset"])
            spf = feat.hop_length / feat.sr
            for i, v in enumerate(feat.onset):
                w.writerow([i, f"{i * spf:.4f}", f"{v:.6f}"])
