"""Core pipeline — orchestrates the full sync workflow."""

from __future__ import annotations

import logging
from contextlib import ExitStack
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
    from adsync.compute import CorrelationBackend
    from adsync.hardware import effective_thread_count, thread_budget, worker_initializer
    from adsync.utils.progress import Progress

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
    from adsync.media.output import OutputPublicationError, validate_output_path
    from adsync.media.probe import probe
    from adsync.rebuild.export import export_wav
    from adsync.rebuild.stitch import stitch_segments
    from adsync.report.json_report import print_summary, write_report

    if mux and output_path is not None:
        validate_output_path(output_path, [video_path, ad_path])
    if report_path is not None:
        protected = [video_path, ad_path]
        if output_path is not None:
            protected.append(output_path)
        validate_output_path(report_path, protected)
    if debug_dir is not None:
        protected = [p for p in (video_path, ad_path, output_path, report_path) if p is not None]
        for name in (
            "video_audio.wav", "ad_audio.wav", "ad_audio_hq.wav", "synced_ad.wav",
            "video_rms.csv", "video_onset.csv", "ad_rms.csv", "ad_onset.csv",
            "plots/rms_comparison.png", "plots/onset_comparison.png",
            "plots/anchors.png", "plots/segments.png",
        ):
            validate_output_path(debug_dir / name, protected)

    progress = Progress()

    workdir = WorkDir(root=debug_dir)
    compute = None
    resources = ExitStack()
    if not keep_temp:
        resources.callback(workdir.cleanup)

    try:
        threads = effective_thread_count(config.threads)
        resources.enter_context(thread_budget(threads))
        compute = CorrelationBackend(config.device)
        resources.callback(compute.close)
        log.info("Per-process CPU budget: %d threads", threads)
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

        # On multi-audio sources, analyze the stream matching the AD language
        # rather than blindly the first — dual-audio releases often put a dub
        # first, and correlating the AD against a dub only matches on the
        # shared music/effects bed.
        vid_audio_index: int | None = None
        original_audio = None
        if config.prepare_audio or len(video_info.audio_streams) > 1:
            from adsync.media.prep import pick_audio_stream
            picked = pick_audio_stream(video_info, language=config.ad_language, strict=config.prepare_audio)
            if config.prepare_audio:
                tag = (picked.language or "").lower()
                if config.ad_language and tag not in {"", "und"} and tag[:2] != config.ad_language.lower()[:2]:
                    raise ValueError(f"Requested {config.ad_language!r} audio, but {video_path.name} provides {tag!r}")
                original_audio = picked
            vid_audio_index = picked.index
            log.info(
                "Analysis audio: stream %d (lang=%s, %d ch)",
                picked.index, picked.language or "untagged", picked.channels or 0,
            )

        # Cache analysis using source content, selected stream and semantic settings.
        # Explicit debug runs still materialize all requested diagnostic WAVs.
        from adsync.cache import content_key, default_cache, derived_key
        cache = default_cache() if debug_dir is None else None
        source_keys = [None, None]
        source_options = [None, None]
        cached_audio = [None, None]
        source_specs = [(video_path, video_info, vid_wav, vid_audio_index),
                        (ad_path, ad_info, ad_wav, None)]
        if cache is not None:
            for index, (source, _, _, stream) in enumerate(source_specs):
                if source.is_file():
                    source_options[index] = {
                        "kind": "preprocessed-audio", "stream": stream,
                        "sr": config.analysis_sr, "mono": config.mono,
                        "highpass_hz": config.highpass_hz, "trim_silence": False,
                    }
                    source_keys[index] = content_key(source, source_options[index])
                    artifact = cache.get(source_keys[index])
                    if artifact is not None and set(artifact) == {"audio"} and artifact["audio"].ndim == 1:
                        cached_audio[index] = artifact["audio"]
                        log.info("Analysis audio cache hit: %s", source.name)

        # Run independent uncached extractions in parallel.
        from concurrent.futures import ThreadPoolExecutor, as_completed
        pair_workers = min(2, threads)
        extraction_threads = max(1, threads // pair_workers)
        with ThreadPoolExecutor(max_workers=pair_workers, initializer=worker_initializer(threads)) as pool:
            futures = [pool.submit(extract_audio, info, wav, sr=config.analysis_sr,
                                   mono=config.mono, stream_index=stream, threads=extraction_threads)
                       for index, (_, info, wav, stream) in enumerate(source_specs)
                       if cached_audio[index] is None]
            for f in as_completed(futures):
                f.result()

        # ── Step 3: Load & preprocess ────────────────────────────────────
        log.info("Step 3/12: Loading and preprocessing audio")
        # trim_silence=False: trimming removes different amounts of leading
        # silence from each track, destroying the timing relationship needed
        # for offset/drift estimation.
        sr = config.analysis_sr
        for index, (_, _, wav, _) in enumerate(source_specs):
            if cached_audio[index] is None:
                y, _ = load_wav(wav, sr=sr)
                cached_audio[index] = preprocess(y, sr, highpass_hz=config.highpass_hz, trim_silence=False)
                if cache is not None and source_keys[index] is not None:
                    if content_key(source_specs[index][0], source_options[index]) != source_keys[index]:
                        raise RuntimeError(f"Source changed during analysis decode: {source_specs[index][0]}")
                    cache.put(source_keys[index], {"audio": cached_audio[index]})
        y_vid, y_ad = cached_audio

        video_duration = float(len(y_vid) / sr)
        ad_duration = float(len(y_ad) / sr)

        # ── Step 4: Compute features ────────────────────────────────────
        log.info("Step 4/12: Computing features")
        # Video and AD features are independent; librosa's FFT/BLAS work
        # releases the GIL, so two threads cut this step roughly in half.
        feat_kwargs = dict(
            hop_length=config.hop_length,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            n_mfcc=config.n_mfcc,
        )
        def features_for(y, index):
            from adsync.models import FeatureBundle
            key = derived_key(source_keys[index], {"kind": "basic-features", **feat_kwargs}) if source_keys[index] else None
            artifact = cache.get(key) if cache is not None and key else None
            if (artifact is not None and set(artifact) == {"rms", "onset", "mel", "mfcc"}
                    and artifact["rms"].ndim == artifact["onset"].ndim == 1
                    and artifact["mel"].ndim == artifact["mfcc"].ndim == 2
                    and artifact["rms"].shape == artifact["onset"].shape
                    and artifact["mel"].shape == (config.n_mels, len(artifact["rms"]))
                    and artifact["mfcc"].shape == (config.n_mfcc, len(artifact["rms"]))):
                log.info("Basic feature cache hit: %s", source_specs[index][0].name)
                return FeatureBundle(sr=sr, hop_length=config.hop_length, duration=len(y) / sr, **artifact)
            result = extract_basic_features(y, sr, **feat_kwargs)
            if cache is not None and key:
                cache.put(key, {name: getattr(result, name) for name in ("rms", "onset", "mel", "mfcc")})
            return result

        with ThreadPoolExecutor(max_workers=pair_workers, initializer=worker_initializer(threads)) as pool:
            vid_feat_future = pool.submit(features_for, y_vid, 0)
            ad_feat_future = pool.submit(features_for, y_ad, 1)
            vid_feat = vid_feat_future.result()
            ad_feat = ad_feat_future.result()

        if debug_dir:
            _save_debug_features(vid_feat, ad_feat, workdir)

        # ── Step 4.5: Fingerprint landmark matching ──────────────────────
        # Global, radius-free evidence: which stretch of the AD sits at which
        # offset, and which stretches match nowhere (bad spots), before any
        # windowed correlation runs.
        fp = None
        if config.fingerprint:
            log.info("Step 4.5/12: Fingerprint landmark matching")
            try:
                from adsync.align.fingerprint import fingerprint_align
                fp = fingerprint_align(y_vid, y_ad, sr)
            except Exception:
                log.exception("Fingerprinting failed — continuing without it")
                fp = None

        # ── Step 4.6: Speed detection & correction ───────────────────────
        # A PAL-style transfer (AD ~4% fast, pitch up) makes the pair look
        # like a total mismatch: probe standard ratios, measure the exact
        # stretch, and correct speed and pitch before any alignment runs.
        speed_stretch: float | None = None
        extra_warnings: list[str] = []
        if config.speed_detect and fp is not None:
            from adsync.align.speed import detect_speed, stretch_fraction

            try:
                det = detect_speed(y_vid, y_ad, sr, baseline=fp)
            except Exception:
                log.exception("Speed detection failed — continuing at native speed")
                det = None

            if det is not None:
                log.info(
                    "Step 4.6/12: AD is speed-shifted ×%.6f (%+.2f%%) — correcting "
                    "speed and pitch (PAL-style transfer)",
                    det.stretch, det.percent,
                )
                from scipy.signal import resample_poly

                up, down = stretch_fraction(det.stretch)
                y_ad = resample_poly(y_ad, up, down).astype(y_ad.dtype, copy=False)
                ad_duration = float(len(y_ad) / sr)
                ad_feat = extract_basic_features(y_ad, sr, **feat_kwargs)
                fp = det.fp
                speed_stretch = det.stretch
                extra_warnings.append(
                    f"AD was speed-corrected by ×{det.stretch:.6f} ({det.percent:+.2f}%) — "
                    "PAL-style transfer detected; original speed and pitch restored"
                )
                # Decode HQ only at rebuild time, with this correction applied.

        identity_check: dict = {}
        identity_failed = False
        if fp is not None:
            from adsync.quality import assess_content_identity
            identity_check = assess_content_identity(fp, ad_duration, video_duration=video_duration)
            identity_failed = identity_check["status"] == "fail"
            if identity_check["status"] != "pass":
                extra_warnings.append(
                    "Content identity needs review: " + "; ".join(identity_check["reasons"])
                )
            if identity_failed:
                extra_warnings.append("Output withheld: distributed content identity was not established")
                log.warning("Content identity failed; alignment evidence will be reported without rendering")

        # Distinct fingerprint offset spans mean the material has edits a
        # single global offset cannot represent.
        fp_edits = (
            fp is not None and fp.strong and len(fp.spans) >= 2
            and (max(s.offset for s in fp.spans) - min(s.offset for s in fp.spans)) > 0.25
        )

        # ── Steps 5-8: Alignment ─────────────────────────────────────────
        mode = config.mode
        global_offset: float | None = None
        drift_ppm: float | None = None
        anchors: list[Anchor] = []
        segments: list[SegmentMap] = []

        # Warp mode also needs offset/drift hints for the decoder.
        offset_windows: list[float] = []
        if mode in ("auto", "offset", "drift", "warp", "partial"):
            log.info("Step 5/12: Estimating global offset")
            offset, offset_conf, offset_windows = estimate_global_offset(
                vid_feat, ad_feat,
                y_vid=y_vid, y_ad=y_ad, sr=sr,
                compute=compute,
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
                    compute=compute,
                )
                progress.update(drift_task, completed=11)
            drift_ppm = drift
            anchors = drift_anchors

            if fp_edits and mode == "auto" and offset_conf >= config.confidence_threshold:
                log.warning(
                    "Fingerprint found %d offset spans (spread %.1f s) — a single "
                    "global offset would misalign part of the track; using warp",
                    len(fp.spans),
                    max(s.offset for s in fp.spans) - min(s.offset for s in fp.spans),
                )
            elif fp_edits and mode in ("offset", "drift"):
                log.warning(
                    "Fingerprint found %d distinct offset spans but --mode %s was "
                    "requested — part of the track will likely be misaligned",
                    len(fp.spans), mode,
                )

            if (mode in ("offset", "drift")
                    or (mode == "auto" and offset_conf >= config.confidence_threshold and not fp_edits)):
                # When drift is significant, use the drift intercept (offset at t=0)
                # rather than the global average offset, because stretching is
                # anchored at t=0 in the source.
                use_drift = (
                    len(drift_anchors) >= 3 and abs(drift_ppm) > 10
                    and drift_conf >= config.confidence_threshold
                )
                if use_drift:
                    base_offset = drift_intercept
                    stretch = 1.0 + drift_ppm / 1e6
                else:
                    base_offset = offset
                    stretch = 1.0

                # Clamp stretch to ±max_stretch
                stretch = max(1.0 - config.max_stretch, min(1.0 + config.max_stretch, stretch))

                dst_end = base_offset + ad_duration * stretch
                best_conf = drift_conf if use_drift else offset_conf
                if abs(drift_ppm) > 10 and not use_drift:
                    extra_warnings.append("Drift estimate was not reliable enough; using the global offset")

                segments = [SegmentMap(
                    src_start=0.0, src_end=ad_duration,
                    dst_start=base_offset,
                    dst_end=dst_end,
                    offset=base_offset, stretch=stretch,
                    confidence=best_conf,
                )]

                if use_drift:
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
            elif mode in ("warp", "partial"):
                log.info(
                    "Offset/drift collected as hints for warp (offset=%.3f s, drift=%.1f ppm)",
                    global_offset, drift_ppm,
                )
            else:
                log.info("Offset mode insufficient (conf=%.3f), trying piecewise", offset_conf)

        warp_fns = None
        warp_segment_ranges = None
        warp_path_result = None
        fp_anchor_windows = 0
        alignment_review_required = False
        partial_diagnostics = None
        timing_debug: dict = {"policy_revision": "measured-cuts-terminal-v1"}
        if fp is not None:
            from dataclasses import asdict
            timing_debug["fingerprint"] = {
                "raw_match_count": fp.n_matches,
                "raw_bucket_support": fp.raw_bucket_support,
                "raw_match_t_ad": fp.raw_match_t_ad.tolist() if fp.raw_match_t_ad is not None else [],
                "raw_match_t_vid": fp.raw_match_t_vid.tolist() if fp.raw_match_t_vid is not None else [],
                "short_terminal_hypotheses": [asdict(s) for s in fp.short_terminal_spans],
                "terminal_support": [asdict(s) for s in fp.terminal_support],
                "match_t_ad": fp.match_t_ad.tolist() if fp.match_t_ad is not None else [],
                "match_t_vid": fp.match_t_vid.tolist() if fp.match_t_vid is not None else [],
            }

        if mode in ("auto", "warp", "partial") and not segments:
            from adsync.align.candidate_lattice import build_candidate_lattice
            from adsync.align.warp_decode import decode_warp_path
            from adsync.align.warp_fit import fit_warp_function

            # Per-window offset hints: fingerprint spans give each window its
            # own expected offset, so the search radius only has to cover
            # local uncertainty, not the size of the edits.
            fp_hint_fn = None
            if fp is not None and fp.strong:
                fallback_offset = global_offset or 0.0

                def fp_hint_fn(t: float) -> float:
                    o = fp.offset_at(t)
                    return fallback_offset if o is None else o

            # Size the search radius so real edits stay inside it.  With
            # fingerprint hints the radius only covers span-boundary slop;
            # otherwise it must reach the largest plausible shift: the
            # video/AD duration gap and the measured offset scatter.
            if config.warp_search_radius is not None:
                search_radius = config.warp_search_radius
            elif fp_hint_fn is not None and mode != "partial":
                max_jump = max(
                    (abs(b.offset - a.offset) for a, b in zip(fp.spans, fp.spans[1:])),
                    default=0.0,
                )
                search_radius = max(30.0, min(90.0, max_jump + 15.0))
            else:
                duration_gap = abs(video_duration - ad_duration)
                offset_spread = (max(offset_windows) - min(offset_windows)) if offset_windows else 0.0
                search_radius = min(240.0, max(30.0, duration_gap + 45.0, offset_spread + 45.0))
            log.info(
                "Step 7/12: Building candidate lattice (search radius ±%.0f s, %s)",
                search_radius,
                f"{len(fp.spans)} fingerprint span hints" if fp_hint_fn is not None
                else f"global hint {global_offset or 0.0:+.2f} s",
            )
            with progress:
                lattice_task = progress.add_task("Building lattice", total=1)
                def _lattice_progress(cur: int, tot: int) -> None:
                    progress.update(lattice_task, total=tot, completed=cur)
                lattice = build_candidate_lattice(
                    vid_feat, ad_feat,
                    y_vid=y_vid, y_ad=y_ad, audio_sr=sr,
                    window_sec=2.0 if mode == "partial" else config.anchor_window_sec,
                    step_sec=1.0 if mode == "partial" else config.anchor_step_sec,
                    search_radius_sec=search_radius,
                    offset_hint=global_offset or 0.0,
                    offset_hint_fn=fp_hint_fn,
                    max_candidates=config.warp_max_candidates,
                    multiband=config.multiband,
                    include_boundary_matches=mode == "partial",
                    on_progress=_lattice_progress,
                    compute=compute,
                    threads=threads,
                )
                progress.update(lattice_task, completed=progress.tasks[lattice_task].total)

            # ── Fingerprint anchoring: when correlation starves (different
            # mixes of the same material) but landmark evidence is strong,
            # the matches themselves become window anchors.
            windows_with_candidates = sum(1 for w in lattice if w.candidates)
            coverage = windows_with_candidates / max(1, len(lattice))
            if mode != "partial" and config.fp_anchor and fp is not None and fp.strong:
                from adsync.align.candidate_lattice import augment_lattice_with_fingerprint

                log.info(
                    "Correlation coverage %.0f%% — checking local fingerprint fallback",
                    100.0 * coverage,
                )
                fp_anchor_windows = augment_lattice_with_fingerprint(
                    lattice, fp, only_weak=coverage >= config.fp_anchor_min_coverage,
                )
                if fp_anchor_windows:
                    windows_with_candidates = sum(1 for w in lattice if w.candidates)
                    extra_warnings.append(
                        f"Alignment anchored on fingerprint landmarks in {fp_anchor_windows} "
                        f"windows (correlation coverage was {coverage:.0%} — differing "
                        "soundtrack mixes)"
                    )

            if mode == "partial":
                if config.fp_anchor and fp is not None:
                    from adsync.align.partial_evidence import augment_partial_evidence
                    fp_anchor_windows = augment_partial_evidence(
                        lattice, fp, max_candidates=config.warp_max_candidates,
                    )
                from adsync.align.partial_fit import fit_partial_alignment

                log.info("Step 8/12: Decoding partial alignment and competing paths")
                warp_fns, warp_segment_ranges, warp_path_result, partial_diagnostics = fit_partial_alignment(
                    lattice, ad_duration, video_duration,
                    window_sec=2.0, step_sec=1.0, max_stretch=config.max_stretch,
                )
                timing_debug["partial_alignment"] = partial_diagnostics
                timing_debug["pre_vetting_path"] = [p.model_dump() for p in warp_path_result.points]
                anchors = [Anchor(source_time=p.source_time, target_time=p.target_time,
                                  score=p.confidence, window=2.0)
                           for p in warp_path_result.anchor_points]
            elif windows_with_candidates < 3:
                log.warning(
                    "Only %d windows with candidates — falling back to piecewise",
                    windows_with_candidates,
                )
                mode = "piecewise"
            else:
                log.info("Step 8/12: Decoding optimal warp path")
                warp_points, warp_cost = decode_warp_path(
                    lattice,
                    lambda_jump=config.warp_lambda_jump,
                    lambda_curve=config.warp_lambda_curve,
                    lambda_speech=config.warp_lambda_speech,
                    drift_hint_ppm=drift_ppm,
                    offset_hint=global_offset,
                    offset_hint_fn=fp_hint_fn,
                    terminal_support=fp.terminal_support if fp is not None else None,
                )
                timing_debug["pre_vetting_path"] = [p.model_dump() for p in warp_points]

                log.info("Step 8.5/12: Fitting continuous warp function")
                warp_fns, warp_segment_ranges, warp_path_result = fit_warp_function(
                    warp_points, ad_duration, video_duration,
                    anchor_fraction=config.warp_anchor_fraction,
                    discontinuity_threshold=config.warp_discontinuity_threshold,
                    terminal_support=fp.terminal_support if fp is not None else None,
                )
                warp_path_result.path_cost = warp_cost
                timing_debug["post_vetting_path"] = [p.model_dump() for p in warp_path_result.points]
                from adsync.align.refine import refine_warp_boundaries
                timing_debug["boundary_refinement"] = refine_warp_boundaries(
                    warp_fns, warp_segment_ranges, warp_path_result, y_vid, y_ad, sr,
                )
                timing_debug["refined_path"] = [p.model_dump() for p in warp_path_result.points]

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

        # Fingerprint residuals measure fit consistency. Landmark-derived
        # anchors reuse this evidence; rendered-output QC is a separate check.
        fp_res_p50: float | None = None
        fp_res_p95: float | None = None
        if (warp_fns is not None and fp is not None
                and fp.match_t_ad is not None and len(fp.match_t_ad)):
            from adsync.align.confidence import local_fingerprint_residuals
            res = local_fingerprint_residuals(
                warp_fns, warp_segment_ranges, fp.match_t_ad, fp.match_t_vid,
            )
            timing_debug["local_fit_residuals"] = res
            if res["p95_ms"] is not None:
                fp_res_p50, fp_res_p95 = res["p50_ms"], res["p95_ms"]
                log.info(
                    "Verification: warp vs %d fingerprint matches — local median %.0f ms, worst local p95 %.0f ms",
                    len(fp.match_t_ad), fp_res_p50, fp_res_p95,
                )
            if res["status"] != "pass":
                bad = [b for b in res["buckets"] if b["strong"] and b["p95_abs_sec"] > .15]
                for bucket in bad:
                    extra_warnings.append(
                        f"Local timing needs review at AD {bucket['start_sec']:.0f}–{bucket['end_sec']:.0f} s: "
                        f"coherent residual {bucket['lag_sec']:+.3f} s"
                    )

        # ── Confidence ───────────────────────────────────────────────────
        confidence, warnings = compute_confidence(
            anchors, segments, ad_duration, video_duration, mode="warp" if mode == "partial" else mode,
            warp_path=warp_path_result,
            fp_residual_p95_ms=fp_res_p95,
        )
        warnings = extra_warnings + warnings
        if partial_diagnostics is not None:
            if not warp_fns:
                confidence = 0.0
                alignment_review_required = True
                warnings.append("Partial alignment found no supported matched segments; output withheld.")
            else:
                source_coverage = sum(hi - lo for lo, hi in warp_segment_ranges) / max(ad_duration, 1e-9)
                confidence = min(confidence, warp_path_result.mean_confidence, source_coverage)
            if partial_diagnostics.get("ambiguous_ranges"):
                alignment_review_required = True
                warnings.append("Competing alignment paths remain plausible; review the partial alignment report.")
            for axis in ("source", "target"):
                gaps = partial_diagnostics.get(f"{axis}_gaps", [])
                if gaps:
                    seconds = sum(gap["end_sec"] - gap["start_sec"] for gap in gaps)
                    warnings.append(f"Partial alignment leaves {seconds:.2f} s of the {axis} timeline unmeasured.")
                    if any(gap["end_sec"] - gap["start_sec"] > 1 / sr for gap in gaps):
                        alignment_review_required = True
            if partial_diagnostics.get("status") != "matched":
                alignment_review_required = True
            if alignment_review_required:
                confidence = min(confidence, max(0.0, config.confidence_threshold - .01))
            partial_diagnostics["review_required"] = alignment_review_required
            partial_diagnostics["playback_offset_adjust_sec"] = config.offset_adjust

        # Keep confidence/landmark verification about the measured alignment.
        # Apply the user's final playback nudge consistently in every mode.
        adjust = config.offset_adjust
        if warp_fns is not None and adjust:
            for fn in warp_fns:
                fn.c[-1] += adjust  # Shift each polynomial's constant only.
            seen: set[int] = set()
            for point in warp_path_result.points + warp_path_result.anchor_points:
                if id(point) not in seen:
                    point.target_time += adjust
                    seen.add(id(point))
        if adjust and mode in ("warp", "piecewise", "partial"):
            for anchor in anchors:
                anchor.target_time += adjust

        # Clip source and destination together. Clamping dst_start alone loses
        # negative offsets and replays the removed leading AD at video time 0.
        clipped_segments: list[SegmentMap] = []
        for seg in segments:
            dst_start = seg.dst_start + adjust
            src_start = max(seg.src_start, seg.src_start - dst_start / seg.stretch)
            src_end = min(seg.src_end, seg.src_start + (video_duration - dst_start) / seg.stretch)
            if src_end <= src_start:
                continue
            clipped_segments.append(seg.model_copy(update={
                "src_start": src_start, "src_end": src_end,
                "dst_start": dst_start + (src_start - seg.src_start) * seg.stretch,
                "dst_end": dst_start + (src_end - seg.src_start) * seg.stretch,
                "offset": dst_start + (src_start - seg.src_start) * seg.stretch - src_start,
            }))
        segments = clipped_segments
        if warp_fns is not None:
            timing_debug["segment_ranges"] = [list(r) for r in warp_segment_ranges]
            timing_debug["fitted_pchip"] = [
                {"x": fn.x.tolist(), "c": fn.c.tolist()} for fn in warp_fns
            ]

        # Fingerprint bad spots: AD stretches whose landmarks match nowhere
        # in the video — different content, missing scenes, or a wrong pair.
        if fp is not None:
            for lo, hi in fp.unmatched:
                warnings.append(
                    f"No fingerprint match for AD {_fmt_mmss(lo)}–{_fmt_mmss(hi)} "
                    "— content may differ from the video there"
                )
            for lo, hi, off in fp.dropped_ranges:
                warnings.append(
                    f"AD {_fmt_mmss(lo)}–{_fmt_mmss(hi)} matched repeated content "
                    f"at an impossible position ({off:+.1f} s) — ignored"
                )

        # Drop the analysis-rate audio (and features, unless we'll plot them)
        # before loading the HQ AD — keeps peak RSS down on long films.
        del y_vid, y_ad
        if not debug_dir:
            del vid_feat, ad_feat

        # ── Step 9: Rebuild ──────────────────────────────────────────────
        synced_y = None
        hq_sr = None

        if (mux or debug_dir) and not identity_failed and (mode != "partial" or warp_fns):
            log.info("Step 9/12: Rebuilding synced AD track (HQ)")
            extract_audio(
                ad_info, ad_hq_wav, sr=ad_hq_sr, mono=False,
                speed_ratio=speed_stretch, sample_fmt="f32",
                threads=threads,
            )
            y_ad_hq, hq_sr = load_wav(ad_hq_wav, sr=ad_hq_sr, mono=False)
            hq_video_duration = video_duration

            if mode in ("warp", "partial") and warp_fns is not None:
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
        publication_error: OutputPublicationError | None = None
        if mux and output_path and synced_y is not None:
            log.info("Step 10–11/12: Encoding & muxing final MKV")
            try:
                mux_ad_track(
                    video_path, synced_y, hq_sr, output_path,
                    codec=config.output_codec,
                    bitrate=config.output_bitrate,
                    language=config.ad_language, title=config.ad_title,
                    n_existing_audio=len(video_info.audio_streams),
                    original_audio=original_audio,
                    threads=threads,
                )
                final_output = str(output_path)
            except OutputPublicationError as exc:
                publication_error = exc
                final_output = str(exc.local_path)
                warnings.append(str(exc))

        # ── Step 12: Report ──────────────────────────────────────────────
        log.info("Step 12/12: Generating report")
        from adsync.models import FingerprintSpan as FingerprintSpanModel

        report = SyncReport(
            mode=mode,
            confidence=confidence,
            alignment_review_required=alignment_review_required,
            compute_requested=compute.requested_backend,
            compute_backend=compute.active_backend,
            compute_device=compute.device_name,
            compute_fallback_reason=compute.fallback_reason,
            gpu_correlations=compute.gpu_calls,
            cpu_correlations=compute.cpu_calls,
            offset_adjust=config.offset_adjust,
            global_offset=global_offset,
            drift_ppm=drift_ppm,
            anchors=anchors,
            segments=segments,
            warnings=warnings,
            output_path=final_output,
            warp_path=warp_path_result,
            fingerprint_spans=[
                FingerprintSpanModel(
                    ad_start=s.ad_start, ad_end=s.ad_end,
                    offset=s.offset, matches=s.matches,
                )
                for s in (fp.spans if fp is not None else [])
            ],
            fingerprint_unmatched=list(fp.unmatched) if fp is not None else [],
            speed_stretch=speed_stretch,
            fp_anchor_windows=fp_anchor_windows,
            fp_residual_p50_ms=fp_res_p50,
            fp_residual_p95_ms=fp_res_p95,
            timing_debug=timing_debug,
            identity_check=identity_check,
        )

        if publication_error is not None:
            recovery_report = publication_error.local_path.with_suffix(".report.json")
            try:
                write_report(report, recovery_report)
                log.warning("Recovery report saved to %s", recovery_report)
            except OSError as exc:
                log.warning("Could not save recovery report: %s. Completed media remains at %s", exc, publication_error.local_path)
        if report_path:
            try:
                write_report(report, report_path)
            except OSError:
                if publication_error is None:
                    raise
                log.warning("Requested report path unavailable; use recovery report at %s", recovery_report)

        if publication_error is not None:
            raise publication_error

        if debug_dir:
            from adsync.report.plots import plot_anchors, plot_features
            plot_features(vid_feat, ad_feat, workdir.subdir("plots"))
            plot_anchors(anchors, segments, workdir.subdir("plots"))

        print_summary(report)
        return report

    finally:
        resources.close()



def _fmt_mmss(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


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
