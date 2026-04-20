"""Debug visualization plots."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from adsync.models import Anchor, FeatureBundle, SegmentMap

log = logging.getLogger("adsync")


def plot_features(
    video_features: FeatureBundle,
    ad_features: FeatureBundle,
    output_dir: Path,
) -> list[Path]:
    """Generate feature comparison plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    hop = video_features.hop_length
    sr = video_features.sr
    spf = hop / sr

    # RMS comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    vid_times = np.arange(len(video_features.rms)) * spf
    ad_times = np.arange(len(ad_features.rms)) * spf
    ax1.plot(vid_times, video_features.rms, linewidth=0.5, label="Video")
    ax1.set_ylabel("RMS")
    ax1.legend()
    ax2.plot(ad_times, ad_features.rms, linewidth=0.5, color="orange", label="AD")
    ax2.set_ylabel("RMS")
    ax2.set_xlabel("Time (s)")
    ax2.legend()
    fig.suptitle("RMS Envelope Comparison")
    fig.tight_layout()
    p = output_dir / "rms_comparison.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    paths.append(p)

    # Onset comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    ax1.plot(vid_times, video_features.onset, linewidth=0.5, label="Video")
    ax1.set_ylabel("Onset")
    ax1.legend()
    ax2.plot(ad_times, ad_features.onset, linewidth=0.5, color="orange", label="AD")
    ax2.set_ylabel("Onset")
    ax2.set_xlabel("Time (s)")
    ax2.legend()
    fig.suptitle("Onset Strength Comparison")
    fig.tight_layout()
    p = output_dir / "onset_comparison.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    paths.append(p)

    log.info("Saved %d feature plots to %s", len(paths), output_dir)
    return paths


def plot_anchors(
    anchors: list[Anchor],
    segments: list[SegmentMap],
    output_dir: Path,
) -> list[Path]:
    """Plot anchor positions and segment map."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    if anchors:
        fig, ax = plt.subplots(figsize=(10, 8))
        src = [a.source_time for a in anchors]
        tgt = [a.target_time for a in anchors]
        scores = [a.score for a in anchors]
        scatter = ax.scatter(src, tgt, c=scores, cmap="RdYlGn", vmin=0, vmax=1, s=20)
        plt.colorbar(scatter, ax=ax, label="Score")
        ax.plot([0, max(src)], [0, max(tgt)], "k--", alpha=0.3, label="Identity")
        ax.set_xlabel("AD Time (s)")
        ax.set_ylabel("Video Time (s)")
        ax.set_title("Anchor Mapping")
        ax.legend()
        fig.tight_layout()
        p = output_dir / "anchors.png"
        fig.savefig(p, dpi=100)
        plt.close(fig)
        paths.append(p)

    if segments:
        fig, ax = plt.subplots(figsize=(12, 5))
        for i, seg in enumerate(segments):
            ax.barh(
                0, seg.src_end - seg.src_start,
                left=seg.src_start, height=0.3,
                color=f"C{i % 10}", alpha=0.7,
            )
            ax.barh(
                1, seg.dst_end - seg.dst_start,
                left=seg.dst_start, height=0.3,
                color=f"C{i % 10}", alpha=0.7,
            )
            # Draw connecting lines
            ax.plot(
                [seg.src_start, seg.dst_start],
                [0.15, 0.85],
                "k-", alpha=0.2, linewidth=0.5,
            )
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["AD (source)", "Video (dest)"])
        ax.set_xlabel("Time (s)")
        ax.set_title("Segment Mapping")
        fig.tight_layout()
        p = output_dir / "segments.png"
        fig.savefig(p, dpi=100)
        plt.close(fig)
        paths.append(p)

    log.info("Saved %d alignment plots to %s", len(paths), output_dir)
    return paths
