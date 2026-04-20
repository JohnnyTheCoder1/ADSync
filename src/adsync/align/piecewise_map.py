"""Build a piecewise time map from anchors."""

from __future__ import annotations

import logging

import numpy as np

from adsync.config import SyncConfig
from adsync.models import Anchor, SegmentMap

log = logging.getLogger("adsync")


def build_piecewise_map(
    anchors: list[Anchor],
    ad_duration: float,
    video_duration: float,
    config: SyncConfig,
) -> list[SegmentMap]:
    """Convert a list of anchors into a monotonic piecewise segment map.

    Each segment describes how a range of AD time maps to video time.
    """
    if not anchors:
        log.warning("No anchors — returning single-segment identity map")
        return [SegmentMap(
            src_start=0.0, src_end=ad_duration,
            dst_start=0.0, dst_end=ad_duration,
            offset=0.0, stretch=1.0, confidence=0.0,
        )]

    segments: list[SegmentMap] = []

    # Prepend a virtual anchor at t=0 based on first anchor's offset
    first = anchors[0]
    virtual_start = Anchor(
        source_time=0.0,
        target_time=first.target_time - first.source_time,
        score=first.score * 0.8,
        window=first.window,
    )

    # Append a virtual anchor at the end
    last = anchors[-1]
    virtual_end = Anchor(
        source_time=ad_duration,
        target_time=last.target_time + (ad_duration - last.source_time),
        score=last.score * 0.8,
        window=last.window,
    )

    all_anchors = [virtual_start] + list(anchors) + [virtual_end]

    for i in range(len(all_anchors) - 1):
        a = all_anchors[i]
        b = all_anchors[i + 1]

        src_start = a.source_time
        src_end = b.source_time
        dst_start = a.target_time
        dst_end = b.target_time

        src_len = src_end - src_start
        dst_len = dst_end - dst_start

        if src_len <= 0:
            continue

        stretch = dst_len / src_len if src_len > 0 else 1.0
        offset = dst_start - src_start

        # Clamp stretch
        max_s = 1.0 + config.max_stretch
        min_s = 1.0 - config.max_stretch
        stretch = max(min_s, min(max_s, stretch))

        seg_confidence = (a.score + b.score) / 2.0

        segments.append(SegmentMap(
            src_start=src_start,
            src_end=src_end,
            dst_start=dst_start,
            dst_end=dst_start + src_len * stretch,
            offset=offset,
            stretch=stretch,
            confidence=seg_confidence,
        ))

    log.info("Built %d segments from %d anchors", len(segments), len(anchors))
    return segments
