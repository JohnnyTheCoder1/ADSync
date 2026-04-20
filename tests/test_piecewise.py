"""Tests for piecewise map building."""

from __future__ import annotations

import pytest

from adsync.align.piecewise_map import build_piecewise_map
from adsync.config import SyncConfig
from adsync.models import Anchor, SegmentMap


def test_empty_anchors() -> None:
    """No anchors should give a single identity segment."""
    config = SyncConfig()
    segments = build_piecewise_map([], ad_duration=100.0, video_duration=100.0, config=config)
    assert len(segments) == 1
    assert segments[0].confidence == 0.0


def test_single_anchor() -> None:
    """A single anchor should produce segments around it."""
    config = SyncConfig()
    anchors = [Anchor(source_time=50.0, target_time=52.0, score=0.9, window=8.0)]
    segments = build_piecewise_map(anchors, ad_duration=100.0, video_duration=100.0, config=config)
    assert len(segments) >= 1
    # All segments should be non-empty
    for seg in segments:
        assert seg.src_end > seg.src_start


def test_multiple_anchors_monotonic() -> None:
    """Multiple anchors should produce monotonic segments."""
    config = SyncConfig()
    anchors = [
        Anchor(source_time=10.0, target_time=12.0, score=0.95, window=8.0),
        Anchor(source_time=50.0, target_time=52.0, score=0.90, window=8.0),
        Anchor(source_time=90.0, target_time=92.0, score=0.85, window=8.0),
    ]
    segments = build_piecewise_map(anchors, ad_duration=100.0, video_duration=100.0, config=config)
    assert len(segments) >= 3

    # Segments should be ordered
    for i in range(1, len(segments)):
        assert segments[i].src_start >= segments[i - 1].src_start


def test_stretch_clamped() -> None:
    """Large offset jumps should be clamped by max_stretch."""
    config = SyncConfig(max_stretch=0.01)
    anchors = [
        Anchor(source_time=10.0, target_time=12.0, score=0.9, window=8.0),
        Anchor(source_time=50.0, target_time=60.0, score=0.9, window=8.0),  # big jump
    ]
    segments = build_piecewise_map(anchors, ad_duration=100.0, video_duration=120.0, config=config)
    for seg in segments:
        assert abs(seg.stretch - 1.0) <= config.max_stretch + 1e-6
