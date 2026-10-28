"""Season runs stop promptly, explain held episodes, and keep their cache bounded."""

import _thread
import hashlib
import threading
import time

import numpy as np
import pytest

from adsync.batch.runner import EpisodeResult
from adsync.cache import ArtifactCache
from test_season_runner import season

REVIEW_REASON = "Local quality checks require review; output has not been published."


def test_review_outcome_names_the_episode_its_reason_and_the_unpublished_render():
    from adsync.batch.cli import _print_outcomes

    result = EpisodeResult(label="S01E04", status="skipped_needs_review", video_path="v", ad_path="a",
                           output_path="o", report_path="r", log_path="l", error=REVIEW_REASON,
                           retained_local_path=r"C:\work\S01E04.AD.mkv")
    lines = []
    _print_outcomes([result], lines.append)
    assert lines == [f"S01E04: needs review ({REVIEW_REASON})",
                     r"  Unpublished render for review: C:\work\S01E04.AD.mkv"]


def test_resumed_review_keeps_its_reason(season):
    season.behaviors["S01E02"] = "needs_review"
    season.run()
    resumed = {result.label: result for result in season.run().results}["S01E02"]
    assert resumed.status == "skipped_needs_review"
    assert resumed.error == REVIEW_REASON


def test_ctrl_c_during_the_content_check_stops_queued_decodes(monkeypatch, tmp_path):
    from adsync.batch import content_pairing
    from adsync.batch.discovery import EpisodeKey, EpisodePair
    from adsync.config import SyncConfig

    calls = []

    def slow_read(path, config, *, video, threads=1):
        calls.append(path)
        time.sleep(0.5)
        raise OSError("not media")

    monkeypatch.setattr(content_pairing, "load_recording_features", slow_read)
    monkeypatch.setattr(content_pairing, "_parallel_reads", lambda files: 2)
    pairs = [EpisodePair(EpisodeKey(1, n), tmp_path / f"v{n}.mkv", tmp_path / f"a{n}.mp3", tmp_path / f"o{n}.mkv")
             for n in range(1, 21)]
    threading.Timer(0.3, _thread.interrupt_main).start()
    started = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        content_pairing.check_season_content(pairs, SyncConfig(device="cpu"), echo=lambda line: None)
    assert time.monotonic() - started < 2.0
    assert len(calls) <= 4


def _key(n):
    return hashlib.sha256(str(n).encode()).hexdigest()


def test_cache_evicts_the_least_recently_used_entries_beyond_its_cap(tmp_path):
    entry = {"audio": np.zeros(4096, dtype=np.float32)}
    probe = ArtifactCache(tmp_path / "probe")
    probe.put(_key(0), entry)
    size = next((tmp_path / "probe").rglob("*.npz")).stat().st_size

    cache = ArtifactCache(tmp_path / "cache", max_bytes=int(2.5 * size))
    cache.put(_key(1), entry)
    time.sleep(0.05)
    cache.put(_key(2), entry)
    time.sleep(0.05)
    assert cache.get(_key(1)) is not None  # entry 1 is now the most recently used
    time.sleep(0.05)
    cache.put(_key(3), entry)
    assert cache.get(_key(2)) is None
    assert cache.get(_key(1)) is not None and cache.get(_key(3)) is not None
