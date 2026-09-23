"""Per-worker thread budgets reach numerical work and child FFmpeg processes."""

from concurrent.futures import ThreadPoolExecutor
import importlib
import os
from types import SimpleNamespace

import numpy as np
import pytest

from test_pipeline_regressions import pipeline_case


def test_thread_budget_limits_native_libraries_and_restores_state(monkeypatch):
    from threadpoolctl import threadpool_info
    hardware = importlib.import_module("adsync.hardware")
    monkeypatch.setenv("ADSYNC_THREADS", "7")
    before_env = dict(os.environ)
    before = {entry["filepath"]: entry["num_threads"] for entry in threadpool_info()}
    with hardware.thread_budget(4):
        assert hardware.effective_thread_count() == 4
        assert all(entry["num_threads"] <= 2 for entry in threadpool_info())
    assert hardware.effective_thread_count() == 7
    assert os.environ == before_env
    after = {entry["filepath"]: entry["num_threads"] for entry in threadpool_info()}
    assert after == before


def test_invalid_thread_environment_and_config_are_rejected(monkeypatch):
    from adsync.config import SyncConfig
    hardware = importlib.import_module("adsync.hardware")
    with pytest.raises(ValueError):
        SyncConfig(threads=0)
    monkeypatch.setenv("ADSYNC_THREADS", "zero")
    with pytest.raises(ValueError, match="ADSYNC_THREADS"):
        hardware.effective_thread_count()


def test_explicit_thread_count_overrides_environment(monkeypatch):
    hardware = importlib.import_module("adsync.hardware")
    monkeypatch.setenv("ADSYNC_THREADS", "2")
    assert hardware.effective_thread_count(3) == 3


def test_worker_initializer_propagates_run_budget_to_python_threads(monkeypatch):
    hardware = importlib.import_module("adsync.hardware")
    monkeypatch.setenv("ADSYNC_THREADS", "8")
    with hardware.thread_budget(2):
        with ThreadPoolExecutor(max_workers=1, initializer=hardware.worker_initializer(2)) as pool:
            assert pool.submit(hardware.effective_thread_count).result() == 2
    assert hardware.effective_thread_count() == 8


@pytest.mark.parametrize("entry", ["run_ffmpeg", "run_ffmpeg_piped", "run_ffmpeg_streamed"])
def test_ffmpeg_limits_apply_to_each_decoder_encoder_and_filters(monkeypatch, entry):
    from adsync.utils import subprocesses
    captured = []

    def run(command, **kwargs):
        captured.append(command)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(subprocesses, "_find_binary", lambda name: name)
    monkeypatch.setattr(subprocesses.subprocess, "run", run)
    monkeypatch.setattr(subprocesses.subprocess, "Popen", run)
    args = ["-i", "video.mkv", "-f", "f32le", "-i", "pipe:0", "-c:a", "libopus", "output.mkv"]
    original = args.copy()
    call = getattr(subprocesses, entry)
    if entry == "run_ffmpeg_piped":
        call(args, b"", threads=2)
    else:
        call(args, threads=2)
    command = captured[0]
    for index, arg in enumerate(command):
        if arg == "-i":
            assert command[index-2:index] == ["-threads", "2"]
    assert command[-3:] == ["-threads", "2", "output.mkv"]
    assert command[command.index("-filter_threads") + 1] == "2"
    assert command[command.index("-filter_complex_threads") + 1] == "2"
    assert args == original


def test_ffmpeg_environment_thread_budget_is_used_without_explicit_argument(monkeypatch):
    from adsync.utils import subprocesses
    captured = []
    monkeypatch.setenv("ADSYNC_THREADS", "3")
    monkeypatch.setattr(subprocesses, "_find_binary", lambda name: name)
    monkeypatch.setattr(subprocesses.subprocess, "run", lambda cmd, **kw: (captured.append(cmd) or SimpleNamespace(returncode=0, stderr="")))
    subprocesses.run_ffmpeg(["-i", "a.wav", "b.wav"])
    assert captured[0][-3:] == ["-threads", "3", "b.wav"]


def test_lattice_obeys_explicit_worker_limit_without_changing_matches(monkeypatch):
    from adsync.align import candidate_lattice
    from adsync.models import FeatureBundle
    used = []

    class CountedPool(ThreadPoolExecutor):
        def __init__(self, max_workers=None, **kwargs):
            used.append(max_workers)
            super().__init__(max_workers=max_workers, **kwargs)

    monkeypatch.setattr(candidate_lattice, "ThreadPoolExecutor", CountedPool)
    onset = np.random.default_rng(3).normal(size=4000)
    features = FeatureBundle(sr=100, hop_length=1, duration=40., rms=np.ones(4000),
                             onset=onset, mel=np.zeros((40, 4000)), mfcc=np.zeros((13, 4000)))
    lattice = candidate_lattice.build_candidate_lattice(
        features, features, threads=2, window_sec=2., step_sec=1., search_radius_sec=1.,
    )
    assert used == [2]
    assert len(lattice) == 38
    # The first window's true match is at a search boundary; the existing
    # peak picker excludes boundaries when interior peaks are present.
    assert all(w.candidates and abs(w.candidates[0].offset_sec) < .001 for w in lattice[1:])


def test_fingerprint_pool_honors_one_thread_worker_budget(monkeypatch):
    from adsync.align import fingerprint
    used = []

    class CountedPool(ThreadPoolExecutor):
        def __init__(self, max_workers=None, **kwargs):
            used.append(max_workers)
            super().__init__(max_workers=max_workers, **kwargs)

    monkeypatch.setenv("ADSYNC_THREADS", "1")
    monkeypatch.setattr(fingerprint, "ThreadPoolExecutor", CountedPool)
    audio = np.random.default_rng(7).normal(size=32000).astype(np.float32)
    result = fingerprint.fingerprint_align(audio, audio, 16000)
    assert used == [1]
    assert result.n_matches > 0


def test_pipeline_splits_ffmpeg_budget_and_honors_one_worker_environment(pipeline_case, monkeypatch):
    import concurrent.futures
    used = []

    class CountedPool(ThreadPoolExecutor):
        def __init__(self, max_workers=None, **kwargs):
            used.append(max_workers)
            super().__init__(max_workers=max_workers, **kwargs)

    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", CountedPool)
    monkeypatch.setenv("ADSYNC_THREADS", "1")
    pipeline_case.run(mux=True)
    assert used == [1, 1]
    assert all(options.get("threads") == 1 for _, options in pipeline_case.extractions)


def test_explicit_pipeline_budget_reaches_ffmpeg_without_mutating_environment(pipeline_case, monkeypatch):
    from pathlib import Path
    from adsync._pipeline import run_pipeline
    from adsync.config import SyncConfig

    monkeypatch.setenv("ADSYNC_THREADS", "8")
    run_pipeline(
        video_path=Path("video.mkv"), ad_path=Path("ad.wav"), output_path=Path("out.mkv"),
        config=SyncConfig(device="cpu", threads=2, mode="offset", fingerprint=False, speed_detect=False),
    )
    budgets = [options.get("threads") for _, options in pipeline_case.extractions]
    assert budgets == [1, 1, 2]
    assert os.environ["ADSYNC_THREADS"] == "8"
