"""Numerical parity and recoverable CUDA failures at the compute boundary."""

from concurrent.futures import ThreadPoolExecutor
import importlib
import threading
import time

import numpy as np
import pytest


def _compute():
    return importlib.import_module("adsync.compute")


def _reference(region, template):
    # Independent direct dot products, without FFT or cumulative sums.
    return np.array([
        np.dot(region[i:i + len(template)], template)
        / (np.linalg.norm(template) * max(np.linalg.norm(region[i:i + len(template)]), 1e-10))
        for i in range(len(region) - len(template) + 1)
    ])


@pytest.mark.parametrize("sizes", [(97, 13), (53, 53), (128, 32)])
def test_cpu_correlation_matches_direct_normalized_dot_products(sizes):
    backend = _compute().CorrelationBackend("cpu")
    rng = np.random.default_rng(22)
    region = rng.normal(size=sizes[0])
    template = rng.normal(size=sizes[1])
    result = backend.normalized_correlation(region, template)
    np.testing.assert_allclose(result, _reference(region, template), atol=2e-14, rtol=2e-12)
    assert result.dtype == np.float64
    assert backend.cpu_calls == 1
    assert backend.gpu_calls == 0


def test_cpu_preserves_input_arrays_and_handles_silence_and_short_region():
    backend = _compute().CorrelationBackend("cpu")
    region = np.arange(16, dtype=np.float32)[::2]
    original = region.copy()
    result = backend.normalized_correlation(region, np.zeros(3))
    np.testing.assert_array_equal(result, np.zeros(6))
    np.testing.assert_array_equal(region, original)
    assert backend.normalized_correlation(np.ones(2), np.ones(3)).size == 0


def test_invalid_backend_is_rejected():
    with pytest.raises(ValueError, match="auto.*cpu.*cuda"):
        _compute().CorrelationBackend("gpu")


def test_cpu_does_not_load_optional_cuda(monkeypatch):
    compute = _compute()
    def unavailable():
        raise AssertionError("CPU must not import CuPy or touch the GPU")
    monkeypatch.setattr(compute, "_load_cuda", unavailable)
    backend = compute.CorrelationBackend("cpu")
    result = backend.normalized_correlation(np.array([1., 2., 3.]), np.array([1., 2.]))
    np.testing.assert_allclose(result, [1., 8. / np.sqrt(65.)])


def test_auto_logs_missing_cuda_and_computes_cpu_result(monkeypatch, caplog):
    compute = _compute()
    def unavailable():
        raise ImportError("CuPy is absent")
    monkeypatch.setattr(compute, "_load_cuda", unavailable)
    backend = compute.CorrelationBackend("auto")
    result = backend.normalized_correlation(np.array([1., 2., 3.]), np.array([1., 2.]))
    np.testing.assert_allclose(result, [1., 8. / np.sqrt(65.)])
    assert backend.active_backend == "cpu"
    assert "CuPy is absent" in backend.fallback_reason
    assert "CPU" in caplog.text


def test_forced_cuda_unavailable_has_actionable_error(monkeypatch):
    compute = _compute()
    def unavailable():
        raise ImportError("CuPy is absent")
    monkeypatch.setattr(compute, "_load_cuda", unavailable)
    with pytest.raises(compute.CudaBackendError, match="(?i)cuda.*cpu"):
        compute.CorrelationBackend("cuda")


class _FailingDevice:
    """Inject a device failure; real CPU recovery remains under test."""
    device_name = "failing CUDA device"

    def normalized_correlation(self, *args):
        raise MemoryError("CUDA out of memory")

    def fftconvolve(self, *args, **kwargs):
        raise MemoryError("CUDA out of memory")

    def close(self):
        pass


def test_auto_runtime_oom_disables_cuda_and_retries_current_work(monkeypatch, caplog):
    compute = _compute()
    monkeypatch.setattr(compute, "_load_cuda", _FailingDevice)
    backend = compute.CorrelationBackend("auto", min_cuda_samples=0)
    for _ in range(2):
        result = backend.normalized_correlation(np.array([1., 2., 3.]), np.array([1., 2.]))
        np.testing.assert_allclose(result, [1., 8. / np.sqrt(65.)])
    assert backend.active_backend == "cpu"
    assert backend.cpu_calls == 2
    assert backend.gpu_calls == 0
    assert "out of memory" in backend.fallback_reason
    assert sum("falling back" in r.message.lower() for r in caplog.records) == 1


def test_forced_cuda_runtime_oom_raises_instead_of_falling_back(monkeypatch):
    compute = _compute()
    monkeypatch.setattr(compute, "_load_cuda", _FailingDevice)
    backend = compute.CorrelationBackend("cuda")
    with pytest.raises(compute.CudaBackendError, match="out of memory"):
        backend.normalized_correlation(np.ones(4), np.ones(2))
    assert backend.cpu_calls == 0


def test_auto_keeps_small_work_on_cpu(monkeypatch):
    compute = _compute()
    monkeypatch.setattr(compute, "_load_cuda", _FailingDevice)
    backend = compute.CorrelationBackend("auto")
    result = backend.normalized_correlation(np.ones(4), np.ones(2))
    np.testing.assert_allclose(result, np.ones(3))
    assert backend.active_backend == "cuda"
    assert backend.fallback_reason is None
    assert backend.cpu_calls == 1


def test_close_race_cannot_silently_turn_forced_cuda_into_cpu(monkeypatch):
    compute = _compute()
    monkeypatch.setattr(compute, "_load_cuda", _FailingDevice)
    backend = compute.CorrelationBackend("cuda")
    routed = threading.Event()
    resume = threading.Event()

    def pause_after_routing(*args):
        routed.set()
        assert resume.wait(5.0)
        return True

    monkeypatch.setattr(backend, "prefers_cuda", pause_after_routing)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(backend.normalized_correlation, np.ones(8), np.ones(3))
        assert routed.wait(5.0)
        backend.close()
        resume.set()
        with pytest.raises(RuntimeError, match="closed"):
            pending.result(timeout=5.0)
    assert backend.cpu_calls == 0


def test_cuda_calls_are_serialized_and_separate_runs_keep_their_selection(monkeypatch):
    compute = _compute()
    busy = threading.Lock()
    device_threads = set()

    class ExclusiveDevice(_FailingDevice):
        def normalized_correlation(self, region, template, template_energy):
            if not busy.acquire(blocking=False):
                raise RuntimeError("overlapping device allocation")
            try:
                device_threads.add(threading.get_ident())
                time.sleep(0.005)
                return _reference(region, template)
            finally:
                busy.release()

    monkeypatch.setattr(compute, "_load_cuda", ExclusiveDevice)
    gpu = compute.CorrelationBackend("cuda")
    cpu = compute.CorrelationBackend("cpu")
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(lambda _: gpu.normalized_correlation(np.ones(8), np.ones(3)), range(8)))
    for output in outputs:
        np.testing.assert_allclose(output, np.ones(6))
    assert gpu.gpu_calls == 8
    # cuFFT plan caches are per thread; many caller threads must share one.
    assert len(device_threads) == 1
    assert cpu.active_backend == "cpu"
    assert cpu.gpu_calls == 0


@pytest.mark.parametrize("sizes", [(97, 13), (53, 53), (300001, 32000)])
def test_real_cuda_matches_cpu_float64_and_peak(sizes):
    compute = _compute()
    try:
        gpu = compute.CorrelationBackend("cuda")
    except compute.CudaBackendError as exc:
        pytest.skip(str(exc))
    rng = np.random.default_rng(103)
    region = rng.normal(size=sizes[0])
    start = (sizes[0] - sizes[1]) // 3
    template = region[start:start + sizes[1]].copy()
    cpu = compute.CorrelationBackend("cpu")
    try:
        actual = gpu.normalized_correlation(region, template)
        expected = cpu.normalized_correlation(region, template)
        np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-9)
        assert np.argmax(actual) == start
        assert actual.dtype == np.float64
        assert gpu.gpu_calls == 1
    finally:
        gpu.close()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_global_and_local_offsets_use_run_backend_and_preserve_delay(device):
    from adsync.align.global_offset import _raw_audio_offset, _norm_cross_correlation
    from adsync.align.drift import _local_offset

    compute = _compute()
    try:
        backend = compute.CorrelationBackend(device)
    except compute.CudaBackendError as exc:
        pytest.skip(str(exc))
    rng = np.random.default_rng(73)
    video = rng.normal(size=16000)
    ad = np.concatenate((np.zeros(125), video[:-125]))
    try:
        offset, confidence, windows = _raw_audio_offset(video, ad, 100, 5., compute=backend)
        assert offset == pytest.approx(-1.25, abs=0.0001)
        assert confidence > 0.99
        assert len(windows) == 5
        anchor = _local_offset(video, ad, 80., .01, 30., 5., compute=backend)
        assert anchor.target_time - anchor.source_time == pytest.approx(1.25, abs=0.0001)
        assert anchor.score > .99
        corr = _norm_cross_correlation(video, ad, 200, compute=backend)
        assert int(np.argmax(corr)) - 200 == -125
        if device == "cuda":
            assert backend.gpu_calls == 7
        else:
            assert backend.cpu_calls == 7
    finally:
        backend.close()


def test_lattice_cuda_preserves_multiband_candidates_and_window_order():
    from adsync.align.candidate_lattice import build_candidate_lattice
    from adsync.models import FeatureBundle

    compute = _compute()
    try:
        gpu = compute.CorrelationBackend("cuda")
    except compute.CudaBackendError as exc:
        pytest.skip(str(exc))
    sr = 4000
    rng = np.random.default_rng(90)
    video = rng.normal(size=sr * 40).astype(np.float32)
    ad = np.concatenate((np.zeros(sr, np.float32), video[:-sr]))
    features = FeatureBundle(
        sr=sr, hop_length=100, duration=40., rms=np.ones(1600),
        onset=np.ones(1600), mel=np.zeros((40, 1600)), mfcc=np.zeros((13, 1600)),
    )
    options = dict(y_vid=video, y_ad=ad, audio_sr=sr, search_radius_sec=5., step_sec=4.)
    cpu = compute.CorrelationBackend("cpu")
    expected = build_candidate_lattice(features, features, compute=cpu, **options)
    progress = []
    try:
        actual = build_candidate_lattice(
            features, features, compute=gpu,
            on_progress=lambda done, total: progress.append((done, total)), **options,
        )
        assert len(actual) == len(expected)
        assert [w.source_center for w in actual] == sorted(w.source_center for w in actual)
        assert progress[-1] == (len(actual), len(actual))
        for a, b in zip(actual, expected):
            assert len(a.candidates) == len(b.candidates)
            np.testing.assert_allclose(
                [(c.offset_sec, c.score, c.peak_sharpness, c.peak_ratio) for c in a.candidates],
                [(c.offset_sec, c.score, c.peak_sharpness, c.peak_ratio) for c in b.candidates],
                rtol=1e-8, atol=1e-9,
            )
        assert actual[3].candidates[0].offset_sec == pytest.approx(-1., abs=.001)
        assert gpu.gpu_calls >= len(actual)
        assert cpu.cpu_calls >= len(expected)
    finally:
        gpu.close()
