"""Per-run CPU/CUDA correlation with float64 normalization and safe fallback.

CuPy is optional and imported only when auto/CUDA is requested. Arrays returned
to alignment code always live on the CPU; peak selection and interpolation keep
their existing NumPy/SciPy implementations.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.signal import fftconvolve as scipy_fftconvolve

log = logging.getLogger("adsync")


class CudaBackendError(RuntimeError):
    """CUDA was explicitly requested but cannot execute the correlation."""


def _cpu_normalized(
    region: NDArray, template: NDArray, template_energy: float,
) -> NDArray:
    raw = scipy_fftconvolve(region, template[::-1], mode="valid")
    npos = len(raw)
    cumulative = np.empty(len(region) + 1, dtype=np.float64)
    cumulative[0] = 0.0
    np.cumsum(region ** 2, out=cumulative[1:])
    norms = np.sqrt(np.maximum(
        cumulative[len(template):len(template) + npos] - cumulative[:npos], 1e-20,
    ))
    return raw / (template_energy * norms)


class _CudaEngine:
    def __init__(self) -> None:
        import cupy as cp
        from cupyx.scipy.signal import fftconvolve

        self.cp = cp
        self._fftconvolve = fftconvolve
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("No CUDA device is available")
        self._device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(self._device.id)
        name = props["name"]
        self.device_name = name.decode() if isinstance(name, bytes) else str(name)
        # Keep allocations owned by this run; do not change CuPy's global pool.
        self._pool = cp.cuda.MemoryPool()
        self._stream = cp.cuda.Stream(non_blocking=True)
        try:
            # Probe the actual FFT, normalization kernels, and host transfer.
            # Device enumeration alone misses absent CUDA DLLs / NVRTC errors.
            result = self.normalized_correlation(
                np.array([1., 2., 3.]), np.array([1., 2.]), np.sqrt(5.),
            )
            if not np.allclose(result, [1., 8. / np.sqrt(65.)], atol=1e-12):
                raise RuntimeError("CUDA correlation self-check returned incorrect values")
        except Exception:
            self.close()
            raise

    def normalized_correlation(
        self, region: NDArray, template: NDArray, template_energy: float,
    ) -> NDArray:
        cp = self.cp
        with self._device, self._stream, cp.cuda.using_allocator(self._pool.malloc):
            r = cp.asarray(region, dtype=cp.float64)
            t = cp.asarray(template, dtype=cp.float64)
            raw = self._fftconvolve(r, t[::-1], mode="valid")
            npos = raw.size
            cumulative = cp.empty(r.size + 1, dtype=cp.float64)
            cumulative[0] = 0.0
            cp.cumsum(r ** 2, out=cumulative[1:])
            norms = cp.sqrt(cp.maximum(
                cumulative[t.size:t.size + npos] - cumulative[:npos], 1e-20,
            ))
            # Blocking copy completes all kernels before releasing the lock or
            # returning success, including asynchronous device error reporting.
            return cp.asnumpy(raw / (template_energy * norms), blocking=True)

    def fftconvolve(self, a: NDArray, b: NDArray, mode: str) -> NDArray:
        cp = self.cp
        with self._device, self._stream, cp.cuda.using_allocator(self._pool.malloc):
            result = self._fftconvolve(
                cp.asarray(a, dtype=cp.float64), cp.asarray(b, dtype=cp.float64), mode=mode,
            )
            return cp.asnumpy(result, blocking=True)

    def memory_info(self) -> tuple[int, int]:
        with self._device:
            free, total = self.cp.cuda.runtime.memGetInfo()
            return int(free), int(total)

    def close(self) -> None:
        # A damaged CUDA context can also reject cleanup. Preserve the original
        # execution failure and allow the CPU retry to continue in that case.
        try:
            with self._device:
                self._stream.synchronize()
                # This engine owns its executor thread, including the local
                # cuFFT cache. Release plans before returning pool allocations.
                self.cp.fft.config.get_plan_cache().clear()
                self._pool.free_all_blocks()
        except Exception:
            log.debug("CUDA allocation cleanup failed", exc_info=True)


def _load_cuda() -> _CudaEngine:
    return _CudaEngine()


class CorrelationBackend:
    """One run's correlation device, counters, and runtime fallback state.

    ``auto`` leaves small FFTs on CPU and permanently switches this run to CPU
    after a CUDA failure. ``cuda`` sends all nonempty correlations to the GPU
    and raises :class:`CudaBackendError` on failure. A single device worker
    bounds concurrent FFT allocations and reuses one thread's cuFFT plan cache,
    while caller threads remain free to parallelize CPU work.
    """

    def __init__(self, backend: str = "auto", *, min_cuda_samples: int = 262_144) -> None:
        if backend not in {"auto", "cpu", "cuda"}:
            raise ValueError("Correlation backend must be auto, cpu, or cuda")
        if min_cuda_samples < 0:
            raise ValueError("min_cuda_samples must be nonnegative")
        self.requested_backend = backend
        self.active_backend = "cpu"
        self.fallback_reason: str | None = None
        self.device_name: str | None = None
        self.gpu_calls = 0
        self.cpu_calls = 0
        self._min_cuda_samples = min_cuda_samples
        self._lock = Lock()
        self._cuda: _CudaEngine | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._closed = False
        if backend != "cpu":
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="adsync-cuda")
            try:
                self._cuda = self._executor.submit(_load_cuda).result()
            except Exception as exc:
                self._executor.shutdown(wait=True)
                self._executor = None
                self._failure(exc)
            else:
                self.active_backend = "cuda"
                self.device_name = self._cuda.device_name
                log.info("Correlation backend: CUDA (%s), float64", self.device_name)
        else:
            log.debug("Correlation backend: CPU")

    def _failure(self, exc: Exception) -> None:
        reason = f"{type(exc).__name__}: {exc}"
        if self.requested_backend == "cuda":
            raise CudaBackendError(
                f"CUDA correlation failed: {reason}. Install adsync[cuda] with a "
                "compatible NVIDIA driver, or use --device cpu (or auto for fallback)."
            ) from exc
        self.fallback_reason = reason
        self.active_backend = "cpu"
        log.warning("CUDA unavailable; falling back to CPU for this run: %s", reason)
        if self._cuda is not None:
            self._executor.submit(self._cuda.close).result()
            self._cuda = None
            self._executor.shutdown(wait=True)
            self._executor = None

    def prefers_cuda(self, region_samples: int, template_samples: int) -> bool:
        """Whether these dimensions merit GPU execution for this run."""
        return self.active_backend == "cuda" and (
            self.requested_backend == "cuda"
            or region_samples + template_samples >= self._min_cuda_samples
        )

    def _execute(
        self, samples: tuple[int, int], cpu_fn: Callable[[], NDArray],
        gpu_fn: Callable[[_CudaEngine], NDArray],
    ) -> NDArray:
        if self._closed:
            raise RuntimeError("Correlation backend is closed")
        if self.prefers_cuda(*samples):
            with self._lock:
                if self._closed:
                    raise RuntimeError("Correlation backend is closed")
                # Another lattice worker may already have disabled CUDA.
                if self._cuda is not None:
                    try:
                        result = self._executor.submit(gpu_fn, self._cuda).result()
                        if not np.isfinite(result).all():
                            raise RuntimeError("CUDA correlation returned nonfinite values")
                    except Exception as exc:
                        self._failure(exc)
                    else:
                        self.gpu_calls += 1
                        return result
        result = cpu_fn()
        with self._lock:
            self.cpu_calls += 1
        return result

    def normalized_correlation(
        self, region: NDArray, template: NDArray, *, template_energy: float | None = None,
    ) -> NDArray:
        """Valid-mode sliding dot products divided by per-position energy.

        Inputs are already centered by the alignment caller, matching its
        existing preprocessing. Both paths compute in float64; neither mutates
        host inputs. A short/empty region yields no positions, silence zeros.
        """
        region = np.asarray(region, dtype=np.float64)
        template = np.asarray(template, dtype=np.float64)
        if region.ndim != 1 or template.ndim != 1:
            raise ValueError("Correlation inputs must be one-dimensional")
        if template.size == 0 or region.size < template.size:
            return np.empty(0, dtype=np.float64)
        if template_energy is None:
            template_energy = float(np.sqrt(np.sum(template ** 2)))
        if template_energy < 1e-10:
            return np.zeros(region.size - template.size + 1, dtype=np.float64)
        return self._execute(
            (region.size, template.size),
            lambda: _cpu_normalized(region, template, template_energy),
            lambda engine: engine.normalized_correlation(region, template, template_energy),
        )

    def fftconvolve(self, a: NDArray, b: NDArray, mode: str = "full") -> NDArray:
        """Float64 FFT convolution for the onset-only global-offset fallback."""
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        return self._execute(
            (a.size, b.size),
            lambda: scipy_fftconvolve(a, b, mode=mode),
            lambda engine: engine.fftconvolve(a, b, mode),
        )

    def close(self) -> None:
        """Release this run's cached CUDA allocations after its work ends."""
        with self._lock:
            if self._cuda is not None:
                self._executor.submit(self._cuda.close).result()
                self._cuda = None
                self._executor.shutdown(wait=True)
                self._executor = None
            self._closed = True

    def memory_info(self) -> tuple[int, int]:
        """Return free/total bytes from the same device that passed the probe."""
        with self._lock:
            if self._closed or self._cuda is None:
                raise RuntimeError("CUDA backend is not active")
            return self._executor.submit(self._cuda.memory_info).result()

    def __enter__(self) -> CorrelationBackend:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
