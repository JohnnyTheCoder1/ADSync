"""Portable hardware detection and conservative per-season resource budgets."""

from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import contextmanager
from contextvars import ContextVar
import math
import os
from typing import Sequence

import psutil

_GIB = 1024 ** 3
_THREAD_BUDGET: ContextVar[int | None] = ContextVar("adsync_thread_budget", default=None)


@dataclass(frozen=True)
class HardwareProfile:
    logical_cpus: int
    physical_cpus: int
    usable_cpus: int
    total_memory_bytes: int
    available_memory_bytes: int
    cuda_available: bool = False
    gpu_name: str | None = None
    gpu_free_bytes: int | None = None
    gpu_total_bytes: int | None = None
    cuda_error: str | None = None


@dataclass(frozen=True)
class TaskResources:
    """Media metadata for one pair, without decoding either complete track."""

    video_duration: float
    ad_duration: float
    channels: int = 2
    sample_rate: int = 48000
    analysis_sr: int = 16000


@dataclass(frozen=True)
class ResourcePlan:
    jobs: int
    threads_per_job: int
    memory_per_job_bytes: int
    gpu_memory_per_job_bytes: int
    ram_budget_bytes: int
    device: str
    reasons: list[str] = field(default_factory=list)


def _usable_cpus() -> int:
    logical = psutil.cpu_count(logical=True) or os.cpu_count() or 1
    try:
        affinity = psutil.Process().cpu_affinity()
    except (AttributeError, NotImplementedError, psutil.Error, OSError):
        affinity = []
    return max(1, min(logical, len(affinity))) if affinity else max(1, logical)


def effective_thread_count(threads: int | None = None) -> int:
    """Resolve explicit, run-local, child-environment, then affinity budgets."""
    if threads is None:
        threads = _THREAD_BUDGET.get()
    if threads is None:
        configured = os.environ.get("ADSYNC_THREADS")
        if configured is not None:
            try:
                threads = int(configured)
            except ValueError as exc:
                raise ValueError("ADSYNC_THREADS must be a positive integer") from exc
    if threads is None:
        return _usable_cpus()
    if threads < 1:
        raise ValueError("Thread budget / ADSYNC_THREADS must be a positive integer")
    return threads


def worker_initializer(threads: int | None = None, *, native_threads: int | None = None):
    """Propagate run budgets and thread-local OpenMP limits into a pool.

    The enclosing thread_budget restores process-wide native settings after
    workers join. Standalone callers without that scope only propagate the
    Python worker budget, avoiding permanent changes to host library settings.
    """
    from threadpoolctl import ThreadpoolController

    count = effective_thread_count(threads)
    controller = ThreadpoolController() if _THREAD_BUDGET.get() is not None else None
    native = native_threads or max(1, count // 2)

    def initialize():
        _THREAD_BUDGET.set(count)
        if controller is not None:
            controller.limit(limits=native)

    return initialize


@contextmanager
def thread_budget(threads: int | None = None):
    """Scope native limits and main-thread nested pools to one pipeline run.

    Native BLAS limits are process-wide, as imposed by the libraries. Season
    workers therefore run in separate processes and set library environments
    before imports. Never run differently-budgeted pipelines in one process.
    """
    from threadpoolctl import threadpool_limits

    count = effective_thread_count(threads)
    token = _THREAD_BUDGET.set(count)
    try:
        # Two feature/fingerprint tasks can execute native code concurrently.
        with threadpool_limits(limits=max(1, count // 2)):
            yield count
    finally:
        _THREAD_BUDGET.reset(token)


def detect_hardware(device: str = "auto") -> HardwareProfile:
    """Inspect usable CPUs/RAM and self-test the requested optional GPU.

    CPU mode never loads CuPy. A CUDA device is reported usable only after
    CorrelationBackend completes its actual FFT/normalization/transfer probe.
    """
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("Device must be auto, cpu, or cuda")
    logical = psutil.cpu_count(logical=True) or os.cpu_count() or 1
    physical = psutil.cpu_count(logical=False) or logical
    memory = psutil.virtual_memory()
    gpu = dict(cuda_available=False)
    if device != "cpu":
        from adsync.compute import CorrelationBackend

        with CorrelationBackend(device) as backend:
            gpu = dict(
                cuda_available=backend.active_backend == "cuda",
                gpu_name=backend.device_name,
                cuda_error=backend.fallback_reason,
            )
            if gpu["cuda_available"]:
                try:
                    free, total = backend.memory_info()
                except Exception as exc:
                    gpu["cuda_error"] = f"VRAM query unavailable: {exc}"
                else:
                    gpu.update(gpu_free_bytes=free, gpu_total_bytes=total)
    return HardwareProfile(
        logical_cpus=max(1, logical), physical_cpus=max(1, physical),
        usable_cpus=_usable_cpus(), total_memory_bytes=int(memory.total),
        available_memory_bytes=int(memory.available), **gpu,
    )


def _fft_sizes(task: TaskResources) -> tuple[int, int]:
    """Upper bounds for padded serial/global and per-worker lattice FFTs."""
    global_samples = (
        min(task.ad_duration, 270.) + min(task.video_duration, 30.)
    ) * task.analysis_sr
    # Match the aligner's integer downsampling factor, including nonstandard
    # rates (e.g. 7,999 Hz remains above 4 kHz because its factor is one).
    correlation_sr = task.analysis_sr / max(1, task.analysis_sr // 4000)
    lattice_samples = (
        min(task.video_duration, 488.) + min(task.ad_duration, 8.)
    ) * correlation_sr
    return tuple(
        1 << max(0, math.ceil(math.log2(max(1., samples))))
        for samples in (global_samples, lattice_samples)
    )


def _memory_estimate(task: TaskResources) -> tuple[int, int]:
    if any(not math.isfinite(d) or d <= 0 for d in (task.video_duration, task.ad_duration)):
        raise ValueError("Positive, known media durations are required for memory planning")
    if min(task.channels, task.sample_rate, task.analysis_sr) < 1:
        raise ValueError("Channels and sample rates must be positive")

    # Concurrent STFTs, both analysis waveforms, mel intermediates, landmark
    # hashes/sort indices, and band copies. HQ loading starts after analysis
    # arrays are released, so these phase estimates need not be added together.
    analysis = (task.video_duration + task.ad_duration) * task.analysis_sr * 32
    # Source + full video output + one full retimed segment can coexist in
    # offset/drift mode. This is the maximum, even though warp uses small blocks.
    render = (
        task.ad_duration + task.video_duration + max(task.ad_duration, task.video_duration)
    ) * task.channels * task.sample_rate * 4
    ram = math.ceil((_GIB + max(analysis, render)) * 1.25)

    # Global offset uses a 30 s template and at most a 270 s search region;
    # lattice search can span 488 s at its ~4 kHz rate. Account for padded
    # double-precision FFT arrays and cached workspaces, plus a CUDA context.
    fft_samples = max(_fft_sizes(task))
    vram = 256 * 1024 ** 2 + 2 * 96 * fft_samples
    return ram, vram


def _cpu_fft_workspace(task: TaskResources, threads: int) -> int:
    global_fft, lattice_fft = _fft_sizes(task)
    # Each lattice worker may allocate independently after automatic CUDA
    # fallback. 128 bytes/sample covers float64 inputs, FFTs, normalization,
    # band temporaries and allocator headroom. Add the serial global workspace
    # conservatively even though global matching precedes lattice execution.
    return 128 * (global_fft + min(32, threads) * lattice_fft)


def plan_resources(
    profile: HardwareProfile,
    tasks: Sequence[TaskResources],
    requested_jobs: int | None = None,
    device: str = "auto",
    requested_threads: int | None = None,
) -> ResourcePlan:
    """Plan independent episode processes without promising a speedup.

    The longest/largest episode controls every worker's RAM budget. Auto caps
    concurrent file readers/writers at four and reserves RAM for the OS and
    other apps. Explicit concurrency can exceed the I/O cap when memory and
    usable CPUs permit it; estimates are conservative rather than hard limits.
    """
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("Device must be auto, cpu, or cuda")
    if requested_jobs is not None and requested_jobs < 1:
        raise ValueError("Requested jobs must be at least 1")
    if requested_threads is not None and requested_threads < 1:
        raise ValueError("Requested threads must be at least 1")
    if device == "cuda" and not profile.cuda_available:
        raise ValueError(f"CUDA is unavailable: {profile.cuda_error or 'device self-test did not pass'}")
    if not tasks:
        return ResourcePlan(0, 0, 0, 0, 0, device, ["No selected tasks"])

    estimates = [_memory_estimate(task) for task in tasks]
    base_ram_per_job = max(estimate[0] for estimate in estimates)
    usable_cpus = max(1, profile.usable_cpus)
    physical_cpus = max(1, min(usable_cpus, profile.physical_cpus))
    reserve = max(2 * _GIB, math.ceil(profile.total_memory_bytes * .15))
    ram_budget = max(0, profile.available_memory_bytes - reserve)
    ram_jobs = ram_budget // base_ram_per_job
    if ram_jobs < 1:
        raise ValueError(
            f"Insufficient available RAM: one episode needs at least {base_ram_per_job/_GIB:.1f} GiB, "
            f"but {ram_budget/_GIB:.1f} GiB remains after the system reserve"
        )
    reasons = [
        f"Reserved {reserve/_GIB:.1f} GiB of available RAM for the system and other applications",
    ]

    uses_cuda = device != "cpu" and profile.cuda_available
    effective_device = "cuda" if uses_cuda else "cpu"
    vram_per_job = max(estimate[1] for estimate in estimates) if uses_cuda else 0
    gpu_jobs = len(tasks)
    if uses_cuda:
        if profile.gpu_free_bytes is None:
            gpu_jobs = 1
            reasons.append("VRAM availability is unknown; limited CUDA to one process")
        else:
            gpu_reserve = max(512 * 1024 ** 2, int((profile.gpu_total_bytes or profile.gpu_free_bytes) * .15))
            gpu_budget = max(0, profile.gpu_free_bytes - gpu_reserve)
            gpu_jobs = gpu_budget // vram_per_job
            if gpu_jobs < 1:
                if device == "cuda":
                    raise ValueError(
                        f"Insufficient CUDA VRAM: estimated {vram_per_job/_GIB:.1f} GiB per worker, "
                        f"{gpu_budget/_GIB:.1f} GiB available after reserve"
                    )
                uses_cuda = False
                effective_device = "cpu"
                vram_per_job = 0
                gpu_jobs = len(tasks)
                reasons.append("Available VRAM cannot fit the FFT budget; using CPU")
            else:
                reasons.append(f"Budgeted {vram_per_job/_GIB:.1f} GiB VRAM per CUDA process")

    if requested_threads is not None and requested_threads > usable_cpus:
        raise ValueError(f"Requested threads exceed the {usable_cpus} usable CPUs")
    cpu_jobs = usable_cpus // (requested_threads or 1)
    if requested_jobs is not None:
        jobs = min(requested_jobs, len(tasks))
        if jobs > ram_jobs:
            raise ValueError(f"Requested {jobs} jobs exceed the RAM budget for {ram_jobs} concurrent jobs")
        if jobs > gpu_jobs:
            raise ValueError(f"Requested {jobs} jobs exceed the CUDA VRAM budget for {gpu_jobs} concurrent jobs")
        if jobs > cpu_jobs:
            raise ValueError(f"Requested jobs and threads exceed the {usable_cpus} usable CPUs")
    else:
        jobs = min(4, len(tasks), ram_jobs, gpu_jobs, cpu_jobs, max(1, usable_cpus // 4))
        reasons.append("Automatic concurrency is capped at four jobs to limit competing media I/O")
    initial_jobs = jobs
    # Fewer processes get more CPU threads, increasing each process's live FFT
    # workspace. Recalculate both budgets together instead of dividing RAM by
    # a constant per-job estimate. Reserve CPU fallback memory in every mode.
    while jobs:
        threads = requested_threads or max(1, min(usable_cpus // jobs, physical_cpus // jobs))
        workspace = max(_cpu_fft_workspace(task, threads) for task in tasks)
        ram_per_job = base_ram_per_job + workspace
        if jobs * ram_per_job <= ram_budget:
            break
        if requested_jobs is not None:
            raise ValueError(
                f"Requested {jobs} jobs need {jobs*ram_per_job/_GIB:.1f} GiB RAM including "
                f"parallel CPU FFT workspace, but only {ram_budget/_GIB:.1f} GiB is budgeted; "
                "reduce --jobs or --threads"
            )
        jobs -= 1
    if not jobs and requested_threads is None:
        # On a many-core, low-memory host even one fully threaded job may not
        # fit. Keep the highest safe automatic thread count instead of rejecting
        # an episode that a smaller worker can process. Explicit limits stay put.
        full_threads = threads
        for candidate_threads in range(min(32, threads - 1), 0, -1):
            workspace = max(_cpu_fft_workspace(task, candidate_threads) for task in tasks)
            ram_per_job = base_ram_per_job + workspace
            threads = candidate_threads
            if ram_per_job <= ram_budget:
                jobs = 1
                reasons.append(
                    f"Reduced the single worker from {full_threads} to {threads} CPU threads "
                    "to fit RAM including FFT workspaces"
                )
                break
    if not jobs:
        raise ValueError(
            f"Insufficient RAM for CPU FFT workspace: one {threads}-thread job needs "
            f"{ram_per_job/_GIB:.1f} GiB, but only {ram_budget/_GIB:.1f} GiB is budgeted; "
            "reduce --threads or free memory"
        )
    if jobs < initial_jobs:
        reasons.append(f"Reduced concurrency from {initial_jobs} to {jobs} jobs to reserve CPU FFT workspaces")
    reasons.append(
        f"Budgeted {ram_per_job/_GIB:.1f} GiB RAM per worker, including {workspace/_GIB:.1f} GiB "
        "for CPU FFT workspaces and possible CUDA fallback"
    )
    return ResourcePlan(jobs, threads, ram_per_job, vram_per_job, ram_budget, effective_device, reasons)
