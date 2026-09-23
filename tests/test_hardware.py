"""Portable planning must obey measured CPU, RAM, and CUDA constraints."""

import importlib
from types import SimpleNamespace

import pytest


GIB = 1024 ** 3


def _hardware():
    return importlib.import_module("adsync.hardware")


def _profile(**changes):
    values = dict(
        logical_cpus=32, physical_cpus=16, usable_cpus=32,
        total_memory_bytes=64 * GIB, available_memory_bytes=50 * GIB,
        cuda_available=False,
    )
    values.update(changes)
    return _hardware().HardwareProfile(**values)


def _tasks(n=10, **changes):
    values = dict(video_duration=3600., ad_duration=3600., channels=2, sample_rate=48000)
    values.update(changes)
    return [_hardware().TaskResources(**values) for _ in range(n)]


def test_cpu_detection_uses_affinity_and_does_not_probe_cuda(monkeypatch):
    hardware = _hardware()
    monkeypatch.setattr(hardware.psutil, "cpu_count", lambda logical=True: 32 if logical else 16)
    monkeypatch.setattr(hardware.psutil, "Process", lambda: SimpleNamespace(cpu_affinity=lambda: [2, 3, 4]))
    monkeypatch.setattr(hardware.psutil, "virtual_memory", lambda: SimpleNamespace(total=32*GIB, available=12*GIB))
    import adsync.compute
    monkeypatch.setattr(adsync.compute, "_load_cuda", lambda: pytest.fail("CPU detection must not touch CUDA"))
    profile = hardware.detect_hardware("cpu")
    assert profile.usable_cpus == 3
    assert profile.physical_cpus == 16
    assert profile.available_memory_bytes == 12 * GIB
    assert not profile.cuda_available


def test_detection_falls_back_when_affinity_is_not_supported(monkeypatch):
    hardware = _hardware()
    monkeypatch.setattr(hardware.psutil, "cpu_count", lambda logical=True: 8 if logical else 4)
    monkeypatch.setattr(hardware.psutil, "Process", lambda: SimpleNamespace())
    assert hardware.detect_hardware("cpu").usable_cpus == 8


def test_cuda_detection_uses_working_backend_and_captures_vram(monkeypatch):
    hardware = _hardware()
    import adsync.compute

    class WorkingBackend:
        active_backend = "cuda"
        device_name = "portable test GPU"
        fallback_reason = None

        def __init__(self, device):
            assert device == "auto"

        def memory_info(self):
            return 6 * GIB, 8 * GIB

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(adsync.compute, "CorrelationBackend", WorkingBackend)
    profile = hardware.detect_hardware("auto")
    assert profile.cuda_available
    assert profile.gpu_free_bytes == 6 * GIB
    assert profile.gpu_total_bytes == 8 * GIB


def test_plan_reserves_ram_and_never_exceeds_affinity_cpu_budget():
    profile = _profile(usable_cpus=6)
    plan = _hardware().plan_resources(profile, _tasks(), device="cpu")
    assert plan.jobs >= 1
    assert plan.jobs * plan.threads_per_job <= 6
    assert plan.jobs * plan.memory_per_job_bytes <= plan.ram_budget_bytes
    assert plan.ram_budget_bytes < profile.available_memory_bytes


def test_more_channels_and_longer_episodes_reduce_memory_concurrency():
    h = _hardware()
    profile = _profile()
    stereo = h.plan_resources(profile, _tasks(), device="cpu")
    surround = h.plan_resources(profile, _tasks(channels=8), device="cpu")
    longer = h.plan_resources(profile, _tasks(video_duration=7200., ad_duration=7200.), device="cpu")
    assert surround.memory_per_job_bytes > stereo.memory_per_job_bytes
    assert longer.memory_per_job_bytes > stereo.memory_per_job_bytes
    assert surround.jobs < stereo.jobs


def test_worst_episode_controls_memory_planning():
    h = _hardware()
    small = _tasks(3, video_duration=300., ad_duration=300.)
    large = _tasks(1, video_duration=7200., ad_duration=7200., channels=6)
    combined = h.plan_resources(_profile(), small + large, device="cpu")
    single = h.plan_resources(_profile(), large, device="cpu")
    assert combined.memory_per_job_bytes == single.memory_per_job_bytes


def test_insufficient_ram_rejects_launch_instead_of_forcing_one_job():
    with pytest.raises(ValueError, match="(?i)memory|ram"):
        _hardware().plan_resources(_profile(available_memory_bytes=GIB), _tasks(), device="cpu")


def test_explicit_threads_and_jobs_cannot_oversubscribe_cpu_or_ram():
    h = _hardware()
    with pytest.raises(ValueError, match="(?i)cpu|thread"):
        h.plan_resources(_profile(usable_cpus=8), _tasks(), requested_jobs=4, requested_threads=4, device="cpu")
    with pytest.raises(ValueError, match="(?i)memory|ram"):
        h.plan_resources(_profile(available_memory_bytes=12*GIB), _tasks(), requested_jobs=8, device="cpu")


def test_explicit_jobs_can_override_automatic_io_cap_when_resources_fit():
    h = _hardware()
    tasks = _tasks(video_duration=60., ad_duration=60.)
    auto = h.plan_resources(_profile(), tasks, device="cpu")
    explicit = h.plan_resources(_profile(), tasks, requested_jobs=8, device="cpu")
    assert auto.jobs <= 4
    assert explicit.jobs == 8
    assert explicit.jobs * explicit.threads_per_job <= 32


def test_gpu_budget_limits_parallel_jobs_and_cpu_selection_ignores_gpu():
    h = _hardware()
    profile = _profile(cuda_available=True, gpu_name="any CUDA GPU", gpu_free_bytes=4*GIB, gpu_total_bytes=8*GIB)
    gpu = h.plan_resources(profile, _tasks(), device="auto")
    cpu = h.plan_resources(profile, _tasks(), device="cpu")
    assert gpu.jobs < cpu.jobs
    assert gpu.gpu_memory_per_job_bytes > 0
    assert cpu.gpu_memory_per_job_bytes == 0


def test_auto_uses_cpu_when_vram_cannot_fit_one_job_but_forced_cuda_errors():
    h = _hardware()
    profile = _profile(cuda_available=True, gpu_free_bytes=256*1024**2, gpu_total_bytes=4*GIB)
    assert h.plan_resources(profile, _tasks(), device="auto").device == "cpu"
    with pytest.raises(ValueError, match="(?i)cuda|vram|gpu"):
        h.plan_resources(profile, _tasks(), device="cuda")


def test_forced_cuda_cannot_plan_an_unavailable_device():
    with pytest.raises(ValueError, match="(?i)cuda"):
        _hardware().plan_resources(_profile(), _tasks(), device="cuda")


def test_invalid_or_unknown_durations_are_not_treated_as_zero_cost():
    h = _hardware()
    with pytest.raises(ValueError, match="duration"):
        h.plan_resources(_profile(), _tasks(video_duration=float("nan")), device="cpu")
    with pytest.raises(ValueError, match="duration"):
        h.plan_resources(_profile(), _tasks(ad_duration=0.), device="cpu")


@pytest.mark.parametrize("device", ["cpu", "auto"])
def test_short_episodes_on_many_core_low_ram_host_reserve_parallel_fft_workspaces(device):
    h = _hardware()
    profile = _profile(
        physical_cpus=16, total_memory_bytes=16*GIB, available_memory_bytes=14*GIB,
        cuda_available=device == "auto", gpu_free_bytes=24*GIB, gpu_total_bytes=24*GIB,
    )
    plan = h.plan_resources(profile, _tasks(4, video_duration=600., ad_duration=600.), device=device)
    # Previously four jobs fit the waveform-only estimate while their concurrent
    # CPU FFTs (including an auto-mode CUDA failure) could exceed available RAM.
    assert plan.jobs < 4
    assert plan.jobs * plan.memory_per_job_bytes <= plan.ram_budget_bytes
    # A 240 s lattice radius at 4 kHz pads to 2^21: 256 MiB / CPU worker.
    # The serial full-rate global FFT adds another 1 GiB workspace.
    assert plan.memory_per_job_bytes > GIB + plan.threads_per_job * 256 * 1024**2


def test_explicit_jobs_and_threads_reject_ram_overcommit_from_fft_workspaces():
    h = _hardware()
    profile = _profile(total_memory_bytes=16*GIB, available_memory_bytes=14*GIB)
    with pytest.raises(ValueError, match="(?i)ram.*fft|fft.*ram|memory"):
        h.plan_resources(profile, _tasks(4, video_duration=600., ad_duration=600.),
                         requested_jobs=4, requested_threads=4, device="cpu")


def test_large_host_keeps_four_six_thread_jobs_after_workspace_reservation():
    h = _hardware()
    profile = _profile(physical_cpus=24, available_memory_bytes=45*GIB,
                       cuda_available=True, gpu_free_bytes=22*GIB, gpu_total_bytes=24*GIB)
    plan = h.plan_resources(profile, _tasks(4, video_duration=600., ad_duration=600.), device="auto")
    assert (plan.jobs, plan.threads_per_job) == (4, 6)
    assert plan.memory_per_job_bytes > 4*GIB


def test_auto_reduces_threads_when_one_full_cpu_job_cannot_fit_memory():
    h = _hardware()
    profile = _profile(total_memory_bytes=8*GIB, available_memory_bytes=7*GIB)
    tasks = _tasks(4, video_duration=600., ad_duration=600.)
    plan = h.plan_resources(profile, tasks, device="cpu")
    assert plan.jobs == 1
    assert 1 <= plan.threads_per_job < profile.physical_cpus
    assert plan.memory_per_job_bytes <= plan.ram_budget_bytes
    with pytest.raises(ValueError, match="(?i)ram|memory"):
        h.plan_resources(profile, tasks, device="cpu", requested_threads=16)


def test_one_thread_minimum_still_fails_when_fft_and_base_memory_cannot_fit():
    h = _hardware()
    profile = _profile(total_memory_bytes=8*GIB, available_memory_bytes=5*GIB)
    with pytest.raises(ValueError, match="(?i)ram|memory"):
        h.plan_resources(profile, _tasks(4, video_duration=600., ad_duration=600.), device="cpu")
