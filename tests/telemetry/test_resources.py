"""Integration tests for host-owned resource interval accounting."""

from __future__ import annotations

import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from rlbench.telemetry import (
    EventLedger,
    GPUDeviceSample,
    ResourceSample,
    ResourceSampler,
)


def test_synthetic_samples_conserve_stage_gpu_and_cpu_hours_once(tmp_path) -> None:
    """Counting per-process samples would double-count one host's devices."""
    start = datetime(2026, 8, 6, 9, tzinfo=UTC)
    learning_gpus = (
        GPUDeviceSample(identity="GPU-0", utilization_percent=50.0),
        GPUDeviceSample(identity="GPU-1", utilization_percent=100.0),
    )
    evaluation_gpus = (
        GPUDeviceSample(identity="GPU-0", utilization_percent=25.0),
        GPUDeviceSample(identity="GPU-1", utilization_percent=75.0),
    )
    samples = (
        ResourceSample(
            timestamp=start,
            stage="learning",
            gpu_devices=learning_gpus,
            gpu_measurement_available=True,
            cpu_count=8,
            cpu_utilization_percent=25.0,
            process_ram_bytes=(100, 200),
            host_ram_bytes=1_000,
        ),
        ResourceSample(
            timestamp=start + timedelta(minutes=30),
            stage="learning",
            gpu_devices=learning_gpus,
            gpu_measurement_available=True,
            cpu_count=8,
            cpu_utilization_percent=25.0,
            process_ram_bytes=(150, 250),
            host_ram_bytes=1_100,
        ),
        ResourceSample(
            timestamp=start + timedelta(minutes=60),
            stage="evaluation",
            gpu_devices=evaluation_gpus,
            gpu_measurement_available=True,
            cpu_count=8,
            cpu_utilization_percent=50.0,
            process_ram_bytes=(300, 400),
            host_ram_bytes=1_200,
        ),
        ResourceSample(
            timestamp=start + timedelta(minutes=90),
            stage="evaluation",
            gpu_devices=evaluation_gpus,
            gpu_measurement_available=True,
            cpu_count=8,
            cpu_utilization_percent=50.0,
            process_ram_bytes=(350, 450),
            host_ram_bytes=1_300,
        ),
    )
    sampler = ResourceSampler(
        run_id="run-2",
        host_id="host-a",
        ledger=EventLedger(tmp_path / "events.jsonl"),
    )

    sampler.extend(samples)
    totals = sampler.totals()

    assert totals["learning"].allocated_gpu_hours == 2.0
    assert totals["learning"].effective_gpu_hours == 1.5
    assert totals["learning"].cpu_core_hours == 2.0
    assert totals["evaluation"].allocated_gpu_hours == 1.0
    assert totals["evaluation"].effective_gpu_hours == 0.5
    assert totals["evaluation"].cpu_core_hours == 2.0
    assert totals["total"].allocated_gpu_hours == 3.0
    assert totals["total"].effective_gpu_hours == 2.0
    assert totals["total"].cpu_core_hours == 4.0
    assert totals["total"].wall_seconds == 5_400.0
    sampler.close()


def test_one_sampler_owns_each_run_host_stream_and_nvml_unknowns_are_null(tmp_path) -> None:
    """A second owner would charge the same host resources twice."""
    ledger = EventLedger(tmp_path / "events.jsonl")
    first = ResourceSampler(run_id="run-3", host_id="host-a", ledger=ledger)
    try:
        with pytest.raises(RuntimeError, match="already owns"):
            ResourceSampler(run_id="run-3", host_id="host-a", ledger=ledger)

        sample = ResourceSample(
            timestamp=datetime(2026, 8, 6, 9, tzinfo=UTC),
            stage="learning",
            gpu_devices=(GPUDeviceSample(identity="GPU-unknown"),),
            gpu_measurement_available=True,
            cpu_count=4,
            cpu_utilization_percent=50.0,
        )
        assert sample.gpu_devices[0].model is None
        assert sample.gpu_devices[0].memory_used_bytes is None
        assert sample.gpu_devices[0].utilization_percent is None
    finally:
        first.close()


def test_populated_direct_gpu_samples_are_available_without_an_explicit_flag(
    tmp_path,
) -> None:
    """Defaulting supplied GPU facts to unknown would erase their measured cost."""
    start = datetime(2026, 8, 6, 9, tzinfo=UTC)
    sampler = ResourceSampler(
        run_id="run-direct-gpu",
        host_id="host-a",
        ledger=EventLedger(tmp_path / "events.jsonl"),
    )
    try:
        sampler.extend(
            (
                ResourceSample(
                    timestamp=start,
                    stage="learning",
                    gpu_devices=(
                        GPUDeviceSample(identity="GPU-0", utilization_percent=50.0),
                    ),
                    cpu_count=4,
                    cpu_utilization_percent=50.0,
                ),
                ResourceSample(
                    timestamp=start + timedelta(hours=1),
                    stage="learning",
                    gpu_devices=(
                        GPUDeviceSample(identity="GPU-0", utilization_percent=50.0),
                    ),
                    cpu_count=4,
                    cpu_utilization_percent=50.0,
                ),
            )
        )

        totals = sampler.totals()

        assert totals["learning"].allocated_gpu_hours == 1.0
        assert totals["learning"].effective_gpu_hours == 0.5
    finally:
        sampler.close()


def test_sample_marks_unavailable_nvml_as_unknown_and_does_not_charge_zero_gpu_hours(
    monkeypatch, tmp_path
) -> None:
    """Treating an unavailable NVML module as zero GPUs loses resource cost."""

    monkeypatch.setitem(sys.modules, "pynvml", None)
    events_path = tmp_path / "events.jsonl"
    sampler = ResourceSampler(
        run_id="run-4",
        host_id="host-a",
        ledger=EventLedger(events_path),
    )
    try:
        first = sampler.sample("learning")
        sampler.add(replace(first, timestamp=first.timestamp + timedelta(hours=1)))

        totals = sampler.totals()

        assert first.gpu_measurement_available is False
        assert totals["learning"].allocated_gpu_hours is None
        assert totals["learning"].effective_gpu_hours is None
        assert totals["total"].allocated_gpu_hours is None
        persisted = list(EventLedger(events_path).read())
        assert [event.event_type for event in persisted] == [
            "resource_sampled",
            "resource_sampled",
        ]
        assert all(event.event_id and event.created_at for event in persisted)
    finally:
        sampler.close()


def test_sample_charges_only_cuda_visible_nvml_devices(monkeypatch, tmp_path) -> None:
    """NVML sees the whole host, but one run owns only its CUDA-visible allocation."""
    fake_nvml = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlDeviceGetCount=lambda: 3,
        nvmlDeviceGetHandleByIndex=lambda index: index,
        nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(
            total=(handle + 1) * 1_000, used=(handle + 1) * 100
        ),
        nvmlDeviceGetUtilizationRates=lambda handle: SimpleNamespace(
            gpu=float(handle * 10)
        ),
        nvmlDeviceGetUUID=lambda handle: f"GPU-{handle}",
        nvmlDeviceGetName=lambda handle: f"accelerator-{handle}",
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,0")
    sampler = ResourceSampler(
        run_id="visible-allocation",
        host_id="host-a",
        ledger=EventLedger(tmp_path / "events.jsonl"),
    )
    try:
        sample = sampler.sample("learning")
    finally:
        sampler.close()

    assert sample.gpu_measurement_available is True
    assert [device.identity for device in sample.gpu_devices] == ["GPU-2", "GPU-0"]
    assert [device.memory_total_bytes for device in sample.gpu_devices] == [3_000, 1_000]
