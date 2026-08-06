"""Host-owned resource samples and interval-based cost accounting."""

from __future__ import annotations

import socket
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import os
from typing import Any, ClassVar, Literal, Mapping, Sequence

import psutil

from .events import Event
from .ledger import EventLedger


Stage = Literal["learning", "evaluation"]


@dataclass(frozen=True, slots=True)
class GPUDeviceSample:
    """One GPU's instantaneous facts; unsupported NVML readings stay ``None``."""

    identity: str
    model: str | None = None
    memory_total_bytes: int | None = None
    memory_used_bytes: int | None = None
    utilization_percent: float | None = None

    def __post_init__(self) -> None:
        if not self.identity:
            raise ValueError("GPU identity must be non-empty")
        if self.utilization_percent is not None and not 0 <= self.utilization_percent <= 100:
            raise ValueError("GPU utilization must be in [0, 100]")


@dataclass(frozen=True, slots=True)
class ResourceSample:
    """A single host sample attributed to one mutually exclusive run stage."""

    timestamp: datetime
    stage: Stage
    gpu_devices: tuple[GPUDeviceSample, ...] = ()
    gpu_measurement_available: bool | None = None
    cpu_count: int | None = None
    cpu_utilization_percent: float | None = None
    cpu_model: str | None = None
    process_ram_bytes: tuple[int, ...] = ()
    host_ram_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.gpu_measurement_available is None:
            object.__setattr__(
                self, "gpu_measurement_available", bool(self.gpu_devices)
            )
        if self.timestamp.tzinfo is None:
            raise ValueError("resource sample timestamps must be timezone-aware")
        if self.cpu_count is not None and self.cpu_count <= 0:
            raise ValueError("cpu_count must be positive")
        if self.cpu_utilization_percent is not None and not 0 <= self.cpu_utilization_percent <= 100:
            raise ValueError("CPU utilization must be in [0, 100]")
        if any(memory < 0 for memory in self.process_ram_bytes):
            raise ValueError("process RAM cannot be negative")
        if self.host_ram_bytes is not None and self.host_ram_bytes < 0:
            raise ValueError("host RAM cannot be negative")


@dataclass(frozen=True, slots=True)
class ResourceTotals:
    """Time-integrated resource costs for a stage."""

    allocated_gpu_hours: float | None = 0.0
    effective_gpu_hours: float | None = 0.0
    cpu_core_hours: float | None = 0.0
    wall_seconds: float = 0.0


@dataclass(slots=True)
class _MutableTotals:
    allocated_gpu_hours: float = 0.0
    effective_gpu_hours: float = 0.0
    cpu_core_hours: float = 0.0
    wall_seconds: float = 0.0
    effective_unknown: bool = False
    gpu_unknown: bool = False
    cpu_unknown: bool = False

    def finalized(self) -> ResourceTotals:
        return ResourceTotals(
            allocated_gpu_hours=None if self.gpu_unknown else self.allocated_gpu_hours,
            effective_gpu_hours=(
                None if self.effective_unknown else self.effective_gpu_hours
            ),
            cpu_core_hours=None if self.cpu_unknown else self.cpu_core_hours,
            wall_seconds=self.wall_seconds,
        )


class ResourceSampler:
    """The sole owner of a run-host stream, preventing shared-device double counts.

    Every observed interval is attributed to its earlier sample's stage.
    """

    _owners: ClassVar[set[tuple[str, str]]] = set()
    _owners_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self, *, run_id: str, host_id: str | None = None, ledger: EventLedger
    ) -> None:
        if not run_id:
            raise ValueError("run_id must be non-empty")
        self.run_id = run_id
        self.host_id = host_id or socket.gethostname()
        self.ledger = ledger
        self._owner_key = (self.run_id, self.host_id)
        with self._owners_lock:
            if self._owner_key in self._owners:
                raise RuntimeError(
                    f"a sampler already owns run-host stream {self._owner_key!r}"
                )
            self._owners.add(self._owner_key)
        self._samples: list[ResourceSample] = []
        self._closed = False

    def __enter__(self) -> ResourceSampler:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def samples(self) -> tuple[ResourceSample, ...]:
        """Return raw immutable facts in timestamp order."""
        return tuple(self._samples)

    def sample(self, stage: Stage) -> ResourceSample:
        """Capture a real host sample using psutil and optional NVML telemetry."""
        self._require_open()
        process_ram: tuple[int, ...]
        try:
            process_ram = (psutil.Process().memory_info().rss,)
        except psutil.Error:
            process_ram = ()
        gpu_devices, gpu_measurement_available = self._sample_gpus()
        sample = ResourceSample(
            timestamp=datetime.now(UTC),
            stage=stage,
            gpu_devices=gpu_devices,
            gpu_measurement_available=gpu_measurement_available,
            cpu_count=psutil.cpu_count(),
            cpu_utilization_percent=psutil.cpu_percent(interval=None),
            process_ram_bytes=process_ram,
            host_ram_bytes=psutil.virtual_memory().used,
        )
        self.add(sample)
        return sample

    def add(self, sample: ResourceSample) -> None:
        """Append an externally captured fact after enforcing time ordering."""
        self._require_open()
        if self._samples and sample.timestamp < self._samples[-1].timestamp:
            raise ValueError("resource sample timestamps must be nondecreasing")
        self.ledger.append(
            Event(
                event_type="resource_sampled",
                run_id=self.run_id,
                stage=sample.stage,
                payload=resource_sample_to_payload(sample, host_id=self.host_id),
            )
        )
        self._samples.append(sample)

    def extend(self, samples: tuple[ResourceSample, ...]) -> None:
        for sample in samples:
            self.add(sample)

    def totals(self) -> dict[str, ResourceTotals]:
        """Integrate host resources separately for learning, evaluation, and total."""
        return summarize_resource_samples(self._samples)

    @staticmethod
    def _sample_gpus() -> tuple[tuple[GPUDeviceSample, ...], bool]:
        try:
            import pynvml  # type: ignore[import-not-found]

            pynvml.nvmlInit()
            try:
                devices: list[GPUDeviceSample] = []
                for handle in _visible_nvml_handles(pynvml):
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    identity = pynvml.nvmlDeviceGetUUID(handle)
                    model = pynvml.nvmlDeviceGetName(handle)
                    devices.append(
                        GPUDeviceSample(
                            identity=(
                                identity.decode() if isinstance(identity, bytes) else identity
                            ),
                            model=model.decode() if isinstance(model, bytes) else model,
                            memory_total_bytes=memory.total,
                            memory_used_bytes=memory.used,
                            utilization_percent=float(utilization.gpu),
                        )
                    )
                return tuple(devices), True
            finally:
                pynvml.nvmlShutdown()
        except Exception:
            return (), False

    def close(self) -> None:
        """Release ownership after the run-host stream stops sampling."""
        if self._closed:
            return
        with self._owners_lock:
            self._owners.discard(self._owner_key)
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("resource sampler is closed")


def summarize_resource_samples(
    samples: Sequence[ResourceSample],
) -> dict[str, ResourceTotals]:
    """Integrate one host stream by attributing each interval to its first fact."""
    by_stage = {"learning": _MutableTotals(), "evaluation": _MutableTotals()}
    for earlier, later in zip(samples, samples[1:]):
        duration_seconds = (later.timestamp - earlier.timestamp).total_seconds()
        if duration_seconds == 0:
            continue
        target = by_stage[earlier.stage]
        hours = duration_seconds / 3600.0
        target.wall_seconds += duration_seconds
        if not earlier.gpu_measurement_available:
            target.gpu_unknown = True
            target.effective_unknown = True
        else:
            target.allocated_gpu_hours += len(earlier.gpu_devices) * hours
            if any(
                device.utilization_percent is None for device in earlier.gpu_devices
            ):
                target.effective_unknown = True
            else:
                target.effective_gpu_hours += sum(
                    device.utilization_percent / 100.0
                    for device in earlier.gpu_devices
                    if device.utilization_percent is not None
                ) * hours
        if earlier.cpu_count is None or earlier.cpu_utilization_percent is None:
            target.cpu_unknown = True
        else:
            target.cpu_core_hours += (
                earlier.cpu_count * earlier.cpu_utilization_percent / 100.0 * hours
            )
    learning = by_stage["learning"].finalized()
    evaluation = by_stage["evaluation"].finalized()
    return {
        "learning": learning,
        "evaluation": evaluation,
        "total": ResourceTotals(
            allocated_gpu_hours=_sum_known(
                learning.allocated_gpu_hours, evaluation.allocated_gpu_hours
            ),
            effective_gpu_hours=_sum_known(
                learning.effective_gpu_hours, evaluation.effective_gpu_hours
            ),
            cpu_core_hours=_sum_known(
                learning.cpu_core_hours, evaluation.cpu_core_hours
            ),
            wall_seconds=learning.wall_seconds + evaluation.wall_seconds,
        ),
    }


def resource_sample_to_payload(
    sample: ResourceSample, *, host_id: str
) -> dict[str, Any]:
    """Encode raw host facts for the append-only event stream."""
    return {
        "host_id": host_id,
        "timestamp": sample.timestamp.astimezone(UTC).isoformat(),
        "gpu_devices": [asdict(device) for device in sample.gpu_devices],
        "gpu_measurement_available": sample.gpu_measurement_available,
        "cpu_count": sample.cpu_count,
        "cpu_utilization_percent": sample.cpu_utilization_percent,
        "cpu_model": sample.cpu_model,
        "process_ram_bytes": list(sample.process_ram_bytes),
        "host_ram_bytes": sample.host_ram_bytes,
    }


def resource_sample_from_payload(
    payload: Mapping[str, Any], *, stage: Stage
) -> ResourceSample:
    """Rebuild a resource fact from its JSONL payload."""
    return ResourceSample(
        timestamp=datetime.fromisoformat(str(payload["timestamp"])),
        stage=stage,
        gpu_devices=tuple(
            GPUDeviceSample(**dict(device)) for device in payload["gpu_devices"]
        ),
        gpu_measurement_available=bool(payload["gpu_measurement_available"]),
        cpu_count=payload.get("cpu_count"),
        cpu_utilization_percent=payload.get("cpu_utilization_percent"),
        cpu_model=payload.get("cpu_model"),
        process_ram_bytes=tuple(payload.get("process_ram_bytes", ())),
        host_ram_bytes=payload.get("host_ram_bytes"),
    )

def _sum_known(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left + right


def _visible_nvml_handles(pynvml: Any) -> tuple[Any, ...]:
    handles = tuple(
        pynvml.nvmlDeviceGetHandleByIndex(index)
        for index in range(pynvml.nvmlDeviceGetCount())
    )
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return handles
    tokens = tuple(token.strip() for token in raw.split(",") if token.strip())
    if not tokens or tokens == ("-1",):
        return ()
    selected: list[Any] = []
    for token in tokens:
        if token.isdecimal():
            index = int(token)
            if index >= len(handles):
                raise ValueError("CUDA_VISIBLE_DEVICES index is outside NVML inventory")
            handle = handles[index]
        else:
            handle = pynvml.nvmlDeviceGetHandleByUUID(token)
        if handle not in selected:
            selected.append(handle)
    return tuple(selected)
