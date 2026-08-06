"""Immutable telemetry facts and learning/evaluation budget counters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Mapping
from uuid import UUID, uuid4

from rlbench.game import StepRecord


Stage = Literal["learning", "evaluation"]
SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class Event:
    """One append-only fact emitted by a benchmark run."""

    event_type: str
    run_id: str
    stage: str | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int | None = None
    event_id: str | None = None
    created_at: datetime | None = None

    @classmethod
    def from_step_record(
        cls,
        *,
        event_type: str,
        run_id: str,
        stage: str,
        record: StepRecord,
        payload: Mapping[str, Any] | None = None,
    ) -> Event:
        """Create an event that preserves a game transition as raw facts."""
        event_payload = dict(payload or {})
        event_payload.update(asdict(record))
        return cls(
            event_type=event_type,
            run_id=run_id,
            stage=stage,
            payload=event_payload,
        )

    def finalized(self, *, created_at: datetime | None = None) -> Event:
        """Inject stable envelope fields without changing the factual payload."""
        if not self.event_type:
            raise ValueError("event_type must be non-empty")
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.event_id is not None:
            UUID(self.event_id)
        timestamp = self.created_at or created_at or datetime.now(UTC)
        if timestamp.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        return Event(
            event_type=self.event_type,
            run_id=self.run_id,
            stage=self.stage,
            payload=dict(self.payload),
            schema_version=self.schema_version or SCHEMA_VERSION,
            event_id=self.event_id or str(uuid4()),
            created_at=timestamp.astimezone(UTC),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize a finalized event using JSON-native values."""
        if (
            self.schema_version is None
            or self.event_id is None
            or self.created_at is None
        ):
            raise ValueError("event must be finalized before serialization")
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "run_id": self.run_id,
            "created_at": self.created_at.astimezone(UTC).isoformat(),
            "stage": self.stage,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Event:
        """Recreate the typed envelope stored in a JSONL event line."""
        created_at = datetime.fromisoformat(str(value["created_at"]))
        if created_at.tzinfo is None:
            raise ValueError("event created_at must be timezone-aware")
        return cls(
            schema_version=int(value["schema_version"]),
            event_id=str(value["event_id"]),
            event_type=str(value["event_type"]),
            run_id=str(value["run_id"]),
            created_at=created_at.astimezone(UTC),
            stage=value.get("stage"),
            payload=dict(value.get("payload", {})),
        )


@dataclass(slots=True)
class BudgetSlice:
    """Counters charged to exactly one run stage."""

    episodes: int = 0
    env_steps: int = 0
    optimizer_steps: int = 0
    mcts_simulations: int = 0
    wall_seconds: float = 0.0
    gpu_hours: float = 0.0
    cpu_core_hours: float = 0.0

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(slots=True)
class BudgetCounters:
    """Learning and evaluation budgets with a derived, non-double-counted total."""

    learning: BudgetSlice = field(default_factory=BudgetSlice)
    evaluation: BudgetSlice = field(default_factory=BudgetSlice)

    @property
    def total(self) -> BudgetSlice:
        """Return a fresh sum so callers cannot mutate stage accounting."""
        return BudgetSlice(
            episodes=self.learning.episodes + self.evaluation.episodes,
            env_steps=self.learning.env_steps + self.evaluation.env_steps,
            optimizer_steps=self.learning.optimizer_steps + self.evaluation.optimizer_steps,
            mcts_simulations=(
                self.learning.mcts_simulations + self.evaluation.mcts_simulations
            ),
            wall_seconds=self.learning.wall_seconds + self.evaluation.wall_seconds,
            gpu_hours=self.learning.gpu_hours + self.evaluation.gpu_hours,
            cpu_core_hours=(
                self.learning.cpu_core_hours + self.evaluation.cpu_core_hours
            ),
        )

    def record_step(self, record: StepRecord, stage: Stage) -> None:
        """Charge a game transition and its terminal episode to one stage."""
        target = self._stage(stage)
        target.env_steps += 1
        if record.terminated:
            target.episodes += 1

    def record_optimizer_step(self, stage: Stage) -> None:
        self._stage(stage).optimizer_steps += 1

    def add_mcts_simulations(self, count: int, stage: Stage) -> None:
        if count < 0:
            raise ValueError("MCTS simulations cannot be negative")
        self._stage(stage).mcts_simulations += count

    def add_wall_seconds(self, seconds: float, stage: Stage) -> None:
        self._add_nonnegative("wall_seconds", seconds, stage)

    def add_resources(
        self, *, gpu_hours: float, cpu_core_hours: float, stage: Stage
    ) -> None:
        self._add_nonnegative("gpu_hours", gpu_hours, stage)
        self._add_nonnegative("cpu_core_hours", cpu_core_hours, stage)

    def as_dict(self) -> dict[str, dict[str, int | float]]:
        return {
            "learning": self.learning.as_dict(),
            "evaluation": self.evaluation.as_dict(),
            "total": self.total.as_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BudgetCounters:
        """Validate and restore conserved stage counters from an event payload."""
        if not isinstance(value, Mapping):
            raise ValueError("budgets must be a mapping")
        fields = tuple(BudgetSlice.__dataclass_fields__)

        def restore(stage: str) -> BudgetSlice:
            raw = value.get(stage)
            if not isinstance(raw, Mapping) or set(raw) != set(fields):
                raise ValueError(f"budget stage {stage!r} has invalid fields")
            restored: dict[str, int | float] = {}
            for name in fields:
                item = raw[name]
                if not isinstance(item, (int, float)) or isinstance(item, bool) or item < 0:
                    raise ValueError(f"budget field {stage}.{name} must be non-negative")
                if name in {
                    "episodes",
                    "env_steps",
                    "optimizer_steps",
                    "mcts_simulations",
                }:
                    if not isinstance(item, int):
                        raise ValueError(f"budget field {stage}.{name} must be an integer")
                    restored[name] = item
                else:
                    restored[name] = float(item)
            return BudgetSlice(**restored)

        counters = cls(learning=restore("learning"), evaluation=restore("evaluation"))
        raw_total = value.get("total")
        if not isinstance(raw_total, Mapping) or counters.total.as_dict() != dict(raw_total):
            raise ValueError("budget total is not conserved")
        return counters

    def require_at_least(self, baseline: BudgetCounters) -> None:
        """Reject event-backed counters that move behind a restored checkpoint."""
        for stage in ("learning", "evaluation"):
            current_slice = self._stage(stage)  # type: ignore[arg-type]
            baseline_slice = baseline._stage(stage)  # type: ignore[arg-type]
            for name in BudgetSlice.__dataclass_fields__:
                if getattr(current_slice, name) < getattr(baseline_slice, name):
                    raise ValueError(f"budget counter regressed at {stage}.{name}")

    def _stage(self, stage: Stage) -> BudgetSlice:
        if stage == "learning":
            return self.learning
        if stage == "evaluation":
            return self.evaluation
        raise ValueError("stage must be 'learning' or 'evaluation'")

    def _add_nonnegative(self, field_name: str, value: float, stage: Stage) -> None:
        if value < 0:
            raise ValueError(f"{field_name} cannot be negative")
        target = self._stage(stage)
        setattr(target, field_name, getattr(target, field_name) + value)
