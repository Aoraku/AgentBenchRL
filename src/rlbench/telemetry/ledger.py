"""Durable append-only event storage and atomic derived summaries."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator, Mapping

from .events import BudgetCounters, Event


class EventLedger:
    """Owns a JSONL fact stream for one run artifact directory."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, event: Event) -> Event:
        """Finalize and durably append one event without rewriting past facts."""
        finalized = event.finalized()
        encoded = json.dumps(
            finalized.to_dict(),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        return finalized

    def read(self) -> Iterator[Event]:
        """Yield stored facts in append order as typed event envelopes."""
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    yield Event.from_dict(raw)
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"invalid event at {self.path}:{line_number}"
                    ) from exc

    def append_budget_snapshot(self, *, run_id: str, counters: BudgetCounters) -> Event:
        """Persist the current budget state as an append-only fact."""
        return self.append(
            Event(
                event_type="budget_snapshot",
                run_id=run_id,
                payload={"budgets": counters.as_dict()},
            )
        )

    def latest_budget_counters(self, *, run_id: str) -> BudgetCounters | None:
        """Restore the latest conserved budget fact for one run."""
        latest: BudgetCounters | None = None
        for event in self.read():
            if event.run_id != run_id or event.event_type != "budget_snapshot":
                continue
            latest = BudgetCounters.from_dict(event.payload.get("budgets"))
        return latest

    def derive_checkpoint_summary(
        self, *, checkpoint_id: str, run_id: str
    ) -> dict[str, Any]:
        """Rebuild a checkpoint summary entirely from persisted run facts."""
        from .resources import (
            ResourceTotals,
            resource_sample_from_payload,
            summarize_resource_samples,
        )

        budgets: dict[str, dict[str, int | float | None]] = BudgetCounters().as_dict()
        samples_by_host: dict[str, list[Any]] = {}
        for event in self.read():
            if event.run_id != run_id:
                continue
            if event.event_type == "budget_snapshot":
                snapshot = event.payload.get("budgets")
                if not isinstance(snapshot, Mapping):
                    raise ValueError("budget_snapshot payload must contain budgets")
                budgets = json.loads(json.dumps(snapshot))
            elif event.event_type == "resource_sampled":
                if event.stage not in {"learning", "evaluation"}:
                    raise ValueError("resource_sampled event requires a valid stage")
                host_id = event.payload.get("host_id")
                if not isinstance(host_id, str) or not host_id:
                    raise ValueError("resource_sampled event requires host_id")
                samples_by_host.setdefault(host_id, []).append(
                    resource_sample_from_payload(event.payload, stage=event.stage)
                )

        host_totals = [
            summarize_resource_samples(samples) for samples in samples_by_host.values()
        ]
        resources = {
            stage: _combine_resource_totals(
                [totals[stage] for totals in host_totals], ResourceTotals
            )
            for stage in ("learning", "evaluation", "total")
        }
        for stage in ("learning", "evaluation", "total"):
            budgets[stage]["gpu_hours"] = resources[stage].allocated_gpu_hours
            budgets[stage]["cpu_core_hours"] = resources[stage].cpu_core_hours
        return {
            "checkpoint_id": checkpoint_id,
            "run_id": run_id,
            "budgets": budgets,
            "resources": {stage: asdict(total) for stage, total in resources.items()},
        }

    def write_checkpoint_summary(
        self,
        path: str | Path,
        *,
        checkpoint_id: str,
        run_id: str,
    ) -> dict[str, Any]:
        """Atomically write a checkpoint summary derived from this fact stream."""
        summary = self.derive_checkpoint_summary(
            checkpoint_id=checkpoint_id, run_id=run_id
        )
        self._atomic_write(path, summary)
        return summary

    @staticmethod
    def _atomic_write(path: str | Path, summary: Mapping[str, Any]) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            text=True,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(summary, stream, allow_nan=False, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, destination)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise


def _combine_resource_totals(
    totals: list[Any], totals_type: type[Any]
) -> Any:
    """Sum independent host streams while preserving unknown measurements."""
    if not totals:
        return totals_type()
    return totals_type(
        allocated_gpu_hours=_sum_optional(
            [total.allocated_gpu_hours for total in totals]
        ),
        effective_gpu_hours=_sum_optional(
            [total.effective_gpu_hours for total in totals]
        ),
        cpu_core_hours=_sum_optional([total.cpu_core_hours for total in totals]),
        wall_seconds=sum(total.wall_seconds for total in totals),
    )


def _sum_optional(values: list[float | None]) -> float | None:
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)
