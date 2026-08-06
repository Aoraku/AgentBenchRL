"""Behavioral tests for the append-only event ledger."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from rlbench.game import StepRecord
from rlbench.telemetry import BudgetCounters, Event, EventLedger


def test_append_finalizes_required_fields_preserves_unknowns_and_round_trips(
    tmp_path,
) -> None:
    """Dropping finalization or coercing unknown values loses audit facts."""
    ledger = EventLedger(tmp_path / "events.jsonl")
    source = Event.from_step_record(
        event_type="episode_finished",
        run_id="run-7",
        stage="learning",
        record=StepRecord(player=1, action=3, terminated=True),
        payload={"opponent_checkpoint": None},
    )

    finalized = ledger.append(source)

    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    stored = json.loads(lines[0])
    assert stored["schema_version"] == 1
    assert stored["event_id"] == finalized.event_id
    assert stored["event_type"] == "episode_finished"
    assert stored["run_id"] == "run-7"
    assert stored["created_at"].endswith("+00:00")
    assert stored["payload"] == {
        "action": 3,
        "opponent_checkpoint": None,
        "player": 1,
        "terminated": True,
    }

    events = list(ledger.read())
    assert events == [finalized]
    assert isinstance(events[0].created_at, datetime)
    assert events[0].created_at.tzinfo == UTC


def test_fresh_ledger_derives_checkpoint_summary_from_budget_and_resource_facts(
    tmp_path,
) -> None:
    """Merging evaluation into learning would make budget curves misleading."""
    counters = BudgetCounters()
    counters.record_step(StepRecord(player=0, action=1, terminated=False), "learning")
    counters.record_step(StepRecord(player=1, action=0, terminated=True), "learning")
    counters.record_step(StepRecord(player=0, action=1, terminated=True), "evaluation")
    counters.record_optimizer_step("learning")
    counters.add_wall_seconds(90.0, "evaluation")

    assert counters.learning.env_steps == 2
    assert counters.learning.episodes == 1
    assert counters.learning.optimizer_steps == 1
    assert counters.evaluation.env_steps == 1
    assert counters.evaluation.episodes == 1
    assert counters.evaluation.wall_seconds == 90.0
    assert counters.total.env_steps == 3
    assert counters.total.episodes == 2
    assert counters.total.optimizer_steps == 1
    assert counters.total.wall_seconds == 90.0

    events_path = tmp_path / "events.jsonl"
    ledger = EventLedger(events_path)
    ledger.append_budget_snapshot(run_id="run-7", counters=counters)
    ledger.append(
        Event(
            event_type="resource_sampled",
            run_id="run-7",
            stage="learning",
            payload={
                "host_id": "host-a",
                "timestamp": "2026-08-06T09:00:00+00:00",
                "gpu_measurement_available": True,
                "gpu_devices": [{"identity": "GPU-0", "utilization_percent": 50.0}],
                "cpu_count": 4,
                "cpu_utilization_percent": 50.0,
                "process_ram_bytes": [100],
                "host_ram_bytes": 1_000,
            },
        )
    )
    ledger.append(
        Event(
            event_type="resource_sampled",
            run_id="run-7",
            stage="learning",
            payload={
                "host_id": "host-a",
                "timestamp": "2026-08-06T10:00:00+00:00",
                "gpu_measurement_available": True,
                "gpu_devices": [{"identity": "GPU-0", "utilization_percent": 50.0}],
                "cpu_count": 4,
                "cpu_utilization_percent": 50.0,
                "process_ram_bytes": [120],
                "host_ram_bytes": 1_100,
            },
        )
    )

    summary_path = tmp_path / "checkpoints" / "checkpoint-0001.json"
    reopened = EventLedger(events_path)
    derived = reopened.write_checkpoint_summary(
        summary_path,
        checkpoint_id="checkpoint-0001",
        run_id="run-7",
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary == derived
    assert summary["checkpoint_id"] == "checkpoint-0001"
    assert summary["budgets"]["learning"]["env_steps"] == 2
    assert summary["budgets"]["evaluation"]["episodes"] == 1
    assert summary["budgets"]["total"]["env_steps"] == 3
    assert summary["budgets"]["learning"]["gpu_hours"] == 1.0
    assert summary["budgets"]["learning"]["cpu_core_hours"] == 2.0
    assert summary["resources"]["learning"]["effective_gpu_hours"] == 0.5
    assert not list(summary_path.parent.glob("*.tmp"))
