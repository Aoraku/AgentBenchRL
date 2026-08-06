"""Immutable stage contract for the Task 10 SnakeGo lineage."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class Task10Stage:
    checkpoint_index: int
    kind: Literal["selfplay", "expert"]
    episodes: int
    training_steps: int
    seed: int | None
    opponent_rank: str | None = None
    expert_demo: bool = False
    opening_moves: int = 0
    opening_weight: float = 1.0
    final_continuation_gpu_hour_ceiling: float | None = None


def task10_stage_plan() -> tuple[Task10Stage, ...]:
    stages = [
        Task10Stage(
            checkpoint_index=index,
            kind="selfplay",
            episodes=16,
            training_steps=64,
            seed=None,
        )
        for index in range(1, 11)
    ]
    expert_stages = (
        (11, "rank15", 64, 1825803012, False, 0, 1.0),
        (12, "rank15", 64, 1289487500, True, 0, 1.0),
        (13, "rank15", 128, 1554186166, True, 0, 1.0),
        (14, "rank6", 128, 684982806, True, 0, 1.0),
        (15, "rank5", 128, 1664431547, True, 0, 1.0),
        (16, "rank15", 256, 1015345658, True, 16, 32.0),
        (17, "rank6", 256, 2145794352, True, 16, 32.0),
        (18, "rank15", 256, 2110560877, True, 16, 32.0),
        (19, "rank6", 256, 1358002110, True, 16, 32.0),
        (20, "rank15", 256, 278777161, True, 16, 32.0),
    )
    for (
        checkpoint_index,
        opponent_rank,
        training_steps,
        seed,
        expert_demo,
        opening_moves,
        opening_weight,
    ) in expert_stages:
        stages.append(
            Task10Stage(
                checkpoint_index=checkpoint_index,
                kind="expert",
                episodes=2,
                training_steps=training_steps,
                seed=seed,
                opponent_rank=opponent_rank,
                expert_demo=expert_demo,
                opening_moves=opening_moves,
                opening_weight=opening_weight,
                final_continuation_gpu_hour_ceiling=(
                    0.15 if checkpoint_index >= 17 else None
                ),
            )
        )
    return tuple(stages)


def task10_plan_payload() -> dict[str, object]:
    stages = task10_stage_plan()
    return {
        "stages": [asdict(stage) for stage in stages],
        "totals": {
            "episodes": sum(stage.episodes for stage in stages),
            "generations": len(stages),
            "optimizer_steps": sum(stage.training_steps for stage in stages),
        },
        "final_continuation_gpu_hour_ceiling": 0.15,
    }


ExpertStageRunner = Callable[[Task10Stage, Path, float | None, str], Path]


def run_task10_workflow(
    *,
    run_dir: str | Path,
    config_path: str | Path,
    stages: Sequence[Task10Stage] | None = None,
    expert_stage_runner: ExpertStageRunner | None = None,
    maximum_stages: int | None = None,
    allocated_gpu_count: int = 1,
) -> dict[str, Any]:
    """Run one lock-protected Task 10 workflow invocation."""
    resolved_run = Path(run_dir).resolve()
    with _workflow_lock(resolved_run):
        return _run_task10_workflow(
            run_dir=resolved_run,
            config_path=config_path,
            stages=stages,
            expert_stage_runner=expert_stage_runner,
            maximum_stages=maximum_stages,
            allocated_gpu_count=allocated_gpu_count,
        )


def _run_task10_workflow(
    *,
    run_dir: str | Path,
    config_path: str | Path,
    stages: Sequence[Task10Stage] | None = None,
    expert_stage_runner: ExpertStageRunner | None = None,
    maximum_stages: int | None = None,
    allocated_gpu_count: int = 1,
) -> dict[str, Any]:
    """Run or resume an immutable stage sequence through real framework calls."""
    from rlbench.cli.main import (
        _load_manifest,
        _validate_checkpoint_lineage,
        main as rlbench_main,
    )
    from rlbench.config import compose_config
    from rlbench.telemetry import Event, EventLedger

    resolved_run = Path(run_dir).resolve()
    resolved_config = Path(config_path).resolve()
    chosen_stages = tuple(task10_stage_plan() if stages is None else stages)
    _validate_stage_sequence(chosen_stages)
    if maximum_stages is not None and maximum_stages < 0:
        raise ValueError("maximum_stages must be non-negative")
    if (
        not isinstance(allocated_gpu_count, int)
        or isinstance(allocated_gpu_count, bool)
        or allocated_gpu_count < 1
    ):
        raise ValueError("allocated_gpu_count must be a positive integer")
    composed = compose_config(
        resolved_config,
        game="snakego",
        algorithm="alphazero",
        output_override=resolved_run,
    )
    allocated_gpu_source = (
        "resource_sampler"
        if bool(composed.canonical["resources"]["sample"])
        else "wall_clock_fallback"
    )
    plan_hash = _stage_plan_hash(chosen_stages)
    state_path = resolved_run / "task10_workflow_state.json"
    state = _load_workflow_state(
        state_path,
        plan_hash=plan_hash,
        allocated_gpu_source=allocated_gpu_source,
    )
    ledger = EventLedger(resolved_run / "events.jsonl")
    attempt_path = resolved_run / "task10_stage_attempt.json"
    checkpoint: Path | None = None
    manifest: Mapping[str, Any] | None = None
    if attempt_path.exists():
        attempt = _load_stage_attempt(attempt_path, plan_hash=plan_hash)
        completed_through = 0 if state is None else int(state["completed_through"])
        commits = [
            event
            for event in ledger.read()
            if event.event_type == "workflow_stage_committed"
            and event.payload.get("attempt_id") == attempt["attempt_id"]
        ]
        if attempt["checkpoint_index"] == completed_through and state is not None:
            if (
                len(commits) != 1
                or commits[0].payload.get("checkpoint_hash")
                != state.get("checkpoint_hash")
                or commits[0].payload.get("plan_hash") != plan_hash
            ):
                raise ValueError("workflow completed attempt is not uniquely committed")
            attempt_path.unlink()
        else:
            planned = resolved_run / str(attempt["planned_checkpoint"])
            saved_target = [
                event
                for event in ledger.read()
                if event.event_type == "checkpoint_saved"
                and event.payload.get("checkpoint_index")
                == attempt["checkpoint_index"]
            ]
            if saved_target or planned.exists():
                state = None
            else:
                _abort_stage_attempt(
                    ledger=ledger,
                    attempt=attempt,
                    run_id=(
                        str(manifest["run_id"])
                        if manifest is not None
                        else "task10-workflow"
                    ),
                )
                attempt_path.unlink()
                if state is not None:
                    state = _state_with_failed_resources(
                        state, ledger=ledger, stages=chosen_stages
                    )
                    _write_workflow_state(state_path, state)
    if state is not None:
        if state.get("allocated_gpu_count") != allocated_gpu_count:
            raise ValueError("workflow state allocated GPU count disagrees")
        if state.get("allocated_gpu_source") != allocated_gpu_source:
            raise ValueError("workflow state allocated GPU source disagrees")
        manifest = _load_manifest(resolved_run / "run_manifest.json")
        checkpoint = (resolved_run / str(state["checkpoint"])).resolve()
        lineage = _validate_checkpoint_lineage(
            resolved_run,
            checkpoint,
            manifest=manifest,
            ledger=ledger,
            require_head=True,
        )
        if lineage.get("checkpoint_hash") != state.get("checkpoint_hash"):
            raise ValueError("workflow state checkpoint hash disagrees with lineage head")
        completed_through = state.get("completed_through")
        if (
            not isinstance(completed_through, int)
            or isinstance(completed_through, bool)
            or not 1 <= completed_through <= len(chosen_stages)
        ):
            raise ValueError("workflow state completed stage is invalid")
        _validate_stage_completion(
            chosen_stages,
            chosen_stages[completed_through - 1],
            lineage,
        )
        learning = lineage["budgets"]["learning"]
        state_learning_hours = state.get("learning_gpu_hours")
        if (
            state.get("episodes") != learning.get("episodes")
            or state.get("optimizer_steps") != learning.get("optimizer_steps")
            or not isinstance(state_learning_hours, (int, float))
            or isinstance(state_learning_hours, bool)
            or not math.isclose(
                float(state_learning_hours),
                _learning_gpu_hours(
                    lineage, allocated_gpu_count=allocated_gpu_count
                ),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("workflow state budgets disagree with checkpoint lineage")
        _validate_continuation_state(
            state=state,
            stages=chosen_stages,
            run_dir=resolved_run,
            manifest=manifest,
            ledger=ledger,
            validate_lineage=_validate_checkpoint_lineage,
            allocated_gpu_count=allocated_gpu_count,
        )
        expected_failed, expected_final_failed = _failed_resource_totals(
            ledger, stages=chosen_stages
        )
        if (
            not math.isclose(
                float(state["failed_attempt_gpu_hours"]),
                expected_failed,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(state["final_continuation_failed_gpu_hours"]),
                expected_final_failed,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("workflow failed-attempt resources disagree")
    elif attempt_path.exists():
        state, checkpoint, manifest = _recover_workflow_state(
            run_dir=resolved_run,
            stages=chosen_stages,
            plan_hash=plan_hash,
            ledger=ledger,
            load_manifest=_load_manifest,
            validate_lineage=_validate_checkpoint_lineage,
            allocated_gpu_count=allocated_gpu_count,
            composed=composed,
            allocated_gpu_source=allocated_gpu_source,
        )
        _write_workflow_state(state_path, state)
        attempt_path.unlink(missing_ok=True)

    completed = 0 if state is None else int(state["completed_through"])
    to_run = chosen_stages[completed:]
    if maximum_stages is not None:
        to_run = to_run[:maximum_stages]
    for stage in to_run:
        attempt_id = str(uuid4())
        _write_stage_attempt(
            attempt_path,
            {
                "schema_version": 2,
                "attempt_id": attempt_id,
                "checkpoint_index": stage.checkpoint_index,
                "plan_hash": plan_hash,
                "stage": asdict(stage),
                "planned_checkpoint": (
                    f"checkpoints/checkpoint_{stage.checkpoint_index:06d}.pt"
                ),
                "prior_checkpoint_hash": (
                    None if state is None else state["checkpoint_hash"]
                ),
                "config_hash": composed.config_hash,
                "manifest_hash": (
                    None if manifest is None else manifest["manifest_hash"]
                ),
                "ledger_event_count": len(list(ledger.read())),
            },
        )
        if stage.kind == "selfplay":
            controls = composed.canonical["training"]
            if (
                int(controls["generations"]) != 1
                or int(controls["self_play_episodes"]) != stage.episodes
                or int(controls["training_steps"]) != stage.training_steps
            ):
                raise ValueError("locked training controls do not match workflow stage")
            arguments = [
                "train",
                "snakego",
                "--algo",
                "alphazero",
                "--config",
                str(resolved_config),
                "--output",
                str(resolved_run),
            ]
            if checkpoint is not None:
                arguments.extend(("--resume", str(checkpoint)))
            rlbench_main(arguments)
            checkpoint = (
                resolved_run
                / "checkpoints"
                / f"checkpoint_{stage.checkpoint_index:06d}.pt"
            )
        else:
            if expert_stage_runner is None or checkpoint is None:
                raise ValueError("expert stages require a validated expert stage runner")
            remaining = _remaining_final_gpu_hours(state, stage)
            try:
                checkpoint = expert_stage_runner(
                    stage, checkpoint, remaining, attempt_id
                ).resolve()
            except TimeoutError:
                _abort_stage_attempt(
                    ledger=ledger,
                    attempt=_load_stage_attempt(attempt_path, plan_hash=plan_hash),
                    run_id=str(manifest["run_id"]),
                )
                state = _state_with_failed_resources(
                    state, ledger=ledger, stages=chosen_stages
                )
                _write_workflow_state(state_path, state)
                attempt_path.unlink()
                raise

        manifest = _load_manifest(resolved_run / "run_manifest.json")
        lineage = _validate_checkpoint_lineage(
            resolved_run,
            checkpoint,
            manifest=manifest,
            ledger=ledger,
            require_head=True,
        )
        _validate_stage_completion(chosen_stages, stage, lineage)
        ledger.append(
            Event(
                "workflow_stage_committed",
                str(manifest["run_id"]),
                stage="learning",
                payload={
                    "attempt_id": attempt_id,
                    "checkpoint_index": stage.checkpoint_index,
                    "checkpoint_hash": lineage["checkpoint_hash"],
                    "plan_hash": plan_hash,
                },
            )
        )
        state = _next_workflow_state(
            plan_hash=plan_hash,
            run_dir=resolved_run,
            stage=stage,
            checkpoint=checkpoint,
            lineage=lineage,
            prior=state,
            allocated_gpu_count=allocated_gpu_count,
            allocated_gpu_source=allocated_gpu_source,
        )
        _write_workflow_state(state_path, state)
        attempt_path.unlink(missing_ok=True)
    if state is None:
        return {
            "schema_version": 1,
            "plan_hash": plan_hash,
            "completed_through": 0,
            "episodes": 0,
            "optimizer_steps": 0,
            "allocated_gpu_count": allocated_gpu_count,
            "allocated_gpu_source": allocated_gpu_source,
            "failed_attempt_gpu_hours": 0.0,
            "final_continuation_failed_gpu_hours": 0.0,
        }
    return dict(state)


def _validate_stage_sequence(stages: Sequence[Task10Stage]) -> None:
    if not stages:
        raise ValueError("workflow requires at least one stage")
    if [stage.checkpoint_index for stage in stages] != list(
        range(1, len(stages) + 1)
    ):
        raise ValueError("workflow checkpoints must be contiguous from one")
    if any(stage.episodes < 1 or stage.training_steps < 1 for stage in stages):
        raise ValueError("workflow stage budgets must be positive")
    bounded = [
        stage for stage in stages if stage.final_continuation_gpu_hour_ceiling is not None
    ]
    if bounded:
        first = bounded[0].checkpoint_index
        ceilings = {stage.final_continuation_gpu_hour_ceiling for stage in bounded}
        if (
            first == 1
            or [stage.checkpoint_index for stage in bounded]
            != list(range(first, len(stages) + 1))
            or len(ceilings) != 1
            or next(iter(ceilings)) <= 0.0
        ):
            raise ValueError("final continuation ceiling must be one positive suffix")


@contextmanager
def _workflow_lock(run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / ".task10_workflow.lock"
    with path.open("a+b") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("task10 workflow is already running") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _recover_workflow_state(
    *,
    run_dir: Path,
    stages: Sequence[Task10Stage],
    plan_hash: str,
    ledger: Any,
    load_manifest: Callable[[Path], Mapping[str, Any]],
    validate_lineage: Callable[..., Mapping[str, Any]],
    allocated_gpu_count: int,
    composed: Any,
    allocated_gpu_source: str,
) -> tuple[dict[str, Any], Path, Mapping[str, Any]]:
    manifest = load_manifest(run_dir / "run_manifest.json")
    attempt_path = run_dir / "task10_stage_attempt.json"
    attempt = _load_stage_attempt(attempt_path, plan_hash=plan_hash)
    saved = [
        event.payload
        for event in ledger.read()
        if event.event_type == "checkpoint_saved"
    ]
    target_index = int(attempt["checkpoint_index"])
    target_saved = [
        payload for payload in saved if payload.get("checkpoint_index") == target_index
    ]
    planned_checkpoint = run_dir / str(attempt["planned_checkpoint"])
    if not target_saved and planned_checkpoint.is_file():
        if attempt["schema_version"] != 2:
            raise ValueError("orphan checkpoint requires a versioned stage journal")
        recovered_payload = _recover_orphan_checkpoint_event(
            run_dir=run_dir,
            checkpoint=planned_checkpoint,
            attempt=attempt,
            stages=stages,
            manifest=manifest,
            ledger=ledger,
            composed=composed,
            allocated_gpu_count=allocated_gpu_count,
        )
        from rlbench.telemetry import Event

        ledger.append(
            Event(
                "checkpoint_saved",
                str(manifest["run_id"]),
                stage="learning",
                payload=recovered_payload,
            )
        )
        saved.append(recovered_payload)
    indices = [payload.get("checkpoint_index") for payload in saved]
    if (
        not indices
        or any(not isinstance(index, int) or isinstance(index, bool) for index in indices)
        or sorted(indices) != list(range(1, max(indices) + 1))
        or len(set(indices)) != len(indices)
        or max(indices) > len(stages)
    ):
        raise ValueError("workflow recovery requires one contiguous checkpoint lineage")
    head_index = max(indices)
    if attempt["checkpoint_index"] != head_index:
        raise ValueError("workflow recovery attempt does not match lineage head")
    head_payload = next(
        payload for payload in saved if payload["checkpoint_index"] == head_index
    )
    checkpoint = (run_dir / str(head_payload["checkpoint"])).resolve()
    lineage = validate_lineage(
        run_dir,
        checkpoint,
        manifest=manifest,
        ledger=ledger,
        require_head=True,
    )
    stage = stages[head_index - 1]
    _validate_stage_completion(stages, stage, lineage)
    commits = [
        event
        for event in ledger.read()
        if event.event_type == "workflow_stage_committed"
        and event.payload.get("attempt_id") == attempt["attempt_id"]
    ]
    if len(commits) > 1:
        raise ValueError("workflow recovery attempt has competing commits")
    if commits and (
        commits[0].payload.get("checkpoint_index") != head_index
        or commits[0].payload.get("checkpoint_hash") != lineage["checkpoint_hash"]
        or commits[0].payload.get("plan_hash") != plan_hash
    ):
        raise ValueError("workflow recovery commit disagrees with lineage head")
    if not commits:
        from rlbench.telemetry import Event

        ledger.append(
            Event(
                "workflow_stage_committed",
                str(manifest["run_id"]),
                stage="learning",
                payload={
                    "attempt_id": attempt["attempt_id"],
                    "checkpoint_index": head_index,
                    "checkpoint_hash": lineage["checkpoint_hash"],
                    "plan_hash": plan_hash,
                    "recovered": True,
                },
            )
        )
    start = _continuation_baseline_hours(
        completed_through=head_index,
        stages=stages,
        run_dir=run_dir,
        manifest=manifest,
        ledger=ledger,
        validate_lineage=validate_lineage,
        allocated_gpu_count=allocated_gpu_count,
    )
    learning = lineage["budgets"]["learning"]
    state = {
        "schema_version": 2,
        "plan_hash": plan_hash,
        "completed_through": head_index,
        "checkpoint": str(checkpoint.relative_to(run_dir)),
        "checkpoint_hash": lineage["checkpoint_hash"],
        "episodes": learning["episodes"],
        "optimizer_steps": learning["optimizer_steps"],
        "learning_gpu_hours": _learning_gpu_hours(
            lineage, allocated_gpu_count=allocated_gpu_count
        ),
        "allocated_gpu_count": allocated_gpu_count,
        "allocated_gpu_source": allocated_gpu_source,
        "final_continuation_start_gpu_hours": start,
    }
    state = _state_with_failed_resources(state, ledger=ledger, stages=stages)
    return state, checkpoint, manifest


def _recover_orphan_checkpoint_event(
    *,
    run_dir: Path,
    checkpoint: Path,
    attempt: Mapping[str, Any],
    stages: Sequence[Task10Stage],
    manifest: Mapping[str, Any],
    ledger: Any,
    composed: Any,
    allocated_gpu_count: int,
) -> dict[str, Any]:
    from rlbench.cli.main import _build_trainer, _sha256_file

    index = int(attempt["checkpoint_index"])
    stage = stages[index - 1]
    expected_relative = f"checkpoints/checkpoint_{index:06d}.pt"
    if (
        attempt.get("stage") != asdict(stage)
        or attempt.get("planned_checkpoint") != expected_relative
        or attempt.get("config_hash") != composed.config_hash
        or manifest.get("config_hash") != composed.config_hash
        or attempt.get("manifest_hash") not in {None, manifest.get("manifest_hash")}
        or checkpoint.resolve() != (run_dir / expected_relative).resolve()
    ):
        raise ValueError("orphan checkpoint journal disagrees with immutable run facts")
    prior_saved = [
        event.payload
        for event in ledger.read()
        if event.event_type == "checkpoint_saved"
        and event.payload.get("manifest_hash") == manifest["manifest_hash"]
    ]
    prior_hash = None if not prior_saved else prior_saved[-1].get("checkpoint_hash")
    if (
        attempt.get("prior_checkpoint_hash") != prior_hash
        or len(prior_saved) != index - 1
        or [payload.get("checkpoint_index") for payload in prior_saved]
        != list(range(1, index))
    ):
        raise ValueError("orphan checkpoint prior lineage disagrees")
    trainer, counter_name = _build_trainer(
        composed, ledger=ledger, run_id=str(manifest["run_id"])
    )
    trainer.load_checkpoint(checkpoint)
    if counter_name != "generation" or trainer.generation != index:
        raise ValueError("orphan checkpoint generation disagrees")
    budgets = trainer.budgets.as_dict()
    learning = budgets["learning"]
    expected_stages = stages[:index]
    if (
        learning["episodes"] != sum(item.episodes for item in expected_stages)
        or learning["optimizer_steps"]
        != sum(item.training_steps for item in expected_stages)
    ):
        raise ValueError("orphan checkpoint budgets disagree with stage plan")
    learning_gpu_hours = (
        float(learning["wall_seconds"]) * allocated_gpu_count / 3600.0
    )
    evaluation_gpu_hours = (
        float(budgets["evaluation"]["wall_seconds"])
        * allocated_gpu_count
        / 3600.0
    )
    return {
        "algorithm": "alphazero",
        "checkpoint_index": index,
        "generation": index,
        "checkpoint": expected_relative,
        "checkpoint_hash": _sha256_file(checkpoint),
        "input_checkpoint_hash": prior_hash,
        "manifest_hash": manifest["manifest_hash"],
        "config_hash": manifest["config_hash"],
        "env_steps": budgets["total"]["env_steps"],
        "optimizer_steps": budgets["total"]["optimizer_steps"],
        "mcts_simulations": budgets["total"]["mcts_simulations"],
        "learning_wall_seconds": learning["wall_seconds"],
        "evaluation_wall_seconds": budgets["evaluation"]["wall_seconds"],
        "wall_seconds": budgets["total"]["wall_seconds"],
        "learning_gpu_hours": learning_gpu_hours,
        "evaluation_gpu_hours": evaluation_gpu_hours,
        "gpu_hours": learning_gpu_hours + evaluation_gpu_hours,
        "budgets": budgets,
        "allocated_gpu_count": allocated_gpu_count,
        "allocation_source": "wall_clock_fallback",
        "attempt_id": attempt["attempt_id"],
        "recovered": True,
    }


def _stage_plan_hash(stages: Sequence[Task10Stage]) -> str:
    encoded = json.dumps(
        [asdict(stage) for stage in stages],
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _load_workflow_state(
    path: Path, *, plan_hash: str, allocated_gpu_source: str
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid task10 workflow state") from exc
    version_one = {
        "schema_version",
        "plan_hash",
        "completed_through",
        "checkpoint",
        "checkpoint_hash",
        "episodes",
        "optimizer_steps",
        "learning_gpu_hours",
        "final_continuation_start_gpu_hours",
        "allocated_gpu_count",
    }
    version_two = version_one | {
        "allocated_gpu_source",
        "failed_attempt_gpu_hours",
        "final_continuation_failed_gpu_hours",
    }
    if not isinstance(state, dict):
        raise ValueError("invalid task10 workflow state schema")
    if state.get("schema_version") == 1 and set(state) == version_one:
        state = {
            **state,
            "schema_version": 2,
            "allocated_gpu_source": allocated_gpu_source,
            "failed_attempt_gpu_hours": 0.0,
            "final_continuation_failed_gpu_hours": 0.0,
        }
    elif state.get("schema_version") != 2 or set(state) != version_two:
        raise ValueError("invalid task10 workflow state schema")
    if state["plan_hash"] != plan_hash:
        raise ValueError("task10 workflow state does not match the locked stage plan")
    return state


def _validate_stage_completion(
    stages: Sequence[Task10Stage],
    stage: Task10Stage,
    lineage: Mapping[str, Any],
) -> None:
    expected = stages[: stage.checkpoint_index]
    budgets = lineage.get("budgets")
    if not isinstance(budgets, Mapping) or not isinstance(
        budgets.get("learning"), Mapping
    ):
        raise ValueError("checkpoint lineage is missing learning budgets")
    learning = budgets["learning"]
    if (
        lineage.get("checkpoint_index") != stage.checkpoint_index
        or learning.get("episodes") != sum(item.episodes for item in expected)
        or learning.get("optimizer_steps")
        != sum(item.training_steps for item in expected)
    ):
        raise ValueError("checkpoint budgets do not satisfy the completed stage plan")


def _learning_gpu_hours(
    lineage: Mapping[str, Any], *, allocated_gpu_count: int
) -> float:
    value = lineage.get("learning_gpu_hours")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    budgets = lineage.get("budgets")
    learning = budgets.get("learning") if isinstance(budgets, Mapping) else None
    wall = learning.get("wall_seconds") if isinstance(learning, Mapping) else None
    if isinstance(wall, (int, float)) and not isinstance(wall, bool):
        return float(wall) * allocated_gpu_count / 3600.0
    raise ValueError("checkpoint lineage has no allocated learning time")


def _remaining_final_gpu_hours(
    state: Mapping[str, Any] | None, stage: Task10Stage
) -> float | None:
    ceiling = stage.final_continuation_gpu_hour_ceiling
    if ceiling is None:
        return None
    if state is None:
        raise ValueError("final continuation requires a prior checkpoint")
    start = state.get("final_continuation_start_gpu_hours")
    if start is None:
        used = float(state["final_continuation_failed_gpu_hours"])
    else:
        used = (
            float(state["learning_gpu_hours"])
            - float(start)
            + float(state["final_continuation_failed_gpu_hours"])
        )
    remaining = ceiling - used
    if remaining <= 0.0:
        raise RuntimeError("final continuation GPU-hour ceiling is exhausted")
    return remaining


def _validate_continuation_state(
    *,
    state: Mapping[str, Any],
    stages: Sequence[Task10Stage],
    run_dir: Path,
    manifest: Mapping[str, Any],
    ledger: Any,
    validate_lineage: Callable[..., Mapping[str, Any]],
    allocated_gpu_count: int,
) -> None:
    stored = state.get("final_continuation_start_gpu_hours")
    expected = _continuation_baseline_hours(
        completed_through=int(state["completed_through"]),
        stages=stages,
        run_dir=run_dir,
        manifest=manifest,
        ledger=ledger,
        validate_lineage=validate_lineage,
        allocated_gpu_count=allocated_gpu_count,
    )
    if expected is None:
        if stored is None:
            return
        raise ValueError("workflow continuation baseline is invalid")
    if (
        not isinstance(stored, (int, float))
        or isinstance(stored, bool)
        or not math.isclose(
            float(stored), expected, rel_tol=0.0, abs_tol=1e-12
        )
    ):
        raise ValueError("workflow continuation baseline is invalid")


def _continuation_baseline_hours(
    *,
    completed_through: int,
    stages: Sequence[Task10Stage],
    run_dir: Path,
    manifest: Mapping[str, Any],
    ledger: Any,
    validate_lineage: Callable[..., Mapping[str, Any]],
    allocated_gpu_count: int,
) -> float | None:
    bounded = [
        stage for stage in stages if stage.final_continuation_gpu_hour_ceiling is not None
    ]
    if not bounded or completed_through < bounded[0].checkpoint_index:
        return None
    baseline_index = bounded[0].checkpoint_index - 1
    baseline = run_dir / "checkpoints" / f"checkpoint_{baseline_index:06d}.pt"
    baseline_lineage = validate_lineage(
        run_dir, baseline, manifest=manifest, ledger=ledger
    )
    return _learning_gpu_hours(
        baseline_lineage, allocated_gpu_count=allocated_gpu_count
    )


def _next_workflow_state(
    *,
    plan_hash: str,
    run_dir: Path,
    stage: Task10Stage,
    checkpoint: Path,
    lineage: Mapping[str, Any],
    prior: Mapping[str, Any] | None,
    allocated_gpu_count: int,
    allocated_gpu_source: str,
) -> dict[str, Any]:
    learning_gpu_hours = _learning_gpu_hours(
        lineage, allocated_gpu_count=allocated_gpu_count
    )
    start = None if prior is None else prior.get("final_continuation_start_gpu_hours")
    if stage.final_continuation_gpu_hour_ceiling is not None and start is None:
        if prior is None:
            raise ValueError("final continuation requires a prior checkpoint")
        start = float(prior["learning_gpu_hours"])
    if start is not None and stage.final_continuation_gpu_hour_ceiling is not None:
        used = (
            learning_gpu_hours
            - float(start)
            + float(
                0.0
                if prior is None
                else prior["final_continuation_failed_gpu_hours"]
            )
        )
        if used > stage.final_continuation_gpu_hour_ceiling:
            raise RuntimeError("final continuation exceeded its GPU-hour ceiling")
    budgets = lineage["budgets"]["learning"]
    return {
        "schema_version": 2,
        "plan_hash": plan_hash,
        "completed_through": stage.checkpoint_index,
        "checkpoint": str(checkpoint.relative_to(run_dir)),
        "checkpoint_hash": lineage["checkpoint_hash"],
        "episodes": budgets["episodes"],
        "optimizer_steps": budgets["optimizer_steps"],
        "learning_gpu_hours": learning_gpu_hours,
        "allocated_gpu_count": allocated_gpu_count,
        "allocated_gpu_source": allocated_gpu_source,
        "final_continuation_start_gpu_hours": start,
        "failed_attempt_gpu_hours": (
            0.0 if prior is None else float(prior["failed_attempt_gpu_hours"])
        ),
        "final_continuation_failed_gpu_hours": (
            0.0
            if prior is None
            else float(prior["final_continuation_failed_gpu_hours"])
        ),
    }


def _failed_resource_totals(
    ledger: Any, *, stages: Sequence[Task10Stage]
) -> tuple[float, float]:
    events = list(ledger.read())
    aborted_ids = {
        str(event.payload["attempt_id"])
        for event in events
        if event.event_type == "workflow_stage_attempt_aborted"
        and isinstance(event.payload.get("attempt_id"), str)
    }
    resources: dict[str, list[Mapping[str, Any]]] = {}
    for event in events:
        attempt_id = event.payload.get("attempt_id")
        if event.event_type != "workflow_stage_resource" or attempt_id not in aborted_ids:
            continue
        resources.setdefault(str(attempt_id), []).append(event.payload)
    first_bounded = next(
        (
            stage.checkpoint_index
            for stage in stages
            if stage.final_continuation_gpu_hour_ceiling is not None
        ),
        len(stages) + 1,
    )
    total = 0.0
    final_total = 0.0
    for attempt_id, facts in resources.items():
        if len(facts) != 1:
            raise ValueError(
                f"workflow attempt {attempt_id} has duplicate resource facts"
            )
        fact = facts[0]
        hours = fact.get("allocated_gpu_hours")
        count = fact.get("allocated_gpu_count")
        source = fact.get("allocation_source")
        index = fact.get("checkpoint_index")
        if (
            not isinstance(hours, (int, float))
            or isinstance(hours, bool)
            or not math.isfinite(float(hours))
            or float(hours) < 0.0
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count < 1
            or source not in {"resource_sampler", "wall_clock_fallback"}
            or not isinstance(index, int)
            or isinstance(index, bool)
        ):
            raise ValueError("workflow failed-attempt resource fact is invalid")
        total += float(hours)
        if index >= first_bounded:
            final_total += float(hours)
    return total, final_total


def _state_with_failed_resources(
    state: Mapping[str, Any] | None,
    *,
    ledger: Any,
    stages: Sequence[Task10Stage],
) -> dict[str, Any]:
    if state is None:
        raise ValueError("failed expert attempt requires a prior workflow state")
    failed, final_failed = _failed_resource_totals(ledger, stages=stages)
    updated = dict(state)
    updated["failed_attempt_gpu_hours"] = failed
    updated["final_continuation_failed_gpu_hours"] = final_failed
    return updated


def _write_workflow_state(path: Path, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.pending")
    payload = json.dumps(state, allow_nan=False, indent=2, sort_keys=True) + "\n"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_stage_attempt(path: Path, *, plan_hash: str) -> dict[str, Any]:
    try:
        attempt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("workflow stage attempt is invalid") from exc
    common = {"schema_version", "attempt_id", "checkpoint_index", "plan_hash"}
    version_two = common | {
        "stage",
        "planned_checkpoint",
        "prior_checkpoint_hash",
        "config_hash",
        "manifest_hash",
        "ledger_event_count",
    }
    if not isinstance(attempt, dict) or attempt.get("plan_hash") != plan_hash:
        raise ValueError("workflow stage attempt is invalid")
    if attempt.get("schema_version") == 1 and set(attempt) == common:
        index = attempt.get("checkpoint_index")
        return {
            **attempt,
            "stage": None,
            "planned_checkpoint": f"checkpoints/checkpoint_{int(index):06d}.pt",
            "prior_checkpoint_hash": None,
            "config_hash": None,
            "manifest_hash": None,
            "ledger_event_count": 0,
        }
    if (
        attempt.get("schema_version") != 2
        or set(attempt) != version_two
        or not isinstance(attempt.get("attempt_id"), str)
        or not attempt["attempt_id"]
        or not isinstance(attempt.get("checkpoint_index"), int)
        or isinstance(attempt["checkpoint_index"], bool)
        or attempt["checkpoint_index"] < 1
        or not isinstance(attempt.get("ledger_event_count"), int)
        or isinstance(attempt["ledger_event_count"], bool)
        or attempt["ledger_event_count"] < 0
    ):
        raise ValueError("workflow stage attempt is invalid")
    return attempt


def _abort_stage_attempt(*, ledger: Any, attempt: Mapping[str, Any], run_id: str) -> None:
    from rlbench.telemetry import Event

    prior_events = list(ledger.read())
    existing = [
        event
        for event in prior_events
        if event.event_type == "workflow_stage_attempt_aborted"
        and event.payload.get("attempt_id") == attempt["attempt_id"]
    ]
    if len(existing) > 1:
        raise ValueError("workflow stage attempt has duplicate abort facts")
    if existing:
        return
    boundary = int(attempt["ledger_event_count"])
    if boundary > len(prior_events):
        raise ValueError("workflow stage attempt ledger boundary is invalid")
    partial = prior_events[boundary:]
    ledger.append(
        Event(
            "workflow_stage_attempt_aborted",
            run_id,
            stage="learning",
            payload={
                "attempt_id": attempt["attempt_id"],
                "checkpoint_index": attempt["checkpoint_index"],
                "plan_hash": attempt["plan_hash"],
                "partial_event_count": len(partial),
                "partial_event_ids": [event.event_id for event in partial],
                "recovered": True,
            },
        )
    )


def _write_stage_attempt(path: Path, attempt: Mapping[str, Any]) -> None:
    if path.exists():
        raise RuntimeError("task10 workflow has an unresolved stage attempt")
    temporary = path.with_suffix(".json.pending")
    payload = json.dumps(attempt, allow_nan=False, indent=2, sort_keys=True) + "\n"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
