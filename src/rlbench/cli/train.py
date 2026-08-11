"""``rlbench train`` subcommand orchestration."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

from rlbench.config import compose_config
from rlbench.population import PopulationManifest
from rlbench.telemetry import Event, EventLedger, ResourceSampler

from ._facts import (
    create_manifest,
    latest_gpu_hours,
    load_manifest,
    print_json,
    restore_event_backed_budgets,
    save_checkpoint_exclusive,
    sha256_file,
    sum_optional_hours,
    validate_checkpoint_lineage,
    validate_resume_manifest,
    write_immutable_manifest,
)
from ._policies import _population_policy, _TrainingRandomPolicy
from .main import _build_trainer, _run_training


def _train_command(
    game_name: str,
    *,
    algorithm: str,
    config_path: str | Path,
    output: str | Path | None,
    resume: str | Path | None,
    initialize: str | Path | None = None,
    population: str | Path | None = None,
    opponent_id: str | None = None,
    local_opponent: str | None = None,
) -> int:
    composed = compose_config(
        config_path,
        game=game_name,
        algorithm=algorithm,
        output_override=output,
        caller_directory=Path.cwd(),
    )
    run_dir = composed.output_dir
    training_opponent, training_inputs = _resolve_training_inputs(
        game_name,
        algorithm=algorithm,
        population=population,
        opponent_id=opponent_id,
        local_opponent=local_opponent,
        training_seed=int(composed.canonical["training"]["seed"]),
        initialize=initialize,
        resume=resume,
    )
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        if resume is None:
            raise FileExistsError("run manifest already exists; use --resume")
        manifest = load_manifest(manifest_path)
        validate_resume_manifest(manifest, composed)
    else:
        if resume is not None:
            raise ValueError("resume requires the original run directory and manifest")
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = create_manifest(composed)
        write_immutable_manifest(manifest_path, manifest)

    run_id = str(manifest["run_id"])
    ledger = EventLedger(run_dir / "events.jsonl")
    if resume is not None:
        _validate_training_inputs(ledger, training_inputs)
    resume_lineage: Mapping[str, Any] | None = None
    if resume is not None:
        resume_lineage = validate_checkpoint_lineage(
            run_dir,
            Path(resume).resolve(),
            manifest=manifest,
            ledger=ledger,
            require_head=True,
        )
    if resume is None:
        ledger.append(
            Event(
                event_type="run_started",
                run_id=run_id,
                payload={
                    "algorithm": algorithm,
                    "game": game_name,
                    "candidate_id": "learner",
                    "config_hash": composed.config_hash,
                    "manifest_hash": manifest["manifest_hash"],
                    "training_inputs": training_inputs,
                },
            )
        )
    else:
        ledger.append(
            Event(
                event_type="run_resumed",
                run_id=run_id,
                stage="learning",
                payload={
                    "checkpoint_hash": resume_lineage["checkpoint_hash"],
                    "checkpoint_index": resume_lineage["checkpoint_index"],
                },
            )
        )

    sampler = ResourceSampler(
        run_id=run_id, host_id=f"learning-{uuid4()}", ledger=ledger
    )
    sample_resources = bool(composed.canonical["resources"]["sample"])
    prior_gpu_hours = latest_gpu_hours(ledger, run_id=run_id)
    started = time.monotonic()
    try:
        if sample_resources:
            sampler.sample("learning")
        trainer, counter_name = _build_trainer(
            composed,
            ledger=ledger,
            run_id=run_id,
            opponent=training_opponent,
            opponent_id=(
                str(training_inputs["opponent"]["agent_id"])
                if training_inputs["opponent"] is not None
                else "opponent"
            ),
        )
        if resume is not None:
            trainer.load_checkpoint(Path(resume).resolve())
            restore_event_backed_budgets(trainer, ledger=ledger, run_id=run_id)
        elif initialize is not None:
            trainer.initialize_model(Path(initialize).resolve())
        total_batches = int(
            composed.canonical["training"][
                "generations" if algorithm == "alphazero" else "iterations"
            ]
        )
        checkpoint_every = int(composed.canonical["training"]["checkpoint_every"])
        last_accounted = started
        checkpoint_path: Path | None = None
        checkpoint_hash: str | None = None
        counter = int(getattr(trainer, counter_name))
        for batch_index, counter in enumerate(_run_training(trainer, composed), 1):
            current = time.monotonic()
            trainer.budgets.add_wall_seconds(current - last_accounted, "learning")
            last_accounted = current
            if counter % checkpoint_every and batch_index != total_batches:
                continue
            checkpoint_path, checkpoint_hash = _record_training_checkpoint(
                trainer=trainer,
                algorithm=algorithm,
                counter_name=counter_name,
                counter=counter,
                run_dir=run_dir,
                run_id=run_id,
                manifest=manifest,
                ledger=ledger,
                sampler=sampler,
                sample_resources=sample_resources,
                prior_gpu_hours=prior_gpu_hours,
            )
        elapsed = time.monotonic() - started
        if checkpoint_path is None or checkpoint_hash is None:
            raise RuntimeError("training produced no checkpoint")
        ledger.append(
            Event(
                event_type="run_finished",
                run_id=run_id,
                payload={
                    "status": "completed",
                    "checkpoint_index": counter,
                    "wall_seconds": elapsed,
                },
            )
        )
    except BaseException as exc:
        ledger.append(
            Event(
                event_type="run_finished",
                run_id=run_id,
                payload={"status": "failed", "error_type": type(exc).__name__},
            )
        )
        raise
    finally:
        close_opponent = getattr(training_opponent, "close", None)
        if callable(close_opponent):
            close_opponent()
        sampler.close()

    print_json(
        {
            "algorithm": algorithm,
            "checkpoint": str(checkpoint_path),
            "checkpoint_hash": checkpoint_hash,
            "run_directory": str(run_dir),
            "run_id": run_id,
        }
    )
    return 0


def _record_training_checkpoint(
    *,
    trainer: Any,
    algorithm: str,
    counter_name: str,
    counter: int,
    run_dir: Path,
    run_id: str,
    manifest: Mapping[str, Any],
    ledger: EventLedger,
    sampler: ResourceSampler,
    sample_resources: bool,
    prior_gpu_hours: Mapping[str, float | None],
) -> tuple[Path, str]:
    if sample_resources:
        sampler.sample("learning")
    totals = sampler.totals()
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoints / f"checkpoint_{counter:06d}.pt"
    save_checkpoint_exclusive(trainer, checkpoint_path)
    ledger.append_budget_snapshot(run_id=run_id, counters=trainer.budgets)
    budgets = trainer.budgets.as_dict()
    invocation_learning_gpu_hours = (
        totals["learning"].allocated_gpu_hours if sample_resources else None
    )
    learning_gpu_hours = sum_optional_hours(
        prior_gpu_hours["learning"], invocation_learning_gpu_hours
    )
    evaluation_gpu_hours = prior_gpu_hours["evaluation"]
    total_gpu_hours = sum_optional_hours(learning_gpu_hours, evaluation_gpu_hours)
    checkpoint_hash = sha256_file(checkpoint_path)
    ledger.append(
        Event(
            event_type="checkpoint_saved",
            run_id=run_id,
            stage="learning",
            payload={
                "algorithm": algorithm,
                "checkpoint_index": counter,
                counter_name: counter,
                "checkpoint": str(checkpoint_path.relative_to(run_dir)),
                "checkpoint_hash": checkpoint_hash,
                "manifest_hash": manifest["manifest_hash"],
                "env_steps": budgets["total"]["env_steps"],
                "optimizer_steps": budgets["total"]["optimizer_steps"],
                "mcts_simulations": budgets["total"]["mcts_simulations"],
                "learning_wall_seconds": budgets["learning"]["wall_seconds"],
                "evaluation_wall_seconds": budgets["evaluation"]["wall_seconds"],
                "wall_seconds": budgets["total"]["wall_seconds"],
                "learning_gpu_hours": learning_gpu_hours,
                "evaluation_gpu_hours": evaluation_gpu_hours,
                "gpu_hours": total_gpu_hours,
                "budgets": budgets,
            },
        )
    )
    return checkpoint_path, checkpoint_hash


def _resolve_training_inputs(
    game_name: str,
    *,
    algorithm: str,
    population: str | Path | None,
    opponent_id: str | None,
    initialize: str | Path | None,
    resume: str | Path | None,
    local_opponent: str | None = None,
    training_seed: int = 0,
) -> tuple[Any, dict[str, Any]]:
    if (population is None) != (opponent_id is None):
        raise ValueError(
            "training population and opponent-id must be provided together"
        )
    if local_opponent is not None and population is not None:
        raise ValueError("built-in and population opponents are mutually exclusive")
    if local_opponent not in (None, "random"):
        raise ValueError(f"unsupported local training opponent: {local_opponent}")
    if algorithm != "ppo" and any(
        value is not None
        for value in (population, opponent_id, local_opponent, initialize)
    ):
        raise ValueError("process-opponent and initialization controls require PPO")
    if initialize is not None and resume is not None:
        raise ValueError("initialize and resume are mutually exclusive")

    initial_record = None
    if initialize is not None:
        initial_path = Path(initialize).resolve()
        initial_record = {
            "checkpoint_hash": sha256_file(initial_path),
            "checkpoint_name": initial_path.name,
        }

    opponent = None
    opponent_record = None
    if local_opponent == "random":
        opponent = _TrainingRandomPolicy(training_seed)
        opponent_record = {
            "agent_id": "random",
            "agent_hash": "builtin:random-v1",
            "agent_kind": "builtin",
            "population_hash": None,
            "protocol": "local",
        }
    elif population is not None:
        manifest = PopulationManifest.from_yaml(population)
        assert opponent_id is not None
        entry = manifest.entry(opponent_id)
        if entry.kind == "test_human":
            raise ValueError("test_human opponents cannot influence training")
        opponent = _population_policy(
            entry, manifest.population_root, game_name=game_name
        )
        opponent_record = {
            "agent_id": entry.agent_id,
            "agent_hash": entry.content_hash,
            "agent_kind": entry.kind,
            "population_hash": manifest.content_hash,
            "protocol": entry.protocol,
        }
    return opponent, {
        "initial_checkpoint": initial_record,
        "opponent": opponent_record,
    }


def _validate_training_inputs(
    ledger: EventLedger, current: Mapping[str, Any]
) -> None:
    started = next(
        (event for event in ledger.read() if event.event_type == "run_started"),
        None,
    )
    if started is None:
        raise ValueError("resume run has no run_started event")
    recorded = started.payload.get("training_inputs")
    recorded_opponent = (
        recorded.get("opponent") if isinstance(recorded, Mapping) else None
    )
    if recorded_opponent != current.get("opponent"):
        raise ValueError("resume training opponent does not match the original run")
