#!/usr/bin/env python3
"""Hardware-neutral entry point for the locked Task 10 SnakeGo experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4


# src/games/snakego/scripts/<file> -> repository root is four levels up.
ROOT = Path(__file__).resolve().parents[4]
AZ_CONFIG = ROOT / "configs/experiments/snakego_task10_alphazero_locked.yaml"
PPO_CONFIG = ROOT / "configs/experiments/snakego_task10_ppo_locked.yaml"
EXPERT_CONTROLS = {
    "opening_moves": 16,
    "opening_weight": 32.0,
    "self_play_episodes": 2,
    "training_steps": 256,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan")

    workflow = commands.add_parser("workflow")
    workflow.add_argument("--output", type=Path, required=True)
    workflow.add_argument("--config", type=Path, default=AZ_CONFIG)
    workflow.add_argument("--population", type=Path, required=True)
    workflow.add_argument("--rank5", required=True)
    workflow.add_argument("--rank6", required=True)
    workflow.add_argument("--rank15", required=True)
    workflow.add_argument("--device", default="cuda")
    workflow.add_argument("--allocated-gpus", type=int, default=1)
    workflow.add_argument("--move-seconds", type=float, default=0.5)
    workflow.add_argument("--maximum-stages", type=int)

    train_az = commands.add_parser("train-alphazero")
    train_az.add_argument("--output", type=Path, required=True)
    train_az.add_argument("--resume", type=Path)

    train_ppo = commands.add_parser("train-ppo")
    train_ppo.add_argument("--output", type=Path, required=True)

    expert = commands.add_parser("expert-generation")
    expert.add_argument("--run-dir", type=Path, required=True)
    expert.add_argument("--checkpoint", type=Path, required=True)
    expert.add_argument("--population", type=Path, required=True)
    expert.add_argument("--opponent", required=True)
    expert.add_argument("--seed", type=int, required=True)
    expert.add_argument("--device", default="cuda")
    expert.add_argument("--allocated-gpus", type=int, default=1)
    expert.add_argument("--move-seconds", type=float, default=0.5)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--population", type=Path, required=True)
    evaluate.add_argument("--seeds", default="101")

    promotion = commands.add_parser("promotion")
    promotion.add_argument("--facts", type=Path, required=True)

    sweep = commands.add_parser("sweep")
    sweep.add_argument("--output", type=Path, required=True)
    sweep.add_argument("--device", default="cpu")
    sweep.add_argument("--dry-run", action="store_true")
    return parser


def _print(value: Any) -> None:
    print(json.dumps(value, allow_nan=False, sort_keys=True))


def _plan() -> dict[str, Any]:
    from games.snakego.experiments import task10_plan_payload

    return {
        "alphazero_config": str(AZ_CONFIG.relative_to(ROOT)),
        "ppo_config": str(PPO_CONFIG.relative_to(ROOT)),
        "expert_demo": EXPERT_CONTROLS,
        "bounded_final_opponent_order": ["rank6", "rank15"],
        "heldout_policy": "frozen evaluation only; never used for tuning or selection",
        **task10_plan_payload(),
    }


def _train(algorithm: str, config: Path, output: Path, resume: Path | None) -> int:
    from rlbench.cli.main import main as rlbench_main

    arguments = [
        "train",
        "snakego",
        "--algo",
        algorithm,
        "--config",
        str(config),
        "--output",
        str(output),
    ]
    if resume is not None:
        arguments.extend(("--resume", str(resume)))
    return rlbench_main(arguments)


def _evaluate(checkpoint: Path, population: Path, seeds: str) -> int:
    from rlbench.cli.main import main as rlbench_main

    return rlbench_main(
        [
            "evaluate",
            "snakego",
            "--checkpoint",
            str(checkpoint),
            "--population",
            str(population),
            "--seeds",
            seeds,
        ]
    )


def _expert_generation(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "workflow_lock_held", False):
        return _expert_generation_unlocked(args)
    from games.snakego.experiments.snakego_task10 import _workflow_lock

    with _workflow_lock(args.run_dir.resolve()):
        return _expert_generation_unlocked(args)


def _expert_generation_unlocked(args: argparse.Namespace) -> dict[str, Any]:
    from rlbench.algorithms.alphazero import (
        AlphaZeroConfig,
        AlphaZeroTrainer,
        PolicyValueNet,
    )
    from rlbench.cli.main import (
        _load_manifest,
        _save_checkpoint_exclusive,
        _validate_checkpoint_lineage,
        game_factory,
    )
    from rlbench.league import LeagueMember, LeagueState
    from rlbench.population import PopulationManifest, SnakeGoProcessPolicy
    from rlbench.telemetry import Event, EventLedger

    run_dir = args.run_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    manifest = _load_manifest(run_dir / "run_manifest.json")
    ledger = EventLedger(run_dir / "events.jsonl")
    input_lineage = _validate_checkpoint_lineage(
        run_dir,
        checkpoint,
        manifest=manifest,
        ledger=ledger,
        require_head=True,
    )
    self_play_episodes = int(
        getattr(args, "self_play_episodes", EXPERT_CONTROLS["self_play_episodes"])
    )
    training_steps = int(
        getattr(args, "training_steps", EXPERT_CONTROLS["training_steps"])
    )
    expert_demo = bool(getattr(args, "expert_demo", True))
    opening_moves = int(
        getattr(args, "opening_moves", EXPERT_CONTROLS["opening_moves"])
    )
    opening_weight = float(
        getattr(args, "opening_weight", EXPERT_CONTROLS["opening_weight"])
    )
    gpu_hour_ceiling = getattr(args, "gpu_hour_ceiling", None)
    if gpu_hour_ceiling is not None and float(gpu_hour_ceiling) <= 0.0:
        raise ValueError("expert generation GPU-hour ceiling must be positive")
    allocated_gpus = int(getattr(args, "allocated_gpus", 1))
    if allocated_gpus < 1:
        raise ValueError("expert generation requires a positive allocated GPU count")
    canonical = manifest["canonical_config"]
    settings = dict(canonical["algorithm"])
    settings["device"] = args.device
    config = AlphaZeroConfig(**settings)
    factory = game_factory("snakego", dict(canonical["game"]))
    population = PopulationManifest.from_yaml(args.population)
    entry = population.entry(args.opponent)
    if entry.kind != "train_human":
        raise ValueError("expert generation requires a train_human opponent")
    member = LeagueMember(entry.agent_id, entry.content_hash, "train_human")
    league = LeagueState(
        anchor_id=entry.agent_id,
        champion_id=entry.agent_id,
        members=(member,),
    )
    started = time.monotonic()
    deadline = (
        None
        if gpu_hour_ceiling is None
        else started + float(gpu_hour_ceiling) * 3600.0 / allocated_gpus
    )
    network = PolicyValueNet.from_game_spec(factory().spec, config, device=args.device)
    trainer = AlphaZeroTrainer(
        network,
        config,
        seed=args.seed,
        ledger=ledger,
        run_id=str(manifest["run_id"]),
        league=league,
    )
    trainer.load_checkpoint(checkpoint)
    trainer.reseed_stage(args.seed)
    process = SnakeGoProcessPolicy(entry, population.population_root)
    attempt_id = str(getattr(args, "attempt_id", None) or uuid4())
    metrics = None
    failure_type: str | None = None
    try:
        try:
            metrics = trainer.run_generation(
                factory,
                self_play_episodes=self_play_episodes,
                training_steps=training_steps,
                processes=1,
                training_opponents={entry.agent_id: process},
                opponent_episodes=self_play_episodes,
                opponent_move_seconds=args.move_seconds,
                expert_demo=expert_demo,
                expert_demo_opening_moves=opening_moves,
                expert_demo_opening_weight=opening_weight,
                deadline_monotonic=deadline,
            )
        except BaseException as exc:
            failure_type = type(exc).__name__
            raise
    finally:
        try:
            process.close()
        except BaseException as exc:
            if failure_type is None:
                failure_type = type(exc).__name__
            raise
        finally:
            elapsed = time.monotonic() - started
            invocation_gpu_hours = elapsed * allocated_gpus / 3600.0
            trainer.budgets.add_wall_seconds(elapsed, "learning")
            trainer.budgets.add_resources(
                gpu_hours=invocation_gpu_hours,
                cpu_core_hours=0.0,
                stage="learning",
            )
            ledger.append(
                Event(
                    "workflow_stage_resource",
                    trainer.run_id,
                    stage="learning",
                    payload={
                        "attempt_id": attempt_id,
                        "checkpoint_index": int(input_lineage["checkpoint_index"]) + 1,
                        "status": (
                            "completed"
                            if failure_type is None
                            else "timed_out"
                            if failure_type == "TimeoutError"
                            else "failed"
                        ),
                        "error_type": failure_type,
                        "elapsed_seconds": elapsed,
                        "allocated_gpu_count": allocated_gpus,
                        "allocation_source": "wall_clock_fallback",
                        "allocated_gpu_hours": invocation_gpu_hours,
                        "executed_stage_seed": args.seed,
                    },
                )
            )
    if gpu_hour_ceiling is not None and invocation_gpu_hours > float(gpu_hour_ceiling):
        raise RuntimeError("expert generation exceeded its GPU-hour ceiling")
    assert metrics is not None
    output = run_dir / "checkpoints" / f"checkpoint_{trainer.generation:06d}.pt"
    _save_checkpoint_exclusive(trainer, output)
    digest = _sha256_file(output)
    budgets = trainer.budgets.as_dict()
    learning_gpu_hours = _lineage_learning_gpu_hours(
        input_lineage, allocated_gpus=allocated_gpus
    ) + invocation_gpu_hours
    ledger.append_budget_snapshot(run_id=trainer.run_id, counters=trainer.budgets)
    ledger.append(
        Event(
            "checkpoint_saved",
            trainer.run_id,
            stage="learning",
            payload={
                "algorithm": "alphazero",
                "checkpoint_index": trainer.generation,
                "generation": trainer.generation,
                "checkpoint": str(output.relative_to(run_dir)),
                "checkpoint_hash": digest,
                "input_checkpoint_hash": input_lineage["checkpoint_hash"],
                "manifest_hash": manifest["manifest_hash"],
                "config_hash": manifest["config_hash"],
                "env_steps": budgets["total"]["env_steps"],
                "optimizer_steps": budgets["total"]["optimizer_steps"],
                "mcts_simulations": budgets["total"]["mcts_simulations"],
                "wall_seconds": budgets["total"]["wall_seconds"],
                "budgets": budgets,
                "training_population_hash": population.content_hash,
                "expert_demo": expert_demo,
                "expert_demo_opening_moves": opening_moves,
                "expert_demo_opening_weight": opening_weight,
                "stage_self_play_episodes": self_play_episodes,
                "stage_training_steps": training_steps,
                "stage_seed": args.seed,
                "stage_gpu_hour_ceiling": gpu_hour_ceiling,
                "stage_allocated_gpu_count": allocated_gpus,
                "stage_allocation_source": "wall_clock_fallback",
                "stage_allocated_gpu_hours": invocation_gpu_hours,
                "attempt_id": attempt_id,
                "learning_gpu_hours": learning_gpu_hours,
            },
        )
    )
    return {
        "checkpoint": str(output),
        "checkpoint_hash": digest,
        "generation": metrics.generation,
        "elapsed_seconds": elapsed,
        "budgets": budgets,
    }


def _workflow(args: argparse.Namespace) -> dict[str, Any]:
    from games.snakego.experiments import run_task10_workflow

    opponent_ids = {
        "rank5": args.rank5,
        "rank6": args.rank6,
        "rank15": args.rank15,
    }

    def run_expert(
        stage,
        checkpoint: Path,
        remaining_gpu_hours: float | None,
        attempt_id: str,
    ) -> Path:
        if stage.opponent_rank not in opponent_ids or stage.seed is None:
            raise ValueError("expert workflow stage has no declared training opponent")
        result = _expert_generation(
            argparse.Namespace(
                run_dir=args.output,
                checkpoint=checkpoint,
                population=args.population,
                opponent=opponent_ids[stage.opponent_rank],
                device=args.device,
                seed=stage.seed,
                move_seconds=args.move_seconds,
                self_play_episodes=stage.episodes,
                training_steps=stage.training_steps,
                expert_demo=stage.expert_demo,
                opening_moves=stage.opening_moves,
                opening_weight=stage.opening_weight,
                gpu_hour_ceiling=remaining_gpu_hours,
                allocated_gpus=args.allocated_gpus,
                workflow_lock_held=True,
                attempt_id=attempt_id,
            )
        )
        return Path(result["checkpoint"])

    with redirect_stdout(io.StringIO()):
        return run_task10_workflow(
            run_dir=args.output,
            config_path=args.config,
            expert_stage_runner=run_expert,
            maximum_stages=args.maximum_stages,
            allocated_gpu_count=args.allocated_gpus,
        )


def _promotion(path: Path) -> dict[str, Any]:
    from rlbench.league import PromotionConfig, evaluate_promotion
    from rlbench.metrics import MatchOutcome

    facts = json.loads(path.read_text(encoding="utf-8"))
    decision = evaluate_promotion(
        candidate_id=str(facts["candidate_id"]),
        champion_id=str(facts["champion_id"]),
        ratings={str(key): float(value) for key, value in facts["ratings"].items()},
        outcomes=tuple(MatchOutcome(**outcome) for outcome in facts["outcomes"]),
        promotion_opponents=frozenset(map(str, facts["promotion_opponents"])),
        protected_reference_scores={
            str(key): float(value)
            for key, value in facts["protected_reference_scores"].items()
        },
        evaluation_complete=bool(facts["evaluation_complete"]),
        config=PromotionConfig(**facts["thresholds"]),
    )
    result = {
        "promoted": decision.promoted,
        "reasons": list(decision.reasons),
        "elo_delta": decision.elo_delta,
        "win_rate_lower_bound": decision.win_rate_lower_bound,
        "protected_scores": dict(decision.protected_scores),
    }
    result["candidate_status"] = (
        "promoted_champion" if decision.promoted else "rejected_candidate"
    )
    return result


def _sweep_variants() -> list[dict[str, Any]]:
    dimensions = {
        "simulations": (8, 16),
        "c_puct": (1.25, 1.75),
        "root_dirichlet_fraction": (0.0, 0.25),
        "temperature_moves": (0, 16),
        "replay_capacity": (64, 256),
        "learning_rate": (0.0003, 0.0005),
        "network_width_depth": ((8, 1), (16, 2)),
        "inference_batch_size": (4, 8),
        "human_mixture_fraction": (0.0, 0.5),
    }
    variants: list[dict[str, Any]] = []
    for dimension, values in dimensions.items():
        for value in values:
            variants.append({"dimension": dimension, "value": value})
    return variants


def _run_sweep(output: Path, *, device: str, dry_run: bool) -> dict[str, Any]:
    variants = _sweep_variants()
    plan = {
        "split": "training_microbenchmark",
        "heldout_used": False,
        "seed": 260806,
        "dimensions": sorted({str(row["dimension"]) for row in variants}),
        "variants": variants,
    }
    if dry_run:
        return plan

    import numpy as np

    from rlbench.algorithms.alphazero import (
        AlphaZeroConfig,
        AlphaZeroTrainer,
        MCTS,
        PolicyValueNet,
        ReplaySample,
    )
    from rlbench.cli.main import game_factory

    class CountingEvaluator:
        def __init__(self, network: PolicyValueNet) -> None:
            self.network = network
            self.batch_sizes: list[int] = []

        def evaluate_batch(self, observations, legal_masks):
            self.batch_sizes.append(len(observations))
            return self.network.evaluate_batch(observations, legal_masks)

    rows: list[dict[str, Any]] = []
    for index, variant in enumerate(variants):
        settings: dict[str, Any] = {
            "simulations": 8,
            "c_puct": 1.75,
            "root_dirichlet_alpha": 0.3,
            "root_dirichlet_fraction": 0.25,
            "self_play_temperature": 1.0,
            "temperature_moves": 16,
            "channels": 8,
            "residual_blocks": 1,
            "learning_rate": 0.0005,
            "weight_decay": 0.0001,
            "batch_size": 2,
            "replay_capacity": 64,
            "min_replay_size": 1,
            "gradient_clip_norm": 5.0,
            "mixed_precision": False,
            "inference_batch_size": 4,
            "device": device,
        }
        dimension = str(variant["dimension"])
        value = variant["value"]
        human_fraction = 0.0
        if dimension == "network_width_depth":
            settings["channels"], settings["residual_blocks"] = value
        elif dimension == "human_mixture_fraction":
            human_fraction = float(value)
        else:
            settings[dimension] = value
        config = AlphaZeroConfig(**settings)
        factory = game_factory("snakego", {"max_round": 512})
        game = factory()
        game.reset(260806 + index)
        network = PolicyValueNet.from_game_spec(game.spec, config, device=device)
        evaluator = CountingEvaluator(network)
        search_started = time.monotonic()
        result = MCTS(config, evaluator, seed=260806 + index).search(
            game,
            training=True,
            move_number=0,
        )
        search_seconds = time.monotonic() - search_started
        trainer = AlphaZeroTrainer(network, config, seed=260806 + index)
        observation = game.observe(game.current_player())
        legal_mask = game.legal_action_mask()
        for sample_index in range(2):
            expert = human_fraction > 0.0 and sample_index == 0
            trainer.replay.add(
                ReplaySample(
                    observation=observation,
                    legal_mask=legal_mask,
                    visit_policy=result.visit_policy,
                    outcome=0.0,
                    player=game.current_player(),
                    source="expert_demo" if expert else "selfplay",
                    sample_weight=32.0 if expert else 1.0,
                    decision_index=0 if expert else -1,
                )
            )
        optimizer_started = time.monotonic()
        metrics = trainer.run_optimizer_steps(1)
        optimizer_seconds = time.monotonic() - optimizer_started
        positions = sum(evaluator.batch_sizes)
        rows.append(
            {
                "variant": index,
                "dimension": dimension,
                "value": json.dumps(value, separators=(",", ":")),
                "completed_simulations": result.completed_simulations,
                "search_seconds": search_seconds,
                "simulations_per_second": result.completed_simulations
                / search_seconds,
                "neural_batches": len(evaluator.batch_sizes),
                "neural_positions": positions,
                "neural_positions_per_second": positions / search_seconds,
                "maximum_neural_batch": max(evaluator.batch_sizes),
                "optimizer_seconds": optimizer_seconds,
                "optimizer_steps": len(metrics),
                "sampled_weight_mass": 33.0 if human_fraction else 2.0,
                "device": device,
                "seed": 260806 + index,
                "split": "training_microbenchmark",
                "heldout_used": False,
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "train_only_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    json_path = output / "train_only_sweep.json"
    json_path.write_text(
        json.dumps({**plan, "results": rows}, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return {
        **plan,
        "rows": len(rows),
        "csv": str(csv_path),
        "json": str(json_path),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _lineage_learning_gpu_hours(
    lineage: dict[str, Any], *, allocated_gpus: int
) -> float:
    value = lineage.get("learning_gpu_hours")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    budgets = lineage.get("budgets")
    learning = budgets.get("learning") if isinstance(budgets, dict) else None
    wall_seconds = learning.get("wall_seconds") if isinstance(learning, dict) else None
    if isinstance(wall_seconds, (int, float)) and not isinstance(wall_seconds, bool):
        return float(wall_seconds) * allocated_gpus / 3600.0
    raise ValueError("checkpoint lineage has no learning allocation time")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "plan":
        _print(_plan())
        return 0
    if args.command == "workflow":
        _print(_workflow(args))
        return 0
    if args.command == "train-alphazero":
        return _train("alphazero", AZ_CONFIG, args.output, args.resume)
    if args.command == "train-ppo":
        return _train("ppo", PPO_CONFIG, args.output, None)
    if args.command == "expert-generation":
        _print(_expert_generation(args))
        return 0
    if args.command == "evaluate":
        return _evaluate(args.checkpoint, args.population, args.seeds)
    if args.command == "promotion":
        _print(_promotion(args.facts))
        return 0
    if args.command == "sweep":
        _print(_run_sweep(args.output, device=args.device, dry_run=args.dry_run))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
