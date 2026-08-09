"""Usable local CLI for validation, training, evaluation, and reporting."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from uuid import UUID, uuid4

import numpy as np
import torch

from rlbench.algorithms.alphazero import (
    AlphaZeroConfig,
    AlphaZeroTrainer,
    MCTS,
    PolicyValueNet,
)
from rlbench.algorithms.ppo_tianshou import PPOConfig, PPORecurrentState, PPOTrainer
from rlbench.config import ComposedConfig, canonical_config_hash, compose_config
from rlbench.evaluation import (
    DeadlineAwareGamePolicy,
    DeadlineAwareLocalPolicy,
    EvaluationRunner,
    build_side_swapped_cases,
)
from rlbench.game import Observation, validate_game
from rlbench.metrics import (
    fit_anchored_elo,
    occupancy_shift,
    policy_kl,
    summarize_information_gain,
    win_rate_summary,
)
from rlbench.population import (
    PopulationEntry,
    PopulationManifest,
    ProcessAgent,
    ProcessMoveTimeout,
    SnakeGoProcessPolicy,
)
from rlbench.registry import ALGORITHMS, GAMES, game_factory
from rlbench.reporting import generate_report
from rlbench.telemetry import BudgetCounters, Event, EventLedger, ResourceSampler


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rlbench")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate-game", help="run the shared game contract")
    validate.add_argument("game", choices=sorted(GAMES))
    validate.add_argument("--seed", type=int, default=0)

    train = commands.add_parser("train", help="run one finite learning job")
    train.add_argument("game", choices=sorted(GAMES))
    train.add_argument("--algo", required=True, choices=sorted(ALGORITHMS))
    train.add_argument("--config", required=True)
    train.add_argument("--output")
    train.add_argument("--resume")
    train.add_argument("--initialize", help="warm-start PPO model weights")
    train.add_argument("--population", help="training population manifest")
    train.add_argument("--opponent-id", help="train-pool process opponent")
    train.add_argument(
        "--opponent", choices=("random",), help="built-in PPO opponent"
    )

    evaluate = commands.add_parser("evaluate", help="evaluate a local checkpoint")
    evaluate.add_argument("game", choices=sorted(GAMES))
    evaluate.add_argument("--checkpoint", required=True)
    opponents = evaluate.add_mutually_exclusive_group()
    opponents.add_argument("--population")
    opponents.add_argument("--opponent", choices=("random",), default="random")
    evaluate.add_argument("--seeds", default=None, help="comma-separated non-negative seeds")

    report = commands.add_parser("report", help="derive tables and curves from facts")
    report.add_argument("run_directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "validate-game":
        return _validate_command(args.game, seed=args.seed)
    if args.command == "train":
        return _train_command(
            args.game,
            algorithm=args.algo,
            config_path=args.config,
            output=args.output,
            resume=args.resume,
            initialize=args.initialize,
            population=args.population,
            opponent_id=args.opponent_id,
            local_opponent=args.opponent,
        )
    if args.command == "evaluate":
        return _evaluate_command(
            args.game,
            checkpoint=args.checkpoint,
            population=args.population,
            opponent=args.opponent,
            seeds=_parse_seeds(args.seeds) if args.seeds else None,
        )
    if args.command == "report":
        report_dir = generate_report(args.run_directory)
        _print_json({"report_directory": str(report_dir)})
        return 0
    raise RuntimeError("unreachable command")


def _validate_command(game_name: str, *, seed: int) -> int:
    game = game_factory(game_name)()
    game.reset(seed)
    validate_game(game)
    spec = game.spec
    _print_json(
        {
            "action_count": len(spec.action_names),
            "game": game_name,
            "observation_planes": len(spec.observation_spec.plane_names),
            "observation_scalars": len(spec.observation_spec.scalar_names),
            "status": "valid",
        }
    )
    return 0


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
        manifest = _load_manifest(manifest_path)
        _validate_resume_manifest(manifest, composed)
    else:
        if resume is not None:
            raise ValueError("resume requires the original run directory and manifest")
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = _create_manifest(composed)
        _write_immutable_manifest(manifest_path, manifest)

    run_id = str(manifest["run_id"])
    ledger = EventLedger(run_dir / "events.jsonl")
    if resume is not None:
        _validate_training_inputs(ledger, training_inputs)
    resume_lineage: Mapping[str, Any] | None = None
    if resume is not None:
        resume_lineage = _validate_checkpoint_lineage(
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
    prior_gpu_hours = _latest_gpu_hours(ledger, run_id=run_id)
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
            _restore_event_backed_budgets(trainer, ledger=ledger, run_id=run_id)
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

    _print_json(
        {
            "algorithm": algorithm,
            "checkpoint": str(checkpoint_path),
            "checkpoint_hash": checkpoint_hash,
            "run_directory": str(run_dir),
            "run_id": run_id,
        }
    )
    return 0


def _build_trainer(
    config: ComposedConfig,
    *,
    ledger: EventLedger,
    run_id: str,
    opponent: Any = None,
    opponent_id: str = "opponent",
) -> tuple[Any, str]:
    factory = game_factory(config.game, config.canonical["game"])
    seed = int(config.canonical["training"]["seed"])
    algorithm_config = config.algorithm_settings()
    if config.algorithm == "alphazero":
        sample = factory()
        network = PolicyValueNet.from_game_spec(
            sample.spec,
            algorithm_config,
            device=_alphazero_device(algorithm_config),
        )
        return (
            AlphaZeroTrainer(
                network,
                algorithm_config,
                seed=seed,
                ledger=ledger,
                run_id=run_id,
            ),
            "generation",
        )
    if config.algorithm == "ppo":
        return (
            PPOTrainer(
                factory,
                algorithm_config,
                seed=seed,
                ledger=ledger,
                run_id=run_id,
                opponent=opponent,
                opponent_id=opponent_id,
            ),
            "iteration",
        )
    raise ValueError(f"unknown algorithm: {config.algorithm}")


def _alphazero_device(config: AlphaZeroConfig) -> str:
    if config.device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return config.device


def _run_training(trainer: Any, config: ComposedConfig) -> Iterator[int]:
    controls = config.canonical["training"]
    if config.algorithm == "alphazero":
        factory = game_factory(config.game, config.canonical["game"])
        for _ in range(int(controls["generations"])):
            metrics = trainer.run_generation(
                factory,
                self_play_episodes=int(controls["self_play_episodes"]),
                training_steps=int(controls["training_steps"]),
                processes=int(controls["processes"]),
            )
            if trainer.ledger is not None:
                trainer.ledger.append(
                    Event(
                        event_type="training_batch_finished",
                        run_id=trainer.run_id,
                        stage="learning",
                        payload={
                            "backend": "alphazero",
                            "checkpoint_index": metrics.generation,
                            **asdict(metrics),
                        },
                    )
                )
            yield int(metrics.generation)
        return
    for _ in range(int(controls["iterations"])):
        metrics = trainer.train_iteration()
        if trainer.ledger is not None:
            trainer.ledger.append(
                Event(
                    event_type="training_batch_finished",
                    run_id=trainer.run_id,
                    stage="learning",
                    payload={
                        "backend": "ppo",
                        "checkpoint_index": metrics.iteration,
                        **asdict(metrics),
                    },
                )
            )
        yield int(metrics.iteration)


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
    _save_checkpoint_exclusive(trainer, checkpoint_path)
    ledger.append_budget_snapshot(run_id=run_id, counters=trainer.budgets)
    budgets = trainer.budgets.as_dict()
    invocation_learning_gpu_hours = (
        totals["learning"].allocated_gpu_hours if sample_resources else None
    )
    learning_gpu_hours = _sum_optional_hours(
        prior_gpu_hours["learning"], invocation_learning_gpu_hours
    )
    evaluation_gpu_hours = prior_gpu_hours["evaluation"]
    total_gpu_hours = _sum_optional_hours(learning_gpu_hours, evaluation_gpu_hours)
    checkpoint_hash = _sha256_file(checkpoint_path)
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


class _AlphaZeroEvaluationPolicy(DeadlineAwareGamePolicy):
    def __init__(
        self,
        trainer: AlphaZeroTrainer,
        config: AlphaZeroConfig,
        seed: int,
        *,
        prior_trainer: AlphaZeroTrainer | None = None,
    ) -> None:
        self.search = MCTS(config, trainer.network, seed=seed)
        self.prior_search = (
            MCTS(config, prior_trainer.network, seed=seed)
            if prior_trainer is not None
            else None
        )
        self.search_calls = 0
        self.completed_simulations = 0
        self.local_kls: list[float] = []

    def act_game(self, game: Any) -> int:
        return self.act_game_with_deadline(game, deadline=None)

    def act_game_with_deadline(
        self, game: Any, *, deadline: float | None
    ) -> int:
        self.search_calls += 1
        move_number = int(getattr(getattr(game, "state", None), "action_count", 0))
        current_deadline = deadline
        if deadline is not None and self.prior_search is not None:
            started = time.monotonic()
            current_deadline = started + max(0.0, deadline - started) / 2.0
        current = self.search.search(
            game,
            training=False,
            move_number=move_number,
            deadline=current_deadline,
        )
        self.completed_simulations += current.completed_simulations
        if self.prior_search is not None:
            prior = self.prior_search.search(
                game,
                training=False,
                move_number=move_number,
                deadline=deadline,
            )
            self.completed_simulations += prior.completed_simulations
            legal = np.flatnonzero(game.legal_action_mask()).tolist()
            current_probabilities = np.asarray(
                [current.visit_policy[action] for action in legal], dtype=np.float64
            )
            prior_probabilities = np.asarray(
                [prior.visit_policy[action] for action in legal], dtype=np.float64
            )
            current_probabilities /= current_probabilities.sum()
            prior_probabilities /= prior_probabilities.sum()
            self.local_kls.append(
                policy_kl(
                    current_probabilities,
                    prior_probabilities,
                    legal_support=legal,
                )
            )
        return current.action


class _PPOEvaluationPolicy(DeadlineAwareGamePolicy, DeadlineAwareLocalPolicy):
    def __init__(
        self, trainer: PPOTrainer, *, prior_trainer: PPOTrainer | None = None
    ) -> None:
        self.trainer = trainer
        self.prior_trainer = prior_trainer
        self.state: PPORecurrentState | None = None
        self.prior_state: PPORecurrentState | None = None
        self.local_kls: list[float] = []

    def reset_episode(self) -> None:
        self.state = None
        self.prior_state = None

    def __call__(
        self, observation: Observation, legal_mask: np.ndarray[Any, Any]
    ) -> int:
        decision = self.trainer.select_action_step(
            observation,
            legal_mask,
            deterministic=True,
            state=self.state,
        )
        if self.prior_trainer is not None:
            prior = self.prior_trainer.action_distribution_step(
                observation,
                legal_mask,
                state=self.prior_state,
            )
            self.prior_state = prior.state
            legal = np.flatnonzero(legal_mask).tolist()
            current_probabilities = np.asarray(
                [decision.probabilities[action] for action in legal], dtype=np.float64
            )
            prior_probabilities = np.asarray(
                [prior.probabilities[action] for action in legal], dtype=np.float64
            )
            current_probabilities /= current_probabilities.sum()
            prior_probabilities /= prior_probabilities.sum()
            self.local_kls.append(
                policy_kl(
                    current_probabilities,
                    prior_probabilities,
                    legal_support=legal,
                )
            )
        self.state = decision.state
        return decision.action

    def act_with_deadline(
        self,
        observation: Observation,
        legal_mask: np.ndarray[Any, Any],
        *,
        deadline: float | None,
    ) -> int:
        if deadline is not None and time.monotonic() >= deadline:
            raise ProcessMoveTimeout("PPO move deadline expired before inference")
        return self(observation, legal_mask)

    def act_game_with_deadline(
        self, game: Any, *, deadline: float | None
    ) -> int:
        if deadline is not None and time.monotonic() >= deadline:
            raise ProcessMoveTimeout("PPO move deadline expired before inference")
        player = int(game.current_player())
        observation = game.observe(player)
        training_action_mask = getattr(game, "training_action_mask", None)
        mask = (
            np.asarray(training_action_mask(player), dtype=np.bool_)
            if callable(training_action_mask)
            else np.asarray(game.legal_action_mask(), dtype=np.bool_)
        )
        return self(observation, mask)


class _RandomPolicy:
    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def start_case(self, case: Any, agent_id: str, side: int) -> None:
        payload = f"{case.seed}|{agent_id}|{side}".encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        self.rng = np.random.default_rng(seed)

    def __call__(self, observation: Observation, legal_mask: np.ndarray[Any, Any]) -> int:
        del observation
        legal = np.flatnonzero(legal_mask)
        return int(self.rng.choice(legal))


class _TrainingRandomPolicy:
    """Stateless pseudo-random opponent with checkpoint-independent replay."""

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)

    def __call__(self, observation: Observation, legal_mask: np.ndarray[Any, Any]) -> int:
        legal = np.flatnonzero(legal_mask)
        digest = hashlib.sha256(self.seed.to_bytes(8, "big", signed=True))
        digest.update(np.asarray(observation.planes, dtype=np.float32).tobytes())
        digest.update(np.asarray(observation.scalars, dtype=np.float32).tobytes())
        digest.update(np.asarray(legal_mask, dtype=np.bool_).tobytes())
        index = int.from_bytes(digest.digest()[:8], "big") % len(legal)
        return int(legal[index])


def _evaluate_command(
    game_name: str,
    *,
    checkpoint: str | Path,
    population: str | Path | None,
    opponent: str,
    seeds: tuple[int, ...] | None,
) -> int:
    checkpoint_path = Path(checkpoint).resolve()
    run_dir = checkpoint_path.parent.parent
    manifest = _load_manifest(run_dir / "run_manifest.json")
    if manifest.get("game") != game_name:
        raise ValueError("checkpoint run manifest game does not match evaluation game")
    algorithm = str(manifest["algorithm"])
    canonical = manifest.get("canonical_config")
    if not isinstance(canonical, Mapping):
        raise ValueError("run manifest has no canonical configuration")
    run_id = str(manifest["run_id"])
    ledger = EventLedger(run_dir / "events.jsonl")
    _validate_checkpoint_lineage(
        run_dir,
        checkpoint_path,
        manifest=manifest,
        ledger=ledger,
    )
    prior_gpu_hours = _latest_gpu_hours(ledger, run_id=run_id)
    game_config = dict(canonical["game"])
    algorithm_config = dict(canonical["algorithm"])
    factory = game_factory(game_name, game_config)
    prior_checkpoint_path = _prior_checkpoint_path(
        run_dir,
        checkpoint_path,
        manifest=manifest,
        ledger=ledger,
    )
    if algorithm == "alphazero":
        az_config = AlphaZeroConfig(**algorithm_config)
        device = _alphazero_device(az_config)
        network = PolicyValueNet.from_game_spec(factory().spec, az_config, device=device)
        trainer: Any = AlphaZeroTrainer(network, az_config, ledger=ledger, run_id=run_id)
        trainer.load_checkpoint(checkpoint_path)
        _restore_event_backed_budgets(trainer, ledger=ledger, run_id=run_id)
        prior_trainer = None
        if prior_checkpoint_path is not None:
            prior_network = PolicyValueNet.from_game_spec(
                factory().spec, az_config, device=device
            )
            prior_trainer = AlphaZeroTrainer(prior_network, az_config)
            prior_trainer.load_checkpoint(prior_checkpoint_path)
        candidate: Any = _AlphaZeroEvaluationPolicy(
            trainer, az_config, seed=0, prior_trainer=prior_trainer
        )
        checkpoint_index = trainer.generation
    elif algorithm == "ppo":
        ppo_config = PPOConfig(**algorithm_config)
        trainer = PPOTrainer(factory, ppo_config, ledger=ledger, run_id=run_id)
        trainer.load_checkpoint(checkpoint_path)
        _restore_event_backed_budgets(trainer, ledger=ledger, run_id=run_id)
        prior_trainer = None
        if prior_checkpoint_path is not None:
            prior_trainer = PPOTrainer(factory, ppo_config, seed=0)
            prior_trainer.load_checkpoint(prior_checkpoint_path)
        candidate = _PPOEvaluationPolicy(trainer, prior_trainer=prior_trainer)
        checkpoint_index = trainer.iteration
    else:
        raise ValueError(f"unknown algorithm in run manifest: {algorithm}")

    chosen_seeds = seeds or tuple(int(seed) for seed in canonical["evaluation"]["seeds"])
    checkpoint_hash = _sha256_file(checkpoint_path)
    agents: dict[str, Any] = {"learner": candidate}
    process_agents: list[ProcessAgent] = []
    cases = []
    move_seconds = canonical["evaluation"].get("move_seconds")
    limits = {} if move_seconds is None else {"move_seconds": move_seconds}
    if population is not None:
        population_manifest = PopulationManifest.from_yaml(population)
        for entry in population_manifest.entries:
            process = _population_policy(
                entry, population_manifest.population_root, game_name=game_name
            )
            process_agents.append(process)
            agents[entry.agent_id] = process
            cases.extend(
                build_side_swapped_cases(
                    candidate_id="learner",
                    candidate_hash=checkpoint_hash,
                    opponent_id=entry.agent_id,
                    opponent_hash=entry.content_hash,
                    seeds=chosen_seeds,
                    game_config=game_config,
                    limits=limits,
                    protocol_version=population_manifest.protocol_version,
                )
            )
    else:
        if opponent != "random":
            raise ValueError(f"unsupported local opponent: {opponent}")
        random_policy = _RandomPolicy(int(chosen_seeds[0]))
        agents["random"] = random_policy
        cases.extend(
            build_side_swapped_cases(
                candidate_id="learner",
                candidate_hash=checkpoint_hash,
                opponent_id="random",
                opponent_hash="builtin:random-v1",
                seeds=chosen_seeds,
                game_config=game_config,
                limits=limits,
            )
        )

    case_set_hash = _case_set_hash(cases)

    evaluation_id = str(uuid4())
    started = time.monotonic()
    sampler = ResourceSampler(
        run_id=run_id,
        host_id=f"evaluation-{evaluation_id}",
        ledger=ledger,
    )
    try:
        sampler.sample("evaluation")
        report = EvaluationRunner(
            lambda value: GAMES[game_name](value), ledger
        ).run(
            cases,
            agents=agents,
            run_id=run_id,
            evaluation_id=evaluation_id,
        )
        sampler.sample("evaluation")
        resource_totals = sampler.totals()
    finally:
        sampler.close()
        for process in process_agents:
            process.close()
    elapsed = time.monotonic() - started
    env_steps = sum(len(result.actions) for result in report.results)
    trainer.budgets.evaluation.episodes += len(report.results)
    trainer.budgets.evaluation.env_steps += env_steps
    if algorithm == "alphazero":
        trainer.budgets.add_mcts_simulations(
            candidate.completed_simulations,
            "evaluation",
        )
    trainer.budgets.add_wall_seconds(elapsed, "evaluation")
    ledger.append_budget_snapshot(run_id=run_id, counters=trainer.budgets)
    for result in report.results:
        ledger.append(
            Event(
                event_type="match_finished",
                run_id=run_id,
                stage="evaluation",
                payload={
                    "checkpoint_index": checkpoint_index,
                    "evaluation_id": evaluation_id,
                    "case_id": result.case_id,
                    "case_hash": result.case_hash,
                    "case_set_hash": case_set_hash,
                    "seed": result.seed,
                    "player_0": result.player_0,
                    "player_1": result.player_1,
                    "actions": list(result.actions),
                    "score_player_0": result.score_player_0,
                    "valid": result.valid,
                    "reason": result.reason,
                },
            )
        )
    metrics_availability: dict[str, dict[str, Any]] = {}
    if prior_checkpoint_path is None:
        metrics_availability["information_gain"] = {
            "available": False,
            "reason": "no prior checkpoint",
        }
    elif candidate.local_kls:
        ig = summarize_information_gain(
            candidate.local_kls, episodes=len(report.results)
        )
        ledger.append(
            Event(
                event_type="policy_ig_measured",
                run_id=run_id,
                stage="evaluation",
                payload={
                    "evaluation_id": evaluation_id,
                    "case_set_hash": case_set_hash,
                    "checkpoint_index": checkpoint_index,
                    "prior_checkpoint": str(prior_checkpoint_path.relative_to(run_dir)),
                    "local_kls": list(ig.local_kls),
                    "nats_per_decision": ig.nats_per_decision,
                    "nats_per_episode": ig.nats_per_episode,
                },
            )
        )
        metrics_availability["information_gain"] = {
            "available": True,
            "reason": None,
        }
    else:
        metrics_availability["information_gain"] = {
            "available": False,
            "reason": "no aligned learner decisions",
        }

    prior_occupancy = _prior_complete_evaluation_states(
        ledger,
        run_id=run_id,
        checkpoint_index=checkpoint_index,
        case_set_hash=case_set_hash,
    )
    current_states = _evaluation_state_ids(
        ledger, run_id=run_id, evaluation_id=evaluation_id
    )
    if prior_occupancy is None:
        metrics_availability["occupancy"] = {
            "available": False,
            "reason": "no prior complete checkpoint evaluation trace",
        }
    elif current_states:
        prior_evaluation_id, prior_states = prior_occupancy
        support = sorted(set(prior_states) | set(current_states))
        prior_counts = Counter(prior_states)
        current_counts = Counter(current_states)
        shift = occupancy_shift(
            {state: float(current_counts[state]) for state in support},
            {state: float(prior_counts[state]) for state in support},
            support=support,
        )
        ledger.append(
            Event(
                event_type="occupancy_measured",
                run_id=run_id,
                stage="evaluation",
                payload={
                    "evaluation_id": evaluation_id,
                    "prior_evaluation_id": prior_evaluation_id,
                    "case_set_hash": case_set_hash,
                    "checkpoint_index": checkpoint_index,
                    "occupancy_shift": shift,
                    "current_state_count": len(current_states),
                    "prior_state_count": len(prior_states),
                },
            )
        )
        metrics_availability["occupancy"] = {
            "available": True,
            "reason": None,
        }
    else:
        metrics_availability["occupancy"] = {
            "available": False,
            "reason": "no state identifiers in current evaluation trace",
        }
    valid_outcomes = tuple(outcome for outcome in report.outcomes if outcome.valid)
    score = win_rate_summary(valid_outcomes, "learner")
    opponents = sorted(
        (
            {result.player_0 for result in report.results}
            | {result.player_1 for result in report.results}
        )
        - {"learner"}
    )
    elo_payload: dict[str, Any] | None = None
    if opponents and valid_outcomes:
        ratings = fit_anchored_elo(valid_outcomes, anchor=opponents[0])
        elo_payload = {
            "rating": ratings.ratings.get("learner"),
            "uncertainty": ratings.uncertainties.get("learner"),
            "anchor": ratings.anchor,
        }
    learning_gpu_hours = prior_gpu_hours["learning"]
    evaluation_gpu_hours = _sum_optional_hours(
        prior_gpu_hours["evaluation"],
        resource_totals["evaluation"].allocated_gpu_hours,
    )
    total_gpu_hours = _sum_optional_hours(learning_gpu_hours, evaluation_gpu_hours)
    ledger.append(
        Event(
            event_type="evaluation_finished",
            run_id=run_id,
            stage="evaluation",
            payload={
                "checkpoint_index": checkpoint_index,
                "evaluation_id": evaluation_id,
                "case_set_hash": case_set_hash,
                "complete": report.complete,
                "valid_games": score.valid_games,
                "raw_matches": len(report.results),
                "env_steps": env_steps,
                "wall_seconds": elapsed,
                "budgets": trainer.budgets.as_dict(),
                "learning_gpu_hours": learning_gpu_hours,
                "evaluation_gpu_hours": evaluation_gpu_hours,
                "gpu_hours": total_gpu_hours,
                "elo": elo_payload,
                "metrics_availability": metrics_availability,
            },
        )
    )
    _print_json(
        {
            "checkpoint_index": checkpoint_index,
            "complete": report.complete,
            "elo": elo_payload,
            "score": score.score,
            "side_swapped": True,
            "valid_games": score.valid_games,
        }
    )
    return 0


def _population_policy(
    entry: PopulationEntry, population_root: str | Path, *, game_name: str
) -> ProcessAgent:
    """Construct the process boundary declared by one immutable manifest entry."""
    if entry.protocol == "snakego_official":
        if game_name != "snakego":
            raise ValueError("snakego_official agents require the SnakeGo game")
        return SnakeGoProcessPolicy(entry, population_root)
    return ProcessAgent(entry, population_root)


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
            "checkpoint_hash": _sha256_file(initial_path),
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


def _create_manifest(config: ComposedConfig) -> dict[str, Any]:
    core = {
        "schema_version": 1,
        "run_id": str(uuid4()),
        "game": config.game,
        "algorithm": config.algorithm,
        "canonical_config": dict(config.canonical),
        "config_hash": config.config_hash,
        "source_hashes": dict(config.source_hashes),
        "software": _software_facts(),
        "hardware": _hardware_facts(),
    }
    encoded = json.dumps(
        core, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return {**core, "manifest_hash": f"sha256:{hashlib.sha256(encoded).hexdigest()}"}


def _software_facts() -> dict[str, Any]:
    facts = {"python": platform.python_version()}
    for package in ("agentbench-rl-frame", "numpy", "torch", "tianshou"):
        try:
            facts[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            facts[package] = None
    return facts


def _hardware_facts() -> dict[str, Any]:
    accelerators = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            accelerators.append(
                {
                    "index": index,
                    "model": properties.name,
                    "memory_bytes": properties.total_memory,
                }
            )
    return {
        "architecture": platform.machine(),
        "cpu_count": os.cpu_count(),
        "operating_system": platform.system(),
        "cuda_available": torch.cuda.is_available(),
        "accelerators": accelerators,
    }


def _write_immutable_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    encoded = json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid run manifest: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("run manifest must be a JSON object")
    _validate_manifest(value)
    return value


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "run_id",
        "game",
        "algorithm",
        "canonical_config",
        "config_hash",
        "source_hashes",
        "software",
        "hardware",
        "manifest_hash",
    }
    if set(manifest) != required:
        raise ValueError("run manifest schema fields are invalid")
    if manifest["schema_version"] != 1:
        raise ValueError("unsupported run manifest schema_version")
    try:
        UUID(str(manifest["run_id"]))
    except ValueError as exc:
        raise ValueError("run manifest run_id is invalid") from exc
    if manifest["game"] not in GAMES or manifest["algorithm"] not in ALGORITHMS:
        raise ValueError("run manifest registry identity is invalid")
    for field in ("canonical_config", "source_hashes", "software", "hardware"):
        if not isinstance(manifest[field], Mapping):
            raise ValueError(f"run manifest {field} must be a mapping")
    canonical_hash = canonical_config_hash(manifest["canonical_config"])
    if manifest["config_hash"] != canonical_hash:
        raise ValueError("run manifest canonical configuration hash mismatch")
    core = dict(manifest)
    stored_hash = core.pop("manifest_hash")
    encoded = json.dumps(
        core, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    actual_hash = f"sha256:{hashlib.sha256(encoded).hexdigest()}"
    if stored_hash != actual_hash:
        raise ValueError("run manifest hash mismatch")


def _validate_checkpoint_lineage(
    run_dir: Path,
    checkpoint_path: Path,
    *,
    manifest: Mapping[str, Any],
    ledger: EventLedger,
    require_head: bool = False,
) -> Mapping[str, Any]:
    resolved_run = run_dir.resolve()
    resolved_checkpoint = checkpoint_path.resolve()
    if not resolved_checkpoint.is_relative_to(resolved_run):
        raise ValueError("checkpoint lineage requires a path inside its run directory")
    saved = [
        event
        for event in ledger.read()
        if event.run_id == manifest["run_id"]
        and event.event_type == "checkpoint_saved"
        and event.payload.get("manifest_hash") == manifest["manifest_hash"]
    ]
    candidates = saved[-1:] if require_head else reversed(saved)
    for event in candidates:
        payload = event.payload
        relative = payload.get("checkpoint")
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError("checkpoint lineage path is invalid")
        recorded_path = (resolved_run / relative).resolve()
        if not recorded_path.is_relative_to(resolved_run):
            raise ValueError("checkpoint lineage path escapes its run directory")
        if require_head and recorded_path != resolved_checkpoint:
            raise ValueError("resume checkpoint is not the latest valid lineage head")
        if recorded_path != resolved_checkpoint:
            continue
        if payload.get("checkpoint_hash") == _sha256_file(recorded_path):
            return payload
        raise ValueError("checkpoint lineage content hash mismatch")
    raise ValueError("checkpoint lineage is not recorded for this run manifest")


def _save_checkpoint_exclusive(trainer: Any, destination: Path) -> None:
    """Commit a checkpoint atomically without replacing historical bytes."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"checkpoint destination already exists: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".pending",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        trainer.save_checkpoint(temporary)
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise FileExistsError(
                f"checkpoint destination already exists: {destination}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _prior_checkpoint_path(
    run_dir: Path,
    checkpoint_path: Path,
    *,
    manifest: Mapping[str, Any],
    ledger: EventLedger,
) -> Path | None:
    saved = [
        event.payload
        for event in ledger.read()
        if event.run_id == manifest["run_id"]
        and event.event_type == "checkpoint_saved"
        and event.payload.get("manifest_hash") == manifest["manifest_hash"]
    ]
    current_position = None
    for position, payload in enumerate(saved):
        relative = payload.get("checkpoint")
        if isinstance(relative, str) and (run_dir / relative).resolve() == checkpoint_path:
            current_position = position
            break
    if current_position is None:
        raise ValueError("checkpoint lineage is not recorded for this run manifest")
    if current_position == 0:
        return None
    prior_relative = saved[current_position - 1].get("checkpoint")
    if not isinstance(prior_relative, str):
        raise ValueError("prior checkpoint lineage path is invalid")
    prior_path = (run_dir / prior_relative).resolve()
    _validate_checkpoint_lineage(
        run_dir, prior_path, manifest=manifest, ledger=ledger
    )
    return prior_path


def _validate_resume_manifest(manifest: Mapping[str, Any], config: ComposedConfig) -> None:
    if (
        manifest.get("game") != config.game
        or manifest.get("algorithm") != config.algorithm
        or manifest.get("config_hash") != config.config_hash
    ):
        raise ValueError("resume configuration does not match the immutable run manifest")


def _sha256_file(path: Path) -> str:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"checkpoint is not readable: {path}") from exc
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise ValueError("seeds must be comma-separated integers") from exc
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be non-negative")
    return seeds


def _restore_event_backed_budgets(
    trainer: Any, *, ledger: EventLedger, run_id: str
) -> None:
    latest = ledger.latest_budget_counters(run_id=run_id)
    if latest is None:
        return
    latest.require_at_least(trainer.budgets)
    trainer.budgets = latest


def _latest_gpu_hours(
    ledger: EventLedger, *, run_id: str
) -> dict[str, float | None]:
    latest: dict[str, float | None] = {
        "learning": 0.0,
        "evaluation": 0.0,
        "total": 0.0,
    }
    for event in ledger.read():
        if event.run_id != run_id or event.event_type not in {
            "checkpoint_saved",
            "evaluation_finished",
        }:
            continue
        payload = event.payload
        if not all(
            key in payload
            for key in (
                "learning_gpu_hours",
                "evaluation_gpu_hours",
                "gpu_hours",
            )
        ):
            continue
        latest = {
            "learning": payload["learning_gpu_hours"],
            "evaluation": payload["evaluation_gpu_hours"],
            "total": payload["gpu_hours"],
        }
    return latest


def _sum_optional_hours(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left) + float(right)


def _evaluation_state_ids(
    ledger: EventLedger, *, run_id: str, evaluation_id: str
) -> tuple[str, ...]:
    return tuple(
        str(event.payload["state_id"])
        for event in ledger.read()
        if event.run_id == run_id
        and event.event_type == "evaluation_move"
        and event.payload.get("evaluation_id") == evaluation_id
        and isinstance(event.payload.get("state_id"), str)
        and event.payload["state_id"]
    )


def _case_set_hash(cases: Sequence[Any]) -> str:
    normalized_cases = []
    for case in cases:
        normalized_cases.append(
            {
                "seed": case.seed,
                "player_0": case.player_0,
                "player_1": case.player_1,
                "player_0_hash": (
                    "candidate:checkpoint-independent"
                    if case.player_0 == "learner"
                    else case.player_0_hash
                ),
                "player_1_hash": (
                    "candidate:checkpoint-independent"
                    if case.player_1 == "learner"
                    else case.player_1_hash
                ),
                "game_config": dict(case.game_config),
                "limits": dict(case.limits),
                "protocol_version": case.protocol_version,
            }
        )
    encoded = json.dumps(
        sorted(
            normalized_cases,
            key=lambda value: json.dumps(
                value, allow_nan=False, separators=(",", ":"), sort_keys=True
            ),
        ),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _prior_complete_evaluation_states(
    ledger: EventLedger,
    *,
    run_id: str,
    checkpoint_index: int,
    case_set_hash: str,
) -> tuple[str, tuple[str, ...]] | None:
    candidates: list[tuple[int, str]] = []
    for event in ledger.read():
        if (
            event.run_id != run_id
            or event.event_type != "evaluation_finished"
            or not event.payload.get("complete")
            or event.payload.get("case_set_hash") != case_set_hash
        ):
            continue
        prior_index = event.payload.get("checkpoint_index")
        evaluation_id = event.payload.get("evaluation_id")
        if (
            isinstance(prior_index, int)
            and prior_index < checkpoint_index
            and isinstance(evaluation_id, str)
        ):
            candidates.append((prior_index, evaluation_id))
    if not candidates:
        return None
    prior_index = max(index for index, _ in candidates)
    evaluation_id = [
        candidate_id
        for index, candidate_id in candidates
        if index == prior_index
    ][-1]
    states = _evaluation_state_ids(
        ledger, run_id=run_id, evaluation_id=evaluation_id
    )
    if not states:
        return None
    return evaluation_id, states


def _print_json(value: Mapping[str, Any]) -> None:
    print(json.dumps(value, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
