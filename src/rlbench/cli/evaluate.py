"""``rlbench evaluate`` subcommand."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from uuid import uuid4

from rlbench.algorithms.alphazero import (
    AlphaZeroConfig,
    AlphaZeroTrainer,
    PolicyValueNet,
)
from rlbench.algorithms.ppo_tianshou import PPOConfig, PPOTrainer
from rlbench.evaluation import EvaluationRunner, build_side_swapped_cases
from rlbench.metrics import (
    fit_anchored_elo,
    occupancy_shift,
    summarize_information_gain,
    win_rate_summary,
)
from rlbench.population import PopulationManifest, ProcessAgent
from rlbench.registry import GAMES, game_factory
from rlbench.telemetry import Event, EventLedger, ResourceSampler

from ._facts import (
    case_set_hash,
    evaluation_state_ids,
    latest_gpu_hours,
    load_manifest,
    print_json,
    prior_checkpoint_path,
    prior_complete_evaluation_states,
    restore_event_backed_budgets,
    sha256_file,
    sum_optional_hours,
    validate_checkpoint_lineage,
)
from ._policies import (
    _AlphaZeroEvaluationPolicy,
    _population_policy,
    _PPOEvaluationPolicy,
    _RandomPolicy,
)
from .main import _alphazero_device


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
    manifest = load_manifest(run_dir / "run_manifest.json")
    if manifest.get("game") != game_name:
        raise ValueError("checkpoint run manifest game does not match evaluation game")
    algorithm = str(manifest["algorithm"])
    canonical = manifest.get("canonical_config")
    if not isinstance(canonical, Mapping):
        raise ValueError("run manifest has no canonical configuration")
    run_id = str(manifest["run_id"])
    ledger = EventLedger(run_dir / "events.jsonl")
    validate_checkpoint_lineage(
        run_dir,
        checkpoint_path,
        manifest=manifest,
        ledger=ledger,
    )
    prior_gpu_hours = latest_gpu_hours(ledger, run_id=run_id)
    game_config = dict(canonical["game"])
    algorithm_config = dict(canonical["algorithm"])
    factory = game_factory(game_name, game_config)
    prior_checkpoint = prior_checkpoint_path(
        run_dir,
        checkpoint_path,
        manifest=manifest,
        ledger=ledger,
    )
    if algorithm == "alphazero":
        az_config = AlphaZeroConfig(**algorithm_config)
        device = _alphazero_device(az_config)
        network = PolicyValueNet.from_game_spec(factory().spec, az_config, device=device)
        trainer: object = AlphaZeroTrainer(network, az_config, ledger=ledger, run_id=run_id)
        trainer.load_checkpoint(checkpoint_path)
        restore_event_backed_budgets(trainer, ledger=ledger, run_id=run_id)
        prior_trainer = None
        if prior_checkpoint is not None:
            prior_network = PolicyValueNet.from_game_spec(
                factory().spec, az_config, device=device
            )
            prior_trainer = AlphaZeroTrainer(prior_network, az_config)
            prior_trainer.load_checkpoint(prior_checkpoint)
        candidate: object = _AlphaZeroEvaluationPolicy(
            trainer, az_config, seed=0, prior_trainer=prior_trainer
        )
        checkpoint_index = trainer.generation
    elif algorithm == "ppo":
        ppo_config = PPOConfig(**algorithm_config)
        trainer = PPOTrainer(factory, ppo_config, ledger=ledger, run_id=run_id)
        trainer.load_checkpoint(checkpoint_path)
        restore_event_backed_budgets(trainer, ledger=ledger, run_id=run_id)
        prior_trainer = None
        if prior_checkpoint is not None:
            prior_trainer = PPOTrainer(factory, ppo_config, seed=0)
            prior_trainer.load_checkpoint(prior_checkpoint)
        candidate = _PPOEvaluationPolicy(trainer, prior_trainer=prior_trainer)
        checkpoint_index = trainer.iteration
    else:
        raise ValueError(f"unknown algorithm in run manifest: {algorithm}")

    chosen_seeds = seeds or tuple(int(seed) for seed in canonical["evaluation"]["seeds"])
    checkpoint_hash = sha256_file(checkpoint_path)
    agents: dict[str, object] = {"learner": candidate}
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

    computed_case_set_hash = case_set_hash(cases)

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
                    "case_set_hash": computed_case_set_hash,
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
    metrics_availability: dict[str, dict[str, object]] = {}
    if prior_checkpoint is None:
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
                    "case_set_hash": computed_case_set_hash,
                    "checkpoint_index": checkpoint_index,
                    "prior_checkpoint": str(prior_checkpoint.relative_to(run_dir)),
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

    prior_occupancy = prior_complete_evaluation_states(
        ledger,
        run_id=run_id,
        checkpoint_index=checkpoint_index,
        case_set_hash=computed_case_set_hash,
    )
    current_states = evaluation_state_ids(
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
                    "case_set_hash": computed_case_set_hash,
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
    elo_payload: dict[str, object] | None = None
    if opponents and valid_outcomes:
        ratings = fit_anchored_elo(valid_outcomes, anchor=opponents[0])
        elo_payload = {
            "rating": ratings.ratings.get("learner"),
            "uncertainty": ratings.uncertainties.get("learner"),
            "anchor": ratings.anchor,
        }
    learning_gpu_hours = prior_gpu_hours["learning"]
    evaluation_gpu_hours = sum_optional_hours(
        prior_gpu_hours["evaluation"],
        resource_totals["evaluation"].allocated_gpu_hours,
    )
    total_gpu_hours = sum_optional_hours(learning_gpu_hours, evaluation_gpu_hours)
    ledger.append(
        Event(
            event_type="evaluation_finished",
            run_id=run_id,
            stage="evaluation",
            payload={
                "checkpoint_index": checkpoint_index,
                "evaluation_id": evaluation_id,
                "case_set_hash": computed_case_set_hash,
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
    print_json(
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
