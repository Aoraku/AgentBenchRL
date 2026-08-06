"""Training-only side-swapped evaluation between learned checkpoints."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from rlbench.algorithms.alphazero import (
    AlphaZeroConfig,
    AlphaZeroTrainer,
    PolicyValueNet,
)
from rlbench.evaluation import EvaluationRunner, build_side_swapped_cases
from rlbench.metrics import fit_anchored_elo, win_rate_summary
from rlbench.registry import game_factory
from rlbench.telemetry import Event, EventLedger


def evaluate_checkpoint_league(
    *,
    run_dir: str | Path,
    candidate_checkpoint: str | Path,
    opponents: Mapping[str, str | Path],
    seeds: Sequence[int],
    output_ledger: str | Path,
    evaluation_split: str,
) -> dict[str, Any]:
    """Evaluate one lineage head against frozen learned history checkpoints."""
    from rlbench.cli.main import (
        _AlphaZeroEvaluationPolicy,
        _alphazero_device,
        _load_manifest,
        _sha256_file,
        _validate_checkpoint_lineage,
    )

    if evaluation_split != "training":
        raise ValueError("checkpoint league evaluation is training-only")
    if not opponents:
        raise ValueError("checkpoint league requires at least one opponent")
    if not seeds:
        raise ValueError("checkpoint league requires at least one seed")
    resolved_run = Path(run_dir).resolve()
    candidate_path = Path(candidate_checkpoint).resolve()
    manifest = _load_manifest(resolved_run / "run_manifest.json")
    if manifest["algorithm"] != "alphazero" or manifest["game"] != "snakego":
        raise ValueError("checkpoint league requires a SnakeGo AlphaZero run")
    source_ledger = EventLedger(resolved_run / "events.jsonl")
    candidate_lineage = _validate_checkpoint_lineage(
        resolved_run,
        candidate_path,
        manifest=manifest,
        ledger=source_ledger,
        require_head=True,
    )
    opponent_lineages = {
        opponent_id: _validate_checkpoint_lineage(
            resolved_run,
            Path(path).resolve(),
            manifest=manifest,
            ledger=source_ledger,
        )
        for opponent_id, path in opponents.items()
    }

    canonical = manifest["canonical_config"]
    config = AlphaZeroConfig(**dict(canonical["algorithm"]))
    factory = game_factory("snakego", dict(canonical["game"]))
    device = _alphazero_device(config)

    def policy(path: Path, *, seed: int):
        network = PolicyValueNet.from_game_spec(factory().spec, config, device=device)
        trainer = AlphaZeroTrainer(network, config)
        trainer.load_checkpoint(path)
        return _AlphaZeroEvaluationPolicy(trainer, config, seed=seed)

    candidate_index = int(candidate_lineage["checkpoint_index"])
    candidate_id = f"checkpoint-{candidate_index}"
    candidate_hash = _sha256_file(candidate_path)
    agents: dict[str, Any] = {candidate_id: policy(candidate_path, seed=0)}
    cases = []
    limits = {}
    move_seconds = canonical["evaluation"].get("move_seconds")
    if move_seconds is not None:
        limits["move_seconds"] = move_seconds
    opponent_hashes: dict[str, str] = {}
    for offset, (opponent_id, path) in enumerate(sorted(opponents.items()), start=1):
        opponent_path = Path(path).resolve()
        opponent_hash = str(opponent_lineages[opponent_id]["checkpoint_hash"])
        opponent_hashes[opponent_id] = opponent_hash
        agents[opponent_id] = policy(opponent_path, seed=offset)
        cases.extend(
            build_side_swapped_cases(
                candidate_id=candidate_id,
                candidate_hash=candidate_hash,
                opponent_id=opponent_id,
                opponent_hash=opponent_hash,
                seeds=seeds,
                game_config=dict(canonical["game"]),
                limits=limits,
                protocol_version="checkpoint-league-v1",
            )
        )
    case_set_hash = _hash_json(sorted(case.content_hash for case in cases))
    population_hash = _hash_json(opponent_hashes)
    ledger = EventLedger(output_ledger)
    evaluation_id = str(uuid4())
    started = time.monotonic()
    report = EvaluationRunner(lambda value: factory(), ledger).run(
        cases,
        agents=agents,
        run_id=str(manifest["run_id"]),
        evaluation_id=evaluation_id,
    )
    wall_seconds = time.monotonic() - started
    env_steps = sum(len(result.actions) for result in report.results)
    completed_mcts_simulations = sum(
        int(getattr(agent, "completed_simulations", 0)) for agent in agents.values()
    )
    for result in report.results:
        ledger.append(
            Event(
                "match_finished",
                str(manifest["run_id"]),
                stage="evaluation",
                payload={
                    "evaluation_id": evaluation_id,
                    "evaluation_split": evaluation_split,
                    "checkpoint_index": candidate_index,
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
    valid = tuple(outcome for outcome in report.outcomes if outcome.valid)
    score = win_rate_summary(valid, candidate_id)
    ratings = fit_anchored_elo(valid, anchor=sorted(opponents)[0])
    ledger.append(
        Event(
            "evaluation_finished",
            str(manifest["run_id"]),
            stage="evaluation",
            payload={
                "evaluation_id": evaluation_id,
                "evaluation_split": evaluation_split,
                "checkpoint_index": candidate_index,
                "candidate_id": candidate_id,
                "candidate_checkpoint_hash": candidate_hash,
                "opponent_checkpoint_hashes": opponent_hashes,
                "population_hash": population_hash,
                "case_set_hash": case_set_hash,
                "complete": report.complete,
                "valid_games": score.valid_games,
                "raw_matches": len(report.results),
                "env_steps": env_steps,
                "wall_seconds": wall_seconds,
                "completed_mcts_simulations": completed_mcts_simulations,
                "score": score.score,
                "wilson_lower": score.wilson_lower,
                "wilson_upper": score.wilson_upper,
                "ratings": dict(ratings.ratings),
                "rating_uncertainties": dict(ratings.uncertainties),
                "heldout_used_for_selection": False,
            },
        )
    )
    return {
        "evaluation_id": evaluation_id,
        "candidate_id": candidate_id,
        "candidate_checkpoint_hash": candidate_hash,
        "opponent_checkpoint_hashes": opponent_hashes,
        "case_set_hash": case_set_hash,
        "complete": report.complete,
        "valid_games": score.valid_games,
        "env_steps": env_steps,
        "wall_seconds": wall_seconds,
        "completed_mcts_simulations": completed_mcts_simulations,
        "score": score.score,
        "heldout_used_for_selection": False,
    }


def _hash_json(value: Any) -> str:
    encoded = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"
