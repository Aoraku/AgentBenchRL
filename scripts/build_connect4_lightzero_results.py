#!/usr/bin/env python3
"""Build auditable Connect4 LightZero Policy Elo snapshot artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rlbench.metrics import MatchOutcome, fit_anchored_elo, win_rate_summary  # noqa: E402


ANCHOR_ID = "lightzero_connect4_rulebot_v1"
ANCHOR_ELO = 1000.0
INITIAL_POLICY_ELO = 1000.0
ELO_SYSTEM = "agentbenchrl-sop-initialized-1000-anchored-bradley-terry-l2-0.01-v2"
FIRST_MEASURED_ITERATION = 10_000


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _score(value: float) -> float:
    if value == 1.0:
        return 1.0
    if value == 0.0 or value == -0.0:
        return 0.5
    if value == -1.0:
        return 0.0
    raise ValueError(f"unsupported Connect4 return: {value!r}")


def _outcomes(policy_id: str, evaluations: Iterable[dict[str, Any]]) -> list[MatchOutcome]:
    results: list[MatchOutcome] = []
    for evaluation in evaluations:
        for value in evaluation["returns"]:
            results.append(MatchOutcome(policy_id, ANCHOR_ID, _score(float(value))))
    return results


def _measurements_batch(
    policy_evaluations: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    outcomes_by_policy = {
        policy_id: _outcomes(policy_id, evaluations)
        for policy_id, evaluations in policy_evaluations.items()
    }
    all_outcomes = [
        outcome
        for policy_id in sorted(outcomes_by_policy)
        for outcome in outcomes_by_policy[policy_id]
    ]
    ratings = fit_anchored_elo(all_outcomes, anchor=ANCHOR_ID)
    measured = {}
    for policy_id, outcomes in outcomes_by_policy.items():
        winrate = win_rate_summary(outcomes, policy_id)
        measured[policy_id] = {
            "elo": float(ratings.ratings[policy_id]),
            "elo_uncertainty": float(ratings.uncertainties[policy_id]),
            "valid_games": int(winrate.valid_games),
            "wins": int(winrate.wins),
            "draws": int(winrate.draws),
            "losses": int(winrate.losses),
            "score_rate": float(winrate.score),
            "wilson_95": [float(winrate.wilson_lower), float(winrate.wilson_upper)],
        }
    return measured


def _validate_source(source: dict[str, Any]) -> None:
    if source.get("schema_version") != 1:
        raise ValueError("source snapshot schema_version must be 1")
    horizon = source.get("common_horizon_iteration")
    if (
        isinstance(horizon, bool)
        or not isinstance(horizon, int)
        or horizon < FIRST_MEASURED_ITERATION
        or horizon % 10_000 != 0
    ):
        raise ValueError(
            "common_horizon_iteration must be at least 10000 and a multiple of 10000"
        )
    seeds = source.get("seeds")
    if not isinstance(seeds, list) or [seed.get("seed") for seed in seeds] != [0, 1, 2, 3]:
        raise ValueError("source snapshot must contain ordered seeds 0, 1, 2, 3")
    expected_iterations = list(range(0, horizon + 1, 10_000))
    for seed in seeds:
        checkpoints = seed.get("checkpoints")
        if not isinstance(checkpoints, list):
            raise ValueError(f"seed {seed['seed']} checkpoints must be a list")
        iterations = [int(item["training_iteration"]) for item in checkpoints]
        if iterations != expected_iterations:
            raise ValueError(f"seed {seed['seed']} has a discontinuous checkpoint sequence")
        previous_games = previous_steps = -1
        for checkpoint in checkpoints:
            games = int(checkpoint["self_play_games"])
            steps = int(checkpoint["env_steps"])
            if games <= previous_games or steps <= previous_steps:
                raise ValueError(f"seed {seed['seed']} budgets must increase strictly")
            previous_games, previous_steps = games, steps
            iteration = int(checkpoint["training_iteration"])
            evaluation = checkpoint["rulebot_eval"]
            if iteration == 0:
                if evaluation is not None:
                    raise ValueError("iteration 0 must remain unmeasured")
            else:
                if evaluation is None or int(evaluation["games"]) != 5:
                    raise ValueError(f"seed {seed['seed']} iteration {iteration} needs five games")
                if len(evaluation["returns"]) != 5:
                    raise ValueError("RuleBot return count must match games")
            transition = checkpoint["transition_from_previous"]
            if iteration == 0 and transition is not None:
                raise ValueError("iteration 0 cannot have transition metadata")
            if iteration > 0:
                mean_ig = float(transition["information_gain"]["mean"])
                if not math.isfinite(mean_ig) or mean_ig < 0.0:
                    raise ValueError("Information Gain must be finite and non-negative")


def _indexed(source: dict[str, Any]) -> dict[int, dict[int, dict[str, Any]]]:
    return {
        int(seed["seed"]): {
            int(checkpoint["training_iteration"]): checkpoint
            for checkpoint in seed["checkpoints"]
        }
        for seed in source["seeds"]
    }


def _seed_measurements(
    source: dict[str, Any],
) -> dict[int, dict[int, dict[str, Any]]]:
    measurements: dict[int, dict[int, dict[str, Any]]] = {}
    for seed in source["seeds"]:
        seed_id = int(seed["seed"])
        measurements[seed_id] = {}
        policy_evaluations = {
            str(checkpoint["policy_id"]): [checkpoint["rulebot_eval"]]
            for checkpoint in seed["checkpoints"]
            if checkpoint["rulebot_eval"] is not None
        }
        measured_by_policy = _measurements_batch(policy_evaluations)
        for checkpoint in seed["checkpoints"]:
            iteration = int(checkpoint["training_iteration"])
            if checkpoint["rulebot_eval"] is None:
                continue
            policy_id = str(checkpoint["policy_id"])
            measurements[seed_id][iteration] = measured_by_policy[policy_id]
    return measurements


def _pooled_measurements(
    source: dict[str, Any], indexed: dict[int, dict[int, dict[str, Any]]]
) -> dict[int, dict[str, Any]]:
    evaluations_by_policy: dict[str, list[dict[str, Any]]] = {}
    for iteration in range(
        FIRST_MEASURED_ITERATION,
        int(source["common_horizon_iteration"]) + 1,
        10_000,
    ):
        policy_id = f"pooled_iter{iteration:06d}"
        evaluations_by_policy[policy_id] = [
            indexed[seed][iteration]["rulebot_eval"] for seed in range(4)
        ]
    measured_by_policy = _measurements_batch(evaluations_by_policy)
    return {
        iteration: measured_by_policy[f"pooled_iter{iteration:06d}"]
        for iteration in range(
            FIRST_MEASURED_ITERATION,
            int(source["common_horizon_iteration"]) + 1,
            10_000,
        )
    }


def _win_rate_metadata(measurement: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": "lightzero-online-rulebot-v1",
        "opponent": "LightZero Connect4RuleBot",
        "agent_seat": "first_player_only",
        "mcts_simulations": 50,
        "games": measurement["valid_games"],
        "wins": measurement["wins"],
        "draws": measurement["draws"],
        "losses": measurement["losses"],
        "score_rate": measurement["score_rate"],
        "wilson_95": measurement["wilson_95"],
    }


def _sop_header(source: dict[str, Any], *, run_id: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "status": source["status"],
        "elo_system": ELO_SYSTEM,
        "elo_protocol": {
            "anchor_policy": ANCHOR_ID,
            "anchor_elo": ANCHOR_ELO,
            "initial_policy_rating": {
                "training_iteration": 0,
                "elo": INITIAL_POLICY_ELO,
                "origin": "initialized_by_sop",
                "rulebot_evaluated": False,
            },
            "fit": "batch Bradley-Terry, draws score 0.5, L2=0.01",
            "uncertainty": "inverse observed Hessian, one standard deviation",
            "opponent_identity": "static",
            "opponent_action_selection": "stochastic fallback",
            "limitations": [
                "five games per checkpoint per seed",
                "learner is first player only",
                "RuleBot is a heuristic stochastic opponent",
                "training was still running at snapshot time",
            ],
        },
        "plot_contract": {
            "x_axis": "cumulative training self-play games_seen",
            "y_axis": "Elo only",
            "information_gain_and_win_rate": "machine-readable metadata only",
        },
    }


def _seed_sop(
    source: dict[str, Any],
    seed: int,
    checkpoints: dict[int, dict[str, Any]],
    measurements: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    payload = _sop_header(source, run_id=f"connect4-lightzero-alphazero-seed{seed}-iter080000")
    iterations = sorted(checkpoints)
    initial_games = int(checkpoints[iterations[0]]["self_play_games"])
    endpoint_games = int(checkpoints[iterations[-1]]["self_play_games"])
    payload["seed"] = seed
    payload["games_seen_accounting"] = {
        "trajectory_origin": "iteration 0 policy checkpoint",
        "self_play_games_before_initial_policy": initial_games,
        "cumulative_games_seen_at_endpoint": endpoint_games - initial_games,
        "absolute_self_play_games_at_endpoint": endpoint_games,
    }
    payload["policy_sources"] = {
        checkpoints[iteration]["policy_id"]: {
            "training_iteration": iteration,
            "env_steps": checkpoints[iteration]["env_steps"],
            "self_play_games": checkpoints[iteration]["self_play_games"],
            "checkpoint_sha256": checkpoints[iteration]["checkpoint_sha256"],
        }
        for iteration in iterations
    }
    rounds = []
    for round_id, (current_iteration, next_iteration) in enumerate(
        zip(iterations, iterations[1:], strict=False), start=1
    ):
        current = checkpoints[current_iteration]
        nxt = checkpoints[next_iteration]
        current_id = str(current["policy_id"])
        next_id = str(nxt["policy_id"])
        current_measurement = measurements.get(current_iteration)
        next_measurement = measurements[next_iteration]
        transition = nxt["transition_from_previous"]
        rounds.append(
            {
                "round": round_id,
                "policy_cur": current_id,
                "policy_nxt": next_id,
                "games_seen": int(nxt["self_play_games"]) - int(current["self_play_games"]),
                "elo": {
                    current_id: INITIAL_POLICY_ELO
                    if current_measurement is None
                    else current_measurement["elo"],
                    next_id: next_measurement["elo"],
                },
                "elo_uncertainty": {
                    current_id: None
                    if current_measurement is None
                    else current_measurement["elo_uncertainty"],
                    next_id: next_measurement["elo_uncertainty"],
                },
                "win_rate": _win_rate_metadata(next_measurement),
                "information_gain": transition["information_gain"],
                "budget": {
                    "policy_cur_env_steps": current["env_steps"],
                    "policy_nxt_env_steps": nxt["env_steps"],
                    "policy_cur_self_play_games": current["self_play_games"],
                    "policy_nxt_self_play_games": nxt["self_play_games"],
                    "training_iteration": next_iteration,
                },
            }
        )
    payload["rounds"] = rounds
    return payload


def _aggregate_ig(
    indexed: dict[int, dict[int, dict[str, Any]]], iteration: int
) -> dict[str, Any]:
    values = {
        f"seed{seed}": float(
            indexed[seed][iteration]["transition_from_previous"]["information_gain"]["mean"]
        )
        for seed in range(4)
    }
    samples = list(values.values())
    return {
        "version": "masked-policy-kl-v1-four-seed-aggregate",
        "unit": "nats",
        "direction": "D_KL(policy_nxt || policy_cur)",
        "mean_across_seeds": float(statistics.fmean(samples)),
        "std_across_seeds": float(statistics.pstdev(samples)),
        "per_seed_mean": values,
        "probe_count_per_seed": 512,
    }


def _pooled_sop(
    source: dict[str, Any],
    indexed: dict[int, dict[int, dict[str, Any]]],
    measurements: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    payload = _sop_header(source, run_id="connect4-lightzero-alphazero-four-seed-pooled-iter080000")
    payload["aggregate_semantics"] = {
        "policy": "same-iteration policies pooled across seeds 0, 1, 2, 3",
        "games_seen": "sum of incremental completed self-play games between adjacent policies across all four seeds",
        "evaluation_games": "sum of the four five-game RuleBot evaluations",
    }
    iterations = sorted(indexed[0])
    initial_games = sum(indexed[seed][iterations[0]]["self_play_games"] for seed in range(4))
    endpoint_games = sum(
        indexed[seed][iterations[-1]]["self_play_games"] for seed in range(4)
    )
    payload["games_seen_accounting"] = {
        "trajectory_origin": "iteration 0 policy checkpoint",
        "self_play_games_before_initial_policy": int(initial_games),
        "cumulative_games_seen_at_endpoint": int(endpoint_games - initial_games),
        "absolute_self_play_games_at_endpoint": int(endpoint_games),
    }
    payload["policy_sources"] = {
        f"pooled_iter{iteration:06d}": {
            "training_iteration": iteration,
            "member_policy_ids": [indexed[seed][iteration]["policy_id"] for seed in range(4)],
            "member_checkpoint_sha256": [
                indexed[seed][iteration]["checkpoint_sha256"] for seed in range(4)
            ],
            "env_steps_sum": sum(indexed[seed][iteration]["env_steps"] for seed in range(4)),
            "self_play_games_sum": sum(
                indexed[seed][iteration]["self_play_games"] for seed in range(4)
            ),
        }
        for iteration in iterations
    }
    rounds = []
    for round_id, (current_iteration, next_iteration) in enumerate(
        zip(iterations, iterations[1:], strict=False), start=1
    ):
        current_id = f"pooled_iter{current_iteration:06d}"
        next_id = f"pooled_iter{next_iteration:06d}"
        current_measurement = measurements.get(current_iteration)
        next_measurement = measurements[next_iteration]
        current_games = sum(indexed[seed][current_iteration]["self_play_games"] for seed in range(4))
        next_games = sum(indexed[seed][next_iteration]["self_play_games"] for seed in range(4))
        rounds.append(
            {
                "round": round_id,
                "policy_cur": current_id,
                "policy_nxt": next_id,
                "games_seen": int(next_games - current_games),
                "elo": {
                    current_id: INITIAL_POLICY_ELO
                    if current_measurement is None
                    else current_measurement["elo"],
                    next_id: next_measurement["elo"],
                },
                "elo_uncertainty": {
                    current_id: None
                    if current_measurement is None
                    else current_measurement["elo_uncertainty"],
                    next_id: next_measurement["elo_uncertainty"],
                },
                "win_rate": _win_rate_metadata(next_measurement),
                "information_gain": _aggregate_ig(indexed, next_iteration),
                "budget": {
                    "policy_cur_env_steps_sum": sum(
                        indexed[seed][current_iteration]["env_steps"] for seed in range(4)
                    ),
                    "policy_nxt_env_steps_sum": sum(
                        indexed[seed][next_iteration]["env_steps"] for seed in range(4)
                    ),
                    "policy_cur_self_play_games_sum": current_games,
                    "policy_nxt_self_play_games_sum": next_games,
                    "training_iteration": next_iteration,
                },
            }
        )
    payload["rounds"] = rounds
    return payload


def _metric_rows(
    source: dict[str, Any],
    measurements: dict[int, dict[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for seed in source["seeds"]:
        seed_id = int(seed["seed"])
        for checkpoint in seed["checkpoints"]:
            iteration = int(checkpoint["training_iteration"])
            measured = measurements[seed_id].get(iteration)
            transition = checkpoint["transition_from_previous"]
            rows.append(
                {
                    "schema_version": 1,
                    "seed": seed_id,
                    "policy_id": checkpoint["policy_id"],
                    "training_iteration": iteration,
                    "env_steps": checkpoint["env_steps"],
                    "self_play_games": checkpoint["self_play_games"],
                    "wall_time_sec": checkpoint["wall_time_sec"],
                    "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                    "elo": None if measured is None else measured["elo"],
                    "elo_uncertainty": None if measured is None else measured["elo_uncertainty"],
                    "win_rate": None if measured is None else _win_rate_metadata(measured),
                    "information_gain": None
                    if transition is None
                    else transition["information_gain"],
                    "games_seen_from_previous": None
                    if transition is None
                    else transition["games_seen"],
                    "env_steps_seen_from_previous": None
                    if transition is None
                    else transition["env_steps_seen"],
                }
            )
    return rows


def _curve_rows(
    indexed: dict[int, dict[int, dict[str, Any]]],
    seed_measurements: dict[int, dict[int, dict[str, Any]]],
    pooled_measurements: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for seed in range(4):
        iterations = sorted(indexed[seed])
        initial_games = int(indexed[seed][iterations[0]]["self_play_games"])
        for iteration in iterations:
            checkpoint = indexed[seed][iteration]
            measured = seed_measurements[seed].get(iteration)
            rows.append(
                {
                    "trajectory": f"seed{seed}",
                    "seed": seed,
                    "training_iteration": iteration,
                    "cumulative_games_seen": int(checkpoint["self_play_games"]) - initial_games,
                    "elo": INITIAL_POLICY_ELO if measured is None else measured["elo"],
                    "elo_uncertainty": None
                    if measured is None
                    else measured["elo_uncertainty"],
                    "rating_origin": "initialized_by_sop"
                    if measured is None
                    else "rulebot_evaluation",
                    "env_steps": checkpoint["env_steps"],
                    "self_play_games": checkpoint["self_play_games"],
                }
            )
    iterations = sorted(indexed[0])
    initial_games = sum(indexed[seed][iterations[0]]["self_play_games"] for seed in range(4))
    for iteration in iterations:
        measured = pooled_measurements.get(iteration)
        total_games = sum(indexed[seed][iteration]["self_play_games"] for seed in range(4))
        rows.append(
            {
                "trajectory": "pooled",
                "seed": "all",
                "training_iteration": iteration,
                "cumulative_games_seen": int(total_games - initial_games),
                "elo": INITIAL_POLICY_ELO if measured is None else measured["elo"],
                "elo_uncertainty": None
                if measured is None
                else measured["elo_uncertainty"],
                "rating_origin": "initialized_by_sop"
                if measured is None
                else "rulebot_evaluation",
                "env_steps": sum(indexed[seed][iteration]["env_steps"] for seed in range(4)),
                "self_play_games": total_games,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_policy_elo_figure(
    curve_rows: list[dict[str, Any]],
) -> tuple[plt.Figure, plt.Text, plt.Text]:
    """Create the publication figure and expose title artists for layout tests."""
    pooled = [row for row in curve_rows if row["trajectory"] == "pooled"]
    x = [float(row["cumulative_games_seen"]) for row in pooled]
    y = [float(row["elo"]) for row in pooled]
    uncertainty = [
        math.nan if row["elo_uncertainty"] in (None, "") else float(row["elo_uncertainty"])
        for row in pooled
    ]
    lower = [rating - error for rating, error in zip(y, uncertainty, strict=True)]
    upper = [rating + error for rating, error in zip(y, uncertainty, strict=True)]

    plt.rcParams.update(
        {
            "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 15,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 2.0,
            "legend.frameon": False,
        }
    )
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.fill_between(x, lower, upper, color="#3775BA", alpha=0.18, linewidth=0)
    axis.plot(
        x,
        y,
        color="#0F4D92",
        marker="o",
        markersize=6,
        linewidth=2.5,
        label="Four-seed pooled Elo",
    )
    axis.axhline(ANCHOR_ELO, color="#767676", linewidth=1.2, linestyle="--", label="RuleBot anchor")
    axis.set_xlabel("Cumulative self-play games since initial policy (four-seed total)")
    axis.set_ylabel("Elo")
    title = figure.suptitle(
        "Connect4 AlphaZero policy Elo",
        x=0.10,
        y=0.98,
        ha="left",
        va="top",
        fontsize=20,
        fontweight="bold",
    )
    subtitle = figure.text(
        0.10,
        0.925,
        "Interim 0–80k snapshot; initial policy = Elo 1000; band = one-standard-deviation fit uncertainty",
        color="#4D4D4D",
        fontsize=10.5,
        ha="left",
        va="top",
    )
    axis.grid(axis="y", color="#CFCECE", alpha=0.45, linewidth=0.8)
    axis.legend(loc="best")
    figure.subplots_adjust(left=0.11, right=0.97, bottom=0.16, top=0.76)
    return figure, title, subtitle


def _render_plot(path: Path, curve_rows: list[dict[str, Any]]) -> None:
    figure, _, _ = make_policy_elo_figure(curve_rows)
    figure.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "AgentBenchRL"},
    )
    plt.close(figure)


def _summary(
    source: dict[str, Any],
    indexed: dict[int, dict[int, dict[str, Any]]],
    seed_measurements: dict[int, dict[int, dict[str, Any]]],
    pooled_measurements: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    horizon = int(source["common_horizon_iteration"])
    initial_games = sum(indexed[seed][0]["self_play_games"] for seed in range(4))
    pooled_history = [
        {
            "training_iteration": 0,
            "cumulative_games_seen": 0,
            "rating": INITIAL_POLICY_ELO,
            "uncertainty": None,
            "rating_origin": "initialized_by_sop",
        }
    ]
    for iteration in sorted(pooled_measurements):
        total_games = sum(indexed[seed][iteration]["self_play_games"] for seed in range(4))
        measured = pooled_measurements[iteration]
        pooled_history.append(
            {
                "training_iteration": iteration,
                "cumulative_games_seen": int(total_games - initial_games),
                "rating": measured["elo"],
                "uncertainty": measured["elo_uncertainty"],
                "rating_origin": "rulebot_evaluation",
            }
        )
    final = pooled_measurements[horizon]
    ig_values = [
        float(indexed[seed][iteration]["transition_from_previous"]["information_gain"]["mean"])
        for seed in range(4)
        for iteration in range(10_000, horizon + 1, 10_000)
    ]
    return {
        "schema_version": 1,
        "run_type": "train",
        "game": "connect4",
        "agent": "lightzero-alphazero",
        "created": source["created_at"],
        "status": source["status"],
        "training_complete": False,
        "seed_count": 4,
        "common_horizon_iteration": horizon,
        "best_elo": max(item["elo"] for item in pooled_measurements.values()),
        "final_elo": final["elo"],
        "final_elo_uncertainty": final["elo_uncertainty"],
        "final_score_rate": final["score_rate"],
        "total_env_steps": sum(indexed[seed][horizon]["env_steps"] for seed in range(4)),
        "total_self_play_games": sum(
            indexed[seed][horizon]["self_play_games"] for seed in range(4)
        ),
        "elo_history": pooled_history,
        "per_seed_final": {
            f"seed{seed}": {
                "elo": seed_measurements[seed][horizon]["elo"],
                "elo_uncertainty": seed_measurements[seed][horizon]["elo_uncertainty"],
                "env_steps": indexed[seed][horizon]["env_steps"],
                "self_play_games": indexed[seed][horizon]["self_play_games"],
            }
            for seed in range(4)
        },
        "evaluation": {
            **source["evaluation_contract"],
            "elo_system": ELO_SYSTEM,
            "pooled_games_per_checkpoint": 20,
        },
        "information_gain": {
            **source["information_gain_contract"],
            "transition_measurement_count": len(ig_values),
            "mean_nats": float(statistics.fmean(ig_values)),
            "std_nats": float(statistics.pstdev(ig_values)),
        },
    }


def _provenance(source: dict[str, Any]) -> dict[str, Any]:
    assets = []
    for seed in source["seeds"]:
        for checkpoint in seed["checkpoints"]:
            assets.append(
                {
                    "seed": seed["seed"],
                    "policy_id": checkpoint["policy_id"],
                    "training_iteration": checkpoint["training_iteration"],
                    "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                }
            )
    return {
        "schema_version": 1,
        "snapshot_id": source["snapshot_id"],
        "created_at": source["created_at"],
        "source_host": source["source_host"],
        "lightzero_commit": source["lightzero_commit"],
        "official_config_sha256": source["official_config_sha256"],
        "common_horizon_iteration": source["common_horizon_iteration"],
        "checkpoint_assets": assets,
        "generator": "scripts/build_connect4_lightzero_results.py",
        "known_limitations": [
            "This is an interim snapshot; all four training processes were still running.",
            "Iteration-0 Elo 1000 is the SOP initialization, not a RuleBot measurement; its uncertainty is null.",
            "Official LightZero evaluation supplies only five games per checkpoint per seed.",
            "The learner occupies the first-player seat only in these evaluations.",
            "The RuleBot identity is fixed but its fallback action choice is stochastic.",
            "Pooled Elo combines independent same-iteration policies across four seeds.",
            "No checkpoint binary, raw log, TensorBoard event, or machine path is published.",
        ],
    }


def build_results(source_path: Path, output_dir: Path) -> None:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    _validate_source(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    indexed = _indexed(source)
    seed_measurements = _seed_measurements(source)
    pooled_measurements = _pooled_measurements(source, indexed)

    for seed in range(4):
        _write_json(
            output_dir / f"policy_elo_seed{seed}.json",
            _seed_sop(source, seed, indexed[seed], seed_measurements[seed]),
        )
    _write_json(
        output_dir / "policy_elo_pooled.json",
        _pooled_sop(source, indexed, pooled_measurements),
    )

    metric_rows = _metric_rows(source, seed_measurements)
    (output_dir / "checkpoint_metrics.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in metric_rows),
        encoding="utf-8",
    )
    curve_rows = _curve_rows(indexed, seed_measurements, pooled_measurements)
    _write_csv(output_dir / "elo_curve.csv", curve_rows)
    _write_json(
        output_dir / "summary.json",
        _summary(source, indexed, seed_measurements, pooled_measurements),
    )
    _write_json(output_dir / "provenance.json", _provenance(source))
    _render_plot(output_dir / "policy_elo_curve.png", curve_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build_results(args.source, args.output_dir)
    print(args.output_dir)


if __name__ == "__main__":
    main()
