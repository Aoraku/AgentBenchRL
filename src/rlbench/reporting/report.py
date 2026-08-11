"""Machine-readable tables and explicit-axis curves from JSONL event facts."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from rlbench.metrics import (
    MatchOutcome,
    build_curve,
    fit_anchored_elo,
    trapezoid_auc,
    win_rate_summary,
)
from rlbench.telemetry import Event, EventLedger


def generate_report(run_directory: str | Path) -> Path:
    """Rebuild all report artifacts from the run's append-only facts."""
    run_dir = Path(run_directory).resolve()
    event_path = run_dir / "events.jsonl"
    if not event_path.is_file():
        raise ValueError(f"run directory has no events.jsonl: {run_dir}")
    events = tuple(EventLedger(event_path).read())
    if not events:
        raise ValueError("events.jsonl contains no facts")
    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = _checkpoint_rows(events)
    (
        match_rows,
        outcomes_by_checkpoint,
        evaluation_availability,
        selected_evaluations,
    ) = _match_rows(events)
    learner = _learner_id(events, match_rows)
    win_rows = _win_rate_rows(outcomes_by_checkpoint, learner)
    elo_rows = _elo_rows(outcomes_by_checkpoint, learner)
    ig_rows = _metric_rows(
        events,
        event_type="policy_ig_measured",
        fields=("nats_per_decision", "nats_per_episode"),
        selected_evaluations=selected_evaluations,
    )
    occupancy_rows = _metric_rows(
        events,
        event_type="occupancy_measured",
        fields=("occupancy_shift",),
        selected_evaluations=selected_evaluations,
    )
    resource_rows = _resource_rows(events)
    auc_rows, auc_availability = _auc_rows(
        checkpoints, elo_rows=elo_rows, win_rows=win_rows, learner=learner
    )

    _write_csv(
        report_dir / "elo.csv",
        elo_rows,
        ("checkpoint_index", "player", "rating", "uncertainty", "anchor", "valid_games"),
    )
    _write_csv(
        report_dir / "win_rate.csv",
        win_rows,
        (
            "checkpoint_index",
            "player",
            "wins",
            "draws",
            "losses",
            "valid_games",
            "score",
            "wilson_lower",
            "wilson_upper",
        ),
    )
    _write_csv(
        report_dir / "information_gain.csv",
        ig_rows,
        ("checkpoint_index", "nats_per_decision", "nats_per_episode"),
    )
    _write_csv(
        report_dir / "occupancy.csv",
        occupancy_rows,
        ("checkpoint_index", "occupancy_shift"),
    )
    _write_csv(
        report_dir / "budgets.csv",
        checkpoints,
        (
            "checkpoint_index",
            "env_steps",
            "optimizer_steps",
            "mcts_simulations",
            "learning_wall_seconds",
            "evaluation_wall_seconds",
            "wall_seconds",
            "learning_gpu_hours",
            "evaluation_gpu_hours",
            "gpu_hours",
        ),
    )
    _write_csv(
        report_dir / "resources.csv",
        resource_rows,
        (
            "created_at",
            "stage",
            "gpu_count",
            "gpu_utilization_percent",
            "cpu_count",
            "cpu_utilization_percent",
            "process_ram_bytes",
            "host_ram_bytes",
        ),
    )
    _write_csv(
        report_dir / "matches.csv",
        match_rows,
        (
            "evaluation_id",
            "checkpoint_index",
            "case_id",
            "case_hash",
            "seed",
            "player_0",
            "player_1",
            "score_player_0",
            "valid",
            "reason",
        ),
    )
    _write_csv(
        report_dir / "auc.csv",
        auc_rows,
        ("name", "x_axis", "y_axis", "value", "point_count"),
    )

    availability = {
        "elo": _availability(bool(elo_rows), "no connected raw match graph"),
        "information_gain": _availability(
            bool(ig_rows), "no policy_ig_measured facts"
        ),
        "occupancy": _availability(
            bool(occupancy_rows), "no occupancy_measured facts"
        ),
        "gpu_hours": _availability(
            any(row.get("gpu_hours") is not None for row in checkpoints),
            "GPU allocation measurement is unavailable",
        ),
        "resource_utilization": _availability(
            any(
                row.get("gpu_utilization_percent") is not None
                or row.get("cpu_utilization_percent") is not None
                for row in resource_rows
            ),
            "no utilization samples",
        ),
        "win_rate": _availability(
            bool(win_rows), "no complete evaluation invocation"
        ),
    }
    summary = {
        "schema_version": 1,
        "run_id": events[0].run_id,
        "source": "events.jsonl",
        "event_count": len(events),
        "checkpoint_count": len(checkpoints),
        "raw_match_count": len(match_rows),
        "availability": availability,
        "evaluation_availability": {
            str(key): value for key, value in sorted(evaluation_availability.items())
        },
        "auc": {str(row["name"]): row for row in auc_rows},
        "auc_availability": auc_availability,
        "win_rate": {
            str(row["player"]): {
                key: row[key]
                for key in (
                    "wins",
                    "draws",
                    "losses",
                    "valid_games",
                    "score",
                    "wilson_lower",
                    "wilson_upper",
                )
            }
            for row in win_rows[-1:]
        },
        "elo": {
            str(row["player"]): {
                "rating": row["rating"],
                "uncertainty": row["uncertainty"],
                "anchor": row["anchor"],
                "valid_games": row["valid_games"],
            }
            for row in elo_rows
            if row["checkpoint_index"] == max(
                (item["checkpoint_index"] for item in elo_rows), default=-1
            )
        },
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Additive cross-repository export: a summary conforming to the shared
    # metrics-schema so A's reporting can aggregate RL runs alongside HL runs.
    schema_summary = _schema_summary(
        run_dir=run_dir,
        events=events,
        learner=learner,
        checkpoints=checkpoints,
        elo_rows=elo_rows,
        win_rows=win_rows,
        ig_rows=ig_rows,
        outcomes_by_checkpoint=outcomes_by_checkpoint,
    )
    (report_dir / "summary.schema.json").write_text(
        json.dumps(schema_summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    _plots(
        report_dir,
        checkpoints=checkpoints,
        elo_rows=[row for row in elo_rows if row["player"] == learner],
        win_rows=win_rows,
        ig_rows=ig_rows,
        occupancy_rows=occupancy_rows,
        resource_rows=resource_rows,
        availability=availability,
    )
    return report_dir


def _checkpoint_rows(events: Iterable[Event]) -> list[dict[str, Any]]:
    events = tuple(events)
    rows: list[dict[str, Any]] = []
    for event in events:
        if event.event_type != "checkpoint_saved":
            continue
        payload = event.payload
        budgets = payload.get("budgets", {})
        learning = budgets.get("learning", {}) if isinstance(budgets, Mapping) else {}
        evaluation = budgets.get("evaluation", {}) if isinstance(budgets, Mapping) else {}
        total = budgets.get("total", {}) if isinstance(budgets, Mapping) else {}
        rows.append(
            {
                "checkpoint_index": int(payload.get("checkpoint_index", len(rows) + 1)),
                "env_steps": payload.get("env_steps", total.get("env_steps")),
                "optimizer_steps": payload.get(
                    "optimizer_steps", total.get("optimizer_steps")
                ),
                "mcts_simulations": payload.get(
                    "mcts_simulations", total.get("mcts_simulations")
                ),
                "learning_wall_seconds": payload.get(
                    "learning_wall_seconds", learning.get("wall_seconds")
                ),
                "evaluation_wall_seconds": payload.get(
                    "evaluation_wall_seconds", evaluation.get("wall_seconds")
                ),
                "wall_seconds": payload.get("wall_seconds", total.get("wall_seconds")),
                "learning_gpu_hours": payload.get(
                    "learning_gpu_hours", learning.get("gpu_hours")
                ),
                "evaluation_gpu_hours": payload.get(
                    "evaluation_gpu_hours", evaluation.get("gpu_hours")
                ),
                "gpu_hours": payload.get("gpu_hours"),
            }
        )
    by_index = {row["checkpoint_index"]: row for row in rows}
    for event in events:
        if event.event_type != "evaluation_finished":
            continue
        checkpoint_index = int(event.payload.get("checkpoint_index", -1))
        row = by_index.get(checkpoint_index)
        budgets = event.payload.get("budgets")
        if row is None or not isinstance(budgets, Mapping):
            continue
        learning = budgets.get("learning", {})
        evaluation = budgets.get("evaluation", {})
        total = budgets.get("total", {})
        row.update(
            {
                "env_steps": total.get("env_steps", row.get("env_steps")),
                "optimizer_steps": total.get(
                    "optimizer_steps", row.get("optimizer_steps")
                ),
                "mcts_simulations": total.get(
                    "mcts_simulations", row.get("mcts_simulations")
                ),
                "learning_wall_seconds": learning.get(
                    "wall_seconds", row.get("learning_wall_seconds")
                ),
                "evaluation_wall_seconds": evaluation.get(
                    "wall_seconds", row.get("evaluation_wall_seconds")
                ),
                "wall_seconds": total.get("wall_seconds", row.get("wall_seconds")),
                "learning_gpu_hours": event.payload.get(
                    "learning_gpu_hours", row.get("learning_gpu_hours")
                ),
                "evaluation_gpu_hours": event.payload.get("evaluation_gpu_hours"),
                "gpu_hours": event.payload.get("gpu_hours"),
            }
        )
    return sorted(rows, key=lambda row: row["checkpoint_index"])


def _match_rows(
    events: Iterable[Event],
) -> tuple[
    list[dict[str, Any]],
    dict[int, list[MatchOutcome]],
    dict[int, dict[str, Any]],
    dict[int, str],
]:
    events = tuple(events)
    preferred_type = (
        "match_finished"
        if any(event.event_type == "match_finished" for event in events)
        else "evaluation_match"
    )
    rows: list[dict[str, Any]] = []
    rows_by_evaluation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event.event_type != preferred_type:
            continue
        payload = event.payload
        player_0 = payload.get("player_0")
        player_1 = payload.get("player_1")
        if not isinstance(player_0, str) or not isinstance(player_1, str):
            continue
        checkpoint_index = int(payload.get("checkpoint_index", 0))
        score = payload.get("score_player_0")
        valid = bool(payload.get("valid", False))
        row = {
            "checkpoint_index": checkpoint_index,
            "seed": payload.get("seed"),
            "player_0": player_0,
            "player_1": player_1,
            "score_player_0": score,
            "valid": valid,
            "reason": payload.get("reason", ""),
            "evaluation_id": payload.get("evaluation_id"),
            "case_id": payload.get("case_id"),
            "case_hash": payload.get("case_hash"),
        }
        rows.append(row)
        evaluation_id = payload.get("evaluation_id")
        group_id = (
            str(evaluation_id)
            if isinstance(evaluation_id, str) and evaluation_id
            else f"legacy:{checkpoint_index}"
        )
        rows_by_evaluation[group_id].append(row)

    finished: dict[int, list[tuple[str, bool]]] = defaultdict(list)
    for event in events:
        if event.event_type != "evaluation_finished":
            continue
        evaluation_id = event.payload.get("evaluation_id")
        checkpoint_index = event.payload.get("checkpoint_index")
        if not isinstance(evaluation_id, str) or not isinstance(checkpoint_index, int):
            continue
        finished[checkpoint_index].append(
            (evaluation_id, bool(event.payload.get("complete", False)))
        )
    for group_id, group_rows in rows_by_evaluation.items():
        if group_id.startswith("legacy:"):
            checkpoint_index = int(group_id.split(":", 1)[1])
            finished.setdefault(checkpoint_index, []).append((group_id, True))

    grouped: dict[int, list[MatchOutcome]] = {}
    availability: dict[int, dict[str, Any]] = {}
    selected_evaluations: dict[int, str] = {}
    for checkpoint_index, invocations in sorted(finished.items()):
        complete_ids = [evaluation_id for evaluation_id, complete in invocations if complete]
        if not complete_ids:
            availability[checkpoint_index] = _availability(
                False, "incomplete evaluation invocation"
            )
            continue
        selected_id = complete_ids[-1]
        selected_evaluations[checkpoint_index] = selected_id
        selected_rows = rows_by_evaluation.get(selected_id, [])
        deduped: list[dict[str, Any]] = []
        seen_cases: set[tuple[Any, Any]] = set()
        for row_index, row in enumerate(selected_rows):
            identity = (row.get("case_id"), row.get("case_hash"))
            if identity == (None, None):
                identity = ("legacy", row_index)
            if identity in seen_cases:
                continue
            seen_cases.add(identity)
            deduped.append(row)
        grouped[checkpoint_index] = [
            MatchOutcome(
                str(row["player_0"]),
                str(row["player_1"]),
                row["score_player_0"],
                valid=bool(row["valid"]),
            )
            for row in deduped
        ]
        availability[checkpoint_index] = _availability(True, "")
    return rows, grouped, availability, selected_evaluations


def _auc_rows(
    checkpoints: list[Mapping[str, Any]],
    *,
    elo_rows: list[Mapping[str, Any]],
    win_rows: list[Mapping[str, Any]],
    learner: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    checkpoint_lookup = {row["checkpoint_index"]: row for row in checkpoints}
    measurements = {
        "elo": [row for row in elo_rows if row["player"] == learner],
        "win_rate": win_rows,
    }
    results: list[dict[str, Any]] = []
    availability: dict[str, dict[str, Any]] = {}
    for y_axis, source_rows in measurements.items():
        source_field = "rating" if y_axis == "elo" else "score"
        for x_axis in (
            "checkpoint_index",
            "env_steps",
            "wall_seconds",
            "gpu_hours",
        ):
            records = []
            for row in source_rows:
                checkpoint = checkpoint_lookup.get(row["checkpoint_index"], {})
                x_value = (
                    row["checkpoint_index"]
                    if x_axis == "checkpoint_index"
                    else checkpoint.get(x_axis)
                )
                if x_value is not None and row.get(source_field) is not None:
                    records.append({x_axis: x_value, y_axis: row[source_field]})
            name = f"AUC_{y_axis}_vs_{x_axis}"
            if len(records) < 2:
                availability[name] = _availability(
                    False, "at least two measured checkpoints are required"
                )
                continue
            curve = build_curve(records, x_axis=x_axis, y_axis=y_axis)
            area = trapezoid_auc(curve)
            results.append(
                {
                    "name": name,
                    "x_axis": x_axis,
                    "y_axis": y_axis,
                    "value": area.value,
                    "point_count": len(curve.points),
                }
            )
            availability[name] = _availability(True, "")
    return results, availability


def _learner_id(events: Iterable[Event], match_rows: list[Mapping[str, Any]]) -> str:
    for event in events:
        if event.event_type == "run_started":
            value = event.payload.get("candidate_id")
            if isinstance(value, str) and value:
                return value
    players = {
        str(row[key]) for row in match_rows for key in ("player_0", "player_1")
    }
    for preferred in ("learner", "candidate", "checkpoint"):
        if preferred in players:
            return preferred
    return sorted(players)[0] if players else "learner"


def _win_rate_rows(
    grouped: Mapping[int, list[MatchOutcome]], learner: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for checkpoint_index, outcomes in sorted(grouped.items()):
        summary = win_rate_summary(outcomes, learner)
        rows.append({"checkpoint_index": checkpoint_index, **asdict(summary)})
    return rows


def _elo_rows(
    grouped: Mapping[int, list[MatchOutcome]], learner: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for checkpoint_index, outcomes in sorted(grouped.items()):
        opponents = sorted(
            {
                player
                for outcome in outcomes
                for player in (outcome.player_a, outcome.player_b)
                if player != learner
            }
        )
        if not opponents:
            continue
        anchor = opponents[0]
        try:
            ratings = fit_anchored_elo(outcomes, anchor=anchor)
        except ValueError:
            continue
        for player in sorted(ratings.ratings):
            rows.append(
                {
                    "checkpoint_index": checkpoint_index,
                    "player": player,
                    "rating": ratings.ratings[player],
                    "uncertainty": ratings.uncertainties[player],
                    "anchor": anchor,
                    "valid_games": ratings.valid_games,
                }
            )
    return rows


def _metric_rows(
    events: Iterable[Event],
    *,
    event_type: str,
    fields: tuple[str, ...],
    selected_evaluations: Mapping[int, str],
) -> list[dict[str, Any]]:
    rows_by_checkpoint: dict[int, dict[str, Any]] = {}
    for event in events:
        if event.event_type != event_type:
            continue
        if not all(field in event.payload for field in fields):
            continue
        checkpoint_index = int(event.payload.get("checkpoint_index", 0))
        evaluation_id = event.payload.get("evaluation_id")
        group_id = (
            str(evaluation_id)
            if isinstance(evaluation_id, str) and evaluation_id
            else f"legacy:{checkpoint_index}"
        )
        if selected_evaluations.get(checkpoint_index) != group_id:
            continue
        rows_by_checkpoint[checkpoint_index] = {
            "checkpoint_index": checkpoint_index,
            **{field: event.payload[field] for field in fields},
        }
    return [rows_by_checkpoint[index] for index in sorted(rows_by_checkpoint)]


def _resource_rows(events: Iterable[Event]) -> list[dict[str, Any]]:
    rows = []
    for event in events:
        if event.event_type != "resource_sampled":
            continue
        devices = event.payload.get("gpu_devices", [])
        utilization = [
            device.get("utilization_percent")
            for device in devices
            if isinstance(device, Mapping)
            and device.get("utilization_percent") is not None
        ]
        process_ram = event.payload.get("process_ram_bytes", [])
        rows.append(
            {
                "created_at": event.created_at.isoformat() if event.created_at else "",
                "stage": event.stage,
                "gpu_count": len(devices),
                "gpu_utilization_percent": (
                    sum(utilization) / len(utilization) if utilization else None
                ),
                "cpu_count": event.payload.get("cpu_count"),
                "cpu_utilization_percent": event.payload.get(
                    "cpu_utilization_percent"
                ),
                "process_ram_bytes": sum(process_ram) if process_ram else None,
                "host_ram_bytes": event.payload.get("host_ram_bytes"),
            }
        )
    return rows


def _schema_budget(
    checkpoint: Mapping[str, Any] | None,
    *,
    kind: str,
) -> dict[str, Any]:
    """Map RL budgets onto the shared metrics-schema budget object.

    Token dimensions do not apply to RL and are recorded as ``null`` rather
    than 0, per the shared missing-value rule; only ``wall_time_s`` is filled.
    """
    wall_field = {
        "learning": "learning_wall_seconds",
        "evaluation": "evaluation_wall_seconds",
        "total": "wall_seconds",
    }[kind]
    wall_time = None if checkpoint is None else checkpoint.get(wall_field)
    return {
        "input_tokens": None,
        "cached_input_tokens": None,
        "output_tokens": None,
        "reasoning_tokens": None,
        "total_tokens": None,
        "wall_time_s": wall_time,
    }


def _h2h(
    outcomes_by_checkpoint: Mapping[int, list[MatchOutcome]],
    *,
    learner: str,
) -> dict[str, float | None]:
    """Learner win rate against each opponent from the final evaluated checkpoint."""
    if not outcomes_by_checkpoint:
        return {}
    final_index = max(outcomes_by_checkpoint)
    tally: dict[str, list[float]] = defaultdict(list)
    for outcome in outcomes_by_checkpoint[final_index]:
        if not outcome.valid:
            continue
        if outcome.player_a == learner:
            opponent, learner_score = outcome.player_b, outcome.score_a
        elif outcome.player_b == learner:
            opponent, learner_score = outcome.player_a, 1.0 - outcome.score_a
        else:
            continue
        tally[opponent].append(learner_score)
    return {
        opponent: (sum(scores) / len(scores) if scores else None)
        for opponent, scores in sorted(tally.items())
    }


def _git_commit(run_dir: Path) -> str | None:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    software = manifest.get("software") if isinstance(manifest, Mapping) else None
    if isinstance(software, Mapping):
        commit = software.get("git_commit")
        if isinstance(commit, str) and commit:
            return commit
    return None


def _schema_summary(
    *,
    run_dir: Path,
    events: tuple[Event, ...],
    learner: str,
    checkpoints: list[dict[str, Any]],
    elo_rows: list[dict[str, Any]],
    win_rows: list[dict[str, Any]],
    ig_rows: list[dict[str, Any]],
    outcomes_by_checkpoint: Mapping[int, list[MatchOutcome]],
) -> dict[str, Any]:
    """Project the native report onto the shared metrics-schema summary.

    This is an additive export for A's ``reporting/aggregate.py``; it never
    replaces the native ``summary.json``. Any metric the RL runs do not
    measure (RL has no raw/evo behavioural curves, no token budgets, and no
    per-side Elo for a single learner) is recorded as ``null``, never 0.
    """
    manifest_game = None
    run_type = "eval"
    for event in events:
        if event.event_type == "run_started":
            manifest_game = event.payload.get("game")
            break

    learner_elo = [row for row in elo_rows if row["player"] == learner]
    elo_history = [
        {
            "checkpoint_index": row["checkpoint_index"],
            "rating": row["rating"],
            "uncertainty": row["uncertainty"],
        }
        for row in learner_elo
    ]
    ratings = [row["rating"] for row in learner_elo if row["rating"] is not None]
    best_elo = max(ratings) if ratings else None
    final_elo = learner_elo[-1]["rating"] if learner_elo else None

    final_win = win_rows[-1] if win_rows else None
    win_rate = final_win["score"] if final_win else None
    score_margin = None if win_rate is None else (2.0 * win_rate - 1.0)

    final_checkpoint = checkpoints[-1] if checkpoints else None
    wall_seconds = (
        None if final_checkpoint is None else final_checkpoint.get("wall_seconds")
    )
    wall_hours = None if wall_seconds is None else wall_seconds / 3600.0
    total_steps = (
        None if final_checkpoint is None else final_checkpoint.get("env_steps")
    )

    # Behavioural information gain: C measures per-decision policy KL (nats).
    # A's AUC_gain is the area under that gain curve; AUC_raw/AUC_evo have no
    # RL counterpart and stay null rather than being coerced to 0.
    gain_records = [
        {"checkpoint_index": row["checkpoint_index"], "gain": row["nats_per_decision"]}
        for row in ig_rows
        if row.get("nats_per_decision") is not None
    ]
    auc_gain = None
    if len(gain_records) >= 2:
        curve = build_curve(gain_records, x_axis="checkpoint_index", y_axis="gain")
        auc_gain = trapezoid_auc(curve).value

    return {
        "schema_version": "1.0",
        "run_type": run_type,
        "game": manifest_game,
        "agent": learner,
        "created": events[0].created_at.isoformat() if events[0].created_at else None,
        "git_commit": _git_commit(run_dir),
        "best_elo": best_elo,
        "final_elo": final_elo,
        "elo_p0": final_elo,
        "elo_p1": None,
        "win_rate": win_rate,
        "score_margin": score_margin,
        "wall_hours": wall_hours,
        "total_steps": total_steps,
        "AUC_raw": None,
        "AUC_evo": None,
        "AUC_gain": auc_gain,
        "budget_learning": _schema_budget(final_checkpoint, kind="learning"),
        "budget_evaluation": _schema_budget(final_checkpoint, kind="evaluation"),
        "budget_total": _schema_budget(final_checkpoint, kind="total"),
        "elo_history": elo_history,
        "h2h": _h2h(outcomes_by_checkpoint, learner=learner),
    }


def _availability(available: bool, reason: str) -> dict[str, Any]:
    return {"available": available, "reason": None if available else reason}


def _write_csv(path: Path, rows: list[Mapping[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plots(
    report_dir: Path,
    *,
    checkpoints: list[Mapping[str, Any]],
    elo_rows: list[Mapping[str, Any]],
    win_rows: list[Mapping[str, Any]],
    ig_rows: list[Mapping[str, Any]],
    occupancy_rows: list[Mapping[str, Any]],
    resource_rows: list[Mapping[str, Any]],
    availability: Mapping[str, Mapping[str, Any]],
) -> None:
    checkpoint_lookup = {row["checkpoint_index"]: row for row in checkpoints}
    for label, axis in (
        ("iteration", "checkpoint_index"),
        ("env_steps", "env_steps"),
        ("wall_time", "wall_seconds"),
        ("gpu_hours", "gpu_hours"),
    ):
        _plot_joined(
            report_dir / f"elo_vs_{label}.png",
            elo_rows,
            checkpoint_lookup,
            x_axis=axis,
            y_axis="rating",
            unavailable=(
                availability["elo"]["reason"]
                if axis != "gpu_hours"
                else availability["gpu_hours"]["reason"]
            ),
        )
        _plot_joined(
            report_dir / f"win_rate_vs_{label}.png",
            win_rows,
            checkpoint_lookup,
            x_axis=axis,
            y_axis="score",
            lower="wilson_lower",
            upper="wilson_upper",
            unavailable=(
                "no raw match facts"
                if axis != "gpu_hours"
                else availability["gpu_hours"]["reason"]
            ),
        )
    _plot_simple(
        report_dir / "information_gain_vs_iteration.png",
        ig_rows,
        x_axis="checkpoint_index",
        y_axis="nats_per_decision",
        unavailable=availability["information_gain"]["reason"],
    )
    _plot_simple(
        report_dir / "occupancy_vs_iteration.png",
        occupancy_rows,
        x_axis="checkpoint_index",
        y_axis="occupancy_shift",
        unavailable=availability["occupancy"]["reason"],
    )
    _plot_simple(
        report_dir / "environment_steps_vs_iteration.png",
        checkpoints,
        x_axis="checkpoint_index",
        y_axis="env_steps",
        unavailable="no checkpoint budget facts",
    )
    _plot_multi(
        report_dir / "wall_time_vs_iteration.png",
        checkpoints,
        x_axis="checkpoint_index",
        y_axes=("learning_wall_seconds", "evaluation_wall_seconds", "wall_seconds"),
        unavailable="no checkpoint wall-time facts",
    )
    _plot_multi(
        report_dir / "gpu_hours_vs_iteration.png",
        checkpoints,
        x_axis="checkpoint_index",
        y_axes=("learning_gpu_hours", "evaluation_gpu_hours", "gpu_hours"),
        unavailable=availability["gpu_hours"]["reason"],
    )
    indexed_resources = [
        {**row, "sample_index": index} for index, row in enumerate(resource_rows)
    ]
    _plot_multi(
        report_dir / "resource_utilization_vs_sample.png",
        indexed_resources,
        x_axis="sample_index",
        y_axes=("cpu_utilization_percent", "gpu_utilization_percent"),
        unavailable=availability["resource_utilization"]["reason"],
    )


def _plot_joined(
    path: Path,
    rows: list[Mapping[str, Any]],
    checkpoints: Mapping[int, Mapping[str, Any]],
    *,
    x_axis: str,
    y_axis: str,
    unavailable: str | None,
    lower: str | None = None,
    upper: str | None = None,
) -> None:
    joined = []
    for row in rows:
        checkpoint = checkpoints.get(int(row["checkpoint_index"]), {})
        x = row.get(x_axis) if x_axis == "checkpoint_index" else checkpoint.get(x_axis)
        joined.append({**row, x_axis: x})
    _plot_simple(
        path,
        joined,
        x_axis=x_axis,
        y_axis=y_axis,
        lower=lower,
        upper=upper,
        unavailable=unavailable,
    )


def _plot_simple(
    path: Path,
    rows: list[Mapping[str, Any]],
    *,
    x_axis: str,
    y_axis: str,
    unavailable: str | None,
    lower: str | None = None,
    upper: str | None = None,
) -> None:
    points = [row for row in rows if row.get(x_axis) is not None and row.get(y_axis) is not None]
    figure, axis = plt.subplots(figsize=(6.4, 4.0), dpi=120)
    axis.set_xlabel(x_axis)
    axis.set_ylabel(y_axis)
    if points:
        x_values = [float(row[x_axis]) for row in points]
        y_values = [float(row[y_axis]) for row in points]
        axis.plot(x_values, y_values, marker="o")
        if lower and upper:
            axis.fill_between(
                x_values,
                [float(row[lower]) for row in points],
                [float(row[upper]) for row in points],
                alpha=0.2,
            )
    else:
        axis.text(0.5, 0.5, unavailable or "unavailable", ha="center", va="center")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _plot_multi(
    path: Path,
    rows: list[Mapping[str, Any]],
    *,
    x_axis: str,
    y_axes: tuple[str, ...],
    unavailable: str | None,
) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 4.0), dpi=120)
    axis.set_xlabel(x_axis)
    axis.set_ylabel(" / ".join(y_axes))
    plotted = False
    for y_axis in y_axes:
        points = [
            row
            for row in rows
            if row.get(x_axis) is not None and row.get(y_axis) is not None
        ]
        if not points:
            continue
        plotted = True
        axis.plot(
            [float(row[x_axis]) for row in points],
            [float(row[y_axis]) for row in points],
            marker="o",
            label=y_axis,
        )
    if plotted:
        axis.legend()
    else:
        axis.text(0.5, 0.5, unavailable or "unavailable", ha="center", va="center")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
