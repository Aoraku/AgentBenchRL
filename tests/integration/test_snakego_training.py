from __future__ import annotations

import json
import os
import pickle
import shutil
import csv
from pathlib import Path

import pytest

from rlbench.cli.main import main
from rlbench.registry import game_factory
from rlbench.telemetry import EventLedger


def _smoke_config(path: Path, algorithm: str) -> Path:
    algorithm_block = (
        "  simulations: 1\n"
        "  channels: 4\n"
        "  residual_blocks: 1\n"
        "  batch_size: 1\n"
        "  replay_capacity: 8\n"
        "  min_replay_size: 1\n"
        "  mixed_precision: false\n"
        if algorithm == "alphazero"
        else "  hidden_size: 8\n"
        "  conv_channels: 4\n"
        "  vector_envs: 1\n"
        "  episodes_per_collect: 2\n"
        "  minibatch_size: 4\n"
        "  update_repetitions: 1\n"
        "  shaping_beta: 0.1\n"
    )
    path.write_text(
        "game:\n"
        "  max_round: 2\n"
        "algorithm:\n"
        f"{algorithm_block}"
        "training:\n"
        "  seed: 3\n"
        "  generations: 1\n"
        "  iterations: 1\n"
        "  self_play_episodes: 1\n"
        "  training_steps: 1\n"
        "  processes: 1\n"
        "evaluation:\n"
        "  seeds: [9]\n",
        encoding="utf-8",
    )
    return path


def test_registered_game_factory_round_trips_across_spawn_boundary() -> None:
    """A local closure in the registry makes every processes>1 run unpicklable."""
    original = game_factory("snakego", {"max_round": 3})
    restored = pickle.loads(pickle.dumps(original))

    game = restored()
    game.reset(17)
    assert game.max_round == 3
    assert game.current_player() == 0


@pytest.mark.parametrize(
    ("algorithm", "counter"),
    [("alphazero", "generation"), ("ppo", "iteration")],
)
def test_real_snakego_training_resumes_evaluates_and_reports(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    algorithm: str,
    counter: str,
) -> None:
    """CLI wiring that bypasses collect/search/update or restore is not a smoke run."""
    config = _smoke_config(tmp_path / f"{algorithm}.yaml", algorithm)
    run = tmp_path / f"run-{algorithm}"

    assert (
        main(
            [
                "train",
                "snakego",
                "--algo",
                algorithm,
                "--config",
                str(config),
                "--output",
                str(run),
            ]
        )
        == 0
    )
    first_output = json.loads(capsys.readouterr().out)
    checkpoint = Path(first_output["checkpoint"])
    assert checkpoint.exists() and checkpoint.stat().st_size > 0

    first_events = list(EventLedger(run / "events.jsonl").read())
    first_checkpoint = [
        event for event in first_events if event.event_type == "checkpoint_saved"
    ][-1]
    assert first_checkpoint.payload[counter] == 1
    assert any(
        event.event_type in {"alphazero_optimizer_step", "ppo_optimizer_step"}
        for event in first_events
    )
    assert first_events[-1].event_type == "run_finished"

    assert (
        main(
            [
                "evaluate",
                "snakego",
                "--checkpoint",
                str(checkpoint),
                "--opponent",
                "random",
                "--seeds",
                "9",
            ]
        )
        == 0
    )
    first_evaluation = json.loads(capsys.readouterr().out)
    assert first_evaluation["valid_games"] == 2
    after_first_evaluation = list(EventLedger(run / "events.jsonl").read())
    first_eval_finished = [
        event
        for event in after_first_evaluation
        if event.event_type == "evaluation_finished"
    ][-1]
    first_eval_budget = first_eval_finished.payload["budgets"]["evaluation"]

    assert (
        main(
            [
                "train",
                "snakego",
                "--algo",
                algorithm,
                "--config",
                str(config),
                "--output",
                str(run),
                "--resume",
                str(checkpoint),
            ]
        )
        == 0
    )
    second_output = json.loads(capsys.readouterr().out)
    assert Path(second_output["checkpoint"]).exists()
    resumed_events = list(EventLedger(run / "events.jsonl").read())
    resumed_checkpoint = [
        event for event in resumed_events if event.event_type == "checkpoint_saved"
    ][-1]
    assert resumed_checkpoint.payload[counter] == 2
    assert any(event.event_type == "run_resumed" for event in resumed_events)
    assert resumed_checkpoint.payload["budgets"]["evaluation"] == first_eval_budget
    assert resumed_checkpoint.payload["budgets"]["total"]["episodes"] == (
        resumed_checkpoint.payload["budgets"]["learning"]["episodes"]
        + resumed_checkpoint.payload["budgets"]["evaluation"]["episodes"]
    )
    assert resumed_checkpoint.payload["gpu_hours"] is None

    before_stale_resume = {
        path.relative_to(run): path.read_bytes()
        for path in run.rglob("*")
        if path.is_file()
    }
    with pytest.raises(ValueError, match="latest.*lineage head"):
        main(
            [
                "train",
                "snakego",
                "--algo",
                algorithm,
                "--config",
                str(config),
                "--output",
                str(run),
                "--resume",
                str(checkpoint),
            ]
        )
    assert {
        path.relative_to(run): path.read_bytes()
        for path in run.rglob("*")
        if path.is_file()
    } == before_stale_resume

    assert (
        main(
            [
                "evaluate",
                "snakego",
                "--checkpoint",
                second_output["checkpoint"],
                "--opponent",
                "random",
                "--seeds",
                "9",
            ]
        )
        == 0
    )
    evaluation = json.loads(capsys.readouterr().out)
    assert evaluation["valid_games"] == 2
    assert evaluation["side_swapped"] is True
    eval_events = list(EventLedger(run / "events.jsonl").read())
    assert len([event for event in eval_events if event.event_type == "match_finished"]) == 4
    evaluation_ids = {
        event.payload["evaluation_id"]
        for event in eval_events
        if event.event_type == "evaluation_finished"
    }
    assert len(evaluation_ids) == 2
    assert all(
        event.payload.get("state_id")
        for event in eval_events
        if event.event_type == "evaluation_move"
    )
    assert any(
        event.event_type == "policy_ig_measured"
        and event.payload["checkpoint_index"] == 2
        for event in eval_events
    )
    assert any(
        event.event_type == "occupancy_measured"
        and event.payload["checkpoint_index"] == 2
        for event in eval_events
    )

    assert main(["report", str(run)]) == 0
    report = json.loads(capsys.readouterr().out)
    report_dir = Path(report["report_directory"])
    assert (report_dir / "summary.json").exists()
    with (report_dir / "auc.csv").open(newline="", encoding="utf-8") as stream:
        auc_rows = list(csv.DictReader(stream))
    with (report_dir / "information_gain.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        ig_rows = list(csv.DictReader(stream))
    with (report_dir / "occupancy.csv").open(newline="", encoding="utf-8") as stream:
        occupancy_rows = list(csv.DictReader(stream))
    assert len(ig_rows) == 1
    assert ig_rows[0]["checkpoint_index"] == "2"
    assert len(occupancy_rows) == 1
    assert occupancy_rows[0]["checkpoint_index"] == "2"
    assert {row["y_axis"] for row in auc_rows} >= {"elo", "win_rate"}
    assert {row["x_axis"] for row in auc_rows} >= {
        "checkpoint_index",
        "env_steps",
        "wall_seconds",
    }
    assert all(path.stat().st_size > 1_000 for path in report_dir.glob("*.png"))

    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    encoded_manifest = json.dumps(manifest).lower()
    assert "ssh" not in encoded_manifest
    assert "hostname" not in encoded_manifest
    assert "credential" not in encoded_manifest
    assert manifest["config_hash"]


def test_disabled_resource_sampling_keeps_gpu_cost_unavailable(tmp_path: Path) -> None:
    """Opting out of measurement must not turn missing GPU evidence into zero cost."""
    config = _smoke_config(tmp_path / "no-resources.yaml", "alphazero")
    with config.open("a", encoding="utf-8") as stream:
        stream.write("resources:\n  sample: false\n")
    run = tmp_path / "run-no-resources"

    assert main(
        [
            "train",
            "snakego",
            "--algo",
            "alphazero",
            "--config",
            str(config),
            "--output",
            str(run),
        ]
    ) == 0

    events = list(EventLedger(run / "events.jsonl").read())
    checkpoint = [event for event in events if event.event_type == "checkpoint_saved"][-1]
    assert checkpoint.payload["gpu_hours"] is None
    assert not any(event.event_type == "resource_sampled" for event in events)


def test_checkpoint_destination_is_never_overwritten(tmp_path: Path, capsys) -> None:
    """A counter/path collision must preserve the historical checkpoint bytes."""
    config = _smoke_config(tmp_path / "exclusive.yaml", "alphazero")
    run = tmp_path / "exclusive-run"
    assert main(
        [
            "train",
            "snakego",
            "--algo",
            "alphazero",
            "--config",
            str(config),
            "--output",
            str(run),
        ]
    ) == 0
    first = json.loads(capsys.readouterr().out)
    destination = run / "checkpoints" / "checkpoint_000002.pt"
    sentinel = b"immutable historical checkpoint"
    destination.write_bytes(sentinel)

    with pytest.raises(FileExistsError, match="checkpoint destination"):
        main(
            [
                "train",
                "snakego",
                "--algo",
                "alphazero",
                "--config",
                str(config),
                "--output",
                str(run),
                "--resume",
                first["checkpoint"],
            ]
        )

    assert destination.read_bytes() == sentinel
    saved_indices = [
        event.payload["checkpoint_index"]
        for event in EventLedger(run / "events.jsonl").read()
        if event.event_type == "checkpoint_saved"
    ]
    assert saved_indices == [1]


def test_manifest_tamper_and_foreign_checkpoint_fail_before_event_mutation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Stored hashes and compatible bytes cannot bypass run lineage."""
    config = _smoke_config(tmp_path / "lineage.yaml", "alphazero")
    run = tmp_path / "lineage-run"
    assert main(
        [
            "train",
            "snakego",
            "--algo",
            "alphazero",
            "--config",
            str(config),
            "--output",
            str(run),
        ]
    ) == 0
    output = json.loads(capsys.readouterr().out)
    checkpoint = Path(output["checkpoint"])
    ledger = EventLedger(run / "events.jsonl")

    foreign = tmp_path / "compatible-foreign.pt"
    shutil.copyfile(checkpoint, foreign)
    before_foreign = len(tuple(ledger.read()))
    with pytest.raises(ValueError, match="lineage"):
        main(
            [
                "train",
                "snakego",
                "--algo",
                "alphazero",
                "--config",
                str(config),
                "--output",
                str(run),
                "--resume",
                str(foreign),
            ]
        )
    assert len(tuple(ledger.read())) == before_foreign

    manifest_path = run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["canonical_config"]["training"]["seed"] += 1
    os.chmod(manifest_path, 0o644)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    before_tamper = len(tuple(ledger.read()))
    with pytest.raises(ValueError, match="manifest"):
        main(
            [
                "evaluate",
                "snakego",
                "--checkpoint",
                str(checkpoint),
                "--opponent",
                "random",
            ]
        )
    assert len(tuple(ledger.read())) == before_tamper
