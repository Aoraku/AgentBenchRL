from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import rlbench.cli.main as cli_module
from rlbench.algorithms.alphazero import AlphaZeroConfig
from rlbench.cli.main import main
from rlbench.config import ConfigError, compose_config
from rlbench.evaluation import build_side_swapped_cases
from rlbench.registry import ALGORITHMS, GAMES
from rlbench.reporting import generate_report
from rlbench.telemetry import Event, EventLedger


def _write_yaml(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_occupancy_comparison_requires_the_identical_frozen_case_set(
    tmp_path: Path,
) -> None:
    """A nearby checkpoint evaluated on different cases is not comparable."""
    ledger = EventLedger(tmp_path / "events.jsonl")
    for evaluation_id, checkpoint_index, case_set_hash in (
        ("matching", 1, "sha256:cases-a"),
        ("different", 2, "sha256:cases-b"),
    ):
        ledger.append(
            Event(
                "evaluation_move",
                "r",
                stage="evaluation",
                payload={
                    "evaluation_id": evaluation_id,
                    "state_id": f"state-{evaluation_id}",
                },
            )
        )
        ledger.append(
            Event(
                "evaluation_finished",
                "r",
                stage="evaluation",
                payload={
                    "evaluation_id": evaluation_id,
                    "checkpoint_index": checkpoint_index,
                    "case_set_hash": case_set_hash,
                    "complete": True,
                },
            )
        )

    result = cli_module._prior_complete_evaluation_states(
        ledger,
        run_id="r",
        checkpoint_index=3,
        case_set_hash="sha256:cases-a",
    )

    assert result == ("matching", ("state-matching",))


def test_case_set_identity_normalizes_only_the_candidate_checkpoint_hash() -> None:
    """Checkpoint comparisons share cases, while opponent changes do not."""

    def cases(candidate_hash: str, opponent_hash: str):
        return build_side_swapped_cases(
            candidate_id="learner",
            candidate_hash=candidate_hash,
            opponent_id="frozen-opponent",
            opponent_hash=opponent_hash,
            seeds=(7, 11),
            game_config={"max_round": 512},
            limits={"move_seconds": 3.0},
        )

    first = cli_module._case_set_hash(cases("sha256:checkpoint-a", "sha256:human"))
    second = cli_module._case_set_hash(cases("sha256:checkpoint-b", "sha256:human"))
    different = cli_module._case_set_hash(
        cases("sha256:checkpoint-b", "sha256:other-human")
    )

    assert first == second
    assert different != first


def test_registry_is_explicit_and_validate_game_runs_contract(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Implicit discovery could silently admit unreviewed games or algorithms."""
    assert set(GAMES) == {"snakego"}
    assert set(ALGORITHMS) == {"alphazero", "ppo"}

    assert main(["validate-game", "snakego", "--seed", "7"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result == {
        "action_count": 6,
        "game": "snakego",
        "observation_planes": 24,
        "observation_scalars": 39,
        "status": "valid",
    }


def test_config_composition_is_deterministic_strict_and_path_aware(
    tmp_path: Path,
) -> None:
    """Order-dependent merges, ignored typos, or caller-relative data break replay."""
    config_dir = tmp_path / "configuration"
    config_dir.mkdir()
    path = _write_yaml(
        config_dir / "experiment.yaml",
        "game:\n"
        "  max_round: 2\n"
        "algorithm:\n"
        "  simulations: 1\n"
        "  channels: 4\n"
        "  residual_blocks: 1\n"
        "  batch_size: 1\n"
        "  min_replay_size: 1\n"
        "training:\n"
        "  seed: 11\n"
        "  generations: 1\n"
        "  self_play_episodes: 1\n"
        "  training_steps: 1\n"
        "run:\n"
        "  output_dir: artifacts\n",
    )

    first = compose_config(path, game="snakego", algorithm="alphazero")
    second = compose_config(path, game="snakego", algorithm="alphazero")

    assert first.canonical == second.canonical
    assert first.config_hash == second.config_hash
    assert first.output_dir == config_dir / "artifacts"
    assert first.canonical["run"]["output_dir"] == "artifacts"

    unknown = _write_yaml(config_dir / "unknown.yaml", "training:\n  genrations: 1\n")
    with pytest.raises(ConfigError, match="unknown configuration key"):
        compose_config(unknown, game="snakego", algorithm="alphazero")

    forbidden = _write_yaml(config_dir / "forbidden.yaml", "connection:\n  endpoint: example\n")
    with pytest.raises(ConfigError, match="forbidden"):
        compose_config(forbidden, game="snakego", algorithm="ppo")

    absolute = _write_yaml(
        config_dir / "absolute.yaml",
        "run:\n  output_dir: /" + "private/machine/run\n",
    )
    with pytest.raises(ConfigError, match="relative"):
        compose_config(absolute, game="snakego", algorithm="ppo")


def test_config_selects_only_the_requested_algorithm_override(tmp_path: Path) -> None:
    """A shared experiment must not leak one backend's controls into the other."""
    path = _write_yaml(
        tmp_path / "shared.yaml",
        "algorithms:\n"
        "  alphazero:\n"
        "    simulations: 2\n"
        "  ppo:\n"
        "    episodes_per_collect: 2\n",
    )

    alphazero = compose_config(path, game="snakego", algorithm="alphazero")
    ppo = compose_config(path, game="snakego", algorithm="ppo")

    assert alphazero.canonical["algorithm"]["simulations"] == 2
    assert "episodes_per_collect" not in alphazero.canonical["algorithm"]
    assert ppo.canonical["algorithm"]["episodes_per_collect"] == 2
    assert "simulations" not in ppo.canonical["algorithm"]


def test_alphazero_device_is_validated_and_resolved_for_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The configured accelerator must reach network construction."""
    path = _write_yaml(
        tmp_path / "cuda.yaml",
        "algorithm:\n  device: cuda:3\n  mixed_precision: true\n",
    )
    composed = compose_config(path, game="snakego", algorithm="alphazero")
    captured: dict[str, object] = {}

    class FakeNetwork:
        @classmethod
        def from_game_spec(cls, spec: object, config: object, *, device: str) -> object:
            captured["device"] = device
            return object()

    class FakeTrainer:
        def __init__(self, network: object, config: object, **kwargs: object) -> None:
            captured["network"] = network

    monkeypatch.setattr(cli_module, "PolicyValueNet", FakeNetwork)
    monkeypatch.setattr(cli_module, "AlphaZeroTrainer", FakeTrainer)
    cli_module._build_trainer(
        composed,
        ledger=EventLedger(tmp_path / "events.jsonl"),
        run_id="cuda-device",
    )

    assert composed.algorithm_settings().device == "cuda:3"
    assert captured["device"] == "cuda:3"

    monkeypatch.setattr(cli_module.torch.cuda, "is_available", lambda: True)
    assert cli_module._alphazero_device(AlphaZeroConfig(device="auto")) == "cuda"
    monkeypatch.setattr(cli_module.torch.cuda, "is_available", lambda: False)
    assert cli_module._alphazero_device(AlphaZeroConfig(device="auto")) == "cpu"


@pytest.mark.parametrize("device", ["gpu", "cuda:-1", "cuda:x", "mps"])
def test_alphazero_device_rejects_ambiguous_values(
    tmp_path: Path, device: str
) -> None:
    path = _write_yaml(tmp_path / "invalid-device.yaml", f"algorithm:\n  device: {device}\n")
    with pytest.raises(ConfigError, match="device"):
        compose_config(path, game="snakego", algorithm="alphazero")


@pytest.mark.parametrize(
    "invalid_yaml",
    [
        "game:\n  max_round: 0\n",
        "algorithm:\n  simulations: 0\n",
        "resources:\n  sample: 'false'\n",
        "evaluation:\n  move_seconds: 0\n",
    ],
)
def test_invalid_config_fails_before_creating_artifacts(
    tmp_path: Path, invalid_yaml: str
) -> None:
    """Late validation must not leave a manifest or event stream for an invalid run."""
    config = _write_yaml(tmp_path / "invalid.yaml", invalid_yaml)
    run = tmp_path / "must-not-exist"

    with pytest.raises((ConfigError, ValueError)):
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
            ]
        )

    assert not run.exists()


def test_train_rejects_unknown_algorithm_without_creating_a_run(
    tmp_path: Path,
) -> None:
    """A misspelled backend must not fall through to a default trainer."""
    config = _write_yaml(tmp_path / "config.yaml", "training:\n  iterations: 1\n")
    with pytest.raises(SystemExit):
        main(
            [
                "train",
                "snakego",
                "--algo",
                "not-real",
                "--config",
                str(config),
                "--output",
                str(tmp_path / "run"),
            ]
        )
    assert not (tmp_path / "run").exists()


def test_report_is_derived_from_raw_events_and_marks_missing_evidence(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Hard-coded scores or zero-filled absent telemetry would misstate evidence."""
    run = tmp_path / "run"
    ledger = EventLedger(run / "events.jsonl")
    facts = [
        Event(
            "run_started",
            "run-1",
            payload={"algorithm": "ppo", "game": "snakego"},
        ),
        Event(
            "checkpoint_saved",
            "run-1",
            stage="learning",
            payload={
                "checkpoint_index": 1,
                "env_steps": 12,
                "wall_seconds": 3.5,
                "learning_wall_seconds": 3.0,
                "evaluation_wall_seconds": 0.5,
                "gpu_hours": None,
            },
        ),
        Event(
            "match_finished",
            "run-1",
            stage="evaluation",
            payload={
                "checkpoint_index": 1,
                "player_0": "learner",
                "player_1": "random",
                "score_player_0": 1.0,
                "valid": True,
                "seed": 5,
            },
        ),
        Event(
            "match_finished",
            "run-1",
            stage="evaluation",
            payload={
                "checkpoint_index": 1,
                "player_0": "random",
                "player_1": "learner",
                "score_player_0": 0.5,
                "valid": True,
                "seed": 5,
            },
        ),
        Event(
            "policy_ig_measured",
            "run-1",
            stage="evaluation",
            payload={
                "checkpoint_index": 1,
                "nats_per_decision": 0.25,
                "nats_per_episode": 0.5,
            },
        ),
        Event(
            "occupancy_measured",
            "run-1",
            stage="evaluation",
            payload={"checkpoint_index": 1, "occupancy_shift": 0.125},
        ),
        Event("run_finished", "run-1", payload={"status": "completed"}),
    ]
    for fact in facts:
        ledger.append(fact)

    assert main(["report", str(run)]) == 0
    output = json.loads(capsys.readouterr().out)
    report_dir = Path(output["report_directory"])
    summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))

    assert summary["win_rate"]["learner"]["wins"] == 1
    assert summary["win_rate"]["learner"]["draws"] == 1
    assert summary["win_rate"]["learner"]["score"] == 0.75
    assert summary["availability"]["gpu_hours"]["available"] is False
    assert summary["availability"]["gpu_hours"]["reason"]
    assert summary["elo"]["learner"]["rating"] > 1000.0

    required_tables = {
        "elo.csv",
        "win_rate.csv",
        "information_gain.csv",
        "occupancy.csv",
        "budgets.csv",
        "resources.csv",
    }
    assert required_tables <= {path.name for path in report_dir.glob("*.csv")}
    with (report_dir / "win_rate.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["wilson_lower"] and rows[0]["wilson_upper"]

    images = list(report_dir.glob("*.png"))
    assert len(images) >= 10
    assert all(path.stat().st_size > 1_000 for path in images)


def test_report_rejects_non_event_aggregate_as_its_only_source(tmp_path: Path) -> None:
    """A summary file without append-only facts is not auditable evidence."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "summary.json").write_text('{"win_rate": 1.0}', encoding="utf-8")
    with pytest.raises(ValueError, match="events.jsonl"):
        main(["report", str(run)])


def test_report_uses_latest_complete_evaluation_without_partial_aggregation(
    tmp_path: Path,
) -> None:
    """Repeated or incomplete invocations must not be pooled into benchmark scores."""
    run = tmp_path / "run"
    ledger = EventLedger(run / "events.jsonl")
    ledger.append(Event("run_started", "r", payload={"candidate_id": "learner"}))
    for index in (1, 2):
        ledger.append(
            Event(
                "checkpoint_saved",
                "r",
                payload={
                    "checkpoint_index": index,
                    "env_steps": index * 10,
                    "wall_seconds": float(index),
                    "gpu_hours": None,
                },
            )
        )
    for evaluation_id, score, metric in (
        ("old", 1.0, 1.0),
        ("latest", 0.0, 2.0),
    ):
        for side in (0, 1):
            ledger.append(
                Event(
                    "match_finished",
                    "r",
                    stage="evaluation",
                    payload={
                        "evaluation_id": evaluation_id,
                        "checkpoint_index": 1,
                        "case_id": f"{evaluation_id}-{side}",
                        "case_hash": f"sha256:{evaluation_id}-{side}",
                        "seed": 4,
                        "player_0": "learner" if side == 0 else "random",
                        "player_1": "random" if side == 0 else "learner",
                        "score_player_0": score if side == 0 else 1.0 - score,
                        "valid": True,
                    },
                )
            )
        ledger.append(
            Event(
                "policy_ig_measured",
                "r",
                stage="evaluation",
                payload={
                    "evaluation_id": evaluation_id,
                    "checkpoint_index": 1,
                    "nats_per_decision": metric,
                    "nats_per_episode": metric * 2.0,
                },
            )
        )
        ledger.append(
            Event(
                "occupancy_measured",
                "r",
                stage="evaluation",
                payload={
                    "evaluation_id": evaluation_id,
                    "checkpoint_index": 1,
                    "occupancy_shift": metric,
                },
            )
        )
        ledger.append(
            Event(
                "evaluation_finished",
                "r",
                stage="evaluation",
                payload={
                    "evaluation_id": evaluation_id,
                    "checkpoint_index": 1,
                    "complete": True,
                },
            )
        )
    ledger.append(
        Event(
            "match_finished",
            "r",
            stage="evaluation",
            payload={
                "evaluation_id": "broken",
                "checkpoint_index": 2,
                "case_id": "broken-0",
                "case_hash": "sha256:broken-0",
                "seed": 8,
                "player_0": "learner",
                "player_1": "random",
                "score_player_0": None,
                "valid": False,
            },
        )
    )
    ledger.append(
        Event(
            "policy_ig_measured",
            "r",
            stage="evaluation",
            payload={
                "evaluation_id": "broken",
                "checkpoint_index": 2,
                "nats_per_decision": 9.0,
                "nats_per_episode": 18.0,
            },
        )
    )
    ledger.append(
        Event(
            "occupancy_measured",
            "r",
            stage="evaluation",
            payload={
                "evaluation_id": "broken",
                "checkpoint_index": 2,
                "occupancy_shift": 9.0,
            },
        )
    )
    ledger.append(
        Event(
            "evaluation_finished",
            "r",
            stage="evaluation",
            payload={
                "evaluation_id": "broken",
                "checkpoint_index": 2,
                "complete": False,
            },
        )
    )

    report_dir = generate_report(run)
    with (report_dir / "win_rate.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    with (report_dir / "information_gain.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        ig_rows = list(csv.DictReader(stream))
    with (report_dir / "occupancy.csv").open(newline="", encoding="utf-8") as stream:
        occupancy_rows = list(csv.DictReader(stream))
    summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))

    assert len(rows) == 1
    assert rows[0]["checkpoint_index"] == "1"
    assert rows[0]["valid_games"] == "2"
    assert rows[0]["score"] == "0.0"
    assert len(ig_rows) == 1
    assert ig_rows[0]["checkpoint_index"] == "1"
    assert ig_rows[0]["nats_per_decision"] == "2.0"
    assert len(occupancy_rows) == 1
    assert occupancy_rows[0]["checkpoint_index"] == "1"
    assert occupancy_rows[0]["occupancy_shift"] == "2.0"
    assert summary["evaluation_availability"]["2"]["available"] is False
    assert "incomplete" in summary["evaluation_availability"]["2"]["reason"]
