from __future__ import annotations

import json
import hashlib
import argparse
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from rlbench.config import compose_config
from rlbench.telemetry import EventLedger


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "src" / "games" / "snakego" / "scripts" / "run_snakego_task10.py"


def _run(*arguments: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _tiny_config(
    path: Path, *, self_play_episodes: int = 1, training_steps: int = 1
) -> Path:
    path.write_text(
        "game:\n"
        "  max_round: 1\n"
        "algorithms:\n"
        "  alphazero:\n"
        "    simulations: 1\n"
        "    channels: 4\n"
        "    residual_blocks: 1\n"
        "    batch_size: 1\n"
        "    replay_capacity: 8\n"
        "    min_replay_size: 1\n"
        "    mixed_precision: false\n"
        "    inference_batch_size: 1\n"
        "    device: cpu\n"
        "training:\n"
        "  seed: 77\n"
        "  generations: 1\n"
        f"  self_play_episodes: {self_play_episodes}\n"
        f"  training_steps: {training_steps}\n"
        "  processes: 1\n"
        "evaluation:\n"
        "  seeds: [9]\n"
        "resources:\n"
        "  sample: false\n"
        "run:\n"
        "  output_dir: ignored\n",
        encoding="utf-8",
    )
    return path


def _tiny_two_checkpoint_run(tmp_path: Path) -> tuple[Path, Path]:
    from games.snakego.experiments import Task10Stage, run_task10_workflow

    config = _tiny_config(tmp_path / "tiny.yaml")
    run_dir = tmp_path / "run"
    run_task10_workflow(
        run_dir=run_dir,
        config_path=config,
        stages=(
            Task10Stage(1, "selfplay", 1, 1, None),
            Task10Stage(2, "selfplay", 1, 1, None),
        ),
    )
    return run_dir, config


def _run_snapshot(run_dir: Path) -> dict[Path, bytes]:
    return {
        path.relative_to(run_dir): path.read_bytes()
        for path in run_dir.rglob("*")
        if path.is_file()
    }


def test_task10_plan_locks_the_reported_configs_and_expert_controls() -> None:
    plan = _run("plan")

    assert plan["alphazero_config"] == (
        "configs/experiments/snakego_task10_alphazero_locked.yaml"
    )
    assert plan["ppo_config"] == (
        "configs/experiments/snakego_task10_ppo_locked.yaml"
    )
    assert plan["expert_demo"] == {
        "opening_moves": 16,
        "opening_weight": 32.0,
        "self_play_episodes": 2,
        "training_steps": 256,
    }
    assert plan["bounded_final_opponent_order"] == ["rank6", "rank15"]
    assert len(plan["stages"]) == 20
    assert plan["totals"] == {
        "episodes": 180,
        "generations": 20,
        "optimizer_steps": 2432,
    }
    assert [
        (stage["checkpoint_index"], stage["opponent_rank"], stage["training_steps"])
        for stage in plan["stages"][16:]
    ] == [
        (17, "rank6", 256),
        (18, "rank15", 256),
        (19, "rank6", 256),
        (20, "rank15", 256),
    ]
    assert plan["final_continuation_gpu_hour_ceiling"] == 0.15
    assert [
        (
            stage["checkpoint_index"],
            stage["opponent_rank"],
            stage["training_steps"],
            stage["seed"],
            stage["expert_demo"],
            stage["opening_moves"],
            stage["opening_weight"],
        )
        for stage in plan["stages"][10:]
    ] == [
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
    ]

    az = compose_config(
        ROOT / str(plan["alphazero_config"]),
        game="snakego",
        algorithm="alphazero",
    )
    ppo = compose_config(
        ROOT / str(plan["ppo_config"]),
        game="snakego",
        algorithm="ppo",
    )
    assert az.config_hash == (
        "sha256:96102f41702eeb1c5f1b0ecfd3db431d74e9157554c519b93b7c8ff92c254af7"
    )
    assert ppo.config_hash == (
        "sha256:14d1da2180bad903578eac02843bd4b3c62822024c0fe82bad6b20a121c8c528"
    )


def test_task10_promotion_command_applies_elo_and_frozen_win_rate_jointly(
    tmp_path: Path,
) -> None:
    facts = {
        "candidate_id": "checkpoint-20",
        "champion_id": "checkpoint-19",
        "ratings": {"checkpoint-20": 900.0, "checkpoint-19": 1000.0},
        "outcomes": [
            {
                "player_a": "checkpoint-20",
                "player_b": "sample-agent",
                "score_a": 0.0,
                "valid": True,
            },
            {
                "player_a": "sample-agent",
                "player_b": "checkpoint-20",
                "score_a": 1.0,
                "valid": True,
            },
        ],
        "promotion_opponents": ["sample-agent"],
        "protected_reference_scores": {"sample-agent": 0.0},
        "evaluation_complete": True,
        "thresholds": {
            "minimum_elo_delta": 0.0,
            "minimum_win_rate_lower_bound": 0.5,
            "maximum_protected_regression": 0.0,
        },
    }
    path = tmp_path / "facts.json"
    path.write_text(json.dumps(facts), encoding="utf-8")

    decision = _run("promotion", "--facts", str(path))

    assert decision["promoted"] is False
    assert decision["candidate_status"] == "rejected_candidate"
    assert set(decision["reasons"]) == {"elo_delta", "win_rate_lower_bound"}


def test_task10_sweep_plan_covers_every_declared_training_parameter(
    tmp_path: Path,
) -> None:
    plan = _run("sweep", "--output", str(tmp_path), "--dry-run")

    assert set(plan["dimensions"]) == {
        "simulations",
        "c_puct",
        "root_dirichlet_fraction",
        "temperature_moves",
        "replay_capacity",
        "learning_rate",
        "network_width_depth",
        "inference_batch_size",
        "human_mixture_fraction",
    }
    assert plan["split"] == "training_microbenchmark"
    assert plan["heldout_used"] is False


def test_task10_local_evidence_is_auditable_and_heldout_is_not_selection() -> None:
    directory = ROOT / "results/snakego"
    matches = [
        json.loads(line)
        for line in (directory / "matches.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    heldout = [row for row in matches if row["evaluation_split"] == "heldout"]
    sample_agent = [
        row
        for row in matches
        if row["evaluation_split"] == "training"
        and "snakego-rank15-wenjie2002-sampleai"
        in (row["player_0"], row["player_1"])
    ]
    learned = [
        row
        for row in matches
        if row["evaluation_id"] == "70f30b16-f543-4f42-b415-736d5da9b03a"
    ]
    assert len(heldout) == 8
    assert len(sample_agent) == 2
    assert len(learned) == 4
    assert all(row["heldout_used_for_selection"] is False for row in matches)
    assert all(row["actions"] for row in matches)
    assert all(row["case_hash"].startswith("sha256:") for row in matches)
    assert all(row["case_set_hash"].startswith("sha256:") for row in matches)
    assert all(row["source_event_id"] for row in matches)
    assert all(
        row["source_event_hash"] == _canonical_sha256(row["source_event"])
        for row in matches
    )
    assert all(
        row["source_ledger_sha256"].startswith("sha256:") for row in matches
    )
    assert {row["source_ledger_sha256"] for row in matches} == {
        "sha256:f7c96df676411ea72047aa4fed8279b948c9a1a4a06a96c0a5ca9139f13cb57b",
        "sha256:cd05503b99c164de66bb56d2bd1114474b73f97497694fd0919f061e52111360",
    }
    executable_hashes = {
        row["player_0"]: row["player_0_executable_hash"] for row in matches
    } | {row["player_1"]: row["player_1_executable_hash"] for row in matches}
    assert executable_hashes["snakego-rank15-wenjie2002-sampleai"] == (
        "sha256:4305ddf156dc4909c6ee0dbbdb42d66e01518d1b50ced5c1f27c4216541dcd17"
    )
    assert executable_hashes["snakego-rank01-omegafantasy-ragnarok"] == (
        "sha256:22463dce353aaacb3469b188e57015f0b5142146fde1b8c6f9bebea2b80b285c"
    )
    serialized = "\n".join(json.dumps(row, sort_keys=True) for row in matches)
    assert "/" + "private/" not in serialized
    assert "/" + "Users/" not in serialized

    moves = [
        json.loads(line)
        for line in (directory / "moves.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(moves) == 17634
    actions_by_case: dict[tuple[str, str], list[int]] = {}
    for row in moves:
        key = (row["evaluation_id"], row["case_id"])
        actions_by_case.setdefault(key, []).append(int(row["action"]))
        assert row["source_event_id"]
        assert row["source_event_hash"] == _canonical_sha256(row["source_event"])
        assert row["agent_executable_hash"].startswith("sha256:")
    for row in matches:
        assert actions_by_case[(row["evaluation_id"], row["case_id"])] == row[
            "actions"
        ]

    evaluations = [
        json.loads(line)
        for line in (directory / "evaluations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(evaluations) == 3
    assert all(row["heldout_used_for_selection"] is False for row in evaluations)
    assert all(
        row["source_event_hash"] == _canonical_sha256(row["source_event"])
        for row in evaluations
    )

    continuation = json.loads(
        (directory / "continuation_accounting.json").read_text()
    )
    assert continuation["baseline"]["learning_gpu_hours"] == 0.5745332479924481
    assert continuation["final"]["learning_gpu_hours"] == 0.6957529932001812
    assert continuation["continuation_gpu_hours"] == 0.12121974520773315
    assert continuation["gpu_hour_ceiling"] == 0.15
    assert continuation["allocated_gpu_count"] == 1
    assert continuation["within_ceiling"] is True
    for key in ("baseline", "final"):
        assert continuation[key]["source_event_hash"] == _canonical_sha256(
            continuation[key]["source_event"]
        )

    promotion = json.loads((directory / "promotion_decision.json").read_text())
    assert promotion == {
        "candidate_status": "rejected_candidate",
        "elo_delta": 161.8719815259824,
        "promoted": False,
        "protected_scores": {
            "bootstrap-checkpoint-5": 0.75,
            "history-checkpoint-10": 0.5,
            "sample-agent-rank15": 0.0,
        },
        "reasons": ["win_rate_lower_bound"],
        "win_rate_lower_bound": 0.0,
    }

    sweep = json.loads((directory / "train_only_sweep.json").read_text())
    assert len(sweep["results"]) == 18
    assert sweep["heldout_used"] is False
    assert all(row["neural_batches"] > 0 for row in sweep["results"])
    assert all(row["neural_positions_per_second"] > 0 for row in sweep["results"])

    for line in (directory / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        assert actual == expected


def test_documented_checksum_command_runs_from_repository_root() -> None:
    """Basename-only checksum entries fail when the documented cwd is the root."""
    completed = subprocess.run(
        ["shasum", "-a", "256", "-c", "results/snakego/SHA256SUMS"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.count("OK") == 17


def test_task10_workflow_runs_real_stages_resumes_and_is_idempotent(
    tmp_path: Path,
) -> None:
    """A metadata-only plan would not create real optimizer/checkpoint facts."""
    from games.snakego.experiments import Task10Stage, run_task10_workflow

    config = _tiny_config(tmp_path / "tiny.yaml")
    stages = (
        Task10Stage(1, "selfplay", 1, 1, None),
        Task10Stage(2, "selfplay", 1, 1, None),
    )
    run_dir = tmp_path / "run"

    first = run_task10_workflow(
        run_dir=run_dir,
        config_path=config,
        stages=stages,
        maximum_stages=1,
    )
    checkpoint_one = run_dir / "checkpoints/checkpoint_000001.pt"
    checkpoint_one_hash = hashlib.sha256(checkpoint_one.read_bytes()).hexdigest()
    assert first["completed_through"] == 1
    assert first["episodes"] == 1
    assert first["optimizer_steps"] == 1

    second = run_task10_workflow(
        run_dir=run_dir,
        config_path=config,
        stages=stages,
    )
    assert second["completed_through"] == 2
    assert second["episodes"] == 2
    assert second["optimizer_steps"] == 2
    assert hashlib.sha256(checkpoint_one.read_bytes()).hexdigest() == checkpoint_one_hash
    assert (run_dir / "checkpoints/checkpoint_000002.pt").is_file()
    assert sum(
        event.event_type == "alphazero_optimizer_step"
        for event in EventLedger(run_dir / "events.jsonl").read()
    ) == 2

    ledger_bytes = (run_dir / "events.jsonl").read_bytes()
    third = run_task10_workflow(
        run_dir=run_dir,
        config_path=config,
        stages=stages,
    )
    assert third == second
    assert (run_dir / "events.jsonl").read_bytes() == ledger_bytes


def test_task10_workflow_rejects_state_budget_tampering_before_resume(
    tmp_path: Path,
) -> None:
    from games.snakego.experiments import Task10Stage, run_task10_workflow

    config = _tiny_config(tmp_path / "tiny.yaml")
    stages = (
        Task10Stage(1, "selfplay", 1, 1, None),
        Task10Stage(2, "selfplay", 1, 1, None),
    )
    run_dir = tmp_path / "run"
    run_task10_workflow(
        run_dir=run_dir,
        config_path=config,
        stages=stages,
        maximum_stages=1,
    )
    state_path = run_dir / "task10_workflow_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["episodes"] = 999
    state_path.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
    before = _run_snapshot(run_dir)

    with pytest.raises(ValueError, match="state budgets disagree"):
        run_task10_workflow(run_dir=run_dir, config_path=config, stages=stages)

    assert _run_snapshot(run_dir) == before


def test_task10_workflow_upgrades_version_one_state_during_resume(
    tmp_path: Path,
) -> None:
    """A durable round-two state must remain resumable under the richer journal."""
    from games.snakego.experiments import Task10Stage, run_task10_workflow

    config = _tiny_config(tmp_path / "tiny.yaml")
    stages = (
        Task10Stage(1, "selfplay", 1, 1, None),
        Task10Stage(2, "selfplay", 1, 1, None),
    )
    run_dir = tmp_path / "run"
    run_task10_workflow(
        run_dir=run_dir,
        config_path=config,
        stages=stages,
        maximum_stages=1,
    )
    state_path = run_dir / "task10_workflow_state.json"
    legacy = json.loads(state_path.read_text(encoding="utf-8"))
    legacy["schema_version"] = 1
    for field in (
        "allocated_gpu_source",
        "failed_attempt_gpu_hours",
        "final_continuation_failed_gpu_hours",
    ):
        legacy.pop(field)
    state_path.write_text(json.dumps(legacy, sort_keys=True), encoding="utf-8")

    resumed = run_task10_workflow(
        run_dir=run_dir,
        config_path=config,
        stages=stages,
    )

    assert resumed["schema_version"] == 2
    assert resumed["completed_through"] == 2
    assert resumed["failed_attempt_gpu_hours"] == 0.0
    assert resumed["final_continuation_failed_gpu_hours"] == 0.0


def test_task10_workflow_recovers_committed_stage_after_state_write_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import games.snakego.experiments.snakego_task10 as workflow_module
    from games.snakego.experiments import Task10Stage, run_task10_workflow

    config = _tiny_config(tmp_path / "tiny.yaml")
    stages = (Task10Stage(1, "selfplay", 1, 1, None),)
    run_dir = tmp_path / "run"
    real_write = workflow_module._write_workflow_state

    def fail_state_write(path: Path, state: object) -> None:
        raise RuntimeError("injected state write crash")

    monkeypatch.setattr(workflow_module, "_write_workflow_state", fail_state_write)
    with pytest.raises(RuntimeError, match="injected"):
        run_task10_workflow(run_dir=run_dir, config_path=config, stages=stages)

    checkpoint = run_dir / "checkpoints/checkpoint_000001.pt"
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    ledger_bytes = (run_dir / "events.jsonl").read_bytes()
    workflow_events = [
        event
        for event in EventLedger(run_dir / "events.jsonl").read()
        if event.event_type.startswith("workflow_stage_")
    ]
    assert [event.event_type for event in workflow_events] == [
        "workflow_stage_committed"
    ]
    attempt = json.loads(
        (run_dir / "task10_stage_attempt.json").read_text(encoding="utf-8")
    )
    assert attempt["attempt_id"] == workflow_events[0].payload["attempt_id"]
    monkeypatch.setattr(workflow_module, "_write_workflow_state", real_write)

    recovered = run_task10_workflow(
        run_dir=run_dir,
        config_path=config,
        stages=stages,
    )

    assert recovered["completed_through"] == 1
    assert not (run_dir / "task10_stage_attempt.json").exists()
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_hash
    assert (run_dir / "events.jsonl").read_bytes() == ledger_bytes


def test_task10_workflow_retries_after_crash_before_attempt_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import games.snakego.experiments.snakego_task10 as workflow_module
    from games.snakego.experiments import Task10Stage, run_task10_workflow

    config = _tiny_config(tmp_path / "tiny.yaml")
    stages = (Task10Stage(1, "selfplay", 1, 1, None),)
    run_dir = tmp_path / "run"
    real_write = workflow_module._write_stage_attempt

    def fail_before_attempt(path: Path, attempt: object) -> None:
        raise RuntimeError("injected before attempt")

    monkeypatch.setattr(workflow_module, "_write_stage_attempt", fail_before_attempt)
    with pytest.raises(RuntimeError, match="injected before attempt"):
        run_task10_workflow(run_dir=run_dir, config_path=config, stages=stages)
    assert not (run_dir / "task10_stage_attempt.json").exists()
    monkeypatch.setattr(workflow_module, "_write_stage_attempt", real_write)

    state = run_task10_workflow(run_dir=run_dir, config_path=config, stages=stages)

    assert state["completed_through"] == 1
    assert len(list((run_dir / "checkpoints").glob("checkpoint_*.pt"))) == 1


def test_task10_workflow_aborts_journal_only_attempt_and_retries_fresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rlbench.cli.main as cli_module
    from games.snakego.experiments import Task10Stage, run_task10_workflow

    config = _tiny_config(tmp_path / "tiny.yaml")
    stages = (Task10Stage(1, "selfplay", 1, 1, None),)
    run_dir = tmp_path / "run"
    real_main = cli_module.main

    def fail_after_attempt(arguments: list[str]) -> int:
        raise RuntimeError("injected after attempt")

    monkeypatch.setattr(cli_module, "main", fail_after_attempt)
    with pytest.raises(RuntimeError, match="injected after attempt"):
        run_task10_workflow(run_dir=run_dir, config_path=config, stages=stages)
    first_attempt = json.loads(
        (run_dir / "task10_stage_attempt.json").read_text(encoding="utf-8")
    )["attempt_id"]
    monkeypatch.setattr(cli_module, "main", real_main)

    state = run_task10_workflow(run_dir=run_dir, config_path=config, stages=stages)
    workflow_events = [
        event
        for event in EventLedger(run_dir / "events.jsonl").read()
        if event.event_type.startswith("workflow_stage_")
    ]

    assert state["completed_through"] == 1
    assert [event.event_type for event in workflow_events] == [
        "workflow_stage_attempt_aborted",
        "workflow_stage_committed",
    ]
    assert workflow_events[0].payload["attempt_id"] == first_attempt
    assert workflow_events[1].payload["attempt_id"] != first_attempt
    assert workflow_events[0].payload["partial_event_count"] == 0
    assert not (run_dir / "task10_stage_attempt.json").exists()


def test_task10_workflow_recovers_exclusive_checkpoint_without_saved_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from games.snakego.experiments import Task10Stage, run_task10_workflow
    from rlbench.telemetry import EventLedger

    config = _tiny_config(tmp_path / "tiny.yaml")
    stages = (Task10Stage(1, "selfplay", 1, 1, None),)
    run_dir = tmp_path / "run"
    real_append = EventLedger.append
    injected = False

    def fail_checkpoint_event(self, event):
        nonlocal injected
        if event.event_type == "checkpoint_saved" and not injected:
            injected = True
            raise RuntimeError("injected after checkpoint file")
        return real_append(self, event)

    monkeypatch.setattr(EventLedger, "append", fail_checkpoint_event)
    with pytest.raises(RuntimeError, match="injected after checkpoint file"):
        run_task10_workflow(run_dir=run_dir, config_path=config, stages=stages)
    checkpoint = run_dir / "checkpoints/checkpoint_000001.pt"
    original_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setattr(EventLedger, "append", real_append)

    state = run_task10_workflow(run_dir=run_dir, config_path=config, stages=stages)
    events = list(EventLedger(run_dir / "events.jsonl").read())

    assert state["completed_through"] == 1
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == original_hash
    saved = [event for event in events if event.event_type == "checkpoint_saved"]
    committed = [
        event for event in events if event.event_type == "workflow_stage_committed"
    ]
    assert len(saved) == len(committed) == 1
    assert saved[0].payload["recovered"] is True
    assert saved[0].payload["attempt_id"] == committed[0].payload["attempt_id"]
    assert not (run_dir / "task10_stage_attempt.json").exists()


@pytest.mark.parametrize("timeout_site", ("episode", "optimizer"))
def test_expert_timeout_durably_records_attempt_resources_without_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    timeout_site: str,
) -> None:
    from rlbench.algorithms.alphazero import AlphaZeroTrainer

    run_dir, _ = _tiny_two_checkpoint_run(tmp_path)
    checkpoint = run_dir / "checkpoints/checkpoint_000002.pt"
    agent = tmp_path / "agents/opponent/agent"
    agent.parent.mkdir(parents=True)
    agent.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    agent.chmod(0o755)
    executable_hash = "sha256:" + hashlib.sha256(agent.read_bytes()).hexdigest()
    population = tmp_path / "population.yaml"
    population.write_text(
        "schema_version: 1\n"
        "population_root: agents\n"
        "protocol_version: test-v1\n"
        "agents:\n"
        "- agent_id: opponent\n"
        "  kind: train_human\n"
        "  protocol: snakego_official\n"
        f"  content_hash: {executable_hash}\n"
        "  command: [opponent/agent]\n"
        "  roles: [player_0, player_1]\n",
        encoding="utf-8",
    )

    def timeout_generation(self, *args, **kwargs):
        if timeout_site == "optimizer":
            self.run_optimizer_steps(
                1, deadline_monotonic=__import__("time").monotonic() - 1.0
            )
        raise TimeoutError(f"{timeout_site} allocation deadline")

    monkeypatch.setattr(AlphaZeroTrainer, "run_generation", timeout_generation)
    namespace = runpy.run_path(str(SCRIPT))
    attempt_id = f"attempt-{timeout_site}"
    arguments = argparse.Namespace(
        run_dir=run_dir,
        checkpoint=checkpoint,
        population=population,
        opponent="opponent",
        device="cpu",
        seed=313,
        move_seconds=0.5,
        self_play_episodes=1,
        training_steps=1,
        expert_demo=True,
        opening_moves=0,
        opening_weight=1.0,
        gpu_hour_ceiling=0.1,
        allocated_gpus=2,
        attempt_id=attempt_id,
    )

    with pytest.raises(TimeoutError):
        namespace["_expert_generation_unlocked"](arguments)

    assert not (run_dir / "checkpoints/checkpoint_000003.pt").exists()
    resource_events = [
        event
        for event in EventLedger(run_dir / "events.jsonl").read()
        if event.event_type == "workflow_stage_resource"
        and event.payload.get("attempt_id") == attempt_id
    ]
    assert len(resource_events) == 1
    resource = resource_events[0].payload
    assert resource["status"] == "timed_out"
    assert resource["allocated_gpu_count"] == 2
    assert resource["allocation_source"] == "wall_clock_fallback"
    assert resource["allocated_gpu_hours"] > 0.0


def test_timed_out_attempt_updates_state_and_reduces_retry_ceiling_once(
    tmp_path: Path,
) -> None:
    from games.snakego.experiments import Task10Stage, run_task10_workflow
    from rlbench.telemetry import Event

    config = _tiny_config(tmp_path / "tiny.yaml")
    run_dir = tmp_path / "run"
    stages = (
        Task10Stage(1, "selfplay", 1, 1, None),
        Task10Stage(
            2,
            "expert",
            1,
            1,
            71,
            opponent_rank="rank15",
            final_continuation_gpu_hour_ceiling=0.01,
        ),
    )
    run_task10_workflow(
        run_dir=run_dir,
        config_path=config,
        stages=stages,
        maximum_stages=1,
    )
    observed_remaining: list[float] = []

    def timeout_runner(stage, checkpoint, remaining, attempt_id):
        del stage, checkpoint
        assert remaining is not None
        observed_remaining.append(remaining)
        spent = 0.004 if len(observed_remaining) == 1 else 0.002
        EventLedger(run_dir / "events.jsonl").append(
            Event(
                "workflow_stage_resource",
                "test-run",
                stage="learning",
                payload={
                    "attempt_id": attempt_id,
                    "checkpoint_index": 2,
                    "status": "timed_out",
                    "error_type": "TimeoutError",
                    "elapsed_seconds": spent * 3600.0,
                    "allocated_gpu_count": 1,
                    "allocation_source": "wall_clock_fallback",
                    "allocated_gpu_hours": spent,
                    "executed_stage_seed": 71,
                },
            )
        )
        raise TimeoutError("injected expert timeout")

    with pytest.raises(TimeoutError, match="injected expert timeout"):
        run_task10_workflow(
            run_dir=run_dir,
            config_path=config,
            stages=stages,
            expert_stage_runner=timeout_runner,
        )
    first_state = json.loads(
        (run_dir / "task10_workflow_state.json").read_text(encoding="utf-8")
    )
    first_events = list(EventLedger(run_dir / "events.jsonl").read())
    assert first_state["completed_through"] == 1
    assert first_state["failed_attempt_gpu_hours"] == pytest.approx(0.004)
    assert first_state["final_continuation_failed_gpu_hours"] == pytest.approx(
        0.004
    )
    assert not (run_dir / "checkpoints/checkpoint_000002.pt").exists()
    assert not (run_dir / "task10_stage_attempt.json").exists()
    assert sum(e.event_type == "workflow_stage_resource" for e in first_events) == 1
    assert sum(
        e.event_type == "workflow_stage_attempt_aborted" for e in first_events
    ) == 1

    unchanged = run_task10_workflow(
        run_dir=run_dir,
        config_path=config,
        stages=stages,
        expert_stage_runner=timeout_runner,
        maximum_stages=0,
    )
    assert unchanged == first_state
    assert len(list(EventLedger(run_dir / "events.jsonl").read())) == len(first_events)

    with pytest.raises(TimeoutError, match="injected expert timeout"):
        run_task10_workflow(
            run_dir=run_dir,
            config_path=config,
            stages=stages,
            expert_stage_runner=timeout_runner,
        )
    second_state = json.loads(
        (run_dir / "task10_workflow_state.json").read_text(encoding="utf-8")
    )
    assert observed_remaining == pytest.approx([0.01, 0.006])
    assert second_state["failed_attempt_gpu_hours"] == pytest.approx(0.006)
    assert second_state["final_continuation_failed_gpu_hours"] == pytest.approx(
        0.006
    )


def test_task10_workflow_rejects_concurrent_stage_attempt(
    tmp_path: Path,
) -> None:
    import games.snakego.experiments.snakego_task10 as workflow_module
    from games.snakego.experiments import Task10Stage, run_task10_workflow

    config = _tiny_config(tmp_path / "tiny.yaml")
    run_dir = tmp_path / "run"
    stages = (Task10Stage(1, "selfplay", 1, 1, None),)

    with workflow_module._workflow_lock(run_dir):
        before = _run_snapshot(run_dir)
        with pytest.raises(RuntimeError, match="already running"):
            run_task10_workflow(run_dir=run_dir, config_path=config, stages=stages)
        assert _run_snapshot(run_dir) == before


def test_task10_workflow_cleans_committed_attempt_after_state_replace_crash(
    tmp_path: Path,
) -> None:
    import games.snakego.experiments.snakego_task10 as workflow_module
    from games.snakego.experiments import Task10Stage, run_task10_workflow

    config = _tiny_config(tmp_path / "tiny.yaml")
    run_dir = tmp_path / "run"
    stages = (Task10Stage(1, "selfplay", 1, 1, None),)
    state = run_task10_workflow(run_dir=run_dir, config_path=config, stages=stages)
    commit = next(
        event
        for event in EventLedger(run_dir / "events.jsonl").read()
        if event.event_type == "workflow_stage_committed"
    )
    workflow_module._write_stage_attempt(
        run_dir / "task10_stage_attempt.json",
        {
            "schema_version": 1,
            "attempt_id": commit.payload["attempt_id"],
            "checkpoint_index": commit.payload["checkpoint_index"],
            "plan_hash": commit.payload["plan_hash"],
        },
    )
    ledger_bytes = (run_dir / "events.jsonl").read_bytes()

    recovered = run_task10_workflow(
        run_dir=run_dir,
        config_path=config,
        stages=stages,
    )

    assert recovered == state
    assert not (run_dir / "task10_stage_attempt.json").exists()
    assert (run_dir / "events.jsonl").read_bytes() == ledger_bytes


def test_task10_driver_workflow_executes_the_real_first_locked_stage(
    tmp_path: Path,
) -> None:
    """The command must dispatch framework training, not only print a plan."""
    config = _tiny_config(
        tmp_path / "stage.yaml",
        self_play_episodes=16,
        training_steps=64,
    )
    run_dir = tmp_path / "workflow-run"

    result = _run(
        "workflow",
        "--output",
        str(run_dir),
        "--config",
        str(config),
        "--population",
        str(tmp_path / "unused-until-expert-stage.yaml"),
        "--rank5",
        "rank5-id",
        "--rank6",
        "rank6-id",
        "--rank15",
        "rank15-id",
        "--maximum-stages",
        "1",
    )

    assert result["completed_through"] == 1
    assert result["episodes"] == 16
    assert result["optimizer_steps"] == 64
    assert (run_dir / "checkpoints/checkpoint_000001.pt").is_file()


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("outside", "inside its run directory"),
        ("tampered", "content hash mismatch"),
        ("stale", "latest valid lineage head"),
    ],
)
def test_expert_generation_rejects_untrusted_checkpoint_before_run_mutation(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    """Population/process construction must not precede checkpoint provenance."""
    run_dir, _ = _tiny_two_checkpoint_run(tmp_path)
    checkpoint_one = run_dir / "checkpoints/checkpoint_000001.pt"
    checkpoint_two = run_dir / "checkpoints/checkpoint_000002.pt"
    if case == "outside":
        checkpoint = tmp_path / "outside.pt"
        checkpoint.write_bytes(checkpoint_two.read_bytes())
    elif case == "tampered":
        checkpoint = checkpoint_two
        checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")
    else:
        checkpoint = checkpoint_one
    before = _run_snapshot(run_dir)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "expert-generation",
            "--run-dir",
            str(run_dir),
            "--checkpoint",
            str(checkpoint),
            "--population",
            str(tmp_path / "missing-population.yaml"),
            "--opponent",
            "training-opponent",
            "--seed",
            "5",
            "--device",
            "cpu",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert message in completed.stderr
    assert _run_snapshot(run_dir) == before
