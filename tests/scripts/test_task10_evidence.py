from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rlbench.cli.main import main as rlbench_main
from rlbench.evaluation import EvaluationCase
from rlbench.telemetry import EventLedger


def _event(event_id: str, event_type: str, payload: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "event_id": event_id,
        "event_type": event_type,
        "run_id": "source-run",
        "stage": "evaluation",
        "created_at": "2026-08-06T00:00:00+00:00",
        "payload": payload,
    }


def _case(
    *, seed: int, player_0: str, player_1: str, hash_0: str, hash_1: str
) -> EvaluationCase:
    return EvaluationCase.create(
        seed=seed,
        player_0=player_0,
        player_1=player_1,
        player_0_hash=hash_0,
        player_1_hash=hash_1,
        protocol_version="test-v1",
    )


CASE_CONTRACT = {
    "game_config": {},
    "limits": {},
    "protocol_version": "test-v1",
}


def _population_manifest(
    *, agent_id: str, executable_hash: str, kind: str = "train_human"
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "population_root": "agents",
        "protocol_version": "test-v1",
        "agents": [
            {
                "agent_id": agent_id,
                "kind": kind,
                "protocol": "line_json",
                "content_hash": executable_hash,
                "command": [f"{agent_id}/agent"],
                "roles": ["player_0", "player_1"],
                "resource_limits": {},
                "provenance": {},
            }
        ],
    }


def test_raw_evidence_extractor_preserves_moves_hashes_and_source_provenance(
    tmp_path: Path,
) -> None:
    """Summary-only output cannot audit actions or trace facts to source events."""
    from rlbench.experiments.evidence import extract_evaluation_evidence

    candidate_hash = "sha256:" + "2" * 64
    history_hash = "sha256:" + "5" * 64
    population_manifest = _population_manifest(
        agent_id="checkpoint-5", executable_hash=history_hash
    )
    population_hash = _canonical_hash(population_manifest)
    case = _case(
        seed=17,
        player_0="checkpoint-20",
        player_1="checkpoint-5",
        hash_0=candidate_hash,
        hash_1=history_hash,
    )
    case_hash = case.content_hash
    common = {
        "evaluation_id": "eval-train",
        "case_id": case.case_id,
        "case_hash": case_hash,
        "seed": 17,
    }
    events = [
        _event(
            "checkpoint-20-source",
            "checkpoint_saved",
            {
                "checkpoint_index": 20,
                "checkpoint_hash": candidate_hash,
                    "training_population_hash": population_hash,
            },
        ),
        _event(
            "move-0",
            "evaluation_move",
            {
                **common,
                "move_index": 0,
                "player": 0,
                "agent_id": "checkpoint-20",
                "state_id": "state-0",
                "action": 3,
                "terminated": False,
            },
        ),
        _event(
            "move-1",
            "evaluation_move",
            {
                **common,
                "move_index": 1,
                "player": 1,
                "agent_id": "checkpoint-5",
                "state_id": "state-1",
                "action": 2,
                "terminated": True,
            },
        ),
        _event(
            "match-1",
            "match_finished",
            {
                **common,
                "checkpoint_index": 20,
                "player_0": "checkpoint-20",
                "player_1": "checkpoint-5",
                "actions": [3, 2],
                "score_player_0": 1.0,
                "valid": True,
                "reason": "completed",
            },
        ),
        _event(
            "finish-1",
            "evaluation_finished",
            {
                "evaluation_id": "eval-train",
                "checkpoint_index": 20,
                "complete": True,
                "valid_games": 1,
                "raw_matches": 1,
                "env_steps": 2,
            },
        ),
    ]
    ledger = tmp_path / "source-events.jsonl"
    ledger_bytes = "".join(
        json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
        for event in events
    ).encode()
    ledger.write_bytes(ledger_bytes)
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "eval-train": {
                    "evaluation_split": "training",
                    "candidate_checkpoint_hash": candidate_hash,
                    "case_contract": CASE_CONTRACT,
                    "population_hash": population_hash,
                    "population_manifest": population_manifest,
                    "executable_hashes": {
                        "checkpoint-20": candidate_hash,
                        "checkpoint-5": history_hash,
                    },
                    "heldout_used_for_selection": False,
                }
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "evidence"

    summary = extract_evaluation_evidence(ledger, metadata, output)

    matches = [json.loads(line) for line in (output / "matches.jsonl").read_text().splitlines()]
    moves = [json.loads(line) for line in (output / "moves.jsonl").read_text().splitlines()]
    evaluations = [
        json.loads(line)
        for line in (output / "evaluations.jsonl").read_text().splitlines()
    ]
    assert summary == {"evaluations": 1, "matches": 1, "moves": 2}
    assert matches[0]["actions"] == [3, 2]
    assert matches[0]["case_hash"] == case_hash
    assert matches[0]["case_set_hash"].startswith("sha256:")
    assert matches[0]["candidate_checkpoint_hash"] == "sha256:" + "2" * 64
    assert matches[0]["player_0_executable_hash"] == "sha256:" + "2" * 64
    assert matches[0]["player_1_executable_hash"] == "sha256:" + "5" * 64
    assert matches[0]["source_event_id"] == "match-1"
    assert matches[0]["source_event_hash"].startswith("sha256:")
    assert matches[0]["source_event"] == events[3]
    assert matches[0]["source_event_hash"] == _canonical_hash(
        matches[0]["source_event"]
    )
    assert matches[0]["source_ledger_sha256"] == (
        "sha256:" + hashlib.sha256(ledger_bytes).hexdigest()
    )
    assert [move["source_event_id"] for move in moves] == ["move-0", "move-1"]
    assert evaluations[0]["source_event_id"] == "finish-1"
    assert evaluations[0]["case_set_hash"] == matches[0]["case_set_hash"]
    assert evaluations[0]["source_payload"] == events[-1]["payload"]
    assert evaluations[0]["source_event_hash"] == _canonical_hash(
        evaluations[0]["source_event"]
    )
    serialized = "".join(path.read_text() for path in output.iterdir())
    assert str(tmp_path) not in serialized


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def test_raw_evidence_rejects_opponent_identity_supplied_only_by_metadata(
    tmp_path: Path,
) -> None:
    """A sidecar label cannot prove an opponent's immutable bytes or split role."""
    from rlbench.experiments.evidence import extract_evaluation_evidence

    candidate_hash = "sha256:" + "a" * 64
    opponent_hash = "sha256:" + "b" * 64
    case = _case(
        seed=7,
        player_0="candidate",
        player_1="opponent",
        hash_0=candidate_hash,
        hash_1=opponent_hash,
    )
    common = {
        "evaluation_id": "metadata-only-opponent",
        "case_id": case.case_id,
        "case_hash": case.content_hash,
        "seed": 7,
    }
    events = [
        _event(
            "checkpoint",
            "checkpoint_saved",
            {"checkpoint_index": 20, "checkpoint_hash": candidate_hash},
        ),
        _event(
            "move",
            "evaluation_move",
            {
                **common,
                "move_index": 0,
                "player": 0,
                "agent_id": "candidate",
                "state_id": "state",
                "action": 1,
                "terminated": True,
            },
        ),
        _event(
            "match",
            "match_finished",
            {
                **common,
                "checkpoint_index": 20,
                "player_0": "candidate",
                "player_1": "opponent",
                "actions": [1],
                "score_player_0": 1.0,
                "valid": True,
                "reason": "completed",
            },
        ),
        _event(
            "finish",
            "evaluation_finished",
            {
                "evaluation_id": "metadata-only-opponent",
                "checkpoint_index": 20,
                "complete": True,
                "valid_games": 1,
                "raw_matches": 1,
                "env_steps": 1,
            },
        ),
    ]
    ledger = tmp_path / "events.jsonl"
    ledger.write_text("".join(json.dumps(event) + "\n" for event in events))
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "metadata-only-opponent": {
                    "evaluation_split": "heldout",
                    "candidate_checkpoint_hash": candidate_hash,
                    "case_contract": CASE_CONTRACT,
                    "population_hash": "sha256:" + "c" * 64,
                    "executable_hashes": {
                        "candidate": candidate_hash,
                        "opponent": opponent_hash,
                    },
                    "heldout_used_for_selection": False,
                }
            }
        )
    )

    with pytest.raises(ValueError, match="immutable population"):
        extract_evaluation_evidence(ledger, metadata, tmp_path / "output")


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("absent_candidate", "not source-anchored"),
        ("forged_population", "metadata population_hash disagrees"),
        ("incomplete_manifest", "population coverage disagrees"),
    ),
)
def test_evidence_authority_rejects_forged_or_incomplete_sidecar_facts(
    case: str,
    message: str,
) -> None:
    from rlbench.experiments.evidence import _resolve_evidence_authority

    candidate_hash = "sha256:" + "a" * 64
    opponent_hash = "sha256:" + "b" * 64
    manifest_agent = "different" if case == "incomplete_manifest" else "opponent"
    manifest = _population_manifest(
        agent_id=manifest_agent, executable_hash=opponent_hash
    )
    population_hash = _canonical_hash(manifest)
    labels = {
        "evaluation_split": "training",
        "candidate_checkpoint_hash": candidate_hash,
        "case_contract": CASE_CONTRACT,
        "population_hash": (
            "sha256:" + "f" * 64 if case == "forged_population" else population_hash
        ),
        "population_manifest": manifest,
        "executable_hashes": {
            "candidate": candidate_hash,
            "opponent": opponent_hash,
        },
        "heldout_used_for_selection": False,
    }
    checkpoint_sources = (
        {}
        if case == "absent_candidate"
        else {
            20: [
                {
                    "payload": {
                        "checkpoint_index": 20,
                        "checkpoint_hash": candidate_hash,
                        "training_population_hash": population_hash,
                    }
                }
            ]
        }
    )
    matches = [
        {
            "payload": {
                "player_0": "candidate",
                "player_1": "opponent",
            }
        }
    ]

    with pytest.raises(ValueError, match=message):
        _resolve_evidence_authority(
            evaluation_id="eval",
            labels=labels,
            finish={"checkpoint_index": 20},
            matches=matches,
            checkpoint_sources=checkpoint_sources,
        )


@pytest.mark.parametrize(
    "unsafe",
    (
        "\\" * 2 + "host\\share\\agent",
        "file" + ":///var/data/agent",
        "C:" + "\\Users\\operator\\agent.exe",
        "prefix /" + "home/operator/agent suffix",
        "prefix /" + "Users/operator/agent suffix",
        "prefix /" + "mnt/worker/agent suffix",
        "prefix /" + "private/tmp/agent suffix",
        "prefix /" + "tmp/agent suffix",
        "s" + "sh://operator@example.invalid/agent",
        {"credential": "opaque"},
    ),
)
def test_raw_evidence_recursively_rejects_paths_hosts_and_credentials(
    unsafe: object,
) -> None:
    """Nested path and connection details must never survive in a preimage."""
    from rlbench.experiments.evidence import _reject_absolute_paths

    with pytest.raises(ValueError, match="unsafe path or connection detail"):
        _reject_absolute_paths({"nested": [unsafe]})


def test_raw_evidence_extractor_rejects_action_trace_disagreement(
    tmp_path: Path,
) -> None:
    """A match action list that disagrees with move facts is not auditable."""
    from rlbench.experiments.evidence import extract_evaluation_evidence

    candidate_hash = "sha256:" + "c" * 64
    history_hash = "sha256:" + "e" * 64
    case = _case(
        seed=1,
        player_0="candidate",
        player_1="history",
        hash_0=candidate_hash,
        hash_1=history_hash,
    )
    case_hash = case.content_hash
    events = [
        _event(
            "move",
            "evaluation_move",
            {
                "evaluation_id": "eval",
                "case_id": case.case_id,
                "case_hash": case_hash,
                "seed": 1,
                "move_index": 0,
                "player": 0,
                "agent_id": "candidate",
                "state_id": "state",
                "action": 1,
                "terminated": True,
            },
        ),
        _event(
            "match",
            "match_finished",
            {
                "evaluation_id": "eval",
                "case_id": case.case_id,
                "case_hash": case_hash,
                "seed": 1,
                "checkpoint_index": 20,
                "candidate_checkpoint_hash": candidate_hash,
                "evaluation_split": "training",
                "population_hash": "sha256:" + "4" * 64,
                "player_0": "candidate",
                "player_1": "history",
                "actions": [2],
                "score_player_0": 1.0,
                "valid": True,
                "reason": "completed",
            },
        ),
        _event(
            "finish",
            "evaluation_finished",
            {
                "evaluation_id": "eval",
                "checkpoint_index": 20,
                "candidate_checkpoint_hash": candidate_hash,
                "candidate_id": "candidate",
                "evaluation_split": "training",
                "population_hash": "sha256:" + "4" * 64,
                "heldout_used_for_selection": False,
                "opponent_checkpoint_hashes": {"history": history_hash},
                "complete": True,
                "valid_games": 1,
                "raw_matches": 1,
                "env_steps": 1,
            },
        ),
    ]
    ledger = tmp_path / "events.jsonl"
    ledger.write_text("".join(json.dumps(event) + "\n" for event in events))
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "eval": {
                    "evaluation_split": "training",
                    "candidate_checkpoint_hash": candidate_hash,
                    "case_contract": CASE_CONTRACT,
                    "population_hash": "sha256:" + "4" * 64,
                    "executable_hashes": {
                        "candidate": candidate_hash,
                        "history": history_hash,
                    },
                    "heldout_used_for_selection": False,
                }
            }
        )
    )

    try:
        extract_evaluation_evidence(ledger, metadata, tmp_path / "output")
    except ValueError as exc:
        assert "action trace" in str(exc)
    else:
        raise AssertionError("mismatched action traces must be rejected")


def test_raw_evidence_extractor_cross_checks_source_completion_facts(
    tmp_path: Path,
) -> None:
    from rlbench.experiments.evidence import extract_evaluation_evidence

    case_hash = "sha256:" + "a" * 64
    events = [
        _event(
            "move",
            "evaluation_move",
            {
                "evaluation_id": "eval",
                "case_id": "case",
                "case_hash": case_hash,
                "seed": 1,
                "move_index": 0,
                "player": 0,
                "agent_id": "candidate",
                "state_id": "state",
                "action": 1,
                "terminated": True,
            },
        ),
        _event(
            "match",
            "match_finished",
            {
                "evaluation_id": "eval",
                "case_id": "case",
                "case_hash": case_hash,
                "seed": 1,
                "checkpoint_index": 20,
                "player_0": "candidate",
                "player_1": "history",
                "actions": [1],
                "score_player_0": 1.0,
                "valid": True,
                "reason": "completed",
            },
        ),
        _event(
            "finish",
            "evaluation_finished",
            {
                "evaluation_id": "eval",
                "evaluation_split": "training",
                "candidate_checkpoint_hash": "sha256:" + "c" * 64,
                "population_hash": "sha256:" + "d" * 64,
                "heldout_used_for_selection": False,
                "checkpoint_index": 20,
                "complete": False,
                "valid_games": 1,
                "raw_matches": 999,
                "env_steps": 1,
            },
        ),
    ]
    ledger = tmp_path / "events.jsonl"
    ledger.write_text("".join(json.dumps(event) + "\n" for event in events))
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "eval": {
                    "evaluation_split": "heldout",
                    "candidate_checkpoint_hash": "sha256:" + "b" * 64,
                    "case_contract": CASE_CONTRACT,
                    "population_hash": "sha256:" + "e" * 64,
                    "executable_hashes": {
                        "candidate": "sha256:" + "b" * 64,
                        "history": "sha256:" + "f" * 64,
                    },
                    "heldout_used_for_selection": False,
                }
            }
        )
    )

    with pytest.raises(ValueError, match="incomplete"):
        extract_evaluation_evidence(ledger, metadata, tmp_path / "output")


def test_raw_evidence_extractor_rejects_paths_in_any_persisted_metadata(
    tmp_path: Path,
) -> None:
    from rlbench.experiments.evidence import extract_evaluation_evidence

    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "eval": {
                    "evaluation_split": "training",
                    "candidate_checkpoint_hash": "sha256:" + "a" * 64,
                    "case_contract": CASE_CONTRACT,
                    "population_hash": "sha256:" + "b" * 64,
                    "executable_hashes": {
                        "/" + "private/secret-agent": "sha256:" + "a" * 64,
                    },
                    "heldout_used_for_selection": False,
                }
            }
        )
    )
    ledger = tmp_path / "events.jsonl"
    ledger.write_text("{}\n")

    with pytest.raises(ValueError, match="unsafe path or connection detail"):
        extract_evaluation_evidence(ledger, metadata, tmp_path / "output")


@pytest.mark.parametrize(
    ("mismatch", "message"),
    (
        ("move", "move contract"),
        ("match", "match contract"),
        ("provenance", "match contract"),
    ),
)
def test_raw_evidence_extractor_rejects_overlapping_source_disagreement(
    tmp_path: Path,
    mismatch: str,
    message: str,
) -> None:
    from rlbench.experiments.evidence import extract_evaluation_evidence

    candidate_hash = "sha256:" + "a" * 64
    opponent_hash = "sha256:" + "b" * 64
    case = _case(
        seed=1,
        player_0="candidate",
        player_1="opponent",
        hash_0=candidate_hash,
        hash_1=opponent_hash,
    )
    events = [
        _event(
            "move",
            "evaluation_move",
            {
                "evaluation_id": "eval",
                "case_id": case.case_id,
                "case_hash": case.content_hash,
                "seed": 999 if mismatch == "move" else 1,
                "move_index": 0,
                "player": 0,
                "agent_id": "candidate",
                "state_id": "state",
                "action": 1,
                "terminated": True,
            },
        ),
        _event(
            "match",
            "match_finished",
            {
                "evaluation_id": "eval",
                "evaluation_split": (
                    "heldout" if mismatch == "match" else "training"
                ),
                "candidate_checkpoint_hash": (
                    "sha256:" + "f" * 64
                    if mismatch == "provenance"
                    else candidate_hash
                ),
                "population_hash": (
                    "sha256:" + "f" * 64
                    if mismatch == "provenance"
                    else "sha256:" + "c" * 64
                ),
                "heldout_used_for_selection": False,
                "case_id": case.case_id,
                "case_hash": case.content_hash,
                "seed": 1,
                "checkpoint_index": 20,
                "player_0": "candidate",
                "player_1": "opponent",
                "actions": [1],
                "score_player_0": 1.0,
                "valid": True,
                "reason": "completed",
            },
        ),
        _event(
            "finish",
            "evaluation_finished",
            {
                "evaluation_id": "eval",
                "evaluation_split": "training",
                "candidate_checkpoint_hash": candidate_hash,
                "candidate_id": "candidate",
                "population_hash": "sha256:" + "c" * 64,
                "heldout_used_for_selection": False,
                "opponent_checkpoint_hashes": {"opponent": opponent_hash},
                "checkpoint_index": 20,
                "complete": True,
                "valid_games": 1,
                "raw_matches": 1,
                "env_steps": 1,
            },
        ),
    ]
    ledger = tmp_path / "events.jsonl"
    ledger.write_text("".join(json.dumps(event) + "\n" for event in events))
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "eval": {
                    "evaluation_split": "training",
                    "candidate_checkpoint_hash": candidate_hash,
                    "case_contract": CASE_CONTRACT,
                    "population_hash": "sha256:" + "c" * 64,
                    "executable_hashes": {
                        "candidate": candidate_hash,
                        "opponent": opponent_hash,
                    },
                    "heldout_used_for_selection": False,
                }
            }
        )
    )

    with pytest.raises(ValueError, match=message):
        extract_evaluation_evidence(ledger, metadata, tmp_path / "output")


def test_continuation_accounting_is_derived_from_checkpoint_event_preimages(
    tmp_path: Path,
) -> None:
    from rlbench.experiments.evidence import extract_continuation_accounting

    events = [
        _event(
            "checkpoint-16",
            "checkpoint_saved",
            {
                "checkpoint_index": 16,
                "checkpoint_hash": "sha256:" + "1" * 64,
                "learning_gpu_hours": 0.57,
                "budgets": {
                    "learning": {"episodes": 172, "optimizer_steps": 1408}
                },
            },
        ),
        _event(
            "checkpoint-20",
            "checkpoint_saved",
            {
                "checkpoint_index": 20,
                "checkpoint_hash": "sha256:" + "2" * 64,
                "learning_gpu_hours": 0.69,
                "budgets": {
                    "learning": {"episodes": 180, "optimizer_steps": 2432}
                },
            },
        ),
    ]
    ledger = tmp_path / "events.jsonl"
    ledger.write_text(
        "".join(
            json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
            for event in events
        )
    )
    output = tmp_path / "continuation.json"

    result = extract_continuation_accounting(
        ledger,
        output,
        baseline_checkpoint=16,
        final_checkpoint=20,
        gpu_hour_ceiling=0.15,
        allocated_gpu_count=1,
    )

    assert result["continuation_gpu_hours"] == pytest.approx(0.12)
    assert result["within_ceiling"] is True
    assert result["allocated_gpu_count"] == 1
    assert result["baseline"]["source_event_hash"] == _canonical_hash(
        result["baseline"]["source_event"]
    )
    assert result["final"]["source_event_hash"] == _canonical_hash(
        result["final"]["source_event"]
    )
    assert json.loads(output.read_text()) == result


def test_checkpoint_league_runs_real_side_swapped_training_evaluation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Learned bootstrap/history baselines must be real checkpoint policies."""
    from rlbench.experiments.checkpoint_league import evaluate_checkpoint_league

    config = tmp_path / "tiny.yaml"
    config.write_text(
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
        "  seed: 31\n"
        "  generations: 1\n"
        "  self_play_episodes: 1\n"
        "  training_steps: 1\n"
        "  processes: 1\n"
        "evaluation:\n"
        "  seeds: [17]\n"
        "  move_seconds: 1.0\n"
        "resources:\n"
        "  sample: false\n"
        "run:\n"
        "  output_dir: ignored\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    assert rlbench_main(
        [
            "train",
            "snakego",
            "--algo",
            "alphazero",
            "--config",
            str(config),
            "--output",
            str(run_dir),
        ]
    ) == 0
    first = json.loads(capsys.readouterr().out)
    assert rlbench_main(
        [
            "train",
            "snakego",
            "--algo",
            "alphazero",
            "--config",
            str(config),
            "--output",
            str(run_dir),
            "--resume",
            first["checkpoint"],
        ]
    ) == 0
    second = json.loads(capsys.readouterr().out)
    output = tmp_path / "checkpoint-league.jsonl"

    summary = evaluate_checkpoint_league(
        run_dir=run_dir,
        candidate_checkpoint=Path(second["checkpoint"]),
        opponents={"bootstrap-checkpoint-1": Path(first["checkpoint"])},
        seeds=(17,),
        output_ledger=output,
        evaluation_split="training",
    )

    assert summary["valid_games"] == 2
    assert summary["heldout_used_for_selection"] is False
    assert summary["completed_mcts_simulations"] == summary["env_steps"]
    assert summary["wall_seconds"] > 0.0
    events = list(EventLedger(output).read())
    matches = [event for event in events if event.event_type == "match_finished"]
    assert len(matches) == 2
    assert all(event.payload["actions"] for event in matches)
    assert all(event.payload["case_hash"].startswith("sha256:") for event in matches)
    finished = next(event for event in events if event.event_type == "evaluation_finished")
    assert finished.payload["case_set_hash"].startswith("sha256:")
    assert finished.payload["evaluation_split"] == "training"
    assert finished.payload["completed_mcts_simulations"] == finished.payload["env_steps"]
    assert finished.payload["candidate_checkpoint_hash"] == second["checkpoint_hash"]
    assert finished.payload["opponent_checkpoint_hashes"] == {
        "bootstrap-checkpoint-1": first["checkpoint_hash"]
    }
