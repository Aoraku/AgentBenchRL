"""Path-free extraction of auditable evaluation facts from an event ledger."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def extract_continuation_accounting(
    ledger_path: str | Path,
    output_path: str | Path,
    *,
    baseline_checkpoint: int,
    final_checkpoint: int,
    gpu_hour_ceiling: float,
    allocated_gpu_count: int,
) -> dict[str, Any]:
    """Extract locally recomputable continuation accounting from checkpoint events."""
    if baseline_checkpoint < 1 or final_checkpoint <= baseline_checkpoint:
        raise ValueError("continuation checkpoint bounds are invalid")
    if gpu_hour_ceiling <= 0.0 or allocated_gpu_count < 1:
        raise ValueError("continuation resource contract is invalid")
    source_bytes = Path(ledger_path).read_bytes()
    ledger_hash = f"sha256:{hashlib.sha256(source_bytes).hexdigest()}"
    events = [json.loads(line) for line in source_bytes.splitlines() if line]

    def checkpoint_event(index: int) -> dict[str, Any]:
        selected = [
            event
            for event in events
            if event.get("event_type") == "checkpoint_saved"
            and isinstance(event.get("payload"), dict)
            and event["payload"].get("checkpoint_index") == index
        ]
        if len(selected) != 1:
            raise ValueError(f"checkpoint {index} requires one source event")
        event = selected[0]
        _reject_absolute_paths(event)
        payload = event["payload"]
        learning_gpu_hours = payload.get("learning_gpu_hours")
        checkpoint_hash = payload.get("checkpoint_hash")
        budgets = payload.get("budgets")
        learning = budgets.get("learning") if isinstance(budgets, Mapping) else None
        if (
            not isinstance(learning_gpu_hours, (int, float))
            or isinstance(learning_gpu_hours, bool)
            or not isinstance(checkpoint_hash, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", checkpoint_hash) is None
            or not isinstance(learning, Mapping)
        ):
            raise ValueError(f"checkpoint {index} source accounting is invalid")
        return {
            "checkpoint_index": index,
            "checkpoint_hash": checkpoint_hash,
            "episodes": learning.get("episodes"),
            "optimizer_steps": learning.get("optimizer_steps"),
            "learning_gpu_hours": float(learning_gpu_hours),
            "source_event": event,
            "source_event_id": str(event["event_id"]),
            "source_event_hash": _hash_json(event),
        }

    baseline = checkpoint_event(baseline_checkpoint)
    final = checkpoint_event(final_checkpoint)
    continuation = final["learning_gpu_hours"] - baseline["learning_gpu_hours"]
    if continuation < 0.0:
        raise ValueError("continuation GPU hours cannot be negative")
    result = {
        "schema_version": 1,
        "allocated_gpu_count": allocated_gpu_count,
        "gpu_hour_ceiling": gpu_hour_ceiling,
        "continuation_gpu_hours": continuation,
        "within_ceiling": continuation <= gpu_hour_ceiling,
        "source_ledger_sha256": ledger_hash,
        "baseline": baseline,
        "final": final,
    }
    _reject_absolute_paths(result)
    Path(output_path).write_text(
        json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def extract_evaluation_evidence(
    ledger_path: str | Path,
    metadata_path: str | Path,
    output_directory: str | Path,
) -> dict[str, int]:
    source = Path(ledger_path)
    source_bytes = source.read_bytes()
    ledger_hash = f"sha256:{hashlib.sha256(source_bytes).hexdigest()}"
    metadata_bytes = Path(metadata_path).read_bytes()
    metadata_hash = f"sha256:{hashlib.sha256(metadata_bytes).hexdigest()}"
    metadata = json.loads(metadata_bytes)
    if not isinstance(metadata, dict) or not metadata:
        raise ValueError("evaluation metadata must be a non-empty mapping")
    _reject_absolute_paths(metadata)
    events = [json.loads(line) for line in source_bytes.splitlines() if line]

    selected = set(metadata)
    checkpoint_sources: dict[int, list[dict[str, Any]]] = defaultdict(list)
    moves_by_evaluation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    matches_by_evaluation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    finishes_by_evaluation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if not isinstance(event, dict) or not isinstance(event.get("payload"), dict):
            raise ValueError("source ledger contains an invalid event")
        if event.get("event_type") == "checkpoint_saved":
            checkpoint_index = event["payload"].get("checkpoint_index")
            if isinstance(checkpoint_index, int) and not isinstance(
                checkpoint_index, bool
            ):
                checkpoint_sources[checkpoint_index].append(event)
        evaluation_id = event["payload"].get("evaluation_id")
        if evaluation_id not in selected:
            continue
        if event.get("event_type") == "evaluation_move":
            moves_by_evaluation[str(evaluation_id)].append(event)
        elif event.get("event_type") == "match_finished":
            matches_by_evaluation[str(evaluation_id)].append(event)
        elif event.get("event_type") == "evaluation_finished":
            finishes_by_evaluation[str(evaluation_id)].append(event)

    match_rows: list[dict[str, Any]] = []
    move_rows: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    for evaluation_id in sorted(selected):
        details = metadata[evaluation_id]
        _validate_metadata(evaluation_id, details)
        matches = matches_by_evaluation[evaluation_id]
        finishes = finishes_by_evaluation[evaluation_id]
        if not matches or len(finishes) != 1:
            raise ValueError(f"evaluation {evaluation_id} is missing raw completion facts")
        case_hashes = sorted(str(event["payload"]["case_hash"]) for event in matches)
        case_set_hash = _hash_json(case_hashes)
        moves_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in moves_by_evaluation[evaluation_id]:
            moves_by_case[str(event["payload"]["case_id"])].append(event)
        finish_event = finishes[0]
        finish = finish_event["payload"]
        details = _validate_source_completion(
            evaluation_id=evaluation_id,
            details=details,
            finish=finish,
            matches=matches,
            moves_by_case=moves_by_case,
            case_set_hash=case_set_hash,
            checkpoint_sources=checkpoint_sources,
        )

        for match_event in sorted(matches, key=lambda event: str(event["payload"]["case_id"])):
            payload = match_event["payload"]
            case_id = str(payload["case_id"])
            case_moves = sorted(
                moves_by_case.get(case_id, ()),
                key=lambda event: int(event["payload"]["move_index"]),
            )
            actions = [int(event["payload"]["action"]) for event in case_moves]
            if actions != payload.get("actions"):
                raise ValueError(f"evaluation {evaluation_id} action trace disagrees with match")
            executable_hashes = details["executable_hashes"]
            player_0 = str(payload["player_0"])
            player_1 = str(payload["player_1"])
            row = {
                "schema_version": 1,
                "evaluation_id": evaluation_id,
                "evaluation_split": details["evaluation_split"],
                "case_id": case_id,
                "case_hash": payload["case_hash"],
                "case_set_hash": case_set_hash,
                "seed": payload["seed"],
                "checkpoint_index": payload["checkpoint_index"],
                "candidate_checkpoint_hash": details["candidate_checkpoint_hash"],
                "population_hash": details["population_hash"],
                "player_0": player_0,
                "player_1": player_1,
                "player_0_executable_hash": executable_hashes[player_0],
                "player_1_executable_hash": executable_hashes[player_1],
                "actions": list(payload["actions"]),
                "score_player_0": payload["score_player_0"],
                "valid": payload["valid"],
                "reason": payload["reason"],
                "heldout_used_for_selection": details[
                    "heldout_used_for_selection"
                ],
                "source_metadata_sha256": metadata_hash,
                **_source_fields(match_event, ledger_hash),
            }
            match_rows.append(row)
            for move_event in case_moves:
                move = move_event["payload"]
                move_rows.append(
                    {
                        "schema_version": 1,
                        "evaluation_id": evaluation_id,
                        "evaluation_split": details["evaluation_split"],
                        "case_id": case_id,
                        "case_hash": move["case_hash"],
                        "case_set_hash": case_set_hash,
                        "seed": move["seed"],
                        "move_index": move["move_index"],
                        "player": move["player"],
                        "agent_id": move["agent_id"],
                        "agent_executable_hash": executable_hashes[
                            str(move["agent_id"])
                        ],
                        "state_id": move["state_id"],
                        "action": move["action"],
                        "terminated": move["terminated"],
                        "candidate_checkpoint_hash": details[
                            "candidate_checkpoint_hash"
                        ],
                        "source_metadata_sha256": metadata_hash,
                        **_source_fields(move_event, ledger_hash),
                    }
                )

        evaluation_rows.append(
            {
                "schema_version": 1,
                "evaluation_id": evaluation_id,
                "evaluation_split": details["evaluation_split"],
                "case_set_hash": case_set_hash,
                "candidate_checkpoint_hash": details["candidate_checkpoint_hash"],
                "population_hash": details["population_hash"],
                "executable_hashes": dict(details["executable_hashes"]),
                "checkpoint_index": finish["checkpoint_index"],
                "complete": finish["complete"],
                "valid_games": finish["valid_games"],
                "raw_matches": finish["raw_matches"],
                "env_steps": finish["env_steps"],
                "source_payload": dict(finish),
                "heldout_used_for_selection": details[
                    "heldout_used_for_selection"
                ],
                "source_metadata_sha256": metadata_hash,
                **_source_fields(finish_event, ledger_hash),
            }
        )

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    _reject_absolute_paths(match_rows)
    _reject_absolute_paths(move_rows)
    _reject_absolute_paths(evaluation_rows)
    _write_jsonl(output / "matches.jsonl", match_rows)
    _write_jsonl(output / "moves.jsonl", move_rows)
    _write_jsonl(output / "evaluations.jsonl", evaluation_rows)
    return {
        "evaluations": len(evaluation_rows),
        "matches": len(match_rows),
        "moves": len(move_rows),
    }


def _validate_metadata(evaluation_id: str, details: Any) -> None:
    required = {
        "evaluation_split",
        "candidate_checkpoint_hash",
        "case_contract",
        "population_hash",
        "executable_hashes",
        "heldout_used_for_selection",
    }
    optional = {"population_manifest"}
    if (
        not isinstance(details, Mapping)
        or not required.issubset(details)
        or set(details) - required - optional
    ):
        raise ValueError(f"evaluation {evaluation_id} metadata schema is invalid")
    if details["heldout_used_for_selection"] is not False:
        raise ValueError("held-out evaluation cannot be marked for selection")
    if not isinstance(details["executable_hashes"], Mapping):
        raise ValueError("executable hashes must be a mapping")
    contract = details["case_contract"]
    if (
        not isinstance(contract, Mapping)
        or set(contract) != {"game_config", "limits", "protocol_version"}
        or not isinstance(contract["game_config"], Mapping)
        or not isinstance(contract["limits"], Mapping)
        or not isinstance(contract["protocol_version"], str)
        or not contract["protocol_version"]
    ):
        raise ValueError("evidence case contract is invalid")
    if details["evaluation_split"] not in {"training", "heldout"}:
        raise ValueError("evaluation split must be training or heldout")
    hashes = (
        details["candidate_checkpoint_hash"],
        details["population_hash"],
        *details["executable_hashes"].values(),
    )
    if not all(
        isinstance(value, str)
        and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None
        for value in hashes
    ):
        raise ValueError("evidence metadata contains an invalid SHA-256")
    if not details["executable_hashes"] or not all(
        isinstance(key, str) and key for key in details["executable_hashes"]
    ):
        raise ValueError("executable hashes require non-empty agent IDs")


def _source_fields(event: Mapping[str, Any], ledger_hash: str) -> dict[str, Any]:
    return {
        "source_event": dict(event),
        "source_event_id": str(event["event_id"]),
        "source_event_hash": _hash_json(event),
        "source_ledger_sha256": ledger_hash,
    }


def _resolve_evidence_authority(
    *,
    evaluation_id: str,
    labels: Mapping[str, Any],
    finish: Mapping[str, Any],
    matches: list[dict[str, Any]],
    checkpoint_sources: Mapping[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    checkpoint_index = finish.get("checkpoint_index")
    if not isinstance(checkpoint_index, int) or isinstance(checkpoint_index, bool):
        raise ValueError(f"evaluation {evaluation_id} checkpoint index is invalid")
    candidate_facts: list[str] = []
    finish_candidate = finish.get("candidate_checkpoint_hash")
    if finish_candidate is not None:
        candidate_facts.append(str(finish_candidate))
    source_checkpoints = checkpoint_sources.get(checkpoint_index, ())
    if source_checkpoints:
        if len(source_checkpoints) != 1:
            raise ValueError(f"evaluation {evaluation_id} checkpoint source is ambiguous")
        checkpoint_hash = source_checkpoints[0]["payload"].get("checkpoint_hash")
        if checkpoint_hash is not None:
            candidate_facts.append(str(checkpoint_hash))
    if not candidate_facts or len(set(candidate_facts)) != 1:
        raise ValueError(
            f"evaluation {evaluation_id} candidate checkpoint is not source-anchored"
        )
    candidate_hash = candidate_facts[0]
    if re.fullmatch(r"sha256:[0-9a-f]{64}", candidate_hash) is None:
        raise ValueError(f"evaluation {evaluation_id} source candidate hash is invalid")

    observed_agents = {
        str(event["payload"][field])
        for event in matches
        for field in ("player_0", "player_1")
    }
    manifest = labels.get("population_manifest")
    if manifest is not None:
        authority = _authority_from_population_manifest(
            evaluation_id=evaluation_id,
            manifest=manifest,
            observed_agents=observed_agents,
            candidate_hash=candidate_hash,
            matches=matches,
        )
    else:
        required_source = {
            "evaluation_split",
            "population_hash",
            "heldout_used_for_selection",
            "opponent_checkpoint_hashes",
        }
        if not required_source.issubset(finish):
            raise ValueError(
                f"evaluation {evaluation_id} requires an immutable population manifest"
            )
        opponent_hashes = finish["opponent_checkpoint_hashes"]
        if not isinstance(opponent_hashes, Mapping):
            raise ValueError(f"evaluation {evaluation_id} opponent source is invalid")
        candidate_agents = observed_agents - set(map(str, opponent_hashes))
        source_candidate_id = finish.get("candidate_id")
        if source_candidate_id is not None:
            if candidate_agents != {str(source_candidate_id)}:
                raise ValueError(
                    f"evaluation {evaluation_id} source candidate identity disagrees"
                )
        elif len(candidate_agents) != 1:
            raise ValueError(
                f"evaluation {evaluation_id} source candidate identity is ambiguous"
            )
        executable_hashes = {
            str(agent_id): str(executable_hash)
            for agent_id, executable_hash in opponent_hashes.items()
        }
        executable_hashes[next(iter(candidate_agents))] = candidate_hash
        authority = {
            "evaluation_split": finish["evaluation_split"],
            "population_hash": finish["population_hash"],
            "heldout_used_for_selection": finish["heldout_used_for_selection"],
            "executable_hashes": executable_hashes,
        }

    resolved = {
        "evaluation_split": authority["evaluation_split"],
        "candidate_checkpoint_hash": candidate_hash,
        "case_contract": labels["case_contract"],
        "population_hash": authority["population_hash"],
        "executable_hashes": authority["executable_hashes"],
        "heldout_used_for_selection": authority["heldout_used_for_selection"],
    }
    for field in (
        "evaluation_split",
        "candidate_checkpoint_hash",
        "population_hash",
        "executable_hashes",
        "heldout_used_for_selection",
    ):
        if labels[field] != resolved[field]:
            raise ValueError(
                f"evaluation {evaluation_id} metadata {field} disagrees with authority"
            )
    _validate_metadata(evaluation_id, resolved)
    return resolved


def _authority_from_population_manifest(
    *,
    evaluation_id: str,
    manifest: Any,
    observed_agents: set[str],
    candidate_hash: str,
    matches: list[dict[str, Any]],
) -> dict[str, Any]:
    required = {"schema_version", "population_root", "protocol_version", "agents"}
    if not isinstance(manifest, Mapping) or set(manifest) != required:
        raise ValueError(f"evaluation {evaluation_id} immutable population is invalid")
    if manifest["schema_version"] != 1 or not isinstance(manifest["agents"], list):
        raise ValueError(f"evaluation {evaluation_id} immutable population is invalid")
    root = manifest["population_root"]
    if not isinstance(root, str) or not root or Path(root).is_absolute():
        raise ValueError(f"evaluation {evaluation_id} immutable population root is invalid")
    entries: dict[str, Mapping[str, Any]] = {}
    for raw in manifest["agents"]:
        if not isinstance(raw, Mapping):
            raise ValueError(f"evaluation {evaluation_id} immutable population is invalid")
        required_entry = {"agent_id", "kind", "content_hash", "roles"}
        if not required_entry.issubset(raw):
            raise ValueError(f"evaluation {evaluation_id} immutable population is invalid")
        agent_id = raw["agent_id"]
        executable_hash = raw["content_hash"]
        roles = raw["roles"]
        if (
            not isinstance(agent_id, str)
            or not agent_id
            or agent_id in entries
            or raw["kind"] not in {"train_human", "test_human"}
            or re.fullmatch(r"sha256:[0-9a-f]{64}", str(executable_hash)) is None
            or not isinstance(roles, list)
            or not set(roles).issubset({"player_0", "player_1"})
        ):
            raise ValueError(f"evaluation {evaluation_id} immutable population is invalid")
        entries[agent_id] = raw
    opponent_agents = observed_agents & set(entries)
    candidate_agents = observed_agents - opponent_agents
    if len(candidate_agents) != 1 or not opponent_agents:
        raise ValueError(f"evaluation {evaluation_id} population coverage disagrees")
    kinds = {str(entries[agent_id]["kind"]) for agent_id in opponent_agents}
    if len(kinds) != 1:
        raise ValueError(f"evaluation {evaluation_id} population split is ambiguous")
    for event in matches:
        for index, field in enumerate(("player_0", "player_1")):
            agent_id = str(event["payload"][field])
            if agent_id in entries and f"player_{index}" not in entries[agent_id]["roles"]:
                raise ValueError(f"evaluation {evaluation_id} population role disagrees")
    split = "training" if kinds == {"train_human"} else "heldout"
    executable_hashes = {
        agent_id: str(entries[agent_id]["content_hash"])
        for agent_id in opponent_agents
    }
    executable_hashes[next(iter(candidate_agents))] = candidate_hash
    return {
        "evaluation_split": split,
        "population_hash": _hash_json(manifest),
        "heldout_used_for_selection": False,
        "executable_hashes": executable_hashes,
    }


def _validate_source_completion(
    *,
    evaluation_id: str,
    details: Mapping[str, Any],
    finish: Mapping[str, Any],
    matches: list[dict[str, Any]],
    moves_by_case: Mapping[str, list[dict[str, Any]]],
    case_set_hash: str,
    checkpoint_sources: Mapping[int, list[dict[str, Any]]],
) -> Mapping[str, Any]:
    from rlbench.evaluation import EvaluationCase

    if finish.get("complete") is not True:
        raise ValueError(f"evaluation {evaluation_id} is incomplete")
    if finish.get("raw_matches") != len(matches):
        raise ValueError(f"evaluation {evaluation_id} raw match count disagrees")
    if finish.get("valid_games") != sum(
        event["payload"].get("valid") is True for event in matches
    ):
        raise ValueError(f"evaluation {evaluation_id} valid game count disagrees")
    env_steps = sum(len(event["payload"].get("actions", ())) for event in matches)
    if finish.get("env_steps") != env_steps:
        raise ValueError(f"evaluation {evaluation_id} environment steps disagree")
    checkpoint_indices = {
        event["payload"].get("checkpoint_index") for event in matches
    }
    if checkpoint_indices != {finish.get("checkpoint_index")}:
        raise ValueError(f"evaluation {evaluation_id} checkpoint indices disagree")
    details = _resolve_evidence_authority(
        evaluation_id=evaluation_id,
        labels=details,
        finish=finish,
        matches=matches,
        checkpoint_sources=checkpoint_sources,
    )
    observed_agents = {
        str(event["payload"][field])
        for event in matches
        for field in ("player_0", "player_1")
    } | {
        str(event["payload"]["agent_id"])
        for events in moves_by_case.values()
        for event in events
    }
    if set(details["executable_hashes"]) != observed_agents:
        raise ValueError(f"evaluation {evaluation_id} executable coverage disagrees")
    if details["candidate_checkpoint_hash"] not in set(
        details["executable_hashes"].values()
    ):
        raise ValueError(f"evaluation {evaluation_id} candidate executable is absent")
    contract = details["case_contract"]
    for event in matches:
        payload = event["payload"]
        expected_case = EvaluationCase.create(
            seed=int(payload["seed"]),
            player_0=str(payload["player_0"]),
            player_1=str(payload["player_1"]),
            player_0_hash=str(
                details["executable_hashes"][str(payload["player_0"])]
            ),
            player_1_hash=str(
                details["executable_hashes"][str(payload["player_1"])]
            ),
            game_config=contract["game_config"],
            limits=contract["limits"],
            protocol_version=str(contract["protocol_version"]),
        )
        if (
            payload.get("case_hash") != expected_case.content_hash
            or payload.get("case_id") != expected_case.case_id
        ):
            raise ValueError(f"evaluation {evaluation_id} case contract disagrees")
        case_moves = sorted(
            moves_by_case.get(str(payload["case_id"]), ()),
            key=lambda move: int(move["payload"]["move_index"]),
        )
        if [move["payload"].get("move_index") for move in case_moves] != list(
            range(len(case_moves))
        ):
            raise ValueError(f"evaluation {evaluation_id} move indices disagree")
        for move in case_moves:
            move_payload = move["payload"]
            overlaps = {
                "evaluation_id": evaluation_id,
                "case_id": payload["case_id"],
                "case_hash": payload["case_hash"],
                "seed": payload["seed"],
            }
            if any(
                move_payload.get(field) != expected
                for field, expected in overlaps.items()
            ):
                raise ValueError(f"evaluation {evaluation_id} move contract disagrees")
    cross_checks = {
        "evaluation_split": details["evaluation_split"],
        "candidate_checkpoint_hash": details["candidate_checkpoint_hash"],
        "population_hash": details["population_hash"],
        "heldout_used_for_selection": details["heldout_used_for_selection"],
        "case_set_hash": case_set_hash,
    }
    for field, expected in cross_checks.items():
        if field in finish and finish[field] != expected:
            raise ValueError(
                f"evaluation {evaluation_id} source {field} disagrees with metadata"
            )
    checkpoint_index = finish["checkpoint_index"]
    finish_has_candidate_anchor = "candidate_checkpoint_hash" in finish
    if checkpoint_sources:
        source_checkpoints = checkpoint_sources.get(checkpoint_index, ())
        if len(source_checkpoints) != 1:
            raise ValueError(
                f"evaluation {evaluation_id} checkpoint source is ambiguous"
            )
        checkpoint_payload = source_checkpoints[0]["payload"]
        if (
            checkpoint_payload.get("checkpoint_hash")
            != details["candidate_checkpoint_hash"]
        ):
            raise ValueError(
                f"evaluation {evaluation_id} candidate checkpoint is not source-anchored"
            )
        training_population_hash = checkpoint_payload.get(
            "training_population_hash"
        )
        if (
            details["evaluation_split"] == "training"
            and training_population_hash != details["population_hash"]
        ):
            raise ValueError(
                f"evaluation {evaluation_id} population is not source-anchored"
            )
    elif not finish_has_candidate_anchor:
        raise ValueError(
            f"evaluation {evaluation_id} has no source checkpoint anchor"
        )
    opponent_hashes = finish.get("opponent_checkpoint_hashes")
    if opponent_hashes is not None:
        candidate_agents = {
            agent_id
            for agent_id, executable_hash in details["executable_hashes"].items()
            if executable_hash == details["candidate_checkpoint_hash"]
        }
        expected_opponents = {
            agent_id: executable_hash
            for agent_id, executable_hash in details["executable_hashes"].items()
            if agent_id not in candidate_agents
        }
        if opponent_hashes != expected_opponents:
            raise ValueError(
                f"evaluation {evaluation_id} opponent checkpoint hashes disagree"
            )
    for event in matches:
        payload = event["payload"]
        match_overlaps = {
            "evaluation_split": details["evaluation_split"],
            "checkpoint_index": checkpoint_index,
            "case_set_hash": case_set_hash,
            "candidate_checkpoint_hash": details["candidate_checkpoint_hash"],
            "population_hash": details["population_hash"],
            "heldout_used_for_selection": details[
                "heldout_used_for_selection"
            ],
        }
        if any(
            field in payload and payload[field] != expected
            for field, expected in match_overlaps.items()
        ):
            raise ValueError(f"evaluation {evaluation_id} match contract disagrees")
        if "case_set_hash" in payload and payload["case_set_hash"] != case_set_hash:
            raise ValueError(f"evaluation {evaluation_id} case set disagrees")
    return details


def _reject_absolute_paths(value: Any) -> None:
    if isinstance(value, str) and _contains_unsafe_connection_detail(value):
        raise ValueError(
            "source evaluation payload contains an unsafe path or connection detail"
        )
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_absolute_paths(key)
            _reject_absolute_paths(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_absolute_paths(item)


def _contains_unsafe_connection_detail(value: str) -> bool:
    lowered = value.casefold()
    normalized_key = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    forbidden_keys = {
        "access_key",
        "api_key",
        "credential",
        "credentials",
        "host",
        "hostname",
        "password",
        "passwd",
        "private_key",
        "remote_host",
        "secret",
        "ssh_host",
        "token",
    }
    if normalized_key in forbidden_keys:
        return True
    if (
        value.startswith(("/", "\\\\", "//"))
        or re.match(r"^[a-zA-Z]:[\\/]", value) is not None
        or re.search(r"/(?:home|users|mnt|private|tmp)/", lowered) is not None
        or re.search(r"(?:file|ssh|sftp|scp)://", lowered) is not None
        or re.search(r"\b[^\s/@:]+@(?:[a-z0-9-]+\.)+[a-z]{2,}\b", lowered)
        is not None
    ):
        return True
    return False


def _hash_json(value: Any) -> str:
    encoded = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, allow_nan=False, separators=(",", ":"), sort_keys=True)
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
