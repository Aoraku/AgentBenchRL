"""Run-manifest, checkpoint-lineage, budget, and small IO helpers for the CLI.

These game-neutral utilities are shared by the ``train``, ``evaluate``, and
``report`` subcommands. They are kept in one module so the subcommand modules
stay focused on orchestration.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import torch

from rlbench.config import ComposedConfig, canonical_config_hash
from rlbench.registry import ALGORITHMS, GAMES
from rlbench.telemetry import EventLedger


def create_manifest(config: ComposedConfig) -> dict[str, Any]:
    core = {
        "schema_version": 1,
        "run_id": str(uuid4()),
        "game": config.game,
        "algorithm": config.algorithm,
        "canonical_config": dict(config.canonical),
        "config_hash": config.config_hash,
        "source_hashes": dict(config.source_hashes),
        "software": software_facts(),
        "hardware": hardware_facts(),
    }
    encoded = json.dumps(
        core, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return {**core, "manifest_hash": f"sha256:{hashlib.sha256(encoded).hexdigest()}"}


def software_facts() -> dict[str, Any]:
    facts = {"python": platform.python_version()}
    for package in ("agentbench-rl-frame", "numpy", "torch", "tianshou"):
        try:
            facts[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            facts[package] = None
    return facts


def hardware_facts() -> dict[str, Any]:
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


def write_immutable_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
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


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid run manifest: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("run manifest must be a JSON object")
    validate_manifest(value)
    return value


def validate_manifest(manifest: Mapping[str, Any]) -> None:
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


def validate_checkpoint_lineage(
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
        if payload.get("checkpoint_hash") == sha256_file(recorded_path):
            return payload
        raise ValueError("checkpoint lineage content hash mismatch")
    raise ValueError("checkpoint lineage is not recorded for this run manifest")


def save_checkpoint_exclusive(trainer: Any, destination: Path) -> None:
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


def prior_checkpoint_path(
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
    validate_checkpoint_lineage(
        run_dir, prior_path, manifest=manifest, ledger=ledger
    )
    return prior_path


def validate_resume_manifest(
    manifest: Mapping[str, Any], config: ComposedConfig
) -> None:
    if (
        manifest.get("game") != config.game
        or manifest.get("algorithm") != config.algorithm
        or manifest.get("config_hash") != config.config_hash
    ):
        raise ValueError("resume configuration does not match the immutable run manifest")


def sha256_file(path: Path) -> str:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"checkpoint is not readable: {path}") from exc
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise ValueError("seeds must be comma-separated integers") from exc
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be non-negative")
    return seeds


def restore_event_backed_budgets(
    trainer: Any, *, ledger: EventLedger, run_id: str
) -> None:
    latest = ledger.latest_budget_counters(run_id=run_id)
    if latest is None:
        return
    latest.require_at_least(trainer.budgets)
    trainer.budgets = latest


def latest_gpu_hours(ledger: EventLedger, *, run_id: str) -> dict[str, float | None]:
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
            for key in ("learning_gpu_hours", "evaluation_gpu_hours", "gpu_hours")
        ):
            continue
        latest = {
            "learning": payload["learning_gpu_hours"],
            "evaluation": payload["evaluation_gpu_hours"],
            "total": payload["gpu_hours"],
        }
    return latest


def sum_optional_hours(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left) + float(right)


def evaluation_state_ids(
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


def case_set_hash(cases: Sequence[Any]) -> str:
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


def prior_complete_evaluation_states(
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
        candidate_id for index, candidate_id in candidates if index == prior_index
    ][-1]
    states = evaluation_state_ids(ledger, run_id=run_id, evaluation_id=evaluation_id)
    if not states:
        return None
    return evaluation_id, states


def print_json(value: Mapping[str, Any]) -> None:
    print(json.dumps(value, allow_nan=False, sort_keys=True))
