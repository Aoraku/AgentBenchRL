"""Strict deterministic YAML composition for reproducible local runs."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml

from rlbench.registry import ALGORITHM_CONFIGS, GAMES


class ConfigError(ValueError):
    """A configuration is ambiguous, unsafe, or outside the supported schema."""


_TRAINING_DEFAULTS: dict[str, Any] = {
    "seed": 0,
    "generations": 1,
    "iterations": 1,
    "self_play_episodes": 1,
    "training_steps": 1,
    "processes": 1,
    "checkpoint_every": 1,
}
_EVALUATION_DEFAULTS: dict[str, Any] = {
    "seeds": [0],
    "move_seconds": None,
}
_RESOURCE_DEFAULTS: dict[str, Any] = {"sample": True}
_RUN_DEFAULTS: dict[str, Any] = {"output_dir": "runs"}
_GAME_SCHEMAS: dict[str, dict[str, Any]] = {"snakego": {"max_round": 512}}
_TOP_LEVEL = {
    "game",
    "algorithm",
    "algorithms",
    "training",
    "evaluation",
    "resources",
    "run",
}
_FORBIDDEN_PARTS = {
    "connection",
    "credential",
    "endpoint",
    "hostname",
    "password",
    "private_key",
    "remote",
    "secret",
    "ssh",
    "token",
    "username",
}


@dataclass(frozen=True, slots=True)
class ComposedConfig:
    """Canonical values plus paths resolved only for local execution."""

    game: str
    algorithm: str
    canonical: Mapping[str, Any]
    config_hash: str
    source_hashes: Mapping[str, str]
    source_path: Path
    output_dir: Path

    def algorithm_settings(self) -> Any:
        return ALGORITHM_CONFIGS[self.algorithm](**dict(self.canonical["algorithm"]))


def compose_config(
    path: str | Path,
    *,
    game: str,
    algorithm: str,
    output_override: str | Path | None = None,
    caller_directory: str | Path | None = None,
) -> ComposedConfig:
    """Merge validated defaults with one YAML document in a stable order."""
    if game not in GAMES:
        raise ConfigError(f"unknown game: {game}")
    if algorithm not in ALGORITHM_CONFIGS:
        raise ConfigError(f"unknown algorithm: {algorithm}")
    source = Path(path).resolve()
    try:
        payload = source.read_bytes()
    except OSError as exc:
        raise ConfigError(f"configuration is not readable: {path}") from exc
    try:
        raw = yaml.safe_load(payload) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML configuration: {source.name}") from exc
    if not isinstance(raw, Mapping):
        raise ConfigError("configuration root must be a mapping")
    _reject_forbidden(raw)
    unknown_top = set(raw) - _TOP_LEVEL
    if unknown_top:
        raise ConfigError(f"unknown configuration key: {sorted(unknown_top)[0]}")

    defaults: dict[str, dict[str, Any]] = {
        "game": dict(_GAME_SCHEMAS[game]),
        "algorithm": asdict(ALGORITHM_CONFIGS[algorithm]()),
        "training": dict(_TRAINING_DEFAULTS),
        "evaluation": dict(_EVALUATION_DEFAULTS),
        "resources": dict(_RESOURCE_DEFAULTS),
        "run": dict(_RUN_DEFAULTS),
    }
    algorithm_overrides = raw.get("algorithms", {})
    if not isinstance(algorithm_overrides, Mapping):
        raise ConfigError("configuration section 'algorithms' must be a mapping")
    unknown_algorithms = set(algorithm_overrides) - set(ALGORITHM_CONFIGS)
    if unknown_algorithms:
        raise ConfigError(f"unknown algorithm: {sorted(unknown_algorithms)[0]}")
    for name, values in algorithm_overrides.items():
        if not isinstance(values, Mapping):
            raise ConfigError(f"algorithms.{name} must be a mapping")
        allowed = set(asdict(ALGORITHM_CONFIGS[name]()))
        unknown = set(values) - allowed
        if unknown:
            raise ConfigError(
                f"unknown configuration key: algorithms.{name}.{sorted(unknown)[0]}"
            )
    canonical: dict[str, Any] = {}
    for section, default_values in defaults.items():
        override = raw.get(section, {})
        if not isinstance(override, Mapping):
            raise ConfigError(f"configuration section {section!r} must be a mapping")
        unknown = set(override) - set(default_values)
        if unknown:
            key = sorted(unknown)[0]
            raise ConfigError(f"unknown configuration key: {section}.{key}")
        canonical[section] = {**default_values, **dict(override)}
    canonical["algorithm"].update(dict(algorithm_overrides.get(algorithm, {})))

    # Preserve hashes for PPO experiments created before optional controls
    # existed. Runtime dataclass defaults supply omitted values; experiments
    # that opt in record their selected values canonically.
    direct_algorithm = raw.get("algorithm", {})
    selected_algorithm = algorithm_overrides.get(algorithm, {})
    if (
        algorithm == "ppo"
        and "residual_blocks" not in direct_algorithm
        and "residual_blocks" not in selected_algorithm
    ):
        canonical["algorithm"].pop("residual_blocks")
    if (
        algorithm == "ppo"
        and "external_opponent_probability" not in direct_algorithm
        and "external_opponent_probability" not in selected_algorithm
    ):
        canonical["algorithm"].pop("external_opponent_probability")
    if (
        algorithm == "ppo"
        and "training_player" not in direct_algorithm
        and "training_player" not in selected_algorithm
    ):
        canonical["algorithm"].pop("training_player")

    _validate_controls(canonical)
    try:
        ALGORITHM_CONFIGS[algorithm](**canonical["algorithm"])
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"invalid {algorithm} algorithm configuration: {exc}") from exc
    configured_output = canonical["run"]["output_dir"]
    if not isinstance(configured_output, str) or not configured_output:
        raise ConfigError("run.output_dir must be a non-empty relative path")
    if Path(configured_output).is_absolute():
        raise ConfigError("run.output_dir must be relative in configuration")
    if output_override is None:
        output_dir = source.parent / configured_output
    else:
        override = Path(output_override)
        output_dir = override if override.is_absolute() else Path(
            caller_directory or Path.cwd()
        ) / override
    output_dir = output_dir.resolve()

    digest = canonical_config_hash(canonical)
    source_digest = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    return ComposedConfig(
        game=game,
        algorithm=algorithm,
        canonical=MappingProxyType(canonical),
        config_hash=digest,
        source_hashes=MappingProxyType({source.name: source_digest}),
        source_path=source,
        output_dir=output_dir,
    )


def canonical_config_hash(config: Mapping[str, Any]) -> str:
    """Hash one canonical JSON-compatible configuration mapping."""
    encoded = json.dumps(
        config, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _reject_forbidden(value: Any, prefix: str = "") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            normalized = key.lower().replace("-", "_")
            if any(part in normalized for part in _FORBIDDEN_PARTS):
                location = f"{prefix}.{key}" if prefix else key
                raise ConfigError(f"forbidden connection or credential field: {location}")
            _reject_forbidden(child, f"{prefix}.{key}" if prefix else key)
    elif isinstance(value, list):
        for child in value:
            _reject_forbidden(child, prefix)


def _validate_controls(config: Mapping[str, Mapping[str, Any]]) -> None:
    game = config["game"]
    max_round = game["max_round"]
    if not isinstance(max_round, int) or isinstance(max_round, bool) or max_round < 1:
        raise ConfigError("game.max_round must be a positive integer")
    training = config["training"]
    for key in (
        "seed",
        "generations",
        "iterations",
        "self_play_episodes",
        "training_steps",
        "processes",
        "checkpoint_every",
    ):
        value = training[key]
        minimum = 0 if key == "seed" else 1
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ConfigError(f"training.{key} must be an integer >= {minimum}")
    seeds = config["evaluation"]["seeds"]
    if not isinstance(seeds, list) or not seeds or any(
        not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
        for seed in seeds
    ):
        raise ConfigError("evaluation.seeds must be a non-empty list of non-negative integers")
    move_seconds = config["evaluation"]["move_seconds"]
    if move_seconds is not None and (
        not isinstance(move_seconds, (int, float))
        or isinstance(move_seconds, bool)
        or not math.isfinite(move_seconds)
        or move_seconds <= 0.0
    ):
        raise ConfigError("evaluation.move_seconds must be null or finite and positive")
    if not isinstance(config["resources"]["sample"], bool):
        raise ConfigError("resources.sample must be a boolean")
