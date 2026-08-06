"""Content-addressed frozen evaluation case manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """One reproducible assignment of agents, sides, seed, and limits."""

    case_id: str
    seed: int
    player_0: str
    player_1: str
    player_0_hash: str
    player_1_hash: str
    game_config: Mapping[str, Any]
    limits: Mapping[str, Any]
    protocol_version: str
    content_hash: str

    @classmethod
    def create(
        cls,
        *,
        seed: int,
        player_0: str,
        player_1: str,
        player_0_hash: str,
        player_1_hash: str,
        game_config: Mapping[str, Any] | None = None,
        limits: Mapping[str, Any] | None = None,
        protocol_version: str = "1",
    ) -> EvaluationCase:
        if not player_0 or not player_1 or player_0 == player_1:
            raise ValueError("evaluation case requires two distinct agents")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        payload = {
            "seed": seed,
            "player_0": player_0,
            "player_1": player_1,
            "player_0_hash": player_0_hash,
            "player_1_hash": player_1_hash,
            "game_config": dict(game_config or {}),
            "limits": dict(limits or {}),
            "protocol_version": protocol_version,
        }
        encoded = json.dumps(
            payload, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        content_hash = f"sha256:{hashlib.sha256(encoded).hexdigest()}"
        return cls(
            case_id=content_hash.removeprefix("sha256:")[:24],
            seed=seed,
            player_0=player_0,
            player_1=player_1,
            player_0_hash=player_0_hash,
            player_1_hash=player_1_hash,
            game_config=_freeze_mapping(payload["game_config"]),
            limits=_freeze_mapping(payload["limits"]),
            protocol_version=protocol_version,
            content_hash=content_hash,
        )


def build_side_swapped_cases(
    *,
    candidate_id: str,
    candidate_hash: str,
    opponent_id: str,
    opponent_hash: str,
    seeds: Iterable[int],
    game_config: Mapping[str, Any] | None = None,
    limits: Mapping[str, Any] | None = None,
    protocol_version: str = "1",
) -> tuple[EvaluationCase, ...]:
    """Create a deterministic pair of opposite-side cases for every seed."""
    cases: list[EvaluationCase] = []
    for seed in seeds:
        cases.extend(
            (
                EvaluationCase.create(
                    seed=seed,
                    player_0=candidate_id,
                    player_1=opponent_id,
                    player_0_hash=candidate_hash,
                    player_1_hash=opponent_hash,
                    game_config=game_config,
                    limits=limits,
                    protocol_version=protocol_version,
                ),
                EvaluationCase.create(
                    seed=seed,
                    player_0=opponent_id,
                    player_1=candidate_id,
                    player_0_hash=opponent_hash,
                    player_1_hash=candidate_hash,
                    game_config=game_config,
                    limits=limits,
                    protocol_version=protocol_version,
                ),
            )
        )
    return tuple(cases)


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value
