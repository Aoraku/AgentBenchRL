from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from games.snakego import (
    IllegalActionError,
    ItemState,
    SnakeGoEngine,
    SnakeGoState,
    SnakeState,
    generate_items,
)


FIXTURE_ROOT = Path(__file__).parent / "fixtures"
CASES = json.loads((FIXTURE_ROOT / "cases.json").read_text(encoding="utf-8"))
EXPECTED = json.loads(
    (FIXTURE_ROOT / "conformance_hashes.json").read_text(encoding="utf-8")
)
REQUIRED_TAGS = {
    "movement",
    "growth",
    "suicide",
    "collision",
    "reversal",
    "items",
    "expiry",
    "split",
    "fire",
    "enclosure",
    "scoring",
    "alternation",
    "termination",
}


def _engine(case: dict[str, Any]) -> SnakeGoEngine:
    items = [
        ItemState(
            id=item["id"],
            x=item["x"],
            y=item["y"],
            spawn_round=item["spawn"],
            item_type=item["type"],
            param=item["param"],
            gotten_round=item.get("gotten", -1),
            owner_snake_id=item.get("owner", -1),
            expired=item.get("expired", False),
        )
        for item in case["items"]
    ]
    snakes = [
        SnakeState(
            id=snake["id"],
            camp=snake["camp"],
            coordinates=[tuple(coord) for coord in snake["coordinates"]],
            length_bank=snake.get("bank", 0),
            split_item_id=snake.get("split_item"),
            railgun_item_id=snake.get("railgun"),
        )
        for snake in case["snakes"]
    ]
    walls = np.full((16, 16), -1, dtype=np.int8)
    for x, y, camp in case["walls"]:
        walls[x, y] = camp
    return SnakeGoEngine.from_state(
        SnakeGoState(
            turn=case["turn"],
            current_player=case["player"],
            max_round=case["max_round"],
            snakes=snakes,
            items=items,
            walls=walls,
            first_item_round=case.get("first_item_round", -1),
            first_item_player=case.get("first_item_player", -1),
            next_snake_id=max((snake.id for snake in snakes), default=-1) + 1,
        )
    )


def _snapshot(engine: SnakeGoEngine) -> dict[str, Any]:
    state = engine.state
    active = None if state.terminated else engine.current_snake.id
    return {
        "turn": state.turn,
        "player": state.current_player,
        "active": active,
        "snakes": tuple(
            (
                snake.id,
                snake.camp,
                tuple(snake.coordinates),
                snake.length_bank,
                snake.split_item_id,
                snake.railgun_item_id,
            )
            for snake in state.snakes
        ),
        "walls": state.walls,
        "item_grid": state.item_grid,
        "available_items": tuple(
            (item.id, item.x, item.y, item.spawn_round, item.item_type, item.param)
            for item in state.items
            if not item.expired and item.owner_snake_id < 0
        ),
        "first": (state.first_item_round, state.first_item_player),
        "scores": engine.scores(),
        "terminated": state.terminated,
        "official_winner": state.official_winner,
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _json_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(child) for child in value]
    return value


def _digest(value: Any) -> str:
    payload = json.dumps(
        _json_value(value), allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def test_conformance_fixture_set_covers_every_required_rule() -> None:
    covered = {tag for case in CASES for tag in case["tags"]}
    assert covered >= REQUIRED_TAGS
    assert EXPECTED["schema_version"] == 1
    assert set(EXPECTED["cases"]) == {case["name"] for case in CASES}


@pytest.mark.parametrize("seed", [0, 7, 417, 2**31 - 1])
def test_seeded_item_schedule_matches_frozen_verified_fixture(seed: int) -> None:
    schedule = [
        (item.id, item.x, item.y, item.spawn_round, item.item_type, item.param)
        for item in generate_items(seed, max_round=512)
    ]
    assert _digest(schedule) == EXPECTED["item_schedules"][str(seed)]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_engine_matches_frozen_verified_transition_fixtures(
    case: dict[str, Any],
) -> None:
    expected = EXPECTED["cases"][case["name"]]
    engine = _engine(case)
    snapshots = [_digest(_snapshot(engine))]

    for action in case["actions"]:
        if expected["error"] is not None:
            assert expected["error"] == "illegal_action"
            with pytest.raises(IllegalActionError):
                engine.step(action)
            break
        engine.step(action)
        snapshots.append(_digest(_snapshot(engine)))

    assert snapshots == expected["snapshots"]
