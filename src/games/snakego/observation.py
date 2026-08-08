"""Player-relative board encoding for SnakeGo."""

from __future__ import annotations

import numpy as np

from rlbench.game import Observation

from .spec import (
    FUTURE_ITEM_SLOTS,
    MAX_ANNOUNCED_ITEMS,
    PLANE_NAMES,
)
from .state import BOARD_SIZE, EMPTY, ItemState, SnakeGoState


_PLANE = {name: index for index, name in enumerate(PLANE_NAMES)}
_MAX_FUTURE_ITEMS_PER_CELL = 32
_EXACT_FUTURE_ITEMS_PER_CELL = 8
_CHECKSUM_MODULUS = 65_521


def canonical_coordinate(x: int, y: int, player: int) -> tuple[int, int]:
    if player == 0:
        return x, y
    return BOARD_SIZE - 1 - x, BOARD_SIZE - 1 - y


def encode_observation(
    state: SnakeGoState, player: int, scores: tuple[int, int]
) -> Observation:
    """Encode all public official state without exposing an absolute camp bit."""
    if player not in (0, 1):
        raise ValueError("player must be 0 or 1")
    planes = np.zeros((len(PLANE_NAMES), BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    active_ids = state.phase_snake_ids or ()
    active_id = (
        active_ids[state.phase_index]
        if not state.terminated and state.phase_index < len(active_ids)
        else None
    )
    active_snake = state.snake(active_id) if active_id is not None else None
    friendly_snakes = [snake for snake in state.snakes if snake.camp == player]
    opponent_snakes = [snake for snake in state.snakes if snake.camp != player]
    snake_slots = {
        snake.id: ("friendly", slot)
        for slot, snake in enumerate(friendly_snakes, start=1)
    }
    snake_slots.update(
        {
            snake.id: ("opponent", slot)
            for slot, snake in enumerate(opponent_snakes, start=1)
        }
    )

    for snake in state.snakes:
        if snake.id == active_id:
            head_plane = _PLANE["active_head"]
            body_plane = _PLANE["active_body"]
        elif snake.camp == player:
            head_plane = _PLANE["friendly_heads"]
            body_plane = _PLANE["friendly_bodies"]
        else:
            head_plane = _PLANE["opponent_heads"]
            body_plane = _PLANE["opponent_bodies"]
        length_plane = _PLANE[
            "friendly_body_length" if snake.camp == player else "opponent_body_length"
        ]
        side, slot = snake_slots[snake.id]
        slot_plane = _PLANE[f"{side}_snake_{slot}_body_order"]
        for index, (x, y) in enumerate(snake.coordinates):
            cx, cy = canonical_coordinate(x, y, player)
            planes[head_plane if index == 0 else body_plane, cx, cy] = 1.0
            planes[length_plane, cx, cy] = min(1.0, len(snake.coordinates) / 256.0)
            planes[slot_plane, cx, cy] = (
                len(snake.coordinates) - index
            ) / len(snake.coordinates)
            if snake.id == active_id:
                planes[_PLANE["active_body_order"], cx, cy] = (
                    len(snake.coordinates) - index
                ) / len(snake.coordinates)

    if active_snake is not None:
        tail_x, tail_y = canonical_coordinate(*active_snake.coordinates[-1], player)
        planes[_PLANE["active_tail"], tail_x, tail_y] = 1.0
        if len(active_snake.coordinates) >= 2:
            neck_x, neck_y = canonical_coordinate(*active_snake.coordinates[1], player)
            planes[_PLANE["active_neck"], neck_x, neck_y] = 1.0

    if active_snake is not None and active_snake.id == active_snake.camp:
        x, y = canonical_coordinate(*active_snake.coordinates[0], player)
        planes[_PLANE["active_founder_head"], x, y] = 1.0

    pending_snakes = []
    for snake_id in active_ids[state.phase_index + 1 :]:
        pending = state.snake(snake_id)
        if pending is not None and pending.camp == state.current_player:
            pending_snakes.append(pending)
    for slot, pending in enumerate(pending_snakes[:3], start=1):
        x, y = canonical_coordinate(*pending.coordinates[0], player)
        planes[_PLANE[f"pending_phase_{slot}_head"], x, y] = 1.0

    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            wall = int(state.walls[x, y])
            if wall == EMPTY:
                continue
            cx, cy = canonical_coordinate(x, y, player)
            if wall == player:
                plane = _PLANE["friendly_walls"]
            elif wall == 1 - player:
                plane = _PLANE["opponent_walls"]
            else:
                plane = _PLANE["blocked"]
            planes[plane, cx, cy] = 1.0

    future_items = []
    for item in state.items:
        if item.expired or item.owner_snake_id >= 0:
            continue
        cx, cy = canonical_coordinate(item.x, item.y, player)
        item_type = item.item_type if item.item_type in (0, 1, 2) else 1
        item_name = ("length", "split", "fire")[item_type]
        if int(state.item_grid[item.x, item.y]) == item.id:
            planes[_PLANE[f"active_{item_name}_items"], cx, cy] = 1.0
            planes[_PLANE["active_item_param"], cx, cy] = _normalized_item_param(
                item.item_type, item.param, state.max_round
            )
        elif item.spawn_round > state.turn:
            future_items.append(item)
            planes[_PLANE[f"future_{item_name}_items"], cx, cy] = 1.0
            normalized_time = min(
                1.0, max(0.0, (item.spawn_round - state.turn) / state.max_round)
            )
            old_time = planes[_PLANE["future_spawn_time"], cx, cy]
            if old_time == 0.0 or normalized_time < old_time:
                planes[_PLANE["future_spawn_time"], cx, cy] = normalized_time
                planes[_PLANE["future_item_param"], cx, cy] = (
                    _normalized_item_param(item.item_type, item.param, state.max_round)
                )
            planes[_PLANE["future_item_count"], cx, cy] = min(
                1.0,
                planes[_PLANE["future_item_count"], cx, cy]
                + 1.0 / _MAX_FUTURE_ITEMS_PER_CELL,
            )
            planes[_PLANE["future_latest_spawn_time"], cx, cy] = max(
                planes[_PLANE["future_latest_spawn_time"], cx, cy],
                normalized_time,
            )

    friendly_count = sum(snake.camp == player for snake in state.snakes)
    opponent_count = len(state.snakes) - friendly_count
    active_length = len(active_snake.coordinates) if active_snake is not None else 0
    active_bank = active_snake.length_bank if active_snake is not None else 0
    fire_inventory = (
        1.0
        if active_snake is not None and active_snake.railgun_item_id is not None
        else 0.0
    )
    split_inventory = (
        1.0
        if active_snake is not None and active_snake.split_item_id is not None
        else 0.0
    )
    score_margin = float(np.clip((scores[player] - scores[1 - player]) / 513.0, -1, 1))
    first_item = (
        0.0
        if state.first_item_player < 0
        else (1.0 if state.first_item_player == player else -1.0)
    )
    scalar_values = [
        np.clip(state.turn / state.max_round, 0.0, 1.0),
        np.clip((state.max_round - state.turn + 1) / state.max_round, 0.0, 1.0),
        np.clip(active_length / 256.0, 0.0, 1.0),
        np.clip(active_bank / 256.0, 0.0, 1.0),
        0.0,
        split_inventory,
        fire_inventory,
        friendly_count / 4.0,
        opponent_count / 4.0,
        score_margin,
        first_item,
        np.clip(state.phase_index / 3.0, 0.0, 1.0),
        np.clip(len(active_ids) / 4.0, 0.0, 1.0),
        (
            snake_slots[active_id][1] / 4.0
            if active_id is not None
            and active_id in snake_slots
            and snake_slots[active_id][0] == "friendly"
            else 0.0
        ),
    ]

    for snakes in (friendly_snakes, opponent_snakes):
        for slot in range(4):
            if slot >= len(snakes):
                scalar_values.extend((0.0,) * 5)
                continue
            snake = snakes[slot]
            scalar_values.extend(
                (
                    1.0,
                    np.clip(len(snake.coordinates) / 256.0, 0.0, 1.0),
                    np.clip(snake.length_bank / 256.0, 0.0, 1.0),
                    1.0 if snake.split_item_id is not None else 0.0,
                    1.0 if snake.railgun_item_id is not None else 0.0,
                )
            )

    future_items.sort(
        key=lambda item: (
            item.spawn_round,
            *canonical_coordinate(item.x, item.y, player),
            item.item_type,
            item.param,
            item.id,
        )
    )
    future_items_by_cell: dict[tuple[int, int], list[ItemState]] = {}
    for item in future_items:
        cell = canonical_coordinate(item.x, item.y, player)
        future_items_by_cell.setdefault(cell, []).append(item)
    for (x, y), cell_items in future_items_by_cell.items():
        for slot, item in enumerate(
            cell_items[:_EXACT_FUTURE_ITEMS_PER_CELL], start=1
        ):
            prefix = f"future_cell_{slot}"
            planes[_PLANE[f"{prefix}_present"], x, y] = 1.0
            item_type = item.item_type if item.item_type in (0, 1, 2) else 1
            item_name = ("length", "split", "fire")[item_type]
            planes[_PLANE[f"{prefix}_{item_name}"], x, y] = 1.0
            planes[_PLANE[f"{prefix}_spawn"], x, y] = np.clip(
                (item.spawn_round - state.turn) / state.max_round,
                0.0,
                1.0,
            )
            planes[_PLANE[f"{prefix}_param"], x, y] = _normalized_item_param(
                item.item_type, item.param, state.max_round
            )
    for slot in range(FUTURE_ITEM_SLOTS):
        if slot >= len(future_items):
            scalar_values.extend((0.0,) * 6)
            continue
        item = future_items[slot]
        x, y = canonical_coordinate(item.x, item.y, player)
        scalar_values.extend(
            (
                1.0,
                x / (BOARD_SIZE - 1),
                y / (BOARD_SIZE - 1),
                item.item_type / 2.0,
                np.clip(
                    (item.spawn_round - state.turn) / state.max_round,
                    0.0,
                    1.0,
                ),
                _normalized_item_param(item.item_type, item.param, state.max_round),
            )
        )

    overflow = future_items[FUTURE_ITEM_SLOTS:]
    scalar_values.extend(
        (
            min(1.0, len(overflow) / MAX_ANNOUNCED_ITEMS),
            _schedule_checksum(overflow, player, "spawn"),
            _schedule_checksum(overflow, player, "position"),
            _schedule_checksum(overflow, player, "type_param"),
        )
    )
    scalars = np.asarray(scalar_values, dtype=np.float32)
    return Observation(planes=planes, scalars=scalars)


def _normalized_item_param(item_type: int, param: int, max_round: int) -> float:
    scale = 5 if item_type == 0 else max_round
    return float(np.clip(param / max(1, scale), 0.0, 1.0))


def _schedule_checksum(items: list[ItemState], player: int, field: str) -> float:
    """Summarize slots beyond the four nearest official announcements."""
    if not items:
        return 0.0
    value = 0
    for index, item in enumerate(items, start=1):
        x, y = canonical_coordinate(item.x, item.y, player)
        if field == "spawn":
            component = item.spawn_round
        elif field == "position":
            component = x * BOARD_SIZE + y
        else:
            component = item.item_type * 1024 + item.param
        value = (value * 257 + component + 17 * index) % _CHECKSUM_MODULUS
    return value / (_CHECKSUM_MODULUS - 1)
