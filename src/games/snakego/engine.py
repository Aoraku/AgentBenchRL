"""Modified port derived from AgentBench's MIT-licensed THUAC 2022 controller."""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
from numpy.typing import NDArray

from .state import BOARD_SIZE, EMPTY, ItemState, SnakeGoState, SnakeState


ITEM_EXPIRE_ROUNDS = 16
AUTO_GROWTH_ROUND = 8
MAX_FRIENDLY_SNAKES = 4
DIRECTIONS: tuple[tuple[int, int], ...] = ((1, 0), (0, 1), (-1, 0), (0, -1))


class IllegalActionError(ValueError):
    """Raised for malformed official operations, never for legal suicides."""


@dataclass(frozen=True, slots=True)
class EngineTransition:
    player: int
    action: int
    snake_id: int
    terminated: bool
    dead_snake_ids: tuple[int, ...] = ()
    solidified: bool = False
    solidified_cells: tuple[tuple[int, int], ...] = ()
    gotten_item_id: int | None = None
    new_snake_id: int | None = None


_SPAWN_CONFIGS = (
    (1, 20, (6, 6, 9, 9), (5, 0, 1)),
    (21, 40, (5, 5, 10, 10), (5, 0, 1)),
    (41, 60, (4, 4, 11, 11), (5, 0, 1)),
    (61, 64, (3, 3, 12, 12), (5, 0, 1)),
    (65, 80, (3, 3, 12, 12), (5, 0, 2)),
    (81, 100, (2, 2, 13, 13), (5, 0, 2)),
    (101, 120, (1, 1, 14, 14), (5, 0, 2)),
    (121, 384, (0, 0, 15, 15), (5, 0, 2)),
    (385, 512, (0, 0, 15, 15), (4, 0, 3)),
)


def generate_items(seed: int, max_round: int = 512) -> list[ItemState]:
    """Reproduce the official seeded item schedule without global RNG or IDs."""
    rng = random.Random(seed)
    occupied_at = np.full((BOARD_SIZE, BOARD_SIZE), -1024, dtype=np.int16)
    items: list[ItemState] = []
    for start, end, area, weights in _SPAWN_CONFIGS:
        for turn in range(start, end + 1):
            if rng.uniform(0, 1) > 0.25:
                continue
            x1, y1, x2, y2 = area
            x = rng.randint(x1, x2)
            y = rng.randint(y1, y2)
            while int(occupied_at[x, y]) + ITEM_EXPIRE_ROUNDS >= turn:
                x = rng.randint(x1, x2)
                y = rng.randint(y1, y2)
            occupied_at[x, y] = turn
            draw = rng.randint(0, sum(weights) - 1)
            if draw < weights[0] + weights[1]:
                item_type = 0
                param = rng.randint(1, 5)
            else:
                item_type = 2
                param = max_round
            items.append(
                ItemState(
                    id=len(items),
                    x=x,
                    y=y,
                    spawn_round=turn,
                    item_type=item_type,
                    param=param,
                )
            )
    return items


def initial_state(seed: int, max_round: int = 512) -> SnakeGoState:
    if max_round < 1 or max_round > 512:
        raise ValueError("max_round must be between 1 and 512")
    state = SnakeGoState(
        turn=1,
        current_player=0,
        max_round=max_round,
        snakes=[
            SnakeState(0, 0, [(0, BOARD_SIZE - 1)]),
            SnakeState(1, 1, [(BOARD_SIZE - 1, 0)]),
        ],
        items=generate_items(seed, max_round),
        next_snake_id=2,
        phase_snake_ids=[0],
    )
    _preprocess_round(state)
    return state


class SnakeGoEngine:
    """One-step-per-active-snake rules engine with official player phasing."""

    def __init__(self, seed: int = 0, *, max_round: int = 512) -> None:
        self.state = initial_state(seed, max_round)

    @classmethod
    def from_state(cls, state: SnakeGoState) -> SnakeGoEngine:
        engine = cls.__new__(cls)
        engine.state = state
        state.rebuild_grids()
        engine._seek_live_active()
        return engine

    @classmethod
    def from_items(
        cls, items: list[ItemState], *, max_round: int = 512
    ) -> SnakeGoEngine:
        """Initialize from the item announcement sent to an official agent."""
        state = SnakeGoState(
            turn=1,
            current_player=0,
            max_round=max_round,
            snakes=[
                SnakeState(0, 0, [(0, BOARD_SIZE - 1)]),
                SnakeState(1, 1, [(BOARD_SIZE - 1, 0)]),
            ],
            items=items,
            next_snake_id=2,
            phase_snake_ids=[0],
        )
        _preprocess_round(state)
        return cls.from_state(state)

    @property
    def current_player(self) -> int:
        return self.state.current_player

    @property
    def current_snake(self) -> SnakeState:
        if self.state.terminated:
            raise RuntimeError("SnakeGo game is terminated")
        active_ids = self.state.phase_snake_ids or ()
        if self.state.phase_index >= len(active_ids):
            raise RuntimeError("SnakeGo phase has no active snake")
        snake = self.state.snake(active_ids[self.state.phase_index])
        if snake is None:
            raise RuntimeError("SnakeGo active snake is missing")
        return snake

    def legal_action_mask(self) -> NDArray[np.bool_]:
        if self.state.terminated:
            return np.zeros(6, dtype=np.bool_)
        snake = self.current_snake
        mask = np.ones(6, dtype=np.bool_)
        if self._reversal_action(snake) is not None:
            mask[self._reversal_action(snake)] = False
        mask[4] = snake.railgun_item_id is not None and len(snake.coordinates) > 1
        mask[5] = (
            len(snake.coordinates) > 1
            and sum(other.camp == snake.camp for other in self.state.snakes)
            < MAX_FRIENDLY_SNAKES
        )
        return mask

    def step(self, action: int) -> EngineTransition:
        if self.state.terminated:
            raise RuntimeError("SnakeGo game is terminated")
        if not isinstance(action, int) or not 0 <= action < 6:
            raise IllegalActionError("action must be an integer from 0 through 5")
        mask = self.legal_action_mask()
        if not bool(mask[action]):
            if action < 4:
                reason = "forbidden reversal"
            elif action == 4:
                reason = "fire requires a railgun and length greater than one"
            else:
                reason = "split requires length greater than one and fewer than four snakes"
            raise IllegalActionError(reason)

        actor = self.state.current_player
        snake_id = self.current_snake.id
        if action < 4:
            result = self._move(action)
        elif action == 4:
            result = self._fire()
        else:
            result = self._split()
        self.state.action_count += 1
        self._advance_active_snake()
        return EngineTransition(
            player=actor,
            action=action,
            snake_id=snake_id,
            terminated=self.state.terminated,
            **result,
        )

    def scores(self) -> tuple[int, int]:
        scores = [0, 0]
        for snake in self.state.snakes:
            scores[snake.camp] += 2 * len(snake.coordinates)
        scores[0] += 2 * int(np.count_nonzero(self.state.walls == 0))
        scores[1] += 2 * int(np.count_nonzero(self.state.walls == 1))
        if self.state.first_item_player >= 0:
            scores[self.state.first_item_player] += 1
        return scores[0], scores[1]

    def score(self, player: int) -> float:
        if player not in (0, 1):
            raise ValueError("player must be 0 or 1")
        return float(self.scores()[player])

    def outcome(self, player: int) -> float | None:
        if player not in (0, 1):
            raise ValueError("player must be 0 or 1")
        if not self.state.terminated:
            return None
        score0, score1 = self.scores()
        if score0 == score1:
            return 0.0
        winner = 0 if score0 > score1 else 1
        return 1.0 if player == winner else -1.0

    @property
    def official_winner(self) -> int | None:
        return self.state.official_winner

    def clone(self) -> SnakeGoEngine:
        return SnakeGoEngine.from_state(self.state.clone())

    def _reversal_action(self, snake: SnakeState) -> int | None:
        coords = snake.coordinates
        auto_grow = self.state.turn <= AUTO_GROWTH_ROUND and snake.camp == snake.id
        reversal_forbidden = len(coords) > 2 or (
            len(coords) == 2 and (auto_grow or snake.length_bank > 0)
        )
        if not reversal_forbidden:
            return None
        hx, hy = coords[0]
        nx, ny = coords[1]
        return DIRECTIONS.index((nx - hx, ny - hy))

    def _move(self, direction: int) -> dict[str, object]:
        state = self.state
        snake = self.current_snake
        snake_index = state.snakes.index(snake)
        old_coordinates = snake.coordinates
        for x, y in old_coordinates:
            state.snake_grid[x, y] = EMPTY
        state.snakes.pop(snake_index)

        dx, dy = DIRECTIONS[direction]
        x, y = old_coordinates[0][0] + dx, old_coordinates[0][1] + dy
        new_coordinates = [(x, y)] if len(old_coordinates) == 1 else [
            (x, y),
            *old_coordinates[:-1],
        ]
        if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE) or state.walls[x, y] != EMPTY:
            return {"dead_snake_ids": (snake.id,)}

        auto_grow = state.turn <= AUTO_GROWTH_ROUND and snake.camp == snake.id
        if auto_grow:
            new_coordinates.append(old_coordinates[-1])
        elif snake.length_bank:
            snake.length_bank -= 1
            new_coordinates.append(old_coordinates[-1])
        snake.coordinates = new_coordinates

        gotten_item_id: int | None = None
        item_id = int(state.item_grid[x, y])
        if item_id != EMPTY:
            gotten_item_id = item_id
            self._give_item(snake, item_id)

        collision_index = next(
            (index for index in range(1, len(new_coordinates)) if new_coordinates[index] == (x, y)),
            None,
        )
        if collision_index is not None:
            solid = new_coordinates[:collision_index]
            extra = _enclosed_cells(solid)
            for coordinate in new_coordinates[collision_index:]:
                if coordinate in extra:
                    solid.append(coordinate)
                    extra.remove(coordinate)
            dead_ids = [snake.id]
            for ex, ey in extra.copy():
                enclosed_id = int(state.snake_grid[ex, ey])
                if enclosed_id != EMPTY:
                    dead_ids.append(enclosed_id)
                    self._delete_snake(enclosed_id)
            all_solid = sorted((*extra, *solid))
            for sx, sy in all_solid:
                state.walls[sx, sy] = state.current_player
            return {
                "dead_snake_ids": tuple(dead_ids),
                "solidified": True,
                "solidified_cells": tuple(all_solid),
                "gotten_item_id": gotten_item_id,
            }

        if state.snake_grid[x, y] != EMPTY:
            return {
                "dead_snake_ids": (snake.id,),
                "gotten_item_id": gotten_item_id,
            }

        state.snakes.insert(snake_index, snake)
        for sx, sy in snake.coordinates:
            state.snake_grid[sx, sy] = snake.id
        return {"gotten_item_id": gotten_item_id}

    def _split(self) -> dict[str, object]:
        state = self.state
        snake = self.current_snake
        index = state.snakes.index(snake)
        middle = (len(snake.coordinates) + 1) // 2
        tail = list(reversed(snake.coordinates[middle:]))
        snake.coordinates = snake.coordinates[:middle]
        child_id = int(state.next_snake_id)
        state.next_snake_id = child_id + 1
        child = SnakeState(
            id=child_id,
            camp=snake.camp,
            coordinates=tail,
            length_bank=snake.length_bank,
        )
        snake.length_bank = 0
        if snake.split_item_id is not None:
            split_item = state.item(snake.split_item_id)
            if split_item is not None:
                split_item.owner_snake_id = -1
                split_item.expired = True
            snake.split_item_id = None
        state.snakes.insert(index + 1, child)
        for x, y in child.coordinates:
            state.snake_grid[x, y] = child.id
        return {"new_snake_id": child.id}

    def _fire(self) -> dict[str, object]:
        snake = self.current_snake
        item_id = snake.railgun_item_id
        snake.railgun_item_id = None
        if item_id is not None:
            item = self.state.item(item_id)
            if item is not None:
                item.owner_snake_id = -1
                item.expired = True
        head_x, head_y = snake.coordinates[0]
        neck_x, neck_y = snake.coordinates[1]
        dx, dy = head_x - neck_x, head_y - neck_y
        x, y = head_x + dx, head_y + dy
        while 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
            self.state.walls[x, y] = EMPTY
            x += dx
            y += dy
        return {}

    def _give_item(self, snake: SnakeState, item_id: int) -> None:
        item = self.state.item(item_id)
        if item is None:
            raise RuntimeError("item grid references a missing item")
        if self.state.first_item_round == -1 or (
            self.state.first_item_round == self.state.turn
            and self.state.first_item_player == 0
        ):
            self.state.first_item_round = self.state.turn
            self.state.first_item_player = snake.camp
        item.gotten_round = self.state.turn
        item.owner_snake_id = snake.id
        self.state.item_grid[item.x, item.y] = EMPTY
        if item.item_type == 0:
            snake.length_bank += item.param
        elif item.item_type == 1:
            if snake.split_item_id is not None:
                replaced = self.state.item(snake.split_item_id)
                if replaced is not None:
                    replaced.owner_snake_id = -1
                    replaced.expired = True
            snake.split_item_id = item.id
        else:
            if snake.railgun_item_id is not None:
                replaced = self.state.item(snake.railgun_item_id)
                if replaced is not None:
                    replaced.owner_snake_id = -1
                    replaced.expired = True
            snake.railgun_item_id = item.id

    def _delete_snake(self, snake_id: int) -> None:
        snake = self.state.snake(snake_id)
        if snake is None:
            return
        for x, y in snake.coordinates:
            self.state.snake_grid[x, y] = EMPTY
        self.state.snakes.remove(snake)

    def _advance_active_snake(self) -> None:
        self.state.phase_index += 1
        self._seek_live_active()

    def _seek_live_active(self) -> None:
        state = self.state
        while not state.terminated:
            phase = state.phase_snake_ids or []
            while state.phase_index < len(phase):
                snake = state.snake(phase[state.phase_index])
                if snake is not None and snake.camp == state.current_player:
                    return
                state.phase_index += 1

            state.current_player = 1 - state.current_player
            if state.current_player == 0:
                state.turn += 1
                self._settle()
                if state.terminated:
                    return
                _preprocess_round(state)
            state.phase_snake_ids = [
                snake.id for snake in state.snakes if snake.camp == state.current_player
            ]
            state.phase_index = 0

    def _settle(self) -> None:
        state = self.state
        no_snakes = not state.snakes
        if no_snakes or state.turn > state.max_round:
            state.terminated = True
            score0, score1 = self.scores()
            state.official_winner = 0 if score0 >= score1 else 1


def _preprocess_round(state: SnakeGoState) -> None:
    for item in state.items:
        if (
            not item.expired
            and item.owner_snake_id < 0
            and item.spawn_round <= state.turn - ITEM_EXPIRE_ROUNDS
        ):
            if state.item_grid[item.x, item.y] == item.id:
                state.item_grid[item.x, item.y] = EMPTY
            item.expired = True
        if (
            not item.expired
            and item.owner_snake_id < 0
            and item.spawn_round == state.turn
        ):
            snake_id = int(state.snake_grid[item.x, item.y])
            if snake_id != EMPTY:
                snake = state.snake(snake_id)
                if snake is not None:
                    _give_item_at_preprocess(state, snake, item)
            else:
                state.item_grid[item.x, item.y] = item.id

    for snake in state.snakes:
        for attribute in ("split_item_id", "railgun_item_id"):
            item_id = getattr(snake, attribute)
            if item_id is None:
                continue
            item = state.item(item_id)
            if item is not None and state.turn - item.gotten_round > item.param:
                setattr(snake, attribute, None)
                item.owner_snake_id = -1
                item.expired = True


def _give_item_at_preprocess(
    state: SnakeGoState, snake: SnakeState, item: ItemState
) -> None:
    if state.first_item_round == -1 or (
        state.first_item_round == state.turn and state.first_item_player == 0
    ):
        state.first_item_round = state.turn
        state.first_item_player = snake.camp
    item.gotten_round = state.turn
    item.owner_snake_id = snake.id
    state.item_grid[item.x, item.y] = EMPTY
    if item.item_type == 0:
        snake.length_bank += item.param
    elif item.item_type == 1:
        if snake.split_item_id is not None:
            replaced = state.item(snake.split_item_id)
            if replaced is not None:
                replaced.owner_snake_id = -1
                replaced.expired = True
        snake.split_item_id = item.id
    else:
        if snake.railgun_item_id is not None:
            replaced = state.item(snake.railgun_item_id)
            if replaced is not None:
                replaced.owner_snake_id = -1
                replaced.expired = True
        snake.railgun_item_id = item.id


def _enclosed_cells(boundary: list[tuple[int, int]]) -> list[tuple[int, int]]:
    table = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    for x, y in boundary:
        table[x, y] = -1
    inner = [True, True, True]

    def convert_direction(
        current: tuple[int, int], previous: tuple[int, int]
    ) -> int:
        x = current[0] - previous[0]
        y = current[1] - previous[1]
        return x + 1 if y == 0 else y + 2

    def flood(start_x: int, start_y: int, color: int) -> None:
        stack = [(start_x, start_y)]
        while stack:
            x, y = stack.pop()
            if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
                inner[color] = False
                continue
            if table[x, y] != 0:
                continue
            table[x, y] = color
            stack.extend(((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)))

    for index, coordinate in enumerate(boundary):
        direction = convert_direction(coordinate, boundary[index - 1])
        left = (direction + 3) % 4
        right = (direction + 1) % 4
        flood(
            coordinate[0] + DIRECTIONS[left][0],
            coordinate[1] + DIRECTIONS[left][1],
            1,
        )
        flood(
            coordinate[0] + DIRECTIONS[right][0],
            coordinate[1] + DIRECTIONS[right][1],
            2,
        )
    return [
        (x, y)
        for color in (1, 2)
        if inner[color]
        for x in range(BOARD_SIZE)
        for y in range(BOARD_SIZE)
        if table[x, y] == color
    ]
