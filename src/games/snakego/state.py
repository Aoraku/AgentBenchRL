"""Compact mutable state values for the in-process SnakeGo engine."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


BOARD_SIZE = 16
EMPTY = -1


@dataclass(slots=True)
class ItemState:
    """One deterministic item announcement and its runtime ownership."""

    id: int
    x: int
    y: int
    spawn_round: int
    item_type: int
    param: int
    gotten_round: int = -1
    owner_snake_id: int = -1
    expired: bool = False

    def clone(self) -> ItemState:
        return ItemState(
            id=self.id,
            x=self.x,
            y=self.y,
            spawn_round=self.spawn_round,
            item_type=self.item_type,
            param=self.param,
            gotten_round=self.gotten_round,
            owner_snake_id=self.owner_snake_id,
            expired=self.expired,
        )


@dataclass(slots=True)
class SnakeState:
    """One live snake, ordered from head to tail."""

    id: int
    camp: int
    coordinates: list[tuple[int, int]]
    length_bank: int = 0
    split_item_id: int | None = None
    railgun_item_id: int | None = None

    def clone(self) -> SnakeState:
        return SnakeState(
            id=self.id,
            camp=self.camp,
            coordinates=self.coordinates.copy(),
            length_bank=self.length_bank,
            split_item_id=self.split_item_id,
            railgun_item_id=self.railgun_item_id,
        )


def _empty_grid(dtype: np.dtype[np.signedinteger] = np.dtype(np.int16)) -> NDArray:
    return np.full((BOARD_SIZE, BOARD_SIZE), EMPTY, dtype=dtype)


@dataclass(slots=True, eq=False)
class SnakeGoState:
    """All transition-relevant state, including the official phase snapshot."""

    turn: int
    current_player: int
    max_round: int
    snakes: list[SnakeState]
    items: list[ItemState] = field(default_factory=list)
    walls: NDArray[np.int8] = field(
        default_factory=lambda: _empty_grid(np.dtype(np.int8))
    )
    first_item_round: int = -1
    first_item_player: int = -1
    next_snake_id: int | None = None
    phase_snake_ids: list[int] | None = None
    phase_index: int = 0
    terminated: bool = False
    official_winner: int | None = None
    action_count: int = 0
    snake_grid: NDArray[np.int16] = field(init=False, repr=False)
    item_grid: NDArray[np.int32] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.walls = np.asarray(self.walls, dtype=np.int8).copy()
        if self.walls.shape != (BOARD_SIZE, BOARD_SIZE):
            raise ValueError("SnakeGo walls must have shape (16, 16)")
        self.snakes = [snake.clone() for snake in self.snakes]
        self.items = [item.clone() for item in self.items]
        if self.next_snake_id is None:
            self.next_snake_id = max((snake.id for snake in self.snakes), default=-1) + 1
        if self.phase_snake_ids is None:
            self.phase_snake_ids = [
                snake.id for snake in self.snakes if snake.camp == self.current_player
            ]
        else:
            self.phase_snake_ids = list(self.phase_snake_ids)
        self.rebuild_grids()

    def rebuild_grids(self) -> None:
        self.snake_grid = _empty_grid(np.dtype(np.int16))
        for snake in self.snakes:
            for x, y in snake.coordinates:
                if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
                    raise ValueError("live snake coordinates must be on the board")
                self.snake_grid[x, y] = snake.id

        self.item_grid = _empty_grid(np.dtype(np.int32))
        for item in self.items:
            if (
                not item.expired
                and item.owner_snake_id < 0
                and item.spawn_round <= self.turn < item.spawn_round + 16
            ):
                self.item_grid[item.x, item.y] = item.id

    def snake(self, snake_id: int) -> SnakeState | None:
        return next((snake for snake in self.snakes if snake.id == snake_id), None)

    def item(self, item_id: int) -> ItemState | None:
        return next((item for item in self.items if item.id == item_id), None)

    def clone(self) -> SnakeGoState:
        return SnakeGoState(
            turn=self.turn,
            current_player=self.current_player,
            max_round=self.max_round,
            snakes=[snake.clone() for snake in self.snakes],
            items=[item.clone() for item in self.items],
            walls=self.walls.copy(),
            first_item_round=self.first_item_round,
            first_item_player=self.first_item_player,
            next_snake_id=self.next_snake_id,
            phase_snake_ids=list(self.phase_snake_ids or ()),
            phase_index=self.phase_index,
            terminated=self.terminated,
            official_winner=self.official_winner,
            action_count=self.action_count,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SnakeGoState):
            return NotImplemented
        return (
            self.turn == other.turn
            and self.current_player == other.current_player
            and self.max_round == other.max_round
            and self.snakes == other.snakes
            and self.items == other.items
            and np.array_equal(self.walls, other.walls)
            and self.first_item_round == other.first_item_round
            and self.first_item_player == other.first_item_player
            and self.next_snake_id == other.next_snake_id
            and self.phase_snake_ids == other.phase_snake_ids
            and self.phase_index == other.phase_index
            and self.terminated == other.terminated
            and self.official_winner == other.official_winner
            and self.action_count == other.action_count
        )
