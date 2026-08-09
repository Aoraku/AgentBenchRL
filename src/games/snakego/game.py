"""DiscreteGame adapter over the official-fidelity SnakeGo engine."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from rlbench.game import Observation, StepRecord

from .engine import SnakeGoEngine
from .observation import canonical_coordinate, encode_observation
from .spec import (
    IDENTITY_SYMMETRY,
    ROTATE_180_SYMMETRY,
    SNAKEGO_SPEC,
    SnakeGoSymmetry,
    canonical_action,
)
from .state import BOARD_SIZE, EMPTY, SnakeGoState


class SnakeGoGame:
    """Framework-neutral fixed-action SnakeGo game."""

    spec = SNAKEGO_SPEC

    def __init__(
        self,
        config: Mapping[str, object] | None = None,
        *,
        max_round: int = 512,
    ) -> None:
        configured_rounds = int((config or {}).get("max_round", max_round))
        self.max_round = configured_rounds
        self.engine = SnakeGoEngine(0, max_round=configured_rounds)

    @classmethod
    def from_state(cls, state: SnakeGoState) -> SnakeGoGame:
        return cls.from_engine(SnakeGoEngine.from_state(state))

    @classmethod
    def from_engine(cls, engine: SnakeGoEngine) -> SnakeGoGame:
        game = cls.__new__(cls)
        game.max_round = engine.state.max_round
        game.engine = engine
        return game

    @property
    def state(self) -> SnakeGoState:
        return self.engine.state

    def reset(self, seed: int) -> None:
        self.engine = SnakeGoEngine(seed, max_round=self.max_round)

    def current_player(self) -> int:
        return self.engine.current_player

    def observe(self, player: int) -> Observation:
        return encode_observation(self.state, player, self.engine.scores())

    def legal_action_mask(self) -> NDArray[np.bool_]:
        absolute = self.engine.legal_action_mask()
        player = self.current_player()
        return np.asarray(
            [absolute[canonical_action(action, player)] for action in range(6)],
            dtype=np.bool_,
        )

    def step(self, action: int) -> StepRecord:
        actor = self.current_player()
        if not isinstance(action, int) or not 0 <= action < 6:
            raise ValueError("action must be an integer from 0 through 5")
        absolute_action = canonical_action(action, actor)
        transition = self.engine.step(absolute_action)
        return StepRecord(player=actor, action=action, terminated=transition.terminated)

    def outcome(self, player: int) -> float | None:
        return self.engine.outcome(player)

    def score(self, player: int) -> float:
        return self.engine.score(player)

    def score_potential(self, player: int) -> float:
        if player not in (0, 1):
            raise ValueError("player must be 0 or 1")
        margin = self.score(player) - self.score(1 - player)
        return float(np.clip(margin / 513.0, -1.0, 1.0))

    def training_potential(self, player: int) -> float:
        """Expose score material plus banked growth as PPO shaping potential."""
        if player not in (0, 1):
            raise ValueError("player must be 0 or 1")

        def effective_score(side: int) -> float:
            banked_growth = sum(
                snake.length_bank for snake in self.state.snakes if snake.camp == side
            )
            return self.score(side) + 2.0 * banked_growth

        margin = effective_score(player) - effective_score(1 - player)
        return float(np.clip(margin / 513.0, -1.0, 1.0))

    @property
    def official_winner(self) -> int | None:
        return self.engine.official_winner

    def clone(self) -> SnakeGoGame:
        return SnakeGoGame.from_engine(self.engine.clone())

    def symmetries(self) -> Sequence[SnakeGoSymmetry]:
        return (IDENTITY_SYMMETRY, ROTATE_180_SYMMETRY)

    def encode_state_id(self, player: int) -> bytes:
        """Return a canonical exact-state digest independent of process-global IDs."""
        if player not in (0, 1):
            raise ValueError("player must be 0 or 1")
        state = self.state
        payload = bytearray()
        winner = (
            255
            if state.official_winner is None
            else int(state.official_winner != player)
        )
        payload.extend(
            struct.pack(
                ">HHBBBBI",
                state.turn,
                state.max_round,
                int(state.current_player != player),
                int(state.terminated),
                winner,
                state.phase_index,
                state.action_count,
            )
        )
        normalized_walls = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.uint8)
        normalized_walls[state.walls == player] = 1
        normalized_walls[state.walls == 1 - player] = 2
        normalized_walls[(state.walls != EMPTY) & (state.walls != 0) & (state.walls != 1)] = 3
        if player == 1:
            normalized_walls = np.rot90(normalized_walls, 2)
        payload.extend(normalized_walls.tobytes(order="C"))

        snake_indexes = {snake.id: index for index, snake in enumerate(state.snakes)}
        payload.extend(struct.pack(">B", len(state.snakes)))
        for snake in state.snakes:
            payload.extend(
                struct.pack(
                    ">BBHHBB",
                    int(snake.camp != player),
                    int(snake.id == snake.camp),
                    len(snake.coordinates),
                    snake.length_bank,
                    int(snake.split_item_id is not None),
                    int(snake.railgun_item_id is not None),
                )
            )
            for x, y in snake.coordinates:
                cx, cy = canonical_coordinate(x, y, player)
                payload.extend(bytes((cx, cy)))

        phase = state.phase_snake_ids or ()
        payload.extend(struct.pack(">B", len(phase)))
        for snake_id in phase:
            payload.extend(struct.pack(">B", snake_indexes.get(snake_id, 255)))

        payload.extend(struct.pack(">H", len(state.items)))
        for item in state.items:
            x, y = canonical_coordinate(item.x, item.y, player)
            owner_index = snake_indexes.get(item.owner_snake_id, 255)
            payload.extend(
                struct.pack(
                    ">HBBiBHiBBB",
                    item.id,
                    x,
                    y,
                    item.spawn_round,
                    item.item_type,
                    item.param,
                    item.gotten_round,
                    owner_index,
                    int(item.expired),
                    int(self.state.item_grid[item.x, item.y] == item.id),
                )
            )
        first_owner = (
            255
            if state.first_item_player < 0
            else int(state.first_item_player != player)
        )
        payload.extend(struct.pack(">iB", state.first_item_round, first_owner))
        return hashlib.sha256(payload).digest()

    def render_replay(self) -> Mapping[str, object]:
        score0, score1 = self.engine.scores()
        return {
            "turn": self.state.turn,
            "scores": (score0, score1),
            "official_winner": self.official_winner,
            "research_outcome": self.outcome(0),
            "actions": self.state.action_count,
        }
