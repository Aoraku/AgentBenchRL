"""Deterministic TicTacToe implementation of the public game contract."""

from __future__ import annotations

import numpy as np

from rlbench.game import (
    BoardObservationSpec,
    DiscreteGameSpec,
    Observation,
    StepRecord,
)


class TicTacToe:
    """A two-player 3x3 game with observations relative to the viewer."""

    spec = DiscreteGameSpec(
        name="tic_tac_toe",
        players=2,
        zero_sum=True,
        action_names=tuple(str(index) for index in range(9)),
        observation_spec=BoardObservationSpec(
            plane_names=("own_stones", "opponent_stones"),
            board_shape=(3, 3),
            scalar_names=(),
        ),
        max_episode_steps=9,
    )

    def __init__(self) -> None:
        self.board = np.full((3, 3), -1, dtype=np.int8)
        self.player = 0
        self.terminal = False
        self.winner: int | None = None
        self.moves = 0

    def reset(self, seed: int) -> None:
        del seed
        self.board.fill(-1)
        self.player = 0
        self.terminal = False
        self.winner = None
        self.moves = 0

    def current_player(self) -> int:
        return self.player

    def observe(self, player: int) -> Observation:
        return Observation(
            planes=np.stack(
                (self.board == player, self.board == 1 - player), axis=0
            ).astype(np.float32),
            scalars=np.empty((0,), dtype=np.float32),
        )

    def legal_action_mask(self) -> np.ndarray:
        if self.terminal:
            return np.zeros(9, dtype=np.bool_)
        return (self.board.reshape(-1) == -1).astype(np.bool_)

    def step(self, action: int) -> StepRecord:
        if self.terminal:
            raise RuntimeError("cannot step a terminal TicTacToe game")
        if action < 0 or action >= 9 or not self.legal_action_mask()[action]:
            raise ValueError("illegal TicTacToe action")
        actor = self.player
        self.board.reshape(-1)[action] = actor
        self.moves += 1
        if self._has_line(actor):
            self.terminal = True
            self.winner = actor
        elif self.moves == 9:
            self.terminal = True
        self.player = 1 - actor
        return StepRecord(player=actor, action=action, terminated=self.terminal)

    def outcome(self, player: int) -> float | None:
        if not self.terminal:
            return None
        if self.winner is None:
            return 0.0
        return 1.0 if player == self.winner else -1.0

    def clone(self) -> TicTacToe:
        copied = TicTacToe()
        copied.board = self.board.copy()
        copied.player = self.player
        copied.terminal = self.terminal
        copied.winner = self.winner
        copied.moves = self.moves
        return copied

    def set_position(self, cells: str, *, player: int) -> None:
        """Load a compact test fixture using X, O, and '.' characters."""
        if len(cells) != 9 or any(cell not in "XO." for cell in cells):
            raise ValueError("position must contain nine X/O/. cells")
        values = {"X": 0, "O": 1, ".": -1}
        self.board = np.array([values[cell] for cell in cells], dtype=np.int8).reshape(
            3, 3
        )
        self.player = player
        self.moves = int(np.count_nonzero(self.board != -1))
        winners = [candidate for candidate in (0, 1) if self._has_line(candidate)]
        self.winner = winners[0] if winners else None
        self.terminal = self.winner is not None or self.moves == 9

    def _has_line(self, player: int) -> bool:
        occupied = self.board == player
        return bool(
            np.any(np.all(occupied, axis=0))
            or np.any(np.all(occupied, axis=1))
            or np.all(np.diag(occupied))
            or np.all(np.diag(np.fliplr(occupied)))
        )
