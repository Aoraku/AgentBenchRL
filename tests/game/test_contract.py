from __future__ import annotations

import numpy as np
import pytest

from rlbench.game import (
    BoardObservationSpec,
    DiscreteGameSpec,
    Observation,
    StepRecord,
    clone_game,
    validate_game,
)


class CounterGame:
    """A deterministic two-action game that ends after two decisions."""

    spec = DiscreteGameSpec(
        name="counter",
        players=2,
        zero_sum=True,
        action_names=("increment", "hold"),
        observation_spec=BoardObservationSpec(
            plane_names=("counter",),
            board_shape=(1, 1),
            scalar_names=("turn",),
        ),
        max_episode_steps=2,
    )

    def __init__(self) -> None:
        self.turn = 0
        self.total = 0
        self.terminal = False

    def reset(self, seed: int) -> None:
        self.turn = 0
        self.total = seed % 2
        self.terminal = False

    def current_player(self) -> int:
        return self.turn % 2

    def observe(self, player: int) -> Observation:
        return Observation(
            planes=np.array([[[self.total]]], dtype=np.float32),
            scalars=np.array([self.turn], dtype=np.float32),
        )

    def legal_action_mask(self) -> np.ndarray:
        return np.array([True, True], dtype=np.bool_)

    def step(self, action: int) -> StepRecord:
        actor = self.current_player()
        if action == 0:
            self.total += 1
        self.turn += 1
        self.terminal = self.turn == 2
        return StepRecord(player=actor, action=action, terminated=self.terminal)

    def outcome(self, player: int) -> float | None:
        if not self.terminal:
            return None
        return 1.0 if player == 0 else -1.0


def test_validate_game_accepts_a_complete_two_action_game() -> None:
    """Removing any required contract behavior makes validation fail."""
    validate_game(CounterGame())


def test_validate_game_rejects_mask_with_wrong_action_count() -> None:
    """A mask that cannot index every declared action is invalid."""

    class WrongMaskCounterGame(CounterGame):
        def legal_action_mask(self) -> np.ndarray:
            return np.array([True], dtype=np.bool_)

    with pytest.raises(ValueError, match="legal_action_mask"):
        validate_game(WrongMaskCounterGame())


def test_validate_game_rejects_terminal_outcomes_that_are_not_zero_sum() -> None:
    """A terminal result must sum to zero across players."""

    class NonZeroSumCounterGame(CounterGame):
        def outcome(self, player: int) -> float | None:
            if not self.terminal:
                return None
            return 1.0

    with pytest.raises(ValueError, match="zero-sum"):
        validate_game(NonZeroSumCounterGame())


def test_clone_game_returns_an_independent_copy() -> None:
    """Mutating a clone does not mutate the source game state."""
    game = CounterGame()
    game.reset(seed=7)
    clone = clone_game(game)

    clone.step(0)

    assert game.turn == 0
    assert game.total == 1
    assert clone.turn == 1
    assert clone.total == 2


def test_clone_game_rejects_an_override_that_returns_the_original_game() -> None:
    """Removing the identity check would allow caller state to be mutated."""

    class SelfCloningCounterGame(CounterGame):
        def clone(self) -> SelfCloningCounterGame:
            return self

    with pytest.raises(ValueError, match="independent"):
        clone_game(SelfCloningCounterGame())
