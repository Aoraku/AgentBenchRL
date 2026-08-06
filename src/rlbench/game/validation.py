"""Runtime checks and copying support for discrete games."""

from __future__ import annotations

import copy
from typing import TypeVar

import numpy as np

from .protocol import DiscreteGame
from .types import BoardObservationSpec, DiscreteGameSpec, Observation, StepRecord


GameT = TypeVar("GameT", bound=DiscreteGame)


def clone_game(game: GameT) -> GameT:
    """Return a copy of ``game``, preferring a game-provided copier."""
    clone = getattr(game, "clone", None)
    copied = clone() if callable(clone) else copy.deepcopy(game)
    if not isinstance(copied, DiscreteGame):
        raise TypeError("clone must return a DiscreteGame")
    if copied is game:
        raise ValueError("clone must return an independent DiscreteGame")
    return copied


def validate_game(game: DiscreteGame) -> None:
    """Raise when a game does not satisfy the stable discrete contract."""
    if not isinstance(game, DiscreteGame):
        raise TypeError("game must implement the DiscreteGame protocol")

    spec = game.spec
    _validate_spec(spec)
    candidate = clone_game(game)
    candidate.reset(seed=0)

    for _ in range(spec.max_episode_steps):
        player = candidate.current_player()
        _validate_player(player, spec.players, "current_player")
        _validate_observation(candidate.observe(player), spec.observation_spec)
        mask = candidate.legal_action_mask()
        _validate_mask(mask, len(spec.action_names))
        action = int(np.flatnonzero(mask)[0])
        record = candidate.step(action)
        _validate_record(record, player, action)

        outcomes = tuple(candidate.outcome(index) for index in range(spec.players))
        if record.terminated:
            _validate_terminal_outcomes(outcomes, spec)
            return
        if any(outcome is not None for outcome in outcomes):
            raise ValueError("outcome must be None before termination")

    raise ValueError("game did not terminate within max_episode_steps")


def _validate_spec(spec: object) -> None:
    if not isinstance(spec, DiscreteGameSpec):
        raise TypeError("spec must be a DiscreteGameSpec")
    if not spec.name:
        raise ValueError("spec.name must not be empty")
    if spec.players != 2:
        raise ValueError("DiscreteGame requires exactly two players")
    if not spec.zero_sum:
        raise ValueError("DiscreteGame must declare zero_sum=True")
    if not spec.action_names or any(not name for name in spec.action_names):
        raise ValueError("spec.action_names must contain named actions")
    if spec.max_episode_steps < 1:
        raise ValueError("spec.max_episode_steps must be positive")
    _validate_observation_spec(spec.observation_spec)


def _validate_observation_spec(spec: object) -> None:
    if not isinstance(spec, BoardObservationSpec):
        raise TypeError("observation_spec must be a BoardObservationSpec")
    if not spec.plane_names or any(not name for name in spec.plane_names):
        raise ValueError("plane_names must contain named planes")
    if len(spec.board_shape) != 2 or any(size < 1 for size in spec.board_shape):
        raise ValueError("board_shape must contain two positive dimensions")
    if any(not name for name in spec.scalar_names):
        raise ValueError("scalar_names must contain named scalars")


def _validate_player(player: object, players: int, source: str) -> None:
    if not isinstance(player, int) or not 0 <= player < players:
        raise ValueError(f"{source} must return a valid player index")


def _validate_observation(observation: object, spec: BoardObservationSpec) -> None:
    if not isinstance(observation, Observation):
        raise TypeError("observe must return an Observation")
    expected_planes = (len(spec.plane_names), *spec.board_shape)
    if observation.planes.shape != expected_planes:
        raise ValueError(f"observation planes must have shape {expected_planes}")
    if observation.scalars.shape != (len(spec.scalar_names),):
        raise ValueError("observation scalars must match scalar_names")


def _validate_mask(mask: object, action_count: int) -> None:
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_:
        raise ValueError("legal_action_mask must be a boolean numpy array")
    if mask.ndim != 1 or mask.shape[0] != action_count:
        raise ValueError("legal_action_mask length must match action_names")
    if not mask.any():
        raise ValueError("legal_action_mask must permit at least one action")


def _validate_record(record: object, player: int, action: int) -> None:
    if not isinstance(record, StepRecord):
        raise TypeError("step must return a StepRecord")
    if record.player != player or record.action != action:
        raise ValueError("StepRecord must identify the applied player and action")


def _validate_terminal_outcomes(
    outcomes: tuple[float | None, ...], spec: DiscreteGameSpec
) -> None:
    if any(outcome is None for outcome in outcomes):
        raise ValueError("outcome must be defined for every player after termination")
    numeric_outcomes = tuple(float(outcome) for outcome in outcomes if outcome is not None)
    if any(outcome not in (-1.0, 0.0, 1.0) for outcome in numeric_outcomes):
        raise ValueError("terminal outcomes must be -1, 0, or 1")
    if spec.zero_sum and sum(numeric_outcomes) != 0.0:
        raise ValueError("terminal outcomes must be zero-sum")
