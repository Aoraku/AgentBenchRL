"""Immutable values exchanged through the discrete game contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class BoardObservationSpec:
    """Declares a planes-first board observation."""

    plane_names: tuple[str, ...]
    board_shape: tuple[int, int]
    scalar_names: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DiscreteGameSpec:
    """Static metadata for a fixed-action, zero-sum game."""

    name: str
    players: int
    zero_sum: bool
    action_names: tuple[str, ...]
    observation_spec: BoardObservationSpec
    max_episode_steps: int


@dataclass(frozen=True, slots=True)
class Observation:
    """Information legally available to one player at a decision point."""

    planes: NDArray[np.float32]
    scalars: NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class StepRecord:
    """Immutable metadata for one applied game-agent decision."""

    player: int
    action: int
    terminated: bool
