"""The six-method protocol every simple discrete game implements."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .types import DiscreteGameSpec, Observation, StepRecord


@runtime_checkable
class DiscreteGame(Protocol):
    """A deterministic, two-player, fixed-discrete-action game."""

    spec: DiscreteGameSpec

    def reset(self, seed: int) -> None:
        """Create the initial state for a deterministic seed."""

    def current_player(self) -> int:
        """Return the player whose action is due."""

    def observe(self, player: int) -> Observation:
        """Return the information legally available to ``player``."""

    def legal_action_mask(self) -> NDArray[np.bool_]:
        """Return protocol-legal actions for the current player."""

    def step(self, action: int) -> StepRecord:
        """Apply one action and return its transition metadata."""

    def outcome(self, player: int) -> float | None:
        """Return terminal outcome for ``player``, or ``None`` in progress."""
