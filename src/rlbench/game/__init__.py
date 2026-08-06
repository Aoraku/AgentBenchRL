"""Stable contracts for framework-neutral discrete games."""

from .protocol import DiscreteGame
from .types import BoardObservationSpec, DiscreteGameSpec, Observation, StepRecord
from .validation import clone_game, validate_game

__all__ = [
    "BoardObservationSpec",
    "DiscreteGame",
    "DiscreteGameSpec",
    "Observation",
    "StepRecord",
    "clone_game",
    "validate_game",
]
