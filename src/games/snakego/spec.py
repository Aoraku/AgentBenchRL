"""Declarative SnakeGo action and observation metadata."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rlbench.game import BoardObservationSpec, DiscreteGameSpec, Observation


PLANE_NAMES = (
    "active_head",
    "active_body",
    "active_neck",
    "active_tail",
    "active_body_order",
    "friendly_body_length",
    "opponent_body_length",
    "friendly_heads",
    "friendly_bodies",
    "opponent_heads",
    "opponent_bodies",
    "friendly_walls",
    "opponent_walls",
    "blocked",
    "active_length_items",
    "active_split_items",
    "active_fire_items",
    "future_length_items",
    "future_split_items",
    "future_fire_items",
    "future_spawn_time",
    "active_founder_head",
    "pending_phase_1_head",
    "pending_phase_2_head",
    "pending_phase_3_head",
    "active_item_param",
    "future_item_param",
    "future_item_count",
    "future_latest_spawn_time",
)

FUTURE_ITEM_SLOTS = 16
MAX_ANNOUNCED_ITEMS = 512

_BASE_SCALAR_NAMES = (
    "round",
    "remaining_rounds",
    "active_length",
    "active_growth_bank",
    "inventory_length",
    "inventory_split",
    "inventory_fire",
    "friendly_snake_count",
    "opponent_snake_count",
    "score_margin",
    "first_item_owner",
)

_FUTURE_SLOT_SCALAR_NAMES = tuple(
    f"future_{slot}_{field}"
    for slot in range(1, FUTURE_ITEM_SLOTS + 1)
    for field in ("present", "x", "y", "type", "spawn", "param")
)

SCALAR_NAMES = (
    *_BASE_SCALAR_NAMES,
    *_FUTURE_SLOT_SCALAR_NAMES,
    "future_overflow_count",
    "future_overflow_spawn_checksum",
    "future_overflow_position_checksum",
    "future_overflow_type_param_checksum",
)

SNAKEGO_SPEC = DiscreteGameSpec(
    name="snakego",
    players=2,
    zero_sum=True,
    action_names=("right", "up", "left", "down", "fire", "split"),
    observation_spec=BoardObservationSpec(
        plane_names=PLANE_NAMES,
        board_shape=(16, 16),
        scalar_names=SCALAR_NAMES,
    ),
    max_episode_steps=512 * 8,
)


_ROTATED_ACTIONS = (2, 3, 0, 1, 4, 5)


def canonical_action(action: int, player: int) -> int:
    """Map an absolute operation to/from the player's canonical direction."""
    if player not in (0, 1):
        raise ValueError("player must be 0 or 1")
    if not 0 <= action < 6:
        raise ValueError("action must be between 0 and 5")
    return action if player == 0 else _ROTATED_ACTIONS[action]


@dataclass(frozen=True, slots=True)
class SnakeGoSymmetry:
    """A geometric training augmentation and its matching action map."""

    name: str
    rotate: bool

    def transform_observation(self, observation: Observation) -> Observation:
        planes = (
            np.rot90(observation.planes, 2, axes=(1, 2)).copy()
            if self.rotate
            else observation.planes.copy()
        )
        return Observation(planes=planes, scalars=observation.scalars.copy())

    def transform_action(self, action: int) -> int:
        return canonical_action(action, 1 if self.rotate else 0)


IDENTITY_SYMMETRY = SnakeGoSymmetry("identity", rotate=False)
ROTATE_180_SYMMETRY = SnakeGoSymmetry("rotate_180", rotate=True)
