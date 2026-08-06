"""Explicit, reviewable registry of supported games and learning backends."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from typing import Any

from games.snakego import SnakeGoGame

from rlbench.algorithms.alphazero import AlphaZeroConfig, AlphaZeroTrainer
from rlbench.algorithms.ppo_tianshou import PPOConfig, PPOTrainer


GameFactory = Callable[[Mapping[str, Any] | None], Any]

GAMES: dict[str, GameFactory] = {"snakego": SnakeGoGame}
ALGORITHMS: dict[str, type[Any]] = {
    "alphazero": AlphaZeroTrainer,
    "ppo": PPOTrainer,
}
ALGORITHM_CONFIGS: dict[str, type[Any]] = {
    "alphazero": AlphaZeroConfig,
    "ppo": PPOConfig,
}


def game_factory(name: str, config: Mapping[str, Any] | None = None) -> Callable[[], Any]:
    """Return a zero-argument factory for one explicitly registered game."""
    try:
        game_type = GAMES[name]
    except KeyError as exc:
        raise ValueError(f"unknown game: {name}") from exc
    frozen_config = dict(config or {})
    return partial(game_type, frozen_config)
