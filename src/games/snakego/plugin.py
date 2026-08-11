"""SnakeGo plugin declaration discovered by the framework registry.

Exposing a single :data:`PLUGIN` is the entire contract a game must satisfy to
be admitted by ``rlbench.registry``. The framework never imports this module
directly; it is found by scanning the ``games`` namespace package.
"""

from __future__ import annotations

from rlbench.plugins import GamePlugin

from .game import SnakeGoGame
from .process_policy import SnakeGoProcessPolicy

PLUGIN = GamePlugin(
    name="snakego",
    game=SnakeGoGame,
    config_schema={"max_round": 512},
    official_protocols={"snakego_official": SnakeGoProcessPolicy},
)
