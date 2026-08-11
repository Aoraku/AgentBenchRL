"""Game-neutral plugin contract discovered at runtime from ``games/<game>/``.

The framework core never imports a concrete game. Each game ships a
``games/<game>/plugin.py`` module that exposes a single :data:`PLUGIN`
instance of :class:`GamePlugin`. The registry scans the ``games`` namespace
package, imports each ``plugin`` submodule lazily, and assembles the public
``GAMES`` mapping and protocol handlers from the declared plugins alone.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

GameFactory = Callable[[Mapping[str, Any] | None], Any]
ProcessPolicyFactory = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class GamePlugin:
    """Everything the framework needs to admit one game without importing it.

    Attributes:
        name: Stable registry key, matching the ``games/<name>/`` directory.
        game: A callable that builds one game instance from a config mapping
            (typically the six-method game class itself).
        config_schema: Default game-configuration values. Replaces the old
            hardcoded ``_GAME_SCHEMAS`` entry in ``config.py``.
        official_protocols: Optional map from a self-declared population
            ``protocol`` name to a factory constructing the matching stateful
            process policy. This removes the ``snakego_official`` special case
            from the CLI: a plugin declares its own protocol handlers.
    """

    name: str
    game: GameFactory
    config_schema: Mapping[str, Any] = field(default_factory=dict)
    official_protocols: Mapping[str, ProcessPolicyFactory] = field(
        default_factory=dict
    )
