"""Runtime plugin discovery for games plus an explicit backend registry.

Games are discovered by scanning the ``games`` namespace package for
``games/<game>/plugin.py`` modules that expose a :data:`PLUGIN` instance of
:class:`rlbench.plugins.GamePlugin`. The framework core never imports a
concrete game module directly; a new game is added by dropping in a
``games/<game>/`` plugin plus its configuration, with no change to framework
source.

Learning backends remain an explicit, reviewable registry: admitting a new
optimizer is a deliberate framework change, not a drop-in plugin.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable, Mapping
from functools import partial
from typing import Any

from rlbench.algorithms.alphazero import AlphaZeroConfig, AlphaZeroTrainer
from rlbench.algorithms.ppo_tianshou import PPOConfig, PPOTrainer
from rlbench.plugins import GamePlugin

GameFactory = Callable[[Mapping[str, Any] | None], Any]

ALGORITHMS: dict[str, type[Any]] = {
    "alphazero": AlphaZeroTrainer,
    "ppo": PPOTrainer,
}
ALGORITHM_CONFIGS: dict[str, type[Any]] = {
    "alphazero": AlphaZeroConfig,
    "ppo": PPOConfig,
}

_PLUGINS: dict[str, GamePlugin] | None = None


def _discover_plugins() -> dict[str, GamePlugin]:
    """Import every ``games/<game>/plugin.py`` and collect its declaration."""
    import games

    discovered: dict[str, GamePlugin] = {}
    for module_info in pkgutil.iter_modules(games.__path__, prefix="games."):
        if not module_info.ispkg:
            continue
        plugin_name = f"{module_info.name}.plugin"
        try:
            module = importlib.import_module(plugin_name)
        except ModuleNotFoundError as exc:
            # A subpackage without a plugin module is not a game plugin.
            if exc.name in (plugin_name, module_info.name):
                continue
            raise
        plugin = getattr(module, "PLUGIN", None)
        if not isinstance(plugin, GamePlugin):
            raise TypeError(
                f"{plugin_name} must expose a GamePlugin named PLUGIN"
            )
        if plugin.name in discovered:
            raise ValueError(f"duplicate game plugin: {plugin.name}")
        discovered[plugin.name] = plugin
    return discovered


def plugins() -> Mapping[str, GamePlugin]:
    """Return the discovered game plugins, scanning once and caching."""
    global _PLUGINS
    if _PLUGINS is None:
        _PLUGINS = _discover_plugins()
    return _PLUGINS


def plugin(name: str) -> GamePlugin:
    """Return one discovered plugin by registry name."""
    try:
        return plugins()[name]
    except KeyError as exc:
        raise ValueError(f"unknown game: {name}") from exc


class _GameRegistry(Mapping[str, GameFactory]):
    """A read-only mapping view over discovered game factories.

    Preserves the historical ``GAMES[name]`` / ``set(GAMES)`` interface while
    the underlying games are discovered as plugins rather than hardcoded.
    """

    def __getitem__(self, name: str) -> GameFactory:
        return plugin(name).game

    def __iter__(self):
        return iter(plugins())

    def __len__(self) -> int:
        return len(plugins())


GAMES: Mapping[str, GameFactory] = _GameRegistry()


def game_factory(
    name: str, config: Mapping[str, Any] | None = None
) -> Callable[[], Any]:
    """Return a zero-argument factory for one discovered game."""
    game_type = plugin(name).game
    frozen_config = dict(config or {})
    return partial(game_type, frozen_config)


def game_config_schema(name: str) -> Mapping[str, Any]:
    """Return the default game-configuration schema declared by a plugin."""
    return dict(plugin(name).config_schema)


def official_process_factory(
    name: str, protocol: str
) -> Callable[..., Any] | None:
    """Return the plugin-declared process policy factory for a protocol."""
    return plugin(name).official_protocols.get(protocol)
