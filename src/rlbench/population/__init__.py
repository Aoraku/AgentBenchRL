"""Immutable opponent populations and isolated process agents.

This framework-core package is game-neutral: it provides the line-JSON
``ProcessAgent`` boundary and content-addressed manifests. Game-specific
process policies (for example SnakeGo's official binary protocol) live in
their own ``games/<game>/`` plugin and are reached through the registry's
protocol handlers, never imported by the core.

``SnakeGoProcessPolicy`` remains importable here as a lazily-resolved
compatibility alias so existing callers keep working; the attribute is
resolved from the SnakeGo plugin on first access rather than imported at
module load time.
"""

from __future__ import annotations

from typing import Any

from .manifest import AgentKind, AgentProtocol, PopulationEntry, PopulationManifest
from .process_agent import (
    AgentInfrastructureError,
    ProcessAgent,
    ProcessMoveTimeout,
)

__all__ = [
    "AgentInfrastructureError",
    "AgentKind",
    "AgentProtocol",
    "PopulationEntry",
    "PopulationManifest",
    "ProcessAgent",
    "ProcessMoveTimeout",
    "SnakeGoProcessPolicy",
]


def __getattr__(name: str) -> Any:
    """Lazily expose game-plugin process policies without a static import."""
    if name == "SnakeGoProcessPolicy":
        from games.snakego.process_policy import SnakeGoProcessPolicy

        return SnakeGoProcessPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
