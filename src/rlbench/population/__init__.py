"""Immutable opponent populations and isolated process agents."""

from .manifest import AgentKind, AgentProtocol, PopulationEntry, PopulationManifest
from .process_agent import (
    AgentInfrastructureError,
    ProcessAgent,
    ProcessMoveTimeout,
)
from .snakego_process import SnakeGoProcessPolicy

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
