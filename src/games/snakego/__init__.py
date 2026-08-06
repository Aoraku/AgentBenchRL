"""SnakeGo reference game plugin."""

from .engine import (
    EngineTransition,
    IllegalActionError,
    SnakeGoEngine,
    generate_items,
)
from .game import SnakeGoGame
from .protocol_agent import (
    OfficialGameOver,
    OfficialProtocolAdapter,
    load_alphazero_policy,
    run_official_agent,
)
from .spec import (
    PLANE_NAMES,
    SCALAR_NAMES,
    SNAKEGO_SPEC,
    SnakeGoSymmetry,
    canonical_action,
)
from .state import ItemState, SnakeGoState, SnakeState

__all__ = [
    "EngineTransition",
    "IllegalActionError",
    "ItemState",
    "OfficialGameOver",
    "OfficialProtocolAdapter",
    "PLANE_NAMES",
    "SCALAR_NAMES",
    "SNAKEGO_SPEC",
    "SnakeGoEngine",
    "SnakeGoGame",
    "SnakeGoState",
    "SnakeGoSymmetry",
    "SnakeState",
    "canonical_action",
    "generate_items",
    "load_alphazero_policy",
    "run_official_agent",
]
