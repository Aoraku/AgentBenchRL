"""SnakeGo reference game plugin."""

from .engine import (
    EngineTransition,
    IllegalActionError,
    SnakeGoEngine,
    generate_items,
)
from .game import SnakeGoGame
from .protocol_agent import (
    INFERENCE_BUNDLE_FORMAT,
    INFERENCE_BUNDLE_SCHEMA_VERSION,
    PPO_INFERENCE_BUNDLE_FORMAT,
    PPO_INFERENCE_BUNDLE_SCHEMA_VERSION,
    OfficialGameOver,
    OfficialProtocolAdapter,
    export_alphazero_inference_bundle,
    export_ppo_inference_bundle,
    load_alphazero_inference_bundle,
    load_alphazero_policy,
    load_ppo_inference_bundle,
    load_ppo_policy,
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
    "INFERENCE_BUNDLE_FORMAT",
    "INFERENCE_BUNDLE_SCHEMA_VERSION",
    "PPO_INFERENCE_BUNDLE_FORMAT",
    "PPO_INFERENCE_BUNDLE_SCHEMA_VERSION",
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
    "export_alphazero_inference_bundle",
    "export_ppo_inference_bundle",
    "generate_items",
    "load_alphazero_inference_bundle",
    "load_alphazero_policy",
    "load_ppo_inference_bundle",
    "load_ppo_policy",
    "run_official_agent",
]
