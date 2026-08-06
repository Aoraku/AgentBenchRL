"""Public boundary for the framework-owned Tianshou PPO backend."""

from .config import PPOConfig
from .env import GymGameEnv, OpponentPolicy
from .network import MaskedActorCritic
from .trainer import (
    OpponentSnapshot,
    PPOActionDecision,
    PPOActionDistribution,
    PPOEvaluationMetrics,
    PPORecurrentState,
    PPOTrainer,
    PPOTrainingMetrics,
)

__all__ = [
    "GymGameEnv",
    "MaskedActorCritic",
    "OpponentPolicy",
    "OpponentSnapshot",
    "PPOActionDecision",
    "PPOActionDistribution",
    "PPOConfig",
    "PPOEvaluationMetrics",
    "PPORecurrentState",
    "PPOTrainer",
    "PPOTrainingMetrics",
]
