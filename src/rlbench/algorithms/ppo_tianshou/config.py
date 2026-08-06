"""Validated configuration for the pinned Tianshou PPO backend."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PPOConfig:
    """Network, rollout, objective, and self-play settings."""

    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    hidden_size: int = 64
    conv_channels: int = 32
    recurrent: bool = False
    gru_hidden_size: int = 64
    vector_envs: int = 2
    episodes_per_collect: int = 8
    minibatch_size: int = 64
    update_repetitions: int = 4
    shaping_beta: float = 0.0
    score_scale: float = 1.0
    snapshot_interval: int = 1
    max_snapshots: int = 8
    device: str = "cpu"

    def __post_init__(self) -> None:
        positive = {
            "learning_rate": self.learning_rate,
            "hidden_size": self.hidden_size,
            "conv_channels": self.conv_channels,
            "gru_hidden_size": self.gru_hidden_size,
            "vector_envs": self.vector_envs,
            "episodes_per_collect": self.episodes_per_collect,
            "minibatch_size": self.minibatch_size,
            "update_repetitions": self.update_repetitions,
            "score_scale": self.score_scale,
            "snapshot_interval": self.snapshot_interval,
            "max_snapshots": self.max_snapshots,
        }
        if any(value <= 0 for value in positive.values()):
            raise ValueError("positive PPO settings must be greater than zero")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay cannot be negative")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")
        if self.clip_epsilon <= 0.0:
            raise ValueError("clip_epsilon must be positive")
        if self.value_coefficient < 0.0 or self.entropy_coefficient < 0.0:
            raise ValueError("loss coefficients cannot be negative")
        if self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        if self.shaping_beta < 0.0:
            raise ValueError("shaping_beta cannot be negative")
