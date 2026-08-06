"""Validated configuration for AlphaZero search and learning."""

from __future__ import annotations

from dataclasses import dataclass
import re


_DEVICE = re.compile(r"(?:cpu|auto|cuda(?::(?:0|[1-9][0-9]*))?)\Z")


@dataclass(frozen=True, slots=True)
class AlphaZeroConfig:
    """Shared search, self-play, network, and optimizer settings."""

    simulations: int = 64
    c_puct: float = 1.5
    root_dirichlet_alpha: float = 0.3
    root_dirichlet_fraction: float = 0.25
    self_play_temperature: float = 1.0
    temperature_moves: int = 8
    channels: int = 32
    residual_blocks: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    replay_capacity: int = 10_000
    min_replay_size: int = 64
    gradient_clip_norm: float = 5.0
    mixed_precision: bool = True
    self_play_workers: int = 1
    inference_batch_size: int = 32
    device: str = "cpu"

    def __post_init__(self) -> None:
        positive = {
            "simulations": self.simulations,
            "c_puct": self.c_puct,
            "root_dirichlet_alpha": self.root_dirichlet_alpha,
            "channels": self.channels,
            "residual_blocks": self.residual_blocks,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "replay_capacity": self.replay_capacity,
            "gradient_clip_norm": self.gradient_clip_norm,
            "self_play_workers": self.self_play_workers,
            "inference_batch_size": self.inference_batch_size,
        }
        if any(value <= 0 for value in positive.values()):
            raise ValueError("positive AlphaZero settings must be greater than zero")
        if not 0.0 <= self.root_dirichlet_fraction <= 1.0:
            raise ValueError("root_dirichlet_fraction must be in [0, 1]")
        if self.self_play_temperature < 0.0:
            raise ValueError("self_play_temperature cannot be negative")
        if self.temperature_moves < 0:
            raise ValueError("temperature_moves cannot be negative")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay cannot be negative")
        if self.min_replay_size < 1 or self.min_replay_size > self.replay_capacity:
            raise ValueError("min_replay_size must be within replay capacity")
        if not isinstance(self.device, str) or not _DEVICE.fullmatch(self.device):
            raise ValueError("device must be cpu, auto, cuda, or cuda:<index>")
