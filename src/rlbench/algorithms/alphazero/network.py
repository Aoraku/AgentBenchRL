"""Compact residual policy-value network and batched inference boundary."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from rlbench.game import BoardObservationSpec, DiscreteGameSpec, Observation

from .config import AlphaZeroConfig


class BatchEvaluator(Protocol):
    """Inference boundary used by search and batched self-play services."""

    def evaluate_batch(
        self,
        observations: Sequence[Observation],
        legal_masks: Sequence[NDArray[np.bool_]],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Return policy logits and current-player values for one batch."""


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(inputs + self.body(inputs))


class PolicyValueNet(nn.Module):
    """A small board residual network with masked policy and tanh value heads."""

    def __init__(
        self,
        observation_spec: BoardObservationSpec,
        action_count: int,
        *,
        channels: int = 32,
        residual_blocks: int = 2,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        if action_count < 1:
            raise ValueError("action_count must be positive")
        height, width = observation_spec.board_shape
        self.observation_spec = observation_spec
        self.action_count = action_count
        self.device = torch.device(device)
        input_channels = len(observation_spec.plane_names) + len(
            observation_spec.scalar_names
        )
        self.trunk = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            *(_ResidualBlock(channels) for _ in range(residual_blocks)),
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * height * width, action_count),
        )
        self.value_features = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(height * width, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 1),
            nn.Tanh(),
        )
        self.to(self.device)

    @classmethod
    def from_game_spec(
        cls,
        game_spec: DiscreteGameSpec,
        config: AlphaZeroConfig,
        *,
        device: str | torch.device = "cpu",
    ) -> PolicyValueNet:
        return cls(
            game_spec.observation_spec,
            len(game_spec.action_names),
            channels=config.channels,
            residual_blocks=config.residual_blocks,
            device=device,
        )

    def forward(
        self,
        planes: Tensor,
        scalars: Tensor | None = None,
        legal_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        planes = planes.to(device=self.device, dtype=torch.float32)
        if planes.ndim != 4:
            raise ValueError("planes must have shape [batch, channels, height, width]")
        scalar_count = len(self.observation_spec.scalar_names)
        if scalar_count:
            if scalars is None or scalars.shape != (planes.shape[0], scalar_count):
                raise ValueError("scalars must match the observation specification")
            height, width = self.observation_spec.board_shape
            scalar_planes = scalars.to(self.device, torch.float32)[:, :, None, None]
            planes = torch.cat(
                (planes, scalar_planes.expand(-1, -1, height, width)), dim=1
            )
        features = self.trunk(planes)
        logits = self.policy_head(features)
        if legal_mask is not None:
            legal_mask = legal_mask.to(device=self.device, dtype=torch.bool)
            if legal_mask.shape != logits.shape:
                raise ValueError("legal_mask must match policy logits")
            logits = logits.masked_fill(~legal_mask, torch.finfo(logits.dtype).min)
        value = self.value_head(self.value_features(features)).squeeze(-1)
        return logits, value

    def evaluate_batch(
        self,
        observations: Sequence[Observation],
        legal_masks: Sequence[NDArray[np.bool_]],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        if not observations or len(observations) != len(legal_masks):
            raise ValueError("observations and legal_masks must be non-empty equal batches")
        planes = torch.as_tensor(
            np.stack([observation.planes for observation in observations]),
            dtype=torch.float32,
            device=self.device,
        )
        scalars = torch.as_tensor(
            np.stack([observation.scalars for observation in observations]),
            dtype=torch.float32,
            device=self.device,
        )
        masks = torch.as_tensor(
            np.stack(legal_masks), dtype=torch.bool, device=self.device
        )
        was_training = self.training
        self.eval()
        try:
            with torch.inference_mode():
                logits, values = self(planes, scalars, masks)
        finally:
            self.train(was_training)
        return (
            logits.detach().cpu().numpy().astype(np.float32, copy=False),
            values.detach().cpu().numpy().astype(np.float32, copy=False),
        )
