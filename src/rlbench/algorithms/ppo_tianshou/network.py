"""Masked board/vector actor-critic networks compatible with Tianshou 2."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from tianshou.data import Batch
from tianshou.utils.net.common import AbstractDiscreteActor, ModuleWithVectorOutput
from torch import Tensor, nn

from rlbench.game import DiscreteGameSpec

from .config import PPOConfig


def _observation_fields(obs: Any) -> tuple[Any, Any | None]:
    if isinstance(obs, (dict, Batch)) and "obs" in obs:
        return obs["obs"], obs.get("mask")
    return obs, None


def _tensor(value: Any, device: torch.device, *, dtype: torch.dtype) -> Tensor:
    return torch.as_tensor(value, dtype=dtype, device=device)


class _VectorEncoder(ModuleWithVectorOutput):
    def __init__(self, observation_size: int, hidden_size: int) -> None:
        super().__init__(hidden_size)
        self.observation_size = observation_size
        self.model = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

    def forward(self, observation: Tensor) -> Tensor:
        if observation.shape[-1] != self.observation_size:
            raise ValueError("observation width does not match network input")
        return self.model(observation)


class _BoardEncoder(ModuleWithVectorOutput):
    def __init__(
        self,
        *,
        plane_shape: tuple[int, int, int],
        scalar_count: int,
        channels: int,
        hidden_size: int,
    ) -> None:
        super().__init__(hidden_size)
        self.plane_shape = plane_shape
        self.scalar_count = scalar_count
        self.observation_size = int(np.prod(plane_shape)) + scalar_count
        self.convolution = nn.Sequential(
            nn.Conv2d(
                plane_shape[0] + scalar_count,
                channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        height, width = plane_shape[1:]
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * height * width, hidden_size),
            nn.Tanh(),
        )

    def forward(self, observation: Tensor) -> Tensor:
        if observation.shape[-1] != self.observation_size:
            raise ValueError("observation width does not match board input")
        leading = observation.shape[:-1]
        flat = observation.reshape(-1, self.observation_size)
        plane_size = int(np.prod(self.plane_shape))
        planes = flat[:, :plane_size].reshape(-1, *self.plane_shape)
        if self.scalar_count:
            height, width = self.plane_shape[1:]
            scalars = flat[:, plane_size:, None, None].expand(-1, -1, height, width)
            planes = torch.cat((planes, scalars), dim=1)
        encoded = self.projection(self.convolution(planes))
        return encoded.reshape(*leading, self.output_dim)


class _MaskedActor(AbstractDiscreteActor):
    def __init__(
        self,
        encoder: ModuleWithVectorOutput,
        action_count: int,
        *,
        recurrent: bool,
        gru_hidden_size: int,
        device: torch.device,
    ) -> None:
        super().__init__(action_count)
        self.encoder = encoder
        self.device = device
        self.gru = (
            nn.GRU(encoder.get_output_dim(), gru_hidden_size, batch_first=True)
            if recurrent
            else None
        )
        self.head = nn.Linear(
            gru_hidden_size if recurrent else encoder.get_output_dim(), action_count
        )

    def get_preprocess_net(self) -> ModuleWithVectorOutput:
        return self.encoder

    def forward(
        self,
        obs: Any,
        state: Batch | dict[str, Any] | None = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[Tensor, Batch | None]:
        del info
        raw_observation, raw_mask = _observation_fields(obs)
        observation = _tensor(raw_observation, self.device, dtype=torch.float32)
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        features = self.encoder(observation)
        next_state: Batch | None = None
        if self.gru is not None:
            sequence = features if features.ndim == 3 else features.unsqueeze(1)
            hidden = torch.zeros(
                self.gru.num_layers,
                sequence.shape[0],
                self.gru.hidden_size,
                dtype=sequence.dtype,
                device=sequence.device,
            )
            if state is not None:
                hidden = _tensor(
                    state["hidden"], self.device, dtype=torch.float32
                ).transpose(0, 1).contiguous()
            pre_hidden = hidden.transpose(0, 1).detach()
            output, hidden = self.gru(sequence, hidden)
            features = output[:, -1]
            next_state = Batch(
                hidden=hidden.transpose(0, 1).detach(),
                pre_hidden=pre_hidden,
            )
        logits = self.head(features)
        if raw_mask is not None:
            mask = _tensor(raw_mask, self.device, dtype=torch.bool)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            if mask.shape != logits.shape:
                raise ValueError("legal mask must match actor logits")
            if torch.any(~mask.any(dim=-1)):
                raise ValueError("every actor observation needs a legal action")
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return logits, next_state


class _ValueCritic(nn.Module):
    def __init__(
        self,
        encoder: ModuleWithVectorOutput,
        *,
        recurrent: bool,
        gru_hidden_size: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.device = device
        self.gru = (
            nn.GRU(encoder.get_output_dim(), gru_hidden_size, batch_first=True)
            if recurrent
            else None
        )
        self.head = nn.Linear(
            gru_hidden_size if recurrent else encoder.get_output_dim(), 1
        )

    def forward(
        self,
        obs: Any,
        state: Batch | dict[str, Any] | None = None,
        info: dict[str, Any] | None = None,
    ) -> Tensor:
        del info
        raw_observation, _ = _observation_fields(obs)
        observation = _tensor(raw_observation, self.device, dtype=torch.float32)
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        features = self.encoder(observation)
        if self.gru is not None:
            sequence = features if features.ndim == 3 else features.unsqueeze(1)
            hidden = None
            if state is not None:
                hidden = _tensor(
                    state["hidden"], self.device, dtype=torch.float32
                ).transpose(0, 1).contiguous()
            features, _ = self.gru(sequence, hidden)
            features = features[:, -1]
        return self.head(features)


class MaskedActorCritic(nn.Module):
    """Own a recurrent-capable actor and a deliberately feed-forward critic."""

    def __init__(
        self,
        observation_size: int,
        action_count: int,
        config: PPOConfig | None = None,
        *,
        board_shape: tuple[int, int, int] | None = None,
        scalar_count: int = 0,
    ) -> None:
        super().__init__()
        config = config or PPOConfig()
        if observation_size <= 0 or action_count <= 0:
            raise ValueError("observation_size and action_count must be positive")
        self.observation_size = observation_size
        self.action_count = action_count
        self.board_shape = board_shape
        self.scalar_count = scalar_count
        self.device = torch.device(config.device)

        def encoder() -> ModuleWithVectorOutput:
            if board_shape is None:
                return _VectorEncoder(observation_size, config.hidden_size)
            expected = int(np.prod(board_shape)) + scalar_count
            if expected != observation_size:
                raise ValueError("board and scalar dimensions do not match observation_size")
            return _BoardEncoder(
                plane_shape=board_shape,
                scalar_count=scalar_count,
                channels=config.conv_channels,
                hidden_size=config.hidden_size,
            )

        self.actor = _MaskedActor(
            encoder(),
            action_count,
            recurrent=config.recurrent,
            gru_hidden_size=config.gru_hidden_size,
            device=self.device,
        )
        self.critic = _ValueCritic(
            encoder(),
            recurrent=False,
            gru_hidden_size=config.gru_hidden_size,
            device=self.device,
        )
        self.to(self.device)

    @classmethod
    def from_game_spec(
        cls, game_spec: DiscreteGameSpec, config: PPOConfig | None = None
    ) -> MaskedActorCritic:
        observation_spec = game_spec.observation_spec
        channels = len(observation_spec.plane_names)
        height, width = observation_spec.board_shape
        scalar_count = len(observation_spec.scalar_names)
        return cls(
            channels * height * width + scalar_count,
            len(game_spec.action_names),
            config,
            board_shape=(channels, height, width),
            scalar_count=scalar_count,
        )

    def forward(
        self, obs: Any, state: Batch | dict[str, Any] | None = None
    ) -> tuple[Tensor, Tensor, Batch | None]:
        logits, next_state = self.actor(obs, state=state)
        return logits, self.critic(obs), next_state
