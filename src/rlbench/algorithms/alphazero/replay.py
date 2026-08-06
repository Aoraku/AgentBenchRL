"""Sliding replay storage for AlphaZero policy-value targets."""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from typing import Any, Literal, Mapping

import numpy as np
from numpy.typing import NDArray

from rlbench.game import Observation


@dataclass(frozen=True, slots=True)
class ReplaySample:
    """One immutable search target from a current-player observation."""

    observation: Observation
    legal_mask: NDArray[np.bool_]
    visit_policy: NDArray[np.float32]
    outcome: float
    player: int = 0
    source: Literal["selfplay", "learner", "expert_demo"] = "selfplay"
    sample_weight: float = 1.0
    decision_index: int = -1

    def __post_init__(self) -> None:
        planes = np.array(self.observation.planes, dtype=np.float32, copy=True)
        scalars = np.array(self.observation.scalars, dtype=np.float32, copy=True)
        legal_mask = np.array(self.legal_mask, dtype=np.bool_, copy=True)
        visit_policy = np.array(self.visit_policy, dtype=np.float32, copy=True)
        if legal_mask.ndim != 1 or visit_policy.shape != legal_mask.shape:
            raise ValueError("legal mask and visit policy must be equal action vectors")
        if not legal_mask.any():
            raise ValueError("a replay sample must contain a legal action")
        if not np.all(np.isfinite(visit_policy)) or np.any(visit_policy < 0.0):
            raise ValueError("visit policy must be finite and non-negative")
        if np.any(visit_policy[~legal_mask] != 0.0):
            raise ValueError("visit policy cannot assign mass to illegal actions")
        if not np.isclose(float(visit_policy.sum()), 1.0, atol=1e-6):
            raise ValueError("visit policy must sum to one")
        if self.outcome not in (-1.0, 0.0, 1.0):
            raise ValueError("outcome must be -1, 0, or 1")
        if self.player not in (0, 1):
            raise ValueError("player must be 0 or 1")
        if self.source not in ("selfplay", "learner", "expert_demo"):
            raise ValueError("invalid replay sample source")
        if not np.isfinite(self.sample_weight) or self.sample_weight <= 0.0:
            raise ValueError("sample weight must be finite and positive")
        if isinstance(self.decision_index, bool) or self.decision_index < -1:
            raise ValueError("decision index must be -1 or a non-negative integer")
        for array in (planes, scalars, legal_mask, visit_policy):
            array.setflags(write=False)
        object.__setattr__(self, "observation", Observation(planes, scalars))
        object.__setattr__(self, "legal_mask", legal_mask)
        object.__setattr__(self, "visit_policy", visit_policy)
        object.__setattr__(self, "outcome", float(self.outcome))


@dataclass(frozen=True, slots=True)
class ReplayBatch:
    planes: NDArray[np.float32]
    scalars: NDArray[np.float32]
    legal_masks: NDArray[np.bool_]
    visit_policies: NDArray[np.float32]
    outcomes: NDArray[np.float32]
    players: NDArray[np.int64]
    sample_weights: NDArray[np.float32]


class ReplayBuffer:
    """A deterministic bounded replay window with resumable sampling state."""

    def __init__(self, capacity: int, *, seed: int = 0) -> None:
        if capacity < 1:
            raise ValueError("replay capacity must be positive")
        self.capacity = capacity
        self._samples: deque[ReplaySample] = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._samples)

    def add(self, sample: ReplaySample) -> None:
        if not isinstance(sample, ReplaySample):
            raise TypeError("replay entries must be ReplaySample values")
        self._samples.append(sample)

    def extend(self, samples: list[ReplaySample]) -> None:
        for sample in samples:
            self.add(sample)

    def sample(self, batch_size: int) -> ReplayBatch:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self._samples):
            raise ValueError("not enough replay samples for the requested batch")
        indices = self.rng.choice(len(self._samples), size=batch_size, replace=False)
        selected = [self._samples[int(index)] for index in indices]
        return ReplayBatch(
            planes=np.stack([sample.observation.planes for sample in selected]),
            scalars=np.stack([sample.observation.scalars for sample in selected]),
            legal_masks=np.stack([sample.legal_mask for sample in selected]),
            visit_policies=np.stack([sample.visit_policy for sample in selected]),
            outcomes=np.asarray([sample.outcome for sample in selected], dtype=np.float32),
            players=np.asarray([sample.player for sample in selected], dtype=np.int64),
            sample_weights=np.asarray(
                [sample.sample_weight for sample in selected], dtype=np.float32
            ),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
            "samples": [
                {
                    "planes": np.array(sample.observation.planes, copy=True),
                    "scalars": np.array(sample.observation.scalars, copy=True),
                    "legal_mask": np.array(sample.legal_mask, copy=True),
                    "visit_policy": np.array(sample.visit_policy, copy=True),
                    "outcome": sample.outcome,
                    "player": sample.player,
                    "source": sample.source,
                    "sample_weight": sample.sample_weight,
                    "decision_index": sample.decision_index,
                }
                for sample in self._samples
            ],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("capacity", -1)) != self.capacity:
            raise ValueError("checkpoint replay capacity does not match configuration")
        raw_samples = state.get("samples")
        rng_state = state.get("rng_state")
        if not isinstance(raw_samples, list) or not isinstance(rng_state, Mapping):
            raise ValueError("invalid replay checkpoint state")
        if len(raw_samples) > self.capacity:
            raise ValueError("checkpoint replay exceeds configured capacity")
        restored = deque(maxlen=self.capacity)
        for raw in raw_samples:
            if not isinstance(raw, Mapping):
                raise ValueError("invalid replay sample state")
            restored.append(
                ReplaySample(
                    observation=Observation(
                        planes=np.asarray(raw["planes"], dtype=np.float32),
                        scalars=np.asarray(raw["scalars"], dtype=np.float32),
                    ),
                    legal_mask=np.asarray(raw["legal_mask"], dtype=np.bool_),
                    visit_policy=np.asarray(raw["visit_policy"], dtype=np.float32),
                    outcome=float(raw["outcome"]),
                    player=int(raw.get("player", 0)),
                    source=str(raw.get("source", "selfplay")),
                    sample_weight=float(raw.get("sample_weight", 1.0)),
                    decision_index=int(raw.get("decision_index", -1)),
                )
            )
        restored_rng = np.random.default_rng()
        restored_rng.bit_generator.state = copy.deepcopy(dict(rng_state))
        self._samples = restored
        self.rng = restored_rng
