"""Deterministic strength-weighted training-opponent selection."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable

from .state import LeagueMember


class OpponentSampler:
    def __init__(self, members: Iterable[LeagueMember], *, seed: int) -> None:
        self._members = tuple(
            member for member in members if member.kind != "test_human"
        )
        if not self._members:
            raise ValueError("training sampler requires at least one eligible opponent")
        self._random = random.Random(seed)

    def sample(self, *, learner_rating: float, count: int = 1) -> tuple[str, ...]:
        if count < 0:
            raise ValueError("count must be non-negative")
        weights = [
            math.exp(max(-8.0, min(8.0, (member.rating - learner_rating) / 400.0)))
            for member in self._members
        ]
        return tuple(
            member.agent_id
            for member in self._random.choices(self._members, weights=weights, k=count)
        )
