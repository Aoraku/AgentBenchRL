"""Policy-change and state-occupancy statistics on explicitly aligned support."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


Distribution = Mapping[Hashable, float] | ArrayLike


@dataclass(frozen=True, slots=True)
class InformationGainSummary:
    """Auditable per-decision policy KL values and their two required rates."""

    local_kls: tuple[float, ...]
    nats_per_decision: float
    nats_per_episode: float


def summarize_information_gain(
    local_kls: Sequence[float], *, episodes: int
) -> InformationGainSummary:
    """Preserve local KL traces and summarize them by decision and episode."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    values = tuple(float(value) for value in local_kls)
    if not all(np.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("local KL values must be finite and non-negative")
    total = sum(values)
    return InformationGainSummary(
        local_kls=values,
        nats_per_decision=total / len(values) if values else 0.0,
        nats_per_episode=total / episodes,
    )


def policy_kl(
    current: Distribution,
    previous: Distribution,
    *,
    legal_support: Sequence[Hashable] | None = None,
    epsilon: float = 1e-12,
) -> float:
    """Return ``KL(current || previous)`` in nats on one legal action support.

    Mapping inputs make action-key alignment explicit.  Array inputs are only
    comparable when their shapes agree (and an optional supplied support has
    exactly that cardinality).  The distributions are not silently renormalized:
    their callers own the policy normalization contract.
    """
    if not np.isfinite(epsilon) or not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be finite and strictly between zero and one")
    current_values, previous_values = _aligned_values(
        current, previous, legal_support=legal_support
    )
    _validate_distribution(current_values, label="current policy")
    _validate_distribution(previous_values, label="previous policy")

    positive_current = current_values > 0.0
    return float(
        np.sum(
            current_values[positive_current]
            * np.log(
                current_values[positive_current]
                / np.maximum(previous_values[positive_current], epsilon)
            )
        )
    )


def occupancy_shift(
    current: Distribution,
    previous: Distribution,
    *,
    support: Sequence[Hashable] | None = None,
) -> float:
    """Return total-variation shift between two normalized occupancy histograms."""
    current_values, previous_values = _aligned_values(
        current, previous, legal_support=support
    )
    _validate_histogram(current_values, label="current occupancy")
    _validate_histogram(previous_values, label="previous occupancy")
    current_normalized = current_values / np.sum(current_values)
    previous_normalized = previous_values / np.sum(previous_values)
    return float(0.5 * np.sum(np.abs(current_normalized - previous_normalized)))


def _aligned_values(
    current: Distribution,
    previous: Distribution,
    *,
    legal_support: Sequence[Hashable] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    current_is_mapping = isinstance(current, Mapping)
    previous_is_mapping = isinstance(previous, Mapping)
    if current_is_mapping != previous_is_mapping:
        raise ValueError("both distributions must use the same legal support form")

    if current_is_mapping:
        current_mapping = current
        previous_mapping = previous
        current_keys = set(current_mapping)
        previous_keys = set(previous_mapping)
        if current_keys != previous_keys:
            raise ValueError("distributions must have identical legal support")
        keys = tuple(legal_support) if legal_support is not None else tuple(sorted(current_keys, key=repr))
        if len(keys) != len(current_keys) or set(keys) != current_keys:
            raise ValueError("legal support must completely and uniquely align distributions")
        return (
            np.asarray([current_mapping[key] for key in keys], dtype=np.float64),
            np.asarray([previous_mapping[key] for key in keys], dtype=np.float64),
        )

    current_values = np.asarray(current, dtype=np.float64)
    previous_values = np.asarray(previous, dtype=np.float64)
    if current_values.ndim != 1 or previous_values.ndim != 1:
        raise ValueError("array distributions must be one-dimensional")
    if current_values.shape != previous_values.shape:
        raise ValueError("array distributions must have identical legal support")
    if legal_support is not None:
        support = tuple(legal_support)
        if len(support) != current_values.size or len(set(support)) != len(support):
            raise ValueError(
                "legal support must completely and uniquely align array distributions"
            )
    return current_values, previous_values


def _validate_distribution(values: NDArray[np.float64], *, label: str) -> None:
    if values.size == 0:
        raise ValueError(f"{label} cannot be empty")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"{label} must contain finite non-negative probabilities")
    if not np.isclose(float(np.sum(values)), 1.0, rtol=0.0, atol=1e-9):
        raise ValueError(f"{label} must sum to one")


def _validate_histogram(values: NDArray[np.float64], *, label: str) -> None:
    if values.size == 0:
        raise ValueError(f"{label} cannot be empty")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"{label} must contain finite non-negative counts")
    if float(np.sum(values)) <= 0.0:
        raise ValueError(f"{label} must have positive total mass")
