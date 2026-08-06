from __future__ import annotations

from math import log

import pytest

from rlbench.metrics import occupancy_shift, policy_kl, summarize_information_gain


def test_policy_kl_is_zero_for_equal_policies_on_the_same_legal_support() -> None:
    """Changing the KL direction or normalization makes identical policies nonzero."""
    policy = {"left": 0.25, "right": 0.75}

    assert policy_kl(policy, policy) == pytest.approx(0.0)


def test_policy_kl_rejects_misaligned_legal_support() -> None:
    """Comparing different action domains mistakes unavailable actions for information."""
    with pytest.raises(ValueError, match="legal support"):
        policy_kl({"left": 0.5, "right": 0.5}, {"left": 1.0})


def test_policy_kl_rejects_duplicate_array_legal_support_entries() -> None:
    """Repeated action labels falsely claim a complete aligned legal support."""
    with pytest.raises(ValueError, match="uniquely"):
        policy_kl(
            [0.5, 0.5],
            [0.5, 0.5],
            legal_support=("left", "left"),
        )


def test_policy_kl_uses_epsilon_to_keep_zero_previous_mass_finite() -> None:
    """An earlier zero-probability legal action must not turn a trace into infinity."""
    value = policy_kl(
        {"left": 0.5, "right": 0.5},
        {"left": 0.0, "right": 1.0},
        epsilon=1e-6,
    )

    assert value == pytest.approx(0.5 * log(250_000.0))


def test_occupancy_shift_normalizes_histograms_without_reusing_policy_kl() -> None:
    """Occupancy movement is a separate trajectory statistic from local policy KL."""
    shift = occupancy_shift(
        {"opening": 3, "endgame": 1},
        {"opening": 1, "endgame": 3},
    )

    assert shift == pytest.approx(0.5)


def test_information_gain_summary_retains_local_values_and_episode_aggregates() -> None:
    """Dropping decision values prevents an aggregate from being audited or regrouped."""
    summary = summarize_information_gain([0.2, 0.4, 0.6], episodes=2)

    assert summary.local_kls == pytest.approx((0.2, 0.4, 0.6))
    assert summary.nats_per_decision == pytest.approx(0.4)
    assert summary.nats_per_episode == pytest.approx(0.6)
