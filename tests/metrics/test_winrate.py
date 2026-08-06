from __future__ import annotations

import pytest

from rlbench.metrics import MatchOutcome, win_rate_summary


def test_win_rate_scores_draws_as_half_and_excludes_invalid_games() -> None:
    """Treating draws as wins or counting outages changes the reported score."""
    outcomes = [
        MatchOutcome("candidate", "baseline", 1.0),
        MatchOutcome("candidate", "baseline", 0.5),
        MatchOutcome("candidate", "baseline", 0.0),
        MatchOutcome("candidate", "baseline", 1.0, valid=False),
    ]

    summary = win_rate_summary(outcomes, "candidate")

    assert summary.wins == 1
    assert summary.draws == 1
    assert summary.losses == 1
    assert summary.valid_games == 3
    assert summary.score == pytest.approx(0.5)
    assert summary.wilson_lower == pytest.approx(0.1253, abs=0.0002)
    assert summary.wilson_upper == pytest.approx(0.8747, abs=0.0002)


def test_win_rate_uses_the_opponent_perspective_for_each_result() -> None:
    """Forgetting to invert score_a reports the opponent's record incorrectly."""
    outcomes = [
        MatchOutcome("candidate", "baseline", 1.0),
        MatchOutcome("candidate", "baseline", 0.0),
        MatchOutcome("candidate", "baseline", 0.5),
    ]

    summary = win_rate_summary(outcomes, "baseline")

    assert (summary.wins, summary.draws, summary.losses) == (1, 1, 1)
    assert summary.score == pytest.approx(0.5)


@pytest.mark.parametrize("z_value", [float("nan"), float("inf"), -1.0, 0.0])
def test_win_rate_rejects_non_finite_or_non_positive_wilson_z_value(
    z_value: float,
) -> None:
    """A malformed critical value must not yield a misleading confidence interval."""
    with pytest.raises(ValueError, match="z_value"):
        win_rate_summary([], "candidate", z_value=z_value)
