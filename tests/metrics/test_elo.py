from __future__ import annotations

import pytest

from rlbench.metrics import MatchOutcome, fit_anchored_elo


def test_batch_elo_keeps_the_named_anchor_at_1000() -> None:
    """Allowing the anchor to drift makes independently fitted graphs incomparable."""
    outcomes = [
        MatchOutcome("strong", "baseline", 1.0),
        MatchOutcome("strong", "baseline", 1.0),
        MatchOutcome("weak", "baseline", 0.0),
        MatchOutcome("weak", "baseline", 0.0),
    ]

    ratings = fit_anchored_elo(outcomes, anchor="baseline")

    assert ratings.ratings["baseline"] == pytest.approx(1000.0)
    assert ratings.ratings["strong"] > 1000.0
    assert ratings.ratings["weak"] < 1000.0
    assert ratings.uncertainties["strong"] > 0.0


def test_batch_elo_rejects_an_anchor_rating_other_than_1000() -> None:
    """A configurable anchor would make independently fitted rating graphs incomparable."""
    with pytest.raises(ValueError, match="1000"):
        fit_anchored_elo(
            [MatchOutcome("candidate", "baseline", 1.0)],
            anchor="baseline",
            anchor_rating=999.0,
        )


@pytest.mark.parametrize("l2", [float("nan"), float("inf"), -0.1, 0.0])
def test_batch_elo_rejects_non_finite_or_non_positive_l2(l2: float) -> None:
    """A malformed penalty makes the optimizer's objective and covariance invalid."""
    with pytest.raises(ValueError, match="l2"):
        fit_anchored_elo([], anchor="baseline", l2=l2)


@pytest.mark.parametrize("tolerance", [float("nan"), float("inf"), -1.0, 0.0])
def test_batch_elo_rejects_non_finite_or_non_positive_tolerance(
    tolerance: float,
) -> None:
    """A malformed convergence threshold must not silently alter optimizer behavior."""
    with pytest.raises(ValueError, match="tolerance"):
        fit_anchored_elo([], anchor="baseline", tolerance=tolerance)


def test_single_decisive_match_has_a_fixed_regularized_rating_and_uncertainty() -> None:
    """A wrong likelihood, Elo scale, or inverse-Hessian conversion changes these literals.

    For one win with l2=0.01, independently solving
    sigmoid(s) - 1 + 0.01*s = 0 gives s about 3.359; multiplying by
    400 / ln(10) gives the stated rating near 1583.5.
    """
    ratings = fit_anchored_elo(
        [MatchOutcome("candidate", "baseline", 1.0)], anchor="baseline"
    )

    assert ratings.ratings["candidate"] == pytest.approx(1583.5, abs=1.0)
    assert ratings.uncertainties["candidate"] == pytest.approx(841.5, abs=3.0)
    assert ratings.uncertainties["candidate"] > 0.0
    assert ratings.uncertainties["candidate"] < 1_000.0


def test_batch_elo_is_order_independent_and_ignores_invalid_games() -> None:
    """Sequential updates or infrastructure failures would make the batch fit unstable."""
    outcomes = [
        MatchOutcome("strong", "baseline", 1.0),
        MatchOutcome("strong", "baseline", 1.0),
        MatchOutcome("strong", "baseline", 1.0),
        MatchOutcome("weak", "baseline", 0.0),
        MatchOutcome("weak", "baseline", 0.0),
        MatchOutcome("weak", "baseline", 0.0),
        MatchOutcome("strong", "baseline", 0.0, valid=False),
    ]

    forward = fit_anchored_elo(outcomes, anchor="baseline")
    backward = fit_anchored_elo(list(reversed(outcomes)), anchor="baseline")

    assert forward.ratings == pytest.approx(backward.ratings, abs=1e-10)
    assert forward.ratings["strong"] > forward.ratings["weak"]
