"""Promotion decisions requiring every strength and safety gate."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Iterable, Mapping

from rlbench.metrics import MatchOutcome, win_rate_summary


@dataclass(frozen=True, slots=True)
class PromotionConfig:
    minimum_elo_delta: float
    minimum_win_rate_lower_bound: float
    maximum_protected_regression: float

    def __post_init__(self) -> None:
        values = (
            self.minimum_elo_delta,
            self.minimum_win_rate_lower_bound,
            self.maximum_protected_regression,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("promotion thresholds must be finite")
        if not 0.0 <= self.minimum_win_rate_lower_bound <= 1.0:
            raise ValueError("win-rate lower bound must be between zero and one")
        if not 0.0 <= self.maximum_protected_regression <= 1.0:
            raise ValueError("protected regression must be between zero and one")


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    promoted: bool
    reasons: tuple[str, ...]
    elo_delta: float
    win_rate_lower_bound: float
    protected_scores: Mapping[str, float]


def evaluate_promotion(
    *,
    candidate_id: str,
    champion_id: str,
    ratings: Mapping[str, float],
    outcomes: Iterable[MatchOutcome],
    promotion_opponents: set[str] | frozenset[str],
    protected_reference_scores: Mapping[str, float],
    evaluation_complete: bool,
    config: PromotionConfig,
) -> PromotionDecision:
    """Promote only when completeness, Elo, confidence, and safety all pass."""
    facts = tuple(outcomes)
    elo_delta = ratings[candidate_id] - ratings[champion_id]
    promotion_facts = tuple(
        outcome
        for outcome in facts
        if _opponent_of(outcome, candidate_id) in promotion_opponents
    )
    promotion_summary = win_rate_summary(promotion_facts, candidate_id)
    protected_summaries = {
        opponent: win_rate_summary(
            (
                outcome
                for outcome in facts
                if _opponent_of(outcome, candidate_id) == opponent
            ),
            candidate_id,
        )
        for opponent in sorted(protected_reference_scores)
    }
    protected_scores = {
        opponent: summary.score
        for opponent, summary in protected_summaries.items()
    }
    missing_protected = tuple(
        opponent
        for opponent, summary in protected_summaries.items()
        if summary.valid_games == 0
    )

    reasons: list[str] = []
    if (
        not evaluation_complete
        or any(not outcome.valid for outcome in facts)
        or missing_protected
    ):
        reasons.append("evaluation_incomplete")
    if elo_delta < config.minimum_elo_delta:
        reasons.append("elo_delta")
    if (
        promotion_summary.wilson_lower
        < config.minimum_win_rate_lower_bound
    ):
        reasons.append("win_rate_lower_bound")
    for opponent, reference in sorted(protected_reference_scores.items()):
        if opponent in missing_protected:
            reasons.append(f"protected_coverage:{opponent}")
            continue
        if protected_scores[opponent] < (
            reference - config.maximum_protected_regression
        ):
            reasons.append(f"protected_regression:{opponent}")
    return PromotionDecision(
        promoted=not reasons,
        reasons=tuple(reasons),
        elo_delta=elo_delta,
        win_rate_lower_bound=promotion_summary.wilson_lower,
        protected_scores=MappingProxyType(protected_scores),
    )


def _opponent_of(outcome: MatchOutcome, player: str) -> str | None:
    if outcome.player_a == player:
        return outcome.player_b
    if outcome.player_b == player:
        return outcome.player_a
    return None
