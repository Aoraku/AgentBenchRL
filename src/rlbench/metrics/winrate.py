"""Win-rate summaries derived from individual match facts."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Iterable


@dataclass(frozen=True, slots=True)
class MatchOutcome:
    """A two-player outcome, recorded from ``player_a``'s perspective."""

    player_a: str
    player_b: str
    score_a: float | None
    valid: bool = True

    def __post_init__(self) -> None:
        if not self.player_a or not self.player_b:
            raise ValueError("match players must be non-empty")
        if self.player_a == self.player_b:
            raise ValueError("a match requires two distinct players")
        if self.valid and self.score_a not in (0.0, 0.5, 1.0):
            raise ValueError("valid match score_a must be 0.0, 0.5, or 1.0")


@dataclass(frozen=True, slots=True)
class WinRateSummary:
    """Score and Wilson interval for one player across valid matches."""

    player: str
    wins: int
    draws: int
    losses: int
    valid_games: int
    score: float
    wilson_lower: float
    wilson_upper: float

    @property
    def wilson_interval(self) -> tuple[float, float]:
        """Return the two-sided 95% Wilson score interval."""
        return (self.wilson_lower, self.wilson_upper)


def win_rate_summary(
    outcomes: Iterable[MatchOutcome], player: str, *, z_value: float = 1.96
) -> WinRateSummary:
    """Summarize a player's valid outcomes with draws worth half a win."""
    if not player:
        raise ValueError("player must be non-empty")
    if not isfinite(z_value) or z_value <= 0.0:
        raise ValueError("z_value must be finite and positive")

    wins = draws = losses = 0
    for outcome in outcomes:
        if not outcome.valid:
            continue
        score = _score_for(outcome, player)
        if score is None:
            continue
        if score == 1.0:
            wins += 1
        elif score == 0.5:
            draws += 1
        else:
            losses += 1

    valid_games = wins + draws + losses
    if valid_games == 0:
        return WinRateSummary(
            player=player,
            wins=0,
            draws=0,
            losses=0,
            valid_games=0,
            score=0.0,
            wilson_lower=0.0,
            wilson_upper=1.0,
        )

    score = (wins + 0.5 * draws) / valid_games
    lower, upper = _wilson_interval(score, valid_games, z_value)
    return WinRateSummary(
        player=player,
        wins=wins,
        draws=draws,
        losses=losses,
        valid_games=valid_games,
        score=score,
        wilson_lower=lower,
        wilson_upper=upper,
    )


def _score_for(outcome: MatchOutcome, player: str) -> float | None:
    if player == outcome.player_a:
        return outcome.score_a
    if player == outcome.player_b:
        if outcome.score_a is None:
            return None
        return 1.0 - outcome.score_a
    return None


def _wilson_interval(score: float, count: int, z_value: float) -> tuple[float, float]:
    z_squared = z_value * z_value
    denominator = 1.0 + z_squared / count
    centre = (score + z_squared / (2.0 * count)) / denominator
    half_width = (
        z_value
        * sqrt(score * (1.0 - score) / count + z_squared / (4.0 * count * count))
        / denominator
    )
    return (max(0.0, centre - half_width), min(1.0, centre + half_width))
