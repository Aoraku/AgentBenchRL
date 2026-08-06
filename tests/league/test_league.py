from __future__ import annotations

from collections import Counter

import pytest

from rlbench.league import (
    LeagueMember,
    LeagueState,
    OpponentSampler,
    PromotionConfig,
    evaluate_promotion,
)
from rlbench.metrics import MatchOutcome


def _member(agent_id: str, rating: float, *, kind: str = "checkpoint", generation: int = 0):
    return LeagueMember(
        agent_id=agent_id,
        content_hash="sha256:" + agent_id.encode().hex().ljust(64, "0")[:64],
        kind=kind,
        rating=rating,
        generation=generation,
    )


def test_league_refreshes_ratings_and_retains_recent_plus_historical_checkpoints() -> None:
    """Stale ratings or all-recency retention erase strength and forgetting signals."""
    state = LeagueState(
        anchor_id="baseline",
        champion_id="champion",
        members=(
            _member("baseline", 1000.0, kind="baseline"),
            _member("champion", 1000.0, kind="champion"),
            *tuple(_member(f"cp-{i}", 1000.0, generation=i) for i in range(1, 6)),
        ),
    )
    outcomes = [
        MatchOutcome("champion", "baseline", 1.0),
        MatchOutcome("cp-5", "baseline", 0.0),
    ]

    refreshed = state.refresh_ratings(outcomes)
    retained = refreshed.retain_checkpoints(recent=2, historical=1)

    assert refreshed.member("champion").rating > refreshed.member("cp-5").rating
    assert [member.agent_id for member in retained.members] == [
        "baseline",
        "champion",
        "cp-1",
        "cp-4",
        "cp-5",
    ]


def test_opponent_sampling_is_seeded_strength_weighted_and_training_safe() -> None:
    """Uniform, nondeterministic, or held-out sampling changes the training objective."""
    members = (
        _member("weak", 700.0),
        _member("strong", 1300.0),
        _member("held-out", 5000.0, kind="test_human"),
    )
    first = OpponentSampler(members, seed=7)
    second = OpponentSampler(members, seed=7)

    assert first.sample(learner_rating=1000.0, count=20) == second.sample(
        learner_rating=1000.0, count=20
    )
    counts = Counter(
        OpponentSampler(members, seed=17).sample(learner_rating=1000.0, count=1000)
    )
    assert counts["strong"] > counts["weak"]
    assert counts["held-out"] == 0


def _promotion_outcomes(*, promotion_wins: int = 20, promotion_losses: int = 0, protected_wins: int = 9):
    return [
        *[
            MatchOutcome("candidate", "promotion", 1.0)
            for _ in range(promotion_wins)
        ],
        *[
            MatchOutcome("candidate", "promotion", 0.0)
            for _ in range(promotion_losses)
        ],
        *[MatchOutcome("candidate", "guard", 1.0) for _ in range(protected_wins)],
        *[MatchOutcome("candidate", "guard", 0.0) for _ in range(10 - protected_wins)],
    ]


def _decision(**overrides):
    arguments = {
        "candidate_id": "candidate",
        "champion_id": "champion",
        "ratings": {"candidate": 1125.0, "champion": 1000.0},
        "outcomes": _promotion_outcomes(),
        "promotion_opponents": {"promotion"},
        "protected_reference_scores": {"guard": 0.85},
        "evaluation_complete": True,
        "config": PromotionConfig(
            minimum_elo_delta=100.0,
            minimum_win_rate_lower_bound=0.80,
            maximum_protected_regression=0.10,
        ),
    }
    arguments.update(overrides)
    return evaluate_promotion(**arguments)


def test_promotion_requires_all_joint_gates() -> None:
    """Replacing joint gates with a scalar lets one metric hide a regression."""
    decision = _decision()

    assert decision.promoted is True
    assert decision.elo_delta == pytest.approx(125.0)
    assert decision.win_rate_lower_bound > 0.80
    assert decision.reasons == ()


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"ratings": {"candidate": 1099.0, "champion": 1000.0}}, "elo_delta"),
        (
            {"outcomes": _promotion_outcomes(promotion_wins=10, promotion_losses=10)},
            "win_rate_lower_bound",
        ),
        (
            {"outcomes": _promotion_outcomes(protected_wins=7)},
            "protected_regression:guard",
        ),
        ({"evaluation_complete": False}, "evaluation_incomplete"),
    ],
)
def test_each_failed_gate_blocks_promotion(override: dict, reason: str) -> None:
    """Skipping any individual gate can promote an unqualified checkpoint."""
    decision = _decision(**override)

    assert decision.promoted is False
    assert reason in decision.reasons


def test_protected_opponent_requires_explicit_valid_match_coverage() -> None:
    """A permissive regression threshold must not disguise missing protected games."""
    promotion_only = [
        MatchOutcome("candidate", "promotion", 1.0) for _ in range(20)
    ]

    decision = _decision(
        outcomes=promotion_only,
        protected_reference_scores={"guard": 0.0},
        config=PromotionConfig(
            minimum_elo_delta=100.0,
            minimum_win_rate_lower_bound=0.80,
            maximum_protected_regression=1.0,
        ),
    )

    assert decision.promoted is False
    assert "evaluation_incomplete" in decision.reasons
    assert "protected_coverage:guard" in decision.reasons
