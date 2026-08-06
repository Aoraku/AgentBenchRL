"""Immutable league membership and batch Elo refresh."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Literal

from rlbench.metrics import MatchOutcome, fit_anchored_elo


MemberKind = Literal[
    "train_human", "test_human", "baseline", "checkpoint", "champion"
]


@dataclass(frozen=True, slots=True)
class LeagueMember:
    agent_id: str
    content_hash: str
    kind: MemberKind
    rating: float = 1000.0
    uncertainty: float = 0.0
    generation: int = 0


@dataclass(frozen=True, slots=True)
class LeagueState:
    anchor_id: str
    champion_id: str
    members: tuple[LeagueMember, ...]

    def __post_init__(self) -> None:
        ids = [member.agent_id for member in self.members]
        if len(ids) != len(set(ids)):
            raise ValueError("league member IDs must be unique")
        if self.anchor_id not in ids:
            raise ValueError("league anchor must be a member")
        if self.champion_id not in ids:
            raise ValueError("league champion must be a member")

    def member(self, agent_id: str) -> LeagueMember:
        for member in self.members:
            if member.agent_id == agent_id:
                return member
        raise KeyError(agent_id)

    def refresh_ratings(self, outcomes: Iterable[MatchOutcome]) -> LeagueState:
        ratings = fit_anchored_elo(outcomes, anchor=self.anchor_id)
        return replace(
            self,
            members=tuple(
                replace(
                    member,
                    rating=ratings.ratings.get(member.agent_id, member.rating),
                    uncertainty=ratings.uncertainties.get(
                        member.agent_id, member.uncertainty
                    ),
                )
                for member in self.members
            ),
        )

    def retain_checkpoints(self, *, recent: int, historical: int) -> LeagueState:
        if recent < 0 or historical < 0:
            raise ValueError("retention counts must be non-negative")
        checkpoints = sorted(
            (member for member in self.members if member.kind == "checkpoint"),
            key=lambda member: (member.generation, member.agent_id),
        )
        recent_members = checkpoints[-recent:] if recent else []
        older = checkpoints[: max(0, len(checkpoints) - len(recent_members))]
        historical_members = older[:historical]
        selected = {
            member.agent_id for member in (*historical_members, *recent_members)
        }
        return replace(
            self,
            members=tuple(
                member
                for member in self.members
                if member.kind != "checkpoint" or member.agent_id in selected
            ),
        )
