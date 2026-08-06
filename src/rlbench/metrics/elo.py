"""Order-independent, anchored Bradley--Terry ratings."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from math import log
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from .winrate import MatchOutcome


_ELO_SCALE = 400.0 / log(10.0)


@dataclass(frozen=True, slots=True)
class EloRatings:
    """Anchored ratings plus covariance-derived one-standard-deviation errors."""

    ratings: dict[str, float]
    uncertainties: dict[str, float]
    anchor: str
    anchor_rating: float
    valid_games: int


def fit_anchored_elo(
    outcomes: Iterable[MatchOutcome],
    *,
    anchor: str,
    anchor_rating: float = 1000.0,
    l2: float = 0.01,
    max_iterations: int = 10_000,
    tolerance: float = 1e-10,
) -> EloRatings:
    """Fit a batch Bradley--Terry model with a fixed, named rating anchor.

    A full-batch deterministic gradient descent minimizes the draw-half
    cross-entropy objective.  L2 regularization stabilizes sparse graphs; the
    inverse observed Hessian supplies rating uncertainty for non-anchor nodes.
    """
    if not anchor:
        raise ValueError("anchor must be non-empty")
    if anchor_rating != 1000.0:
        raise ValueError("anchor_rating is fixed at 1000")
    if not np.isfinite(l2) or l2 <= 0.0:
        raise ValueError("l2 must be finite and positive")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    valid = sorted(
        (
            outcome
            for outcome in outcomes
            if outcome.valid and outcome.score_a is not None
        ),
        key=lambda item: (item.player_a, item.player_b, float(item.score_a)),
    )
    players = {anchor}
    for outcome in valid:
        players.add(outcome.player_a)
        players.add(outcome.player_b)
    _require_anchor_connectivity(players, valid, anchor)

    unknown_players = tuple(sorted(players - {anchor}))
    if not unknown_players:
        return EloRatings(
            ratings={anchor: float(anchor_rating)},
            uncertainties={anchor: 0.0},
            anchor=anchor,
            anchor_rating=float(anchor_rating),
            valid_games=0,
        )

    index = {player: position for position, player in enumerate(unknown_players)}
    design, targets = _design_matrix(valid, index)
    strengths = np.zeros(len(unknown_players), dtype=np.float64)

    for _ in range(max_iterations):
        logits = design @ strengths
        probabilities = _sigmoid(logits)
        gradient = design.T @ (probabilities - targets) + l2 * strengths
        if float(np.max(np.abs(gradient))) <= tolerance:
            break
        objective = _objective(logits, targets, strengths, l2)
        direction = gradient
        step = 1.0
        while step > 1e-14:
            candidate = strengths - step * direction
            candidate_objective = _objective(
                design @ candidate, targets, candidate, l2
            )
            if candidate_objective <= objective - 1e-4 * step * float(direction @ direction):
                strengths = candidate
                break
            step *= 0.5
        else:
            break

    probabilities = _sigmoid(design @ strengths)
    hessian = (design.T * (probabilities * (1.0 - probabilities))) @ design
    hessian += l2 * np.eye(len(unknown_players))
    covariance = np.linalg.inv(hessian)

    ratings = {anchor: float(anchor_rating)}
    uncertainties = {anchor: 0.0}
    for player, position in index.items():
        ratings[player] = float(anchor_rating + _ELO_SCALE * strengths[position])
        uncertainties[player] = float(
            _ELO_SCALE * np.sqrt(max(0.0, covariance[position, position]))
        )
    return EloRatings(
        ratings=ratings,
        uncertainties=uncertainties,
        anchor=anchor,
        anchor_rating=float(anchor_rating),
        valid_games=len(valid),
    )


def _design_matrix(
    outcomes: list[MatchOutcome], index: dict[str, int]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    design = np.zeros((len(outcomes), len(index)), dtype=np.float64)
    targets = np.zeros(len(outcomes), dtype=np.float64)
    for row, outcome in enumerate(outcomes):
        if outcome.player_a in index:
            design[row, index[outcome.player_a]] = 1.0
        if outcome.player_b in index:
            design[row, index[outcome.player_b]] = -1.0
        targets[row] = float(outcome.score_a)
    return design, targets


def _require_anchor_connectivity(
    players: set[str], outcomes: list[MatchOutcome], anchor: str
) -> None:
    graph: dict[str, set[str]] = defaultdict(set)
    for outcome in outcomes:
        graph[outcome.player_a].add(outcome.player_b)
        graph[outcome.player_b].add(outcome.player_a)
    reachable = {anchor}
    queue = deque([anchor])
    while queue:
        player = queue.popleft()
        for neighbour in graph[player]:
            if neighbour not in reachable:
                reachable.add(neighbour)
                queue.append(neighbour)
    disconnected = sorted(players - reachable)
    if disconnected:
        raise ValueError("every player must be connected to the anchor")


def _sigmoid(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.where(
        values >= 0.0,
        1.0 / (1.0 + np.exp(-values)),
        np.exp(values) / (1.0 + np.exp(values)),
    )


def _objective(
    logits: NDArray[np.float64],
    targets: NDArray[np.float64],
    strengths: NDArray[np.float64],
    l2: float,
) -> float:
    # logaddexp gives stable -log(sigmoid) terms at extreme rating gaps.
    return float(
        np.sum(np.logaddexp(0.0, logits) - targets * logits)
        + 0.5 * l2 * (strengths @ strengths)
    )
