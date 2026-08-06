"""Behavior tests for legal-mask PUCT and actor-aware value backup."""

from __future__ import annotations

import time

import numpy as np

from games.snakego import SnakeGoGame, SnakeGoState, SnakeState
from rlbench.algorithms.alphazero import AlphaZeroConfig, MCTS
from rlbench.game import (
    BoardObservationSpec,
    DiscreteGameSpec,
    Observation,
    StepRecord,
)

from tests.toy_games.tictactoe import TicTacToe


class ConstantBatchEvaluator:
    """Record the batch boundary while returning fixed network predictions."""

    def __init__(self, logits: np.ndarray | None = None, value: float = 0.0) -> None:
        self.logits = (
            np.zeros(9, dtype=np.float32) if logits is None else logits.astype(np.float32)
        )
        self.value = value
        self.batch_sizes: list[int] = []

    def evaluate_batch(
        self, observations: list[object], legal_masks: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        self.batch_sizes.append(len(observations))
        return (
            np.stack([self.logits.copy() for _ in observations]),
            np.full(len(legal_masks), self.value, dtype=np.float32),
        )


class SamePlayerTwiceGame:
    """A tiny game whose first edge retains the actor and second edge changes it."""

    spec = DiscreteGameSpec(
        name="same-player-twice",
        players=2,
        zero_sum=True,
        action_names=("continue",),
        observation_spec=BoardObservationSpec(
            plane_names=("depth",), board_shape=(1, 1)
        ),
        max_episode_steps=2,
    )

    def __init__(self) -> None:
        self.depth = 0
        self.player = 0

    def reset(self, seed: int) -> None:
        del seed
        self.depth = 0
        self.player = 0

    def current_player(self) -> int:
        return self.player

    def observe(self, player: int) -> Observation:
        del player
        return Observation(
            planes=np.array([[[self.depth]]], dtype=np.float32),
            scalars=np.empty((0,), dtype=np.float32),
        )

    def legal_action_mask(self) -> np.ndarray:
        return np.array([self.depth < 2], dtype=np.bool_)

    def step(self, action: int) -> StepRecord:
        if action != 0 or self.depth >= 2:
            raise ValueError("illegal action")
        actor = self.player
        self.depth += 1
        if self.depth == 2:
            self.player = 1
        return StepRecord(player=actor, action=action, terminated=self.depth == 2)

    def outcome(self, player: int) -> float | None:
        if self.depth < 2:
            return None
        return 1.0 if player == 0 else -1.0

    def clone(self) -> SamePlayerTwiceGame:
        copied = SamePlayerTwiceGame()
        copied.depth = self.depth
        copied.player = self.player
        return copied


def test_mcts_expands_only_actions_permitted_by_the_legal_mask() -> None:
    """Ignoring the legal mask would create edges for occupied cells."""
    game = TicTacToe()
    game.set_position("XO.X.....", player=1)
    logits = np.array([100.0, 90.0, 0.0, 80.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    result = MCTS(_config(simulations=12), ConstantBatchEvaluator(logits)).search(game)

    assert set(result.expanded_actions) == {2, 4, 5, 6, 7, 8}
    assert np.all(result.visit_policy[[0, 1, 3]] == 0.0)


def test_mcts_discovers_an_immediate_win() -> None:
    """Removing terminal leaf handling would miss the forced winning move."""
    game = TicTacToe()
    game.set_position("XX.OO....", player=0)

    result = MCTS(_config(simulations=48), ConstantBatchEvaluator()).search(game)

    assert result.action == 2
    assert result.action_values[2] == 1.0
    assert result.visit_counts[2] > max(result.visit_counts[5:])


def test_mcts_backs_up_values_in_the_parent_player_perspective() -> None:
    """Omitting the perspective flip would score a winning edge as a loss."""
    game = TicTacToe()
    game.set_position("XX.OO....", player=0)

    result = MCTS(_config(simulations=16), ConstantBatchEvaluator()).search(game)

    assert result.action_values[2] == 1.0


def test_mcts_exports_normalized_root_visits_and_uses_a_batch_boundary() -> None:
    """Leaf search must aggregate real inference batches within the configured cap."""
    evaluator = ConstantBatchEvaluator()
    result = MCTS(
        _config(simulations=14, inference_batch_size=4), evaluator
    ).search(TicTacToe())

    assert np.isclose(result.visit_policy.sum(), 1.0)
    assert np.allclose(
        result.visit_policy,
        result.visit_counts / result.visit_counts.sum(),
    )
    assert evaluator.batch_sizes
    assert max(evaluator.batch_sizes) > 1
    assert max(evaluator.batch_sizes) <= 4


def test_mcts_never_mutates_the_supplied_root_game() -> None:
    """Searching cloned simulations must leave the caller-owned state untouched."""
    game = TicTacToe()
    game.set_position("XO.X.....", player=1)
    original_board = game.board.copy()
    original_player = game.current_player()

    MCTS(_config(simulations=20), ConstantBatchEvaluator()).search(game)

    assert np.array_equal(game.board, original_board)
    assert game.current_player() == original_player
    assert game.outcome(0) is None


def test_evaluation_search_is_deterministic_and_does_not_add_root_noise() -> None:
    """Evaluation must ignore training-only randomness and temperature."""
    config = _config(simulations=18, root_dirichlet_fraction=0.75, temperature=2.0)

    first = MCTS(config, ConstantBatchEvaluator(), seed=1).search(
        TicTacToe(), training=False
    )
    second = MCTS(config, ConstantBatchEvaluator(), seed=999).search(
        TicTacToe(), training=False
    )

    assert first.action == second.action
    assert np.array_equal(first.visit_counts, second.visit_counts)
    assert np.array_equal(first.visit_policy, second.visit_policy)


def test_evaluation_with_only_a_root_expansion_still_selects_a_legal_action() -> None:
    """Zero child visits must not make argmax default to an illegal action."""
    game = TicTacToe()
    game.set_position("X........", player=1)
    logits = np.arange(9, dtype=np.float32)

    result = MCTS(_config(simulations=1), ConstantBatchEvaluator(logits)).search(game)

    assert result.action == 8
    assert game.legal_action_mask()[result.action]


def test_mcts_stops_at_a_completed_simulation_wave_after_deadline() -> None:
    """Ignoring the deadline between waves makes trusted search overrun every move."""

    class SlowBatchEvaluator(ConstantBatchEvaluator):
        def evaluate_batch(
            self, observations: list[object], legal_masks: list[np.ndarray]
        ) -> tuple[np.ndarray, np.ndarray]:
            time.sleep(0.01)
            return super().evaluate_batch(observations, legal_masks)

    game = TicTacToe()
    evaluator = SlowBatchEvaluator()
    started = time.monotonic()
    result = MCTS(
        _config(simulations=100, inference_batch_size=4), evaluator
    ).search(game, deadline=started + 0.035)
    elapsed = time.monotonic() - started

    assert game.legal_action_mask()[result.action]
    assert sum(evaluator.batch_sizes) < 100
    assert result.completed_simulations < 100
    assert result.completed_simulations == int(result.visit_counts.sum()) + 1
    assert elapsed < 0.15


def test_mcts_without_deadline_preserves_the_exact_simulation_budget() -> None:
    """A deadline hook must not reduce ordinary training or unlimited evaluation search."""
    evaluator = ConstantBatchEvaluator()

    result = MCTS(
        _config(simulations=14, inference_batch_size=4), evaluator
    ).search(TicTacToe(), deadline=None)

    assert sum(evaluator.batch_sizes) == 14
    assert result.completed_simulations == 14
    assert int(result.visit_counts.sum()) == 13


def test_single_simulation_training_renormalizes_float32_root_policy() -> None:
    """Float32 softmax rounding must not make minimum-budget sampling invalid."""
    game = TicTacToe()
    game.set_position("XOXOOXX..", player=1)
    logits = np.zeros(9, dtype=np.float32)
    logits[7:] = (0.64042265, 0.10490012)

    result = MCTS(
        _config(simulations=1), ConstantBatchEvaluator(logits), seed=0
    ).search(game, training=True, move_number=0)

    assert game.legal_action_mask()[result.action]


def test_nonterminal_network_value_flips_across_two_alternating_plies() -> None:
    """A depth-two value for the root player must remain positive at the root edge."""

    class DepthValueEvaluator(ConstantBatchEvaluator):
        def evaluate_batch(
            self, observations: list[object], legal_masks: list[np.ndarray]
        ) -> tuple[np.ndarray, np.ndarray]:
            self.batch_sizes.append(len(observations))
            values = np.array(
                [
                    0.8 if float(observation.planes.sum()) == 2.0 else 0.0
                    for observation in observations
                ],
                dtype=np.float32,
            )
            logits = np.tile(
                np.arange(9, 0, -1, dtype=np.float32),
                (len(observations), 1),
            )
            return logits, values

    result = MCTS(
        _config(simulations=3, inference_batch_size=1), DepthValueEvaluator()
    ).search(TicTacToe())

    assert np.isclose(result.action_values[0], 0.4)


def test_nonterminal_network_value_does_not_flip_across_a_same_player_edge() -> None:
    """Treating every edge as alternating reverses a retained actor's evaluation."""

    class DepthEvaluator(ConstantBatchEvaluator):
        def evaluate_batch(
            self, observations: list[object], legal_masks: list[np.ndarray]
        ) -> tuple[np.ndarray, np.ndarray]:
            self.batch_sizes.append(len(observations))
            return (
                np.zeros((len(observations), 1), dtype=np.float32),
                np.array(
                    [
                        0.75 if float(observation.planes[0, 0, 0]) == 1.0 else 0.0
                        for observation in observations
                    ],
                    dtype=np.float32,
                ),
            )

    result = MCTS(
        _config(simulations=2, inference_batch_size=1), DepthEvaluator()
    ).search(SamePlayerTwiceGame())

    assert result.action_values[0] == 0.75


def test_mcts_uses_the_next_active_snake_actor_from_real_snakego_state() -> None:
    """A SnakeGo edge can retain player zero while advancing to its second snake."""
    game = SnakeGoGame.from_state(
        SnakeGoState(
            turn=9,
            current_player=0,
            max_round=20,
            snakes=[
                SnakeState(0, 0, [(3, 3), (2, 3)]),
                SnakeState(2, 0, [(7, 7)]),
                SnakeState(1, 1, [(13, 13)]),
            ],
            phase_snake_ids=[0, 2],
        )
    )
    evaluator = ConstantBatchEvaluator(
        logits=np.array([9.0, 0.0, 0.0, 0.0, 0.0, 0.0]), value=0.625
    )

    result = MCTS(
        _config(simulations=2, inference_batch_size=1), evaluator
    ).search(game)

    assert result.action == 0
    assert result.action_values[0] == 0.625


def _config(
    *,
    simulations: int,
    root_dirichlet_fraction: float = 0.0,
    temperature: float = 1.0,
    inference_batch_size: int = 32,
) -> AlphaZeroConfig:
    return AlphaZeroConfig(
        simulations=simulations,
        c_puct=1.5,
        root_dirichlet_alpha=0.3,
        root_dirichlet_fraction=root_dirichlet_fraction,
        self_play_temperature=temperature,
        temperature_moves=9,
        inference_batch_size=inference_batch_size,
    )
