"""Replay, learning, self-play, and checkpoint behavior tests."""

from __future__ import annotations

import copy
from dataclasses import replace
import random
import time

import numpy as np
import pytest
import torch

from rlbench.algorithms.alphazero import (
    AlphaZeroConfig,
    AlphaZeroTrainer,
    ExpertTrajectory,
    PolicyValueNet,
    ReplayBuffer,
    ReplaySample,
    SelfPlayWorker,
)
from rlbench.algorithms import PolicyCheckpoint
from rlbench.league import LeagueMember, LeagueState
from rlbench.population import ProcessMoveTimeout
from rlbench.telemetry import EventLedger

from tests.toy_games.tictactoe import TicTacToe


torch.set_num_threads(1)


class TacticalTicTacToe(TicTacToe):
    """A legal midgame whose current player has a non-lowest immediate win."""

    def reset(self, seed: int) -> None:
        del seed
        self.set_position("OOX..X...", player=0)


def test_replay_retains_observation_mask_visit_policy_and_outcome() -> None:
    """Dropping any learning target would make sampled batches incomplete."""
    game = _training_position()
    sample = _winning_sample(game)
    replay = ReplayBuffer(capacity=4, seed=7)

    replay.add(sample)
    batch = replay.sample(1)

    assert np.array_equal(batch.planes[0], sample.observation.planes)
    assert np.array_equal(batch.scalars[0], sample.observation.scalars)
    assert np.array_equal(batch.legal_masks[0], sample.legal_mask)
    assert np.array_equal(batch.visit_policies[0], sample.visit_policy)
    assert batch.outcomes.tolist() == [1.0]


def test_replay_rejects_over_capacity_checkpoint_payload_before_loading() -> None:
    """A bounded deque must not silently truncate a malformed checkpoint."""
    replay = ReplayBuffer(capacity=2, seed=7)
    replay.add(_winning_sample(_training_position()))
    state = replay.state_dict()
    state["samples"] = state["samples"] * 3

    with pytest.raises(ValueError, match="exceeds"):
        ReplayBuffer(capacity=2).load_state_dict(state)


def test_one_optimizer_step_reduces_loss_on_a_fixed_batch() -> None:
    """Skipping either policy or value optimization would not fit this target."""
    torch.manual_seed(3)
    game = _training_position()
    trainer = _trainer(game, learning_rate=0.05)
    for _ in range(8):
        trainer.replay.add(_winning_sample(game))
    batch = trainer.replay.sample(8)

    before = trainer.loss_on_batch(batch).total
    trainer.train_step(batch)
    after = trainer.loss_on_batch(batch).total

    assert after < before
    assert trainer.optimizer_steps == 1


def test_optimizer_continuation_checks_deadline_before_next_step() -> None:
    game = _training_position()
    trainer = _trainer(game, learning_rate=0.05)
    for _ in range(8):
        trainer.replay.add(_winning_sample(game))

    with pytest.raises(TimeoutError, match="allocation deadline"):
        trainer.run_optimizer_steps(
            4,
            deadline_monotonic=time.monotonic() - 1.0,
        )

    assert trainer.optimizer_steps == 0


def test_complete_expert_trajectories_use_generic_replay_and_budget_path(
    tmp_path,
) -> None:
    ledger = EventLedger(tmp_path / "trajectory-events.jsonl")
    trainer = _trainer(TicTacToe(), learning_rate=0.02, ledger=ledger)
    winning_game = (0, 3, 1, 4, 2)

    result = trainer.distill_expert_trajectories(
        TicTacToe,
        (
            ExpertTrajectory(seed=101, actions=winning_game),
            ExpertTrajectory(seed=102, actions=winning_game),
        ),
        training_steps=2,
        fresh_replay=True,
        base_weight=2.0,
        opening_moves=1,
        opening_weight=5.0,
    )

    samples = tuple(trainer.replay._samples)
    assert result.generation == 1
    assert result.episodes == 2
    assert result.replay_samples == 10
    assert result.optimizer_steps == 2
    assert trainer.budgets.learning.episodes == 2
    assert trainer.budgets.learning.env_steps == 10
    assert {sample.source for sample in samples} == {"expert_demo"}
    assert [sample.outcome for sample in samples[:5]] == [
        1.0,
        -1.0,
        1.0,
        -1.0,
        1.0,
    ]
    assert [sample.decision_index for sample in samples[:5]] == [0, 0, 1, 1, 2]
    assert [sample.sample_weight for sample in samples[:5]] == [
        5.0,
        5.0,
        2.0,
        2.0,
        2.0,
    ]
    event = next(
        event
        for event in ledger.read()
        if event.event_type == "alphazero_expert_distillation"
    )
    assert event.payload["expert_samples"] == 10
    assert event.payload["completed_optimizer_steps"] == 2


def test_expert_trajectory_requires_a_complete_legal_game() -> None:
    trainer = _trainer(TicTacToe(), learning_rate=0.02)

    with pytest.raises(ValueError, match="terminal state"):
        trainer.distill_expert_trajectories(
            TicTacToe,
            (ExpertTrajectory(seed=1, actions=(0, 3)),),
            training_steps=0,
        )

    with pytest.raises(ValueError, match="illegal action"):
        trainer.distill_expert_trajectories(
            TicTacToe,
            (ExpertTrajectory(seed=1, actions=(0, 0, 1, 4, 2)),),
            training_steps=0,
        )


def test_checkpoint_restores_identical_logits_rng_and_counters(tmp_path) -> None:
    """A partial resume would change inference, randomness, or accounting."""
    torch.manual_seed(11)
    game = _training_position()
    trainer = _trainer(game, learning_rate=0.02)
    trainer.replay.add(_winning_sample(game))
    trainer.train_step(trainer.replay.sample(1))
    trainer.generation = 4
    trainer.budgets.learning.env_steps = 7
    planes, scalars, mask = _network_inputs(game)
    with torch.inference_mode():
        saved_logits = trainer.network(planes, scalars, mask)[0].clone()
    saved_optimizer = copy.deepcopy(trainer.optimizer.state_dict())
    saved_replay = copy.deepcopy(trainer.replay.state_dict())
    saved_scaler = copy.deepcopy(trainer.scaler.state_dict())
    saved_budgets = copy.deepcopy(trainer.budgets.as_dict())

    checkpoint_path = tmp_path / "policy.pt"
    trainer.save_checkpoint(checkpoint_path)
    expected_python = [random.random() for _ in range(4)]
    expected_global_numpy = np.random.random(4)
    expected_numpy = trainer.rng.random(4)
    expected_torch = torch.rand(4)
    saved_steps = trainer.optimizer_steps
    with torch.no_grad():
        for parameter in trainer.network.parameters():
            parameter.add_(1.0)
    trainer.generation = 999
    trainer.optimizer_steps = 999

    trainer.load_checkpoint(checkpoint_path)

    with torch.inference_mode():
        restored_logits = trainer.network(planes, scalars, mask)[0]
    assert torch.equal(restored_logits, saved_logits)
    assert [random.random() for _ in range(4)] == expected_python
    assert np.array_equal(np.random.random(4), expected_global_numpy)
    assert np.array_equal(trainer.rng.random(4), expected_numpy)
    assert torch.equal(torch.rand(4), expected_torch)
    assert trainer.generation == 4
    assert trainer.optimizer_steps == saved_steps
    assert trainer.budgets.as_dict() == saved_budgets
    assert trainer.scaler.state_dict() == saved_scaler
    _assert_nested_equal(trainer.optimizer.state_dict(), saved_optimizer)
    _assert_nested_equal(trainer.replay.state_dict(), saved_replay)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_checkpoint_loaded_to_cuda_restores_cuda_rng(tmp_path) -> None:
    """map_location=cuda must not leave RNG state tensors on the CUDA device."""
    torch.manual_seed(53)
    torch.cuda.manual_seed_all(59)
    game = _training_position()
    config = _config()
    network = PolicyValueNet.from_game_spec(game.spec, config, device="cuda")
    trainer = AlphaZeroTrainer(network, config, seed=61)
    checkpoint_path = tmp_path / "cuda-rng.pt"
    trainer.save_checkpoint(checkpoint_path)
    expected = torch.rand(4, device="cuda")

    torch.cuda.manual_seed_all(67)
    trainer.load_checkpoint(checkpoint_path)

    assert torch.equal(torch.rand(4, device="cuda"), expected)


@pytest.mark.parametrize("corruption", ["optimizer", "scaler", "trainer"])
def test_malformed_checkpoint_restore_is_transactional(tmp_path, corruption) -> None:
    """Any malformed late field must leave all live trainer state untouched."""
    torch.manual_seed(31)
    game = _training_position()
    trainer = _trainer(game, learning_rate=0.02)
    trainer.replay.add(_winning_sample(game))
    trainer.train_step(trainer.replay.sample(1))
    checkpoint_path = tmp_path / f"malformed-{corruption}.pt"
    trainer.save_checkpoint(checkpoint_path)

    with torch.no_grad():
        for parameter in trainer.network.parameters():
            parameter.add_(0.25)
    trainer.replay.add(_losing_sample(game))
    trainer.replay.sample(1)
    trainer.generation = 8
    trainer.optimizer_steps = 9
    trainer.budgets.learning.env_steps = 12
    trainer.rng.random(3)
    random.random()
    np.random.random()
    torch.rand(1)
    live_state = _trainer_state_snapshot(trainer, game)

    payload = torch.load(checkpoint_path, weights_only=False)
    if corruption == "optimizer":
        payload["optimizer_state"] = {"state": {}, "param_groups": []}
    elif corruption == "scaler":
        payload["trainer_state"]["scaler"] = "not-a-scaler-state"
    else:
        payload["trainer_state"]["optimizer_steps"] = -1
    torch.save(payload, checkpoint_path)

    with pytest.raises(ValueError):
        trainer.load_checkpoint(checkpoint_path)

    _assert_trainer_snapshot(trainer, game, live_state)


def test_checkpoint_rejects_shape_incompatible_adam_slots_transactionally(
    tmp_path,
) -> None:
    """AdamW accepts malformed slot shapes at load but the next step cannot use them."""
    torch.manual_seed(37)
    game = _training_position()
    trainer = _trainer(game, learning_rate=0.02)
    trainer.replay.add(_winning_sample(game))
    trainer.train_step(trainer.replay.sample(1))
    checkpoint_path = tmp_path / "malformed-slot-shape.pt"
    trainer.save_checkpoint(checkpoint_path)

    with torch.no_grad():
        for parameter in trainer.network.parameters():
            parameter.add_(0.5)
    trainer.replay.add(_losing_sample(game))
    trainer.replay.sample(1)
    trainer.generation = 6
    trainer.optimizer_steps = 7
    trainer.budgets.learning.env_steps = 10
    trainer.rng.random(2)
    random.random()
    np.random.random()
    torch.rand(1)
    live_state = _trainer_state_snapshot(trainer, game)

    payload = torch.load(checkpoint_path, weights_only=False)
    for state in payload["optimizer_state"]["state"].values():
        if "exp_avg" not in state or state["exp_avg"].numel() <= 1:
            continue
        state["exp_avg"] = torch.zeros(
            (1,), dtype=state["exp_avg"].dtype, device=state["exp_avg"].device
        )
        state["exp_avg_sq"] = torch.zeros(
            (1,),
            dtype=state["exp_avg_sq"].dtype,
            device=state["exp_avg_sq"].device,
        )
        break
    else:
        raise AssertionError("fixture did not find a non-scalar AdamW parameter state")
    torch.save(payload, checkpoint_path)

    with pytest.raises(ValueError, match="optimizer"):
        trainer.load_checkpoint(checkpoint_path)

    _assert_trainer_snapshot(trainer, game, live_state)


def test_policy_checkpoint_round_trips_valid_adafactor_slots(tmp_path) -> None:
    """Factored row and column slots must remain valid for generic checkpoints."""
    torch.manual_seed(41)
    model = torch.nn.Linear(4, 3, bias=False)
    optimizer = torch.optim.Adafactor(model.parameters(), lr=0.01)
    inputs = torch.tensor(
        [[1.0, -2.0, 0.5, 3.0], [-1.0, 0.25, 2.0, -0.5]],
        dtype=torch.float32,
    )
    model(inputs).square().sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    with torch.inference_mode():
        expected = model(inputs).clone()

    checkpoint_path = tmp_path / "adafactor.pt"
    PolicyCheckpoint.save(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
    )
    restored_model = torch.nn.Linear(4, 3, bias=False)
    restored_optimizer = torch.optim.Adafactor(restored_model.parameters(), lr=0.01)

    PolicyCheckpoint.load(checkpoint_path).restore(
        model=restored_model,
        optimizer=restored_optimizer,
    )

    with torch.inference_mode():
        assert torch.equal(restored_model(inputs), expected)
    restored_model(inputs).square().sum().backward()
    restored_optimizer.step()
    assert all(
        torch.isfinite(parameter).all() for parameter in restored_model.parameters()
    )


def test_self_play_records_normalized_legal_visit_targets_and_outcomes(tmp_path) -> None:
    """Self-play must store search behavior and final current-player outcomes."""
    torch.manual_seed(5)
    game = TicTacToe()
    config = _config(simulations=8)
    network = PolicyValueNet.from_game_spec(game.spec, config)
    ledger = EventLedger(tmp_path / "events.jsonl")
    worker = SelfPlayWorker(
        TicTacToe, network, config, seed=13, ledger=ledger, run_id="selfplay-test"
    )

    samples = worker.play_episode(seed=17)
    episode = worker.last_episode

    assert 5 <= len(samples) <= 9
    assert all(np.isclose(sample.visit_policy.sum(), 1.0) for sample in samples)
    assert all(np.all(sample.visit_policy[~sample.legal_mask] == 0.0) for sample in samples)
    assert all(sample.outcome in (-1.0, 0.0, 1.0) for sample in samples)
    assert {sample.player for sample in samples} == {0, 1}
    assert all(sample.outcome == episode.outcomes[sample.player] for sample in samples)
    events = list(ledger.read())
    assert sum(event.event_type == "alphazero_self_play_step" for event in events) == len(
        samples
    )
    assert sum(
        event.event_type == "alphazero_self_play_episode" for event in events
    ) == 1


def test_training_episode_against_opponent_keeps_only_learner_search_targets(
    tmp_path,
) -> None:
    """Training on a human must not label its actions as learner MCTS targets."""

    class LifecycleOpponent:
        def __init__(self) -> None:
            self.begun: list[tuple[str, int]] = []
            self.observed: list[tuple[int, int]] = []
            self.ended = 0

        def begin_game(self, case, agent_id: str, side: int, game) -> None:
            del case, game
            self.begun.append((agent_id, side))

        def act_game_process(self, game, *, timeout_seconds: float | None) -> int:
            assert timeout_seconds == 0.5
            return int(np.flatnonzero(game.legal_action_mask())[0])

        def observe_action(self, game, actor: int, action: int) -> None:
            del game
            self.observed.append((actor, action))

        def end_game(self, game, result) -> None:
            del game
            assert result.valid is True
            assert result.reason == "completed"
            self.ended += 1

    opponent = LifecycleOpponent()
    config = _config(simulations=8)
    worker = SelfPlayWorker(
        TicTacToe,
        PolicyValueNet.from_game_spec(TicTacToe.spec, config),
        config,
        ledger=EventLedger(tmp_path / "mixed-events.jsonl"),
    )

    samples = worker.play_episode_against(
        opponent,
        opponent_id="train-human",
        learner_player=0,
        seed=29,
        timeout_seconds=0.5,
    )

    episode = worker.last_episode
    assert samples
    assert all(sample.player == 0 for sample in samples)
    assert all(sample.outcome == episode.outcomes[0] for sample in samples)
    assert episode.stats.mcts_simulations == len(samples) * config.simulations
    assert opponent.begun == [("train-human", 1)]
    assert len(opponent.observed) == len(episode.decisions)
    assert opponent.ended == 1

    other_side_samples = worker.play_episode_against(
        opponent,
        opponent_id="train-human",
        learner_player=1,
        seed=31,
        timeout_seconds=0.5,
    )

    other_episode = worker.last_episode
    assert other_side_samples
    assert all(sample.player == 1 for sample in other_side_samples)
    assert all(
        sample.outcome == other_episode.outcomes[1]
        for sample in other_side_samples
    )
    assert opponent.begun == [("train-human", 1), ("train-human", 0)]
    assert opponent.ended == 2


def test_expert_demo_replay_uses_one_hot_actions_and_human_value_perspective(
    tmp_path,
) -> None:
    """Expert labels must encode the human action and that same camp's result."""

    class HighestLegalOpponent:
        def act_game(self, game) -> int:
            return int(np.flatnonzero(game.legal_action_mask())[-1])

    config = _config(simulations=4)
    ledger = EventLedger(tmp_path / "expert-events.jsonl")
    worker = SelfPlayWorker(
        TicTacToe,
        PolicyValueNet.from_game_spec(TicTacToe.spec, config),
        config,
        ledger=ledger,
    )

    samples = worker.play_episode_against(
        HighestLegalOpponent(),
        opponent_id="train-human",
        learner_player=0,
        seed=43,
        expert_demo=True,
    )

    episode = worker.last_episode
    expert_samples = [sample for sample in samples if sample.source == "expert_demo"]
    expert_decisions = [
        decision for decision in episode.decisions if decision.player == 1
    ]
    assert len(expert_samples) == len(expert_decisions) > 0
    for sample, decision in zip(expert_samples, expert_decisions, strict=True):
        expected = np.zeros_like(sample.visit_policy)
        expected[decision.action] = 1.0
        assert np.array_equal(sample.visit_policy, expected)
        assert sample.player == decision.player
        assert sample.outcome == episode.outcomes[decision.player]
    assert all(
        sample.source == "learner" for sample in samples if sample.player == 0
    )
    episode_event = next(
        event
        for event in ledger.read()
        if event.event_type == "alphazero_self_play_episode"
    )
    assert episode_event.payload["sample_sources"] == {
        "expert_demo": len(expert_samples),
        "learner": len(samples) - len(expert_samples),
        "selfplay": 0,
    }


@pytest.mark.parametrize("learner_player", [0, 1])
def test_expert_demo_values_follow_expert_camp_for_both_sides(
    learner_player,
) -> None:
    """Shared-network demonstrations retain the acting human camp's perspective."""

    class FirstLegalOpponent:
        def act_game(self, game) -> int:
            return int(np.flatnonzero(game.legal_action_mask())[0])

    config = _config(simulations=4)
    worker = SelfPlayWorker(
        TicTacToe,
        PolicyValueNet.from_game_spec(TicTacToe.spec, config),
        config,
    )

    samples = worker.play_episode_against(
        FirstLegalOpponent(),
        opponent_id="train-human",
        learner_player=learner_player,
        seed=47 + learner_player,
        expert_demo=True,
    )

    expert_player = 1 - learner_player
    expert_samples = [sample for sample in samples if sample.source == "expert_demo"]
    assert expert_samples
    assert {sample.player for sample in expert_samples} == {expert_player}
    assert all(
        sample.outcome == worker.last_episode.outcomes[expert_player]
        for sample in expert_samples
    )


def test_training_opponent_failure_closes_its_episode_process() -> None:
    """A timed-out human process must not survive a failed mixed episode."""

    class TimedOutOpponent:
        def __init__(self) -> None:
            self.closed = 0

        def act_game_process(self, game, *, timeout_seconds: float | None) -> int:
            del game, timeout_seconds
            raise ProcessMoveTimeout("training move deadline")

        def close(self) -> None:
            self.closed += 1

    opponent = TimedOutOpponent()
    config = _config(simulations=4)
    worker = SelfPlayWorker(
        TicTacToe,
        PolicyValueNet.from_game_spec(TicTacToe.spec, config),
        config,
    )

    with pytest.raises(ProcessMoveTimeout, match="training move deadline"):
        worker.play_episode_against(
            opponent,
            opponent_id="train-human",
            learner_player=1,
            seed=37,
            timeout_seconds=0.5,
        )

    assert opponent.closed == 1


def test_expert_episode_deadline_reaches_learner_mcts_wave_boundary() -> None:
    """A stage deadline must stop learner search inside an expert episode."""

    class SlowEvaluator:
        def __init__(self, delegate) -> None:
            self.delegate = delegate

        def evaluate_batch(self, observations, legal_masks):
            time.sleep(0.003)
            return self.delegate.evaluate_batch(observations, legal_masks)

    config = _config(simulations=128)
    network = PolicyValueNet.from_game_spec(TicTacToe.spec, config)
    worker = SelfPlayWorker(TicTacToe, SlowEvaluator(network), config)

    with pytest.raises(TimeoutError, match="allocation deadline"):
        worker.play_episode(
            seed=41,
            deadline_monotonic=time.monotonic() + 0.01,
        )


def test_expert_episode_preflights_configured_opponent_bound() -> None:
    """An opponent move cannot start when its configured bound exceeds remaining time."""

    class CountingOpponent:
        def __init__(self) -> None:
            self.calls = 0

        def act_game_process(self, game, *, timeout_seconds: float | None) -> int:
            del timeout_seconds
            self.calls += 1
            return int(np.flatnonzero(game.legal_action_mask())[0])

    opponent = CountingOpponent()
    config = _config(simulations=4)
    worker = SelfPlayWorker(
        TicTacToe,
        PolicyValueNet.from_game_spec(TicTacToe.spec, config),
        config,
    )

    with pytest.raises(TimeoutError, match="bounded unit"):
        worker.play_episode_against(
            opponent,
            opponent_id="train-human",
            learner_player=1,
            seed=43,
            timeout_seconds=0.5,
            deadline_monotonic=time.monotonic() + 0.01,
        )

    assert opponent.calls == 0


def test_trainer_mixture_samples_train_members_and_never_held_out_humans() -> None:
    """Allowing a held-out member into replay invalidates final evaluation."""

    class CountingOpponent:
        def __init__(self) -> None:
            self.actions = 0

        def act_game(self, game) -> int:
            self.actions += 1
            return int(np.flatnonzero(game.legal_action_mask())[0])

    train_opponent = CountingOpponent()
    held_out_opponent = CountingOpponent()
    config = _config(simulations=4)
    network = PolicyValueNet.from_game_spec(TicTacToe.spec, config)
    league = LeagueState(
        anchor_id="train-human",
        champion_id="train-human",
        members=(
            LeagueMember(
                "train-human", "sha256:" + "a" * 64, "train_human"
            ),
            LeagueMember(
                "held-out", "sha256:" + "b" * 64, "test_human"
            ),
        ),
    )
    trainer = AlphaZeroTrainer(network, config, seed=31, league=league)

    result = trainer.run_generation(
        TicTacToe,
        self_play_episodes=2,
        training_steps=0,
        processes=1,
        training_opponents={
            "train-human": train_opponent,
            "held-out": held_out_opponent,
        },
        opponent_episodes=2,
        opponent_move_seconds=0.5,
    )

    assert result.episodes == 2
    assert train_opponent.actions > 0
    assert held_out_opponent.actions == 0
    assert trainer.budgets.learning.episodes == 2

    with pytest.raises(ValueError, match="sequential"):
        trainer.run_generation(
            TicTacToe,
            self_play_episodes=1,
            training_steps=0,
            processes=2,
            training_opponents={"train-human": train_opponent},
            opponent_episodes=1,
        )


def test_expert_demo_mode_strictly_rejects_test_human_policy() -> None:
    """Held-out actions must never enter demonstration replay, even unsampled."""
    config = _config(simulations=4)
    league = LeagueState(
        anchor_id="train-human",
        champion_id="train-human",
        members=(
            LeagueMember("train-human", "sha256:" + "a" * 64, "train_human"),
            LeagueMember("held-out", "sha256:" + "b" * 64, "test_human"),
        ),
    )
    trainer = AlphaZeroTrainer(
        PolicyValueNet.from_game_spec(TicTacToe.spec, config),
        config,
        seed=53,
        league=league,
    )

    with pytest.raises(ValueError, match="test_human"):
        trainer.run_generation(
            TicTacToe,
            self_play_episodes=1,
            training_steps=0,
            processes=1,
            training_opponents={"train-human": object(), "held-out": object()},
            opponent_episodes=1,
            expert_demo=True,
        )


def test_replay_checkpoint_round_trips_sample_source() -> None:
    """Resume must retain whether a target came from search or an expert."""
    game = _training_position()
    original = _winning_sample(game, source="expert_demo")
    replay = ReplayBuffer(capacity=4, seed=59)
    replay.add(original)

    restored = ReplayBuffer(capacity=4, seed=59)
    restored.load_state_dict(replay.state_dict())

    state = restored.state_dict()
    assert state["samples"][0]["source"] == "expert_demo"


def test_expert_demo_opening_boost_sets_weight_and_decision_index() -> None:
    """Only the configured prefix of expert decisions receives extra weight."""

    class FirstLegalOpponent:
        def act_game(self, game) -> int:
            return int(np.flatnonzero(game.legal_action_mask())[0])

    config = _config(simulations=4)
    worker = SelfPlayWorker(
        TicTacToe,
        PolicyValueNet.from_game_spec(TicTacToe.spec, config),
        config,
    )

    samples = worker.play_episode_against(
        FirstLegalOpponent(),
        opponent_id="train-human",
        learner_player=0,
        seed=61,
        expert_demo=True,
        expert_demo_opening_moves=2,
        expert_demo_opening_weight=7.0,
    )

    expert = [sample for sample in samples if sample.source == "expert_demo"]
    assert [sample.decision_index for sample in expert] == list(range(len(expert)))
    assert [sample.sample_weight for sample in expert] == [
        7.0 if index < 2 else 1.0 for index in range(len(expert))
    ]
    assert all(
        sample.sample_weight == 1.0 and sample.decision_index == -1
        for sample in samples
        if sample.source == "learner"
    )


def test_expert_demo_limit_keeps_only_the_configured_opening_prefix() -> None:
    """Long post-elimination play must not flood replay with irrelevant demos."""

    class FirstLegalOpponent:
        def act_game(self, game) -> int:
            return int(np.flatnonzero(game.legal_action_mask())[0])

    config = _config(simulations=4)
    worker = SelfPlayWorker(
        TicTacToe,
        PolicyValueNet.from_game_spec(TicTacToe.spec, config),
        config,
    )

    samples = worker.play_episode_against(
        FirstLegalOpponent(),
        opponent_id="train-human",
        learner_player=0,
        seed=71,
        expert_demo=True,
        expert_demo_max_decisions=1,
    )

    expert = [sample for sample in samples if sample.source == "expert_demo"]
    assert len(expert) == 1
    assert expert[0].decision_index == 0


@pytest.mark.parametrize("limit", [-1, True])
def test_expert_demo_limit_rejects_invalid_values(limit) -> None:
    config = _config(simulations=4)
    worker = SelfPlayWorker(
        TicTacToe,
        PolicyValueNet.from_game_spec(TicTacToe.spec, config),
        config,
    )

    with pytest.raises(ValueError, match="expert_demo_max_decisions"):
        worker.play_episode_against(
            object(),
            opponent_id="train-human",
            learner_player=0,
            seed=73,
            expert_demo=True,
            expert_demo_max_decisions=limit,
        )


def test_trainer_propagates_expert_demo_limit_to_mixed_episodes() -> None:
    class FirstLegalOpponent:
        def act_game(self, game) -> int:
            return int(np.flatnonzero(game.legal_action_mask())[0])

    config = _config(simulations=4)
    opponent_id = "train-human"
    league = LeagueState(
        anchor_id=opponent_id,
        champion_id=opponent_id,
        members=(
            LeagueMember(opponent_id, "sha256:" + "a" * 64, "train_human"),
        ),
    )
    trainer = AlphaZeroTrainer(
        PolicyValueNet.from_game_spec(TicTacToe.spec, config),
        config,
        seed=79,
        league=league,
    )

    trainer.run_generation(
        TicTacToe,
        self_play_episodes=1,
        training_steps=0,
        processes=1,
        training_opponents={opponent_id: FirstLegalOpponent()},
        opponent_episodes=1,
        expert_demo=True,
        expert_demo_max_decisions=1,
    )

    sources = [sample["source"] for sample in trainer.replay.state_dict()["samples"]]
    assert sources.count("expert_demo") == 1


def test_pure_selfplay_targets_remain_uniformly_weighted() -> None:
    """Opening emphasis must not alter ordinary self-play optimization."""
    config = _config(simulations=4)
    worker = SelfPlayWorker(
        TicTacToe,
        PolicyValueNet.from_game_spec(TicTacToe.spec, config),
        config,
    )

    samples = worker.play_episode(seed=67)

    assert all(sample.source == "selfplay" for sample in samples)
    assert all(sample.sample_weight == 1.0 for sample in samples)
    assert all(sample.decision_index == -1 for sample in samples)


def test_training_opponent_is_closed_when_begin_game_raises() -> None:
    """A partially started process must be cleaned up on lifecycle failure."""

    class FailingOpponent:
        def __init__(self) -> None:
            self.closed = 0

        def begin_game(self, case, agent_id: str, side: int, game) -> None:
            del case, agent_id, side, game
            raise RuntimeError("begin failed")

        def close(self) -> None:
            self.closed += 1

    opponent = FailingOpponent()
    config = _config(simulations=1)
    worker = SelfPlayWorker(
        TicTacToe,
        PolicyValueNet.from_game_spec(TicTacToe.spec, config),
        config,
    )

    with pytest.raises(RuntimeError, match="begin failed"):
        worker.play_episode_against(
            opponent,
            opponent_id="failing",
            learner_player=0,
        )

    assert opponent.closed == 1


def test_old_replay_checkpoint_defaults_weight_and_decision_index() -> None:
    """Checkpoints written before weighted replay remain loadable and uniform."""
    replay = ReplayBuffer(capacity=4, seed=71)
    replay.add(_winning_sample(_training_position()))
    state = replay.state_dict()
    state["samples"][0].pop("sample_weight")
    state["samples"][0].pop("decision_index")

    restored = ReplayBuffer(capacity=4, seed=71)
    restored.load_state_dict(state)

    sample = restored._samples[0]
    assert sample.sample_weight == 1.0
    assert sample.decision_index == -1


def test_optimizer_only_continuation_reports_sampled_weight_mass(
    tmp_path,
) -> None:
    """A replay-only phase must train without inventing episodes or generations."""
    game = _training_position()
    ledger = EventLedger(tmp_path / "optimizer-only-events.jsonl")
    trainer = _trainer(game, learning_rate=0.02, ledger=ledger)
    for _ in range(8):
        trainer.replay.add(_winning_sample(game, sample_weight=3.0))

    metrics = trainer.run_optimizer_steps(3)

    assert len(metrics) == 3
    assert trainer.optimizer_steps == 3
    assert trainer.generation == 0
    assert trainer.budgets.learning.episodes == 0
    event = next(
        event
        for event in ledger.read()
        if event.event_type == "alphazero_optimizer_continuation"
    )
    assert event.payload["batch_draws"] == 24
    assert event.payload["sampled_weight_mass"] == 72.0
    assert "effective_weighted_draws" not in event.payload


def test_sample_weights_scale_policy_and_value_loss() -> None:
    """Weighted replay must affect both heads with a normalized weighted mean."""
    game = _training_position()
    trainer = _trainer(game, learning_rate=0.02)
    trainer.replay.add(_winning_sample(game, sample_weight=5.0))
    trainer.replay.add(_losing_sample(game))
    batch = trainer.replay.sample(2)

    actual = trainer.loss_on_batch(batch)
    planes = torch.as_tensor(batch.planes, dtype=torch.float32)
    scalars = torch.as_tensor(batch.scalars, dtype=torch.float32)
    masks = torch.as_tensor(batch.legal_masks, dtype=torch.bool)
    was_training = trainer.network.training
    trainer.network.eval()
    try:
        with torch.inference_mode():
            logits, values = trainer.network(planes, scalars, masks)
    finally:
        trainer.network.train(was_training)
    targets = torch.as_tensor(batch.visit_policies, dtype=torch.float32)
    outcomes = torch.as_tensor(batch.outcomes, dtype=torch.float32)
    weights = torch.as_tensor(batch.sample_weights, dtype=torch.float32)
    policy_each = -(targets * torch.log_softmax(logits, dim=1)).sum(dim=1)
    value_each = (values - outcomes).square()
    expected_policy = float((policy_each * weights).sum() / weights.sum())
    expected_value = float((value_each * weights).sum() / weights.sum())

    assert actual.policy == pytest.approx(expected_policy)
    assert actual.value == pytest.approx(expected_value)
    assert actual.total == pytest.approx(expected_policy + expected_value)


def test_short_alphazero_loop_improves_deterministically_against_random(
    tmp_path,
) -> None:
    """Self-play root visits must drive measurable improvement through the loop."""
    torch.manual_seed(19)
    game = _training_position()
    ledger = EventLedger(tmp_path / "training-events.jsonl")
    trainer = _trainer(game, learning_rate=0.03, ledger=ledger)
    policy_linear = trainer.network.policy_head[-1]
    torch.nn.init.zeros_(policy_linear.weight)
    torch.nn.init.zeros_(policy_linear.bias)
    before = _score_against_random(trainer.network, seeds=range(24))

    result = trainer.run_generation(
        TacticalTicTacToe,
        self_play_episodes=8,
        training_steps=4,
        processes=1,
    )

    after = _score_against_random(trainer.network, seeds=range(24))
    assert result.optimizer_steps == 4
    assert len(trainer.replay) >= 8
    assert any(event.event_type == "budget_snapshot" for event in ledger.read())
    assert after > before


def test_post_load_stage_reseed_controls_episode_seed_actions_and_all_rngs(
    tmp_path,
) -> None:
    """Loading a checkpoint must not override the literal seed of the next stage."""
    config = replace(
        _config(simulations=8),
        root_dirichlet_fraction=0.5,
        self_play_temperature=1.0,
        temperature_moves=9,
    )
    network = PolicyValueNet.from_game_spec(TicTacToe().spec, config)
    saved = AlphaZeroTrainer(network, config, seed=991)
    checkpoint = tmp_path / "stage-seed.pt"
    saved.save_checkpoint(checkpoint)

    def resumed_trace(stage_seed: int, suffix: str):
        ledger = EventLedger(tmp_path / f"events-{suffix}.jsonl")
        resumed = AlphaZeroTrainer(
            PolicyValueNet.from_game_spec(TicTacToe().spec, config),
            config,
            seed=123,
            ledger=ledger,
            run_id=f"seed-{suffix}",
        )
        resumed.load_checkpoint(checkpoint)
        resumed.reseed_stage(stage_seed)

        expected_rng = np.random.default_rng(stage_seed).bit_generator.state
        assert resumed.rng.bit_generator.state == expected_rng
        assert resumed.replay.rng.bit_generator.state == expected_rng
        assert random.random() == random.Random(stage_seed).random()
        assert np.random.random() == np.random.RandomState(stage_seed).random_sample()
        expected_torch = torch.Generator().manual_seed(stage_seed)
        assert torch.rand(1).item() == torch.rand(1, generator=expected_torch).item()

        resumed.reseed_stage(stage_seed)
        resumed.run_generation(
            TicTacToe,
            self_play_episodes=1,
            training_steps=0,
            processes=1,
        )
        events = list(ledger.read())
        episode = next(
            event for event in events if event.event_type == "alphazero_self_play_episode"
        )
        actions = tuple(
            event.payload["action"]
            for event in events
            if event.event_type == "alphazero_self_play_step"
        )
        return episode.payload["seed"], actions

    first = resumed_trace(101, "first")
    repeated = resumed_trace(101, "repeated")
    different = resumed_trace(202, "different")

    assert first == repeated
    assert first[0] == 666291129
    assert different[0] == 399595374
    assert different[1] != first[1]


def _config(*, simulations: int = 12, learning_rate: float = 0.03) -> AlphaZeroConfig:
    return AlphaZeroConfig(
        simulations=simulations,
        c_puct=1.5,
        root_dirichlet_fraction=0.0,
        self_play_temperature=0.0,
        temperature_moves=0,
        channels=8,
        residual_blocks=1,
        learning_rate=learning_rate,
        weight_decay=0.0,
        batch_size=8,
        replay_capacity=64,
        min_replay_size=1,
        mixed_precision=False,
        inference_batch_size=4,
    )


def _trainer(
    game: TicTacToe,
    *,
    learning_rate: float,
    ledger: EventLedger | None = None,
) -> AlphaZeroTrainer:
    config = _config(learning_rate=learning_rate)
    network = PolicyValueNet.from_game_spec(game.spec, config)
    return AlphaZeroTrainer(network, config, seed=23, ledger=ledger, run_id="test-run")


def _training_position() -> TicTacToe:
    game = TicTacToe()
    game.set_position("OOX..X...", player=0)
    return game


def _winning_sample(
    game: TicTacToe,
    *,
    source: str = "selfplay",
    sample_weight: float = 1.0,
) -> ReplaySample:
    policy = np.zeros(9, dtype=np.float32)
    policy[8] = 1.0
    return ReplaySample(
        observation=game.observe(game.current_player()),
        legal_mask=game.legal_action_mask(),
        visit_policy=policy,
        outcome=1.0,
        player=game.current_player(),
        source=source,
        sample_weight=sample_weight,
    )


def _losing_sample(game: TicTacToe) -> ReplaySample:
    policy = np.zeros(9, dtype=np.float32)
    policy[8] = 1.0
    return ReplaySample(
        observation=game.observe(game.current_player()),
        legal_mask=game.legal_action_mask(),
        visit_policy=policy,
        outcome=-1.0,
        player=game.current_player(),
    )


def _network_inputs(game: TicTacToe) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    observation = game.observe(game.current_player())
    return (
        torch.as_tensor(observation.planes[None]),
        torch.as_tensor(observation.scalars[None]),
        torch.as_tensor(game.legal_action_mask()[None]),
    )


def _score_against_random(network: PolicyValueNet, seeds: range) -> float:
    outcomes: list[float] = []
    for seed in seeds:
        game = _training_position()
        rng = np.random.default_rng(seed)
        while game.outcome(0) is None:
            legal = game.legal_action_mask()
            if game.current_player() == 0:
                logits, _ = network.evaluate_batch(
                    [game.observe(0)], [legal]
                )
                action = int(np.argmax(logits[0]))
            else:
                action = int(rng.choice(np.flatnonzero(legal)))
            game.step(action)
        outcome = game.outcome(0)
        assert outcome is not None
        outcomes.append(outcome)
    return float(np.mean(outcomes))


def _trainer_state_snapshot(
    trainer: AlphaZeroTrainer, game: TicTacToe
) -> dict[str, object]:
    planes, scalars, mask = _network_inputs(game)
    was_training = trainer.network.training
    trainer.network.eval()
    try:
        with torch.inference_mode():
            logits = trainer.network(planes, scalars, mask)[0].clone()
    finally:
        trainer.network.train(was_training)
    return {
        "logits": logits,
        "model": {
            name: tensor.detach().clone()
            for name, tensor in trainer.network.state_dict().items()
        },
        "optimizer": copy.deepcopy(trainer.optimizer.state_dict()),
        "replay": copy.deepcopy(trainer.replay.state_dict()),
        "scaler": copy.deepcopy(trainer.scaler.state_dict()),
        "budgets": copy.deepcopy(trainer.budgets.as_dict()),
        "generation": trainer.generation,
        "optimizer_steps": trainer.optimizer_steps,
        "trainer_rng": copy.deepcopy(trainer.rng.bit_generator.state),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state().clone(),
    }


def _assert_trainer_snapshot(
    trainer: AlphaZeroTrainer,
    game: TicTacToe,
    expected: dict[str, object],
) -> None:
    actual = _trainer_state_snapshot(trainer, game)
    _assert_nested_equal(actual, expected)


def _assert_nested_equal(actual, expected) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert torch.equal(actual, expected)
    elif isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        assert np.array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_equal(actual_item, expected_item)
    else:
        assert actual == expected
