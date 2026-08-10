from __future__ import annotations

import copy
from dataclasses import replace
from importlib.metadata import version

import numpy as np
import pytest
import torch
from tianshou.algorithm import PPO

from rlbench.algorithms import PolicyCheckpoint
from rlbench.algorithms.ppo_tianshou import GymGameEnv, PPOConfig, PPOTrainer
from rlbench.game import (
    BoardObservationSpec,
    DiscreteGameSpec,
    Observation,
    StepRecord,
)
from rlbench.telemetry import EventLedger


class CounterGame:
    """Player zero chooses win or loss; player one then performs a forced hold."""

    spec = DiscreteGameSpec(
        name="ppo-counter",
        players=2,
        zero_sum=True,
        action_names=("increment", "decrement"),
        observation_spec=BoardObservationSpec(
            plane_names=("constant",),
            board_shape=(1, 1),
            scalar_names=("viewer",),
        ),
        max_episode_steps=2,
    )

    def __init__(self) -> None:
        self.turn = 0
        self.result = 0.0

    def reset(self, seed: int) -> None:
        self.turn = 0
        self.result = 0.0

    def current_player(self) -> int:
        return self.turn % 2

    def observe(self, player: int) -> Observation:
        return Observation(
            planes=np.zeros((1, 1, 1), dtype=np.float32),
            scalars=np.array([player], dtype=np.float32),
        )

    def legal_action_mask(self) -> np.ndarray:
        if self.current_player() == 0:
            return np.array([True, True], dtype=np.bool_)
        return np.array([False, True], dtype=np.bool_)

    def step(self, action: int) -> StepRecord:
        actor = self.current_player()
        if not self.legal_action_mask()[action]:
            raise ValueError("illegal action")
        if actor == 0:
            self.result = 1.0 if action == 0 else -1.0
        self.turn += 1
        return StepRecord(player=actor, action=action, terminated=self.turn == 2)

    def outcome(self, player: int) -> float | None:
        if self.turn != 2:
            return None
        return self.result if player == 0 else -self.result


class SequenceGame:
    """Two controlled decisions with identical observations and intervening turns."""

    spec = replace(CounterGame.spec, name="ppo-sequence", max_episode_steps=4)

    def __init__(self) -> None:
        self.turn = 0
        self.result = 0.0

    def reset(self, seed: int) -> None:
        self.turn = 0
        self.result = 0.0

    def current_player(self) -> int:
        return self.turn % 2

    def observe(self, player: int) -> Observation:
        return Observation(
            planes=np.zeros((1, 1, 1), dtype=np.float32),
            scalars=np.array([player], dtype=np.float32),
        )

    def legal_action_mask(self) -> np.ndarray:
        if self.current_player() == 0:
            return np.array([True, True], dtype=np.bool_)
        return np.array([False, True], dtype=np.bool_)

    def step(self, action: int) -> StepRecord:
        actor = self.current_player()
        if not self.legal_action_mask()[action]:
            raise ValueError("illegal action")
        if actor == 0 and self.turn == 2:
            self.result = 1.0 if action == 0 else -1.0
        self.turn += 1
        return StepRecord(player=actor, action=action, terminated=self.turn == 4)

    def outcome(self, player: int) -> float | None:
        if self.turn != 4:
            return None
        return self.result if player == 0 else -self.result


class ShapingLossGame(CounterGame):
    """Positive official-score potential with a terminal game loss."""

    spec = replace(CounterGame.spec, name="ppo-shaping-loss")

    def score(self, player: int) -> float:
        if self.turn < 2:
            return 0.0
        return 1.0 if player == 0 else 0.0

    def outcome(self, player: int) -> float | None:
        if self.turn != 2:
            return None
        return -1.0 if player == 0 else 1.0


def small_config(**overrides: object) -> PPOConfig:
    values: dict[str, object] = {
        "learning_rate": 0.01,
        "hidden_size": 16,
        "conv_channels": 8,
        "vector_envs": 2,
        "episodes_per_collect": 16,
        "minibatch_size": 16,
        "update_repetitions": 3,
        "entropy_coefficient": 0.0,
        "snapshot_interval": 1,
        "max_snapshots": 3,
    }
    values.update(overrides)
    return PPOConfig(**values)


def observation_and_mask() -> tuple[Observation, np.ndarray]:
    game = CounterGame()
    game.reset(seed=0)
    return game.observe(0), game.legal_action_mask()


def test_constructs_the_pinned_tianshou_two_ppo_algorithm() -> None:
    """Falling back to a local PPO loop would bypass the pinned backend contract."""
    trainer = PPOTrainer(CounterGame, small_config(), seed=3)

    assert version("tianshou") == "2.0.1"
    assert trainer.tianshou_version == "2.0.1"
    assert isinstance(trainer.algorithm, PPO)
    assert trainer.algorithm.policy is trainer.policy
    assert trainer.algorithm.critic is trainer.network.critic


def test_action_mapper_runs_through_vector_collection_and_snapshot_self_play() -> None:
    calls = 0

    def mapper(game: CounterGame, player: int) -> tuple[int, ...]:
        nonlocal calls
        del player
        calls += 1
        return tuple(
            reversed(
                [int(action) for action in np.flatnonzero(game.legal_action_mask())]
            )
        )

    trainer = PPOTrainer(
        CounterGame,
        small_config(vector_envs=2, episodes_per_collect=4),
        seed=11,
        action_mapper=mapper,
        action_mapper_id="reverse-legal-v1",
    )

    metrics = trainer.train_iteration()

    assert metrics.episodes == 4
    assert calls > 0


def test_stateful_process_opponent_requires_single_vector_environment() -> None:
    """One process cannot safely carry independent state for concurrent games."""

    class ProcessOpponent:
        def act_game_process(self, game, *, timeout_seconds=None) -> int:
            return 0

    with pytest.raises(ValueError, match="vector_envs=1"):
        PPOTrainer(
            CounterGame,
            small_config(vector_envs=2),
            opponent=ProcessOpponent(),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("probability", "expects_external_calls"),
    ((0.0, False), (1.0, True)),
)
def test_external_opponent_probability_mixes_human_and_snapshot_training(
    probability: float, expects_external_calls: bool
) -> None:
    class CountingOpponent:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, observation, legal_mask) -> int:
            del observation
            self.calls += 1
            return int(np.flatnonzero(legal_mask)[0])

    opponent = CountingOpponent()
    trainer = PPOTrainer(
        CounterGame,
        small_config(
            vector_envs=1,
            episodes_per_collect=4,
            external_opponent_probability=probability,
        ),
        seed=17,
        opponent=opponent,
        opponent_id="teacher",
    )

    trainer.train_iteration()

    assert (opponent.calls > 0) is expects_external_calls


@pytest.mark.parametrize("probability", (-0.01, 1.01, float("nan")))
def test_external_opponent_probability_requires_a_unit_interval(
    probability: float,
) -> None:
    with pytest.raises(ValueError, match="external_opponent_probability"):
        PPOConfig(external_opponent_probability=probability)


@pytest.mark.parametrize("player", (-1, 2, True))
def test_training_player_requires_one_role_or_null(player: object) -> None:
    with pytest.raises(ValueError, match="training_player"):
        PPOConfig(training_player=player)  # type: ignore[arg-type]


def test_training_player_focuses_collection_on_one_role(tmp_path) -> None:
    ledger = EventLedger(tmp_path / "events.jsonl")
    trainer = PPOTrainer(
        CounterGame,
        small_config(
            vector_envs=1,
            episodes_per_collect=4,
            training_player=1,
        ),
        seed=18,
        ledger=ledger,
        run_id="ppo-role-curriculum",
    )

    trainer.train_iteration()

    transitions = [
        event
        for event in ledger.read()
        if event.event_type == "ppo_transition"
    ]
    assert len(transitions) == 4
    assert {event.payload["controlled_player"] for event in transitions} == {1}


def test_synchronous_collection_updates_ppo_and_emits_events(tmp_path) -> None:
    """A trainer that only constructs PPO but never collects or updates cannot learn."""
    ledger = EventLedger(tmp_path / "events.jsonl")
    trainer = PPOTrainer(
        CounterGame,
        small_config(),
        seed=4,
        ledger=ledger,
        run_id="ppo-test",
    )

    metrics = trainer.train_iteration()

    assert metrics.iteration == 1
    assert metrics.episodes == 16
    assert metrics.env_steps == 16
    assert metrics.optimizer_steps > 0
    assert np.isfinite(metrics.loss)
    assert trainer.budgets.learning.episodes == 16
    assert trainer.budgets.learning.env_steps == 16
    assert trainer.budgets.learning.optimizer_steps == metrics.optimizer_steps
    assert len(trainer.opponent_snapshots) == 2
    events = list(ledger.read())
    transitions = [event for event in events if event.event_type == "ppo_transition"]
    assert len(transitions) == 16
    assert len({event.payload["env_step_id"] for event in transitions}) == 16
    assert len({event.payload["episode_id"] for event in transitions}) == 16
    assert all(
        set(event.payload)
        >= {
            "episode_id",
            "env_step_id",
            "episode_step",
            "terminal_outcome",
            "terminal_reward",
            "shaping_reward",
            "combined_reward",
        }
        for event in transitions
    )
    assert all(
        event.event_id is not None
        and event.created_at is not None
        and event.schema_version is not None
        for event in transitions
    )
    assert all(event.payload["terminal_outcome"] in (-1.0, 1.0) for event in transitions)
    assert all(
        event.payload["terminal_reward"] == event.payload["combined_reward"]
        for event in transitions
    )
    event_types = [event.event_type for event in events[-4:]]
    assert event_types == [
        "ppo_collection_completed",
        "ppo_optimizer_step",
        "budget_snapshot",
        "ppo_snapshot_created",
    ]


def test_action_distribution_is_normalized_masked_and_deterministic() -> None:
    """Exporting unmasked scores invalidates both play and information-gain metrics."""
    trainer = PPOTrainer(CounterGame, small_config(), seed=5)
    observation, _ = observation_and_mask()
    mask = np.array([False, True], dtype=np.bool_)

    first = trainer.action_distribution(observation, mask)
    second = trainer.action_distribution(observation, mask)

    assert first.tolist() == pytest.approx([0.0, 1.0])
    assert second.tolist() == pytest.approx(first.tolist())
    assert trainer.select_action(observation, mask, deterministic=True) == 1

    stateful_distribution = trainer.action_distribution_step(observation, mask)
    stateful_decision = trainer.select_action_step(
        observation, mask, deterministic=True
    )

    assert stateful_distribution.probabilities.tolist() == pytest.approx(
        first.tolist()
    )
    assert stateful_distribution.state is None
    assert stateful_decision.action == 1
    assert stateful_decision.probabilities.tolist() == pytest.approx(first.tolist())
    assert stateful_decision.state is None


def test_deployment_logit_bias_changes_distribution_and_action() -> None:
    trainer = PPOTrainer(CounterGame, small_config(), seed=6)
    observation, mask = observation_and_mask()
    baseline = trainer.action_distribution(observation, mask)
    baseline_action = int(np.argmax(baseline))
    target_action = 1 - baseline_action
    bias = np.zeros_like(baseline)
    bias[target_action] = (
        np.log(baseline[baseline_action]) - np.log(baseline[target_action]) + 1.0
    )

    biased = trainer.action_distribution(
        observation, mask, logit_bias=bias
    )
    decision = trainer.select_action_step(
        observation,
        mask,
        deterministic=True,
        logit_bias=bias,
    )

    assert biased.sum() == pytest.approx(1.0)
    assert int(np.argmax(biased)) == target_action
    assert decision.action == target_action
    assert decision.probabilities.tolist() == pytest.approx(biased.tolist())


@pytest.mark.parametrize(
    "bias",
    (np.zeros(1, dtype=np.float32), np.array([0.0, np.nan], dtype=np.float32)),
)
def test_deployment_logit_bias_is_validated(bias: np.ndarray) -> None:
    trainer = PPOTrainer(CounterGame, small_config(), seed=7)
    observation, mask = observation_and_mask()

    with pytest.raises(ValueError, match="logit bias"):
        trainer.action_distribution(observation, mask, logit_bias=bias)


def test_checkpoint_restore_recovers_model_optimizer_counters_rng_and_snapshots(
    tmp_path,
) -> None:
    """Restoring weights alone changes the next PPO update and self-play population."""
    config = small_config(episodes_per_collect=8, minibatch_size=8)
    original = PPOTrainer(CounterGame, config, seed=9)
    original.train_iteration()
    path = tmp_path / "ppo.pt"

    saved = original.save_checkpoint(path)
    expected_metrics = original.train_iteration()
    expected_parameters = {
        name: parameter.detach().clone()
        for name, parameter in original.network.state_dict().items()
    }

    restored = PPOTrainer(CounterGame, config, seed=999)
    loaded = restored.load_checkpoint(path)
    actual_metrics = restored.train_iteration()

    assert isinstance(saved, PolicyCheckpoint)
    assert isinstance(loaded, PolicyCheckpoint)
    assert actual_metrics == expected_metrics
    assert restored.iteration == original.iteration
    assert restored.optimizer_steps == original.optimizer_steps
    assert restored.budgets.as_dict() == original.budgets.as_dict()
    assert len(restored.opponent_snapshots) == len(original.opponent_snapshots)
    for name, parameter in restored.network.state_dict().items():
        torch.testing.assert_close(parameter, expected_parameters[name], rtol=0, atol=0)


def test_checkpoint_accepts_the_legacy_zero_residual_config(tmp_path) -> None:
    config = small_config(residual_blocks=0)
    source = PPOTrainer(CounterGame, config, seed=19)
    path = tmp_path / "legacy-zero-residual.pt"
    source.save_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    del payload["trainer_state"]["ppo_config"]["residual_blocks"]
    torch.save(payload, path)

    restored = PPOTrainer(CounterGame, config, seed=20)
    restored.load_checkpoint(path)

    for name, parameter in restored.network.state_dict().items():
        torch.testing.assert_close(
            parameter,
            source.network.state_dict()[name],
            rtol=0,
            atol=0,
        )


def test_model_initialization_loads_weights_into_a_fresh_ppo_run(tmp_path) -> None:
    source = PPOTrainer(CounterGame, small_config(), seed=21)
    source.train_iteration()
    path = tmp_path / "warm-start.pt"
    source.save_checkpoint(path)

    ledger = EventLedger(tmp_path / "events.jsonl")
    target = PPOTrainer(
        CounterGame,
        small_config(learning_rate=1e-4, entropy_coefficient=0.0),
        seed=22,
        ledger=ledger,
        run_id="warm-start",
    )
    target.initialize_model(path)

    assert target.iteration == 0
    assert target.optimizer_steps == 0
    assert target.budgets.as_dict()["total"]["optimizer_steps"] == 0
    assert not target.optimizer.state
    for name, parameter in target.network.state_dict().items():
        torch.testing.assert_close(
            parameter, source.network.state_dict()[name], rtol=0, atol=0
        )
        torch.testing.assert_close(
            target.opponent_snapshots[0].network.state_dict()[name],
            parameter,
            rtol=0,
            atol=0,
        )
    assert [event.event_type for event in ledger.read()] == [
        "ppo_model_initialized"
    ]


@pytest.mark.parametrize(
    "mismatched_config",
    [
        small_config(gamma=0.8),
        small_config(shaping_beta=0.25),
    ],
)
def test_checkpoint_rejects_config_mismatch_without_mutating_live_state(
    tmp_path, mismatched_config: PPOConfig
) -> None:
    source = PPOTrainer(CounterGame, small_config(), seed=31)
    source.train_iteration()
    path = tmp_path / "config-mismatch.pt"
    source.save_checkpoint(path)
    target = PPOTrainer(CounterGame, mismatched_config, seed=32)
    before = copy.deepcopy(target.network.state_dict())
    counters_before = target.budgets.as_dict()

    with pytest.raises(ValueError, match="PPOConfig"):
        target.load_checkpoint(path)

    assert target.iteration == 0
    assert target.optimizer_steps == 0
    assert target.budgets.as_dict() == counters_before
    for name, parameter in target.network.state_dict().items():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)


def test_checkpoint_rejects_game_fingerprint_mismatch_without_mutation(tmp_path) -> None:
    class RenamedCounterGame(CounterGame):
        spec = replace(CounterGame.spec, name="renamed-counter")

    source = PPOTrainer(CounterGame, small_config(), seed=33)
    path = tmp_path / "game-mismatch.pt"
    source.save_checkpoint(path)
    target = PPOTrainer(RenamedCounterGame, small_config(), seed=34)
    before = copy.deepcopy(target.network.state_dict())

    with pytest.raises(ValueError, match="game specification"):
        target.load_checkpoint(path)

    assert target.iteration == 0
    for name, parameter in target.network.state_dict().items():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)


def test_checkpoint_binds_residual_action_mapper_identity(tmp_path) -> None:
    def mapper(game: CounterGame, player: int) -> tuple[int, ...]:
        del player
        return tuple(int(action) for action in np.flatnonzero(game.legal_action_mask()))

    source = PPOTrainer(
        CounterGame,
        small_config(),
        seed=35,
        action_mapper=mapper,
        action_mapper_id="legal-prior-v1",
    )
    path = tmp_path / "mapped.pt"
    source.save_checkpoint(path)

    matching = PPOTrainer(
        CounterGame,
        small_config(),
        seed=36,
        action_mapper=mapper,
        action_mapper_id="legal-prior-v1",
    )
    matching.load_checkpoint(path)

    mismatched = PPOTrainer(CounterGame, small_config(), seed=37)
    with pytest.raises(ValueError, match="action mapper"):
        mismatched.load_checkpoint(path)


def test_evaluation_is_deterministic_and_does_not_charge_learning_budget() -> None:
    """Sampling during evaluation makes checkpoint comparisons irreproducible."""
    trainer = PPOTrainer(CounterGame, small_config(), seed=6)
    learning_before = trainer.budgets.learning.env_steps

    first = trainer.evaluate(episodes=6, seed=100)
    second = trainer.evaluate(episodes=6, seed=100)

    assert first == second
    assert trainer.budgets.learning.env_steps == learning_before
    assert trainer.budgets.evaluation.episodes == 12
    assert trainer.budgets.evaluation.env_steps == 12


def test_evaluation_classifies_terminal_outcome_not_positive_shaped_return() -> None:
    config = small_config(shaping_beta=3.0, score_scale=1.0)
    env = GymGameEnv(
        ShapingLossGame,
        controlled_player=0,
        opponent=lambda observation, mask: 1,
        shaping_beta=config.shaping_beta,
        gamma=1.0,
        score_scale=config.score_scale,
    )
    env.reset(seed=0)
    _, reward, terminated, _, info = env.step(0)
    assert terminated
    assert info["terminal_outcome"] == -1.0
    assert info["terminal_reward"] == -1.0
    assert info["shaping_reward"] == 3.0
    assert info["combined_reward"] == 2.0
    assert reward == 2.0

    trainer = PPOTrainer(ShapingLossGame, config, seed=35)
    result = trainer.evaluate(episodes=1, seed=0)

    assert result.mean_return == pytest.approx(1.97)
    assert (result.wins, result.draws, result.losses) == (0, 0, 1)


def test_recurrent_history_changes_distribution_and_critic_is_feed_forward() -> None:
    config = small_config(recurrent=True, gru_hidden_size=8)
    trainer = PPOTrainer(SequenceGame, config, seed=41)
    observation, mask = observation_and_mask()
    encoded = {
        "obs": np.concatenate((observation.planes.reshape(-1), observation.scalars))[None],
        "mask": mask[None],
    }

    first_logits, first_state = trainer.network.actor(encoded)
    second_logits, _ = trainer.network.actor(encoded, state=first_state)
    reset_logits, _ = trainer.network.actor(encoded)

    assert first_state is not None
    assert not torch.allclose(first_logits, second_logits)
    torch.testing.assert_close(first_logits, reset_logits, rtol=0, atol=0)
    assert trainer.network.actor.gru is not None
    assert trainer.network.critic.gru is None


def test_public_recurrent_deployment_state_propagates_and_resets() -> None:
    trainer = PPOTrainer(
        SequenceGame,
        small_config(recurrent=True, gru_hidden_size=8),
        seed=44,
    )
    observation, mask = observation_and_mask()

    first = trainer.action_distribution_step(observation, mask)
    continued = trainer.action_distribution_step(
        observation,
        mask,
        state=first.state,
    )
    reset = trainer.action_distribution_step(observation, mask, state=None)

    assert first.state is not None
    assert continued.state is not None
    assert not np.allclose(first.probabilities, continued.probabilities)
    np.testing.assert_array_equal(first.probabilities, reset.probabilities)
    assert not first.probabilities.flags.writeable
    assert not first.state.hidden.flags.writeable

    forced = trainer.select_action_step(
        observation,
        np.array([False, True], dtype=np.bool_),
        deterministic=True,
        state=first.state,
    )

    assert forced.action == 1
    np.testing.assert_array_equal(
        forced.probabilities,
        np.array([0.0, 1.0], dtype=np.float32),
    )


def test_recurrent_ppo_update_uses_nonzero_behavior_time_hidden_state() -> None:
    config = small_config(
        recurrent=True,
        gru_hidden_size=8,
        vector_envs=2,
        episodes_per_collect=8,
        minibatch_size=16,
        update_repetitions=2,
    )
    trainer = PPOTrainer(SequenceGame, config, seed=42)
    assert trainer.network.actor.gru is not None
    before = trainer.network.actor.gru.weight_hh_l0.detach().clone()

    metrics = trainer.train_iteration()

    assert metrics.env_steps == 16
    assert not torch.equal(trainer.network.actor.gru.weight_hh_l0, before)


def _state_input_norm(state: object) -> float | None:
    if state is None:
        return None
    return float(torch.linalg.vector_norm(state["hidden"]).item())  # type: ignore[index]


def test_evaluation_and_snapshot_opponents_maintain_and_reset_recurrent_state(
    monkeypatch,
) -> None:
    config = small_config(recurrent=True, gru_hidden_size=8)
    trainer = PPOTrainer(SequenceGame, config, seed=43)
    learner_states: list[float | None] = []
    opponent_states: list[float | None] = []
    learner_forward = trainer.network.actor.forward
    snapshot_forward = trainer.opponent_snapshots[-1].network.actor.forward

    def record_learner(obs, state=None, info=None):
        learner_states.append(_state_input_norm(state))
        return learner_forward(obs, state=state, info=info)

    def record_snapshot(obs, state=None, info=None):
        opponent_states.append(_state_input_norm(state))
        return snapshot_forward(obs, state=state, info=info)

    monkeypatch.setattr(trainer.network.actor, "forward", record_learner)
    monkeypatch.setattr(
        trainer.opponent_snapshots[-1].network.actor, "forward", record_snapshot
    )

    trainer.evaluate(episodes=2, seed=0)

    assert learner_states[0] is None
    assert learner_states[1] is not None and learner_states[1] > 0
    assert learner_states[2] is None
    assert learner_states[3] is not None and learner_states[3] > 0
    assert opponent_states[0] is None
    assert opponent_states[1] is not None and opponent_states[1] > 0
    assert opponent_states[2] is None
    assert opponent_states[3] is not None and opponent_states[3] > 0


def test_ppo_improves_the_winning_action_probability_on_counter_game() -> None:
    """Collection and updates that do not improve a solved toy game are miswired."""
    trainer = PPOTrainer(CounterGame, small_config(), seed=11)
    observation, mask = observation_and_mask()
    before = float(trainer.action_distribution(observation, mask)[0])

    trainer.train(iterations=12)

    after = float(trainer.action_distribution(observation, mask)[0])
    assert after > before + 0.2
    assert after > 0.8
