"""Synchronous self-play lifecycle built on the Tianshou 2.0.1 PPO API."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gymnasium.spaces import Discrete
from numpy.typing import NDArray
from tianshou.algorithm import PPO
from tianshou.algorithm.modelfree.reinforce import DiscreteActorPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Batch, Collector, VectorReplayBuffer
from tianshou.data.types import DistBatchProtocol, ObsBatchProtocol
from tianshou.env import DummyVectorEnv
from tianshou.utils.torch_utils import policy_within_training_step

from rlbench.algorithms.checkpoint import PolicyCheckpoint
from rlbench.game import DiscreteGame, DiscreteGameSpec, Observation
from rlbench.league import LeagueState
from rlbench.telemetry import BudgetCounters, Event, EventLedger

from .config import PPOConfig
from .env import ActionMapper, GymGameEnv, OpponentPolicy
from .network import MaskedActorCritic


@dataclass(frozen=True, slots=True)
class PPOTrainingMetrics:
    iteration: int
    episodes: int
    env_steps: int
    optimizer_steps: int
    mean_return: float
    loss: float


@dataclass(frozen=True, slots=True)
class PPOEvaluationMetrics:
    episodes: int
    env_steps: int
    mean_return: float
    wins: int
    draws: int
    losses: int


def _immutable_probabilities(
    probabilities: NDArray[np.float32],
) -> NDArray[np.float32]:
    result = np.array(probabilities, dtype=np.float32, copy=True)
    if result.ndim != 1 or result.size == 0:
        raise ValueError("action probabilities must be a non-empty vector")
    if not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ValueError("action probabilities must be finite and non-negative")
    if not np.isclose(float(result.sum()), 1.0):
        raise ValueError("action probabilities must sum to one")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class PPORecurrentState:
    """Opaque single-environment actor state carried between deployment calls."""

    hidden: NDArray[np.float32]

    def __post_init__(self) -> None:
        hidden = np.array(self.hidden, dtype=np.float32, copy=True)
        if hidden.ndim != 3 or hidden.shape[0] != 1:
            raise ValueError(
                "recurrent hidden state must have shape "
                "(1, recurrent_layers, hidden_size)"
            )
        if not np.all(np.isfinite(hidden)):
            raise ValueError("recurrent hidden state must be finite")
        hidden.setflags(write=False)
        object.__setattr__(self, "hidden", hidden)


@dataclass(frozen=True, slots=True)
class PPOActionDistribution:
    """Masked action probabilities and the recurrent state for the next call."""

    probabilities: NDArray[np.float32]
    state: PPORecurrentState | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "probabilities",
            _immutable_probabilities(self.probabilities),
        )


@dataclass(frozen=True, slots=True)
class PPOActionDecision:
    """Selected action, its distribution, and the recurrent state to carry."""

    action: int
    probabilities: NDArray[np.float32]
    state: PPORecurrentState | None

    def __post_init__(self) -> None:
        probabilities = _immutable_probabilities(self.probabilities)
        if isinstance(self.action, bool) or not 0 <= self.action < len(probabilities):
            raise ValueError("action must index the probability vector")
        object.__setattr__(self, "probabilities", probabilities)


@dataclass(slots=True)
class OpponentSnapshot:
    generation: int
    network: MaskedActorCritic


LeagueEvaluation = Callable[
    [MaskedActorCritic, LeagueState, int], Mapping[str, Any]
]
EventCallback = Callable[[Event], None]


class RecurrentDiscreteActorPolicy(DiscreteActorPolicy):
    """Recover behavior-time GRU input state from Tianshou rollout batches."""

    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | Batch | np.ndarray | None = None,
    ) -> DistBatchProtocol:
        if state is None:
            policy_entry = batch.get("policy")
            if isinstance(policy_entry, Batch):
                stored_state = policy_entry.get("hidden_state")
                if isinstance(stored_state, Batch):
                    pre_hidden = stored_state.get("pre_hidden")
                    if pre_hidden is not None:
                        if pre_hidden.shape[0] != len(batch.obs):
                            raise ValueError(
                                "stored recurrent state is not batch-aligned"
                            )
                        state = Batch(hidden=pre_hidden)
        return super().forward(batch, state=state)


class _StatefulOpponent:
    def __init__(
        self,
        trainer: PPOTrainer,
        snapshot: OpponentSnapshot,
        *,
        deterministic: bool,
    ) -> None:
        self.trainer = trainer
        self.snapshot = snapshot
        self.deterministic = deterministic
        self.state: Batch | None = None

    def reset(self) -> None:
        self.state = None

    def __call__(self, observation: Observation, mask: NDArray[np.bool_]) -> int:
        probabilities, self.state = self.trainer._network_distribution_step(
            self.snapshot.network,
            observation,
            mask,
            state=self.state,
        )
        if self.deterministic:
            return int(np.argmax(probabilities))
        return int(self.trainer.rng.choice(len(probabilities), p=probabilities))


class PPOTrainer:
    """Own the pinned PPO algorithm, vector collection, snapshots, and resume state."""

    def __init__(
        self,
        game_factory: Callable[[], DiscreteGame],
        config: PPOConfig,
        *,
        network: MaskedActorCritic | None = None,
        seed: int = 0,
        ledger: EventLedger | None = None,
        run_id: str = "ppo",
        league: LeagueState | None = None,
        evaluation_callback: LeagueEvaluation | None = None,
        event_callback: EventCallback | None = None,
        opponent: OpponentPolicy | None = None,
        opponent_id: str = "opponent",
        action_mapper: ActionMapper | None = None,
        action_mapper_id: str | None = None,
    ) -> None:
        self.tianshou_version = version("tianshou")
        if self.tianshou_version != "2.0.1":
            raise RuntimeError("PPO backend requires tianshou==2.0.1")
        if not run_id:
            raise ValueError("run_id must be non-empty")
        if not opponent_id:
            raise ValueError("opponent_id must be non-empty")
        if (action_mapper is None) != (action_mapper_id is None):
            raise ValueError("action_mapper and action_mapper_id must be set together")
        if action_mapper_id is not None and not action_mapper_id:
            raise ValueError("action_mapper_id must be non-empty")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.game_factory = game_factory
        self.config = config
        sample_game = game_factory()
        self.game_spec = sample_game.spec
        self.network = network or MaskedActorCritic.from_game_spec(
            self.game_spec, config
        )
        if self.network.action_count != len(self.game_spec.action_names):
            raise ValueError("network action count does not match the game")

        probe_env = GymGameEnv(game_factory, action_mapper=action_mapper)
        self.policy = RecurrentDiscreteActorPolicy(
            actor=self.network.actor,
            action_space=Discrete(len(self.game_spec.action_names)),
            observation_space=probe_env.observation_space,
            deterministic_eval=True,
        )
        self.algorithm = PPO(
            policy=self.policy,
            critic=self.network.critic,
            optim=AdamOptimizerFactory(
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            ),
            eps_clip=config.clip_epsilon,
            advantage_normalization=True,
            vf_coef=config.value_coefficient,
            ent_coef=config.entropy_coefficient,
            max_grad_norm=config.max_grad_norm,
            gae_lambda=config.gae_lambda,
            max_batchsize=config.minibatch_size,
            gamma=config.gamma,
        )
        # Tianshou 2 owns the torch optimizer behind its gradient-clipping wrapper.
        self.optimizer: torch.optim.Optimizer = self.algorithm.optim._optim
        self.iteration = 0
        self.optimizer_steps = 0
        self.budgets = BudgetCounters()
        self.ledger = ledger
        self.run_id = run_id
        self.league = league
        self.evaluation_callback = evaluation_callback
        self.event_callback = event_callback
        self.external_opponent = opponent
        self.opponent_id = opponent_id
        self.action_mapper = action_mapper
        self.action_mapper_id = action_mapper_id
        if (
            opponent is not None
            and callable(getattr(opponent, "act_game_process", None))
            and config.vector_envs != 1
        ):
            raise ValueError(
                "a stateful game-process opponent requires vector_envs=1"
            )
        self.opponent_snapshots: list[OpponentSnapshot] = []
        self.last_vector_env_class: type[DummyVectorEnv] | None = None
        self._telemetry_env_step_id = 0
        self._transition_episode_ids: dict[tuple[int, int], str] = {}
        self._create_snapshot(emit=False)

    def initialize_model(self, path: str | Path) -> PolicyCheckpoint:
        """Warm-start policy/value weights while keeping a fresh PPO optimizer."""
        checkpoint = PolicyCheckpoint.load(path, map_location=self.network.device)
        state = checkpoint.trainer_state
        fingerprint = state.get("game_spec_fingerprint")
        if fingerprint is not None and fingerprint != _game_spec_fingerprint(
            self.game_spec
        ):
            raise ValueError("initial checkpoint game specification does not match")
        if state.get("action_mapper_id") != self.action_mapper_id:
            raise ValueError("initial checkpoint action mapper does not match")
        checkpoint.validate_restore(model=self.network)
        self.network.load_state_dict(checkpoint.model_state, strict=True)
        self.opponent_snapshots[0].network.load_state_dict(
            self.network.state_dict(), strict=True
        )
        self._emit(
            Event(
                event_type="ppo_model_initialized",
                run_id=self.run_id,
                stage="learning",
                payload={"schema_version": checkpoint.schema_version},
            )
        )
        return checkpoint

    def train_iteration(self) -> PPOTrainingMetrics:
        """Collect complete episodes synchronously and perform one PPO update."""
        self._transition_episode_ids = {}
        vector_env = self._make_vector_env()
        self.last_vector_env_class = type(vector_env)
        total_size = max(
            self.config.vector_envs * 2,
            self.config.episodes_per_collect * self.game_spec.max_episode_steps,
        )
        buffer = VectorReplayBuffer(total_size, self.config.vector_envs)
        collector = Collector(self.algorithm, vector_env, buffer=buffer)
        try:
            with policy_within_training_step(self.policy):
                collect_stats = collector.collect(
                    n_episode=self.config.episodes_per_collect,
                    reset_before_collect=True,
                )
                training_stats = self.algorithm.update(
                    buffer,
                    batch_size=self.config.minibatch_size,
                    repeat=self.config.update_repetitions,
                )
        finally:
            collector.close()

        gradient_steps = int(training_stats.gradient_steps)
        self.iteration += 1
        self.optimizer_steps += gradient_steps
        self.budgets.learning.episodes += int(collect_stats.n_collected_episodes)
        self.budgets.learning.env_steps += int(collect_stats.n_collected_steps)
        for _ in range(gradient_steps):
            self.budgets.record_optimizer_step("learning")
        mean_return = (
            float(np.mean(collect_stats.returns))
            if collect_stats.returns.size
            else 0.0
        )
        loss = float(training_stats.loss.mean)
        metrics = PPOTrainingMetrics(
            iteration=self.iteration,
            episodes=int(collect_stats.n_collected_episodes),
            env_steps=int(collect_stats.n_collected_steps),
            optimizer_steps=self.optimizer_steps,
            mean_return=mean_return,
            loss=loss,
        )
        self._emit(
            Event(
                event_type="ppo_collection_completed",
                run_id=self.run_id,
                stage="learning",
                payload={
                    "iteration": self.iteration,
                    "episodes": metrics.episodes,
                    "env_steps": metrics.env_steps,
                    "mean_return": metrics.mean_return,
                },
            )
        )
        self._emit(
            Event(
                event_type="ppo_optimizer_step",
                run_id=self.run_id,
                stage="learning",
                payload={
                    "iteration": self.iteration,
                    "optimizer_steps": self.optimizer_steps,
                    "gradient_steps": gradient_steps,
                    "loss": loss,
                },
            )
        )
        if self.ledger is not None:
            finalized = self.ledger.append_budget_snapshot(
                run_id=self.run_id, counters=self.budgets
            )
            if self.event_callback is not None:
                self.event_callback(finalized)
        if self.iteration % self.config.snapshot_interval == 0:
            self._create_snapshot(emit=True)
        self.evaluate_league()
        return metrics

    def train(self, *, iterations: int) -> tuple[PPOTrainingMetrics, ...]:
        if iterations < 0:
            raise ValueError("iterations cannot be negative")
        return tuple(self.train_iteration() for _ in range(iterations))

    def action_distribution(
        self,
        observation: Observation,
        legal_mask: NDArray[np.bool_],
    ) -> NDArray[np.float32]:
        return self._network_distribution(self.network, observation, legal_mask)

    def action_distribution_step(
        self,
        observation: Observation,
        legal_mask: NDArray[np.bool_],
        *,
        state: PPORecurrentState | None = None,
    ) -> PPOActionDistribution:
        """Return one deployment distribution and the state for the next decision.

        Pass ``state=None`` for the first decision of every episode. Pass the
        returned state into the next call within that episode; discarding it
        resets recurrent history. Feed-forward policies always return ``None``.
        """
        probabilities, next_state = self._network_distribution_step(
            self.network,
            observation,
            legal_mask,
            state=self._deployment_state_batch(state),
        )
        return PPOActionDistribution(
            probabilities=probabilities,
            state=self._deployment_state(next_state),
        )

    def select_action(
        self,
        observation: Observation,
        legal_mask: NDArray[np.bool_],
        *,
        deterministic: bool,
    ) -> int:
        probabilities = self.action_distribution(observation, legal_mask)
        if deterministic:
            return int(np.argmax(probabilities))
        return int(self.rng.choice(len(probabilities), p=probabilities))

    def select_action_step(
        self,
        observation: Observation,
        legal_mask: NDArray[np.bool_],
        *,
        deterministic: bool,
        state: PPORecurrentState | None = None,
    ) -> PPOActionDecision:
        """Select one deployment action and return state for the next decision.

        Initialize or reset an episode with ``state=None``. Within an episode,
        pass each result's state into the following call. The legacy
        :meth:`select_action` remains a stateless convenience method.
        """
        distribution = self.action_distribution_step(
            observation,
            legal_mask,
            state=state,
        )
        if deterministic:
            action = int(np.argmax(distribution.probabilities))
        else:
            action = int(
                self.rng.choice(
                    len(distribution.probabilities),
                    p=distribution.probabilities,
                )
            )
        return PPOActionDecision(
            action=action,
            probabilities=distribution.probabilities,
            state=distribution.state,
        )

    def evaluate(self, *, episodes: int, seed: int = 0) -> PPOEvaluationMetrics:
        if episodes <= 0:
            raise ValueError("episodes must be positive")
        returns: list[float] = []
        env_steps = 0
        wins = draws = losses = 0
        latest = self.opponent_snapshots[-1]
        was_training = self.network.training
        self.network.eval()
        try:
            for episode in range(episodes):
                opponent = self.external_opponent or self._snapshot_policy(
                    latest, deterministic=True
                )
                learner_state: Batch | None = None
                env = GymGameEnv(
                    self.game_factory,
                    controlled_player=episode % 2,
                    opponent=opponent,
                    opponent_id=self.opponent_id,
                    shaping_beta=self.config.shaping_beta,
                    gamma=self.config.gamma,
                    score_scale=self.config.score_scale,
                    action_mapper=self.action_mapper,
                )
                gym_observation, _ = env.reset(seed=seed + episode)
                done = False
                total = 0.0
                terminal_outcome = 0.0
                while not done:
                    action, learner_state = self._action_from_encoded(
                        gym_observation,
                        deterministic=True,
                        state=learner_state,
                    )
                    gym_observation, reward, terminated, truncated, info = env.step(
                        action
                    )
                    total += float(reward)
                    env_steps += 1
                    done = terminated or truncated
                    if info["terminal_outcome"] is not None:
                        terminal_outcome = float(info["terminal_outcome"])
                returns.append(total)
                if terminal_outcome > 0:
                    wins += 1
                elif terminal_outcome < 0:
                    losses += 1
                else:
                    draws += 1
        finally:
            self.network.train(was_training)
        self.budgets.evaluation.episodes += episodes
        self.budgets.evaluation.env_steps += env_steps
        result = PPOEvaluationMetrics(
            episodes=episodes,
            env_steps=env_steps,
            mean_return=float(np.mean(returns)),
            wins=wins,
            draws=draws,
            losses=losses,
        )
        self._emit(
            Event(
                event_type="ppo_evaluation_completed",
                run_id=self.run_id,
                stage="evaluation",
                payload={
                    "iteration": self.iteration,
                    "episodes": episodes,
                    "env_steps": env_steps,
                    "mean_return": result.mean_return,
                    "wins": wins,
                    "draws": draws,
                    "losses": losses,
                },
            )
        )
        return result

    def evaluate_league(self) -> Mapping[str, Any] | None:
        if self.league is None or self.evaluation_callback is None:
            return None
        was_training = self.network.training
        self.network.eval()
        try:
            with torch.inference_mode():
                result = dict(
                    self.evaluation_callback(
                        self.network, self.league, self.iteration
                    )
                )
        finally:
            self.network.train(was_training)
        self._emit(
            Event(
                event_type="ppo_league_evaluated",
                run_id=self.run_id,
                stage="evaluation",
                payload={"iteration": self.iteration, **result},
            )
        )
        return result

    def save_checkpoint(self, path: str | Path) -> PolicyCheckpoint:
        checkpoint = PolicyCheckpoint.save(
            path,
            model=self.network,
            optimizer=self.optimizer,
            trainer_state={
                "ppo_config": asdict(self.config),
                "game_spec": _game_spec_payload(self.game_spec),
                "game_spec_fingerprint": _game_spec_fingerprint(self.game_spec),
                "action_mapper_id": self.action_mapper_id,
                "iteration": self.iteration,
                "optimizer_steps": self.optimizer_steps,
                "budgets": self.budgets.as_dict(),
                "return_statistics": {
                    "mean": copy.deepcopy(self.algorithm.ret_rms.mean),
                    "var": copy.deepcopy(self.algorithm.ret_rms.var),
                    "count": self.algorithm.ret_rms.count,
                },
                "snapshots": [
                    {
                        "generation": snapshot.generation,
                        "model_state": {
                            name: tensor.detach().cpu().clone()
                            for name, tensor in snapshot.network.state_dict().items()
                        },
                    }
                    for snapshot in self.opponent_snapshots
                ],
            },
            custom_rng_state={
                "trainer": copy.deepcopy(self.rng.bit_generator.state)
            },
        )
        self._emit(
            Event(
                event_type="ppo_checkpoint_saved",
                run_id=self.run_id,
                stage="learning",
                payload={
                    "iteration": self.iteration,
                    "optimizer_steps": self.optimizer_steps,
                    "schema_version": checkpoint.schema_version,
                },
            )
        )
        return checkpoint

    def load_checkpoint(self, path: str | Path) -> PolicyCheckpoint:
        checkpoint = PolicyCheckpoint.load(path, map_location=self.network.device)
        state = checkpoint.trainer_state
        required = {
            "ppo_config",
            "game_spec",
            "game_spec_fingerprint",
            "iteration",
            "optimizer_steps",
            "budgets",
            "return_statistics",
            "snapshots",
        }
        if not required.issubset(state):
            raise ValueError("PPO checkpoint trainer state is incomplete")
        try:
            checkpoint_config = PPOConfig(**dict(state["ppo_config"]))
        except (TypeError, ValueError) as exc:
            raise ValueError("checkpoint PPOConfig is invalid") from exc
        if asdict(checkpoint_config) != asdict(self.config):
            raise ValueError("checkpoint PPOConfig does not match this trainer")
        expected_game_spec = _game_spec_payload(self.game_spec)
        expected_fingerprint = _game_spec_fingerprint(self.game_spec)
        if (
            state["game_spec"] != expected_game_spec
            or state["game_spec_fingerprint"] != expected_fingerprint
        ):
            raise ValueError("checkpoint game specification does not match this trainer")
        if state.get("action_mapper_id") != self.action_mapper_id:
            raise ValueError("checkpoint action mapper does not match this trainer")
        checkpoint.validate_restore(model=self.network, optimizer=self.optimizer)
        iteration = _nonnegative_int(state["iteration"], "iteration")
        optimizer_steps = _nonnegative_int(
            state["optimizer_steps"], "optimizer_steps"
        )
        budgets = _restore_budgets(state["budgets"])
        if budgets.learning.optimizer_steps != optimizer_steps:
            raise ValueError("optimizer step counter disagrees with budget counters")
        return_statistics = _validate_return_statistics(state["return_statistics"])
        snapshots = self._stage_snapshots(state["snapshots"])
        custom_rng = checkpoint.custom_rng_state.get("trainer")
        staged_rng = np.random.default_rng()
        try:
            staged_rng.bit_generator.state = copy.deepcopy(dict(custom_rng))
        except Exception as exc:
            raise ValueError("invalid PPO trainer RNG state") from exc

        checkpoint.restore(model=self.network, optimizer=self.optimizer)
        self.iteration = iteration
        self.optimizer_steps = optimizer_steps
        self.budgets = budgets
        self.algorithm.ret_rms.mean = return_statistics["mean"]
        self.algorithm.ret_rms.var = return_statistics["var"]
        self.algorithm.ret_rms.count = return_statistics["count"]
        self.opponent_snapshots = snapshots
        self.rng = staged_rng
        self._telemetry_env_step_id = budgets.learning.env_steps
        self._transition_episode_ids = {}
        self._emit(
            Event(
                event_type="ppo_checkpoint_loaded",
                run_id=self.run_id,
                stage="learning",
                payload={
                    "iteration": self.iteration,
                    "optimizer_steps": self.optimizer_steps,
                    "schema_version": checkpoint.schema_version,
                },
            )
        )
        return checkpoint

    def _make_vector_env(self) -> DummyVectorEnv:
        env_functions: list[Callable[[], GymGameEnv]] = []
        seeds: list[int] = []
        for env_index in range(self.config.vector_envs):
            snapshot = self.opponent_snapshots[
                int(self.rng.integers(0, len(self.opponent_snapshots)))
            ]
            use_external = self.external_opponent is not None and (
                self.rng.random() < self.config.external_opponent_probability
            )
            opponent = (
                self.external_opponent
                if use_external
                else self._snapshot_policy(snapshot, deterministic=False)
            )
            training_opponent_id = (
                self.opponent_id
                if use_external
                else f"ppo-snapshot-{snapshot.generation}"
            )
            transition_sink = self._transition_sink(env_index)
            env_functions.append(
                partial(
                    GymGameEnv,
                    self.game_factory,
                    controlled_player=self.config.training_player,
                    opponent=opponent,
                    opponent_id=training_opponent_id,
                    shaping_beta=self.config.shaping_beta,
                    gamma=self.config.gamma,
                    score_scale=self.config.score_scale,
                    opponent_training_actions=not use_external,
                    action_mapper=self.action_mapper,
                    transition_callback=transition_sink,
                )
            )
            seeds.append(int(self.rng.integers(0, 2**31 - 1)))
        vector_env = DummyVectorEnv(env_functions)
        vector_env.seed(seeds)
        return vector_env

    def _transition_sink(
        self, env_index: int
    ) -> Callable[[Mapping[str, Any]], None]:
        def sink(fact: Mapping[str, Any]) -> None:
            self._telemetry_env_step_id += 1
            local_episode = int(fact["episode_index"])
            episode_key = (env_index, local_episode)
            if episode_key not in self._transition_episode_ids:
                self._transition_episode_ids[episode_key] = (
                    f"{self.run_id}:learning:{self.iteration + 1}:"
                    f"{env_index}:{self._telemetry_env_step_id}"
                )
            self._emit(
                Event(
                    event_type="ppo_transition",
                    run_id=self.run_id,
                    stage="learning",
                    payload={
                        **dict(fact),
                        "iteration": self.iteration + 1,
                        "vector_env_id": env_index,
                        "episode_id": self._transition_episode_ids[episode_key],
                        "env_step_id": self._telemetry_env_step_id,
                    },
                )
            )

        return sink

    def _create_snapshot(self, *, emit: bool) -> OpponentSnapshot:
        frozen = copy.deepcopy(self.network)
        frozen.eval()
        for parameter in frozen.parameters():
            parameter.requires_grad_(False)
        snapshot = OpponentSnapshot(generation=self.iteration, network=frozen)
        self.opponent_snapshots.append(snapshot)
        self.opponent_snapshots = self.opponent_snapshots[
            -self.config.max_snapshots :
        ]
        if emit:
            self._emit(
                Event(
                    event_type="ppo_snapshot_created",
                    run_id=self.run_id,
                    stage="learning",
                    payload={
                        "iteration": self.iteration,
                        "snapshots": len(self.opponent_snapshots),
                    },
                )
            )
        return snapshot

    def _snapshot_policy(
        self, snapshot: OpponentSnapshot, *, deterministic: bool
    ) -> OpponentPolicy:
        return _StatefulOpponent(self, snapshot, deterministic=deterministic)

    def _network_distribution(
        self,
        network: MaskedActorCritic,
        observation: Observation,
        legal_mask: NDArray[np.bool_],
    ) -> NDArray[np.float32]:
        probabilities, _ = self._network_distribution_step(
            network, observation, legal_mask, state=None
        )
        return probabilities

    def _network_distribution_step(
        self,
        network: MaskedActorCritic,
        observation: Observation,
        legal_mask: NDArray[np.bool_],
        *,
        state: Batch | None,
    ) -> tuple[NDArray[np.float32], Batch | None]:
        flat = self._flatten_observation(observation)
        mask = np.asarray(legal_mask, dtype=np.bool_)
        if mask.shape != (network.action_count,) or not np.any(mask):
            raise ValueError("legal_mask must provide non-empty action support")
        was_training = network.training
        network.eval()
        try:
            with torch.inference_mode():
                logits, next_state = network.actor(
                    {"obs": flat[None, :], "mask": mask[None, :]},
                    state=state,
                )
                probabilities = torch.softmax(logits, dim=-1)[0]
        finally:
            network.train(was_training)
        result = probabilities.detach().cpu().numpy().astype(np.float32, copy=True)
        result[~mask] = 0.0
        result /= result.sum()
        return result, next_state

    def _deployment_state_batch(
        self, state: PPORecurrentState | None
    ) -> Batch | None:
        if state is None:
            return None
        if not isinstance(state, PPORecurrentState):
            raise TypeError("state must be PPORecurrentState or None")
        gru = self.network.actor.gru
        if gru is None:
            raise ValueError("feed-forward PPO policies do not accept recurrent state")
        expected = (1, gru.num_layers, gru.hidden_size)
        if state.hidden.shape != expected:
            raise ValueError(f"recurrent hidden state must have shape {expected}")
        return Batch(
            hidden=torch.tensor(
                state.hidden,
                dtype=torch.float32,
                device=self.network.device,
            )
        )

    @staticmethod
    def _deployment_state(state: Batch | None) -> PPORecurrentState | None:
        if state is None:
            return None
        hidden = state.get("hidden")
        if not isinstance(hidden, torch.Tensor):
            raise RuntimeError("recurrent actor returned invalid hidden state")
        return PPORecurrentState(hidden.detach().cpu().numpy())

    def _action_from_encoded(
        self,
        observation: Mapping[str, NDArray[Any]],
        *,
        deterministic: bool,
        state: Batch | None,
    ) -> tuple[int, Batch | None]:
        with torch.inference_mode():
            logits, next_state = self.network.actor(
                {
                    "obs": np.asarray(observation["obs"])[None, :],
                    "mask": np.asarray(observation["mask"])[None, :],
                },
                state=state,
            )
            probabilities = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        if deterministic:
            return int(np.argmax(probabilities)), next_state
        return int(self.rng.choice(len(probabilities), p=probabilities)), next_state

    def _flatten_observation(self, observation: Observation) -> NDArray[np.float32]:
        flat = np.concatenate(
            (
                np.asarray(observation.planes, dtype=np.float32).reshape(-1),
                np.asarray(observation.scalars, dtype=np.float32).reshape(-1),
            )
        ).astype(np.float32, copy=False)
        if flat.shape != (self.network.observation_size,):
            raise ValueError("observation does not match the PPO network")
        return flat

    def _stage_snapshots(self, raw: Any) -> list[OpponentSnapshot]:
        if not isinstance(raw, list) or not raw:
            raise ValueError("PPO checkpoint snapshot pool must be non-empty")
        if len(raw) > self.config.max_snapshots:
            raise ValueError("PPO checkpoint exceeds configured snapshot retention")
        snapshots: list[OpponentSnapshot] = []
        for item in raw:
            if not isinstance(item, Mapping) or not {
                "generation",
                "model_state",
            }.issubset(item):
                raise ValueError("invalid PPO checkpoint snapshot")
            generation = _nonnegative_int(item["generation"], "generation")
            model_state = item["model_state"]
            if not isinstance(model_state, Mapping):
                raise ValueError("invalid PPO snapshot model state")
            frozen = copy.deepcopy(self.network)
            try:
                frozen.load_state_dict(model_state, strict=True)
            except Exception as exc:
                raise ValueError("invalid PPO snapshot model state") from exc
            frozen.eval()
            for parameter in frozen.parameters():
                parameter.requires_grad_(False)
            snapshots.append(OpponentSnapshot(generation, frozen))
        return snapshots

    def _emit(self, event: Event) -> Event:
        finalized = self.ledger.append(event) if self.ledger is not None else event.finalized()
        if self.event_callback is not None:
            self.event_callback(finalized)
        return finalized


def _validate_return_statistics(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or not {"mean", "var", "count"}.issubset(raw):
        raise ValueError("invalid PPO return statistics")
    count = _nonnegative_int(raw["count"], "return count")
    mean = copy.deepcopy(raw["mean"])
    variance = copy.deepcopy(raw["var"])
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(variance)):
        raise ValueError("PPO return statistics must be finite")
    if np.any(np.asarray(variance) < 0):
        raise ValueError("PPO return variance cannot be negative")
    return {"mean": mean, "var": variance, "count": count}


def _game_spec_payload(game_spec: DiscreteGameSpec) -> dict[str, Any]:
    observation_spec = game_spec.observation_spec
    return {
        "name": game_spec.name,
        "players": game_spec.players,
        "zero_sum": game_spec.zero_sum,
        "action_names": list(game_spec.action_names),
        "observation_spec": {
            "plane_names": list(observation_spec.plane_names),
            "board_shape": list(observation_spec.board_shape),
            "scalar_names": list(observation_spec.scalar_names),
        },
        "max_episode_steps": game_spec.max_episode_steps,
    }


def _game_spec_fingerprint(game_spec: DiscreteGameSpec) -> str:
    encoded = json.dumps(
        _game_spec_payload(game_spec),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _restore_budgets(raw: Any) -> BudgetCounters:
    if not isinstance(raw, Mapping):
        raise ValueError("invalid checkpoint budget counters")
    counters = BudgetCounters()
    for stage in ("learning", "evaluation"):
        values = raw.get(stage)
        if not isinstance(values, Mapping):
            raise ValueError("invalid checkpoint budget stage")
        target = getattr(counters, stage)
        for name in ("episodes", "env_steps", "optimizer_steps", "mcts_simulations"):
            setattr(target, name, _nonnegative_int(values.get(name), name))
        for name in ("wall_seconds", "gpu_hours", "cpu_core_hours"):
            value = values.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"invalid checkpoint budget value: {name}")
            setattr(target, name, float(value))
    total = raw.get("total")
    if not isinstance(total, Mapping) or counters.total.as_dict() != dict(total):
        raise ValueError("checkpoint total budget counters are inconsistent")
    return counters


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} cannot be negative")
    return result
