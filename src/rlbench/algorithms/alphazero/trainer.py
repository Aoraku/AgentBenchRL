"""AlphaZero replay learner, generation loop, telemetry, and resume lifecycle."""

from __future__ import annotations

import copy
import math
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from rlbench.algorithms.checkpoint import PolicyCheckpoint
from rlbench.league import LeagueState, OpponentSampler
from rlbench.telemetry import BudgetCounters, Event, EventLedger

from .config import AlphaZeroConfig
from .network import PolicyValueNet
from .replay import ReplayBatch, ReplayBuffer
from .selfplay import SelfPlayWorker


@dataclass(frozen=True, slots=True)
class TrainingMetrics:
    total: float
    policy: float
    value: float
    gradient_norm: float = 0.0


@dataclass(frozen=True, slots=True)
class GenerationMetrics:
    generation: int
    episodes: int
    replay_samples: int
    optimizer_steps: int
    evaluation: Mapping[str, Any] | None = None


LeagueEvaluation = Callable[
    [PolicyValueNet, LeagueState, int], Mapping[str, Any]
]


def _require_allocation_time(deadline_monotonic: float | None) -> None:
    if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
        raise TimeoutError("training allocation deadline is exhausted")


class AlphaZeroTrainer:
    """Own policy-value optimization and the resumable generation lifecycle."""

    def __init__(
        self,
        network: PolicyValueNet,
        config: AlphaZeroConfig,
        *,
        replay: ReplayBuffer | None = None,
        seed: int = 0,
        ledger: EventLedger | None = None,
        run_id: str = "alphazero",
        league: LeagueState | None = None,
        evaluation_callback: LeagueEvaluation | None = None,
    ) -> None:
        self.network = network
        self.config = config
        self.replay = replay if replay is not None else ReplayBuffer(
            config.replay_capacity, seed=seed
        )
        self.rng = np.random.default_rng(seed)
        self.optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self._amp_enabled = bool(
            config.mixed_precision
            and network.device.type == "cuda"
            and torch.cuda.is_available()
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self._amp_enabled)
        self.optimizer_steps = 0
        self.generation = 0
        self.budgets = BudgetCounters()
        self.ledger = ledger
        self.run_id = run_id
        self.league = league
        self.evaluation_callback = evaluation_callback

    def reseed_stage(self, seed: int) -> None:
        """Make a literal stage seed authoritative after checkpoint restore."""
        if (
            not isinstance(seed, int)
            or isinstance(seed, bool)
            or seed < 0
            or seed >= 2**32
        ):
            raise ValueError("stage seed must be an unsigned 32-bit integer")
        self.rng = np.random.default_rng(seed)
        self.replay.rng = np.random.default_rng(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def loss_on_batch(self, batch: ReplayBatch) -> TrainingMetrics:
        tensors = self._batch_tensors(batch)
        was_training = self.network.training
        self.network.eval()
        try:
            with torch.inference_mode():
                total, policy, value = self._loss_tensors(*tensors)
        finally:
            self.network.train(was_training)
        return TrainingMetrics(
            total=float(total.item()),
            policy=float(policy.item()),
            value=float(value.item()),
        )

    def train_step(self, batch: ReplayBatch | None = None) -> TrainingMetrics:
        if batch is None:
            batch = self.replay.sample(self.config.batch_size)
        tensors = self._batch_tensors(batch)
        self.network.train()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=self.network.device.type,
            enabled=self._amp_enabled,
        ):
            total, policy, value = self._loss_tensors(*tensors)
        self.scaler.scale(total).backward()
        self.scaler.unscale_(self.optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.network.parameters(), self.config.gradient_clip_norm
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer_steps += 1
        self.budgets.record_optimizer_step("learning")
        metrics = TrainingMetrics(
            total=float(total.detach().item()),
            policy=float(policy.detach().item()),
            value=float(value.detach().item()),
            gradient_norm=float(gradient_norm),
        )
        if self.ledger is not None:
            self.ledger.append(
                Event(
                    event_type="alphazero_optimizer_step",
                    run_id=self.run_id,
                    stage="learning",
                    payload={
                        "optimizer_step": self.optimizer_steps,
                        "generation": self.generation,
                        "total_loss": metrics.total,
                        "policy_loss": metrics.policy,
                        "value_loss": metrics.value,
                    },
                )
            )
        return metrics

    def run_generation(
        self,
        game_factory: Callable[[], Any],
        *,
        self_play_episodes: int,
        training_steps: int,
        processes: int | None = None,
        training_opponents: Mapping[str, Any] | None = None,
        opponent_episodes: int = 0,
        opponent_move_seconds: float | None = None,
        expert_demo: bool = False,
        expert_demo_opening_moves: int = 0,
        expert_demo_opening_weight: float = 1.0,
        expert_demo_max_decisions: int | None = None,
        deadline_monotonic: float | None = None,
    ) -> GenerationMetrics:
        _require_allocation_time(deadline_monotonic)
        if opponent_episodes < 0 or opponent_episodes > self_play_episodes:
            raise ValueError("opponent_episodes must fit within self_play_episodes")
        if opponent_episodes and (self.league is None or not training_opponents):
            raise ValueError("opponent episodes require a league and policies")
        if opponent_episodes and processes not in (None, 1):
            raise ValueError("opponent episodes require sequential generation")
        if expert_demo and not opponent_episodes:
            raise ValueError("expert_demo requires opponent episodes")
        if expert_demo:
            assert self.league is not None and training_opponents is not None
            held_out = tuple(
                agent_id
                for agent_id in training_opponents
                if any(
                    member.agent_id == agent_id and member.kind == "test_human"
                    for member in self.league.members
                )
            )
            if held_out:
                raise ValueError("expert_demo cannot use test_human policies")
        worker = SelfPlayWorker(
            game_factory,
            self.network,
            self.config,
            seed=int(self.rng.integers(0, 2**31)),
            ledger=self.ledger,
            run_id=self.run_id,
        )
        samples: list[Any] = []
        completed_episodes: list[Any] = []
        first_seed = worker.seed
        if opponent_episodes:
            assert self.league is not None and training_opponents is not None
            available = tuple(
                member
                for member in self.league.members
                if member.agent_id in training_opponents
                and (not expert_demo or member.kind == "train_human")
            )
            sampler = OpponentSampler(
                available, seed=int(self.rng.integers(0, 2**31))
            )
            opponent_ids = sampler.sample(
                learner_rating=self.league.member(self.league.champion_id).rating,
                count=opponent_episodes,
            )
            for index, opponent_id in enumerate(opponent_ids):
                _require_allocation_time(deadline_monotonic)
                samples.extend(
                    worker.play_episode_against(
                        training_opponents[opponent_id],
                        opponent_id=opponent_id,
                        learner_player=index % 2,
                        seed=first_seed + index,
                        timeout_seconds=opponent_move_seconds,
                        expert_demo=expert_demo,
                        expert_demo_opening_moves=expert_demo_opening_moves,
                        expert_demo_opening_weight=expert_demo_opening_weight,
                        expert_demo_max_decisions=expert_demo_max_decisions,
                        deadline_monotonic=deadline_monotonic,
                    )
                )
                completed_episodes.append(worker.last_episode)
                _require_allocation_time(deadline_monotonic)
        pure_episodes = self_play_episodes - opponent_episodes
        if pure_episodes:
            samples.extend(
                worker.play_episodes(
                    pure_episodes,
                    start_seed=first_seed + opponent_episodes,
                    processes=processes,
                    deadline_monotonic=deadline_monotonic,
                )
            )
            completed_episodes.extend(worker.completed_episodes)
        self.replay.extend(samples)
        for episode in completed_episodes:
            self.budgets.learning.episodes += 1
            self.budgets.learning.env_steps += episode.stats.env_steps
            self.budgets.add_mcts_simulations(
                episode.stats.mcts_simulations, "learning"
            )
            if self.ledger is not None:
                self.ledger.append_budget_snapshot(
                    run_id=self.run_id,
                    counters=self.budgets,
                )
        self.run_optimizer_steps(
            training_steps, deadline_monotonic=deadline_monotonic
        )
        self.generation += 1
        evaluation = self.evaluate_league()
        return GenerationMetrics(
            generation=self.generation,
            episodes=self_play_episodes,
            replay_samples=len(samples),
            optimizer_steps=self.optimizer_steps,
            evaluation=evaluation,
        )

    def run_optimizer_steps(
        self,
        training_steps: int,
        *,
        deadline_monotonic: float | None = None,
    ) -> tuple[TrainingMetrics, ...]:
        """Continue optimization from replay without generating new episodes."""
        if isinstance(training_steps, bool) or training_steps < 0:
            raise ValueError("training_steps must be non-negative")
        completed: list[TrainingMetrics] = []
        sampled_weight_mass = 0.0
        for _ in range(training_steps):
            _require_allocation_time(deadline_monotonic)
            if len(self.replay) < max(self.config.min_replay_size, self.config.batch_size):
                break
            batch = self.replay.sample(self.config.batch_size)
            sampled_weight_mass += float(batch.sample_weights.sum())
            completed.append(self.train_step(batch))
        if self.ledger is not None:
            self.ledger.append(
                Event(
                    event_type="alphazero_optimizer_continuation",
                    run_id=self.run_id,
                    stage="learning",
                    payload={
                        "generation": self.generation,
                        "requested_steps": training_steps,
                        "completed_steps": len(completed),
                        "batch_draws": len(completed) * self.config.batch_size,
                        "sampled_weight_mass": sampled_weight_mass,
                        "replay_samples": len(self.replay),
                    },
                )
            )
        return tuple(completed)

    def evaluate_league(self) -> Mapping[str, Any] | None:
        if self.league is None or self.evaluation_callback is None:
            return None
        was_training = self.network.training
        self.network.eval()
        try:
            with torch.inference_mode():
                result = dict(
                    self.evaluation_callback(
                        self.network, self.league, self.generation
                    )
                )
        finally:
            self.network.train(was_training)
        if self.ledger is not None:
            self.ledger.append(
                Event(
                    event_type="alphazero_league_evaluated",
                    run_id=self.run_id,
                    stage="evaluation",
                    payload={"generation": self.generation, **result},
                )
            )
        return result

    def save_checkpoint(self, path: str | Path) -> PolicyCheckpoint:
        checkpoint = PolicyCheckpoint.save(
            path,
            model=self.network,
            optimizer=self.optimizer,
            replay_state=self.replay.state_dict(),
            trainer_state={
                "generation": self.generation,
                "optimizer_steps": self.optimizer_steps,
                "budgets": self.budgets.as_dict(),
                "scaler": self.scaler.state_dict(),
            },
            custom_rng_state=copy.deepcopy(self.rng.bit_generator.state),
        )
        if self.ledger is not None:
            self.ledger.append(
                Event(
                    event_type="alphazero_checkpoint_saved",
                    run_id=self.run_id,
                    stage="learning",
                    payload={
                        "generation": self.generation,
                        "optimizer_steps": self.optimizer_steps,
                        "schema_version": checkpoint.schema_version,
                    },
                )
            )
        return checkpoint

    def load_checkpoint(self, path: str | Path) -> PolicyCheckpoint:
        checkpoint = PolicyCheckpoint.load(path, map_location=self.network.device)
        checkpoint.validate_restore(model=self.network, optimizer=self.optimizer)
        validated_replay = ReplayBuffer(self.config.replay_capacity)
        validated_replay.load_state_dict(checkpoint.replay_state)
        trainer_state = checkpoint.trainer_state
        required = {"generation", "optimizer_steps", "budgets", "scaler"}
        if not isinstance(trainer_state, Mapping) or not required.issubset(trainer_state):
            raise ValueError("checkpoint trainer state is incomplete")
        generation = _nonnegative_int(trainer_state["generation"], "generation")
        optimizer_steps = _nonnegative_int(
            trainer_state["optimizer_steps"], "optimizer_steps"
        )
        restored_budgets = _restore_budgets(trainer_state["budgets"])
        if restored_budgets.learning.optimizer_steps != optimizer_steps:
            raise ValueError("optimizer step counter disagrees with budget counters")
        scaler_state = trainer_state["scaler"]
        if not isinstance(scaler_state, Mapping):
            raise ValueError("invalid checkpoint scaler state")
        if not self._amp_enabled and scaler_state:
            raise ValueError("disabled mixed precision requires an empty scaler state")
        try:
            staged_scaler = copy.deepcopy(self.scaler)
            staged_scaler.load_state_dict(dict(scaler_state))
        except Exception as exc:
            raise ValueError("invalid checkpoint scaler state") from exc
        custom_rng = np.random.default_rng()
        try:
            custom_rng.bit_generator.state = copy.deepcopy(
                dict(checkpoint.custom_rng_state)
            )
        except Exception as exc:
            raise ValueError("invalid checkpoint trainer RNG state") from exc

        snapshot = self._transaction_snapshot()
        try:
            checkpoint.restore(model=self.network, optimizer=self.optimizer)
            self.scaler.load_state_dict(dict(scaler_state))
            self.replay = validated_replay
            self.generation = generation
            self.optimizer_steps = optimizer_steps
            self.budgets = restored_budgets
            self.rng = custom_rng
        except Exception as exc:
            self._rollback_transaction(snapshot)
            raise ValueError("AlphaZero checkpoint restore failed transactionally") from exc
        return checkpoint

    def _transaction_snapshot(self) -> dict[str, Any]:
        return {
            "model": {
                name: tensor.detach().clone()
                for name, tensor in self.network.state_dict().items()
            },
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "scaler": copy.deepcopy(self.scaler.state_dict()),
            "replay": self.replay,
            "replay_state": copy.deepcopy(self.replay.state_dict()),
            "generation": self.generation,
            "optimizer_steps": self.optimizer_steps,
            "budgets": copy.deepcopy(self.budgets),
            "trainer_rng": copy.deepcopy(self.rng.bit_generator.state),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state().clone(),
            "cuda_rng": [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None,
        }

    def _rollback_transaction(self, snapshot: Mapping[str, Any]) -> None:
        self.network.load_state_dict(snapshot["model"], strict=True)
        self.optimizer.load_state_dict(snapshot["optimizer"])
        self.scaler.load_state_dict(snapshot["scaler"])
        replay = snapshot["replay"]
        replay.load_state_dict(snapshot["replay_state"])
        self.replay = replay
        self.generation = snapshot["generation"]
        self.optimizer_steps = snapshot["optimizer_steps"]
        self.budgets = snapshot["budgets"]
        restored_rng = np.random.default_rng()
        restored_rng.bit_generator.state = copy.deepcopy(snapshot["trainer_rng"])
        self.rng = restored_rng
        random.setstate(snapshot["python_rng"])
        np.random.set_state(snapshot["numpy_rng"])
        torch.set_rng_state(snapshot["torch_rng"].cpu())
        if torch.cuda.is_available() and snapshot["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(snapshot["cuda_rng"])

    def _batch_tensors(
        self, batch: ReplayBatch
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        device = self.network.device
        return (
            torch.as_tensor(batch.planes, dtype=torch.float32, device=device),
            torch.as_tensor(batch.scalars, dtype=torch.float32, device=device),
            torch.as_tensor(batch.legal_masks, dtype=torch.bool, device=device),
            torch.as_tensor(batch.visit_policies, dtype=torch.float32, device=device),
            torch.as_tensor(batch.outcomes, dtype=torch.float32, device=device),
            torch.as_tensor(batch.sample_weights, dtype=torch.float32, device=device),
        )

    def _loss_tensors(
        self,
        planes: Tensor,
        scalars: Tensor,
        legal_masks: Tensor,
        target_policies: Tensor,
        target_values: Tensor,
        sample_weights: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        logits, predicted_values = self.network(planes, scalars, legal_masks)
        weight_total = sample_weights.sum()
        policy_losses = -(
            target_policies * torch.log_softmax(logits, dim=1)
        ).sum(dim=1)
        value_losses = torch.nn.functional.mse_loss(
            predicted_values, target_values, reduction="none"
        )
        policy_loss = (policy_losses * sample_weights).sum() / weight_total
        value_loss = (value_losses * sample_weights).sum() / weight_total
        return policy_loss + value_loss, policy_loss, value_loss


def _restore_budgets(raw: Any) -> BudgetCounters:
    if not isinstance(raw, Mapping):
        raise ValueError("invalid checkpoint budget counters")
    counters = BudgetCounters()
    for stage in ("learning", "evaluation"):
        values = raw.get(stage)
        if not isinstance(values, Mapping):
            raise ValueError("invalid checkpoint budget stage")
        target = getattr(counters, stage)
        for name in (
            "episodes",
            "env_steps",
            "optimizer_steps",
            "mcts_simulations",
        ):
            setattr(target, name, _nonnegative_int(values.get(name), name))
        for name in (
            "wall_seconds",
            "gpu_hours",
            "cpu_core_hours",
        ):
            value = values.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"invalid checkpoint budget field: {name}")
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0.0:
                raise ValueError(f"invalid checkpoint budget field: {name}")
            setattr(target, name, numeric)
    return counters


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value
