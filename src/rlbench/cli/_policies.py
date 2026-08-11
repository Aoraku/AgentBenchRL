"""Deadline-aware evaluation policies and the protocol-aware population seam."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np

from rlbench.algorithms.alphazero import MCTS, AlphaZeroConfig, AlphaZeroTrainer
from rlbench.algorithms.ppo_tianshou import PPORecurrentState, PPOTrainer
from rlbench.evaluation import DeadlineAwareGamePolicy, DeadlineAwareLocalPolicy
from rlbench.game import Observation
from rlbench.metrics import policy_kl
from rlbench.population import PopulationEntry, ProcessAgent, ProcessMoveTimeout
from rlbench.registry import official_process_factory


class _AlphaZeroEvaluationPolicy(DeadlineAwareGamePolicy):
    def __init__(
        self,
        trainer: AlphaZeroTrainer,
        config: AlphaZeroConfig,
        seed: int,
        *,
        prior_trainer: AlphaZeroTrainer | None = None,
    ) -> None:
        self.search = MCTS(config, trainer.network, seed=seed)
        self.prior_search = (
            MCTS(config, prior_trainer.network, seed=seed)
            if prior_trainer is not None
            else None
        )
        self.search_calls = 0
        self.completed_simulations = 0
        self.local_kls: list[float] = []

    def act_game(self, game: Any) -> int:
        return self.act_game_with_deadline(game, deadline=None)

    def act_game_with_deadline(self, game: Any, *, deadline: float | None) -> int:
        self.search_calls += 1
        move_number = int(getattr(getattr(game, "state", None), "action_count", 0))
        current_deadline = deadline
        if deadline is not None and self.prior_search is not None:
            started = time.monotonic()
            current_deadline = started + max(0.0, deadline - started) / 2.0
        current = self.search.search(
            game,
            training=False,
            move_number=move_number,
            deadline=current_deadline,
        )
        self.completed_simulations += current.completed_simulations
        if self.prior_search is not None:
            prior = self.prior_search.search(
                game,
                training=False,
                move_number=move_number,
                deadline=deadline,
            )
            self.completed_simulations += prior.completed_simulations
            legal = np.flatnonzero(game.legal_action_mask()).tolist()
            current_probabilities = np.asarray(
                [current.visit_policy[action] for action in legal], dtype=np.float64
            )
            prior_probabilities = np.asarray(
                [prior.visit_policy[action] for action in legal], dtype=np.float64
            )
            current_probabilities /= current_probabilities.sum()
            prior_probabilities /= prior_probabilities.sum()
            self.local_kls.append(
                policy_kl(
                    current_probabilities,
                    prior_probabilities,
                    legal_support=legal,
                )
            )
        return current.action


class _PPOEvaluationPolicy(DeadlineAwareGamePolicy, DeadlineAwareLocalPolicy):
    def __init__(
        self, trainer: PPOTrainer, *, prior_trainer: PPOTrainer | None = None
    ) -> None:
        self.trainer = trainer
        self.prior_trainer = prior_trainer
        self.state: PPORecurrentState | None = None
        self.prior_state: PPORecurrentState | None = None
        self.local_kls: list[float] = []

    def reset_episode(self) -> None:
        self.state = None
        self.prior_state = None

    def __call__(
        self, observation: Observation, legal_mask: np.ndarray[Any, Any]
    ) -> int:
        decision = self.trainer.select_action_step(
            observation,
            legal_mask,
            deterministic=True,
            state=self.state,
        )
        if self.prior_trainer is not None:
            prior = self.prior_trainer.action_distribution_step(
                observation,
                legal_mask,
                state=self.prior_state,
            )
            self.prior_state = prior.state
            legal = np.flatnonzero(legal_mask).tolist()
            current_probabilities = np.asarray(
                [decision.probabilities[action] for action in legal], dtype=np.float64
            )
            prior_probabilities = np.asarray(
                [prior.probabilities[action] for action in legal], dtype=np.float64
            )
            current_probabilities /= current_probabilities.sum()
            prior_probabilities /= prior_probabilities.sum()
            self.local_kls.append(
                policy_kl(
                    current_probabilities,
                    prior_probabilities,
                    legal_support=legal,
                )
            )
        self.state = decision.state
        return decision.action

    def act_with_deadline(
        self,
        observation: Observation,
        legal_mask: np.ndarray[Any, Any],
        *,
        deadline: float | None,
    ) -> int:
        if deadline is not None and time.monotonic() >= deadline:
            raise ProcessMoveTimeout("PPO move deadline expired before inference")
        return self(observation, legal_mask)

    def act_game_with_deadline(self, game: Any, *, deadline: float | None) -> int:
        if deadline is not None and time.monotonic() >= deadline:
            raise ProcessMoveTimeout("PPO move deadline expired before inference")
        player = int(game.current_player())
        observation = game.observe(player)
        training_action_mask = getattr(game, "training_action_mask", None)
        mask = (
            np.asarray(training_action_mask(player), dtype=np.bool_)
            if callable(training_action_mask)
            else np.asarray(game.legal_action_mask(), dtype=np.bool_)
        )
        return self(observation, mask)


class _RandomPolicy:
    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def start_case(self, case: Any, agent_id: str, side: int) -> None:
        payload = f"{case.seed}|{agent_id}|{side}".encode()
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        self.rng = np.random.default_rng(seed)

    def __call__(
        self, observation: Observation, legal_mask: np.ndarray[Any, Any]
    ) -> int:
        del observation
        legal = np.flatnonzero(legal_mask)
        return int(self.rng.choice(legal))


class _TrainingRandomPolicy:
    """Stateless pseudo-random opponent with checkpoint-independent replay."""

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)

    def __call__(
        self, observation: Observation, legal_mask: np.ndarray[Any, Any]
    ) -> int:
        legal = np.flatnonzero(legal_mask)
        digest = hashlib.sha256(self.seed.to_bytes(8, "big", signed=True))
        digest.update(np.asarray(observation.planes, dtype=np.float32).tobytes())
        digest.update(np.asarray(observation.scalars, dtype=np.float32).tobytes())
        digest.update(np.asarray(legal_mask, dtype=np.bool_).tobytes())
        index = int.from_bytes(digest.digest()[:8], "big") % len(legal)
        return int(legal[index])


def _population_policy(
    entry: PopulationEntry, population_root: str | Path, *, game_name: str
) -> ProcessAgent:
    """Construct the process boundary declared by one immutable manifest entry.

    Game-specific wire protocols are self-declared by the game plugin. The CLI
    asks the registry for a handler matching ``entry.protocol`` and falls back
    to the generic line-JSON ``ProcessAgent`` for the default protocol.
    """
    if entry.protocol != "line_json":
        handler = official_process_factory(game_name, entry.protocol)
        if handler is None:
            raise ValueError(
                f"game {game_name!r} does not declare protocol {entry.protocol!r}"
            )
        return handler(entry, population_root)
    return ProcessAgent(entry, population_root)
