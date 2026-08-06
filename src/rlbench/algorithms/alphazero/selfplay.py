"""Sequential and process-parallel AlphaZero self-play generation."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import math
import multiprocessing as mp
import time
from types import SimpleNamespace
from typing import Any

import numpy as np

from rlbench.game import DiscreteGame, Observation, StepRecord
from rlbench.telemetry import Event, EventLedger

from .config import AlphaZeroConfig
from .mcts import MCTS
from .network import BatchEvaluator
from .replay import ReplaySample


@dataclass(frozen=True, slots=True)
class SelfPlayStats:
    episodes: int
    env_steps: int
    mcts_simulations: int


@dataclass(frozen=True, slots=True)
class SelfPlayDecision:
    player: int
    action: int
    terminated: bool
    visit_counts: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SelfPlayEpisode:
    seed: int
    samples: tuple[ReplaySample, ...]
    decisions: tuple[SelfPlayDecision, ...]
    outcomes: tuple[float, float]
    opponent_id: str | None = None
    learner_player: int | None = None

    @property
    def stats(self) -> SelfPlayStats:
        steps = len(self.decisions)
        simulations = sum(
            sum(decision.visit_counts) + 1
            for decision in self.decisions
            if decision.visit_counts
        )
        return SelfPlayStats(episodes=1, env_steps=steps, mcts_simulations=simulations)


_PROCESS_WORKER: SelfPlayWorker | None = None


class SelfPlayWorker:
    """Generate replay targets from training-only noisy, tempered search."""

    def __init__(
        self,
        game_factory: Callable[[], DiscreteGame],
        evaluator: BatchEvaluator,
        config: AlphaZeroConfig,
        *,
        seed: int = 0,
        ledger: EventLedger | None = None,
        run_id: str = "alphazero",
    ) -> None:
        self.game_factory = game_factory
        self.evaluator = evaluator
        self.config = config
        self.seed = seed
        self.ledger = ledger
        self.run_id = run_id
        self.last_stats = SelfPlayStats(episodes=0, env_steps=0, mcts_simulations=0)
        self._last_episode: SelfPlayEpisode | None = None
        self._completed_episodes: tuple[SelfPlayEpisode, ...] = ()

    @property
    def last_episode(self) -> SelfPlayEpisode:
        if self._last_episode is None:
            raise RuntimeError("no self-play episode has completed")
        return self._last_episode

    @property
    def completed_episodes(self) -> tuple[SelfPlayEpisode, ...]:
        return self._completed_episodes

    def play_episode(
        self,
        *,
        seed: int | None = None,
        deadline_monotonic: float | None = None,
    ) -> list[ReplaySample]:
        episode_seed = self.seed if seed is None else seed
        episode = self._generate_episode(
            episode_seed, deadline_monotonic=deadline_monotonic
        )
        self._last_episode = episode
        self._completed_episodes = (episode,)
        self.last_stats = episode.stats
        self._emit_episode(episode)
        return list(episode.samples)

    def play_episode_against(
        self,
        opponent: Any,
        *,
        opponent_id: str,
        learner_player: int,
        seed: int | None = None,
        timeout_seconds: float | None = None,
        expert_demo: bool = False,
        expert_demo_opening_moves: int = 0,
        expert_demo_opening_weight: float = 1.0,
        deadline_monotonic: float | None = None,
    ) -> list[ReplaySample]:
        """Generate search targets and optional expert demonstrations."""
        if learner_player not in (0, 1):
            raise ValueError("learner_player must be zero or one")
        if timeout_seconds is not None and (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0.0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        if (
            isinstance(expert_demo_opening_moves, bool)
            or expert_demo_opening_moves < 0
        ):
            raise ValueError("expert_demo_opening_moves must be non-negative")
        if (
            not isinstance(expert_demo_opening_weight, (int, float))
            or isinstance(expert_demo_opening_weight, bool)
            or not math.isfinite(expert_demo_opening_weight)
            or expert_demo_opening_weight < 1.0
        ):
            raise ValueError("expert_demo_opening_weight must be finite and at least one")
        if not expert_demo and (
            expert_demo_opening_moves or expert_demo_opening_weight != 1.0
        ):
            raise ValueError("expert opening emphasis requires expert_demo")
        episode_seed = self.seed if seed is None else seed
        episode = self._generate_episode(
            episode_seed,
            opponent=opponent,
            opponent_id=opponent_id,
            learner_player=learner_player,
            timeout_seconds=timeout_seconds,
            expert_demo=expert_demo,
            expert_demo_opening_moves=expert_demo_opening_moves,
            expert_demo_opening_weight=float(expert_demo_opening_weight),
            deadline_monotonic=deadline_monotonic,
        )
        self._last_episode = episode
        self._completed_episodes = (episode,)
        self.last_stats = episode.stats
        self._emit_episode(episode)
        return list(episode.samples)

    def _generate_episode(
        self,
        episode_seed: int,
        *,
        opponent: Any | None = None,
        opponent_id: str | None = None,
        learner_player: int | None = None,
        timeout_seconds: float | None = None,
        expert_demo: bool = False,
        expert_demo_opening_moves: int = 0,
        expert_demo_opening_weight: float = 1.0,
        deadline_monotonic: float | None = None,
    ) -> SelfPlayEpisode:
        _require_deadline(deadline_monotonic)
        game = self.game_factory()
        game.reset(episode_seed)
        search = MCTS(self.config, self.evaluator, seed=episode_seed)
        pending: list[tuple[Observation, object, object, int, str, float, int]] = []
        decisions: list[SelfPlayDecision] = []
        step_count = 0
        expert_decision_index = 0
        try:
            if opponent is not None:
                begin_game = getattr(opponent, "begin_game", None)
                if callable(begin_game):
                    begin_game(None, opponent_id, 1 - int(learner_player), game)
            while game.outcome(game.current_player()) is None:
                player = game.current_player()
                observation = game.observe(player)
                legal_mask = game.legal_action_mask()
                if opponent is None or player == learner_player:
                    _require_deadline(deadline_monotonic)
                    result = search.search(
                        game,
                        training=True,
                        move_number=step_count,
                        deadline=deadline_monotonic,
                    )
                    _require_deadline(deadline_monotonic)
                    pending.append(
                        (
                            observation,
                            legal_mask,
                            result.visit_policy,
                            player,
                            "selfplay" if opponent is None else "learner",
                            1.0,
                            -1,
                        )
                    )
                    action = result.action
                    visit_counts = tuple(
                        int(count) for count in result.visit_counts
                    )
                else:
                    _require_deadline(
                        deadline_monotonic, unit_bound_seconds=timeout_seconds
                    )
                    action = _opponent_action(
                        opponent,
                        game,
                        observation,
                        legal_mask,
                        timeout_seconds=timeout_seconds,
                    )
                    _require_deadline(deadline_monotonic)
                    visit_counts = ()
                if (
                    not isinstance(action, int)
                    or isinstance(action, bool)
                    or action < 0
                    or action >= len(legal_mask)
                    or not bool(legal_mask[action])
                ):
                    raise ValueError("training opponent selected an illegal action")
                if opponent is not None and player != learner_player and expert_demo:
                    expert_policy = np.zeros(len(legal_mask), dtype=np.float32)
                    expert_policy[action] = 1.0
                    pending.append(
                        (
                            observation,
                            legal_mask,
                            expert_policy,
                            player,
                            "expert_demo",
                            expert_demo_opening_weight
                            if expert_decision_index < expert_demo_opening_moves
                            else 1.0,
                            expert_decision_index,
                        )
                    )
                    expert_decision_index += 1
                record = game.step(action)
                decisions.append(
                    SelfPlayDecision(
                        player=record.player,
                        action=record.action,
                        terminated=record.terminated,
                        visit_counts=visit_counts,
                    )
                )
                if opponent is not None:
                    observe_action = getattr(opponent, "observe_action", None)
                    if callable(observe_action):
                        observe_action(game, record.player, record.action)
                step_count += 1
        except BaseException:
            close = getattr(opponent, "close", None)
            if callable(close):
                close()
            raise
        samples: list[ReplaySample] = []
        for (
            observation,
            legal_mask,
            policy,
            player,
            source,
            sample_weight,
            decision_index,
        ) in pending:
            outcome = game.outcome(player)
            if outcome is None:
                raise ValueError("completed self-play game must expose outcomes")
            samples.append(
                ReplaySample(
                    observation=observation,
                    legal_mask=legal_mask,
                    visit_policy=policy,
                    outcome=outcome,
                    player=player,
                    source=source,
                    sample_weight=sample_weight,
                    decision_index=decision_index,
                )
            )
        outcomes = (game.outcome(0), game.outcome(1))
        if outcomes[0] is None or outcomes[1] is None:
            raise ValueError("completed self-play game must expose both outcomes")
        if opponent is not None:
            end_game = getattr(opponent, "end_game", None)
            if callable(end_game):
                score_player_0 = (
                    1.0 if outcomes[0] > 0.0 else 0.0 if outcomes[0] < 0.0 else 0.5
                )
                end_game(
                    game,
                    SimpleNamespace(
                        valid=True,
                        reason="completed",
                        score_player_0=score_player_0,
                    ),
                )
        return SelfPlayEpisode(
            seed=episode_seed,
            samples=tuple(samples),
            decisions=tuple(decisions),
            outcomes=(float(outcomes[0]), float(outcomes[1])),
            opponent_id=opponent_id,
            learner_player=learner_player,
        )

    def _emit_episode(self, episode: SelfPlayEpisode) -> None:
        if self.ledger is None:
            return
        for decision in episode.decisions:
            self.ledger.append(
                Event.from_step_record(
                    event_type="alphazero_self_play_step",
                    run_id=self.run_id,
                    stage="learning",
                    record=StepRecord(
                        player=decision.player,
                        action=decision.action,
                        terminated=decision.terminated,
                    ),
                    payload={
                        "seed": episode.seed,
                        "visit_counts": list(decision.visit_counts),
                    },
                )
            )
        sample_sources = {
            source: sum(sample.source == source for sample in episode.samples)
            for source in ("expert_demo", "learner", "selfplay")
        }
        self.ledger.append(
            Event(
                event_type="alphazero_self_play_episode",
                run_id=self.run_id,
                stage="learning",
                payload={
                    "seed": episode.seed,
                    "env_steps": episode.stats.env_steps,
                    "mcts_simulations": episode.stats.mcts_simulations,
                    "outcomes": list(episode.outcomes),
                    "opponent_id": episode.opponent_id,
                    "learner_player": episode.learner_player,
                    "sample_sources": sample_sources,
                },
            )
        )

    def play_episodes(
        self,
        episodes: int,
        *,
        start_seed: int | None = None,
        processes: int | None = None,
        deadline_monotonic: float | None = None,
    ) -> list[ReplaySample]:
        if episodes < 1:
            raise ValueError("episodes must be positive")
        first_seed = self.seed if start_seed is None else start_seed
        seeds = list(range(first_seed, first_seed + episodes))
        process_count = self.config.self_play_workers if processes is None else processes
        if process_count < 1:
            raise ValueError("processes must be positive")
        if deadline_monotonic is not None and process_count != 1:
            raise ValueError("allocation deadlines require sequential self-play")
        if process_count == 1:
            completed = []
            for episode_seed in seeds:
                _require_deadline(deadline_monotonic)
                completed.append(
                    self._generate_episode(
                        episode_seed, deadline_monotonic=deadline_monotonic
                    )
                )
        else:
            context = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=process_count,
                mp_context=context,
                initializer=_initialize_process_worker,
                initargs=(self.game_factory, self.evaluator, self.config),
            ) as executor:
                completed = list(
                    executor.map(_play_initialized_episode, seeds, chunksize=1)
                )
        for episode in completed:
            self._emit_episode(episode)
        self._last_episode = completed[-1]
        self._completed_episodes = tuple(completed)
        samples = [sample for episode in completed for sample in episode.samples]
        self.last_stats = SelfPlayStats(
            episodes=episodes,
            env_steps=sum(episode.stats.env_steps for episode in completed),
            mcts_simulations=sum(
                episode.stats.mcts_simulations for episode in completed
            ),
        )
        return samples


def _initialize_process_worker(
    game_factory: Callable[[], DiscreteGame],
    evaluator: BatchEvaluator,
    config: AlphaZeroConfig,
) -> None:
    global _PROCESS_WORKER
    _PROCESS_WORKER = SelfPlayWorker(game_factory, evaluator, config)


def _play_initialized_episode(seed: int) -> SelfPlayEpisode:
    if _PROCESS_WORKER is None:
        raise RuntimeError("self-play process was not initialized")
    return _PROCESS_WORKER._generate_episode(seed)


def _opponent_action(
    opponent: Any,
    game: DiscreteGame,
    observation: Observation,
    legal_mask: Any,
    *,
    timeout_seconds: float | None,
) -> int:
    act_game_process = getattr(opponent, "act_game_process", None)
    if callable(act_game_process):
        return act_game_process(game, timeout_seconds=timeout_seconds)
    act_game = getattr(opponent, "act_game", None)
    if callable(act_game):
        return act_game(game)
    return opponent(observation, legal_mask.copy())


def _require_deadline(
    deadline_monotonic: float | None,
    *,
    unit_bound_seconds: float | None = None,
) -> None:
    if deadline_monotonic is None:
        return
    remaining = deadline_monotonic - time.monotonic()
    if remaining <= 0.0:
        raise TimeoutError("training allocation deadline is exhausted")
    if unit_bound_seconds is not None and remaining < unit_bound_seconds:
        raise TimeoutError("remaining allocation cannot cover bounded unit")
