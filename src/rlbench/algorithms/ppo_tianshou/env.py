"""Gymnasium adapter for one controlled player in a two-player game."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import Any, Protocol

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from rlbench.game import DiscreteGame, Observation, StepRecord


class OpponentPolicy(Protocol):
    def __call__(
        self, observation: Observation, legal_mask: NDArray[np.bool_]
    ) -> int: ...


TransitionCallback = Callable[[Mapping[str, Any]], None]


class GymGameEnv(gym.Env[dict[str, NDArray[Any]], int]):
    """Expose controlled decisions while advancing all opponent turns internally."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        game_factory: Callable[[], DiscreteGame],
        *,
        controlled_player: int | None = None,
        opponent: OpponentPolicy | None = None,
        shaping_beta: float = 0.0,
        gamma: float = 0.99,
        score_scale: float = 1.0,
        opponent_id: str = "opponent",
        opponent_move_seconds: float | None = None,
        transition_callback: TransitionCallback | None = None,
    ) -> None:
        super().__init__()
        if controlled_player not in (None, 0, 1):
            raise ValueError("controlled_player must be 0, 1, or None")
        if shaping_beta < 0.0:
            raise ValueError("shaping_beta cannot be negative")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if score_scale <= 0.0:
            raise ValueError("score_scale must be positive")
        if not opponent_id:
            raise ValueError("opponent_id must be non-empty")
        if opponent_move_seconds is not None and (
            isinstance(opponent_move_seconds, bool)
            or not isinstance(opponent_move_seconds, (int, float))
            or not math.isfinite(opponent_move_seconds)
            or opponent_move_seconds <= 0.0
        ):
            raise ValueError("opponent_move_seconds must be finite and positive")
        self.game_factory = game_factory
        self.game = game_factory()
        self._configured_player = controlled_player
        self.controlled_player = controlled_player
        self.opponent = opponent
        self.shaping_beta = shaping_beta
        self.gamma = gamma
        self.score_scale = score_scale
        self.opponent_id = opponent_id
        self.opponent_move_seconds = opponent_move_seconds
        self.transition_callback = transition_callback
        self._done = False
        self._game_steps = 0
        self._episode_index = -1
        self._episode_step = 0
        self._opponent_started = False

        observation_spec = self.game.spec.observation_spec
        self._observation_size = (
            len(observation_spec.plane_names)
            * int(np.prod(observation_spec.board_shape))
            + len(observation_spec.scalar_names)
        )
        self.action_space = spaces.Discrete(len(self.game.spec.action_names))
        self.observation_space = spaces.Dict(
            {
                "obs": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self._observation_size,),
                    dtype=np.float32,
                ),
                "mask": spaces.MultiBinary(int(self.action_space.n)),
            }
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
        super().reset(seed=seed)
        del options
        game_seed = (
            int(seed)
            if seed is not None
            else int(self.np_random.integers(0, 2**31 - 1))
        )
        self._close_game_aware_opponent()
        self.game = self.game_factory()
        self.game.reset(game_seed)
        self._episode_index += 1
        self.controlled_player = (
            self._episode_index % 2
            if self._configured_player is None
            else self._configured_player
        )
        self._done = False
        self._game_steps = 0
        self._episode_step = 0
        reset_opponent = getattr(self.opponent, "reset", None)
        if callable(reset_opponent):
            reset_opponent()
        begin_game = getattr(self.opponent, "begin_game", None)
        if callable(begin_game):
            self._opponent_started = True
            try:
                begin_game(
                    None,
                    self.opponent_id,
                    1 - int(self.controlled_player),
                    self.game,
                )
            except BaseException:
                self._close_game_aware_opponent()
                raise
        opponent_records = self._advance_opponent()
        if self._done:
            raise RuntimeError("game terminated before the controlled player's first turn")
        return self._encoded_observation(), {
            "controlled_player": self.controlled_player,
            "opponent_steps": len(opponent_records),
        }

    def step(
        self, action: int
    ) -> tuple[dict[str, NDArray[Any]], float, bool, bool, dict[str, Any]]:
        if self._done:
            raise RuntimeError("cannot step a terminal environment")
        player = self._require_controlled_turn()
        action = int(action)
        legal_mask = self.game.legal_action_mask()
        if action < 0 or action >= self.action_space.n or not bool(legal_mask[action]):
            raise ValueError(f"illegal controlled action: {action}")
        before = self.potential(player)
        records = [self._apply(action)]
        if not self._done:
            records.extend(self._advance_opponent())
        after = self.potential(player)
        terminated = any(record.terminated for record in records)
        truncated = self._done and not terminated
        terminal_reward = 0.0
        terminal_outcome: float | None = None
        if terminated:
            outcome = self.game.outcome(player)
            if outcome is None:
                raise ValueError("terminated game did not expose an outcome")
            terminal_outcome = float(outcome)
            terminal_reward = terminal_outcome
        shaping_reward = self.shaping_beta * (self.gamma * after - before)
        reward = terminal_reward + shaping_reward
        raw_scores: tuple[float | None, float | None] = (None, None)
        score = getattr(self.game, "score", None)
        if self._done and callable(score):
            raw_scores = (float(score(0)), float(score(1)))
        self._episode_step += 1
        info = {
            "acting_player": player,
            "controlled_player": player,
            "opponent_steps": len(records) - 1,
            "terminal_outcome": terminal_outcome,
            "terminal_reward": terminal_reward,
            "shaping_reward": shaping_reward,
            "combined_reward": reward,
            "raw_scores": raw_scores,
            "game_steps": self._game_steps,
            "episode_index": self._episode_index,
            "episode_step": self._episode_step,
        }
        if self.transition_callback is not None:
            self.transition_callback(info)
        return self._encoded_observation(), reward, terminated, truncated, info

    def potential(self, player: int) -> float:
        training_potential = getattr(self.game, "training_potential", None)
        if callable(training_potential):
            value = float(training_potential(player))
            if not math.isfinite(value):
                raise ValueError("game training potential must be finite")
            return float(np.clip(value, -1.0, 1.0))
        score = getattr(self.game, "score", None)
        if not callable(score):
            return 0.0
        margin = float(score(player)) - float(score(1 - player))
        return float(np.clip(margin / self.score_scale, -1.0, 1.0))

    def _advance_opponent(self) -> list[StepRecord]:
        records: list[StepRecord] = []
        while not self._done and self.game.current_player() != self.controlled_player:
            observation = self.game.observe(self.game.current_player())
            mask = np.asarray(self.game.legal_action_mask(), dtype=np.bool_)
            legal = np.flatnonzero(mask)
            if legal.size == 0:
                raise ValueError("opponent turn has no legal actions")
            act_game_process = getattr(self.opponent, "act_game_process", None)
            if callable(act_game_process):
                action = int(
                    act_game_process(
                        self.game,
                        timeout_seconds=self.opponent_move_seconds,
                    )
                )
            else:
                action = (
                    int(self.opponent(observation, mask.copy()))
                    if self.opponent
                    else int(legal[0])
                )
            if action < 0 or action >= self.action_space.n or not bool(mask[action]):
                raise ValueError(f"opponent selected illegal action: {action}")
            records.append(self._apply(action))
        return records

    def _apply(self, action: int) -> StepRecord:
        record = self.game.step(action)
        observe_action = getattr(self.opponent, "observe_action", None)
        if self._opponent_started and callable(observe_action):
            observe_action(self.game, record.player, record.action)
        self._game_steps += 1
        if record.terminated or self._game_steps >= self.game.spec.max_episode_steps:
            self._done = True
            self._finish_game_aware_opponent(terminated=record.terminated)
        return record

    def _finish_game_aware_opponent(self, *, terminated: bool) -> None:
        if not self._opponent_started:
            return
        end_game = getattr(self.opponent, "end_game", None)
        try:
            if callable(end_game):
                outcome = self.game.outcome(0) if terminated else None
                score_player_0 = (
                    1.0
                    if outcome is not None and outcome > 0.0
                    else 0.0
                    if outcome is not None and outcome < 0.0
                    else 0.5
                    if outcome == 0.0
                    else None
                )
                end_game(
                    self.game,
                    SimpleNamespace(
                        valid=terminated,
                        reason="completed" if terminated else "max_episode_steps",
                        score_player_0=score_player_0,
                    ),
                )
        finally:
            self._opponent_started = False

    def _close_game_aware_opponent(self) -> None:
        if not self._opponent_started:
            return
        close = getattr(self.opponent, "close", None)
        try:
            if callable(close):
                close()
        finally:
            self._opponent_started = False

    def close(self) -> None:
        self._close_game_aware_opponent()
        super().close()

    def _require_controlled_turn(self) -> int:
        if self.controlled_player is None:
            raise RuntimeError("environment must be reset before stepping")
        player = self.game.current_player()
        if player != self.controlled_player:
            raise RuntimeError("game is not at the controlled player's turn")
        return player

    def _encoded_observation(self) -> dict[str, NDArray[Any]]:
        if self.controlled_player is None:
            raise RuntimeError("environment must be reset before observation")
        observation = self.game.observe(self.controlled_player)
        flat = np.concatenate(
            (
                np.asarray(observation.planes, dtype=np.float32).reshape(-1),
                np.asarray(observation.scalars, dtype=np.float32).reshape(-1),
            )
        ).astype(np.float32, copy=False)
        if flat.shape != (self._observation_size,):
            raise ValueError("game observation does not match its declared specification")
        mask = np.zeros(self.action_space.n, dtype=np.bool_)
        if not self._done:
            legal_mask = np.asarray(self.game.legal_action_mask(), dtype=np.bool_)
            if legal_mask.shape != (self.action_space.n,):
                raise ValueError("legal action mask does not match the action space")
            training_action_mask = getattr(self.game, "training_action_mask", None)
            mask = (
                np.asarray(
                    training_action_mask(self.controlled_player), dtype=np.bool_
                ).copy()
                if callable(training_action_mask)
                else legal_mask.copy()
            )
            if mask.shape != (self.action_space.n,):
                raise ValueError("training action mask does not match the action space")
            if np.any(mask & ~legal_mask):
                raise ValueError("game training action mask must be a legal subset")
            if not np.any(mask):
                raise ValueError("game training action mask must retain one action")
        if mask.shape != (self.action_space.n,):
            raise ValueError("legal action mask does not match the action space")
        return {"obs": flat, "mask": mask}
