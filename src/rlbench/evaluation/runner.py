"""Deterministic local-policy and process-policy match runner."""

from __future__ import annotations

import math
import multiprocessing
import threading
import time
import warnings
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from rlbench.game import DiscreteGame, Observation, clone_game
from rlbench.metrics import MatchOutcome
from rlbench.population import AgentInfrastructureError, ProcessMoveTimeout
from rlbench.telemetry import Event, EventLedger

from .cases import EvaluationCase


class ProcessPolicy(Protocol):
    def act(
        self, request: Mapping[str, Any], *, timeout_seconds: float | None = None
    ) -> int: ...


class GameAwarePolicy(Protocol):
    """A local tree-search policy that requires the complete current state."""

    def act_game(self, game: DiscreteGame) -> int: ...


class DeadlineAwareGamePolicy:
    """Nominal base for framework-owned cooperative game policies."""

    def act_game_with_deadline(
        self, game: DiscreteGame, *, deadline: float | None
    ) -> int:
        raise NotImplementedError


class DeadlineAwareLocalPolicy:
    """Nominal base for framework-owned cooperative recurrent policies."""

    def act_with_deadline(
        self,
        observation: Observation,
        legal_mask: NDArray[np.bool_],
        *,
        deadline: float | None,
    ) -> int:
        raise NotImplementedError


LocalPolicy = Callable[[Observation, NDArray[np.bool_]], int]
Policy = (
    LocalPolicy
    | ProcessPolicy
    | GameAwarePolicy
    | DeadlineAwareGamePolicy
    | DeadlineAwareLocalPolicy
)
_WORKER_CLEANUP_SECONDS = 0.05
_LOCAL_RESULT_PENDING = 0
_LOCAL_RESULT_OK = 1
_LOCAL_RESULT_TIMEOUT = 2
_LOCAL_RESULT_ILLEGAL = 3
_LOCAL_RESULT_ERROR = 4


@dataclass(frozen=True, slots=True)
class MatchResult:
    case_id: str
    case_hash: str
    seed: int
    player_0: str
    player_1: str
    actions: tuple[int, ...]
    score_player_0: float | None
    valid: bool
    reason: str

    def as_outcome(self) -> MatchOutcome:
        return MatchOutcome(
            self.player_0, self.player_1, self.score_player_0, valid=self.valid
        )


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    results: tuple[MatchResult, ...]
    complete: bool

    @property
    def outcomes(self) -> tuple[MatchOutcome, ...]:
        return tuple(result.as_outcome() for result in self.results)


class EvaluationRunner:
    """Run frozen cases and preserve both moves and results as raw facts."""

    def __init__(
        self,
        game_factory: Callable[[Mapping[str, Any]], DiscreteGame],
        ledger: EventLedger,
    ) -> None:
        self.game_factory = game_factory
        self.ledger = ledger

    def run(
        self,
        cases: Iterable[EvaluationCase],
        *,
        agents: Mapping[str, Policy],
        run_id: str,
        evaluation_id: str | None = None,
    ) -> EvaluationReport:
        unique_cases: list[EvaluationCase] = []
        seen_ids: dict[str, str] = {}
        seen_hashes: set[str] = set()
        for case in cases:
            prior_hash = seen_ids.get(case.case_id)
            if prior_hash is not None and prior_hash != case.content_hash:
                raise ValueError("evaluation case_id collision")
            if case.case_id in seen_ids or case.content_hash in seen_hashes:
                continue
            seen_ids[case.case_id] = case.content_hash
            seen_hashes.add(case.content_hash)
            unique_cases.append(case)
        results = tuple(
            self._run_case(
                case,
                agents=agents,
                run_id=run_id,
                evaluation_id=evaluation_id,
            )
            for case in unique_cases
        )
        return EvaluationReport(
            results=results, complete=all(result.valid for result in results)
        )

    def _run_case(
        self,
        case: EvaluationCase,
        *,
        agents: Mapping[str, Policy],
        run_id: str,
        evaluation_id: str | None,
    ) -> MatchResult:
        game = self.game_factory(case.game_config)
        game.reset(case.seed)
        actions: list[int] = []
        for side, agent_id in enumerate((case.player_0, case.player_1)):
            policy = agents.get(agent_id)
            try:
                start_case = getattr(policy, "start_case", None)
                if callable(start_case):
                    start_case(case, agent_id, side)
                reset_episode = getattr(policy, "reset_episode", None)
                if callable(reset_episode):
                    reset_episode()
                begin_game = getattr(policy, "begin_game", None)
                if callable(begin_game):
                    begin_game(case, agent_id, side, game)
            except Exception as exc:
                result = self._infrastructure_result(case, actions, str(exc))
                return self._finish_case(
                    game, case, agents, result, run_id, evaluation_id
                )
        for move_index in range(game.spec.max_episode_steps):
            actor = game.current_player()
            agent_id = case.player_0 if actor == 0 else case.player_1
            try:
                policy = agents[agent_id]
            except KeyError:
                result = self._infrastructure_result(
                    case, actions, f"missing agent: {agent_id}"
                )
                return self._finish_case(
                    game, case, agents, result, run_id, evaluation_id
                )
            observation = game.observe(actor)
            legal_mask = game.legal_action_mask()
            encode_state_id = getattr(game, "encode_state_id", None)
            state_id = (
                encode_state_id(actor).hex() if callable(encode_state_id) else None
            )
            try:
                action = self._select_action(
                    policy,
                    game=game,
                    observation=observation,
                    legal_mask=legal_mask,
                    case=case,
                    actor=actor,
                    move_index=move_index,
                    timeout_seconds=_move_timeout(case),
                )
            except ProcessMoveTimeout:
                result = self._rule_loss(case, actions, actor, "rule_timeout")
                return self._finish_case(
                    game, case, agents, result, run_id, evaluation_id
                )
            except Exception as exc:
                result = self._infrastructure_result(case, actions, str(exc))
                return self._finish_case(
                    game, case, agents, result, run_id, evaluation_id
                )
            if (
                not isinstance(action, int)
                or isinstance(action, bool)
                or action < 0
                or action >= len(legal_mask)
                or not bool(legal_mask[action])
            ):
                result = self._rule_loss(case, actions, actor, "illegal_action")
                return self._finish_case(
                    game, case, agents, result, run_id, evaluation_id
                )
            record = game.step(action)
            actions.append(action)
            try:
                for observed_agent_id in (case.player_0, case.player_1):
                    observe_action = getattr(
                        agents.get(observed_agent_id), "observe_action", None
                    )
                    if callable(observe_action):
                        observe_action(game, actor, action)
            except Exception as exc:
                result = self._infrastructure_result(case, actions, str(exc))
                return self._finish_case(
                    game, case, agents, result, run_id, evaluation_id
                )
            self.ledger.append(
                Event.from_step_record(
                    event_type="evaluation_move",
                    run_id=run_id,
                    stage="evaluation",
                    record=record,
                    payload={
                        "case_id": case.case_id,
                        "case_hash": case.content_hash,
                        "seed": case.seed,
                        "move_index": move_index,
                        "agent_id": agent_id,
                        "evaluation_id": evaluation_id,
                        "state_id": state_id,
                    },
                )
            )
            if record.terminated:
                result = MatchResult(
                    case_id=case.case_id,
                    case_hash=case.content_hash,
                    seed=case.seed,
                    player_0=case.player_0,
                    player_1=case.player_1,
                    actions=tuple(actions),
                    score_player_0=_outcome_score(game.outcome(0)),
                    valid=True,
                    reason="completed",
                )
                return self._finish_case(
                    game, case, agents, result, run_id, evaluation_id
                )
        actor = game.current_player()
        result = self._rule_loss(case, actions, actor, "rule_timeout")
        return self._finish_case(game, case, agents, result, run_id, evaluation_id)

    def _finish_case(
        self,
        game: DiscreteGame,
        case: EvaluationCase,
        agents: Mapping[str, Policy],
        result: MatchResult,
        run_id: str,
        evaluation_id: str | None,
    ) -> MatchResult:
        for agent_id in (case.player_0, case.player_1):
            end_game = getattr(agents.get(agent_id), "end_game", None)
            if callable(end_game):
                try:
                    end_game(game, result)
                except Exception:
                    result = self._infrastructure_result(
                        case, list(result.actions), "end-game hook failed"
                    )
        self._log_match(result, run_id=run_id, evaluation_id=evaluation_id)
        return result

    @staticmethod
    def _select_action(
        policy: Policy,
        *,
        game: DiscreteGame,
        observation: Observation,
        legal_mask: NDArray[np.bool_],
        case: EvaluationCase,
        actor: int,
        move_index: int,
        timeout_seconds: float | None,
    ) -> int:
        if isinstance(policy, DeadlineAwareGamePolicy):
            deadline = (
                None
                if timeout_seconds is None
                else time.monotonic() + timeout_seconds
            )
            action = policy.act_game_with_deadline(
                clone_game(game), deadline=deadline
            )
            if deadline is not None and time.monotonic() > deadline:
                raise ProcessMoveTimeout("trusted game policy exceeded move deadline")
            return action
        if isinstance(policy, DeadlineAwareLocalPolicy):
            deadline = (
                None
                if timeout_seconds is None
                else time.monotonic() + timeout_seconds
            )
            action = policy.act_with_deadline(
                observation, legal_mask.copy(), deadline=deadline
            )
            if deadline is not None and time.monotonic() > deadline:
                raise ProcessMoveTimeout("trusted local policy exceeded move deadline")
            return action
        act_game_process = getattr(policy, "act_game_process", None)
        if callable(act_game_process):
            return act_game_process(game, timeout_seconds=timeout_seconds)
        act = getattr(policy, "act", None)
        if callable(act):
            return act(
                {
                    "protocol_version": case.protocol_version,
                    "case_id": case.case_id,
                    "seed": case.seed,
                    "player": actor,
                    "move_index": move_index,
                    "observation": {
                        "planes": observation.planes.tolist(),
                        "scalars": observation.scalars.tolist(),
                    },
                    "legal_actions": np.flatnonzero(legal_mask).tolist(),
                },
                timeout_seconds=timeout_seconds,
            )
        act_game = getattr(policy, "act_game", None)
        if callable(act_game):
            policy_game = clone_game(game)
            if timeout_seconds is None:
                return act_game(policy_game)
            return _run_timed_local_policy(
                _local_game_policy_worker,
                (policy, policy_game),
                timeout_seconds,
            )
        if timeout_seconds is None:
            return policy(observation, legal_mask.copy())  # type: ignore[operator]
        return _run_timed_local_policy(
            _local_policy_worker,
            (policy, observation, legal_mask.copy()),
            timeout_seconds,
        )

    @staticmethod
    def _rule_loss(
        case: EvaluationCase, actions: list[int], actor: int, reason: str
    ) -> MatchResult:
        return MatchResult(
            case_id=case.case_id,
            case_hash=case.content_hash,
            seed=case.seed,
            player_0=case.player_0,
            player_1=case.player_1,
            actions=tuple(actions),
            score_player_0=0.0 if actor == 0 else 1.0,
            valid=True,
            reason=reason,
        )

    @staticmethod
    def _infrastructure_result(
        case: EvaluationCase, actions: list[int], detail: str
    ) -> MatchResult:
        del detail
        return MatchResult(
            case_id=case.case_id,
            case_hash=case.content_hash,
            seed=case.seed,
            player_0=case.player_0,
            player_1=case.player_1,
            actions=tuple(actions),
            score_player_0=None,
            valid=False,
            reason="infrastructure_failure",
        )

    def _log_match(
        self, result: MatchResult, *, run_id: str, evaluation_id: str | None
    ) -> None:
        self.ledger.append(
            Event(
                event_type="evaluation_match",
                run_id=run_id,
                stage="evaluation",
                payload={
                    "case_id": result.case_id,
                    "case_hash": result.case_hash,
                    "seed": result.seed,
                    "player_0": result.player_0,
                    "player_1": result.player_1,
                    "actions": list(result.actions),
                    "score_player_0": result.score_player_0,
                    "valid": result.valid,
                    "reason": result.reason,
                    "evaluation_id": evaluation_id,
                },
            )
        )


def _outcome_score(outcome: float | None) -> float:
    if outcome is None:
        raise ValueError("terminated game must expose an outcome")
    if outcome > 0:
        return 1.0
    if outcome < 0:
        return 0.0
    return 0.5


def _move_timeout(case: EvaluationCase) -> float | None:
    value = case.limits.get("move_seconds")
    if value is None:
        return None
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0.0
    ):
        raise ValueError("case move_seconds must be finite and positive")
    return float(value)


def _local_policy_worker(
    result: Any,
    policy: LocalPolicy,
    observation: Observation,
    legal_mask: NDArray[np.bool_],
) -> None:
    try:
        action = policy(observation, legal_mask)
    except ProcessMoveTimeout:
        result[0] = _LOCAL_RESULT_TIMEOUT
    except BaseException:
        result[0] = _LOCAL_RESULT_ERROR
    else:
        if (
            isinstance(action, int)
            and not isinstance(action, bool)
            and 0 <= action < len(legal_mask)
        ):
            result[1] = action
            result[0] = _LOCAL_RESULT_OK
        else:
            result[0] = _LOCAL_RESULT_ILLEGAL


def _local_game_policy_worker(
    result: Any,
    policy: GameAwarePolicy,
    game: DiscreteGame,
) -> None:
    try:
        action = policy.act_game(game)
    except ProcessMoveTimeout:
        result[0] = _LOCAL_RESULT_TIMEOUT
    except BaseException:
        result[0] = _LOCAL_RESULT_ERROR
    else:
        if isinstance(action, int) and not isinstance(action, bool):
            result[0] = _LOCAL_RESULT_OK
            result[1] = action
        else:
            result[0] = _LOCAL_RESULT_ILLEGAL


def _run_timed_local_policy(
    target: Callable[..., None],
    arguments: tuple[Any, ...],
    timeout_seconds: float,
) -> int:
    context = multiprocessing.get_context("fork")
    result = context.RawArray("q", 2)
    worker = context.Process(
        target=target,
        args=(result, *arguments),
        daemon=True,
    )
    deadline = time.monotonic() + timeout_seconds
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"This process .* is multi-threaded, use of fork.*",
            category=DeprecationWarning,
        )
        worker.start()
    try:
        remaining = deadline - time.monotonic()
        if remaining > 0.0:
            worker.join(timeout=remaining)
        if time.monotonic() >= deadline or worker.is_alive():
            raise ProcessMoveTimeout(f"local policy exceeded {timeout_seconds}s")
        status = int(result[0])
        action = int(result[1])
    finally:
        _stop_local_worker(worker)
    if status == _LOCAL_RESULT_OK:
        return action
    if status == _LOCAL_RESULT_TIMEOUT:
        raise ProcessMoveTimeout("local policy reported a move timeout")
    if status == _LOCAL_RESULT_ILLEGAL:
        return -1
    raise AgentInfrastructureError("local policy worker failed")


def _stop_local_worker(worker: multiprocessing.Process) -> None:
    if worker.is_alive():
        worker.kill()
    worker.join(timeout=_WORKER_CLEANUP_SECONDS)
    if worker.is_alive():
        threading.Thread(target=worker.join, daemon=True).start()
    else:
        worker.close()
