from __future__ import annotations

import time
import multiprocessing
import signal
from collections.abc import Mapping
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rlbench.evaluation import (
    EvaluationCase,
    EvaluationRunner,
    build_side_swapped_cases,
)
from rlbench.evaluation.runner import (
    DeadlineAwareGamePolicy,
    DeadlineAwareLocalPolicy,
)
from rlbench.game import (
    BoardObservationSpec,
    DiscreteGameSpec,
    Observation,
    StepRecord,
)
from rlbench.population import AgentInfrastructureError, ProcessMoveTimeout
from rlbench.telemetry import EventLedger


class OneMoveGame:
    spec = DiscreteGameSpec(
        name="one-move",
        players=2,
        zero_sum=True,
        action_names=("player-zero-wins", "player-one-wins"),
        observation_spec=BoardObservationSpec(
            plane_names=("seed",), board_shape=(1, 1)
        ),
        max_episode_steps=1,
    )

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.seed = 0
        self.terminal = False
        self.winner = 0

    def reset(self, seed: int) -> None:
        self.seed = seed
        self.terminal = False
        self.winner = 0

    def current_player(self) -> int:
        return 0

    def observe(self, player: int) -> Observation:
        return Observation(
            planes=np.array([[[self.seed]]], dtype=np.float32),
            scalars=np.array([player], dtype=np.float32),
        )

    def legal_action_mask(self) -> np.ndarray:
        return np.array([True, True], dtype=np.bool_)

    def step(self, action: int) -> StepRecord:
        action_zero_winner = int(self.config.get("action_zero_winner", 0))
        self.winner = action_zero_winner if action == 0 else 1 - action_zero_winner
        self.terminal = True
        return StepRecord(player=0, action=action, terminated=True)

    def outcome(self, player: int) -> float | None:
        if not self.terminal:
            return None
        return 1.0 if player == self.winner else -1.0


class TwoMoveSamePlayerGame:
    """Tiny stateful-policy fixture whose first actor takes both moves."""

    spec = DiscreteGameSpec(
        name="two-move-same-player",
        players=2,
        zero_sum=True,
        action_names=("zero", "one"),
        observation_spec=BoardObservationSpec(
            plane_names=("move",), board_shape=(1, 1)
        ),
        max_episode_steps=2,
    )

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        del config
        self.move = 0

    def reset(self, seed: int) -> None:
        del seed
        self.move = 0

    def current_player(self) -> int:
        return 0

    def observe(self, player: int) -> Observation:
        del player
        return Observation(
            planes=np.array([[[self.move]]], dtype=np.float32),
            scalars=np.empty((0,), dtype=np.float32),
        )

    def legal_action_mask(self) -> np.ndarray:
        return np.array([True, True], dtype=np.bool_)

    def step(self, action: int) -> StepRecord:
        record = StepRecord(player=0, action=action, terminated=self.move == 1)
        self.move += 1
        return record

    def outcome(self, player: int) -> float | None:
        if self.move < 2:
            return None
        return 1.0 if player == 0 else -1.0

    def clone(self) -> TwoMoveSamePlayerGame:
        copied = TwoMoveSamePlayerGame()
        copied.move = self.move
        return copied


def _cases():
    return build_side_swapped_cases(
        candidate_id="candidate",
        candidate_hash="sha256:" + "a" * 64,
        opponent_id="baseline",
        opponent_hash="sha256:" + "b" * 64,
        seeds=[13, 21],
        game_config={"variant": "tiny"},
        limits={"move_seconds": 0.1},
        protocol_version="1",
    )


def test_side_swapped_cases_are_frozen_content_hashed_and_deterministic() -> None:
    """Dropping a side, seed stability, or frozen config invalidates comparisons."""
    first = _cases()
    second = _cases()

    assert first == second
    assert [(case.seed, case.player_0, case.player_1) for case in first] == [
        (13, "candidate", "baseline"),
        (13, "baseline", "candidate"),
        (21, "candidate", "baseline"),
        (21, "baseline", "candidate"),
    ]
    assert all(case.content_hash.startswith("sha256:") for case in first)
    with pytest.raises(TypeError):
        first[0].game_config["variant"] = "changed"  # type: ignore[index]


def test_evaluation_case_deep_freezes_mutable_values_inside_tuples() -> None:
    """Tuple-contained dictionaries and lists must not mutate a hashed case."""
    nested = {"layers": ({"values": ["original"]},)}
    case = EvaluationCase.create(
        seed=1,
        player_0="candidate",
        player_1="baseline",
        player_0_hash="sha256:" + "a" * 64,
        player_1_hash="sha256:" + "b" * 64,
        game_config=nested,
    )
    content_hash = case.content_hash

    nested["layers"][0]["values"].append("external-mutation")

    frozen_layer = case.game_config["layers"][0]
    assert frozen_layer["values"] == ("original",)
    assert case.content_hash == content_hash
    with pytest.raises(TypeError):
        frozen_layer["new"] = "mutation"  # type: ignore[index]


def test_runner_uses_case_seeds_and_logs_raw_moves_and_matches(tmp_path: Path) -> None:
    """Ignoring frozen seeds or omitting raw facts makes evaluation irreproducible."""
    ledger = EventLedger(tmp_path / "events.jsonl")
    runner = EvaluationRunner(OneMoveGame, ledger)

    report = runner.run(
        _cases()[:2],
        agents={
            "candidate": lambda observation, legal_mask: int(
                observation.planes[0, 0, 0] % 2
            ),
            "baseline": lambda observation, legal_mask: 0,
        },
        run_id="evaluation-1",
    )

    assert report.complete is True
    assert [result.seed for result in report.results] == [13, 13]
    assert [result.actions for result in report.results] == [(1,), (0,)]
    events = list(ledger.read())
    assert [event.event_type for event in events] == [
        "evaluation_move",
        "evaluation_match",
        "evaluation_move",
        "evaluation_match",
    ]
    assert events[0].payload["seed"] == 13
    assert events[1].payload["case_hash"] == _cases()[0].content_hash


def test_runner_prefers_game_aware_local_policy_for_search(tmp_path: Path) -> None:
    """Reducing tree-search policies to observation callables bypasses real MCTS."""
    seen: list[tuple[int, int]] = []

    class GameAwarePolicy:
        def act_game(self, game: OneMoveGame) -> int:
            seen.append((game.seed, game.current_player()))
            return game.seed % 2

        def __call__(self, observation: Observation, legal_mask: np.ndarray) -> int:
            raise AssertionError("observation-only fallback bypassed game-aware search")

    case = EvaluationCase.create(
        seed=7,
        player_0="candidate",
        player_1="baseline",
        player_0_hash="sha256:" + "a" * 64,
        player_1_hash="sha256:" + "b" * 64,
    )
    report = EvaluationRunner(
        OneMoveGame, EventLedger(tmp_path / "game-aware.jsonl")
    ).run(
        (case,),
        agents={
            "candidate": GameAwarePolicy(),
            "baseline": lambda observation, mask: 0,
        },
        run_id="game-aware",
    )

    assert seen == [(7, 0)]
    assert report.results[0].actions == (1,)


def test_framework_policy_cooperates_with_deadline_without_losing_parent_state(
    tmp_path: Path,
) -> None:
    """Forking a trusted policy per move discards recurrent/search state updates."""

    class FrameworkPolicy(DeadlineAwareGamePolicy):
        def __init__(self) -> None:
            self.calls = 0
            self.deadlines: list[float | None] = []

        def act_game_with_deadline(
            self, game: TwoMoveSamePlayerGame, *, deadline: float | None
        ) -> int:
            del game
            self.calls += 1
            self.deadlines.append(deadline)
            return self.calls - 1

        def act_game(self, game: TwoMoveSamePlayerGame) -> int:
            del game
            self.calls += 1
            return self.calls - 1

    policy = FrameworkPolicy()
    case = EvaluationCase.create(
        seed=17,
        player_0="candidate",
        player_1="baseline",
        player_0_hash="sha256:" + "a" * 64,
        player_1_hash="sha256:" + "b" * 64,
        limits={"move_seconds": 0.5},
    )

    report = EvaluationRunner(
        TwoMoveSamePlayerGame, EventLedger(tmp_path / "cooperative-deadline.jsonl")
    ).run(
        (case,),
        agents={
            "candidate": policy,
            "baseline": lambda observation, mask: 0,
        },
        run_id="cooperative-deadline",
    )

    assert report.complete is True
    assert report.results[0].actions == (0, 1)
    assert policy.calls == 2
    assert len(policy.deadlines) == 2
    assert all(deadline is not None for deadline in policy.deadlines)


def test_framework_observation_policy_preserves_recurrent_state_under_deadline(
    tmp_path: Path,
) -> None:
    """Forking a trusted recurrent policy per move repeats its initial action."""

    class RecurrentFrameworkPolicy(DeadlineAwareLocalPolicy):
        def __init__(self) -> None:
            self.state = 0

        def act_with_deadline(
            self,
            observation: Observation,
            legal_mask: np.ndarray,
            *,
            deadline: float | None,
        ) -> int:
            del observation, legal_mask
            assert deadline is not None
            action = self.state
            self.state += 1
            return action

        def __call__(self, observation: Observation, legal_mask: np.ndarray) -> int:
            del observation, legal_mask
            action = self.state
            self.state += 1
            return action

    policy = RecurrentFrameworkPolicy()
    case = EvaluationCase.create(
        seed=18,
        player_0="candidate",
        player_1="baseline",
        player_0_hash="sha256:" + "a" * 64,
        player_1_hash="sha256:" + "b" * 64,
        limits={"move_seconds": 0.5},
    )

    report = EvaluationRunner(
        TwoMoveSamePlayerGame, EventLedger(tmp_path / "recurrent-deadline.jsonl")
    ).run(
        (case,),
        agents={"candidate": policy, "baseline": lambda observation, mask: 0},
        run_id="recurrent-deadline",
    )

    assert report.results[0].actions == (0, 1)
    assert policy.state == 2


def test_trusted_policy_returning_after_deadline_is_a_rule_timeout(
    tmp_path: Path,
) -> None:
    """Removing the post-return clock check accepts an action produced too late."""

    class SlowTrustedPolicy(DeadlineAwareGamePolicy):
        def act_game_with_deadline(
            self, game: OneMoveGame, *, deadline: float | None
        ) -> int:
            del game, deadline
            time.sleep(0.05)
            return 0

    case = EvaluationCase.create(
        seed=23,
        player_0="candidate",
        player_1="baseline",
        player_0_hash="sha256:" + "a" * 64,
        player_1_hash="sha256:" + "b" * 64,
        limits={"move_seconds": 0.005},
    )

    report = EvaluationRunner(
        OneMoveGame, EventLedger(tmp_path / "trusted-post-deadline.jsonl")
    ).run(
        (case,),
        agents={
            "candidate": SlowTrustedPolicy(),
            "baseline": lambda observation, mask: 0,
        },
        run_id="trusted-post-deadline",
    )

    assert report.results[0].reason == "rule_timeout"
    assert report.results[0].score_player_0 == 0.0


def test_trusted_game_policy_always_receives_an_independent_clone(
    tmp_path: Path,
) -> None:
    """Falling back to the root lets a cooperative policy mutate match state."""
    created_games: list[OneMoveGame] = []

    def factory(config: Mapping[str, Any]) -> OneMoveGame:
        game = OneMoveGame(config)
        created_games.append(game)
        return game

    class MutatingTrustedPolicy(DeadlineAwareGamePolicy):
        received: OneMoveGame | None = None

        def act_game_with_deadline(
            self, game: OneMoveGame, *, deadline: float | None
        ) -> int:
            del deadline
            self.received = game
            game.winner = 1
            return 0

    policy = MutatingTrustedPolicy()
    case = EvaluationCase.create(
        seed=29,
        player_0="candidate",
        player_1="baseline",
        player_0_hash="sha256:" + "a" * 64,
        player_1_hash="sha256:" + "b" * 64,
        limits={"move_seconds": 0.5},
    )

    report = EvaluationRunner(
        factory, EventLedger(tmp_path / "trusted-clone.jsonl")
    ).run(
        (case,),
        agents={"candidate": policy, "baseline": lambda observation, mask: 0},
        run_id="trusted-clone",
    )

    assert policy.received is not created_games[0]
    assert report.results[0].score_player_0 == 1.0


def test_duck_typed_deadline_method_retains_hard_process_isolation(
    tmp_path: Path,
) -> None:
    """Treating any matching method name as trusted permits parent-side effects."""
    parent_mutations: list[str] = []

    class DuckTypedPolicy:
        def act_game_with_deadline(
            self, game: OneMoveGame, *, deadline: float | None
        ) -> int:
            del game, deadline
            parent_mutations.append("duck-deadline")
            time.sleep(0.05)
            return 0

        def act_game(self, game: OneMoveGame) -> int:
            del game
            parent_mutations.append("duck-game")
            time.sleep(0.05)
            return 0

    case = EvaluationCase.create(
        seed=31,
        player_0="candidate",
        player_1="baseline",
        player_0_hash="sha256:" + "a" * 64,
        player_1_hash="sha256:" + "b" * 64,
        limits={"move_seconds": 0.005},
    )

    report = EvaluationRunner(
        OneMoveGame, EventLedger(tmp_path / "duck-isolation.jsonl")
    ).run(
        (case,),
        agents={
            "candidate": DuckTypedPolicy(),
            "baseline": lambda observation, mask: 0,
        },
        run_id="duck-isolation",
    )

    assert report.results[0].reason == "rule_timeout"
    assert parent_mutations == []


def test_runner_resets_stateful_local_policies_for_each_case(tmp_path: Path) -> None:
    """Recurrent hidden state leaking across frozen cases changes their outcomes."""
    resets: list[str] = []

    class StatefulPolicy:
        def reset_episode(self) -> None:
            resets.append("reset")

        def __call__(self, observation: Observation, legal_mask: np.ndarray) -> int:
            return 0

    policy = StatefulPolicy()
    report = EvaluationRunner(
        OneMoveGame, EventLedger(tmp_path / "stateful.jsonl")
    ).run(
        _cases()[:2],
        agents={"candidate": policy, "baseline": policy},
        run_id="stateful",
    )

    assert report.complete is True
    assert resets == ["reset", "reset", "reset", "reset"]


def test_case_lifecycle_makes_policy_results_order_independent(tmp_path: Path) -> None:
    """Case-local random state must not depend on the surrounding case order."""
    class SeededPolicy:
        def __init__(self) -> None:
            self.rng = np.random.default_rng(0)

        def start_case(self, case: EvaluationCase, agent_id: str, side: int) -> None:
            payload = f"{case.seed}|{agent_id}|{side}".encode()
            seed = int.from_bytes(__import__("hashlib").sha256(payload).digest()[:8], "big")
            self.rng = np.random.default_rng(seed)

        def __call__(self, observation: Observation, legal_mask: np.ndarray) -> int:
            del observation
            return int(self.rng.choice(np.flatnonzero(legal_mask)))

    def execute(cases, suffix: str):
        return EvaluationRunner(
            OneMoveGame, EventLedger(tmp_path / f"{suffix}.jsonl")
        ).run(
            cases,
            agents={"candidate": SeededPolicy(), "baseline": SeededPolicy()},
            run_id=suffix,
            evaluation_id=suffix,
        )

    forward = execute(_cases(), "forward")
    reverse = execute(tuple(reversed(_cases())), "reverse")

    assert {
        result.case_id: (result.actions, result.score_player_0)
        for result in forward.results
    } == {
        result.case_id: (result.actions, result.score_player_0)
        for result in reverse.results
    }


def test_runner_deduplicates_cases_within_one_evaluation(tmp_path: Path) -> None:
    """Duplicate case facts within an invocation must not double benchmark weight."""
    case = _cases()[0]
    report = EvaluationRunner(
        OneMoveGame, EventLedger(tmp_path / "dedupe.jsonl")
    ).run(
        (case, case),
        agents={
            "candidate": lambda observation, mask: 0,
            "baseline": lambda observation, mask: 0,
        },
        run_id="dedupe",
        evaluation_id="evaluation-1",
    )

    assert len(report.results) == 1


def test_runner_passes_each_frozen_game_config_to_the_factory(tmp_path: Path) -> None:
    """Ignoring game_config can make distinct content-hashed cases execute identically."""
    observed: list[dict[str, Any]] = []

    def factory(config: Mapping[str, Any]) -> OneMoveGame:
        observed.append(dict(config))
        return OneMoveGame(config)

    common = {
        "seed": 5,
        "player_0": "candidate",
        "player_1": "baseline",
        "player_0_hash": "sha256:" + "a" * 64,
        "player_1_hash": "sha256:" + "b" * 64,
    }
    cases = (
        EvaluationCase.create(**common, game_config={"action_zero_winner": 0}),
        EvaluationCase.create(**common, game_config={"action_zero_winner": 1}),
    )

    report = EvaluationRunner(
        factory, EventLedger(tmp_path / "configured.jsonl")
    ).run(
        cases,
        agents={
            "candidate": lambda observation, mask: 0,
            "baseline": lambda observation, mask: 0,
        },
        run_id="configured",
    )

    assert observed == [
        {"action_zero_winner": 0},
        {"action_zero_winner": 1},
    ]
    assert [result.score_player_0 for result in report.results] == [1.0, 0.0]


def test_case_move_deadline_controls_local_policy_execution(tmp_path: Path) -> None:
    """A hashed move limit that is not enforced permits unbounded policy execution."""
    case = EvaluationCase.create(
        seed=8,
        player_0="slow",
        player_1="baseline",
        player_0_hash="sha256:" + "c" * 64,
        player_1_hash="sha256:" + "b" * 64,
        limits={"move_seconds": 0.02},
    )

    def slow_policy(observation: Observation, legal_mask: np.ndarray) -> int:
        time.sleep(0.20)
        return 0

    started = time.monotonic()
    report = EvaluationRunner(
        OneMoveGame, EventLedger(tmp_path / "deadline.jsonl")
    ).run(
        (case,),
        agents={"slow": slow_policy, "baseline": lambda observation, mask: 0},
        run_id="deadline",
    )
    elapsed = time.monotonic() - started

    assert elapsed < 0.15
    assert report.complete is True
    assert report.results[0].reason == "rule_timeout"
    assert report.results[0].score_player_0 == 0.0


def test_timed_local_policy_oversized_integer_result_cannot_block_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pickle larger than the pipe buffer must not bypass the move deadline."""
    oversized_action = 1 << (16 * 1024 * 1024 * 8)
    original_recv = Connection.recv

    def delayed_recv(connection: Connection) -> object:
        time.sleep(0.25)
        return original_recv(connection)

    monkeypatch.setattr(Connection, "recv", delayed_recv)
    case = EvaluationCase.create(
        seed=11,
        player_0="oversized",
        player_1="baseline",
        player_0_hash="sha256:" + "e" * 64,
        player_1_hash="sha256:" + "b" * 64,
        limits={"move_seconds": 0.20},
    )

    def oversized_policy(
        observation: Observation, legal_mask: np.ndarray
    ) -> int:
        del observation, legal_mask
        return oversized_action

    started = time.monotonic()
    report = EvaluationRunner(
        OneMoveGame, EventLedger(tmp_path / "oversized-result.jsonl")
    ).run(
        (case,),
        agents={
            "oversized": oversized_policy,
            "baseline": lambda observation, mask: 0,
        },
        run_id="oversized-result",
    )
    elapsed = time.monotonic() - started

    assert elapsed < 0.20, (elapsed, report.results[0].reason)
    assert report.complete is True
    assert report.results[0].reason == "illegal_action"


def test_timed_local_policy_is_fork_isolated_and_workers_do_not_accumulate(
    tmp_path: Path,
) -> None:
    """A policy that catches timeout exceptions must not escape its isolated worker."""
    case = EvaluationCase.create(
        seed=9,
        player_0="slow",
        player_1="baseline",
        player_0_hash="sha256:" + "c" * 64,
        player_1_hash="sha256:" + "b" * 64,
        limits={"move_seconds": 0.02},
    )
    parent_mutations: list[str] = []

    def resistant_policy(observation: Observation, legal_mask: np.ndarray) -> int:
        finish = time.monotonic() + 0.20
        while time.monotonic() < finish:
            try:
                time.sleep(0.05)
            except BaseException:
                continue
        parent_mutations.append("continued-after-timeout")
        return 0

    children_before = {child.pid for child in multiprocessing.active_children()}
    started = time.monotonic()
    report = EvaluationRunner(
        OneMoveGame, EventLedger(tmp_path / "fork-deadline.jsonl")
    ).run(
        (case,) * 3,
        agents={
            "slow": resistant_policy,
            "baseline": lambda observation, mask: 0,
        },
        run_id="fork-deadline",
    )
    elapsed = time.monotonic() - started
    time.sleep(0.05)
    children_after = {child.pid for child in multiprocessing.active_children()}

    assert all(result.reason == "rule_timeout" for result in report.results)
    assert elapsed < 0.40
    assert parent_mutations == []
    assert children_after <= children_before


def test_timed_local_policy_does_not_reset_existing_real_timer(tmp_path: Path) -> None:
    """Evaluation deadlines must not install or reset process-global signal state."""
    case = EvaluationCase.create(
        seed=10,
        player_0="policy",
        player_1="baseline",
        player_0_hash="sha256:" + "d" * 64,
        player_1_hash="sha256:" + "b" * 64,
        limits={"move_seconds": 0.50},
    )

    def policy(observation: Observation, legal_mask: np.ndarray) -> int:
        time.sleep(0.15)
        return 0

    original_handler = signal.getsignal(signal.SIGALRM)
    original_timer = signal.getitimer(signal.ITIMER_REAL)

    def prior_handler(signum: int, frame: object) -> None:
        del signum, frame

    signal.signal(signal.SIGALRM, prior_handler)
    signal.setitimer(signal.ITIMER_REAL, 2.0)
    started = time.monotonic()
    try:
        report = EvaluationRunner(
            OneMoveGame, EventLedger(tmp_path / "existing-timer.jsonl")
        ).run(
            (case,),
            agents={
                "policy": policy,
                "baseline": lambda observation, mask: 0,
            },
            run_id="existing-timer",
        )
        elapsed = time.monotonic() - started
        remaining = signal.getitimer(signal.ITIMER_REAL)[0]
        handler_after = signal.getsignal(signal.SIGALRM)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, original_handler)
        signal.setitimer(signal.ITIMER_REAL, *original_timer)

    assert report.results[0].reason == "completed"
    assert remaining <= 2.0 - elapsed + 0.05
    assert handler_after is prior_handler


def test_infrastructure_failure_is_missing_and_marks_evaluation_incomplete(
    tmp_path: Path,
) -> None:
    """Counting a process outage as a game result biases metrics and promotion."""
    def crashed(observation: Observation, legal_mask: np.ndarray) -> int:
        raise AgentInfrastructureError("worker disappeared")

    report = EvaluationRunner(
        OneMoveGame, EventLedger(tmp_path / "events.jsonl")
    ).run(
        _cases()[:1],
        agents={"candidate": crashed, "baseline": lambda observation, mask: 0},
        run_id="evaluation-incomplete",
    )

    assert report.complete is False
    assert report.results[0].valid is False
    assert report.results[0].score_player_0 is None
    assert report.results[0].reason == "infrastructure_failure"
    assert report.outcomes[0].valid is False
    assert report.outcomes[0].score_a is None


@pytest.mark.parametrize(
    ("failing_policy", "reason"),
    [
        (
            lambda observation, legal_mask: (_ for _ in ()).throw(
                ProcessMoveTimeout("deadline")
            ),
            "rule_timeout",
        ),
        (lambda observation, legal_mask: 99, "illegal_action"),
    ],
)
def test_rule_timeout_and_illegal_action_are_valid_losses(
    tmp_path: Path, failing_policy, reason: str
) -> None:
    """Treating rule violations as missing would let weak agents escape losses."""
    report = EvaluationRunner(
        OneMoveGame, EventLedger(tmp_path / f"{reason}.jsonl")
    ).run(
        _cases()[:1],
        agents={
            "candidate": failing_policy,
            "baseline": lambda observation, legal_mask: 0,
        },
        run_id=f"evaluation-{reason}",
    )

    assert report.complete is True
    assert report.results[0].valid is True
    assert report.results[0].score_player_0 == 0.0
    assert report.results[0].reason == reason
    assert report.outcomes[0].valid is True
