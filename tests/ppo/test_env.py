from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch
from gymnasium.spaces import Discrete
from tianshou.algorithm.modelfree.reinforce import DiscreteActorPolicy
from tianshou.data import Batch

from rlbench.algorithms.ppo_tianshou import (
    GymGameEnv,
    MaskedActorCritic,
    PPOConfig,
)
from rlbench.game import (
    BoardObservationSpec,
    DiscreteGameSpec,
    Observation,
    StepRecord,
)


class AlternatingGame:
    """Small game whose observations and masks reveal the acting player."""

    spec = DiscreteGameSpec(
        name="alternating",
        players=2,
        zero_sum=True,
        action_names=("left", "middle", "right"),
        observation_spec=BoardObservationSpec(
            plane_names=("move_count",),
            board_shape=(1, 1),
            scalar_names=("viewer",),
        ),
        max_episode_steps=4,
    )

    def __init__(self, terminal_after: int = 4) -> None:
        self.terminal_after = terminal_after
        self.moves: list[tuple[int, int]] = []
        self.terminal = False

    def reset(self, seed: int) -> None:
        self.moves = []
        self.terminal = False

    def current_player(self) -> int:
        return len(self.moves) % 2

    def observe(self, player: int) -> Observation:
        return Observation(
            planes=np.array([[[len(self.moves)]]], dtype=np.float32),
            scalars=np.array([player], dtype=np.float32),
        )

    def legal_action_mask(self) -> np.ndarray:
        if self.current_player() == 0:
            return np.array([True, False, True], dtype=np.bool_)
        return np.array([False, True, True], dtype=np.bool_)

    def step(self, action: int) -> StepRecord:
        actor = self.current_player()
        if not self.legal_action_mask()[action]:
            raise ValueError("illegal")
        self.moves.append((actor, action))
        self.terminal = len(self.moves) == self.terminal_after
        return StepRecord(player=actor, action=action, terminated=self.terminal)

    def outcome(self, player: int) -> float | None:
        if not self.terminal:
            return None
        return 1.0 if player == 0 else -1.0


def factory(*, terminal_after: int = 4) -> Callable[[], AlternatingGame]:
    return lambda: AlternatingGame(terminal_after=terminal_after)


def test_reset_and_step_observe_the_current_controlled_player() -> None:
    """Observing a stale or opponent player breaks canonical policy inputs."""
    env = GymGameEnv(factory(), controlled_player=0, opponent=lambda obs, mask: 1)

    initial, _ = env.reset(seed=7)
    following, reward, terminated, truncated, info = env.step(0)

    assert initial["obs"].tolist() == [0.0, 0.0]
    assert initial["mask"].tolist() == [True, False, True]
    assert following["obs"].tolist() == [2.0, 0.0]
    assert following["mask"].tolist() == [True, False, True]
    assert reward == 0.0
    assert not terminated
    assert not truncated
    assert info["acting_player"] == 0


def test_unspecified_controlled_player_alternates_roles_between_episodes() -> None:
    env = GymGameEnv(factory(), opponent=lambda obs, mask: int(np.flatnonzero(mask)[0]))

    first, first_info = env.reset(seed=7)
    second, second_info = env.reset(seed=8)

    assert first_info["controlled_player"] == 0
    assert first["obs"].tolist() == [0.0, 0.0]
    assert second_info["controlled_player"] == 1
    assert second["obs"].tolist() == [1.0, 1.0]


def test_opponent_advances_with_its_own_observation_and_legal_mask() -> None:
    """Skipping opponent turns or passing the learner mask corrupts transitions."""
    calls: list[tuple[list[float], list[bool]]] = []

    def opponent(observation: Observation, mask: np.ndarray) -> int:
        calls.append((observation.scalars.tolist(), mask.tolist()))
        return 1

    env = GymGameEnv(factory(), controlled_player=0, opponent=opponent)
    env.reset(seed=0)

    env.step(0)

    assert calls == [([1.0], [False, True, True])]
    assert env.game.moves == [(0, 0), (1, 1)]


class GameProcessOpponent:
    def __init__(self) -> None:
        self.events: list[tuple[object, ...]] = []

    def begin_game(self, case, opponent_id: str, side: int, game) -> None:
        self.events.append(("begin", case, opponent_id, side, game.current_player()))

    def act_game_process(self, game, *, timeout_seconds: float | None = None) -> int:
        self.events.append(("act", game.current_player(), timeout_seconds))
        return int(np.flatnonzero(game.legal_action_mask())[0])

    def observe_action(self, game, actor: int, action: int) -> None:
        self.events.append(("observe", actor, action, len(game.moves)))

    def end_game(self, game, result) -> None:
        self.events.append(
            (
                "end",
                result.valid,
                result.reason,
                result.score_player_0,
                game.outcome(0),
            )
        )

    def close(self) -> None:
        self.events.append(("close",))


def test_game_process_opponent_receives_complete_episode_lifecycle() -> None:
    """Official process opponents require state initialization and every action."""
    opponent = GameProcessOpponent()
    env = GymGameEnv(
        factory(terminal_after=2),
        controlled_player=0,
        opponent=opponent,  # type: ignore[arg-type]
        opponent_id="train-human",
        opponent_move_seconds=0.25,
    )

    env.reset(seed=13)
    _, reward, terminated, truncated, _ = env.step(0)

    assert terminated and not truncated and reward == 1.0
    assert opponent.events == [
        ("begin", None, "train-human", 1, 0),
        ("observe", 0, 0, 1),
        ("act", 1, 0.25),
        ("observe", 1, 1, 2),
        ("end", True, "completed", 1.0, 1.0),
    ]


@pytest.mark.parametrize(
    ("controlled_player", "expected_reward"),
    [(0, 1.0), (1, -1.0)],
)
def test_terminal_reward_uses_the_controlled_actor_perspective(
    controlled_player: int, expected_reward: float
) -> None:
    """Returning player-zero reward for both sides violates zero-sum perspective."""
    env = GymGameEnv(
        factory(terminal_after=2),
        controlled_player=controlled_player,
        opponent=lambda obs, mask: int(np.flatnonzero(mask)[0]),
    )
    env.reset(seed=0)

    _, reward, terminated, truncated, info = env.step(
        0 if controlled_player == 0 else 1
    )

    assert reward == expected_reward
    assert terminated
    assert not truncated
    assert info["terminal_reward"] == expected_reward


def test_terminal_observation_has_an_empty_mask_and_rejects_further_steps() -> None:
    """A terminal mask with legal support lets collectors act after game end."""
    env = GymGameEnv(
        factory(terminal_after=2),
        controlled_player=0,
        opponent=lambda obs, mask: 1,
    )
    env.reset(seed=0)

    observation, _, terminated, truncated, _ = env.step(0)

    assert terminated
    assert not truncated
    assert observation["mask"].tolist() == [False, False, False]
    with pytest.raises(RuntimeError, match="terminal"):
        env.step(0)


class PotentialGame(AlternatingGame):
    def __init__(self) -> None:
        super().__init__(terminal_after=4)
        self.points = [0.0, 0.0]

    def reset(self, seed: int) -> None:
        super().reset(seed)
        self.points = [0.0, 0.0]

    def step(self, action: int) -> StepRecord:
        actor = self.current_player()
        record = super().step(action)
        if action == 2:
            self.points[actor] += 1.0
        return record

    def score(self, player: int) -> float:
        return self.points[player]

    def outcome(self, player: int) -> float | None:
        if not self.terminal:
            return None
        margin = self.points[player] - self.points[1 - player]
        return float(np.sign(margin))


class AdapterPotentialGame(PotentialGame):
    def training_potential(self, player: int) -> float:
        margin = self.points[player] - self.points[1 - player]
        return margin / 4.0


class AdapterActionMaskGame(AlternatingGame):
    def training_action_mask(self, player: int) -> np.ndarray:
        mask = self.legal_action_mask().copy()
        if player == 0:
            mask[2] = False
        return mask


def test_potential_shaping_telescopes_over_controlled_decisions() -> None:
    """Using inconsistent players or states prevents potential differences telescoping."""
    env = GymGameEnv(
        PotentialGame,
        controlled_player=0,
        opponent=lambda obs, mask: 1 if len(env.game.moves) == 1 else 2,
        shaping_beta=0.5,
        gamma=1.0,
        score_scale=2.0,
    )
    initial, _ = env.reset(seed=0)
    assert initial["obs"].tolist() == [0.0, 0.0]

    _, first_reward, first_done, _, first_info = env.step(2)
    _, second_reward, second_done, _, second_info = env.step(0)

    assert not first_done
    assert second_done
    assert first_info["shaping_reward"] == pytest.approx(0.25)
    assert second_info["shaping_reward"] == pytest.approx(-0.25)
    assert first_reward + second_reward == pytest.approx(0.0)
    assert first_info["shaping_reward"] + second_info["shaping_reward"] == pytest.approx(
        0.5 * (env.potential(0) - 0.0)
    )


def test_game_adapter_training_potential_overrides_generic_score_scaling() -> None:
    env = GymGameEnv(
        AdapterPotentialGame,
        controlled_player=0,
        opponent=lambda obs, mask: 1,
        shaping_beta=1.0,
        gamma=1.0,
        score_scale=1000.0,
    )
    env.reset(seed=0)

    _, reward, terminated, _, info = env.step(2)

    assert not terminated
    assert reward == pytest.approx(0.25)
    assert info["shaping_reward"] == pytest.approx(0.25)


def test_game_adapter_can_constrain_only_the_controlled_training_actions() -> None:
    opponent_masks: list[list[bool]] = []

    def opponent(observation: Observation, mask: np.ndarray) -> int:
        del observation
        opponent_masks.append(mask.tolist())
        return 1

    env = GymGameEnv(
        AdapterActionMaskGame,
        controlled_player=0,
        opponent=opponent,
    )
    initial, _ = env.reset(seed=0)

    env.step(0)

    assert initial["mask"].tolist() == [True, False, False]
    assert opponent_masks == [[False, True, True]]


def test_masked_actor_critic_masks_logits_before_categorical_sampling() -> None:
    """Applying the legal mask after sampling can emit protocol-invalid actions."""
    config = PPOConfig(hidden_size=16, conv_channels=8)
    network = MaskedActorCritic.from_game_spec(AlternatingGame.spec, config)
    batch = Batch(
        obs={
            "obs": np.zeros((256, 2), dtype=np.float32),
            "mask": np.tile(np.array([True, False, True]), (256, 1)),
        },
        info=Batch(),
    )
    policy = DiscreteActorPolicy(
        actor=network.actor,
        action_space=Discrete(3),
        deterministic_eval=False,
    )

    result = policy(batch)

    assert torch.all(result.logits[:, 1] == torch.finfo(result.logits.dtype).min)
    assert not torch.any(result.act == 1)
    assert torch.all(result.dist.probs[:, 1] == 0)


def test_vector_and_gru_network_contracts_return_expected_shapes() -> None:
    """Hard-wiring the board trunk or dropping recurrent state breaks PPO variants."""
    config = PPOConfig(hidden_size=12, recurrent=True, gru_hidden_size=10)
    network = MaskedActorCritic(
        observation_size=5,
        action_count=3,
        config=config,
    )
    observation = {
        "obs": np.zeros((4, 5), dtype=np.float32),
        "mask": np.ones((4, 3), dtype=np.bool_),
    }

    logits, state = network.actor(observation)
    values = network.critic(observation)

    assert logits.shape == (4, 3)
    assert values.shape == (4, 1)
    assert state is not None
    assert state.hidden.shape == (4, 1, 10)
