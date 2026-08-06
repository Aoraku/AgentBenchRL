from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from games.snakego import (
    PLANE_NAMES,
    SCALAR_NAMES,
    SNAKEGO_SPEC,
    ItemState,
    OfficialProtocolAdapter,
    SnakeGoGame,
    SnakeGoState,
    SnakeState,
    canonical_action,
    export_alphazero_inference_bundle,
    load_alphazero_inference_bundle,
    load_alphazero_policy,
)
from rlbench.algorithms.alphazero import (
    AlphaZeroConfig,
    AlphaZeroTrainer,
    PolicyValueNet,
)
from rlbench.game import validate_game


def _walls() -> np.ndarray:
    walls = np.full((16, 16), -1, dtype=np.int8)
    walls[2, 7] = 0
    walls[13, 8] = 1
    walls[8, 8] = 2
    return walls


def _symmetric_games() -> tuple[SnakeGoGame, SnakeGoGame]:
    items_a = [
        ItemState(0, 6, 5, 9, 0, 3),
        ItemState(1, 7, 6, 12, 2, 512),
        ItemState(2, 9, 2, 14, 1, 1),
    ]
    state_a = SnakeGoState(
        turn=9,
        current_player=0,
        max_round=32,
        snakes=[
            SnakeState(0, 0, [(4, 4), (3, 4), (2, 4)], length_bank=2),
            SnakeState(2, 0, [(8, 3), (8, 2)]),
            SnakeState(1, 1, [(12, 12), (13, 12)]),
        ],
        items=items_a,
        walls=_walls(),
        first_item_round=4,
        first_item_player=1,
        next_snake_id=3,
        phase_snake_ids=[0, 2],
    )

    def rotate(coord: tuple[int, int]) -> tuple[int, int]:
        return 15 - coord[0], 15 - coord[1]

    id_map = {0: 1, 1: 0, 2: 3}
    snakes_b = [
        SnakeState(
            id_map[snake.id],
            1 - snake.camp,
            [rotate(coord) for coord in snake.coordinates],
            length_bank=snake.length_bank,
            railgun_item_id=snake.railgun_item_id,
        )
        for snake in state_a.snakes
    ]
    items_b = [
        ItemState(
            item.id,
            *rotate((item.x, item.y)),
            item.spawn_round,
            item.item_type,
            item.param,
            gotten_round=item.gotten_round,
            owner_snake_id=id_map.get(item.owner_snake_id, -1),
            expired=item.expired,
        )
        for item in state_a.items
    ]
    rotated_walls = np.rot90(state_a.walls, 2).copy()
    rotated_walls[rotated_walls == 0] = 3
    rotated_walls[rotated_walls == 1] = 0
    rotated_walls[rotated_walls == 3] = 1
    state_b = SnakeGoState(
        turn=9,
        current_player=1,
        max_round=32,
        snakes=snakes_b,
        items=items_b,
        walls=rotated_walls,
        first_item_round=4,
        first_item_player=0,
        next_snake_id=4,
        phase_snake_ids=[1, 3],
    )
    return SnakeGoGame.from_state(state_a), SnakeGoGame.from_state(state_b)


def test_spec_declares_exact_action_and_observation_layout() -> None:
    """Changing metadata silently mis-shapes both PPO and AlphaZero networks."""
    assert SNAKEGO_SPEC.name == "snakego"
    assert SNAKEGO_SPEC.players == 2
    assert SNAKEGO_SPEC.zero_sum is True
    assert SNAKEGO_SPEC.action_names == (
        "right",
        "up",
        "left",
        "down",
        "fire",
        "split",
    )
    assert SNAKEGO_SPEC.observation_spec.board_shape == (16, 16)
    assert SNAKEGO_SPEC.observation_spec.plane_names == PLANE_NAMES
    assert SNAKEGO_SPEC.observation_spec.scalar_names == SCALAR_NAMES
    assert SNAKEGO_SPEC.max_episode_steps == 4096


def test_observation_has_exact_shape_finite_normalized_values_and_named_content() -> None:
    """Missing planes or unbounded features destabilize shared network construction."""
    game, _ = _symmetric_games()
    observation = game.observe(0)

    assert observation.planes.shape == (29, 16, 16)
    assert observation.scalars.shape == (111,)
    assert observation.planes.dtype == np.float32
    assert observation.scalars.dtype == np.float32
    assert np.isfinite(observation.planes).all()
    assert np.isfinite(observation.scalars).all()
    assert np.all((observation.planes >= 0.0) & (observation.planes <= 1.0))
    assert np.all((observation.scalars >= -1.0) & (observation.scalars <= 1.0))

    plane = {name: index for index, name in enumerate(PLANE_NAMES)}
    assert observation.planes[plane["active_head"], 4, 4] == 1.0
    assert observation.planes[plane["active_body"], 3, 4] == 1.0
    assert observation.planes[plane["active_neck"], 3, 4] == 1.0
    assert observation.planes[plane["active_tail"], 2, 4] == 1.0
    assert observation.planes[plane["active_body_order"], 4, 4] == 1.0
    assert observation.planes[plane["active_body_order"], 2, 4] == pytest.approx(
        1 / 3
    )
    assert observation.planes[plane["friendly_body_length"], 3, 4] == pytest.approx(
        3 / 256
    )
    assert observation.planes[plane["friendly_heads"], 8, 3] == 1.0
    assert observation.planes[plane["opponent_heads"], 12, 12] == 1.0
    assert observation.planes[plane["friendly_walls"], 2, 7] == 1.0
    assert observation.planes[plane["opponent_walls"], 13, 8] == 1.0
    assert observation.planes[plane["blocked"], 8, 8] == 1.0
    assert observation.planes[plane["active_length_items"], 6, 5] == 1.0
    assert observation.planes[plane["future_fire_items"], 7, 6] == 1.0
    assert observation.planes[plane["future_split_items"], 9, 2] == 1.0
    assert observation.planes[plane["future_spawn_time"], 7, 6] == pytest.approx(3 / 32)


def test_player_one_rotation_hides_absolute_camp_and_preserves_action_symmetry() -> None:
    """A policy must see identical tensors for 180-degree side-swapped positions."""
    game_a, game_b = _symmetric_games()

    observation_a = game_a.observe(0)
    observation_b = game_b.observe(1)

    np.testing.assert_array_equal(observation_a.planes, observation_b.planes)
    np.testing.assert_allclose(observation_a.scalars, observation_b.scalars)
    np.testing.assert_array_equal(
        game_a.legal_action_mask(), game_b.legal_action_mask()
    )
    assert game_a.encode_state_id(0) == game_b.encode_state_id(1)

    game_a.step(0)
    game_b.step(0)
    np.testing.assert_array_equal(game_a.observe(0).planes, game_b.observe(1).planes)
    np.testing.assert_allclose(game_a.observe(0).scalars, game_b.observe(1).scalars)


@pytest.mark.parametrize(
    ("action", "expected"), [(0, 2), (1, 3), (2, 0), (3, 1), (4, 4), (5, 5)]
)
def test_player_one_action_transform_matches_board_rotation(
    action: int, expected: int
) -> None:
    """A rotated policy action must map back to the corresponding official direction."""
    assert canonical_action(action, player=1) == expected
    assert canonical_action(expected, player=1) == action
    assert canonical_action(action, player=0) == action


def test_score_potential_is_clipped_normalized_and_antisymmetric() -> None:
    """A non-antisymmetric shaping hook would violate the zero-sum objective."""
    game, _ = _symmetric_games()

    potential0 = game.score_potential(0)
    potential1 = game.score_potential(1)

    assert -1.0 <= potential0 <= 1.0
    assert potential0 == pytest.approx(-potential1)
    assert potential0 == pytest.approx((game.score(0) - game.score(1)) / 513.0)


def test_scalar_inventory_distinguishes_split_items_from_railguns() -> None:
    """Collapsing item types prevents a policy from observing official inventory state."""
    item = ItemState(
        0,
        0,
        0,
        1,
        1,
        20,
        gotten_round=2,
        owner_snake_id=0,
    )
    game = SnakeGoGame.from_state(
        SnakeGoState(
            turn=4,
            current_player=0,
            max_round=20,
            snakes=[
                SnakeState(0, 0, [(4, 4), (3, 4)], split_item_id=0),
                SnakeState(1, 1, [(12, 12)]),
            ],
            items=[item],
        )
    )
    scalars = dict(zip(SCALAR_NAMES, game.observe(0).scalars, strict=True))

    assert scalars["inventory_split"] == 1.0
    assert scalars["inventory_fire"] == 0.0


def _observations_are_equal(first, second) -> bool:
    return bool(
        np.array_equal(first.planes, second.planes)
        and np.array_equal(first.scalars, second.scalars)
    )


def test_observation_distinguishes_founder_from_split_child_during_auto_growth() -> None:
    """Founder identity changes whether the same move retains the old tail."""

    def game(active_id: int) -> SnakeGoGame:
        return SnakeGoGame.from_state(
            SnakeGoState(
                turn=4,
                current_player=0,
                max_round=20,
                snakes=[
                    SnakeState(active_id, 0, [(4, 4), (3, 4)]),
                    SnakeState(1, 1, [(12, 12)]),
                ],
                phase_snake_ids=[active_id],
            )
        )

    founder = game(0).observe(0)
    split_child = game(2).observe(0)

    assert not _observations_are_equal(founder, split_child)


def test_observation_distinguishes_the_order_of_pending_active_snakes() -> None:
    """The pending phase order determines which friendly snake acts next."""

    def game(order: list[int]) -> SnakeGoGame:
        return SnakeGoGame.from_state(
            SnakeGoState(
                turn=9,
                current_player=0,
                max_round=20,
                snakes=[
                    SnakeState(0, 0, [(4, 4), (3, 4)]),
                    SnakeState(2, 0, [(7, 5)]),
                    SnakeState(3, 0, [(9, 8)]),
                    SnakeState(1, 1, [(12, 12)]),
                ],
                phase_snake_ids=order,
            )
        )

    second_then_third = game([0, 2, 3]).observe(0)
    third_then_second = game([0, 3, 2]).observe(0)

    assert not _observations_are_equal(second_then_third, third_then_second)


def test_observation_distinguishes_active_length_item_parameters() -> None:
    """Length-item parameters change the bank awarded by an identical pickup."""

    def game(param: int) -> SnakeGoGame:
        return SnakeGoGame.from_state(
            SnakeGoState(
                turn=4,
                current_player=0,
                max_round=20,
                snakes=[
                    SnakeState(0, 0, [(4, 4), (3, 4)]),
                    SnakeState(1, 1, [(12, 12)]),
                ],
                items=[ItemState(0, 5, 4, 4, 0, param)],
            )
        )

    one_growth = game(1).observe(0)
    five_growth = game(5).observe(0)

    assert not _observations_are_equal(one_growth, five_growth)


def test_observation_distinguishes_later_public_items_after_same_earliest_item() -> None:
    """A shared earliest event must not hide a different later announced item."""

    def game(later_round: int) -> SnakeGoGame:
        return SnakeGoGame.from_state(
            SnakeGoState(
                turn=4,
                current_player=0,
                max_round=64,
                snakes=[
                    SnakeState(0, 0, [(4, 4), (3, 4)]),
                    SnakeState(1, 1, [(12, 12)]),
                ],
                items=[
                    ItemState(0, 8, 8, 8, 0, 2),
                    ItemState(1, 8, 8, later_round, 0, 4),
                ],
            )
        )

    earlier_later_item = game(24).observe(0)
    later_later_item = game(40).observe(0)

    assert not _observations_are_equal(earlier_later_item, later_later_item)


def test_seeded_schedules_and_state_ids_are_local_and_deterministic() -> None:
    """Class-global IDs make identical resets depend on unrelated earlier games."""
    first = SnakeGoGame()
    unrelated = SnakeGoGame()
    second = SnakeGoGame()
    first.reset(417)
    unrelated.reset(999)
    second.reset(417)

    first_items = [
        (item.id, item.x, item.y, item.spawn_round, item.item_type, item.param)
        for item in first.state.items
    ]
    second_items = [
        (item.id, item.x, item.y, item.spawn_round, item.item_type, item.param)
        for item in second.state.items
    ]
    assert first_items == second_items
    assert [item.id for item in first.state.items] == list(range(len(first_items)))
    assert first.encode_state_id(0) == second.encode_state_id(0)

    first.step(int(np.flatnonzero(first.legal_action_mask())[0]))
    assert first.encode_state_id(first.current_player()) != second.encode_state_id(
        second.current_player()
    )


def test_clone_is_independent_and_does_not_use_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MCTS clone throughput must not depend on pickle or JSON round trips."""
    game = SnakeGoGame()
    game.reset(7)
    monkeypatch.setattr(pickle, "dumps", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError))
    monkeypatch.setattr(json, "dumps", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError))

    clone = game.clone()
    clone.step(int(np.flatnonzero(clone.legal_action_mask())[0]))

    assert clone is not game
    assert clone.state is not game.state
    assert clone.state.walls is not game.state.walls
    assert clone.encode_state_id(clone.current_player()) != game.encode_state_id(
        game.current_player()
    )


def test_snakego_satisfies_the_shared_discrete_game_contract() -> None:
    """A plugin-specific shortcut must not bypass shared validation."""
    validate_game(SnakeGoGame(max_round=2))


class _RecordingMCTS:
    def __init__(self) -> None:
        self.calls: list[tuple[bool, int]] = []

    def search(self, game: SnakeGoGame, *, training: bool, move_number: int):
        self.calls.append((training, move_number))
        legal = np.flatnonzero(game.legal_action_mask())
        return type("Search", (), {"action": int(legal[0])})()


def _official_config(player: int, max_round: int = 8) -> bytes:
    return bytes((16, 16)) + max_round.to_bytes(2, "big", signed=True) + bytes((player,))


def _official_items(items: list[tuple[int, int, int, int, int]]) -> bytes:
    message = bytearray((0x10,))
    message.extend(len(items).to_bytes(2, "big", signed=True))
    for x, y, item_type, spawn_round, param in items:
        message.extend((x, y, item_type))
        message.extend(spawn_round.to_bytes(2, "big", signed=True))
        message.extend(param.to_bytes(2, "big", signed=True))
    return bytes(message)


def test_protocol_adapter_consumes_official_stream_and_emits_framed_mcts_actions() -> None:
    """The deployment wrapper must maintain echoed operations and deterministic search state."""
    policy = _RecordingMCTS()
    adapter = OfficialProtocolAdapter(policy)

    assert adapter.consume(_official_config(player=0)) is None
    outgoing = adapter.consume(_official_items([(8, 8, 0, 4, 2)]))

    assert outgoing == b"\x00\x00\x00\x01\x01"
    assert policy.calls == [(False, 0)]
    assert adapter.consume(b"\x01") is None  # official echo of our move

    outgoing = adapter.consume(b"\x03")  # opponent moves left

    assert outgoing == b"\x00\x00\x00\x01\x01"
    assert policy.calls == [(False, 0), (False, 2)]
    assert adapter.game is not None
    assert adapter.game.state.turn == 2


def test_protocol_adapter_rejects_bad_echo_without_advancing_state() -> None:
    """Accepting a mismatched judge echo silently desynchronizes every later decision."""
    adapter = OfficialProtocolAdapter(lambda observation, mask: int(np.flatnonzero(mask)[0]))
    adapter.consume(_official_config(player=0))
    adapter.consume(_official_items([(8, 8, 0, 4, 2)]))
    assert adapter.game is not None
    state_id = adapter.game.encode_state_id(adapter.game.current_player())

    with pytest.raises(ValueError, match="echo"):
        adapter.consume(b"\x02")

    assert adapter.game.encode_state_id(adapter.game.current_player()) == state_id


def _deployment_config() -> AlphaZeroConfig:
    return AlphaZeroConfig(
        simulations=2,
        c_puct=1.5,
        root_dirichlet_fraction=0.0,
        self_play_temperature=0.0,
        temperature_moves=0,
        channels=4,
        residual_blocks=1,
        batch_size=1,
        replay_capacity=4,
        min_replay_size=1,
        mixed_precision=False,
        inference_batch_size=1,
    )


def _save_real_alphazero_checkpoint(path: Path) -> tuple[AlphaZeroConfig, dict]:
    torch.manual_seed(73)
    config = _deployment_config()
    network = PolicyValueNet.from_game_spec(SNAKEGO_SPEC, config)
    trainer = AlphaZeroTrainer(network, config, seed=19)
    trainer.save_checkpoint(path)
    return config, {
        name: tensor.detach().clone() for name, tensor in network.state_dict().items()
    }


def test_public_loader_restores_a_real_alphazero_checkpoint_deterministically(
    tmp_path: Path,
) -> None:
    """A deployment loader must restore framework weights rather than use a fake policy."""
    checkpoint = tmp_path / "snakego-alphazero.pt"
    config, expected_state = _save_real_alphazero_checkpoint(checkpoint)

    policy = load_alphazero_policy(checkpoint, config=config, seed=11)

    restored_state = policy.evaluator.state_dict()
    assert set(restored_state) == set(expected_state)
    assert all(
        torch.equal(restored_state[name], expected_state[name])
        for name in expected_state
    )
    game = SnakeGoGame(max_round=2)
    game.reset(5)
    first = policy.search(game, training=False, move_number=0)
    second = load_alphazero_policy(checkpoint, config=config, seed=999).search(
        game, training=False, move_number=0
    )
    assert first.action == second.action
    assert np.array_equal(first.visit_counts, second.visit_counts)
    assert bool(game.legal_action_mask()[first.action]) is True


def test_compact_inference_bundle_embeds_network_config_and_game_schema(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "snakego-training.pt"
    bundle = tmp_path / "snakego-inference.pt"
    config, expected_state = _save_real_alphazero_checkpoint(checkpoint)

    exported = export_alphazero_inference_bundle(
        checkpoint,
        bundle,
        config=config,
    )
    payload = torch.load(exported, map_location="cpu", weights_only=False)
    policy = load_alphazero_inference_bundle(
        bundle,
        simulations=2,
        seed=11,
    )

    assert set(payload) == {
        "format",
        "schema_version",
        "game_spec",
        "config",
        "model_state",
    }
    assert payload["config"]["channels"] == config.channels
    assert payload["config"]["residual_blocks"] == config.residual_blocks
    assert tuple(payload["game_spec"]["plane_names"]) == PLANE_NAMES
    restored_state = policy.evaluator.state_dict()
    assert all(
        torch.equal(restored_state[name], expected_state[name])
        for name in expected_state
    )


def test_inference_bundle_rejects_an_observation_schema_mismatch(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "snakego-training.pt"
    bundle = tmp_path / "snakego-inference.pt"
    config, _ = _save_real_alphazero_checkpoint(checkpoint)
    export_alphazero_inference_bundle(checkpoint, bundle, config=config)
    payload = torch.load(bundle, map_location="cpu", weights_only=False)
    payload["game_spec"]["plane_names"] = ("wrong",)
    torch.save(payload, bundle)

    with pytest.raises(ValueError, match="game schema"):
        load_alphazero_inference_bundle(bundle)


def test_module_agent_loads_checkpoint_and_processes_one_raw_byte_stream(
    tmp_path: Path,
) -> None:
    """The runnable boundary must parse raw official bytes and emit a framed decision."""
    checkpoint = tmp_path / "snakego-process.pt"
    _save_real_alphazero_checkpoint(checkpoint)
    raw_input = (
        _official_config(player=0)
        + _official_items([(8, 8, 0, 4, 2)])
        + b"\x11\x00\x00\x00\x02\x00\x02"
    )
    environment = os.environ.copy()
    source_root = str(Path(__file__).parents[2] / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (source_root, environment.get("PYTHONPATH", "")))
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "games.snakego",
            "--checkpoint",
            str(checkpoint),
            "--simulations",
            "2",
            "--channels",
            "4",
            "--residual-blocks",
            "1",
            "--inference-batch-size",
            "1",
        ],
        input=raw_input,
        capture_output=True,
        check=False,
        timeout=30,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    assert len(completed.stdout) == 5
    assert completed.stdout[:4] == b"\x00\x00\x00\x01"
    assert 1 <= completed.stdout[4] <= 6


def test_module_agent_loads_compact_bundle_without_network_arguments(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "snakego-process.pt"
    bundle = tmp_path / "snakego-process-inference.pt"
    config, _ = _save_real_alphazero_checkpoint(checkpoint)
    export_alphazero_inference_bundle(checkpoint, bundle, config=config)
    raw_input = (
        _official_config(player=0)
        + _official_items([(8, 8, 0, 4, 2)])
        + b"\x11\x00\x00\x00\x02\x00\x02"
    )
    environment = os.environ.copy()
    source_root = str(Path(__file__).parents[2] / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (source_root, environment.get("PYTHONPATH", "")))
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "games.snakego",
            "--bundle",
            str(bundle),
            "--simulations",
            "2",
        ],
        input=raw_input,
        capture_output=True,
        check=False,
        timeout=30,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    assert len(completed.stdout) == 5
    assert completed.stdout[:4] == b"\x00\x00\x00\x01"
    assert 1 <= completed.stdout[4] <= 6
