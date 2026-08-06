from __future__ import annotations

import numpy as np
import pytest

from games.snakego import (
    IllegalActionError,
    ItemState,
    SnakeGoEngine,
    SnakeGoState,
    SnakeState,
)


def _snake(
    snake_id: int,
    camp: int,
    *coords: tuple[int, int],
    bank: int = 0,
    split_item: int | None = None,
    railgun: int | None = None,
) -> SnakeState:
    return SnakeState(
        id=snake_id,
        camp=camp,
        coordinates=list(coords),
        length_bank=bank,
        split_item_id=split_item,
        railgun_item_id=railgun,
    )


def _engine(
    *snakes: SnakeState,
    turn: int = 9,
    player: int = 0,
    max_round: int = 512,
    items: list[ItemState] | None = None,
    walls: np.ndarray | None = None,
) -> SnakeGoEngine:
    state = SnakeGoState(
        turn=turn,
        current_player=player,
        max_round=max_round,
        snakes=list(snakes),
        items=list(items or []),
        walls=(
            np.full((16, 16), -1, dtype=np.int8)
            if walls is None
            else walls
        ),
    )
    return SnakeGoEngine.from_state(state)


def test_move_uses_official_direction_numbers_and_consumes_the_tail() -> None:
    """Swapping either axis or retaining the tail breaks official movement."""
    engine = _engine(
        _snake(0, 0, (4, 4), (3, 4), (2, 4)),
        _snake(1, 1, (15, 0)),
    )

    engine.step(1)  # up

    assert engine.state.snakes[0].coordinates == [(4, 5), (4, 4), (3, 4)]
    assert engine.current_snake.id == 1


def test_initial_snakes_auto_grow_for_the_first_eight_rounds() -> None:
    """Disabling official auto-growth shortens the two founding snakes."""
    engine = _engine(
        _snake(0, 0, (2, 2), (1, 2)),
        _snake(1, 1, (13, 13)),
        turn=8,
    )

    engine.step(0)

    assert engine.state.snakes[0].coordinates == [(3, 2), (2, 2), (1, 2)]


def test_growth_bank_extends_once_then_decrements() -> None:
    """A banked unit must preserve the old tail for exactly one movement."""
    engine = _engine(
        _snake(0, 0, (2, 2), (1, 2), bank=1),
        _snake(1, 1, (13, 13)),
    )

    engine.step(0)

    grown = engine.state.snakes[0]
    assert grown.coordinates == [(3, 2), (2, 2), (1, 2)]
    assert grown.length_bank == 0


@pytest.mark.parametrize(
    ("target", "walls"),
    [
        ((-1, 0), None),
        (
            (1, 0),
            np.pad(np.array([[0]], dtype=np.int8), ((1, 14), (0, 15)), constant_values=-1),
        ),
    ],
)
def test_boundary_and_wall_moves_are_legal_suicides(
    target: tuple[int, int], walls: np.ndarray | None
) -> None:
    """Masking a strategically fatal move would change protocol support."""
    head = (0, 0) if target[0] < 0 else (0, 0)
    engine = _engine(
        _snake(0, 0, head),
        _snake(1, 1, (15, 15)),
        walls=walls,
    )
    action = 2 if target[0] < 0 else 0

    assert bool(engine.legal_action_mask()[action]) is True
    engine.step(action)

    assert [snake.id for snake in engine.state.snakes] == [1]


def test_moving_into_another_snake_is_a_legal_suicide() -> None:
    """Opponent occupancy is fatal but does not make a direction malformed."""
    engine = _engine(
        _snake(0, 0, (3, 3), (2, 3)),
        _snake(1, 1, (4, 3)),
    )

    assert bool(engine.legal_action_mask()[0]) is True
    engine.step(0)

    assert [snake.id for snake in engine.state.snakes] == [1]


def test_forbidden_reversal_is_masked_and_rejected_without_mutation() -> None:
    """Allowing an occupied neck reversal violates the official protocol."""
    engine = _engine(
        _snake(0, 0, (4, 4), (3, 4), (2, 4)),
        _snake(1, 1, (15, 0)),
    )
    before = engine.state.clone()

    assert engine.legal_action_mask().tolist() == [True, True, False, True, False, True]
    with pytest.raises(IllegalActionError, match="reversal"):
        engine.step(2)

    assert engine.state == before


def test_length_two_can_reverse_only_when_its_tail_will_move() -> None:
    """Applying the length-three reversal rule to a shrinking length-two snake is wrong."""
    engine = _engine(
        _snake(0, 0, (4, 4), (3, 4)),
        _snake(1, 1, (15, 0)),
        turn=9,
    )

    assert bool(engine.legal_action_mask()[2]) is True
    engine.step(2)

    assert engine.state.snakes[0].coordinates == [(3, 4), (4, 4)]


def test_length_pickup_banks_growth_and_awards_first_item_bonus() -> None:
    """A length item must disappear, bank its literal parameter, and set first ownership."""
    item = ItemState(id=4, x=5, y=4, spawn_round=9, item_type=0, param=3)
    engine = _engine(
        _snake(0, 0, (4, 4), (3, 4)),
        _snake(1, 1, (15, 0)),
        items=[item],
    )

    transition = engine.step(0)

    assert transition.gotten_item_id == 4
    assert engine.state.snakes[0].length_bank == 3
    assert engine.state.first_item_player == 0
    assert engine.scores() == (5, 2)


def test_item_spawn_pickup_and_expiry_run_once_per_complete_round() -> None:
    """Preprocessing at a player boundary would spawn or expire items twice."""
    spawning = ItemState(id=0, x=2, y=2, spawn_round=2, item_type=0, param=2)
    expiring = ItemState(id=1, x=8, y=8, spawn_round=-14, item_type=0, param=1)
    engine = _engine(
        _snake(0, 0, (1, 1)),
        _snake(1, 1, (2, 2)),
        turn=1,
        items=[spawning, expiring],
    )

    engine.step(0)
    engine.step(2)

    opponent = next(snake for snake in engine.state.snakes if snake.id == 1)
    assert engine.state.turn == 2
    assert opponent.length_bank == 2
    assert engine.state.item(0).owner_snake_id == 1
    assert engine.state.item(1).expired is True


def test_type_one_pickup_is_split_inventory_not_a_railgun() -> None:
    """Treating every non-length item as a railgun corrupts declared item-type state."""
    item = ItemState(id=0, x=5, y=4, spawn_round=9, item_type=1, param=8)
    engine = _engine(
        _snake(0, 0, (4, 4), (3, 4)),
        _snake(1, 1, (15, 0)),
        items=[item],
    )

    engine.step(0)

    snake = engine.state.snake(0)
    assert snake is not None
    assert snake.split_item_id == 0
    assert snake.railgun_item_id is None


def test_split_discards_held_type_one_item_without_requiring_it() -> None:
    """The official split partitions inventory and drops its unused type-one item."""
    item = ItemState(
        id=0,
        x=0,
        y=0,
        spawn_round=1,
        item_type=1,
        param=20,
        gotten_round=4,
        owner_snake_id=0,
    )
    engine = _engine(
        _snake(0, 0, (5, 5), (4, 5), (3, 5), (2, 5), split_item=0),
        _snake(1, 1, (15, 0)),
        items=[item],
    )

    engine.step(5)

    parent, child = engine.state.snakes[:2]
    assert parent.split_item_id is None
    assert child.split_item_id is None
    assert engine.state.item(0).expired is True


def test_split_reverses_tail_transfers_bank_and_delays_child_turn() -> None:
    """Scheduling a split child immediately or retaining the parent's bank changes a round."""
    engine = _engine(
        _snake(0, 0, (5, 5), (4, 5), (3, 5), (2, 5), bank=2),
        _snake(2, 0, (9, 9)),
        _snake(1, 1, (15, 0)),
    )

    transition = engine.step(5)

    parent, child = engine.state.snakes[:2]
    assert transition.new_snake_id == child.id == 3
    assert parent.coordinates == [(5, 5), (4, 5)]
    assert parent.length_bank == 0
    assert child.coordinates == [(2, 5), (3, 5)]
    assert child.length_bank == 2
    assert engine.current_snake.id == 2


def test_split_requires_length_and_stops_at_four_friendly_snakes() -> None:
    """Either missing split constraint permits an official illegal action."""
    short = _engine(_snake(0, 0, (1, 1)), _snake(1, 1, (15, 0)))
    crowded = _engine(
        _snake(0, 0, (5, 5), (4, 5)),
        _snake(2, 0, (7, 7)),
        _snake(3, 0, (8, 8)),
        _snake(4, 0, (9, 9)),
        _snake(1, 1, (15, 0)),
    )

    assert bool(short.legal_action_mask()[5]) is False
    assert bool(crowded.legal_action_mask()[5]) is False
    with pytest.raises(IllegalActionError, match="split"):
        crowded.step(5)


def test_fire_requires_railgun_and_length_then_clears_forward_walls() -> None:
    """Railgun must be consumed and clear the full heading ray without moving."""
    walls = np.full((16, 16), -1, dtype=np.int8)
    walls[6, 4] = 0
    walls[10, 4] = 1
    item = ItemState(
        id=9,
        x=0,
        y=0,
        spawn_round=1,
        item_type=2,
        param=512,
        gotten_round=4,
        owner_snake_id=0,
    )
    engine = _engine(
        _snake(0, 0, (4, 4), (3, 4), railgun=9),
        _snake(1, 1, (15, 0)),
        items=[item],
        walls=walls,
    )

    assert bool(engine.legal_action_mask()[4]) is True
    engine.step(4)

    assert engine.state.snakes[0].railgun_item_id is None
    assert np.all(engine.state.walls[5:, 4] == -1)
    assert engine.state.snakes[0].coordinates == [(4, 4), (3, 4)]


def test_fire_without_inventory_or_with_length_one_is_protocol_invalid() -> None:
    """A railgun operation is not merely strategically bad when prerequisites fail."""
    no_item = _engine(
        _snake(0, 0, (4, 4), (3, 4)),
        _snake(1, 1, (15, 0)),
    )
    short_with_item = _engine(
        _snake(0, 0, (4, 4), railgun=7),
        _snake(1, 1, (15, 0)),
        items=[
            ItemState(
                id=7,
                x=0,
                y=0,
                spawn_round=1,
                item_type=2,
                param=512,
                gotten_round=1,
                owner_snake_id=0,
            )
        ],
    )

    assert bool(no_item.legal_action_mask()[4]) is False
    assert bool(short_with_item.legal_action_mask()[4]) is False


def test_self_collision_solidifies_loop_and_enclosed_region() -> None:
    """Closing a loop must convert both its boundary and bounded interior to walls."""
    engine = _engine(
        _snake(
            0,
            0,
            (1, 1),
            (1, 2),
            (2, 2),
            (3, 2),
            (3, 1),
            (3, 0),
            (2, 0),
            (1, 0),
            (0, 0),
        ),
        _snake(1, 1, (15, 0)),
    )

    transition = engine.step(3)

    assert transition.solidified is True
    assert set(transition.dead_snake_ids) == {0}
    assert np.all(engine.state.walls[1:4, 0:3] == 0)
    assert engine.state.walls[2, 1] == 0


def test_enclosure_eliminates_snakes_inside_the_solidified_area() -> None:
    """Leaving an enclosed opponent alive creates snake cells beneath walls."""
    engine = _engine(
        _snake(
            0,
            0,
            (1, 1),
            (1, 2),
            (2, 2),
            (3, 2),
            (3, 1),
            (3, 0),
            (2, 0),
            (1, 0),
            (0, 0),
        ),
        _snake(1, 1, (2, 1)),
    )

    transition = engine.step(3)

    assert set(transition.dead_snake_ids) == {0, 1}
    assert engine.state.terminated is True
    assert engine.scores() == (18, 0)


def test_active_snakes_follow_snapshot_order_then_alternate_players() -> None:
    """Changing actor order lets split/death effects leak into the same player turn."""
    engine = _engine(
        _snake(0, 0, (2, 2)),
        _snake(2, 0, (5, 5)),
        _snake(1, 1, (13, 13)),
    )

    assert (engine.current_player, engine.current_snake.id) == (0, 0)
    engine.step(0)
    assert (engine.current_player, engine.current_snake.id) == (0, 2)
    engine.step(1)
    assert (engine.current_player, engine.current_snake.id) == (1, 1)
    engine.step(2)
    assert (engine.state.turn, engine.current_player, engine.current_snake.id) == (10, 0, 0)


def test_scores_count_two_per_snake_or_wall_cell_plus_first_item() -> None:
    """Scoring must use occupied cells rather than snake count or wall components."""
    walls = np.full((16, 16), -1, dtype=np.int8)
    walls[4, 4] = walls[5, 4] = 0
    walls[10, 10] = 1
    engine = _engine(
        _snake(0, 0, (1, 1), (2, 1), (3, 1)),
        _snake(1, 1, (14, 14), (13, 14)),
        walls=walls,
    )
    engine.state.first_item_player = 1

    assert engine.scores() == (10, 7)
    assert engine.score(0) == 10.0
    assert engine.score(1) == 7.0


def test_both_snakes_zero_and_max_round_end_with_research_draws_on_ties() -> None:
    """Official p0 tie metadata must not turn an equal research score into a win."""
    extinction = _engine(
        _snake(0, 0, (0, 0)),
        _snake(1, 1, (15, 15)),
        max_round=20,
    )
    extinction.step(2)
    final = extinction.step(0)

    assert final.terminated is True
    assert extinction.outcome(0) == extinction.outcome(1) == 0.0
    assert extinction.official_winner == 0

    timeout = _engine(
        _snake(0, 0, (2, 2)),
        _snake(1, 1, (13, 13)),
        turn=1,
        max_round=1,
    )
    timeout.step(0)
    timeout.step(2)

    assert timeout.state.turn == 2
    assert timeout.state.terminated is True
    assert timeout.outcome(0) == timeout.outcome(1) == 0.0
    assert timeout.official_winner == 0
