"""Stateful adapter for the official SnakeGo agent binary protocol."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Sequence

import numpy as np

from .engine import SnakeGoEngine
from .game import SnakeGoGame
from .spec import SNAKEGO_SPEC, canonical_action
from .state import ItemState

if TYPE_CHECKING:
    from rlbench.algorithms.alphazero import AlphaZeroConfig, MCTS


@dataclass(frozen=True, slots=True)
class OfficialGameOver:
    result_type: int
    winner: int
    scores: tuple[int, int]


class OfficialProtocolAdapter:
    """Consume judge messages and emit official framed deterministic decisions."""

    def __init__(self, policy: Any) -> None:
        self.policy = policy
        self.local_player: int | None = None
        self.max_round: int | None = None
        self.game: SnakeGoGame | None = None
        self.awaiting_echo: int | None = None
        self.policy_state: object | None = None
        self.game_over: OfficialGameOver | None = None

    def consume(self, message: bytes) -> bytes | None:
        if self.local_player is None:
            self._consume_config(message)
            return None
        if self.game is None:
            self._consume_items(message)
            return self._maybe_choose()
        if not message:
            raise ValueError("official protocol message must not be empty")
        if message[0] == 0x11:
            self._consume_game_over(message)
            return None
        if len(message) != 1 or not 1 <= message[0] <= 6:
            raise ValueError("invalid official operation message")
        operation = int(message[0])
        if self.awaiting_echo is not None:
            if operation != self.awaiting_echo:
                raise ValueError("official operation echo does not match emitted action")
            self.awaiting_echo = None
        else:
            if self.game.current_player() == self.local_player:
                raise ValueError("received an opponent operation during the local turn")
            self.game.engine.step(operation - 1)
        return self._maybe_choose()

    def _consume_config(self, message: bytes) -> None:
        if len(message) != 5:
            raise ValueError("official config message must contain five bytes")
        length, width = message[0], message[1]
        if (length, width) != (16, 16):
            raise ValueError("SnakeGo protocol requires a 16 by 16 board")
        max_round = int.from_bytes(message[2:4], "big", signed=True)
        player = int(message[4])
        if not 1 <= max_round <= 512 or player not in (0, 1):
            raise ValueError("invalid official SnakeGo config")
        self.max_round = max_round
        self.local_player = player

    def _consume_items(self, message: bytes) -> None:
        if len(message) < 3 or message[0] != 0x10:
            raise ValueError("official item message must start with 0x10")
        item_count = int.from_bytes(message[1:3], "big", signed=True)
        if item_count <= 0 or len(message) != 3 + 7 * item_count:
            raise ValueError("invalid official item message length")
        items: list[ItemState] = []
        offset = 3
        for item_id in range(item_count):
            x, y, item_type = message[offset : offset + 3]
            spawn_round = int.from_bytes(message[offset + 3 : offset + 5], "big")
            param = int.from_bytes(message[offset + 5 : offset + 7], "big")
            if x >= 16 or y >= 16 or item_type not in (0, 1, 2):
                raise ValueError("invalid official item entry")
            items.append(
                ItemState(item_id, x, y, spawn_round, item_type, param)
            )
            offset += 7
        assert self.max_round is not None
        self.game = SnakeGoGame.from_engine(
            SnakeGoEngine.from_items(items, max_round=self.max_round)
        )

    def _consume_game_over(self, message: bytes) -> None:
        if len(message) != 7:
            raise ValueError("official game-over message must contain seven bytes")
        self.game_over = OfficialGameOver(
            result_type=message[1],
            winner=message[2],
            scores=(
                int.from_bytes(message[3:5], "big", signed=True),
                int.from_bytes(message[5:7], "big", signed=True),
            ),
        )

    def _maybe_choose(self) -> bytes | None:
        if (
            self.game is None
            or self.game.state.terminated
            or self.game.current_player() != self.local_player
            or self.awaiting_echo is not None
        ):
            return None
        action = self._policy_action()
        mask = self.game.legal_action_mask()
        if not isinstance(action, int) or not 0 <= action < 6 or not bool(mask[action]):
            raise ValueError("policy emitted an illegal SnakeGo action")
        absolute_action = canonical_action(action, self.local_player)
        self.game.engine.step(absolute_action)
        operation = absolute_action + 1
        self.awaiting_echo = operation
        return b"\x00\x00\x00\x01" + bytes((operation,))

    def _policy_action(self) -> int:
        assert self.game is not None
        observation = self.game.observe(self.game.current_player())
        mask = self.game.legal_action_mask()
        search = getattr(self.policy, "search", None)
        if callable(search):
            result = search(
                self.game,
                training=False,
                move_number=self.game.state.action_count,
            )
            return int(result.action)
        select_step = getattr(self.policy, "select_action_step", None)
        if callable(select_step):
            decision = select_step(
                observation,
                mask.copy(),
                deterministic=True,
                state=self.policy_state,
            )
            self.policy_state = decision.state
            return int(decision.action)
        select = getattr(self.policy, "select_action", None)
        if callable(select):
            return int(select(observation, mask.copy(), deterministic=True))
        if callable(self.policy):
            return int(self.policy(observation, mask.copy()))
        raise TypeError("policy must be callable or expose deterministic selection")


def load_alphazero_policy(
    checkpoint_path: str | Path,
    *,
    config: AlphaZeroConfig,
    device: str = "cpu",
    seed: int = 0,
) -> MCTS:
    """Restore a framework AlphaZero checkpoint as a deterministic search policy."""
    from rlbench.algorithms import PolicyCheckpoint
    from rlbench.algorithms.alphazero import MCTS, PolicyValueNet

    network = PolicyValueNet.from_game_spec(SNAKEGO_SPEC, config, device=device)
    PolicyCheckpoint.load(checkpoint_path, map_location=device).restore(model=network)
    network.eval()
    return MCTS(config, network, seed=seed)


def run_official_agent(
    policy: Any,
    *,
    input_stream: BinaryIO | None = None,
    output_stream: BinaryIO | None = None,
) -> OfficialGameOver:
    """Run one official SnakeGo game over exact unframed stdin/stdout messages."""
    source = input_stream if input_stream is not None else sys.stdin.buffer
    destination = output_stream if output_stream is not None else sys.stdout.buffer
    adapter = OfficialProtocolAdapter(policy)

    adapter.consume(_read_exact(source, 5))
    item_header = _read_exact(source, 3)
    if item_header[0] != 0x10:
        raise ValueError("official item message must start with 0x10")
    item_count = int.from_bytes(item_header[1:3], "big", signed=True)
    if item_count <= 0:
        raise ValueError("official item count must be positive")
    outgoing = adapter.consume(item_header + _read_exact(source, 7 * item_count))
    _write_decision(destination, outgoing)

    while adapter.game_over is None:
        marker = _read_exact(source, 1)
        message = marker + _read_exact(source, 6) if marker == b"\x11" else marker
        outgoing = adapter.consume(message)
        _write_decision(destination, outgoing)

    return adapter.game_over


def _read_exact(stream: BinaryIO, byte_count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = byte_count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError(
                f"official SnakeGo stream ended with {remaining} bytes outstanding"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _write_decision(stream: BinaryIO, decision: bytes | None) -> None:
    if decision is None:
        return
    stream.write(decision)
    stream.flush()


def main(argv: Sequence[str] | None = None) -> int:
    """Load an AlphaZero checkpoint and serve the official binary protocol."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--channels", type=int, required=True)
    parser.add_argument("--residual-blocks", type=int, required=True)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args(argv)

    from rlbench.algorithms.alphazero import AlphaZeroConfig

    config = AlphaZeroConfig(
        simulations=arguments.simulations,
        c_puct=arguments.c_puct,
        root_dirichlet_fraction=0.0,
        self_play_temperature=0.0,
        temperature_moves=0,
        channels=arguments.channels,
        residual_blocks=arguments.residual_blocks,
        mixed_precision=False,
        inference_batch_size=arguments.inference_batch_size,
    )
    policy = load_alphazero_policy(
        arguments.checkpoint,
        config=config,
        device=arguments.device,
        seed=arguments.seed,
    )
    run_official_agent(policy)
    return 0
