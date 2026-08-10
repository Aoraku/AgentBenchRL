"""Stateful adapter for the official SnakeGo agent binary protocol."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, BinaryIO, Sequence

import numpy as np

from .engine import SnakeGoEngine
from .game import SnakeGoGame
from .spec import SNAKEGO_SPEC, canonical_action
from .state import ItemState

if TYPE_CHECKING:
    from rlbench.algorithms.alphazero import AlphaZeroConfig, MCTS
    from rlbench.algorithms.ppo_tianshou import ActionMapper, PPOTrainer


INFERENCE_BUNDLE_FORMAT = "agentbench-rl-frame-alphazero-inference"
INFERENCE_BUNDLE_SCHEMA_VERSION = 1
PPO_INFERENCE_BUNDLE_FORMAT = "agentbench-rl-frame-ppo-inference"
PPO_INFERENCE_BUNDLE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class OfficialGameOver:
    result_type: int
    winner: int
    scores: tuple[int, int]


class OfficialProtocolAdapter:
    """Consume judge messages and emit official framed deterministic decisions."""

    def __init__(
        self,
        policy: Any,
        *,
        action_mapper: ActionMapper | None = None,
        policy_logit_biases: (
            tuple[tuple[float, ...], tuple[float, ...]] | None
        ) = None,
    ) -> None:
        self.policy = policy
        self.action_mapper = action_mapper
        self.policy_logit_biases = policy_logit_biases
        if policy_logit_biases is not None:
            try:
                biases = np.asarray(policy_logit_biases, dtype=np.float32)
            except ValueError as exc:
                raise ValueError(
                    "policy logit biases must have shape (2, action_count)"
                ) from exc
            if biases.shape != (2, len(SNAKEGO_SPEC.action_names)):
                raise ValueError("policy logit biases must have shape (2, action_count)")
            if not np.all(np.isfinite(biases)):
                raise ValueError("policy logit biases must be finite")
            self.policy_logit_biases = (
                tuple(float(value) for value in biases[0]),
                tuple(float(value) for value in biases[1]),
            )
        self.local_player: int | None = None
        self.max_round: int | None = None
        self.game: SnakeGoGame | None = None
        self.awaiting_echo: int | None = None
        self.policy_state: object | None = None
        self.game_over: OfficialGameOver | None = None
        self._action_mapper_started = False

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
            actor = self.game.current_player()
            self._apply_action(canonical_action(operation - 1, actor))
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
        begin_mapping = getattr(self.action_mapper, "begin_game", None)
        if callable(begin_mapping):
            assert self.local_player is not None
            try:
                begin_mapping(
                    None,
                    "action-mapper",
                    self.local_player,
                    self.game,
                )
            except BaseException:
                self.close()
                raise
            self._action_mapper_started = True

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
        self._finish_action_mapper()

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
        self._apply_action(action)
        operation = absolute_action + 1
        self.awaiting_echo = operation
        return b"\x00\x00\x00\x01" + bytes((operation,))

    def _apply_action(self, action: int) -> None:
        assert self.game is not None
        record = self.game.step(action)
        observe_mapping = getattr(self.action_mapper, "observe_action", None)
        if self._action_mapper_started and callable(observe_mapping):
            observe_mapping(self.game, record.player, record.action)

    def _finish_action_mapper(self) -> None:
        if not self._action_mapper_started:
            return
        assert self.game is not None and self.game_over is not None
        if self.game_over.winner == 0:
            score_player_0 = 1.0
        elif self.game_over.winner == 1:
            score_player_0 = 0.0
        else:
            score_player_0 = 0.5
        result = SimpleNamespace(
            valid=self.game_over.result_type != 0x20,
            reason={0x10: "rule_timeout", 0x11: "illegal_action"}.get(
                self.game_over.result_type,
                "completed",
            ),
            score_player_0=score_player_0,
        )
        end_game = getattr(self.action_mapper, "end_game", None)
        try:
            if callable(end_game):
                end_game(self.game, result)
        finally:
            self._action_mapper_started = False

    def close(self) -> None:
        close_mapper = getattr(self.action_mapper, "close", None)
        try:
            if callable(close_mapper):
                close_mapper()
        finally:
            self._action_mapper_started = False

    def _policy_action(self) -> int:
        assert self.game is not None
        observation = self.game.observe(self.game.current_player())
        legal_mask = self.game.legal_action_mask()
        search = getattr(self.policy, "search", None)
        if callable(search):
            result = search(
                self.game,
                training=False,
                move_number=self.game.state.action_count,
            )
            return int(result.action)
        select_step = getattr(self.policy, "select_action_step", None)
        training_action_mask = getattr(self.game, "training_action_mask", None)
        mapping = self._residual_action_mapping()
        if mapping is not None:
            mask = np.zeros(len(legal_mask), dtype=np.bool_)
            mask[: len(mapping)] = True
        else:
            mask = (
                training_action_mask(self.game.current_player())
                if callable(training_action_mask)
                else legal_mask
            )
        if callable(select_step):
            selection_arguments = {
                "deterministic": True,
                "state": self.policy_state,
            }
            if self.policy_logit_biases is not None:
                selection_arguments["logit_bias"] = np.asarray(
                    self.policy_logit_biases[self.game.current_player()],
                    dtype=np.float32,
                )
            decision = select_step(
                observation,
                mask.copy(),
                **selection_arguments,
            )
            self.policy_state = decision.state
            return self._mapped_policy_action(mapping, int(decision.action))
        select = getattr(self.policy, "select_action", None)
        if callable(select):
            action = int(select(observation, mask.copy(), deterministic=True))
            return self._mapped_policy_action(mapping, action)
        if callable(self.policy):
            action = int(self.policy(observation, mask.copy()))
            return self._mapped_policy_action(mapping, action)
        raise TypeError("policy must be callable or expose deterministic selection")

    def _residual_action_mapping(self) -> tuple[int, ...] | None:
        if self.action_mapper is None:
            return None
        assert self.game is not None
        player = self.game.current_player()
        mapping = tuple(
            int(action) for action in self.action_mapper(self.game, player)
        )
        legal_mask = self.game.legal_action_mask()
        if (
            not mapping
            or len(mapping) > len(legal_mask)
            or len(set(mapping)) != len(mapping)
            or any(
                action < 0
                or action >= len(legal_mask)
                or not bool(legal_mask[action])
                for action in mapping
            )
        ):
            raise ValueError("deployment action mapping must be unique and legal")
        return mapping

    @staticmethod
    def _mapped_policy_action(
        mapping: tuple[int, ...] | None,
        policy_action: int,
    ) -> int:
        if mapping is None:
            return policy_action
        if policy_action < 0 or policy_action >= len(mapping):
            raise ValueError("policy emitted an illegal residual action")
        return mapping[policy_action]


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


def load_ppo_policy(
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
    seed: int = 0,
    action_mapper: ActionMapper | None = None,
    action_mapper_id: str | None = None,
) -> PPOTrainer:
    """Restore PPO actor-critic weights as a deterministic deployment policy."""
    from rlbench.algorithms import PolicyCheckpoint
    from rlbench.algorithms.ppo_tianshou import PPOConfig, PPOTrainer

    checkpoint = PolicyCheckpoint.load(checkpoint_path, map_location="cpu")
    raw_config = checkpoint.trainer_state.get("ppo_config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("PPO checkpoint is missing its configuration")
    config = replace(PPOConfig(**dict(raw_config)), device=device)
    trainer = PPOTrainer(
        lambda: SnakeGoGame({"max_round": 512}),
        config,
        seed=seed,
        action_mapper=action_mapper,
        action_mapper_id=action_mapper_id,
    )
    if checkpoint.trainer_state.get("action_mapper_id") != action_mapper_id:
        raise ValueError("PPO checkpoint action mapper does not match deployment")
    checkpoint.validate_restore(model=trainer.network)
    checkpoint.restore(model=trainer.network)
    trainer.network.eval()
    return trainer


def export_alphazero_inference_bundle(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    config: AlphaZeroConfig,
) -> Path:
    """Strip optimizer, replay, trainer, and RNG state from a SnakeGo checkpoint."""
    import torch

    from rlbench.algorithms import PolicyCheckpoint
    from rlbench.algorithms.alphazero import PolicyValueNet

    network = PolicyValueNet.from_game_spec(SNAKEGO_SPEC, config, device="cpu")
    PolicyCheckpoint.load(checkpoint_path, map_location="cpu").restore(model=network)
    inference_config = replace(
        config,
        root_dirichlet_fraction=0.0,
        self_play_temperature=0.0,
        temperature_moves=0,
        mixed_precision=False,
        device="cpu",
    )
    payload = {
        "format": INFERENCE_BUNDLE_FORMAT,
        "schema_version": INFERENCE_BUNDLE_SCHEMA_VERSION,
        "game_spec": {
            "name": SNAKEGO_SPEC.name,
            "action_names": SNAKEGO_SPEC.action_names,
            "plane_names": SNAKEGO_SPEC.observation_spec.plane_names,
            "scalar_names": SNAKEGO_SPEC.observation_spec.scalar_names,
        },
        "config": asdict(inference_config),
        "model_state": {
            name: tensor.detach().cpu()
            for name, tensor in network.state_dict().items()
        },
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary_name)
        with open(temporary_name, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def export_ppo_inference_bundle(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    action_mapper: ActionMapper | None = None,
    action_mapper_id: str | None = None,
) -> Path:
    """Strip PPO optimizer, snapshots, counters, and RNG state for deployment."""
    import torch

    trainer = load_ppo_policy(
        checkpoint_path,
        device="cpu",
        action_mapper=action_mapper,
        action_mapper_id=action_mapper_id,
    )
    payload = {
        "format": PPO_INFERENCE_BUNDLE_FORMAT,
        "schema_version": PPO_INFERENCE_BUNDLE_SCHEMA_VERSION,
        "game_spec": {
            "name": SNAKEGO_SPEC.name,
            "action_names": SNAKEGO_SPEC.action_names,
            "plane_names": SNAKEGO_SPEC.observation_spec.plane_names,
            "scalar_names": SNAKEGO_SPEC.observation_spec.scalar_names,
        },
        "config": asdict(trainer.config),
        "action_mapper_id": trainer.action_mapper_id,
        "model_state": {
            name: tensor.detach().cpu()
            for name, tensor in trainer.network.state_dict().items()
        },
    }
    return _write_inference_bundle(payload, output_path, torch_module=torch)


def load_alphazero_inference_bundle(
    bundle_path: str | Path,
    *,
    device: str = "cpu",
    seed: int = 0,
    simulations: int | None = None,
    c_puct: float | None = None,
    inference_batch_size: int | None = None,
) -> MCTS:
    """Restore a compact, schema-bound SnakeGo inference bundle."""
    import torch

    from rlbench.algorithms.alphazero import AlphaZeroConfig, MCTS, PolicyValueNet

    try:
        payload = torch.load(bundle_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(f"invalid inference bundle: {bundle_path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("inference bundle must contain a mapping")
    if payload.get("format") != INFERENCE_BUNDLE_FORMAT:
        raise ValueError("unsupported inference bundle format")
    if payload.get("schema_version") != INFERENCE_BUNDLE_SCHEMA_VERSION:
        raise ValueError("unsupported inference bundle schema version")
    _validate_inference_game_spec(payload.get("game_spec"))
    raw_config = payload.get("config")
    model_state = payload.get("model_state")
    if not isinstance(raw_config, Mapping) or not isinstance(model_state, Mapping):
        raise ValueError("inference bundle is missing config or model state")
    embedded = AlphaZeroConfig(**dict(raw_config))
    config = replace(
        embedded,
        simulations=embedded.simulations if simulations is None else simulations,
        c_puct=embedded.c_puct if c_puct is None else c_puct,
        inference_batch_size=(
            embedded.inference_batch_size
            if inference_batch_size is None
            else inference_batch_size
        ),
        root_dirichlet_fraction=0.0,
        self_play_temperature=0.0,
        temperature_moves=0,
        mixed_precision=False,
        device=device,
    )
    network = PolicyValueNet.from_game_spec(SNAKEGO_SPEC, config, device=device)
    try:
        network.load_state_dict(dict(model_state), strict=True)
    except Exception as exc:
        raise ValueError("inference bundle model state does not match its schema") from exc
    network.eval()
    return MCTS(config, network, seed=seed)


def load_ppo_inference_bundle(
    bundle_path: str | Path,
    *,
    device: str = "cpu",
    seed: int = 0,
    action_mapper: ActionMapper | None = None,
    action_mapper_id: str | None = None,
) -> PPOTrainer:
    """Restore a compact PPO bundle, including recurrent deployment state."""
    import torch

    from rlbench.algorithms.ppo_tianshou import PPOConfig, PPOTrainer

    payload = _read_inference_bundle(bundle_path, torch_module=torch)
    if payload.get("format") != PPO_INFERENCE_BUNDLE_FORMAT:
        raise ValueError("unsupported PPO inference bundle format")
    if payload.get("schema_version") != PPO_INFERENCE_BUNDLE_SCHEMA_VERSION:
        raise ValueError("unsupported PPO inference bundle schema version")
    _validate_inference_game_spec(payload.get("game_spec"))
    raw_config = payload.get("config")
    model_state = payload.get("model_state")
    if not isinstance(raw_config, Mapping) or not isinstance(model_state, Mapping):
        raise ValueError("PPO inference bundle is missing config or model state")
    if payload.get("action_mapper_id") != action_mapper_id:
        raise ValueError("PPO inference action mapper does not match its bundle")
    config = replace(PPOConfig(**dict(raw_config)), device=device)
    trainer = PPOTrainer(
        lambda: SnakeGoGame({"max_round": 512}),
        config,
        seed=seed,
        action_mapper=action_mapper,
        action_mapper_id=action_mapper_id,
    )
    try:
        trainer.network.load_state_dict(dict(model_state), strict=True)
    except Exception as exc:
        raise ValueError("PPO inference model state does not match its schema") from exc
    trainer.network.eval()
    return trainer


def _read_inference_bundle(
    bundle_path: str | Path, *, torch_module: Any
) -> Mapping[str, Any]:
    try:
        payload = torch_module.load(
            bundle_path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as exc:
        raise ValueError(f"invalid inference bundle: {bundle_path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("inference bundle must contain a mapping")
    return payload


def _write_inference_bundle(
    payload: Mapping[str, Any],
    output_path: str | Path,
    *,
    torch_module: Any,
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    try:
        torch_module.save(dict(payload), temporary_name)
        with open(temporary_name, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def _validate_inference_game_spec(raw: object) -> None:
    if not isinstance(raw, Mapping):
        raise ValueError("inference bundle has no game schema")
    expected = {
        "name": SNAKEGO_SPEC.name,
        "action_names": tuple(SNAKEGO_SPEC.action_names),
        "plane_names": tuple(SNAKEGO_SPEC.observation_spec.plane_names),
        "scalar_names": tuple(SNAKEGO_SPEC.observation_spec.scalar_names),
    }
    actual = {
        "name": raw.get("name"),
        "action_names": tuple(raw.get("action_names", ())),
        "plane_names": tuple(raw.get("plane_names", ())),
        "scalar_names": tuple(raw.get("scalar_names", ())),
    }
    if actual != expected:
        raise ValueError("inference bundle game schema does not match SnakeGo")


def run_official_agent(
    policy: Any,
    *,
    action_mapper: ActionMapper | None = None,
    policy_logit_biases: tuple[tuple[float, ...], tuple[float, ...]] | None = None,
    input_stream: BinaryIO | None = None,
    output_stream: BinaryIO | None = None,
) -> OfficialGameOver:
    """Run one official SnakeGo game over exact unframed stdin/stdout messages."""
    source = input_stream if input_stream is not None else sys.stdin.buffer
    destination = output_stream if output_stream is not None else sys.stdout.buffer
    adapter = OfficialProtocolAdapter(
        policy,
        action_mapper=action_mapper,
        policy_logit_biases=policy_logit_biases,
    )
    try:
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
    finally:
        adapter.close()


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
    """Load an AlphaZero or PPO policy and serve the official binary protocol."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--bundle", type=Path)
    parser.add_argument("--simulations", type=int)
    parser.add_argument("--c-puct", type=float)
    parser.add_argument("--channels", type=int)
    parser.add_argument("--residual-blocks", type=int)
    parser.add_argument("--inference-batch-size", type=int)
    parser.add_argument("--algorithm", choices=("alphazero", "ppo"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args(argv)

    if arguments.bundle is not None:
        import torch

        payload = _read_inference_bundle(arguments.bundle, torch_module=torch)
        bundle_format = payload.get("format")
        if bundle_format == PPO_INFERENCE_BUNDLE_FORMAT:
            if arguments.algorithm not in (None, "ppo"):
                parser.error("bundle contains PPO but --algorithm requests AlphaZero")
            policy = load_ppo_inference_bundle(
                arguments.bundle,
                device=arguments.device,
                seed=arguments.seed,
            )
        elif bundle_format == INFERENCE_BUNDLE_FORMAT:
            if arguments.algorithm not in (None, "alphazero"):
                parser.error("bundle contains AlphaZero but --algorithm requests PPO")
            policy = load_alphazero_inference_bundle(
                arguments.bundle,
                device=arguments.device,
                seed=arguments.seed,
                simulations=arguments.simulations,
                c_puct=arguments.c_puct,
                inference_batch_size=arguments.inference_batch_size,
            )
        else:
            parser.error("bundle has an unsupported policy format")
    else:
        algorithm = arguments.algorithm or "alphazero"
        if algorithm == "ppo":
            policy = load_ppo_policy(
                arguments.checkpoint,
                device=arguments.device,
                seed=arguments.seed,
            )
        else:
            from rlbench.algorithms.alphazero import AlphaZeroConfig

            if arguments.channels is None or arguments.residual_blocks is None:
                parser.error("--checkpoint requires --channels and --residual-blocks")
            config = AlphaZeroConfig(
                simulations=(
                    64 if arguments.simulations is None else arguments.simulations
                ),
                c_puct=1.5 if arguments.c_puct is None else arguments.c_puct,
                root_dirichlet_fraction=0.0,
                self_play_temperature=0.0,
                temperature_moves=0,
                channels=arguments.channels,
                residual_blocks=arguments.residual_blocks,
                mixed_precision=False,
                inference_batch_size=(
                    32
                    if arguments.inference_batch_size is None
                    else arguments.inference_batch_size
                ),
            )
            policy = load_alphazero_policy(
                arguments.checkpoint,
                config=config,
                device=arguments.device,
                seed=arguments.seed,
            )
    run_official_agent(policy)
    return 0
