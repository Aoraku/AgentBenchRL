"""Official stateful binary-protocol process policy for SnakeGo."""

from __future__ import annotations

import math
import os
import selectors
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from rlbench.population import (
    AgentInfrastructureError,
    PopulationEntry,
    ProcessAgent,
)

from .spec import canonical_action


class SnakeGoProcessPolicy(ProcessAgent):
    """Run one official SnakeGo executable with fresh state for every game."""

    def __init__(self, entry: PopulationEntry, population_root: str | Path) -> None:
        if entry.protocol != "snakego_official":
            raise ValueError("SnakeGo process policy requires snakego_official protocol")
        super().__init__(entry, population_root)
        self._side: int | None = None
        self._pending_operation: int | None = None

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> None:
        if self._process is not None:
            return
        executable_fd, executable, first_line = self._open_verified_executable()
        try:
            command = self._fd_launch_command(executable_fd, first_line)
            self._process = subprocess.Popen(
                [*command, *self.entry.command[1:]],
                cwd=self.population_root,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                pass_fds=(executable_fd,),
            )
        except OSError as exc:
            raise AgentInfrastructureError(
                f"failed to launch {self.entry.agent_id} from {executable.name}"
            ) from exc
        finally:
            os.close(executable_fd)
        assert self._process.stderr is not None
        assert self._process.stdin is not None and self._process.stdout is not None
        os.set_blocking(self._process.stdin.fileno(), False)
        os.set_blocking(self._process.stdout.fileno(), False)
        self._stderr_thread = threading.Thread(
            target=self._capture_stderr,
            args=(self._process.stderr,),
            daemon=True,
        )
        self._stderr_thread.start()

    def begin_game(self, case: Any, agent_id: str, side: int, game: Any) -> None:
        del case, agent_id
        if side not in (0, 1):
            raise ValueError("SnakeGo side must be 0 or 1")
        self.close()
        self._side = side
        self._pending_operation = None
        self._stdout_buffer.clear()
        self.start()
        max_round = int(game.max_round)
        items = game.state.items
        if not items or len(items) > 32767:
            raise AgentInfrastructureError("official SnakeGo item count is invalid")
        startup = bytearray((16, 16))
        startup.extend(max_round.to_bytes(2, "big", signed=True))
        startup.append(side)
        startup.append(0x10)
        startup.extend(len(items).to_bytes(2, "big", signed=True))
        for item in items:
            startup.extend((item.x, item.y, item.item_type))
            startup.extend(item.spawn_round.to_bytes(2, "big", signed=True))
            startup.extend(item.param.to_bytes(2, "big", signed=True))
        self._write_request(bytes(startup), time.monotonic() + self.move_timeout)

    def act_game_process(
        self, game: Any, *, timeout_seconds: float | None = None
    ) -> int:
        del game
        if self._side is None:
            raise AgentInfrastructureError("official SnakeGo game has not started")
        timeout = self.move_timeout if timeout_seconds is None else timeout_seconds
        if (
            not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or not math.isfinite(timeout)
            or timeout <= 0.0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        response = self._read_exact(5, time.monotonic() + float(timeout))
        if response[:4] != b"\x00\x00\x00\x01" or response[4] not in range(1, 7):
            raise AgentInfrastructureError(
                "official SnakeGo action must be a five-byte framed operation"
            )
        operation = int(response[4])
        self._pending_operation = operation
        return canonical_action(operation - 1, self._side)

    def observe_action(self, game: Any, actor: int, action: int) -> None:
        del game
        if not self.running:
            return
        operation = canonical_action(action, actor) + 1
        if actor == self._side:
            if self._pending_operation != operation:
                raise AgentInfrastructureError(
                    "official SnakeGo local action does not match process output"
                )
            self._pending_operation = None
        self._write_request(bytes((operation,)), time.monotonic() + self.move_timeout)

    def end_game(self, game: Any, result: Any) -> None:
        try:
            if not self.running:
                return
            score_0 = int(round(game.score(0)))
            score_1 = int(round(game.score(1)))
            score_player_0 = getattr(result, "score_player_0", None)
            if score_player_0 == 1.0:
                winner = 0
            elif score_player_0 == 0.0:
                winner = 1
            elif score_player_0 == 0.5:
                winner = getattr(game, "official_winner", None)
                if winner not in (0, 1):
                    winner = 0 if score_0 >= score_1 else 1
            elif bool(getattr(result, "valid", False)):
                winner = getattr(game, "official_winner", None)
                if winner not in (0, 1):
                    winner = 0 if score_0 >= score_1 else 1
            else:
                winner = 0xFF
            result_type = {
                "rule_timeout": 0x10,
                "illegal_action": 0x11,
            }.get(
                getattr(result, "reason", ""),
                0x00 if bool(getattr(result, "valid", False)) else 0x20,
            )
            if result_type in (0x10, 0x11):
                if winner == 0:
                    score_1 = -100
                else:
                    score_0 = -100
            payload = bytearray((0x11, result_type, winner))
            payload.extend(score_0.to_bytes(2, "big", signed=True))
            payload.extend(score_1.to_bytes(2, "big", signed=True))
            self._write_request(bytes(payload), time.monotonic() + self.move_timeout)
            process = self._process
            if process is not None:
                try:
                    process.wait(timeout=min(0.5, self.move_timeout))
                except subprocess.TimeoutExpired:
                    pass
        finally:
            self.close()

    def close(self) -> None:
        process = self._process
        if process is not None and process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=0.05)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=1.0)
        self._process = None
        self._stderr_thread = None

    def _read_exact(self, byte_count: int, deadline: float) -> bytes:
        process = self._require_process()
        assert process.stdout is not None
        selector = selectors.DefaultSelector()
        payload = bytearray()
        try:
            selector.register(process.stdout, selectors.EVENT_READ)
            while len(payload) < byte_count:
                self._wait_ready(selector, deadline)
                try:
                    chunk = os.read(process.stdout.fileno(), byte_count - len(payload))
                except BlockingIOError:
                    continue
                except OSError as exc:
                    raise AgentInfrastructureError(
                        "failed to read official SnakeGo action"
                    ) from exc
                if not chunk:
                    raise AgentInfrastructureError(
                        "official SnakeGo process exited before sending a complete action"
                    )
                payload.extend(chunk)
        finally:
            selector.close()
        return bytes(payload)
