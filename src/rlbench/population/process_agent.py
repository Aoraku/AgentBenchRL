"""Line-delimited JSON subprocess boundary for external agents."""

from __future__ import annotations

import hashlib
import json
import math
import os
import selectors
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, BinaryIO, Mapping

from .manifest import PopulationEntry


_PROCESS_CLEANUP_SECONDS = 0.05


class AgentInfrastructureError(RuntimeError):
    """The agent process or protocol failed independently of game rules."""


class ProcessMoveTimeout(TimeoutError):
    """The process exceeded its rule-defined per-move deadline."""


class ProcessAgent:
    """A lazily started external policy using one JSON object per line."""

    def __init__(self, entry: PopulationEntry, population_root: str | Path) -> None:
        self.entry = entry
        self.population_root = Path(population_root).resolve()
        if self.population_root != entry.population_root.resolve():
            raise AgentInfrastructureError(
                "process population root differs from the manifest population root"
            )
        self._validate_executable()
        timeout = entry.resource_limits.get("move_seconds", 1.0)
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("move_seconds must be positive")
        self.move_timeout = float(timeout)
        self._process: subprocess.Popen[bytes] | None = None
        self._stderr_parts: list[str] = []
        self._stderr_thread: threading.Thread | None = None
        self._stdout_buffer = bytearray()

    @property
    def stderr(self) -> str:
        return "".join(self._stderr_parts)

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

    def act(
        self,
        request: Mapping[str, Any],
        *,
        timeout_seconds: float | None = None,
    ) -> int:
        self.start()
        process = self._require_process()
        if process.poll() is not None:
            raise AgentInfrastructureError(
                f"agent process exited with status {process.returncode}"
            )
        timeout = self.move_timeout if timeout_seconds is None else timeout_seconds
        if (
            not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or not math.isfinite(timeout)
            or timeout <= 0.0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        deadline = time.monotonic() + float(timeout)
        try:
            encoded = (
                json.dumps(
                    dict(request),
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise AgentInfrastructureError("failed to encode agent request") from exc
        self._write_request(encoded, deadline)
        line = self._read_response_line(deadline)
        try:
            response = json.loads(line.decode("utf-8"))
            action = response["action"]
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise AgentInfrastructureError("agent response must contain JSON action") from exc
        if not isinstance(action, int) or isinstance(action, bool):
            raise AgentInfrastructureError("agent action must be an integer")
        return action

    def close(self) -> None:
        self._stop_process()
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=1.0)

    def __enter__(self) -> ProcessAgent:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _write_request(self, payload: bytes, deadline: float) -> None:
        process = self._require_process()
        assert process.stdin is not None
        view = memoryview(payload)
        selector = selectors.DefaultSelector()
        try:
            selector.register(process.stdin, selectors.EVENT_WRITE)
            while view:
                self._wait_ready(selector, deadline)
                try:
                    written = os.write(process.stdin.fileno(), view)
                except BlockingIOError:
                    continue
                except (BrokenPipeError, OSError) as exc:
                    raise AgentInfrastructureError(
                        "failed to send agent request"
                    ) from exc
                if written == 0:
                    raise AgentInfrastructureError("agent stdin closed during request")
                view = view[written:]
        finally:
            selector.close()

    def _read_response_line(self, deadline: float) -> bytes:
        process = self._require_process()
        assert process.stdout is not None
        selector = selectors.DefaultSelector()
        try:
            selector.register(process.stdout, selectors.EVENT_READ)
            while True:
                newline = self._stdout_buffer.find(b"\n")
                if newline >= 0:
                    line = bytes(self._stdout_buffer[:newline])
                    del self._stdout_buffer[: newline + 1]
                    return line
                self._wait_ready(selector, deadline)
                try:
                    chunk = os.read(process.stdout.fileno(), 65_536)
                except BlockingIOError:
                    continue
                except OSError as exc:
                    raise AgentInfrastructureError("failed to read agent response") from exc
                if not chunk:
                    raise AgentInfrastructureError(
                        f"agent process exited with status {process.poll()} before replying"
                    )
                self._stdout_buffer.extend(chunk)
        finally:
            selector.close()

    def _wait_ready(self, selector: selectors.BaseSelector, deadline: float) -> None:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0 or not selector.select(remaining):
            self._kill_process()
            raise ProcessMoveTimeout(
                f"agent {self.entry.agent_id} exceeded its move deadline"
            )

    def _capture_stderr(self, stream: BinaryIO) -> None:
        for part in iter(stream.readline, b""):
            self._stderr_parts.append(part.decode("utf-8", errors="replace"))

    def _validate_executable(self) -> Path:
        executable_fd, executable, _ = self._open_verified_executable()
        os.close(executable_fd)
        return executable

    def _open_verified_executable(self) -> tuple[int, Path, bytes]:
        relative = Path(self.entry.command[0])
        if relative.is_absolute():
            raise AgentInfrastructureError(
                "command must be relative to the population root"
            )
        executable = (self.population_root / relative).resolve()
        if not executable.is_relative_to(self.population_root):
            raise AgentInfrastructureError("command escapes the population root")
        try:
            executable_fd = os.open(executable, os.O_RDONLY)
        except OSError as exc:
            raise AgentInfrastructureError("agent executable is not readable") from exc
        digest_builder = hashlib.sha256()
        try:
            while chunk := os.read(executable_fd, 1024 * 1024):
                digest_builder.update(chunk)
            os.lseek(executable_fd, 0, os.SEEK_SET)
            first_line = os.pread(executable_fd, 4096, 0).split(b"\n", 1)[0]
        except OSError as exc:
            os.close(executable_fd)
            raise AgentInfrastructureError("agent executable cannot be hashed") from exc
        digest = digest_builder.hexdigest()
        if f"sha256:{digest}" != self.entry.content_hash:
            os.close(executable_fd)
            raise AgentInfrastructureError("agent executable content hash changed")
        return executable_fd, executable, first_line

    @staticmethod
    def _fd_launch_command(executable_fd: int, first_line: bytes) -> list[str]:
        proc_path = Path(f"/proc/self/fd/{executable_fd}")
        if proc_path.exists():
            return [str(proc_path)]
        dev_path = Path(f"/dev/fd/{executable_fd}")
        if dev_path.exists():
            if sys.platform == "darwin":
                if not first_line.startswith(b"#!"):
                    raise AgentInfrastructureError(
                        "Darwin fd launch requires a script shebang"
                    )
                try:
                    interpreter = shlex.split(first_line[2:].decode("utf-8"))
                except (UnicodeDecodeError, ValueError) as exc:
                    raise AgentInfrastructureError("invalid agent script shebang") from exc
                if not interpreter:
                    raise AgentInfrastructureError("invalid agent script shebang")
                return [*interpreter, str(dev_path)]
            return [str(dev_path)]
        raise AgentInfrastructureError("platform does not expose inherited file descriptors")

    def _require_process(self) -> subprocess.Popen[bytes]:
        if self._process is None:
            raise AgentInfrastructureError("agent process is not running")
        return self._process

    def _stop_process(self) -> None:
        if self._process is None:
            return
        if self._process.stdin is not None:
            try:
                self._process.stdin.close()
            except OSError:
                pass
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()

    def _kill_process(self) -> None:
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.kill()
        try:
            self._process.wait(timeout=_PROCESS_CLEANUP_SECONDS)
        except subprocess.TimeoutExpired:
            threading.Thread(
                target=self._process.wait,
                daemon=True,
            ).start()
