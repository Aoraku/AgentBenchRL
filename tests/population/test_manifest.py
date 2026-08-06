from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import pytest

import rlbench.population.process_agent as process_agent_module
from rlbench.population import (
    AgentInfrastructureError,
    PopulationManifest,
    ProcessAgent,
    ProcessMoveTimeout,
)


def _sha256(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _write_manifest(path: Path, entries: str) -> Path:
    path.write_text(
        "population_root: agents\n"
        "protocol_version: '1'\n"
        "agents:\n"
        f"{entries}",
        encoding="utf-8",
    )
    return path


def test_manifest_is_immutable_content_hashed_and_training_safe(tmp_path: Path) -> None:
    """Mutable entries, unstable hashes, or test selection break frozen populations."""
    agents = tmp_path / "agents"
    agents.mkdir()
    train_payload = b"#!/bin/sh\n"
    test_payload = b"#!/bin/sh\nexit 1\n"
    (agents / "train-agent").write_bytes(train_payload)
    (agents / "test-agent").write_bytes(test_payload)
    manifest_path = _write_manifest(
        tmp_path / "population.yaml",
        "  - agent_id: train-human\n"
        "    kind: train_human\n"
        f"    content_hash: {_sha256(train_payload)}\n"
        "    command: [train-agent]\n"
        "    roles: [player]\n"
        "    resource_limits: {move_seconds: 0.25}\n"
        "    provenance: {author: fixture}\n"
        "  - agent_id: held-out-human\n"
        "    kind: test_human\n"
        f"    content_hash: {_sha256(test_payload)}\n"
        "    command: [test-agent]\n"
        "    roles: [player]\n"
        "    resource_limits: {move_seconds: 0.25}\n"
        "    provenance: {author: fixture}\n",
    )

    first = PopulationManifest.from_yaml(manifest_path)
    second = PopulationManifest.from_yaml(manifest_path)

    assert first.content_hash == second.content_hash
    assert first.population_root == agents.resolve()
    assert [entry.agent_id for entry in first.training_entries()] == ["train-human"]
    assert first.entry("held-out-human").kind == "test_human"
    with pytest.raises((AttributeError, TypeError)):
        first.entries[0].content_hash = "sha256:mutable"  # type: ignore[misc]
    with pytest.raises(TypeError):
        first.entries[0].resource_limits["move_seconds"] = 99  # type: ignore[index]


def test_manifest_rejects_duplicate_ids_hash_mismatch_and_escaping_paths(
    tmp_path: Path,
) -> None:
    """Dropping identity, byte-integrity, or root-boundary checks admits ambiguity."""
    agents = tmp_path / "agents"
    agents.mkdir()
    (agents / "agent").write_text("agent", encoding="utf-8")
    valid_entry = (
        "  - agent_id: duplicate\n"
        "    kind: baseline\n"
        f"    content_hash: {_sha256(b'agent')}\n"
        "    command: [agent]\n"
    )

    with pytest.raises(ValueError, match="duplicate agent_id"):
        PopulationManifest.from_yaml(
            _write_manifest(tmp_path / "duplicate.yaml", valid_entry + valid_entry)
        )

    mismatch = valid_entry.replace(_sha256(b"agent"), "sha256:" + "0" * 64)
    with pytest.raises(ValueError, match="content hash"):
        PopulationManifest.from_yaml(
            _write_manifest(tmp_path / "mismatch.yaml", mismatch)
        )

    escaping = valid_entry.replace("command: [agent]", "command: [../outside]")
    with pytest.raises(ValueError, match="population root"):
        PopulationManifest.from_yaml(
            _write_manifest(tmp_path / "escaping.yaml", escaping)
        )


def test_manifest_deep_freezes_mutable_values_nested_inside_tuples(
    tmp_path: Path,
) -> None:
    """A tuple must not hide mutable descendants from the frozen manifest snapshot."""
    root = tmp_path / "agents"
    root.mkdir()
    script = root / "agent"
    script.write_bytes(b"agent")
    nested = {"layers": ({"values": ["original"]},)}
    manifest = PopulationManifest.from_data(
        {
            "agents": [
                {
                    "agent_id": "nested",
                    "kind": "baseline",
                    "content_hash": _sha256(b"agent"),
                    "command": ["agent"],
                    "resource_limits": nested,
                }
            ]
        },
        population_root=root,
    )
    content_hash = manifest.content_hash

    nested["layers"][0]["values"].append("external-mutation")

    frozen_layer = manifest.entries[0].resource_limits["layers"][0]
    assert frozen_layer["values"] == ("original",)
    assert manifest.content_hash == content_hash
    with pytest.raises(TypeError):
        frozen_layer["new"] = "mutation"  # type: ignore[index]


def test_process_agent_exchanges_json_lines_and_captures_stderr(tmp_path: Path) -> None:
    """Using a non-JSON protocol or discarding diagnostics breaks process isolation."""
    root = tmp_path / "agents"
    root.mkdir()
    script = root / "agent.py"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "for line in sys.stdin:\n"
        "    request = json.loads(line)\n"
        "    print('handled:' + request['case_id'], file=sys.stderr, flush=True)\n"
        "    print(json.dumps({'action': request['legal_actions'][-1]}), flush=True)\n",
        encoding="utf-8",
    )
    os.chmod(script, 0o755)
    manifest = PopulationManifest.from_data(
        {
            "population_root": ".",
            "agents": [
                {
                    "agent_id": "process",
                    "kind": "baseline",
                    "content_hash": _sha256(script.read_bytes()),
                    "command": ["agent.py"],
                    "resource_limits": {"move_seconds": 1.0},
                }
            ],
        },
        population_root=root,
    )

    with ProcessAgent(manifest.entry("process"), root) as agent:
        assert agent.act({"case_id": "case-7", "legal_actions": [0, 2]}) == 2
        agent.close()
        assert "handled:case-7" in agent.stderr


def test_process_agent_rejects_a_root_other_than_the_manifest_root(
    tmp_path: Path,
) -> None:
    """Rebinding an entry to another root can launch bytes outside its manifest."""
    declared_root = tmp_path / "declared"
    alternate_root = tmp_path / "alternate"
    declared_root.mkdir()
    alternate_root.mkdir()
    payload = b"#!/bin/sh\nexit 0\n"
    for root in (declared_root, alternate_root):
        (root / "agent").write_bytes(payload)
        os.chmod(root / "agent", 0o755)
    entry = PopulationManifest.from_data(
        {
            "agents": [
                {
                    "agent_id": "bound",
                    "kind": "baseline",
                    "content_hash": _sha256(payload),
                    "command": ["agent"],
                }
            ]
        },
        population_root=declared_root,
    ).entries[0]

    with pytest.raises(AgentInfrastructureError, match="population root"):
        ProcessAgent(entry, alternate_root)


def test_process_agent_rechecks_launched_bytes_immediately_before_start(
    tmp_path: Path,
) -> None:
    """A file changed after manifest loading must not execute under the old hash."""
    root = tmp_path / "agents"
    root.mkdir()
    script = root / "agent"
    original = b"#!/bin/sh\nexit 0\n"
    script.write_bytes(original)
    os.chmod(script, 0o755)
    entry = PopulationManifest.from_data(
        {
            "agents": [
                {
                    "agent_id": "changed",
                    "kind": "baseline",
                    "content_hash": _sha256(original),
                    "command": ["agent"],
                }
            ]
        },
        population_root=root,
    ).entries[0]
    agent = ProcessAgent(entry, root)
    script.write_bytes(b"#!/bin/sh\necho changed\n")

    with pytest.raises(AgentInfrastructureError, match="content hash"):
        agent.start()


def test_process_agent_launches_the_opened_hashed_inode_during_path_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A path replacement at Popen must never substitute bytes verified earlier."""
    root = tmp_path / "agents"
    root.mkdir()
    script = root / "agent.py"
    original = (
        b"#!/usr/bin/env python3\n"
        b"import json, sys\n"
        b"for line in sys.stdin:\n"
        b"    json.loads(line)\n"
        b"    print(json.dumps({'action': 0}), flush=True)\n"
    )
    replacement = original.replace(b"'action': 0", b"'action': 1")
    script.write_bytes(original)
    os.chmod(script, 0o755)
    entry = PopulationManifest.from_data(
        {
            "agents": [
                {
                    "agent_id": "swap-resistant",
                    "kind": "baseline",
                    "content_hash": _sha256(original),
                    "command": ["agent.py"],
                    "resource_limits": {"move_seconds": 1.0},
                }
            ]
        },
        population_root=root,
    ).entries[0]
    agent = ProcessAgent(entry, root)
    real_popen = process_agent_module.subprocess.Popen

    def swapping_popen(*args, **kwargs):
        alternate = root / "alternate.py"
        alternate.write_bytes(replacement)
        os.chmod(alternate, 0o755)
        os.replace(alternate, script)
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(process_agent_module.subprocess, "Popen", swapping_popen)

    with agent:
        action = agent.act({"case_id": "swap", "legal_actions": [0, 1]})

    assert action == 0


def test_process_agent_distinguishes_move_timeout_from_crash(tmp_path: Path) -> None:
    """Conflating a rule deadline with a crashed process corrupts valid-game counts."""
    root = tmp_path / "agents"
    root.mkdir()
    sleepy = root / "sleepy.py"
    sleepy.write_text(
        "#!/usr/bin/env python3\nimport sys, time\n"
        "for line in sys.stdin:\n    time.sleep(2)\n",
        encoding="utf-8",
    )
    crash = root / "crash.py"
    crash.write_text("#!/usr/bin/env python3\nraise RuntimeError('boom')\n", encoding="utf-8")
    os.chmod(sleepy, 0o755)
    os.chmod(crash, 0o755)

    def entry_for(filename: str, timeout: float):
        return PopulationManifest.from_data(
            {
                "population_root": ".",
                "agents": [
                    {
                        "agent_id": filename,
                        "kind": "baseline",
                        "content_hash": _sha256((root / filename).read_bytes()),
                        "command": [filename],
                        "resource_limits": {"move_seconds": timeout},
                    }
                ],
            },
            population_root=root,
        ).entries[0]

    with ProcessAgent(entry_for("sleepy.py", 0.02), root) as agent:
        with pytest.raises(ProcessMoveTimeout):
            agent.act({"case_id": "slow", "legal_actions": [0]})

    with ProcessAgent(entry_for("crash.py", 1.0), root) as agent:
        with pytest.raises(AgentInfrastructureError, match="exited"):
            agent.act({"case_id": "crash", "legal_actions": [0]})
        agent.close()
        assert "boom" in agent.stderr


@pytest.mark.parametrize(
    ("script_body", "agent_request"),
    [
        (
            "import time\ntime.sleep(1.0)\n",
            {"case_id": "blocked-write", "payload": "x" * 2_000_000},
        ),
        (
            "import sys, time\nsys.stdin.readline()\n"
            "sys.stdout.write('{\"action\": 0}')\nsys.stdout.flush()\n"
            "time.sleep(1.0)\n",
            {"case_id": "partial-line", "legal_actions": [0]},
        ),
    ],
)
def test_process_deadline_covers_blocked_writes_and_incomplete_response_lines(
    tmp_path: Path, script_body: str, agent_request: dict[str, object]
) -> None:
    """Blocking on either pipe direction must not escape the per-move deadline."""
    root = tmp_path / "agents"
    root.mkdir()
    script = root / "blocking.py"
    script.write_text("#!/usr/bin/env python3\n" + script_body, encoding="utf-8")
    os.chmod(script, 0o755)
    entry = PopulationManifest.from_data(
        {
            "population_root": ".",
            "agents": [
                {
                    "agent_id": "blocking",
                    "kind": "baseline",
                    "content_hash": _sha256(script.read_bytes()),
                    "command": ["blocking.py"],
                    "resource_limits": {"move_seconds": 0.30},
                }
            ],
        },
        population_root=root,
    ).entries[0]

    started = time.monotonic()
    with ProcessAgent(entry, root) as agent:
        with pytest.raises(ProcessMoveTimeout):
            agent.act(agent_request)
    elapsed = time.monotonic() - started

    assert elapsed < 0.65


def test_process_deadline_kills_and_reaps_child_that_ignores_sigterm(
    tmp_path: Path,
) -> None:
    """Deadline cleanup must not add the graceful-close one-second wait."""
    root = tmp_path / "agents"
    root.mkdir()
    script = root / "ignore-term.py"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import signal, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "print('READY', file=sys.stderr, flush=True)\n"
        "sys.stdin.readline()\n"
        "time.sleep(5)\n",
        encoding="utf-8",
    )
    os.chmod(script, 0o755)
    entry = PopulationManifest.from_data(
        {
            "agents": [
                {
                    "agent_id": "ignore-term",
                    "kind": "baseline",
                    "content_hash": _sha256(script.read_bytes()),
                    "command": ["ignore-term.py"],
                    "resource_limits": {"move_seconds": 0.30},
                }
            ]
        },
        population_root=root,
    ).entries[0]

    with ProcessAgent(entry, root) as agent:
        readiness_deadline = time.monotonic() + 1.0
        while "READY" not in agent.stderr and time.monotonic() < readiness_deadline:
            time.sleep(0.005)
        assert "READY" in agent.stderr
        started = time.monotonic()
        with pytest.raises(ProcessMoveTimeout):
            agent.act({"case_id": "ignore-term", "legal_actions": [0]})
        elapsed = time.monotonic() - started

    assert elapsed < 0.65


def test_process_timeout_return_is_bounded_when_wait_does_not_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pathological reap must not turn a move deadline into an unbounded wait."""
    root = tmp_path / "agents"
    root.mkdir()
    script = root / "slow.py"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import sys, time\n"
        "sys.stdin.readline()\n"
        "time.sleep(5)\n",
        encoding="utf-8",
    )
    os.chmod(script, 0o755)
    entry = PopulationManifest.from_data(
        {
            "agents": [
                {
                    "agent_id": "slow-wait",
                    "kind": "baseline",
                    "content_hash": _sha256(script.read_bytes()),
                    "command": ["slow.py"],
                    "resource_limits": {"move_seconds": 0.05},
                }
            ]
        },
        population_root=root,
    ).entries[0]
    agent = ProcessAgent(entry, root)
    agent.start()
    assert agent._process is not None
    process = agent._process
    real_wait = process.wait

    def pathological_wait(timeout: float | None = None) -> int:
        if timeout is not None:
            raise subprocess.TimeoutExpired(process.args, timeout)
        time.sleep(0.50)
        return real_wait()

    monkeypatch.setattr(process, "wait", pathological_wait)
    started = time.monotonic()
    try:
        with pytest.raises(ProcessMoveTimeout):
            agent.act({"case_id": "bounded-reap", "legal_actions": [0]})
        elapsed = time.monotonic() - started
    finally:
        monkeypatch.setattr(process, "wait", real_wait)
        agent.close()

    assert elapsed < 0.25
