"""End-to-end population build and official SnakeGo protocol coverage."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest
import yaml
import numpy as np

import rlbench.population as population_module
from games.snakego import SnakeGoGame
import rlbench.cli.main as cli_module
from rlbench.evaluation import (
    EvaluationCase,
    EvaluationRunner,
    build_side_swapped_cases,
)
from rlbench.config import compose_config
from rlbench.population import PopulationManifest
from rlbench.telemetry import EventLedger


REPOSITORY_ROOT = Path(__file__).parents[2]


def _sha256(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _external_data_root() -> Path:
    configured = os.environ.get("AGENTBENCH_DATA_ROOT")
    if not configured:
        pytest.skip("AGENTBENCH_DATA_ROOT is required for external population tests")
    return Path(configured).resolve()


def _official_fixture(root: Path) -> PopulationManifest:
    script = root / "official_fixture.py"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "def read_exact(count):\n"
        "    value = sys.stdin.buffer.read(count)\n"
        "    if len(value) != count:\n"
        "        raise SystemExit(91)\n"
        "    return value\n"
        "config = read_exact(5)\n"
        "header = read_exact(3)\n"
        "count = int.from_bytes(header[1:3], 'big', signed=True)\n"
        "read_exact(7 * count)\n"
        "print('startup=' + (config + header).hex(), file=sys.stderr, flush=True)\n"
        "side = config[4]\n"
        "turn = 0\n"
        "while True:\n"
        "    if turn % 2 == side:\n"
        "        sys.stdout.buffer.write(b'\\x00\\x00\\x00\\x01\\x01')\n"
        "        sys.stdout.buffer.flush()\n"
        "    marker = read_exact(1)\n"
        "    if marker == b'\\x11':\n"
        "        gameover = marker + read_exact(6)\n"
        "        print('gameover=' + gameover.hex(), file=sys.stderr, flush=True)\n"
        "        break\n"
        "    print(('echo=' if turn % 2 == side else 'opponent=') + marker.hex(), "
        "file=sys.stderr, flush=True)\n"
        "    turn += 1\n",
        encoding="utf-8",
    )
    os.chmod(script, 0o755)
    return PopulationManifest.from_data(
        {
            "protocol_version": "snakego-official-v1",
            "agents": [
                {
                    "agent_id": "official-fixture",
                    "kind": "test_human",
                    "protocol": "snakego_official",
                    "content_hash": _sha256(script.read_bytes()),
                    "command": [script.name],
                    "roles": ["player_0", "player_1"],
                    "resource_limits": {"move_seconds": 1.0},
                }
            ],
        },
        population_root=root,
    )


def test_official_policy_exchanges_binary_protocol_and_game_lifecycle(
    tmp_path: Path,
) -> None:
    """A JSON request path or missing action broadcasts desynchronizes official AIs."""
    policy_type = getattr(population_module, "SnakeGoProcessPolicy", None)
    assert policy_type is not None, "SnakeGo official process policy is not implemented"
    root = tmp_path / "population"
    root.mkdir()
    manifest = _official_fixture(root)
    policy = policy_type(manifest.entries[0], manifest.population_root)
    case = EvaluationCase.create(
        seed=0,
        player_0="official-fixture",
        player_1="local",
        player_0_hash=manifest.entries[0].content_hash,
        player_1_hash="builtin:local",
        game_config={"max_round": 1},
        limits={"move_seconds": 1.0},
        protocol_version=manifest.protocol_version,
    )

    report = EvaluationRunner(
        SnakeGoGame, EventLedger(tmp_path / "events.jsonl")
    ).run(
        (case,),
        agents={
            "official-fixture": policy,
            "local": lambda observation, legal: 0,
        },
        run_id="official-protocol",
    )

    assert report.complete is True
    assert report.results[0].reason == "completed"
    assert report.results[0].actions == (0, 0)
    assert "startup=1010000100100082" in policy.stderr
    assert "echo=01" in policy.stderr
    assert "opponent=03" in policy.stderr
    assert "gameover=11000000040004" in policy.stderr
    assert policy.running is False


def test_official_policy_restarts_concurrently_without_executable_copy_races(
    tmp_path: Path,
) -> None:
    root = tmp_path / "concurrent-population"
    root.mkdir()
    manifest = _official_fixture(root)

    def restart(worker: int) -> None:
        policy = population_module.SnakeGoProcessPolicy(
            manifest.entries[0], manifest.population_root
        )
        try:
            for episode in range(8):
                game = SnakeGoGame({"max_round": 1})
                game.reset(worker * 8 + episode)
                policy.begin_game(None, manifest.entries[0].agent_id, 0, game)
                policy.close()
        finally:
            policy.close()

    with ThreadPoolExecutor(max_workers=16) as executor:
        tuple(executor.map(restart, range(16)))


def test_population_builder_generates_loadable_disjoint_runtime_manifests(
    tmp_path: Path,
) -> None:
    """Missing builds, mutable identities, or split overlap invalidate benchmark roles."""
    external_data_root = _external_data_root()
    data_root = tmp_path / "snakego-population"
    result = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "build_population_manifest.py"),
            "--repo-root",
            str(REPOSITORY_ROOT),
            "--data-root",
            str(data_root),
            "--agentbench-data-root",
            str(external_data_root),
        ],
        cwd=REPOSITORY_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    train = PopulationManifest.from_yaml(data_root / "manifests" / "train.yaml")
    test = PopulationManifest.from_yaml(data_root / "manifests" / "test.yaml")
    train_ids = {entry.agent_id for entry in train.entries}
    test_ids = {entry.agent_id for entry in test.entries}

    assert train_ids == {
        "snakego-rank03-guyan-bob",
        "snakego-rank05-ouuan-nauuo",
        "snakego-rank06-aysn-ai1",
        "snakego-rank15-wenjie2002-sampleai",
    }
    assert test_ids == {
        "snakego-rank01-omegafantasy-ragnarok",
        "snakego-rank02-viridian-riverside",
        "snakego-rank08-jhdjames37-notasample",
        "snakego-rank13-swastika-one",
    }
    assert train_ids.isdisjoint(test_ids)
    assert {entry.kind for entry in train.entries} == {"train_human"}
    assert {entry.kind for entry in test.entries} == {"test_human"}
    assert all(entry.protocol == "snakego_official" for entry in (*train.entries, *test.entries))
    assert all(entry.roles == ("player_0", "player_1") for entry in (*train.entries, *test.entries))
    assert all(
        entry.provenance["source_tree_hash"].startswith("sha256:")
        and entry.provenance["source_archive_hash"].startswith("sha256:")
        and entry.provenance["build_recipe_hash"].startswith("sha256:")
        for entry in (*train.entries, *test.entries)
    )
    generated_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((data_root / "manifests").glob("*.yaml"))
    )
    assert str(REPOSITORY_ROOT) not in generated_text
    assert str(Path.home()) not in generated_text


def _copy_population_blueprints(destination: Path) -> list[dict[str, object]]:
    blueprint_directory = destination / "populations" / "snakego"
    blueprint_directory.mkdir(parents=True)
    blueprints = []
    for set_name in ("train", "test"):
        source = REPOSITORY_ROOT / "populations" / "snakego" / f"{set_name}.yaml"
        shutil.copy2(source, blueprint_directory / source.name)
        blueprints.append(yaml.safe_load(source.read_text(encoding="utf-8")))
    return blueprints


def _copy_external_population_data(destination: Path) -> None:
    source_root = _external_data_root()
    blueprints = [
        yaml.safe_load(
            (REPOSITORY_ROOT / "populations" / "snakego" / f"{set_name}.yaml").read_text(
                encoding="utf-8"
            )
        )
        for set_name in ("train", "test")
    ]
    copied: set[Path] = set()
    for blueprint in blueprints:
        for raw in blueprint["agents"]:
            for field in ("source_archive", "source_path"):
                relative = Path(raw[field])
                if relative in copied:
                    continue
                copied.add(relative)
                source = source_root / relative
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if source.is_dir():
                    shutil.copytree(source, target)
                else:
                    shutil.copy2(source, target)


def test_population_builder_rejects_full_tree_tampering_before_outputs(
    tmp_path: Path,
) -> None:
    """Hashing only a main file would admit modified headers into benchmark builds."""
    repository = tmp_path / "repository"
    _copy_population_blueprints(repository)
    external_data_root = tmp_path / "external-data"
    _copy_external_population_data(external_data_root)
    blueprint = yaml.safe_load(
        (repository / "populations/snakego/train.yaml").read_text(encoding="utf-8")
    )
    tampered = external_data_root / blueprint["agents"][3]["source_path"] / "adk.hpp"
    tampered.write_bytes(tampered.read_bytes() + b"\n// tampered\n")
    data_root = tmp_path / "must-not-exist"

    result = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "build_population_manifest.py"),
            "--repo-root",
            str(repository),
            "--data-root",
            str(data_root),
            "--agentbench-data-root",
            str(external_data_root),
        ],
        cwd=REPOSITORY_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )

    assert result.returncode == 1
    assert "full source tree hash mismatch" in result.stderr
    assert not data_root.exists()


def test_strength_configs_compose_with_consumed_comparable_controls() -> None:
    """Unknown or backend-inert strength keys make advertised experiments unusable."""
    alphazero = compose_config(
        REPOSITORY_ROOT / "configs/experiments/snakego_alphazero_strength.yaml",
        game="snakego",
        algorithm="alphazero",
    )
    ppo = compose_config(
        REPOSITORY_ROOT / "configs/experiments/snakego_ppo_baseline.yaml",
        game="snakego",
        algorithm="ppo",
    )

    assert alphazero.canonical["training"] == {
        "seed": 260806,
        "generations": 200,
        "iterations": 1,
        "self_play_episodes": 128,
        "training_steps": 256,
        "processes": 8,
        "checkpoint_every": 1,
    }
    assert alphazero.algorithm_settings().simulations == 256
    assert alphazero.algorithm_settings().inference_batch_size == 256
    assert alphazero.algorithm_settings().replay_capacity == 500_000
    assert alphazero.algorithm_settings().device == "auto"
    assert ppo.canonical["training"]["iterations"] == 2_000
    assert ppo.algorithm_settings().vector_envs == 8
    assert ppo.algorithm_settings().snapshot_interval == 10
    assert ppo.algorithm_settings().recurrent is True
    expected_seeds = [101, 211, 307, 401, 503, 601, 701, 809]
    assert alphazero.canonical["evaluation"]["seeds"] == expected_seeds
    assert ppo.canonical["evaluation"]["seeds"] == expected_seeds
    assert alphazero.canonical["evaluation"]["move_seconds"] == 3.0
    assert ppo.canonical["evaluation"]["move_seconds"] == 3.0
    assert alphazero.canonical["resources"]["sample"] is True
    assert ppo.canonical["resources"]["sample"] is True


def test_cli_policy_seam_runs_real_side_swapped_human_match(tmp_path: Path) -> None:
    """Routing an official binary through line JSON prevents any real human match."""
    external_data_root = _external_data_root()
    data_root = tmp_path / "snakego-population"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "build_population_manifest.py"),
            "--repo-root",
            str(REPOSITORY_ROOT),
            "--data-root",
            str(data_root),
            "--agentbench-data-root",
            str(external_data_root),
        ],
        cwd=REPOSITORY_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=90,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    manifest = PopulationManifest.from_yaml(data_root / "manifests" / "train.yaml")
    entry = manifest.entry("snakego-rank15-wenjie2002-sampleai")
    factory = getattr(cli_module, "_population_policy", None)
    assert factory is not None, "CLI has no protocol-aware population policy seam"
    human = factory(entry, manifest.population_root, game_name="snakego")
    cases = build_side_swapped_cases(
        candidate_id="local",
        candidate_hash="builtin:first-legal",
        opponent_id=entry.agent_id,
        opponent_hash=entry.content_hash,
        seeds=[0],
        game_config={"max_round": 1},
        limits={"move_seconds": 3.0},
        protocol_version=manifest.protocol_version,
    )
    ledger = EventLedger(tmp_path / "real-match.jsonl")

    report = EvaluationRunner(SnakeGoGame, ledger).run(
        cases,
        agents={
            "local": lambda observation, legal: int(np.flatnonzero(legal)[0]),
            entry.agent_id: human,
        },
        run_id="real-human-match",
    )

    assert report.complete is True
    assert [(result.player_0, result.player_1) for result in report.results] == [
        ("local", entry.agent_id),
        (entry.agent_id, "local"),
    ]
    assert all(result.reason == "completed" for result in report.results)
    assert human.running is False
    events = tuple(ledger.read())
    assert sum(event.event_type == "evaluation_match" for event in events) == 2


def test_population_manifest_rejects_unsupported_roles(tmp_path: Path) -> None:
    """An unknown role cannot be assigned a deterministic side by evaluation."""
    executable = tmp_path / "agent"
    executable.write_bytes(b"agent")
    with pytest.raises(ValueError, match="unsupported population role"):
        PopulationManifest.from_data(
            {
                "agents": [
                    {
                        "agent_id": "invalid-role",
                        "kind": "baseline",
                        "content_hash": _sha256(executable.read_bytes()),
                        "command": ["agent"],
                        "roles": ["spectator"],
                    }
                ]
            },
            population_root=tmp_path,
        )


def _single_script_manifest(
    root: Path, *, name: str, source: str, move_seconds: float
) -> PopulationManifest:
    executable = root / name
    executable.write_text(source, encoding="utf-8")
    os.chmod(executable, 0o755)
    return PopulationManifest.from_data(
        {
            "protocol_version": "snakego-official-v1",
            "agents": [
                {
                    "agent_id": name,
                    "kind": "test_human",
                    "protocol": "snakego_official",
                    "content_hash": _sha256(executable.read_bytes()),
                    "command": [name],
                    "roles": ["player_0", "player_1"],
                    "resource_limits": {"move_seconds": move_seconds},
                }
            ],
        },
        population_root=root,
    )


def test_official_partial_action_times_out_and_cleans_up(tmp_path: Path) -> None:
    """A stalled partial frame must not survive its per-move rule deadline."""
    root = tmp_path / "slow-population"
    root.mkdir()
    manifest = _single_script_manifest(
        root,
        name="slow.py",
        move_seconds=0.05,
        source=(
            "#!/usr/bin/env python3\n"
            "import sys, time\n"
            "def read_exact(n):\n"
            "    value = sys.stdin.buffer.read(n)\n"
            "    assert len(value) == n\n"
            "    return value\n"
            "read_exact(5)\n"
            "header = read_exact(3)\n"
            "read_exact(7 * int.from_bytes(header[1:3], 'big', signed=True))\n"
            "print('waiting', file=sys.stderr, flush=True)\n"
            "sys.stdout.buffer.write(b'\\x00\\x00')\n"
            "sys.stdout.buffer.flush()\n"
            "time.sleep(5)\n"
        ),
    )
    human = population_module.SnakeGoProcessPolicy(
        manifest.entries[0], manifest.population_root
    )
    case = EvaluationCase.create(
        seed=0,
        player_0=manifest.entries[0].agent_id,
        player_1="local",
        player_0_hash=manifest.entries[0].content_hash,
        player_1_hash="builtin:first-legal",
        game_config={"max_round": 1},
        limits={"move_seconds": 0.05},
        protocol_version=manifest.protocol_version,
    )

    started = time.monotonic()
    report = EvaluationRunner(
        SnakeGoGame, EventLedger(tmp_path / "timeout.jsonl")
    ).run(
        (case,),
        agents={
            manifest.entries[0].agent_id: human,
            "local": lambda observation, legal: int(np.flatnonzero(legal)[0]),
        },
        run_id="official-timeout",
    )

    assert time.monotonic() - started < 0.75
    assert report.complete is True
    assert report.results[0].reason == "rule_timeout"
    assert report.results[0].score_player_0 == 0.0
    assert human.running is False


@pytest.mark.parametrize(
    ("reason", "local_policy", "expected_frame"),
    [
        ("illegal_action", lambda observation, legal: 99, "gameover=111101ff9c0002"),
        (
            "rule_timeout",
            lambda observation, legal: (time.sleep(5), 0)[1],
            "gameover=111001ff9c0002",
        ),
    ],
)
def test_official_forfeit_frame_names_error_and_non_forfeiting_winner(
    tmp_path: Path, reason: str, local_policy: object, expected_frame: str
) -> None:
    """Official cleanup must agree with the runner's valid rule-loss result."""
    root = tmp_path / reason
    root.mkdir()
    manifest = _official_fixture(root)
    human = population_module.SnakeGoProcessPolicy(
        manifest.entries[0], manifest.population_root
    )
    case = EvaluationCase.create(
        seed=0,
        player_0="local",
        player_1=manifest.entries[0].agent_id,
        player_0_hash="builtin:bad-local",
        player_1_hash=manifest.entries[0].content_hash,
        game_config={"max_round": 1},
        limits={"move_seconds": 0.05},
        protocol_version=manifest.protocol_version,
    )

    report = EvaluationRunner(
        SnakeGoGame, EventLedger(tmp_path / f"{reason}.jsonl")
    ).run(
        (case,),
        agents={"local": local_policy, manifest.entries[0].agent_id: human},
        run_id=f"official-{reason}",
    )

    assert report.results[0].reason == reason
    assert report.results[0].score_player_0 == 0.0
    assert expected_frame in human.stderr
    assert human.running is False


def test_official_policy_revalidates_executable_hash_before_each_game(
    tmp_path: Path,
) -> None:
    """Replacing a built human after manifest loading must block process launch."""
    root = tmp_path / "changed-population"
    root.mkdir()
    manifest = _official_fixture(root)
    human = population_module.SnakeGoProcessPolicy(
        manifest.entries[0], manifest.population_root
    )
    executable = root / manifest.entries[0].command[0]
    executable.write_bytes(executable.read_bytes() + b"\n# changed\n")
    case = EvaluationCase.create(
        seed=0,
        player_0=manifest.entries[0].agent_id,
        player_1="local",
        player_0_hash=manifest.entries[0].content_hash,
        player_1_hash="builtin:first-legal",
        game_config={"max_round": 1},
        limits={"move_seconds": 1.0},
        protocol_version=manifest.protocol_version,
    )

    report = EvaluationRunner(
        SnakeGoGame, EventLedger(tmp_path / "changed.jsonl")
    ).run(
        (case,),
        agents={
            manifest.entries[0].agent_id: human,
            "local": lambda observation, legal: int(np.flatnonzero(legal)[0]),
        },
        run_id="changed-human",
    )

    assert report.complete is False
    assert report.results[0].reason == "infrastructure_failure"
    assert human.running is False


def test_population_builder_rejects_escaping_agent_id_before_outputs(
    tmp_path: Path,
) -> None:
    """A blueprint identity is never an output path supplied to the compiler."""
    repository = tmp_path / "repository"
    _copy_population_blueprints(repository)
    train_path = repository / "populations" / "snakego" / "train.yaml"
    train = yaml.safe_load(train_path.read_text(encoding="utf-8"))
    outside = tmp_path / "escaped-agent"
    train["agents"][0]["agent_id"] = str(outside)
    train_path.write_text(
        yaml.safe_dump(train, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    data_root = tmp_path / "must-not-exist"
    external_data_root = tmp_path / "external-data"
    external_data_root.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "build_population_manifest.py"),
            "--repo-root",
            str(repository),
            "--data-root",
            str(data_root),
            "--agentbench-data-root",
            str(external_data_root),
        ],
        cwd=REPOSITORY_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )

    assert result.returncode == 1
    assert "agent_id" in result.stderr
    assert not outside.exists()
    assert not data_root.exists()


def test_population_builder_handles_non_utf8_compiler_diagnostics(
    tmp_path: Path,
) -> None:
    """Compiler diagnostics may quote archived source bytes outside UTF-8."""
    repository = tmp_path / "repository"
    _copy_population_blueprints(repository)
    external_data_root = tmp_path / "external-data"
    _copy_external_population_data(external_data_root)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    compiler = fake_bin / "c++"
    compiler.write_bytes(b"#!/bin/sh\nprintf '\\265\\n' >&2\nexit 1\n")
    os.chmod(compiler, 0o755)
    data_root = tmp_path / "failed-build"

    result = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "build_population_manifest.py"),
            "--repo-root",
            str(repository),
            "--data-root",
            str(data_root),
            "--agentbench-data-root",
            str(external_data_root),
        ],
        cwd=REPOSITORY_ROOT,
        env={**os.environ, "PATH": str(fake_bin)},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )

    assert result.returncode == 1
    assert "build failed" in result.stderr
    assert "codec can't decode" not in result.stderr
