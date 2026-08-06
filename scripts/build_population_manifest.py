#!/usr/bin/env python3
"""Build verified SnakeGo populations from a separately supplied data root."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import yaml


class BuildError(RuntimeError):
    """A blueprint, source input, compiler, or protocol probe failed."""


_AGENT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")
_AGENT_FIELDS = {
    "agent_id",
    "kind",
    "source_archive",
    "source_archive_hash",
    "source_path",
    "source_tree_hash",
    "move_seconds",
}
BUILD_RECIPES: Mapping[str, Mapping[str, Any]] = {
    "snakego-rank01-omegafantasy-ragnarok": {
        "compiler": "c++",
        "flags": ["-O3", "-std=c++11", "-I."],
        "sources": ["main.cpp"],
    },
    "snakego-rank02-viridian-riverside": {
        "compiler": "c++",
        "flags": ["-O2", "-std=c++11"],
        "sources": ["main.cpp", "Snakeenv.cpp", "Snakeio.cpp", "Snakestrategy.cpp"],
    },
    "snakego-rank03-guyan-bob": {
        "compiler": "c++",
        "flags": ["-O2", "-std=c++11"],
        "sources": ["main.cpp"],
    },
    "snakego-rank05-ouuan-nauuo": {
        "compiler": "c++",
        "flags": ["-O2", "-std=c++17"],
        "sources": [
            "src/main.cpp",
            "src/Agent.cpp",
            "src/Channel.cpp",
            "src/Logic.cpp",
            "src/Mcts.cpp",
        ],
    },
    "snakego-rank06-aysn-ai1": {
        "compiler": "c++",
        "flags": ["-O2", "-std=c++11"],
        "sources": ["main.cpp"],
    },
    "snakego-rank08-jhdjames37-notasample": {
        "compiler": "c++",
        "flags": ["-O2", "-std=c++14"],
        "sources": ["main.cpp", "ai.cpp", "odk.cpp"],
    },
    "snakego-rank13-swastika-one": {
        "compiler": "c++",
        "flags": ["-O2", "-std=c++11"],
        "sources": ["main.cpp"],
    },
    "snakego-rank15-wenjie2002-sampleai": {
        "compiler": "c++",
        "flags": ["-O2", "-std=c++11", "-I."],
        "sources": ["main.cpp"],
    },
}


def _validated_agent_id(raw: Any) -> str:
    if not isinstance(raw, str) or not _AGENT_ID.fullmatch(raw) or raw in {".", ".."}:
        raise BuildError("agent_id must be one safe portable path component")
    return raw


def _sha256_file(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def source_tree_hash(path: Path) -> str:
    """Hash every regular file name, byte length, and content digest in a source tree."""
    if path.is_file():
        files = (path,)
    elif path.is_dir():
        files = tuple(
            sorted(
                (candidate for candidate in path.rglob("*") if candidate.is_file()),
                key=lambda candidate: candidate.relative_to(path).as_posix(),
            )
        )
    else:
        raise BuildError(f"source path is missing: {path.name}")
    digest = hashlib.sha256()
    for source in files:
        relative = source.name if path.is_file() else source.relative_to(path).as_posix()
        relative_bytes = relative.encode("utf-8")
        payload = source.read_bytes()
        digest.update(len(relative_bytes).to_bytes(8, "big"))
        digest.update(relative_bytes)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(hashlib.sha256(payload).digest())
    return f"sha256:{digest.hexdigest()}"


def _relative_path(root: Path, raw: Any, field: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise BuildError(f"{field} must be a non-empty relative path")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise BuildError(f"{field} must be a contained relative path")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise BuildError(f"{field} escapes the external data root")
    return resolved


def _load_blueprint(path: Path) -> Mapping[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("schema_version") != 1:
        raise BuildError(f"invalid population blueprint: {path.name}")
    if raw.get("set") not in ("train", "test") or not isinstance(raw.get("agents"), list):
        raise BuildError(f"invalid population set: {path.name}")
    return raw


def _validate_agent(
    raw: Mapping[str, Any], *, external_data_root: Path
) -> tuple[Path, Path, Mapping[str, Any]]:
    if set(raw) != _AGENT_FIELDS:
        raise BuildError("population agent fields are invalid")
    agent_id = _validated_agent_id(raw.get("agent_id"))
    if raw.get("kind") not in {"train_human", "test_human"}:
        raise BuildError(f"{agent_id} kind is invalid")
    if not isinstance(raw.get("move_seconds"), (int, float)) or float(
        raw["move_seconds"]
    ) <= 0:
        raise BuildError(f"{agent_id} move_seconds is invalid")
    for field in ("source_archive_hash", "source_tree_hash"):
        if not isinstance(raw.get(field), str) or not _HASH.fullmatch(raw[field]):
            raise BuildError(f"{agent_id} {field} is invalid")

    archive = _relative_path(
        external_data_root, raw.get("source_archive"), "source_archive"
    )
    source_path = _relative_path(
        external_data_root, raw.get("source_path"), "source_path"
    )
    if _sha256_file(archive) != raw["source_archive_hash"]:
        raise BuildError(f"{agent_id} source archive hash mismatch")
    if source_tree_hash(source_path) != raw["source_tree_hash"]:
        raise BuildError(f"{agent_id} full source tree hash mismatch")
    recipe = BUILD_RECIPES.get(agent_id)
    if recipe is None:
        raise BuildError(f"{agent_id} has no audited build recipe")
    return archive, source_path, recipe


def _build_agent(
    raw: Mapping[str, Any], *, external_data_root: Path, agents_root: Path
) -> Mapping[str, Any]:
    archive, source_path, build = _validate_agent(
        raw, external_data_root=external_data_root
    )
    compiler = shutil.which(str(build["compiler"]))
    if compiler is None:
        raise BuildError(f"compiler is unavailable: {build['compiler']}")
    working_directory = source_path if source_path.is_dir() else source_path.parent
    sources: list[str] = []
    for value in build["sources"]:
        source = _relative_path(working_directory, value, "build source")
        if not source.is_file():
            raise BuildError(f"build source is not a file: {source.name}")
        sources.append(value)
    flags = [str(value) for value in build["flags"]]
    if any(not value or "\x00" in value for value in flags):
        raise BuildError("build flags must be non-empty strings")
    agent_id = _validated_agent_id(raw.get("agent_id"))
    output_directory = agents_root / agent_id
    output_directory.mkdir(parents=True, exist_ok=False)
    executable = output_directory / "agent"
    completed = subprocess.run(
        [compiler, *flags, *sources, "-o", str(executable)],
        cwd=working_directory,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        diagnostics = completed.stderr.decode("utf-8", errors="replace")
        detail = diagnostics.strip().splitlines()[-1:] or ["compiler failed"]
        raise BuildError(f"{agent_id} build failed: {detail[0]}")
    if not executable.is_file() or executable.stat().st_size == 0:
        raise BuildError(f"{agent_id} produced no executable")
    os.chmod(executable, 0o500)
    recipe = {
        "compiler": build["compiler"],
        "flags": flags,
        "sources": sources,
        "output": "agent",
    }
    recipe_hash = _canonical_hash(recipe)
    build_input_hash = _canonical_hash(
        {
            "source_archive_hash": _sha256_file(archive),
            "source_tree_hash": source_tree_hash(source_path),
            "build_recipe_hash": recipe_hash,
        }
    )
    entry = {
        "agent_id": agent_id,
        "kind": raw["kind"],
        "protocol": "snakego_official",
        "content_hash": _sha256_file(executable),
        "command": [f"{agent_id}/agent"],
        "roles": ["player_0", "player_1"],
        "resource_limits": {"move_seconds": float(raw["move_seconds"])},
        "provenance": {
            "source_archive": raw["source_archive"],
            "source_archive_hash": raw["source_archive_hash"],
            "source_tree_hash": raw["source_tree_hash"],
            "build_recipe_hash": recipe_hash,
            "build_input_hash": build_input_hash,
        },
    }
    _probe_protocol(entry, agents_root)
    return entry


def _probe_protocol(entry: Mapping[str, Any], agents_root: Path) -> None:
    repo_src = Path(__file__).resolve().parents[1] / "src"
    if str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))
    from games.snakego import SnakeGoGame
    from rlbench.population import PopulationManifest, SnakeGoProcessPolicy

    manifest = PopulationManifest.from_data(
        {"protocol_version": "snakego-official-v1", "agents": [entry]},
        population_root=agents_root,
    )
    policy = SnakeGoProcessPolicy(manifest.entries[0], agents_root)
    game = SnakeGoGame({"max_round": 512})
    game.reset(0)
    try:
        policy.begin_game(None, str(entry["agent_id"]), 0, game)
        action = policy.act_game_process(
            game,
            timeout_seconds=float(entry["resource_limits"]["move_seconds"]),
        )
        if not bool(game.legal_action_mask()[action]):
            raise BuildError(f"protocol probe emitted illegal action for {entry['agent_id']}")
        game.step(action)
        policy.observe_action(game, 0, action)
        policy.end_game(game, SimpleNamespace(valid=True))
    finally:
        policy.close()


def build_populations(
    repo_root: Path, data_root: Path, external_data_root: Path
) -> Mapping[str, Any]:
    repo_root = repo_root.resolve()
    data_root = data_root.resolve()
    external_data_root = external_data_root.resolve()
    blueprint_root = repo_root / "populations" / "snakego"
    blueprints = tuple(
        _load_blueprint(blueprint_root / f"{set_name}.yaml")
        for set_name in ("train", "test")
    )
    all_agents = [raw for blueprint in blueprints for raw in blueprint["agents"]]
    if not all(isinstance(raw, Mapping) for raw in all_agents):
        raise BuildError("population agents must be mappings")
    ids = [_validated_agent_id(raw.get("agent_id")) for raw in all_agents]
    if len(ids) != len(set(ids)):
        raise BuildError("train and test population agent IDs must be disjoint")
    for raw in all_agents:
        _validate_agent(raw, external_data_root=external_data_root)

    manifests_root = data_root / "manifests"
    manifests_root.mkdir(parents=True, exist_ok=False)
    agents_root = manifests_root / "agents"
    agents_root.mkdir(parents=True, exist_ok=False)
    built: dict[str, Mapping[str, Any]] = {}
    for raw in sorted(all_agents, key=lambda value: str(value["agent_id"])):
        built[str(raw["agent_id"])] = _build_agent(
            raw, external_data_root=external_data_root, agents_root=agents_root
        )
    manifest_paths: list[str] = []
    for blueprint in blueprints:
        set_name = str(blueprint["set"])
        runtime = {
            "schema_version": 1,
            "population_root": "agents",
            "protocol_version": blueprint["protocol_version"],
            "agents": [built[str(raw["agent_id"])] for raw in blueprint["agents"]],
        }
        destination = manifests_root / f"{set_name}.yaml"
        destination.write_text(
            yaml.safe_dump(runtime, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        manifest_paths.append(destination.relative_to(data_root).as_posix())
    report = {
        "schema_version": 1,
        "built_agents": sorted(built),
        "manifests": manifest_paths,
    }
    (data_root / "build-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--agentbench-data-root", type=Path)
    arguments = parser.parse_args(argv)
    external_data_root = arguments.agentbench_data_root
    if external_data_root is None:
        configured = os.environ.get("AGENTBENCH_DATA_ROOT")
        if configured:
            external_data_root = Path(configured)
    if external_data_root is None:
        print(
            "population build failed: provide --agentbench-data-root or AGENTBENCH_DATA_ROOT",
            file=sys.stderr,
        )
        return 1
    try:
        report = build_populations(
            arguments.repo_root, arguments.data_root, external_data_root
        )
    except (BuildError, OSError, subprocess.SubprocessError, ValueError) as exc:
        print(f"population build failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
