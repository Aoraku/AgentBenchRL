#!/usr/bin/env python3
"""Extract the standalone AgentBenchRLFrame release tree from the source checkout."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import yaml


TREE_ROOTS = (
    ".github",
    "configs",
    "populations",
    "provenance",
    "reports/snakego_strength.md",
    "results/snakego",
    "scripts",
    "src",
    "tests",
)
ROOT_FILES = (
    ".gitignore",
    "CHANGELOG.md",
    "CITATION.cff",
    "LICENSE",
    "LICENSES/AgentBench-MIT.txt",
    "MANIFEST.in",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
)
IGNORED_NAMES = {
    ".DS_Store",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
}


class ExtractionError(RuntimeError):
    """The release tree cannot be created without risking an overwrite."""


def _ignored(_directory: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name in IGNORED_NAMES
        or name.endswith((".egg-info", ".pyc", ".pyo"))
    }


def _contained_path(root: Path, raw: Any, *, field: str) -> tuple[Path, Path]:
    if not isinstance(raw, str) or not raw:
        raise ExtractionError(f"unsafe population path in {field}")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ExtractionError(f"unsafe population path in {field}: {raw}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ExtractionError(f"unsafe population path in {field}: {raw}")
    return relative, resolved


def _copy(source_root: Path, destination: Path, raw: str) -> None:
    relative, source = _contained_path(source_root, raw, field="release input")
    target = (destination / relative).resolve()
    if not target.is_relative_to(destination):
        raise ExtractionError(f"unsafe release target: {raw}")
    if not source.exists():
        raise ExtractionError(f"required release input is missing: {raw}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target, ignore=_ignored)
    else:
        shutil.copy2(source, target)


def _validate_population_paths(source_root: Path, destination: Path) -> None:
    for name in ("train", "test"):
        blueprint_path = source_root / f"populations/snakego/{name}.yaml"
        blueprint = yaml.safe_load(blueprint_path.read_text(encoding="utf-8"))
        if not isinstance(blueprint, Mapping) or not isinstance(
            blueprint.get("agents"), list
        ):
            raise ExtractionError(f"invalid population blueprint: {name}.yaml")
        for agent in blueprint["agents"]:
            if not isinstance(agent, Mapping):
                raise ExtractionError(f"invalid population agent: {name}.yaml")
            for field in ("source_archive", "source_path"):
                relative, _source = _contained_path(
                    source_root, agent.get(field), field=field
                )
                target = (destination / relative).resolve()
                if not target.is_relative_to(destination):
                    raise ExtractionError(
                        f"unsafe population path in {field}: {agent.get(field)}"
                    )


def extract_release(source_root: Path, destination: Path) -> None:
    """Copy only inputs required to build, test, reproduce, and audit the release."""
    source_root = source_root.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise ExtractionError(f"destination already exists: {destination}")
    _validate_population_paths(source_root, destination)
    destination.mkdir(parents=True)
    try:
        for relative in ROOT_FILES:
            _copy(source_root, destination, relative)
        for relative in TREE_ROOTS:
            _copy(source_root, destination, relative)
    except Exception:
        shutil.rmtree(destination)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        extract_release(arguments.source_root, arguments.destination)
    except (ExtractionError, OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    print(arguments.destination.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
