#!/usr/bin/env python3
"""Install one built wheel in an isolated environment and verify its CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from typing import Sequence
import zipfile


def _run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        timeout=600,
    )


UPSTREAM_LICENSE = "LICENSES/AgentBench-MIT.txt"
PROVENANCE = "provenance/agentbench-snakego-controller"


def _verify_distribution_licenses(wheel: Path, sdist: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        if not any(
            name.endswith(f".dist-info/licenses/{UPSTREAM_LICENSE}")
            for name in archive.namelist()
        ):
            raise RuntimeError("wheel is missing the AgentBench MIT notice")
    with tarfile.open(sdist, mode="r:gz") as archive:
        if not any(
            member.name.endswith(f"/{UPSTREAM_LICENSE}")
            for member in archive.getmembers()
        ):
            raise RuntimeError("sdist is missing the AgentBench MIT notice")


def _verify_sdist_provenance(sdist: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    provenance_root = repository_root / PROVENANCE
    manifest_bytes = (provenance_root / "ORIGIN.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    with tarfile.open(sdist, mode="r:gz") as archive:
        members = archive.getmembers()

        def payload(relative: str) -> bytes:
            matches = [
                member
                for member in members
                if member.isfile() and member.name.endswith(f"/{PROVENANCE}/{relative}")
            ]
            if len(matches) != 1:
                raise RuntimeError(f"sdist provenance member is missing: {relative}")
            stream = archive.extractfile(matches[0])
            if stream is None:
                raise RuntimeError(f"sdist provenance member is unreadable: {relative}")
            return stream.read()

        if payload("ORIGIN.json") != manifest_bytes:
            raise RuntimeError("sdist provenance manifest differs from the repository")
        for required in ("LICENSE", "README.md", "RIGHTS.md"):
            if payload(required) != (provenance_root / required).read_bytes():
                raise RuntimeError(f"sdist provenance member differs: {required}")
        for name, record in manifest["files"].items():
            digest = hashlib.sha256(payload(f"controller/{name}")).hexdigest()
            if digest != record["sha256"]:
                raise RuntimeError(f"sdist controller digest differs: {name}")


def verify_wheel(wheel: Path, sdist: Path, python: Path) -> dict[str, object]:
    wheel = wheel.resolve(strict=True)
    sdist = sdist.resolve(strict=True)
    _verify_distribution_licenses(wheel, sdist)
    _verify_sdist_provenance(sdist)
    with tempfile.TemporaryDirectory(prefix="agentbench-wheel-") as raw:
        root = Path(raw)
        environment = root / "venv"
        subprocess.run(
            [str(python), "-m", "venv", str(environment)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
        )
        executable = environment / (
            "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
        )
        cli = environment / (
            "Scripts/rlbench.exe" if sys.platform == "win32" else "bin/rlbench"
        )
        exporter = environment / (
            "Scripts/snakego-export-policy.exe"
            if sys.platform == "win32"
            else "bin/snakego-export-policy"
        )
        _run(
            [
                str(executable),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                str(wheel),
            ],
            cwd=root,
        )
        probe = _run(
            [
                str(executable),
                "-c",
                (
                    "import importlib.metadata,json,rlbench,games.snakego;"
                    "print(json.dumps({'version':importlib.metadata.version("
                    "'agentbench-rl-frame'),'rlbench':rlbench.__file__,"
                    "'snakego':games.snakego.__file__}))"
                ),
            ],
            cwd=root,
        )
        metadata = json.loads(probe.stdout)
        if not cli.is_file():
            raise RuntimeError("wheel did not install the rlbench console entry")
        if not exporter.is_file():
            raise RuntimeError("wheel did not install the SnakeGo exporter entry")
        _run([str(exporter), "--help"], cwd=root)
        validation = json.loads(
            _run([str(cli), "validate-game", "snakego", "--seed", "7"], cwd=root).stdout
        )
        for field in ("rlbench", "snakego"):
            if not Path(metadata[field]).resolve().is_relative_to(environment.resolve()):
                raise RuntimeError(f"{field} import escaped the isolated environment")
        return {
            "license_files": "verified",
            "provenance_snapshot": "verified",
            "version": metadata["version"],
            "validation": validation,
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--sdist", required=True, type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    arguments = parser.parse_args(argv)
    print(
        json.dumps(
            verify_wheel(arguments.wheel, arguments.sdist, arguments.python),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
