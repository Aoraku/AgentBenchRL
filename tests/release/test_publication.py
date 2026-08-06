"""Publication contracts for the standalone AgentBenchRLFrame release."""

from __future__ import annotations

import os
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest
import yaml


ROOT = Path(__file__).parents[2]
UPSTREAM_LICENSE = ROOT / "LICENSES" / "AgentBench-MIT.txt"
UPSTREAM_COMMIT = "b17a1fe7d39a0a82eeca4da80a2a30c6db663f03"
CONTROLLER_COMMIT = "b581bca3ba3d2d7d58a2f8c6bbddd060fc7fdc87"
PROVENANCE_ROOT = ROOT / "provenance" / "agentbench-snakego-controller"
CONTROLLER_HASHES = {
    "communication.py": "7dd339d3492e5f02c3f1185c3002ee383a8dbc64106747700311b3270d4ebdb3",
    "constants.py": "0b83c20a65f1a97a5c08927938dddaec88fae43c8b730af169169b545f87ab63",
    "formatter.py": "d95dfa37d66d3bd4879d353e3e190427578400a4728e271978f626783ef019cf",
    "infrastructure.py": "154bc94b0a45f33131e2b3b09be8b061ea6a97ca1bd215f7fcd4473ff7fa6518",
    "operate.py": "28bc0e8bcd28d36903d0cb78849724ffc32b826fa1dd4172e47e47084cc157cc",
    "result.py": "c57710c489442af28f82f15aac7d1957431cc29858741de1273cc757c42bc6a8",
    "spawn.py": "7aaa2fddead772ff8dd3c9fb732c27d8847c6a3bfcc9e24ee37b623c9501b975",
}


def test_release_metadata_declares_an_installable_mit_project() -> None:
    """Missing machine-readable metadata makes wheels and citations ambiguous."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]

    assert project["name"] == "agentbench-rl-frame"
    assert project["version"] == "0.1.0"
    assert project["license"] == "MIT"
    assert project["license-files"] == ["LICENSE", "LICENSES/*.txt"]
    assert project["authors"]
    assert project["scripts"] == {"rlbench": "rlbench.cli.main:main"}
    assert "Programming Language :: Python :: 3.11" in project["classifiers"]
    assert "License :: OSI Approved :: MIT License" not in project["classifiers"]
    assert (ROOT / "LICENSE").read_text(encoding="utf-8").startswith("MIT License\n")
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert "does not redistribute" in notices
    assert "contestant" in notices
    assert "modified port" in notices.lower()
    assert "https://github.com/Aoraku/AgentBench.git" in notices
    assert UPSTREAM_COMMIT in notices
    assert "immutable public verification source" in notices
    assert "may require authorization" in notices
    assert "independently implemented SnakeGo" not in notices
    assert UPSTREAM_LICENSE.read_text(encoding="utf-8") == (
        ROOT / "LICENSE"
    ).read_text(encoding="utf-8")
    assert "Copyright (c) 2026 Qingle" in UPSTREAM_LICENSE.read_text(
        encoding="utf-8"
    )
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "modified port" in readme.lower()
    assert UPSTREAM_COMMIT in readme
    assert "immutable public verification source" in readme
    assert "independently implemented SnakeGo" not in readme
    engine_header = (ROOT / "src/games/snakego/engine.py").read_text(
        encoding="utf-8"
    ).splitlines()[0]
    assert "modified port" in engine_header.lower()
    assert (ROOT / "CHANGELOG.md").is_file()
    assert (ROOT / "MANIFEST.in").is_file()

    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert citation["cff-version"] == "1.2.0"
    assert citation["title"] == "AgentBenchRLFrame"
    assert citation["version"] == "0.1.0"
    assert citation["license"] == "MIT"
    assert citation["authors"]


def test_public_controller_provenance_snapshot_is_exact_and_narrow() -> None:
    manifest = json.loads((PROVENANCE_ROOT / "ORIGIN.json").read_text("utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["repository"] == "https://github.com/Aoraku/AgentBench.git"
    assert manifest["source_commit"] == CONTROLLER_COMMIT
    assert manifest["license_commit"] == UPSTREAM_COMMIT
    assert manifest["rights_declaration"] == "RIGHTS.md"
    assert manifest["historical_repository_access"] == "may require authorization"
    assert set(manifest["files"]) == set(CONTROLLER_HASHES)
    for name, expected_hash in CONTROLLER_HASHES.items():
        record = manifest["files"][name]
        assert record == {
            "source_path": (
                "backend_sources/corpus/26_snakego/logic/"
                f"gamecode_logic/logic/{name}"
            ),
            "sha256": expected_hash,
        }
        payload = (PROVENANCE_ROOT / "controller" / name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected_hash

    assert (PROVENANCE_ROOT / "LICENSE").read_bytes() == (
        ROOT / "LICENSES/AgentBench-MIT.txt"
    ).read_bytes()
    assert {
        path.relative_to(PROVENANCE_ROOT).as_posix()
        for path in PROVENANCE_ROOT.rglob("*")
        if path.is_file()
    } == {
        "LICENSE",
        "ORIGIN.json",
        "README.md",
        "RIGHTS.md",
        *(f"controller/{name}" for name in CONTROLLER_HASHES),
    }
    provenance_readme = (PROVENANCE_ROOT / "README.md").read_text("utf-8")
    assert "immutable public verification source" in provenance_readme
    assert "contestant" in provenance_readme
    assert "does not include" in provenance_readme
    rights = (PROVENANCE_ROOT / "RIGHTS.md").read_text("utf-8")
    assert "Copyright (c) 2026 Qingle" in rights
    assert "Qingle publishes this narrowly" in rights
    assert "makes no claim" in rights
    assert "contestant submissions" in rights


def test_ci_runs_the_supported_cpu_matrix_and_release_build() -> None:
    """A release tested on only one interpreter does not support its declared matrix."""
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/tests.yml").read_text(encoding="utf-8")
    )
    jobs = workflow["jobs"]
    cpu = jobs["cpu-tests"]

    assert cpu["strategy"]["matrix"]["python-version"] == ["3.11", "3.12", "3.13"]
    commands = "\n".join(
        str(step.get("run", "")) for step in cpu["steps"] if isinstance(step, dict)
    )
    assert "pytest" in commands
    assert "not gpu" in commands
    assert "python -m build" in commands
    assert "verify_built_wheel.py" in commands
    assert "agentbench_rl_frame-0.1.0.tar.gz" in commands
    for step in cpu["steps"]:
        if isinstance(step, dict) and "uses" in step:
            action, revision = step["uses"].rsplit("@", 1)
            assert action in {"actions/checkout", "actions/setup-python"}
            assert len(revision) == 40
            int(revision, 16)


def test_release_extractor_emits_a_standalone_audited_tree(tmp_path: Path) -> None:
    """Copying the parent repository would publish unrelated or private corpora."""
    destination = tmp_path / "AgentBenchRLFrame"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/extract_release_repository.py"),
            "--source-root",
            str(ROOT),
            "--destination",
            str(destination),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr

    required = {
        "LICENSE",
        "README.md",
        "CHANGELOG.md",
        "CITATION.cff",
        "pyproject.toml",
        "src/rlbench/__init__.py",
        "src/games/snakego/__init__.py",
        "populations/snakego/train.yaml",
        "populations/snakego/test.yaml",
        "results/snakego/SHA256SUMS",
        "THIRD_PARTY_NOTICES.md",
        "LICENSES/AgentBench-MIT.txt",
        "MANIFEST.in",
        "provenance/agentbench-snakego-controller/ORIGIN.json",
        "provenance/agentbench-snakego-controller/RIGHTS.md",
        "provenance/agentbench-snakego-controller/controller/operate.py",
    }
    assert required.issubset(
        {path.relative_to(destination).as_posix() for path in destination.rglob("*")}
    )
    assert not (destination / ".git").exists()
    assert not (destination / ".superpowers").exists()
    assert not (destination / "docs/superpowers").exists()
    assert not (destination / "backend_sources").exists()
    assert not (destination / "top_algorithms").exists()
    assert not any(path.name == "__pycache__" for path in destination.rglob("*"))
    assert not any(path.suffix == ".pyc" for path in destination.rglob("*"))
    assert not any(path.name.endswith(".egg-info") for path in destination.rglob("*"))

    allowed_agent_fields = {
        "agent_id",
        "kind",
        "source_archive",
        "source_archive_hash",
        "source_path",
        "source_tree_hash",
        "move_seconds",
    }
    for name in ("train", "test"):
        blueprint = yaml.safe_load(
            (destination / f"populations/snakego/{name}.yaml").read_text(
                encoding="utf-8"
            )
        )
        for agent in blueprint["agents"]:
            assert set(agent) == allowed_agent_fields
            for field in ("source_archive", "source_path"):
                relative = Path(agent[field])
                assert not relative.is_absolute()
                assert ".." not in relative.parts
                assert relative.parts[:2] == ("snakego", "agents")

    checksum_lines = (
        destination / "results/snakego/SHA256SUMS"
    ).read_text(encoding="utf-8").splitlines()
    assert len(checksum_lines) == 17

    scan = subprocess.run(
        [
            sys.executable,
            str(destination / "scripts/verify_release_repository.py"),
            str(destination),
        ],
        cwd=destination,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert scan.returncode == 0, scan.stderr


def _extract(source: Path, destination: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/extract_release_repository.py"),
            "--source-root",
            str(source),
            "--destination",
            str(destination),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )


@pytest.mark.parametrize(
    "malicious_path",
    [
        "/" + "private/outside/source",
        "snakego/agents/../../outside/source",
    ],
)
def test_release_extractor_rejects_population_path_escape_without_mutation(
    tmp_path: Path, malicious_path: str
) -> None:
    """Unchecked blueprint paths can read or write outside release roots."""
    clean_source = tmp_path / "clean-source"
    assert _extract(ROOT, clean_source).returncode == 0
    blueprint = clean_source / "populations/snakego/train.yaml"
    data = yaml.safe_load(blueprint.read_text(encoding="utf-8"))
    data["agents"][0]["source_path"] = malicious_path
    blueprint.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    sentinel = tmp_path / "outside" / "source"
    sentinel.parent.mkdir()
    sentinel.write_text("sentinel", encoding="utf-8")
    destination = tmp_path / "must-not-exist"

    completed = _extract(clean_source, destination)

    assert completed.returncode != 0
    assert "unsafe population path" in completed.stderr
    assert not destination.exists()
    assert sentinel.read_text(encoding="utf-8") == "sentinel"


def _unsafe_samples() -> tuple[str, ...]:
    slash = "/"
    backslash = "\\"
    begin = "BEGIN "
    return (
        slash + "private/worker/run",
        slash + "home/worker/run",
        slash + "Users/worker/run",
        slash + "mnt/worker/run",
        "C:" + backslash + "Users" + backslash + "worker" + backslash + "run",
        backslash * 2 + "server" + backslash + "share" + backslash + "run",
        "file" + "://" + slash + "private/worker/run",
        begin + "RSA" + " PRIVATE KEY",
        begin + "EC" + " PRIVATE KEY",
        begin + "DSA" + " PRIVATE KEY",
        begin + "OPENSSH" + " PRIVATE KEY",
        begin + "PGP" + " PRIVATE KEY BLOCK",
        "gh" + "p_" + "a" * 36,
        "github" + "_pat_" + "a" * 82,
        "AK" + "IA" + "A" * 16,
        "s" + "sh worker@host.invalid command",
        "s" + "cp worker@host.invalid" + ":" + "relative/path local",
    )


@pytest.mark.parametrize("unsafe", _unsafe_samples())
def test_release_audit_rejects_sensitive_paths_connections_and_keys(
    tmp_path: Path, unsafe: str
) -> None:
    """A clean result must cover common credential and remote-execution forms."""
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "sample.txt").write_text(unsafe, encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/verify_release_repository.py"),
            str(repository),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 1
    assert "sample.txt" in completed.stdout


def test_built_wheel_installs_and_validates_from_an_isolated_environment() -> None:
    """Editable imports can hide a wheel with missing packages or console metadata."""
    wheel = os.environ.get("AGENTBENCH_RELEASE_WHEEL")
    if wheel is None:
        pytest.skip("set AGENTBENCH_RELEASE_WHEEL after python -m build")
    sdist = os.environ.get("AGENTBENCH_RELEASE_SDIST")
    if sdist is None:
        pytest.skip("set AGENTBENCH_RELEASE_SDIST after python -m build")
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/verify_built_wheel.py"),
            wheel,
            "--sdist",
            sdist,
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr
    result = yaml.safe_load(completed.stdout)
    assert result["version"] == "0.1.0"
    assert result["validation"]["status"] == "valid"
    assert result["license_files"] == "verified"
    assert result["provenance_snapshot"] == "verified"
