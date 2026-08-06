#!/usr/bin/env python3
"""Audit a standalone AgentBenchRLFrame repository for private execution details."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
from typing import Iterable, Sequence


IGNORED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
}
FORBIDDEN_TREES = (
    ".superpowers",
    "docs/superpowers",
    "backend_sources",
    "top_algorithms",
)
ARCHIVE_SUFFIXES = (".7z", ".gz", ".rar", ".tar", ".tgz", ".whl", ".zip")


def _candidate_files(root: Path) -> Iterable[Path]:
    if (root / ".git").exists():
        completed = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        for raw in completed.stdout.split(b"\0"):
            if raw:
                yield root / raw.decode("utf-8", errors="surrogateescape")
        return
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if path.is_file() and not any(part in IGNORED_PARTS for part in relative.parts):
            yield path


def _find_private_details(text: str) -> list[str]:
    findings: list[str] = []
    key_types = ("RSA", "EC", "DSA", "OPEN" + "SSH", "PGP")
    if any(
        "BEGIN " + key_type + " PRIVATE KEY" in text
        or "BEGIN " + key_type + " PRIVATE KEY BLOCK" in text
        for key_type in key_types
    ):
        findings.append("private-key material")

    posix_pattern = "/" + r"(?:private|home|Users|mnt)(?:/|\\)"
    if re.search(posix_pattern, text):
        findings.append("private absolute path")
    if re.search(r"\b[A-Za-z]:\\[^\s]+", text):
        findings.append("Windows absolute path")
    if re.search(r"\\\\[A-Za-z0-9._-]+\\[A-Za-z0-9$._-]+", text):
        findings.append("UNC path")

    file_scheme = "file" + "://"
    ssh_scheme = "s" + "sh://"
    if file_scheme in text.lower():
        findings.append("file URI")
    if ssh_scheme in text.lower():
        findings.append("SSH endpoint")
    if re.search(r"(?m)^\s*s" + r"sh(?:\s+-\S+)*\s+\S+", text):
        findings.append("SSH command")
    if re.search(
        r"\b[A-Za-z0-9._-]+@[A-Za-z0-9.-]+:[^\s]+", text
    ):
        findings.append("remote copy target")

    github_classic = "gh" + "p_"
    github_fine_grained = "github" + "_pat_"
    if re.search(re.escape(github_classic) + r"[A-Za-z0-9]{20,}", text):
        findings.append("GitHub token")
    if re.search(re.escape(github_fine_grained) + r"[A-Za-z0-9_]{20,}", text):
        findings.append("GitHub token")
    aws_prefix = "AK" + "IA"
    if re.search(re.escape(aws_prefix) + r"[A-Z0-9]{16}", text):
        findings.append("AWS access key")
    return findings


def audit_repository(root: Path) -> list[str]:
    root = root.resolve()
    findings: list[str] = []
    for relative in FORBIDDEN_TREES:
        if (root / relative).exists():
            findings.append(f"forbidden release tree: {relative}")
    for path in _candidate_files(root):
        relative = path.relative_to(root)
        if any(part in IGNORED_PARTS for part in relative.parts):
            continue
        if path.name.lower().endswith(ARCHIVE_SUFFIXES):
            findings.append(f"{relative.as_posix()}: unaudited archive")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for category in _find_private_details(text):
            findings.append(f"{relative.as_posix()}: {category}")
    return sorted(set(findings))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    arguments = parser.parse_args(argv)
    findings = audit_repository(arguments.root)
    if findings:
        for finding in findings:
            print(finding)
        return 1
    print("release repository audit: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
