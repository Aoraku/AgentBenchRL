#!/usr/bin/env python3
"""Render Task 10 curves and write a content hash manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(directory: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    with (directory / "strength_curve.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        strength = list(csv.DictReader(stream))
    checkpoints = [int(row["checkpoint"]) for row in strength]
    survival = [
        int(row["training_opening_survival"])
        / int(row["total_training_openings"])
        for row in strength
    ]
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.plot(checkpoints, survival, marker="o", color="#2864DC")
    axis.set(xlabel="Checkpoint", ylabel="Training opening survival", ylim=(0, 1.05))
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(directory / "strength_curve.png", dpi=160)
    plt.close(figure)

    with (directory / "training_curve.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        training = list(csv.DictReader(stream))
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    az = [row for row in training if row["algorithm"] == "alphazero"]
    axes[0].plot(
        [row["metric"].removeprefix("total_loss_") for row in az],
        [float(row["value"]) for row in az],
        marker="o",
        color="#2864DC",
    )
    axes[0].set(title="AlphaZero checkpoint 20", ylabel="Total loss")
    ppo = [row for row in training if row["algorithm"] == "ppo"]
    axes[1].plot(
        [int(row["step"]) for row in ppo],
        [float(row["value"]) for row in ppo],
        marker="o",
        color="#D96027",
    )
    axes[1].set(title="PPO", xlabel="Iteration", ylabel="Total loss")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(directory / "training_curve.png", dpi=160)
    plt.close(figure)

    manifest = directory / "SHA256SUMS"
    # src/games/snakego/scripts/<file> -> repository root is four levels up.
    repository_root = Path(__file__).resolve().parents[4]
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.name != manifest.name
    )
    manifest.write_text(
        "".join(
            f"{_sha256(path)}  {_manifest_name(path, repository_root)}\n"
            for path in paths
        ),
        encoding="utf-8",
    )


def _manifest_name(path: Path, repository_root: Path) -> str:
    try:
        return path.relative_to(repository_root).as_posix()
    except ValueError:
        return path.name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path(__file__).resolve().parents[4] / "results/snakego",
    )
    args = parser.parse_args()
    build(args.directory.resolve())


if __name__ == "__main__":
    main()
