from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from scripts import build_connect4_lightzero_results as builder


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/connect4/source_snapshot.json"
BUILDER = ROOT / "scripts/build_connect4_lightzero_results.py"


def _run_builder(output_dir: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--source",
            str(SOURCE),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def _assert_sop_chain(path: Path, *, rounds: int, cumulative_games: int) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["status"] == "interim_training_in_progress"
    assert payload["elo_protocol"]["anchor_policy"] == "lightzero_connect4_rulebot_v1"
    assert payload["elo_protocol"]["anchor_elo"] == 1000.0
    assert payload["plot_contract"] == {
        "x_axis": "cumulative training self-play games_seen",
        "y_axis": "Elo only",
        "information_gain_and_win_rate": "machine-readable metadata only",
    }
    records = payload["rounds"]
    assert len(records) == rounds
    assert [record["round"] for record in records] == list(range(1, rounds + 1))
    assert sum(record["games_seen"] for record in records) == cumulative_games
    for previous, current in zip(records, records[1:], strict=False):
        assert previous["policy_nxt"] == current["policy_cur"]
    for record in records:
        assert record["games_seen"] >= 0
        assert set(record["elo"]) >= {record["policy_cur"], record["policy_nxt"]}
        assert set(record["elo_uncertainty"]) >= {
            record["policy_cur"],
            record["policy_nxt"],
        }
        assert all(isinstance(value, float) for value in record["elo"].values())
        assert all(value >= 0.0 for value in record["elo_uncertainty"].values())


def test_builder_emits_continuous_sop_and_auditable_metrics(tmp_path: Path) -> None:
    output = tmp_path / "results"

    _run_builder(output)

    for seed in range(4):
        _assert_sop_chain(
            output / f"policy_elo_seed{seed}.json",
            rounds=7,
            cumulative_games=11_200,
        )
    _assert_sop_chain(
        output / "policy_elo_pooled.json",
        rounds=7,
        cumulative_games=44_800,
    )

    metric_rows = [
        json.loads(line)
        for line in (output / "checkpoint_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(metric_rows) == 36
    assert {row["seed"] for row in metric_rows} == {0, 1, 2, 3}
    assert {row["training_iteration"] for row in metric_rows} == set(range(0, 80_001, 10_000))
    assert all(row["information_gain"] is None for row in metric_rows if row["training_iteration"] == 0)
    assert all(
        row["information_gain"]["mean"] >= 0.0
        for row in metric_rows
        if row["information_gain"] is not None
    )

    curve_path = output / "elo_curve.csv"
    assert b"\r\n" not in curve_path.read_bytes()
    with curve_path.open(newline="", encoding="utf-8") as stream:
        curve_rows = list(csv.DictReader(stream))
    assert len(curve_rows) == 40
    assert {row["trajectory"] for row in curve_rows} == {
        "seed0", "seed1", "seed2", "seed3", "pooled",
    }
    pooled_curve = [row for row in curve_rows if row["trajectory"] == "pooled"]
    assert int(pooled_curve[0]["cumulative_games_seen"]) == 6_464
    assert int(pooled_curve[-1]["cumulative_games_seen"]) == 51_264
    for seed in range(4):
        seed_curve = [row for row in curve_rows if row["trajectory"] == f"seed{seed}"]
        assert int(seed_curve[0]["cumulative_games_seen"]) == 1_616
        assert int(seed_curve[-1]["cumulative_games_seen"]) == 12_816
    assert (output / "policy_elo_curve.png").stat().st_size > 20_000

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["game"] == "connect4"
    assert summary["seed_count"] == 4
    assert summary["common_horizon_iteration"] == 80_000
    assert summary["training_complete"] is False
    assert summary["evaluation"]["games_per_checkpoint_per_seed"] == 5
    assert summary["evaluation"]["learner_seat"] == "first_player_only"
    assert summary["information_gain"]["probe_count"] == 512


def test_builder_is_deterministic_and_removes_machine_paths(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    _run_builder(first)
    _run_builder(second)

    first_files = sorted(path.relative_to(first) for path in first.iterdir() if path.is_file())
    second_files = sorted(path.relative_to(second) for path in second.iterdir() if path.is_file())
    assert first_files == second_files
    for relative in first_files:
        assert (first / relative).read_bytes() == (second / relative).read_bytes()

    committed_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in first.iterdir()
        if path.suffix in {".json", ".jsonl", ".csv"}
    )
    forbidden_roots = [
        str(Path("/").joinpath("Users")),
        str(Path("/").joinpath("data")),
        str(Path("/").joinpath("private", "tmp")),
    ]
    assert all(root not in committed_text for root in forbidden_roots)


def test_policy_elo_figure_title_does_not_overlap_subtitle() -> None:
    assert hasattr(builder, "make_policy_elo_figure")
    rows = [
        {
            "trajectory": "pooled",
            "cumulative_games_seen": 0,
            "elo": 1000.0,
            "elo_uncertainty": 100.0,
        },
        {
            "trajectory": "pooled",
            "cumulative_games_seen": 6400,
            "elo": 1200.0,
            "elo_uncertainty": 120.0,
        },
    ]

    figure, title, subtitle = builder.make_policy_elo_figure(rows)
    figure.canvas.draw()

    assert not title.get_window_extent().overlaps(subtitle.get_window_extent())
    plt.close(figure)
