#!/usr/bin/env python3
"""Run training-only Task 10 learned-checkpoint baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from games.snakego.experiments.checkpoint_league import evaluate_checkpoint_league


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--bootstrap", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--seeds", default="101")
    parser.add_argument("--output-ledger", type=Path, required=True)
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(","))
    result = evaluate_checkpoint_league(
        run_dir=args.run_dir,
        candidate_checkpoint=args.candidate,
        opponents={
            "bootstrap-checkpoint-5": args.bootstrap,
            "history-checkpoint-10": args.history,
        },
        seeds=seeds,
        output_ledger=args.output_ledger,
        evaluation_split="training",
    )
    print(json.dumps(result, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
