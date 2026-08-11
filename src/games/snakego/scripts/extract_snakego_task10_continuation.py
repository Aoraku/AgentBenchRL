#!/usr/bin/env python3
"""Extract checkpoint-16-to-20 Task 10 allocation accounting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from games.snakego.experiments.evidence import extract_continuation_accounting


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allocated-gpus", type=int, default=1)
    args = parser.parse_args()
    result = extract_continuation_accounting(
        args.ledger,
        args.output,
        baseline_checkpoint=16,
        final_checkpoint=20,
        gpu_hour_ceiling=0.15,
        allocated_gpu_count=args.allocated_gpus,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
