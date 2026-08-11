#!/usr/bin/env python3
"""Extract compact, path-free Task 10 evaluation evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from games.snakego.experiments.evidence import extract_evaluation_evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = extract_evaluation_evidence(args.ledger, args.metadata, args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
