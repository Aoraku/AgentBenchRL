"""Export a compact SnakeGo AlphaZero or PPO policy for official play."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from rlbench.algorithms.alphazero import AlphaZeroConfig

from .protocol_agent import (
    export_alphazero_inference_bundle,
    export_ppo_inference_bundle,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--algorithm", choices=("alphazero", "ppo"), default="alphazero"
    )
    parser.add_argument("--channels", type=int)
    parser.add_argument("--residual-blocks", type=int)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--c-puct", type=float, default=1.75)
    parser.add_argument("--inference-batch-size", type=int, default=256)
    args = parser.parse_args(argv)

    if args.algorithm == "ppo":
        output = export_ppo_inference_bundle(args.checkpoint, args.output)
        details = {"algorithm": "ppo"}
    else:
        if args.channels is None or args.residual_blocks is None:
            parser.error("AlphaZero export requires --channels and --residual-blocks")
        config = AlphaZeroConfig(
            simulations=args.simulations,
            c_puct=args.c_puct,
            root_dirichlet_fraction=0.0,
            self_play_temperature=0.0,
            temperature_moves=0,
            channels=args.channels,
            residual_blocks=args.residual_blocks,
            mixed_precision=False,
            inference_batch_size=args.inference_batch_size,
            device="cpu",
        )
        output = export_alphazero_inference_bundle(
            args.checkpoint,
            args.output,
            config=config,
        )
        details = {
            "algorithm": "alphazero",
            "channels": args.channels,
            "residual_blocks": args.residual_blocks,
        }
    print(
        json.dumps(
            {
                "bundle": str(output.resolve()),
                "bytes": output.stat().st_size,
                **details,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
