"""Usable local CLI for validation, training, evaluation, and reporting.

The subcommands are implemented in focused modules (``validate``, ``train``,
``evaluate``) and their shared helpers live in ``_facts`` and ``_policies``.
This module wires the argument parser to those handlers and re-exports the
historically public symbols so existing callers (experiment pipelines, tests,
and scripts that import ``rlbench.cli.main``) keep working unchanged.

Trainer construction (:func:`_build_trainer`, :func:`_alphazero_device`,
:func:`_run_training`) lives here rather than in ``train`` because the unit
tests monkeypatch ``PolicyValueNet``/``AlphaZeroTrainer`` on this module and
then call ``_build_trainer`` directly; keeping them co-located preserves that
contract. ``train`` imports them from here, and the ``train`` handler is
imported lazily inside :func:`main` to avoid an import cycle.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
from dataclasses import asdict
from typing import Any

import torch  # re-exported: tests monkeypatch ``cli_module.torch.cuda``

from rlbench.algorithms.alphazero import (
    AlphaZeroConfig,
    AlphaZeroTrainer,
    PolicyValueNet,
)
from rlbench.algorithms.ppo_tianshou import PPOTrainer
from rlbench.config import ComposedConfig
from rlbench.registry import ALGORITHMS, GAMES, game_factory
from rlbench.reporting import generate_report
from rlbench.telemetry import Event, EventLedger

from ._facts import (
    case_set_hash as _case_set_hash,
)
from ._facts import (
    create_manifest as _create_manifest,
)
from ._facts import (
    evaluation_state_ids as _evaluation_state_ids,
)
from ._facts import (
    hardware_facts as _hardware_facts,
)
from ._facts import (
    latest_gpu_hours as _latest_gpu_hours,
)
from ._facts import (
    load_manifest as _load_manifest,
)
from ._facts import (
    parse_seeds as _parse_seeds,
)
from ._facts import (
    print_json as _print_json,
)
from ._facts import (
    prior_checkpoint_path as _prior_checkpoint_path,
)
from ._facts import (
    prior_complete_evaluation_states as _prior_complete_evaluation_states,
)
from ._facts import (
    restore_event_backed_budgets as _restore_event_backed_budgets,
)
from ._facts import (
    save_checkpoint_exclusive as _save_checkpoint_exclusive,
)
from ._facts import (
    sha256_file as _sha256_file,
)
from ._facts import (
    software_facts as _software_facts,
)
from ._facts import (
    sum_optional_hours as _sum_optional_hours,
)
from ._facts import (
    validate_checkpoint_lineage as _validate_checkpoint_lineage,
)
from ._facts import (
    validate_manifest as _validate_manifest,
)
from ._facts import (
    validate_resume_manifest as _validate_resume_manifest,
)
from ._facts import (
    write_immutable_manifest as _write_immutable_manifest,
)
from ._policies import (
    _AlphaZeroEvaluationPolicy,
    _population_policy,
    _PPOEvaluationPolicy,
    _RandomPolicy,
    _TrainingRandomPolicy,
)
from .validate import validate_command as _validate_command

# The leading names are the framework-facing API; the underscore-prefixed
# entries are re-exported for backward compatibility with experiment
# pipelines, scripts, and tests that import them from ``rlbench.cli.main``.
__all__ = [
    "_AlphaZeroEvaluationPolicy",
    "_PPOEvaluationPolicy",
    "_RandomPolicy",
    "_TrainingRandomPolicy",
    "_alphazero_device",
    "_build_trainer",
    "_case_set_hash",
    "_create_manifest",
    "_evaluation_state_ids",
    "_hardware_facts",
    "_latest_gpu_hours",
    "_load_manifest",
    "_parse_seeds",
    "_population_policy",
    "_print_json",
    "_prior_checkpoint_path",
    "_prior_complete_evaluation_states",
    "_restore_event_backed_budgets",
    "_run_training",
    "_save_checkpoint_exclusive",
    "_sha256_file",
    "_software_facts",
    "_sum_optional_hours",
    "_validate_checkpoint_lineage",
    "_validate_manifest",
    "_validate_resume_manifest",
    "_write_immutable_manifest",
    "game_factory",
    "main",
]

# Training-input helpers live in ``train`` but are part of the historical
# ``rlbench.cli.main`` surface exercised directly by unit tests. Expose them
# lazily so importing them here does not create a module import cycle.
_TRAIN_REEXPORTS = frozenset(
    {
        "_resolve_training_inputs",
        "_validate_training_inputs",
        "_record_training_checkpoint",
    }
)


def __getattr__(name: str) -> Any:
    if name in _TRAIN_REEXPORTS:
        from . import train

        return getattr(train, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _alphazero_device(config: AlphaZeroConfig) -> str:
    if config.device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return config.device


def _build_trainer(
    config: ComposedConfig,
    *,
    ledger: EventLedger,
    run_id: str,
    opponent: Any = None,
    opponent_id: str = "opponent",
) -> tuple[Any, str]:
    factory = game_factory(config.game, config.canonical["game"])
    seed = int(config.canonical["training"]["seed"])
    algorithm_config = config.algorithm_settings()
    if config.algorithm == "alphazero":
        sample = factory()
        network = PolicyValueNet.from_game_spec(
            sample.spec,
            algorithm_config,
            device=_alphazero_device(algorithm_config),
        )
        return (
            AlphaZeroTrainer(
                network,
                algorithm_config,
                seed=seed,
                ledger=ledger,
                run_id=run_id,
            ),
            "generation",
        )
    if config.algorithm == "ppo":
        return (
            PPOTrainer(
                factory,
                algorithm_config,
                seed=seed,
                ledger=ledger,
                run_id=run_id,
                opponent=opponent,
                opponent_id=opponent_id,
            ),
            "iteration",
        )
    raise ValueError(f"unknown algorithm: {config.algorithm}")


def _run_training(trainer: Any, config: ComposedConfig) -> Iterator[int]:
    controls = config.canonical["training"]
    if config.algorithm == "alphazero":
        factory = game_factory(config.game, config.canonical["game"])
        for _ in range(int(controls["generations"])):
            metrics = trainer.run_generation(
                factory,
                self_play_episodes=int(controls["self_play_episodes"]),
                training_steps=int(controls["training_steps"]),
                processes=int(controls["processes"]),
            )
            if trainer.ledger is not None:
                trainer.ledger.append(
                    Event(
                        event_type="training_batch_finished",
                        run_id=trainer.run_id,
                        stage="learning",
                        payload={
                            "backend": "alphazero",
                            "checkpoint_index": metrics.generation,
                            **asdict(metrics),
                        },
                    )
                )
            yield int(metrics.generation)
        return
    for _ in range(int(controls["iterations"])):
        metrics = trainer.train_iteration()
        if trainer.ledger is not None:
            trainer.ledger.append(
                Event(
                    event_type="training_batch_finished",
                    run_id=trainer.run_id,
                    stage="learning",
                    payload={
                        "backend": "ppo",
                        "checkpoint_index": metrics.iteration,
                        **asdict(metrics),
                    },
                )
            )
        yield int(metrics.iteration)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rlbench")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate-game", help="run the shared game contract")
    validate.add_argument("game", choices=sorted(GAMES))
    validate.add_argument("--seed", type=int, default=0)

    train = commands.add_parser("train", help="run one finite learning job")
    train.add_argument("game", choices=sorted(GAMES))
    train.add_argument("--algo", required=True, choices=sorted(ALGORITHMS))
    train.add_argument("--config", required=True)
    train.add_argument("--output")
    train.add_argument("--resume")
    train.add_argument("--initialize", help="warm-start PPO model weights")
    train.add_argument("--population", help="training population manifest")
    train.add_argument("--opponent-id", help="train-pool process opponent")
    train.add_argument(
        "--opponent", choices=("random",), help="built-in PPO opponent"
    )

    evaluate = commands.add_parser("evaluate", help="evaluate a local checkpoint")
    evaluate.add_argument("game", choices=sorted(GAMES))
    evaluate.add_argument("--checkpoint", required=True)
    opponents = evaluate.add_mutually_exclusive_group()
    opponents.add_argument("--population")
    opponents.add_argument("--opponent", choices=("random",), default="random")
    evaluate.add_argument(
        "--seeds", default=None, help="comma-separated non-negative seeds"
    )

    commands.add_parser("report", help="derive tables and curves from facts").add_argument(
        "run_directory"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from .evaluate import _evaluate_command
    from .train import _train_command

    args = _parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "validate-game":
        return _validate_command(args.game, seed=args.seed)
    if args.command == "train":
        return _train_command(
            args.game,
            algorithm=args.algo,
            config_path=args.config,
            output=args.output,
            resume=args.resume,
            initialize=args.initialize,
            population=args.population,
            opponent_id=args.opponent_id,
            local_opponent=args.opponent,
        )
    if args.command == "evaluate":
        return _evaluate_command(
            args.game,
            checkpoint=args.checkpoint,
            population=args.population,
            opponent=args.opponent,
            seeds=_parse_seeds(args.seeds) if args.seeds else None,
        )
    if args.command == "report":
        report_dir = generate_report(args.run_directory)
        _print_json({"report_directory": str(report_dir)})
        return 0
    raise RuntimeError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
