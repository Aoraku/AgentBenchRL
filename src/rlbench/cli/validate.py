"""``rlbench validate-game`` subcommand."""

from __future__ import annotations

from rlbench.game import validate_game
from rlbench.registry import game_factory

from ._facts import print_json


def validate_command(game_name: str, *, seed: int) -> int:
    game = game_factory(game_name)()
    game.reset(seed)
    validate_game(game)
    spec = game.spec
    print_json(
        {
            "action_count": len(spec.action_names),
            "game": game_name,
            "observation_planes": len(spec.observation_spec.plane_names),
            "observation_scalars": len(spec.observation_spec.scalar_names),
            "status": "valid",
        }
    )
    return 0
