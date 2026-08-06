# AgentBench RL Frame

AgentBench RL Frame is an installable research framework for deterministic,
two-player AgentBench games. It provides AlphaZero and Tianshou PPO backends,
immutable experiment facts, side-balanced evaluation, population manifests,
league utilities, telemetry, and reproducible reports. Frozen expected-state
fixtures cover SnakeGo rule conformance without bundling the upstream
controller corpus or contestant programs. The SnakeGo engine is a modified
port derived from the MIT-licensed AgentBench/THUAC 2022 controller.

Python 3.11 through 3.13 is supported.

The unified game, algorithm, evaluation, and telemetry boundaries are specified
in [the architecture design](docs/architecture.md).

## License and SnakeGo provenance

AgentBenchRLFrame is distributed under the MIT license. The implementation in
`src/games/snakego` is a modified port derived from AgentBench/THUAC 2022
SnakeGo controller semantics and code under the AgentBench MIT license. The
source repository is `https://github.com/Aoraku/AgentBench.git`; the controller
corpus is identified by commit
`b581bca3ba3d2d7d58a2f8c6bbddd060fc7fdc87`, and the upstream root MIT notice
by commit `b17a1fe7d39a0a82eeca4da80a2a30c6db663f03`. The notice is reproduced at
`LICENSES/AgentBench-MIT.txt`.

The historical upstream URL may require authorization. The
immutable public verification source is
`provenance/agentbench-snakego-controller/`, containing the seven exact
controller files relevant to this port, original paths and SHA-256 digests,
source and license commits, the AgentBench MIT notice, and a rights declaration
grounded in the Qingle copyright and commit-author records.

The AgentBenchRLFrame implementation lineage starts at
`5c1dbfe3a9c7fd453e3d462b601a43c1fe3bbbfa`; the SnakeGo port enters that
lineage at `7caffd83721e69717c3797a530f974f7bd1adae2`. Contestant submissions are
separate works and are not covered or relicensed by either repository's MIT
license.

## Install and verify

```bash
python -m pip install 'agentbench-rl-frame @ git+https://github.com/Aoraku/AgentBenchRL.git@v0.1.0'
python -m pip install -e '.[dev]'  # contributor checkout
python -m pytest
rlbench validate-game snakego --seed 7
```

The PPO dependency is pinned to Tianshou 2.0.1. Native SnakeGo population
builds require a C++ compiler available as `c++`.

## Six-method game plugin

A discrete game declares a static `DiscreteGameSpec` and implements exactly
six methods:

```python
class MyGame:
    spec = DiscreteGameSpec(...)

    def reset(self, seed: int) -> None: ...
    def current_player(self) -> int: ...
    def observe(self, player: int) -> Observation: ...
    def legal_action_mask(self) -> np.ndarray: ...
    def step(self, action: int) -> StepRecord: ...
    def outcome(self, player: int) -> float | None: ...
```

Register the class in `rlbench.registry.GAMES`. The framework owns training,
checkpointing, evaluation, metrics, telemetry, and reporting. A plugin may
also expose `clone`, `symmetries`, `encode_state_id`, `score`, or
`score_potential` for backend-specific acceleration and diagnostics.

## Train and resume

AlphaZero strength run:

```bash
rlbench train snakego \
  --algo alphazero \
  --config configs/experiments/snakego_alphazero_strength.yaml \
  --output runs/snakego-az
```

PPO comparison run:

```bash
rlbench train snakego \
  --algo ppo \
  --config configs/experiments/snakego_ppo_baseline.yaml \
  --output runs/snakego-ppo
```

Resume uses the original run directory, its immutable run manifest, and its
latest recorded checkpoint:

```bash
rlbench train snakego \
  --algo alphazero \
  --config configs/experiments/snakego_alphazero_strength.yaml \
  --output runs/snakego-az \
  --resume runs/snakego-az/checkpoints/checkpoint_000200.pt
```

The strength configurations declare only controls consumed by the standalone
CLI: self-play volume and workers, optimizer volume, search/network/replay
settings, PPO vector collection and snapshot settings, deterministic
evaluation seeds, move deadlines, and resource sampling. AlphaZero uses pure
self-play and selects CUDA when available; PPO uses learner and retained policy
snapshots. Raw human opponent
mixtures and promotion cadence are library-level orchestration concerns:
construct a `LeagueState`, attach a trainer evaluation callback, schedule
frozen matches, and apply `evaluate_promotion` to the resulting facts.
`AlphaZeroTrainer.run_generation` can record one-hot actions from
`train_human` opponents as expert replay. The optional
`expert_demo_max_decisions` bound retains only an opening prefix from each
episode; `expert_demo_opening_moves` and `expert_demo_opening_weight` control
the weight of that prefix. These controls are game-independent, and held-out
`test_human` policies are rejected from expert replay.

Complete expert-versus-expert games use the same optimizer through a compact,
game-independent trajectory interface. A trajectory contains only the reset
seed and canonical action sequence; the trainer reconstructs observations,
legal masks, player-relative terminal values, replay weights, and budgets:

```python
from rlbench.algorithms.alphazero import ExpertTrajectory

trainer.distill_expert_trajectories(
    game_factory,
    [ExpertTrajectory(seed=101, actions=(0, 3, 1, 4, 2))],
    training_steps=8192,
    fresh_replay=True,
    opening_moves=8,
    opening_weight=4.0,
)
```

## Build SnakeGo human populations

Committed population files are external-data blueprints. They contain stable
agent IDs, archive hashes, full source-tree hashes, and neutral paths under a
caller-supplied data root. AgentBench official logic and contestant source
bytes are not distributed by this repository. Prepare the neutral external
layout declared by each blueprint entry. Every agent has an `archive` file and
a `source` directory beneath
`$AGENTBENCH_DATA_ROOT/snakego/agents/<agent-id>/`; the single-file rank-03
entry uses `source/main.cpp`. Build native executables and runtime manifests
under a separate output root:

```bash
export AGENTBENCH_DATA_ROOT=/path/to/licensed-or-authorized-agentbench-data
python scripts/build_population_manifest.py \
  --data-root build/snakego-populations
```

Generated files:

- `build/snakego-populations/manifests/train.yaml` — ranks 3, 5, 6, and 15,
  labeled `train_human`;
- `build/snakego-populations/manifests/test.yaml` — held-out ranks 1, 2, 8,
  and 13, labeled `test_human`;
- `build/snakego-populations/manifests/agents/` — verified local executables;
- `build/snakego-populations/build-report.json` — selected agents and manifest
  names.

Each build validates the selected archive SHA-256, complete source-tree
SHA-256, framework-owned compiler recipe, executable bytes, and SnakeGo
startup/action/echo/game-over protocol. Runtime commands stay relative to the
declared population root. Train and test agent IDs are disjoint, and
`test_human` entries are excluded by `PopulationManifest.training_entries()`
and `OpponentSampler`.

## Evaluate and report

Evaluation against every held-out human uses fixed seeds from the run manifest
and automatically creates both side orders:

```bash
rlbench evaluate snakego \
  --checkpoint runs/snakego-az/checkpoints/checkpoint_000200.pt \
  --population build/snakego-populations/manifests/test.yaml
```

An explicit seed subset is also supported:

```bash
rlbench evaluate snakego \
  --checkpoint runs/snakego-az/checkpoints/checkpoint_000200.pt \
  --population build/snakego-populations/manifests/test.yaml \
  --seeds 101,211,307
```

Official ladder agents run as stateful binary-protocol processes. Every case
starts a fresh process, sends the five-byte configuration and full item
announcement, enforces the move deadline on five-byte decisions, broadcasts
one-byte local echoes and opponent operations, sends the seven-byte game-over
message, captures stderr, and closes the process.

Generate tables, figures, and availability annotations from the append-only
facts:

```bash
rlbench report runs/snakego-az
```

## Official SnakeGo submission wrapper

A compact inference bundle omits replay, optimizer, trainer, and RNG state while
embedding the exact network configuration and SnakeGo observation schema:

```bash
snakego-export-policy \
  --checkpoint runs/snakego-az/checkpoints/checkpoint_000200.pt \
  --output submissions/snakego/policy.pt \
  --channels 128 \
  --residual-blocks 8 \
  --simulations 256

python -m games.snakego \
  --bundle submissions/snakego/policy.pt \
  --device cpu \
  --seed 0
```

A complete AlphaZero training checkpoint can also serve the competition
stdin/stdout binary protocol directly. Network dimensions must match the
training configuration.

```bash
python -m games.snakego \
  --checkpoint runs/snakego-az/checkpoints/checkpoint_000200.pt \
  --channels 128 \
  --residual-blocks 8 \
  --simulations 256 \
  --inference-batch-size 256 \
  --device cpu \
  --seed 0
```

The wrapper consumes raw judge messages from stdin and writes only framed
actions to stdout. Diagnostics belong on stderr.

## Artifact facts

Every run directory contains evidence suitable for replay and audit:

- `run_manifest.json` records the canonical configuration hash, source hashes,
  software facts, hardware facts, and run identity;
- `events.jsonl` records raw moves, matches, budgets, checkpoints, evaluation
  completion, information gain, occupancy, and resource samples;
- `checkpoints/` contains content-hashed model, optimizer, replay, trainer, and
  random-state snapshots;
- `reports/` contains derived CSV tables, figures, and summary availability
  metadata.

Population runtime manifests add executable SHA-256, archive SHA-256,
full-source-tree SHA-256, build-recipe SHA-256, roles, and per-move limits.
Infrastructure-invalid games remain invalid facts; rule timeouts and illegal
actions remain valid game losses.
