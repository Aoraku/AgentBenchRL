# Connect4 LightZero AlphaZero Interim Policy Elo Snapshot

## Result

This snapshot freezes the largest checkpoint horizon shared by four independent
LightZero AlphaZero runs: iterations 0 through 80,000 in 10,000-iteration
increments. At iteration 80,000, the four runs had completed 51,264 self-play
games and 1,647,241 environment steps in total. All four training processes
were still running when the snapshot was taken on 2026-08-24.

The supplied Policy Elo SOP requires cumulative completed games on the x-axis
and Elo only on the y-axis. The committed figure therefore plots the absolute
cumulative self-play total for a pooled four-seed trajectory from iteration
10,000 to 80,000. The strict SOP `rounds[].games_seen` fields remain the
incremental budgets between the seven measured transitions; each SOP file also
records the nonzero game-count offset at its first evaluated policy. Iteration
0 is retained in the raw checkpoint metrics but excluded from Elo because
LightZero did not evaluate that checkpoint against RuleBot; no unmeasured Elo
value is fabricated.

| Iteration | Four-seed W-D-L | Pooled Elo | Fit uncertainty (1 s.d.) | Cumulative games seen |
| ---: | ---: | ---: | ---: | ---: |
| 10,000 | 6-2-12 | 892.7 | 81.3 | 6,464 |
| 20,000 | 16-3-1 | 1336.5 | 116.8 | 12,864 |
| 30,000 | 13-7-0 | 1268.4 | 101.9 | 19,264 |
| 40,000 | 12-1-7 | 1088.6 | 80.1 | 25,664 |
| 50,000 | 14-1-5 | 1168.0 | 86.8 | 32,064 |
| 60,000 | 17-0-3 | 1300.2 | 108.3 | 38,464 |
| 70,000 | 14-5-1 | 1268.4 | 101.9 | 44,864 |
| 80,000 | 19-1-0 | 1624.1 | 238.3 | 51,264 |

The decrease between iterations 20,000 and 40,000 is present after pooling all
four seeds: the observed result changes from 16-3-1 to 12-1-7. It is not caused
by a changing opponent pool. Every game uses the same named
`LightZero Connect4RuleBot` anchor at Elo 1000. The result is nevertheless not
evidence of a precisely measured strength regression: each pooled point has
only 20 games, every learner plays first, RuleBot uses stochastic fallback
actions, and AlphaZero training itself has no monotonic-improvement guarantee.
The later 19-1-0 point is close to separation, which explains its larger rating
uncertainty.

## Experimental Contract

- Game: standard 7-column × 6-row Connect Four.
- Algorithm: LightZero AlphaZero self-play.
- Seeds: 0, 1, 2, and 3.
- Shared checkpoint horizon: iteration 80,000.
- Target training budget: 1,000,000 environment steps per seed.
- MCTS: 50 simulations.
- Collector environments: 8.
- Episodes per collect: 8.
- Learner updates per collect: 50.
- Batch size: 256.
- Evaluator: five games per checkpoint and seed against the fixed RuleBot.
- Evaluation seat: learner first-player only.
- LightZero commit: `9f717c80c86d5c69b39a56d8ade18e684ce6311a`.
- Official configuration SHA-256:
  `090c3fd71aabc0e0e1671775e4ecabd888bf841ccccc663e214d291d51cf4e65`.

The snapshot-time run counters were:

| Seed | Training iteration | Environment steps | Self-play games | Published checkpoint horizon |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 99,900 | 497,789 | 16,000 | 80,000 |
| 1 | 101,900 | 500,390 | 16,320 | 80,000 |
| 2 | 87,800 | 479,667 | 14,064 | 80,000 |
| 3 | 88,300 | 473,038 | 14,144 | 80,000 |

Only the common 80,000-iteration horizon is compared. Later checkpoints from
the faster runs are deliberately excluded rather than extrapolated or mixed
with shorter seeds.

## Elo Protocol

Each saved policy is connected to one fixed RuleBot anchor by its individual
game outcomes. Ratings use AgentBenchRL's order-independent anchored
Bradley--Terry fit with draws worth 0.5, RuleBot fixed at 1000, and L2=0.01.
The uncertainty is one standard deviation from the inverse observed Hessian.

The pooled policy at an iteration means the same-iteration policies from all
four seeds treated as one experimental condition. Its 20 evaluation games are
the union of the four five-game evaluations. Its `games_seen` is the sum of
the actual completed self-play games across the four runs. Per-seed SOP JSON
is also committed; those trajectories are not overlaid on the pooled figure
because their game budgets use a different scale.

This protocol is an interim benchmark snapshot, not an official LightZero Elo
league. LightZero's native evaluator reports episode returns rather than Elo;
the Elo transformation is versioned and fully recorded here.

## Information Gain and Win Rate

Information Gain is measured between adjacent checkpoints on one frozen set of
512 legal probe positions generated with seed 20260823. For every seed and
transition, the metric is

`D_KL(policy_nxt || policy_cur)`

after masking illegal columns and renormalizing. It is reported in nats and is
kept separate from win rate and Elo. The mean across all 32 seed-transition
measurements is 0.7299 nats. The pooled transition means decrease from 1.0927
nats at 10k→20k to 0.4911 nats at 70k→80k. This measures policy change, not
policy quality; a larger value is not automatically better.

Win rate, W-D-L counts, Wilson intervals, Information Gain distributions,
environment steps, wall time, and checkpoint hashes are stored in JSON/JSONL.
The figure has no secondary axis and displays Elo only, as required by the
supplied SOP.

## Reproduction

Generate all derived artifacts deterministically from the normalized source
snapshot:

```text
PYTHONPATH=src python scripts/build_connect4_lightzero_results.py \
  --source results/connect4/source_snapshot.json \
  --output-dir results/connect4
```

Validate the builder, Elo implementation, curve contracts, and file hashes:

```text
PYTHONPATH=src python -m pytest \
  tests/scripts/test_connect4_lightzero_results.py \
  tests/metrics/test_elo.py tests/metrics/test_curves.py -q
shasum -a 256 -c results/connect4/SHA256SUMS
```

## Artifact Inventory

| Artifact | Purpose |
| --- | --- |
| `results/connect4/source_snapshot.json` | Normalized immutable source facts and checkpoint digests |
| `results/connect4/checkpoint_metrics.jsonl` | Per-seed checkpoint Elo, win rate, Information Gain, and budgets |
| `results/connect4/policy_elo_seed{0..3}.json` | Strict per-seed Policy Elo SOP trajectories |
| `results/connect4/policy_elo_pooled.json` | Strict pooled four-seed Policy Elo SOP trajectory |
| `results/connect4/elo_curve.csv` | Pooled and per-seed curve table |
| `results/connect4/policy_elo_curve.png` | Publication figure with Elo as the only y-axis |
| `results/connect4/summary.json` | Run-level result summary |
| `results/connect4/provenance.json` | Software, metric, checkpoint, and limitation provenance |
| `results/connect4/SHA256SUMS` | Per-file integrity hashes |

Checkpoint binaries, raw logs, TensorBoard event files, temporary archives,
host filesystem paths, and credentials are not committed.
