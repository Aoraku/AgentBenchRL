# SnakeGo GPU Training and Strength Evaluation

## Result

AlphaZero checkpoint 20 is the frozen candidate. Its SHA-256 is
`816070471a8020a2e9a3dcc8d61be34626fa0d831daaf8af24ff5e51f0f16614`.
The candidate completed all declared training-population checks: 8/8 fixed-
opening survival cases; side-swapped learned-checkpoint matches against frozen
checkpoints 5 and 10; and a side-swapped match against training rank 15.

Checkpoint 20 scored 0.75 against bootstrap checkpoint 5, 0.50 against
historical checkpoint 10, and 0.00 against rank 15. A joint six-game Elo fit
anchored checkpoint 5 at 1000 and rated checkpoint 20 at
1161.872 ± 263.863. The joint promotion gate rejected checkpoint 20 because
the rank-15 95% Wilson lower bound was 0.000, below 0.50. Checkpoint 5 is a
frozen bootstrap reference, not a formally promoted champion. Checkpoint 20 is
not a champion.

The held-out evaluation was executed once after the candidate contract was
frozen. Checkpoint 20 scored 0/8 against ranks 1, 2, 8, and 13. The aggregate
score was 0.000 with a 95% Wilson interval of [0.000, 0.324]. This frozen fact
was not used for tuning, checkpoint selection, promotion, or any other training
decision. The result does not support a state-of-the-art or human-competitive
claim.

## Experimental Contract

- Game: SnakeGo, `max_round=512`.
- Candidate selection and promotion inputs: training population only.
- Training humans: ranks 3, 5, 6, and 15.
- Training sample-agent baseline: rank 15, seed 101, both side orders.
- Frozen learned baselines: bootstrap checkpoint 5 and historical checkpoint
  10, training seed 101, both side orders.
- Held-out humans: ranks 1, 2, 8, and 13.
- Held-out evaluation: seed 101, both side orders, 3.0-second move deadline.
- AlphaZero search: 128 simulations, `c_puct=1.75`, inference batch 128.
- Network: 128 channels and eight residual blocks.
- Optimization: learning rate 0.0005, batch size 512, replay capacity 50,000,
  mixed precision.
- Expert replay: the first 16 expert decisions per mixed game have weight 32;
  all other expert, learner, and self-play samples have weight 1.
- Population isolation: expert demonstrations accept only `train_human`
  policies; `test_human` policies are rejected.
- Expert-stage provenance: the checkpoint must resolve inside the run
  directory, match its checkpoint-file digest and canonical manifest, and be
  the event-ledger lineage head. Validation precedes network construction,
  process launch, event append, and checkpoint creation.

The exact AlphaZero canonical configuration hash is
`sha256:96102f41702eeb1c5f1b0ecfd3db431d74e9157554c519b93b7c8ff92c254af7`.
The exact PPO canonical configuration hash is
`sha256:14d1da2180bad903578eac02843bd4b3c62822024c0fe82bad6b20a121c8c528`.
The locked YAML files are
`configs/experiments/snakego_task10_alphazero_locked.yaml` and
`configs/experiments/snakego_task10_ppo_locked.yaml`.

## Evaluation Trust Boundary

Framework policies opt into cooperative in-process deadlines through nominal
`DeadlineAwareGamePolicy` or `DeadlineAwareLocalPolicy` base classes. Every
game policy receives an independent clone, including nominal framework
policies. A return after the absolute monotonic deadline is recorded as a rule
timeout. Duck-typed and generic policies remain under hard process isolation.

MCTS reports `completed_simulations` from completed search waves. Learning and
evaluation budgets accumulate that exact count, including both candidate and
aligned prior-checkpoint searches. A truncated deadline search therefore does
not receive the configured simulation count.

Occupancy comparisons require an identical content-addressed case set. The
case-set hash covers frozen opponents and executable hashes, seeds, side
assignments, game configuration, limits, and protocol version. The evaluated
candidate checkpoint hash is normalized so two checkpoints can share one
frozen case contract.

## Bounded Train-Only Sweep and Throughput

The bounded SnakeGo microbenchmark executed 18 real search-and-optimization
variants on CPU. It used only training seed family 260806–260823 and no
held-out opponent, seed, match, score, or checkpoint. Each row records exact
completed simulations, inference calls and batch sizes, neural positions per
second, optimizer time, optimizer steps, and sampled replay weight mass.

| Dimension | Values |
| --- | --- |
| MCTS simulations | 8, 16 |
| `c_puct` | 1.25, 1.75 |
| Root noise fraction | 0.00, 0.25 |
| Temperature horizon | 0, 16 moves |
| Replay capacity | 64, 256 |
| Learning rate | 0.0003, 0.0005 |
| Network width/depth | 8×1, 16×2 |
| Inference batch cap | 4, 8 |
| Human replay mixture | 0.0, 0.5 |

Observed neural throughput ranged from 387.5 to 1,721.4 positions/s across
the 18 short variants. The inference-batch-cap rows reached maximum actual
batches of 4 and 7. These low-power measurements establish that batched neural
inference and every declared control executed; they are not strength estimates.
The complete rows are in `results/snakego/train_only_sweep.csv` and
`results/snakego/train_only_sweep.json`.

The CUDA self-play calibration measured:

| Workers | Episodes | Environment steps/s | MCTS simulations/s | Seconds/episode |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 8.146 | 521.313 | 3.422 |
| 8 | 8 | 9.177 | 587.332 | 3.950 |

Eight workers increased aggregate environment and MCTS throughput by 12.7%.
GPU utilization during the continuation allocation ranged from 28% to 59%,
with approximately 3.36 GiB device memory in use.

## AlphaZero Training Evidence

The executable workflow is a persisted checkpoint-1-to-checkpoint-20 stage
machine. Resume requires the exact plan hash, canonical manifest, cumulative
episode and optimizer budgets, explicit allocated GPU count, and recorded
checkpoint lineage head. A run-scoped exclusive lock prevents competing stage
attempts; checkpoints use exclusive creation. The versioned attempt journal
binds each attempt ID to its stage, planned checkpoint, prior lineage hash,
immutable configuration, and ledger boundary. Journal-only attempts receive a
single abort fact. Orphan checkpoint bytes are accepted only after exact stage,
configuration, manifest, generation, trainer/replay state, budget, and prior-
lineage validation. The five interruption boundaries—before journal creation,
after journal creation, after checkpoint creation, after checkpoint event, and
after state replacement—produce one contiguous checkpoint/event/state lineage.
A repeated call on a complete run is idempotent.

Expert restore is followed by a literal stage reseed of trainer episode, replay
sampling, Python, NumPy, Torch, and CUDA generators. An absolute allocation
deadline is checked around opponent and learner bounded units, passed into
learner MCTS for wave-boundary termination, and checked before optimizer steps.
Every expert attempt stores attempt ID, executed seed, elapsed time, allocated
GPU count, allocation source, and allocated GPU-hours. Timeout resource facts
are derived into workflow state, reduce the retry ceiling exactly once, and do
not authorize checkpoint creation.

| Checkpoint stage | Source | Episodes | Optimizer steps | Seed | Expert/opening control |
| ---: | --- | ---: | ---: | ---: | --- |
| 1–10 | self-play | 16 each | 64 each | canonical training seed | none |
| 11 | rank 15 | 2 | 64 | 1825803012 | no expert replay |
| 12 | rank 15 | 2 | 64 | 1289487500 | expert, weight 1 |
| 13 | rank 15 | 2 | 128 | 1554186166 | expert, weight 1 |
| 14 | rank 6 | 2 | 128 | 684982806 | expert, weight 1 |
| 15 | rank 5 | 2 | 128 | 1664431547 | expert, weight 1 |
| 16 | rank 15 | 2 | 256 | 1015345658 | first 16 expert decisions, weight 32 |
| 17 | rank 6 | 2 | 256 | 2145794352 | first 16 expert decisions, weight 32 |
| 18 | rank 15 | 2 | 256 | 2110560877 | first 16 expert decisions, weight 32 |
| 19 | rank 6 | 2 | 256 | 1358002110 | first 16 expert decisions, weight 32 |
| 20 | rank 15 | 2 | 256 | 278777161 | first 16 expert decisions, weight 32 |

The stage totals are exactly 180 episodes and 2,432 optimizer steps. The
checkpoint-16 learning allocation was 0.574533 GPU-hours and checkpoint 20 was
0.695753 GPU-hours. The checkpoint-17-to-20 continuation consumed 0.121220
GPU-hours, below its 0.15 GPU-hour ceiling.

Checkpoint 20 has the following cumulative learning budget:

| Metric | Value |
| --- | ---: |
| Episodes | 180 |
| Environment steps | 40,697 |
| MCTS simulations | 680,576 |
| Optimizer steps | 2,432 |
| Learning wall-seconds | 2,504.711 |
| Allocated learning GPU-hours | 0.695753 |
| Replay samples | 36,867 |
| Replay weight mass | 41,827 |

The final rank-15 mixed generation added 3,719 expert-demo samples and 16
learner samples. Its 256 optimizer steps sampled replay weight mass 148,401.
Total loss was 0.5944 at the first step, 0.4441 on average, and 0.3324 at the
last step.

| Checkpoint | Fixed opening survival | Rank-15 games | Rank-15 score | Status |
| ---: | ---: | ---: | ---: | --- |
| 11 | 5/8 | 2/2 valid | 0.000 | candidate |
| 14 | 5/8 | 2/2 valid | 0.000 | candidate |
| 16 | 6/8 | 2/2 valid | 0.000 | candidate |
| 19 | 8/8 | not scored | unavailable | candidate |
| 20 | 8/8 | 2/2 valid | 0.000 | rejected candidate |

On the aligned rank-15 seed-101 camp-0 state, the human action is action 0.
Checkpoint 14 assigned it probability 0.2514 and ranked it second. Checkpoint
16 assigned it probability 0.2605, ranked it first, and selected it through
MCTS. Checkpoint 20 selected the other safe action, action 3. Its value head
ranked the safe successors above the suicidal successors: original-player
values were -0.0087 and -0.0536 for safe actions, versus -0.1100 for both
suicidal actions. This is a measurable training signal without a non-zero
sample-agent match score.

Checkpoint 5 completed a 16-game side-swapped random evaluation with score
0.625, anchored Elo 1088.504 ± 89.573, 592 evaluation moves, and 0.037038
allocated evaluation GPU-hours. Random-baseline strength is not evidence of
human competitiveness.

## Joint Promotion Decision

| Gate | Threshold | Candidate fact | Result |
| --- | ---: | ---: | --- |
| Elo delta vs bootstrap checkpoint 5 | ≥ 0 | +161.872 | pass |
| Frozen rank-15 win-rate Wilson lower bound | ≥ 0.50 | 0.000 | fail |
| Protected score vs bootstrap checkpoint 5 | ≥ 0.00 | 0.75 | pass |
| Protected score vs historical checkpoint 10 | ≥ 0.00 | 0.50 | pass |
| Protected score vs training rank 15 | ≥ 0.00 | 0.00 | pass |
| Evaluation completeness | complete | 6/6 valid | pass |

The executable decision input is `results/snakego/promotion_facts.json`; the
decision is `results/snakego/promotion_decision.json`. All six games are from
training-only frozen references. The held-out 0/8 result is absent from the
promotion facts.

The learned-checkpoint evaluation used real AlphaZero checkpoint policies,
128 configured simulations, seed 101, and both side orders. Its four games
contained 132 moves and 16,896 completed MCTS simulations. Evaluation wall
time was 23.689 seconds; the maximum interval between consecutive move events
was 0.266 seconds under the 3.0-second move deadline. All four games have
`valid=true` and `reason=completed`.

## PPO Budget Baseline

The recurrent masked-action PPO baseline completed six collect/update
iterations and saved checkpoint 6 with SHA-256
`3dc8dba00fa16989a6aa46549501ede3365d2ff299147bdf1cae2a8340fe4d6a`.

| Metric | PPO | AlphaZero checkpoint 20 |
| --- | ---: | ---: |
| Episodes | 48 | 180 |
| Environment steps | 1,004 | 40,697 |
| Optimizer steps | 24 | 2,432 |
| Learning wall-seconds | 42.692 | 2,504.711 |
| Allocated learning GPU-hours | 0.011856 | 0.695753 |

PPO loss was 0.20343 on iteration 1 and 0.00570 on iteration 6; the final
collected mean return was -0.25068. The PPO allocation is materially smaller
than AlphaZero's and does not support an equal-budget strength comparison.

## Frozen Held-Out Evaluation

| Held-out opponent | Games | Learner W-D-L | Score | 95% Wilson interval | Moves by side order |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rank 1 | 2 | 0-0-2 | 0.000 | [0.000, 0.658] | 2,012 / 1,974 |
| Rank 2 | 2 | 0-0-2 | 0.000 | [0.000, 0.658] | 1,159 / 984 |
| Rank 8 | 2 | 0-0-2 | 0.000 | [0.000, 0.658] | 1,776 / 1,721 |
| Rank 13 | 2 | 0-0-2 | 0.000 | [0.000, 0.658] | 2,013 / 1,978 |
| **Aggregate** | **8** | **0-0-8** | **0.000** | **[0.000, 0.324]** | **13,617 total** |

All eight games have `valid=true` and `reason=completed`. Evaluation wall time
was 1,176.258 seconds, corresponding to 0.326738 allocated GPU-hours on one
GPU. Information gain against checkpoint 19 is 0.025984 nats per aligned
learner decision and 0.708062 nats per episode.

Held-out occupancy shift is unavailable because the event ledger contains no
complete reference checkpoint evaluation with the identical held-out case-set
hash. No cross-opponent or cross-seed occupancy value is reported.

The held-out evaluation contains one seed group. Each per-human interval has
only two games, so the estimate has low statistical power.

## Reproduction

The hardware-neutral entry point prints the locked plan and runs every stage:

```text
python scripts/run_snakego_task10.py plan

CUDA_VISIBLE_DEVICES=<idle-device> python scripts/run_snakego_task10.py \
  workflow --output <alpha-zero-run> --population <training-manifest> \
  --rank5 <rank-5-agent-id> --rank6 <rank-6-agent-id> \
  --rank15 <rank-15-agent-id> --device cuda --allocated-gpus 1

CUDA_VISIBLE_DEVICES=<idle-device> python scripts/run_snakego_task10.py \
  train-ppo --output <ppo-run>

python scripts/run_snakego_task10.py promotion \
  --facts results/snakego/promotion_facts.json

CUDA_VISIBLE_DEVICES=<idle-device> python \
  scripts/evaluate_snakego_checkpoint_league.py \
  --run-dir <alpha-zero-run> \
  --candidate <alpha-zero-run>/checkpoints/checkpoint_000020.pt \
  --bootstrap <alpha-zero-run>/checkpoints/checkpoint_000005.pt \
  --history <alpha-zero-run>/checkpoints/checkpoint_000010.pt \
  --seeds 101 --output-ledger <checkpoint-league-ledger>

CUDA_VISIBLE_DEVICES=<idle-device> python scripts/run_snakego_task10.py \
  evaluate --checkpoint <alpha-zero-run>/checkpoints/checkpoint_000020.pt \
  --population <heldout-manifest> --seeds 101
```

The train-only sweep is reproducible with:

```text
python scripts/run_snakego_task10.py sweep \
  --output results/snakego --device cpu
python scripts/build_snakego_task10_artifacts.py

shasum -a 256 -c results/snakego/SHA256SUMS
```

## Local Artifact Inventory

| Artifact | Local path |
| --- | --- |
| Match-level training and held-out facts, including actions | `results/snakego/matches.jsonl` |
| Per-move raw facts | `results/snakego/moves.jsonl` |
| Evaluation completion facts | `results/snakego/evaluations.jsonl` |
| Source extraction metadata | `results/snakego/source_metadata_main.json`, `source_metadata_league.json` |
| Checkpoint-16-to-20 allocation accounting | `results/snakego/continuation_accounting.json` |
| Canonical configurations and hashes | `results/snakego/configs.json` |
| Promotion facts and decision | `results/snakego/promotion_facts.json`, `promotion_decision.json` |
| Full train-only sweep rows | `results/snakego/train_only_sweep.csv`, `train_only_sweep.json` |
| Strength and loss curves | `results/snakego/strength_curve.csv`, `training_curve.csv` |
| Rendered curves | `results/snakego/strength_curve.png`, `training_curve.png` |
| Hardware and resource facts | `results/snakego/resource_summary.json` |
| Checkpoint release manifest | `results/snakego/checkpoint_assets.json` |
| Per-file integrity hashes | `results/snakego/SHA256SUMS` |

Large checkpoint binaries are represented by filename, canonical configuration
hash, and SHA-256 in `checkpoint_assets.json` for the tagged artifact release.
The local raw extract contains 3 evaluation records, 14 match records, and
17,634 move records. Every match and move includes its sanitized canonical
source-event preimage, source event ID, recomputable canonical event hash,
source-ledger SHA-256, case and case-set hashes,
checkpoint/population/executable provenance, and no host path. Completion and
count fields, checkpoint anchors, case sets, population roles, and exact
executable coverage are cross-validated before extraction. Candidate identity
comes from unique source checkpoint facts. Human opponent identity, split,
roles, executable hashes, and population hash come from embedded canonical
immutable manifests; checkpoint-league identity comes from source completion
facts. Sidecar labels must equal those derived facts. Recursive filtering
rejects UNC paths, file URIs, Windows drives, embedded system paths, connection
URIs, host identifiers, user-at-host values, and credential fields from every
key, value, and source preimage. The two source
ledger digests are
`f7c96df676411ea72047aa4fed8279b948c9a1a4a06a96c0a5ca9139f13cb57b`
and
`cd05503b99c164de66bb56d2bd1114474b73f97497694fd0919f061e52111360`.
The repository contains the compact evidence required to recompute the tables,
promotion result, throughput summary, and figures.

## Verification

The behavior suite covers trusted and generic deadline policies, post-return
timeouts, cloned game isolation, exact MCTS completion counts, identical
case-set occupancy matching, self-play lifecycle cleanup, locked configuration
hashes, joint promotion gates, sweep coverage, and evidence hashes.

```text
python -m compileall -q src scripts tests
exit 0

python -m pytest -q
282 passed, 1 skipped in 67.51s

git diff --check
exit 0

shasum -a 256 -c results/snakego/SHA256SUMS
17 artifacts OK
```
