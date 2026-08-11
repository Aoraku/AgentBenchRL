# Metrics-schema mapping (RL -> shared contract)

`rlbench report` writes two summaries into `<run>/report/`:

- `summary.json` — the native, stable RL report (unchanged; source of truth
  for this repository's own tables, curves, and tests).
- `summary.schema.json` — an **additive** projection onto the shared
  cross-repository contract in `AgentBench/docs/metrics-schema.md`, so A's
  `reporting/aggregate.py` can rank RL runs alongside HL runs on the same axes.

The mapping is implemented by `_schema_summary` in
`src/rlbench/reporting/report.py`. It never mutates the native schema.

## Field mapping

| metrics-schema (`summary.json` §3) | RL source | Notes |
| --- | --- | --- |
| `schema_version` `"1.0"` | constant | RL native uses int `1`; the export emits the string `"1.0"`. |
| `run_type` | constant `"eval"` | `rlbench report` summarizes an evaluated run. |
| `game` | `run_started.payload.game` | e.g. `"snakego"`. |
| `agent` | learner id (`run_started.candidate_id`) | the candidate under study. |
| `created` | first event `created_at` (ISO 8601) | run start time. |
| `git_commit` | `run_manifest.json` `software.git_commit` | `null` if not recorded. |
| `best_elo` | max learner Elo over checkpoints | `null` if no connected match graph. |
| `final_elo` / `elo_p0` | last learner Elo | anchored Elo of the final checkpoint. |
| `elo_p1` | `null` | a single learner has no second-seat rating. |
| `win_rate` | final checkpoint `win_rate_summary.score` | learner score, draws worth 0.5. |
| `score_margin` | `2 * win_rate - 1` | centered margin. |
| `wall_hours` | final `checkpoint.wall_seconds / 3600` | total wall clock. |
| `total_steps` | final `checkpoint.env_steps` | environment steps. |
| `AUC_gain` | trapezoid AUC of `policy_ig_measured.nats_per_decision` vs checkpoint | behavioural information-gain curve; `null` if < 2 measured points. |
| `AUC_raw`, `AUC_evo` | `null` | RL runs measure no raw/evo behavioural curves; recorded as `null`, never 0. |
| `budget_learning/evaluation/total` | RL budgets `*_wall_seconds` -> `wall_time_s` | token dimensions are `null` (not applicable to RL); never 0. |
| `elo_history` | per-checkpoint learner Elo | `[{checkpoint_index, rating, uncertainty}]` for curves. |
| `h2h` | learner win rate per opponent, final checkpoint | `{opponent: rate}` for the heatmap. |

## Invariants preserved

- **KL and occupancy stay orthogonal.** `local_policy_kl` (via
  `policy_ig_measured`) and `occupancy_shift` (via `occupancy_measured`) are
  reported separately and never summed, matching the shared rule.
- **Missing means `null`, never 0.** Any unmeasured or inapplicable quantity
  (token budgets, RL-absent raw/evo AUC, single-seat Elo, unrecorded commit)
  is emitted as `null` so A's reporting skips it instead of averaging in a 0.
