# Connect4 LightZero Results Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish an auditable four-seed interim Connect4 AlphaZero result snapshot in AgentBenchRL, following the supplied Policy Elo SOP and the repository's existing `results/<game>/` conventions.

**Architecture:** Freeze a common checkpoint horizon across the four xulab LightZero runs, normalize their training/evaluation/checkpoint facts into one immutable source snapshot, and generate strict per-seed SOP JSON plus a pooled four-seed Elo curve. Elo is fit against one named static RuleBot anchor with the repository's anchored Bradley--Terry implementation; win rate, Information Gain, uncertainty, budgets, provenance, and limitations remain in machine-readable artifacts while the figure's vertical axis contains Elo only.

**Tech Stack:** Python 3.11, LightZero/PyTorch for checkpoint policy extraction, AgentBenchRL `fit_anchored_elo`, matplotlib, JSON/JSONL/CSV, SHA-256, pytest.

**Spec:** `docs/metrics-schema-mapping.md`, the user-supplied `policy_elo_sop/SOP.md`, and the established `results/snakego/` publication layout.

## Global Constraints

- Use only completed checkpoints present for all four seeds; never extrapolate unfinished training.
- Preserve the official LightZero configuration: 50 MCTS simulations, five RuleBot evaluator games, learner first-player only.
- Treat the RuleBot identity as static but its fallback action selection as stochastic.
- Follow the supplied SOP literally: initialize the iteration-0 `policy_cur` at Elo 1000 and cumulative `games_seen = 0`, include the 0→10k transition, and plot only Elo on y.
- Keep SOP trajectory increments separate from absolute training counters: the former exclude games completed before the iteration-0 policy boundary.
- Store missing or unavailable measurements as JSON `null`, never numeric zero.
- Do not commit model checkpoints, TensorBoard events, raw logs, machine paths, or archives.
- State prominently that this is an interim, first-player-only, low-game-count snapshot and not a final strength claim.

---

### Task 1: Freeze and normalize the xulab snapshot

**Files:**
- Create: `results/connect4/source_snapshot.json`
- Create: `results/connect4/provenance.json`

**Interfaces:**
- Consumes: the four `remote_seed{0..3}_retry1` LightZero run directories on xulab.
- Produces: one normalized JSON object with `seeds[].checkpoints[]`, including iteration, env steps, self-play games, wall time, RuleBot returns, checkpoint SHA-256, and adjacent-checkpoint Information Gain.

- [ ] **Step 1: Read all four run states and choose the greatest 10,000-iteration checkpoint common to every seed.**

Run: `ssh xulab '<read-only progress/checkpoint inventory>'`

Expected: all four processes remain live and every selected checkpoint exists with a stable SHA-256 digest.

- [ ] **Step 2: Export normalized checkpoint facts.**

Run the existing Connect4 `build_policy_metrics.py` logic against each frozen run using one shared 512-position probe set and probe seed `20260823`.

Expected: every adjacent pair has finite non-negative masked-policy KL in nats; every saved nonzero checkpoint has exactly five parsed RuleBot returns.

- [ ] **Step 3: Remove machine-specific paths and write provenance.**

Record the LightZero commit, official-config SHA-256, selected horizon, checkpoint digests, metric versions, source host label `xulab`, snapshot time, and known evaluation limitations. Do not retain `/data/...` paths.

### Task 2: Build reproducible SOP and repository artifacts

**Files:**
- Create: `scripts/build_connect4_lightzero_results.py`
- Create: `tests/scripts/test_connect4_lightzero_results.py`
- Create: `results/connect4/policy_elo_seed0.json`
- Create: `results/connect4/policy_elo_seed1.json`
- Create: `results/connect4/policy_elo_seed2.json`
- Create: `results/connect4/policy_elo_seed3.json`
- Create: `results/connect4/policy_elo_pooled.json`
- Create: `results/connect4/checkpoint_metrics.jsonl`
- Create: `results/connect4/elo_curve.csv`
- Create: `results/connect4/summary.json`

**Interfaces:**
- Consumes: `results/connect4/source_snapshot.json`.
- Produces: deterministic SOP JSON, native AgentBenchRL results, and a pooled curve table.

- [ ] **Step 1: Write failing schema and determinism tests.**

Tests must assert round numbering and policy-chain continuity, the initialized iteration-0 Elo of 1000, the 0→10k first transition, non-negative incremental `games_seen`, exact cumulative-game endpoints, fixed RuleBot anchor metadata, nullable uncertainty only for the unmeasured initialized policy, no machine paths, and byte-identical JSON/CSV on repeated builds.

- [ ] **Step 2: Run the focused tests and verify failure.**

Run: `PYTHONPATH=src python -m pytest tests/scripts/test_connect4_lightzero_results.py -q`

Expected: FAIL because the builder does not yet exist.

- [ ] **Step 3: Implement the deterministic builder.**

Convert each W/D/L return into `MatchOutcome(policy, "lightzero_connect4_rulebot_v1", score)` records, fit anchored Elo at 1000 with `fit_anchored_elo`, and pool same-iteration outcomes across seeds only for the explicitly labeled aggregate trajectory. Retain per-seed Information Gain and win-rate facts outside the plot contract.

- [ ] **Step 4: Generate all JSON/JSONL/CSV artifacts and rerun tests.**

Run: `PYTHONPATH=src python scripts/build_connect4_lightzero_results.py --source results/connect4/source_snapshot.json --output-dir results/connect4`

Expected: focused tests PASS and repeated generation yields no Git diff.

### Task 3: Render and document the publication snapshot

**Files:**
- Create: `results/connect4/policy_elo_curve.png`
- Create: `reports/connect4_lightzero.md`
- Create: `results/connect4/SHA256SUMS`

**Interfaces:**
- Consumes: `results/connect4/elo_curve.csv`, summary, and provenance.
- Produces: a publication-ready figure, an honest experimental report, and integrity hashes.

- [ ] **Step 1: Render the figure with the scientific-figure house style.**

Show the pooled four-seed Elo as the only plotted line, beginning at `(games_seen=0, Elo=1000)`, with fit uncertainty only where measured. Sum post-initial-policy self-play increments across the four seeds on x and show Elo only on y. Keep individual seed trajectories in JSON/CSV rather than overlaying incompatible per-seed game budgets. Export a print-readable PNG with no win-rate or IG secondary axis.

- [ ] **Step 2: Write the report.**

Document progress at snapshot time, training configuration, static-but-stochastic RuleBot behavior, five-game/first-seat limitation, pooled-vs-per-seed semantics, Information Gain definition, exact reproduction command, and the fact that training was still running.

- [ ] **Step 3: Hash all committed result artifacts.**

Run: `cd results/connect4 && shasum -a 256 <all files except SHA256SUMS> > SHA256SUMS`

Expected: `shasum -a 256 -c results/connect4/SHA256SUMS` reports every file OK.

### Task 4: Validate and submit the PR

**Files:**
- Modify only if required by publication indexing: `README.md`

**Interfaces:**
- Consumes: all artifacts from Tasks 1--3.
- Produces: one reviewable Git commit and one GitHub pull request to `Aoraku/AgentBenchRL` main.

- [ ] **Step 1: Run focused and repository validation.**

Run: `PYTHONPATH=src python -m pytest tests/scripts/test_connect4_lightzero_results.py tests/metrics/test_elo.py tests/metrics/test_curves.py -q`

Run: `shasum -a 256 -c results/connect4/SHA256SUMS`

Expected: all focused tests and hashes pass. Record any unrelated full-suite dependency limitation separately.

- [ ] **Step 2: Inspect the final diff and artifact sizes.**

Run: `git status --short && git diff --check && git diff --stat && find results/connect4 -type f -maxdepth 1 -exec stat -f '%z %N' {} \;`

Expected: no checkpoint binaries, logs, archives, caches, private paths, or oversized raw products.

- [ ] **Step 3: Commit and push.**

Run: `git add docs/superpowers/plans/2026-08-24-connect4-lightzero-results.md scripts/build_connect4_lightzero_results.py tests/scripts/test_connect4_lightzero_results.py results/connect4 reports/connect4_lightzero.md && git commit -m 'data(connect4): publish interim LightZero Elo snapshot'`

Run: `git push -u origin codex/connect4-lightzero-results`

Expected: branch is available on origin with a clean worktree.

- [ ] **Step 4: Open the pull request.**

Run: `gh pr create --base main --head codex/connect4-lightzero-results --title 'data(connect4): publish interim LightZero Elo snapshot' --body-file <prepared-body>`

Expected: PR URL returned; body reports data horizon, four-seed scope, validation, and limitations.
