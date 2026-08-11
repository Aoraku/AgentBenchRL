# Changelog

All notable changes to AgentBenchRLFrame are documented in this file. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Games are now discovered as runtime plugins: `rlbench.registry` scans the
  `games` namespace for `games/<name>/plugin.py` (`GamePlugin`) instead of
  importing a concrete game, so adding a game needs no framework-source edits.
- Game configuration schema is declared by `GamePlugin.config_schema` rather
  than a hardcoded `config._GAME_SCHEMAS`; the redundant
  `configs/games/snakego.yaml` was removed.
- Population process protocols are self-declared by game plugins
  (`official_protocols`); the CLI resolves them by name, removing the
  `snakego_official` special case.
- SnakeGo-specific pipelines, the official binary process policy, and the
  SnakeGo scripts moved out of the framework package into `games/snakego/`.
- The CLI (`rlbench.cli.main`) was split into focused subcommand and helper
  modules with an unchanged public import surface.

### Documentation

- `results/` and `reports/` are documented as committed, content-addressed
  audit evidence (backed up under `experiments-data/`), and only reproducible
  training output (`runs/`) is ignored.

## [0.1.0] - 2026-08-06

### Added

- Six-method discrete-game contract and deterministic SnakeGo plugin.
- AlphaZero and Tianshou PPO training backends with resumable checkpoints.
- Immutable population manifests, side-balanced evaluation, Elo and win-rate
  promotion gates, telemetry, and fact-derived reports.
- Reproducible SnakeGo strength artifacts and external-data population
  blueprints without redistributed contestant source bytes.
