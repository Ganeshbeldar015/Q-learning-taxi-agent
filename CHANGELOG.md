# Changelog

All notable changes to this project will be documented in this file.

## [Sprint 3] - 2026-10-25
### Added
- Implemented hyperparameter tuning and multi-seed retraining engine in `src/tune.py`.
- Conducted Stage A single-factor sweeps across $\alpha$, $\gamma$, and decay schedules, followed by Stage B grid search (63 trials total) saved to `results/tuning_results.csv`.
- Discovered optimal hyperparameters: $\alpha=0.7$, $\gamma=0.99$, $\text{decay}=0.999$, $\epsilon_{\text{min}}=0.01$, yielding convergence at episode 800 (nearly $3\times$ faster than baseline $\alpha=0.1$ at episode 2300) with mean validation reward +8.30.
- Exported winning configuration to `config/best.yaml`.
- Executed 10,000-episode retraining across 5 independent seeds (101–105) achieving 100% convergence across all seeds (mean convergence episode 800.0).
- Saved multi-seed checkpoints and training logs to `results/final_runs/` and exported primary production model to `results/q_table_final.npy` (derived from seed 104 with +8.34 validation reward).
- Added unit tests in `tests/test_tune.py` covering candidate selection algorithm, config generation, and multi-seed retraining pipelines (61 tests passing).

## [Sprint 2] - 2026-10-25
### Added
- Implemented `QLearningAgent` in `src/agent.py` with Bellman update rule, unbiased uniform random tie-breaking, and `.npy` save/load.
- Implemented `MultiplicativeSchedule` and `LinearSchedule` in `src/schedules.py` with floor constraints.
- Implemented `src/train.py` with greedy validation over fixed evaluation seeds, convergence detector (3-in-a-row $\ge 7.0$), and CSV episodic metric logging.
- Added comprehensive unit tests in `tests/test_agent.py`, `tests/test_schedules.py`, and `tests/test_train.py` (57 tests passing).
- Verified baseline training run reaching +8.34 validation reward with convergence at episode 2300 in 17.78 seconds.

## [Sprint 1] - 2026-10-25
### Added
- Created `requirements.txt` with pinned dependencies (`gymnasium`, `numpy`, `pandas`, `matplotlib`, `pyyaml`, `pytest`, `ruff`, `mypy`).
- Created baseline configuration in `config/default.yaml`.
- Implemented `src/env_utils.py` featuring:
  - `load_config`: Strict schema and type checking for YAML configs.
  - `get_seeded_rng`: Seeded NumPy Generator creation.
  - `make_env`: Backward compatible Taxi-v3 / Taxi-v4 instantiation per ADR-0001.
  - `decode_state` and `encode_state`: Bidirectional state index transformations.
  - `verify_state_roundtrip`: Exhaustive verification across all 500 Taxi states.
  - `run_random_baseline`: 1,000-episode random baseline evaluation.
- Added comprehensive unit tests in `tests/test_env.py` covering all acceptance criteria.
- Generated empirical benchmark `results/baseline_random.json`.
- Created project `README.md` with setup, verification instructions, and baseline benchmarks.

## [Sprint 0] - 2026-10-25
### Added
- Scaffolding of Harness Engineering infrastructure.
- Automated self-testing linter `scripts/check_harness.py`.
- Path guard `scripts/guard_paths.py` and circuit breaker `scripts/breaker.py`.
- Harness test suite in `tests/harness/` (39 tests passing).
