# Changelog

All notable changes to this project will be documented in this file.

## [Sprint 4] - 2026-10-25
### Added
- Implemented `src/evaluate.py` with 95% bootstrap confidence interval calculation, greedy test evaluation, and comparative policy benchmarking across 4 policies (Random, Untrained, Baseline, Tuned).
- Conducted 1,000 unseen test episodes (seeds 10000–10999) on the primary production model (`results/q_table_final.npy`): mean reward +7.87 (95% CI: [7.69, 8.03]), mean steps 13.13, 100.0% success rate, exactly 0 illegal actions.
- Generated `results/comparison_table.csv` and `results/test_metrics.json`.
- Implemented `src/plots.py` generating publication-quality figures V1–V8 (PNG, 150+ DPI) and V9 (animated GIF):
  - `V1_learning_curve.png` (5-seed mean training trajectory with $\pm 1\sigma$ band)
  - `V2_steps_per_episode.png` (policy length compression)
  - `V3_epsilon_schedule.png` (multiplicative decay & floor)
  - `V4_validation_curve.png` (validation reward with +7.0 threshold & convergence marker)
  - `V5_tuning_heatmap.png` (parameter convergence speed matrix)
  - `V6_test_reward_distribution.png` (test rewards histogram & KDE)
  - `V7_test_steps_distribution.png` (test steps histogram)
  - `V8_policy_grid.png` (5x5 spatial grid with directional policy arrows)
  - `V9_replay.gif` (animated greedy rollout demonstration)
- Created interactive Jupyter notebook `notebooks/taxi_q_learning.ipynb` executing end-to-end with 0 errors.
- Authored comprehensive technical report `report/REPORT.md` documenting mathematical formulation, ablation studies, and empirical benchmarks.
- Added unit tests in `tests/test_evaluate.py` and `tests/test_plots.py` (68 total tests passing).

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
