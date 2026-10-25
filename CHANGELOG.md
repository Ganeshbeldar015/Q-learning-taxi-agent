# Changelog

All notable changes to this project will be documented in this file.

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
