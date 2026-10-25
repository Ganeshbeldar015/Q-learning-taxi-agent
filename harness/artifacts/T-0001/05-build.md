---
status: complete
owner: builder
task: T-0001
inputs:
  - harness/artifacts/T-0001/03-plan.md
  - harness/artifacts/T-0001/04-tests.md
acceptance_refs:
  - AC-001-a
  - AC-001-b
  - AC-002-a
  - AC-002-b
  - AC-002-c
---
# 05-build — T-0001: Implementation Summary

## 1. What Changed and Why
- **`requirements.txt`**: Added pinned dependencies for gymnasium, numpy, pandas, matplotlib, pyyaml, pytest, ruff, and mypy.
- **`config/default.yaml`**: Created baseline configuration with seed=42, Taxi-v4, training parameters, and evaluation schedules.
- **`src/__init__.py`**: Initialized Python package structure.
- **`src/env_utils.py`**:
  - Implemented `load_config` with strict type checking and required key validation.
  - Implemented `get_seeded_rng(seed)` returning a NumPy Generator.
  - Implemented `make_env` with automatic fallback adapter handling Taxi-v3 / Taxi-v4 compatibility per ADR-0001.
  - Implemented bidirectional `decode_state` and `encode_state` formulas and `verify_state_roundtrip`.
  - Implemented `run_random_baseline` evaluating 1,000 episodes and saving empirical results.
- **`results/baseline_random.json`**: Generated 1,000-episode random baseline metrics.

## 2. Gate G1 Verification Commands & Results
- **Formatting**: `python -m ruff format --check src tests` &rarr; 0 errors, all files formatted.
- **Linting**: `python -m ruff check src tests` &rarr; 0 errors.
- **Type Checking**: `python -m mypy src --ignore-missing-imports` &rarr; Success: no issues found.
- **Unit Tests**: `pytest tests/test_env.py -v` &rarr; 7 passed in 1.13s.

## 3. Empirical Baseline Metrics Summary
```json
{
  "policy": "random",
  "episodes": 1000,
  "seed": 42,
  "mean_reward": -772.629,
  "std_reward": 98.968,
  "mean_steps": 196.563,
  "median_steps": 200.0,
  "success_rate": 0.05,
  "truncation_rate": 0.95,
  "illegal_actions": 64124
}
```
All AC criteria satisfied; ready for review.
