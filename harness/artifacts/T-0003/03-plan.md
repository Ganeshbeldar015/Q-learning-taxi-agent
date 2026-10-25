---
status: complete
owner: tech-lead
task: T-0003
inputs:
  - harness/artifacts/T-0003/01-spec.md
  - harness/artifacts/T-0003/02-design.md
acceptance_refs:
  - AC-005-a
  - AC-005-b
  - AC-006-a
---
# 03-plan — T-0003: Implementation Plan & Lane Decomposition

## 1. File Modification Roadmap

| File | Purpose |
|---|---|
| `src/tune.py` | Sweep executor (Stage A/B), multiprocessing worker pool, ranking selection, and final 5-seed training. |
| `config/best.yaml` | Output configuration containing winning hyperparameters. |
| `results/tuning_results.csv` | Full CSV of tuning trial results. |
| `results/final_runs/` | Checkpoints and logs for 5-seed training runs. |
| `results/q_table_final.npy` | Primary trained Q-table exported from the best validation run. |
| `tests/test_tune.py` | Unit tests for ranking logic, tie-breaking, best.yaml generation, and small sweep smoke tests. |

## 2. Lane Decomposition & Ownership Locks
- **Lane**: `lane-tuning`
- **Owner**: `builder`
- **Exclusive Paths**:
  - `src/tune.py`
  - `config/best.yaml`
  - `results/tuning_results.csv`
  - `results/final_runs/**`
  - `results/q_table_final.npy`

## 3. Test Strategy for Test Author
- `tests/test_tune.py`:
  - `test_select_best_configuration`: Verify selection ranking rule (highest conv rate, lowest episodes-to-convergence).
  - `test_save_best_config_format`: Verify `config/best.yaml` is valid YAML and matches schema.
  - `test_run_trial_execution`: Smoke test executing single trial for 20 episodes.
  - `test_final_5_seed_retraining_smoke`: Runs 5 seeds for small episode count and asserts `results/q_table_final.npy` creation.

## 4. Rollback & Feasibility
`src/tune.py` imports existing `train` and `QLearningAgent` without modifying their core APIs.
multiprocessing uses standard Python library `concurrent.futures`.
