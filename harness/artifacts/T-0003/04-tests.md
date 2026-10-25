---
status: complete
owner: test-author
task: T-0003
inputs:
  - harness/artifacts/T-0003/01-spec.md
  - harness/artifacts/T-0003/03-plan.md
acceptance_refs:
  - AC-005-a
  - AC-005-b
  - AC-006-a
---
# 04-tests — T-0003: Red Test Run Evidence

## 1. Acceptance Criteria to Test Mapping

| Acceptance Criterion | Test Function | Description |
|---|---|---|
| `AC-005-a` | `tests/test_tune.py::test_select_best_configuration` | Validates selection rule (lowest episodes-to-convergence among $\ge 80\%$ conv rate). |
| `AC-005-a` | `tests/test_tune.py::test_save_best_config` | Validates generation of valid `config/best.yaml`. |
| `AC-005-a` | `tests/test_tune.py::test_run_trial_execution` | Smoke test verifying single trial metric dictionary output. |
| `AC-006-a` | `tests/test_tune.py::test_retrain_final_models_smoke` | Smoke test verifying 5-seed training, checkpoint creation, and primary Q-table export. |

## 2. Red Run Captured Output
```
collected 0 items / 1 error
ERROR tests/test_tune.py - ModuleNotFoundError: No module named 'src.tune'
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
```
Red tests confirmed failing due to missing `src.tune`. Handing off to Builder.
