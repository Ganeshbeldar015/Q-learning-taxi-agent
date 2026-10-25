---
status: complete
owner: test-author
task: T-0004
inputs:
  - harness/artifacts/T-0004/01-spec.md
  - harness/artifacts/T-0004/03-plan.md
acceptance_refs:
  - AC-007-a
  - AC-007-b
  - AC-008-a
  - AC-008-b
  - AC-009-a
  - AC-009-b
  - AC-010-a
---
# 04-tests — T-0004: Red Test Run Evidence

## 1. Acceptance Criteria to Test Mapping

| Acceptance Criterion | Test Function | Description |
|---|---|---|
| `AC-007-a` | `tests/test_evaluate.py::test_compute_bootstrap_ci` | Validates 95% bootstrap confidence interval computation. |
| `AC-007-a` | `tests/test_evaluate.py::test_evaluate_policy_untrained_smoke` | Validates policy evaluation return schema and metrics calculation. |
| `AC-007-a` | `tests/test_evaluate.py::test_evaluate_policy_loaded_q_table` | Validates evaluation using pre-loaded NumPy Q-table. |
| `AC-007-b` | `tests/test_evaluate.py::test_generate_comparison_table` | Validates 4-policy comparison table generation and CSV output. |
| `AC-008-a` | `tests/test_plots.py::test_plot_v3_epsilon_schedule` | Validates V3 epsilon schedule figure generation. |
| `AC-008-a` | `tests/test_plots.py::test_plot_v6_and_v7_distributions` | Validates V6 and V7 reward and steps distribution histogram generation. |
| `AC-008-a` | `tests/test_plots.py::test_plot_v8_policy_grid` | Validates V8 5x5 spatial action arrow grid generation. |
| `AC-010-a` | Pytest test suite execution across all test files | Validates zero failures and complete regression safety. |

## 2. Red Run Captured Output
```
collected 0 items / 2 errors
ERROR tests/test_evaluate.py - ModuleNotFoundError: No module named 'src.evaluate'
ERROR tests/test_plots.py - ModuleNotFoundError: No module named 'src.plots'
!!!!!!!!!!!!!!!!!!! Interrupted: 2 errors during collection !!!!!!!!!!!!!!!!!!!
============================== 2 errors in 0.43s ==============================
```

Red tests confirmed failing cleanly due to missing source modules. Handing off to Builder.
