---
status: complete
owner: test-author
task: T-0002
inputs:
  - harness/artifacts/T-0002/01-spec.md
  - harness/artifacts/T-0002/03-plan.md
acceptance_refs:
  - AC-003-a
  - AC-003-b
  - AC-003-c
  - AC-004-a
  - AC-004-b
---
# 04-tests — T-0002: Red Test Run Evidence

## 1. Acceptance Criteria to Test Mapping

| Acceptance Criterion | Test File & Function | Description |
|---|---|---|
| `AC-003-a` | `tests/test_agent.py::test_bellman_update_hand_computed` | Bellman update matches hand-calculated values. |
| `AC-003-a` | `tests/test_agent.py::test_bellman_update_terminal_no_bootstrap` | Zero bootstrap on `terminated=True`. |
| `AC-003-a` | `tests/test_agent.py::test_bellman_update_truncated_with_bootstrap` | Future value bootstrapped on `truncated=True`. |
| `AC-003-b` | `tests/test_agent.py::test_random_tie_breaking` | Uniform action selection over 1,200 trials on equal Q. |
| `AC-003-c` | `tests/test_schedules.py::test_multiplicative_schedule_decay_and_floor` | Multiplicative decay and floor enforcement. |
| `AC-003-c` | `tests/test_schedules.py::test_linear_schedule_decay_and_floor` | Linear decay slope and boundary bounds. |
| `AC-004-a` | `tests/test_train.py::test_train_creates_csv_and_saves_q_table` | Training loop emits CSV with all required metrics. |
| `AC-004-b` | `tests/test_train.py::test_convergence_detector_logic` | Convergence detected after 3 consecutive evaluations $\ge 7.0$. |

## 2. Red Run Captured Evidence
```
collected 0 items / 3 errors
ERROR tests/test_agent.py - ModuleNotFoundError: No module named 'src.agent'
ERROR tests/test_schedules.py - ModuleNotFoundError: No module named 'src.schedules'
ERROR tests/test_train.py - ModuleNotFoundError: No module named 'src.train'
```
Red tests confirmed failing due to absent implementation. Handing off to Builder.
