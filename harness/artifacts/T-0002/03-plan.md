---
status: complete
owner: tech-lead
task: T-0002
inputs:
  - harness/artifacts/T-0002/01-spec.md
  - harness/artifacts/T-0002/02-design.md
acceptance_refs:
  - AC-003-a
  - AC-003-b
  - AC-003-c
  - AC-004-a
  - AC-004-b
---
# 03-plan — T-0002: Implementation Roadmap & Test Strategy

## 1. File Modification Roadmap

| File | Purpose |
|---|---|
| `src/agent.py` | `QLearningAgent` with vectorized Bellman update, tie-breaking, and save/load. |
| `src/schedules.py` | `MultiplicativeSchedule` and `LinearSchedule` classes with floor enforcement. |
| `src/train.py` | `train` loop, periodic validation on fixed seeds, CSV logging, and convergence detector. |
| `tests/test_agent.py` | Unit tests for Bellman update (hand-computed, terminal vs truncated), tie-breaking, and save/load. |
| `tests/test_schedules.py` | Unit tests for decay schedules at episode boundaries ($t=0, t=1000, t=10000$). |
| `tests/test_train.py` | Fast integration tests for training loop, CSV creation, and convergence detection. |

## 2. Lane Decomposition & Ownership Locks
- **Lane**: `lane-agent-train`
- **Owner**: `builder`
- **Exclusive Paths**:
  - `src/agent.py`
  - `src/schedules.py`
  - `src/train.py`
  - `results/training_log.csv`

## 3. Test Strategy for Test Author
- `tests/test_agent.py`:
  - `test_bellman_update_hand_computed`: Verify update on $s=0, a=1, r=-1, s'=5, \alpha=0.1, \gamma=0.99$.
  - `test_bellman_update_terminal_no_bootstrap`: When `terminated=True`, future return is 0 regardless of $Q(s')$.
  - `test_bellman_update_truncated_with_bootstrap`: When `truncated=True`, target bootstraps $\gamma \max Q(s')$.
  - `test_random_tie_breaking`: Verify uniform action distribution over 1,000 choices when all Q-values are 0.
  - `test_q_table_save_and_load`: Save to `.npy` and verify bitwise identical reload.
- `tests/test_schedules.py`:
  - `test_multiplicative_schedule_decay_and_floor`: Check monotonicity and `min_epsilon`.
  - `test_linear_schedule_decay_and_floor`: Check linear slope and boundary floor.
- `tests/test_train.py`:
  - `test_train_loop_creates_log_and_improves`: Run 50 episodes, check CSV format and headers.
  - `test_convergence_detection`: Synthetic validation scores to verify convergence patience logic.

## 4. Rollback & Feasibility
All additions are self-contained modular files in `src/` and `tests/`. No breaking changes to existing `src/env_utils.py`.
