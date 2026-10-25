---
status: complete
owner: builder
task: T-0002
inputs:
  - harness/artifacts/T-0002/03-plan.md
  - harness/artifacts/T-0002/04-tests.md
acceptance_refs:
  - AC-003-a
  - AC-003-b
  - AC-003-c
  - AC-004-a
  - AC-004-b
---
# 05-build — T-0002: Implementation Summary

## 1. What Changed and Why
- **`src/agent.py`**:
  - Implemented `QLearningAgent` with tabular Q-table of shape `(500, 6)`.
  - Implemented `update` adhering strictly to Bellman target: zero future bootstrap on `terminated=True`; future value bootstrap preserved on `truncated=True`.
  - Implemented `greedy_action` using `np.flatnonzero` and `rng.choice` for uniform random tie-breaking.
  - Implemented `save` and `load` for `.npy` Q-table persistence.
- **`src/schedules.py`**:
  - Implemented `MultiplicativeSchedule` ($\epsilon = \max(\epsilon_{\text{min}}, \text{start} \times \text{decay}^t)$).
  - Implemented `LinearSchedule` ($\epsilon$ linearly decaying to $\epsilon_{\text{min}}$ over $N$ episodes).
  - Implemented `get_schedule` factory function.
- **`src/train.py`**:
  - Implemented `evaluate_greedy` evaluating pure exploitation on fixed evaluation seeds without modifying the Q-table.
  - Implemented `check_convergence` detecting when validation reward sustains $\ge 7.0$ across 3 consecutive evaluation cycles.
  - Implemented `train` logging episodic metrics to CSV and saving Q-table.
- **Artifacts Created**:
  - `results/training_log.csv`: Full 5,000-episode metrics.
  - `results/q_table_baseline.npy`: Checkpoint of converged model.

## 2. Empirical Verification
- 5,000-episode training run completed in 17.78s.
- Validation reward reached **8.34** (exceeding target 7.5).
- Convergence detected at **Episode 2300**.
- All 57 tests passing.
