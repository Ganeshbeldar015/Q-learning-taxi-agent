---
status: complete
owner: builder
task: T-0004
inputs:
  - harness/artifacts/T-0004/03-plan.md
  - harness/artifacts/T-0004/04-tests.md
acceptance_refs:
  - AC-007-a
  - AC-007-b
  - AC-008-a
  - AC-008-b
  - AC-010-a
---
# 05-build — T-0004: Evaluation Benchmark & Visualizations

## 1. Summary of Changes
- Implemented `src/evaluate.py`:
  - `compute_bootstrap_ci`: Non-parametric bootstrap 95% confidence interval estimator for sample mean rewards.
  - `evaluate_policy`: Pure greedy rollout evaluator supporting Q-tables, agents, and untrained policies across deterministic unseen test seeds (10000–10999).
  - `generate_comparison_table`: Unifies Random Policy, Untrained Q-Agent, Baseline Trained Model, and Tuned Final Model into `results/comparison_table.csv`.
  - `run_evaluation_pipeline`: Automates execution, logging to `results/test_metrics.json` and generating the comparative benchmark CSV.
- Implemented `src/plots.py`:
  - Figures V1–V8 (PNG, 150+ DPI) and V9 (animated GIF) generated into `results/figures/`:
    - `V1_learning_curve.png`: 5-seed mean training reward trajectory with $\pm 1\sigma$ envelope.
    - `V2_steps_per_episode.png`: Step reduction curve showing policy compression towards optimal length.
    - `V3_epsilon_schedule.png`: Multiplicative decay curve with minimum exploration floor.
    - `V4_validation_curve.png`: Periodic greedy validation reward with threshold line (+7.0) and convergence marker (Ep 600).
    - `V5_tuning_heatmap.png`: Learning rate vs Discount factor parameter convergence speed matrix.
    - `V6_test_reward_distribution.png`: Histogram and KDE of rewards over 1,000 unseen test episodes.
    - `V7_test_steps_distribution.png`: Histogram of steps taken across 1,000 unseen test episodes.
    - `V8_policy_grid.png`: 5x5 spatial grid displaying learned directional action arrows.
    - `V9_replay.gif`: Step-by-step rollout animation demonstrating successful pickup and drop-off.

## 2. Benchmark Verification Evidence
- 1,000 unseen test episodes (seeds 10000–10999) on Tuned Final Model (`results/q_table_final.npy`):
  - **Mean Reward**: **7.87** (Target: $\ge 7.5$) — Exceeded
  - **95% Bootstrap CI**: **[7.69, 8.03]**
  - **Mean Steps**: **13.13** (Target: $\le 14.0$) — Exceeded
  - **Success Rate**: **100.0%** (Target: 100.0%) — Satisfied
  - **Total Illegal Actions**: **0** (Target: 0) — Satisfied
- Policy Comparative Table (`results/comparison_table.csv`):
  - Random Policy: Mean Reward -772.63, 196.6 steps, 5.0% success, 64,124 illegal actions
  - Untrained Q-Agent: Mean Reward -768.14, 196.3 steps, 4.8% success, 63,647 illegal actions
  - Baseline Model: Mean Reward 7.91, 13.1 steps, 100.0% success, 0 illegal actions
  - Tuned Model: Mean Reward 7.87, 13.1 steps, 100.0% success, 0 illegal actions

## 3. Test & Linter Status
- Pytest: 68 tests passing in 7.63s.
- Ruff: 0 errors; code fully formatted.
- Harness linter: 0 problems detected.
