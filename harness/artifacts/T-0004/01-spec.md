---
status: complete
owner: po-spec
task: T-0004
inputs:
  - docs/raw.md
  - docs/harness.md
  - harness/tasks/T-0004.md
acceptance_refs:
  - AC-007-a
  - AC-007-b
  - AC-008-a
  - AC-008-b
  - AC-009-a
  - AC-009-b
  - AC-010-a
---
# 01-spec — T-0004: Evaluation Benchmark, Visualizations & Deliverables

## 1. Problem Statement & Functional Requirements
Complete the final project phase by evaluating the tuned Q-learning model across 1,000 unseen test seeds, generating publication-grade visualization figures (V1 through V9), establishing a comparative policy table, creating an interactive Jupyter notebook (`notebooks/taxi_q_learning.ipynb`), and synthesizing the formal technical report (`report/REPORT.md`).

## 2. Detailed Acceptance Criteria
1. **`AC-007-a`**:
   - Evaluate primary model (`results/q_table_final.npy`) with greedy action selection ($\epsilon=0$) over 1,000 test episodes with deterministic seeds starting at offset 10000 (seeds 10000 to 10999).
   - Compute:
     - Mean reward: target $\ge 7.5$
     - Mean steps: target $\le 14.0$
     - Success rate: target $100\%$
     - Illegal actions: target $0$
     - Bootstrap 95% confidence interval for mean reward
   - Output saved to `results/test_metrics.json`.
2. **`AC-007-b`**:
   - Construct comparative benchmark table contrasting 4 policies:
     1. Random Policy (from `results/baseline_random.json`)
     2. Untrained Agent (all-zero Q-table with uniform random tie-breaking)
     3. Baseline Trained Model (`results/q_table_baseline.npy`)
     4. Tuned Winning Model (`results/q_table_final.npy`)
   - Save table to `results/comparison_table.csv`.
3. **`AC-008-a`**:
   - Generate publication-quality figures at $\ge 150$ DPI in `results/figures/`:
     - `V1_learning_curve.png`: Mean reward vs episode with standard deviation band across the 5 retrained seeds.
     - `V2_steps_per_episode.png`: Episode length / steps curve showing convergence to optimal path length.
     - `V3_epsilon_schedule.png`: Exploration decay schedule showing initial value, decay rate, and floor.
     - `V4_validation_curve.png`: Greedy validation reward trajectory with horizontal convergence threshold line at $+7.0$.
     - `V5_tuning_heatmap.png`: Learning rate ($\alpha$) vs discount factor ($\gamma$) performance heatmap / parameter sweep curves.
     - `V6_test_reward_distribution.png`: Histogram and KDE of rewards over the 1,000 unseen test episodes.
     - `V7_test_steps_distribution.png`: Histogram and KDE of steps taken across the 1,000 unseen test episodes.
     - `V8_policy_grid.png`: 5x5 grid map with action arrows depicting learned optimal spatial routing for key passenger/destination states.
4. **`AC-008-b`**:
   - `V9_replay.gif`: Rendered animated GIF capturing step-by-step rollout of a greedy agent executing a complete pickup and drop-off episode.
5. **`AC-009-a`**:
   - Interactive Jupyter notebook `notebooks/taxi_q_learning.ipynb` containing complete narrative, live model evaluation, and embedded figures.
6. **`AC-009-b`**:
   - Formal technical report `report/REPORT.md` referencing exact numbers and tables matching `results/`.
7. **`AC-010-a`**:
   - Automated unit tests in `tests/test_evaluate.py` and `tests/test_plots.py` passing with zero errors.
