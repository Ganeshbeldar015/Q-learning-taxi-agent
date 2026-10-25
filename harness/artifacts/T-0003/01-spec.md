---
status: complete
owner: po-spec
task: T-0003
inputs:
  - docs/raw.md
  - docs/harness.md
  - harness/tasks/T-0003.md
acceptance_refs:
  - AC-005-a
  - AC-005-b
  - AC-006-a
---
# 01-spec — T-0003: Hyperparameter Search & Multi-Seed Final Training

## 1. Problem & Intent
Optimize the hyperparameters of the Q-learning agent to accelerate convergence and maximize final policy reward. The baseline configuration required 2300 episodes to converge. We execute a two-stage search (single-factor sweep and focused grid) to discover optimal values for $\alpha$, $\gamma$, and $\epsilon$-decay, saving `config/best.yaml`. Subsequently, we train the winning configuration across 5 independent seeds and persist `results/q_table_final.npy`.

## 2. Scope & Boundaries
- **In Scope**:
  - `src/tune.py`: Hyperparameter sweep runner supporting parallel workers via multiprocessing.
  - Stage A: One-factor-at-a-time search covering learning rate $\alpha \in [0.05, 0.1, 0.3, 0.5, 0.7]$, discount factor $\gamma \in [0.8, 0.9, 0.95, 0.99]$, and $\epsilon$-decay $\in [0.99, 0.995, 0.999, 0.9995]$.
  - Stage B: Focused grid of top parameter combinations evaluated across multiple seeds.
  - Objective evaluation & ranking: configurations with $\ge 80\%$ convergence rate and mean validation reward $\ge 7.0$, ranked by minimum episodes-to-convergence.
  - Export of all trial results to `results/tuning_results.csv`.
  - Output of winning parameters to `config/best.yaml`.
  - Final multi-seed training on 5 distinct seeds (e.g. 101, 102, 103, 104, 105), storing all checkpoints in `results/final_runs/` and primary Q-table in `results/q_table_final.npy`.
- **Out of Scope**:
  - Test set evaluation on seeds 10000–10999 (strictly deferred to `T-0004` to prevent hyperparameter overfit).
  - Visualization plots (deferred to `T-0004`).

## 3. Acceptance Criteria & Scenarios

### AC-005-a: Tuning Execution & Outputs
- **Given** search space in `raw.md §8`,
- **When** `tune()` executes,
- **Then** all tested trials are logged to `results/tuning_results.csv` with columns (`alpha`, `gamma`, `epsilon_decay`, `seed`, `converged`, `convergence_episode`, `final_val_reward`, `wall_clock_time`), and the winning parameters are written to `config/best.yaml`.

### AC-005-b: Convergence Speed Improvement
- **Given** baseline configuration with mean convergence episode $\approx 2300$,
- **When** evaluating the winning configuration in `config/best.yaml` across 5 training seeds,
- **Then** the mean episodes-to-convergence is strictly lower than the baseline configuration.

### AC-006-a: Multi-Seed Final Model Training
- **Given** `config/best.yaml`,
- **When** trained over 5 seeds (distinct from validation and test sets),
- **Then** checkpoints for each seed are saved in `results/final_runs/`, and the primary Q-table with highest validation reward is saved to `results/q_table_final.npy`.

## 4. Boundary & Constraint Rules
- Seed isolation: Training seeds (e.g. 101–105) must never intersect with validation seeds (5000–5049) or test seeds (10000–10999).
- Model selection: Primary model must be chosen strictly by validation reward, never by test reward.
