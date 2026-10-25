---
status: complete
owner: designer
task: T-0004
inputs:
  - harness/artifacts/T-0004/01-spec.md
acceptance_refs:
  - AC-007-a
  - AC-007-b
  - AC-008-a
  - AC-008-b
  - AC-009-a
  - AC-009-b
  - AC-010-a
---
# 02-design — T-0004: Evaluation & Visualization Architecture

## 1. Module Architecture & Interface Specifications

### 1.1 `src/evaluate.py`
```python
def compute_bootstrap_ci(
    data: Sequence[float],
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute non-parametric bootstrap confidence interval for the mean."""
    ...

def evaluate_policy(
    policy_source: QLearningAgent | np.ndarray | str | None,
    episodes: int = 1000,
    seed_offset: int = 10000,
    max_steps: int = 200,
    env_name: str = "Taxi-v4",
) -> dict[str, Any]:
    """Evaluate a greedy policy (or untrained/random) across deterministic test seeds.
    Returns:
        {
            "mean_reward": float,
            "std_reward": float,
            "ci_95": [float, float],
            "mean_steps": float,
            "std_steps": float,
            "success_rate": float,
            "total_illegal_actions": int,
            "mean_illegal_actions": float,
            "episodes": int,
            "raw_rewards": list[float],
            "raw_steps": list[int],
        }
    """
    ...

def generate_comparison_table(
    results_dir: str = "results",
    out_csv_path: str = "results/comparison_table.csv",
) -> pd.DataFrame:
    """Benchmark 4 policies (Random, Untrained, Baseline, Tuned) into a unified DataFrame."""
    ...
```

### 1.2 `src/plots.py`
All visualization routines save 150+ DPI figures to `results/figures/`:
- `plot_v1_learning_curve(final_runs_dir: str, out_path: str)`: Aggregates 5 seeds, plots mean training and validation reward curves with shaded standard deviation envelopes.
- `plot_v2_steps_per_episode(training_log_path: str, out_path: str)`: Moving average of episode steps demonstrating policy compression.
- `plot_v3_epsilon_schedule(config_path: str, episodes: int, out_path: str)`: Dual plot of multiplicative decay curve and floor cut-off.
- `plot_v4_validation_curve(training_log_path: str, out_path: str)`: Periodic greedy evaluations, threshold marker line ($+7.0$), and annotated convergence episode.
- `plot_v5_tuning_heatmap(tuning_csv_path: str, out_path: str)`: Alpha vs Gamma parameter grid showing mean episodes to convergence.
- `plot_v6_test_reward_distribution(raw_rewards: list[float], out_path: str)`: Frequency histogram with KDE, median line, and empirical bounds.
- `plot_v7_test_steps_distribution(raw_steps: list[int], out_path: str)`: Step count distribution highlighting optimal path concentration ($\le 14$ steps).
- `plot_v8_policy_grid(q_table: np.ndarray, passenger_loc: int, dest_loc: int, out_path: str)`: 5x5 spatial grid layout showing directional navigation arrows ($\uparrow, \downarrow, \rightarrow, \leftarrow$, pickup, dropoff) from every taxi cell.
- `plot_v9_replay_gif(q_table: np.ndarray, out_gif_path: str, seed: int = 10001)`: Visual frame-by-frame animated demonstration of complete pickup-dropoff run.
- `generate_all_figures(...)`: Batch generator executing V1 through V9 sequentially.

### 1.3 `notebooks/taxi_q_learning.ipynb` & `report/REPORT.md`
- Jupyter notebook with structured Markdown and code blocks reproducing:
  1. Environment exploration & Baseline results
  2. Q-learning agent formulation & Training loop
  3. Hyperparameter tuning & Multi-seed ablation
  4. Final 1,000-episode evaluation & Comparative metrics
  5. Interactive visualizer & Rollout demonstration
- Technical report in `report/REPORT.md` presenting complete mathematical derivations, experimental setup, empirical tables, and error analyses.
