# Taxi Route Optimization with Reinforcement Learning

End-to-end reinforcement learning project where a tabular Q-learning agent learns to solve the Gymnasium Taxi problem: picking up passengers and routing them to target destinations in minimal steps with zero illegal moves.

## Setup & Installation

### Prerequisites
- Python 3.10+ (tested on Python 3.14.7)
- Virtual environment recommended

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Running Verification & Tests

### Execute Gate G0 (Harness Integrity)
```bash
python scripts/check_harness.py && pytest tests/harness -v
```

### Run Unit Tests
```bash
pytest tests/ -v
```

### Run Random Baseline
```bash
python -c "from src.env_utils import make_env, run_random_baseline; env = make_env(); run_random_baseline(env, episodes=1000, seed=42, output_path='results/baseline_random.json'); env.close()"
```

### Run Training
```bash
python -c "from src.env_utils import load_config; from src.train import train; cfg = load_config(); train(cfg, log_csv_path='results/training_log.csv', q_table_save_path='results/q_table_baseline.npy')"
```

### Run Hyperparameter Tuning & Final Multi-Seed Retraining
```bash
python -c "from src.tune import run_tuning_pipeline, retrain_final_models; from src.env_utils import load_config; best_cfg = run_tuning_pipeline(); retrain_final_models(load_config('config/best.yaml'))"
```

### Run Evaluation Benchmark (1,000 Unseen Test Episodes)
```bash
python -m src.evaluate
```

### Generate Publication Figures (V1 through V9)
```bash
python -m src.plots
```

## Empirical Benchmarks

### 1,000 Unseen Test Episodes Comparison (Seeds 10000–10999)
Source: [results/comparison_table.csv](results/comparison_table.csv) and [results/test_metrics.json](results/test_metrics.json)

| Policy | Episodes | Mean Reward | Std Reward | 95% Bootstrap CI | Mean Steps | Success Rate | Illegal Actions |
|---|---|---|---|---|---|---|---|
| **Random Policy** | 1,000 | -772.63 | 98.97 | N/A | 196.6 | 5.0% | 64,124 |
| **Untrained Q-Agent** | 1,000 | -768.14 | 105.68 | [-774.66, -760.83] | 196.3 | 4.8% | 63,647 |
| **Baseline Trained Model** | 1,000 | **+7.91** | 2.57 | [7.75, 8.07] | **13.1** | **100.0%** | **0** |
| **Tuned Final Model** | 1,000 | **+7.87** | 2.61 | [7.69, 8.03] | **13.1** | **100.0%** | **0** |

### Key Benchmark Highlights
- **100.0% Success Rate**: The tuned agent reliably completes all 1,000 unseen test configurations.
- **Zero Illegal Actions**: Across 13,130 test actions, exactly 0 illegal pickups or dropoffs were attempted.
- **Optimal Step Bound**: Mean path length of 13.13 steps beats the project requirement of $\le 14.0$ steps.
- **Convergence Speedup**: Hyperparameter tuning discovered $\alpha=0.7$ achieves convergence in **800 episodes** ($2.8\times$ faster than the baseline $\alpha=0.1$ at episode 2,300).

## Deliverables & Visualizations
- **Formal Technical Report**: [report/REPORT.md](report/REPORT.md)
- **Interactive Jupyter Notebook**: [notebooks/taxi_q_learning.ipynb](notebooks/taxi_q_learning.ipynb)
- **Publication Figures**:
  - [V1: Learning Curve Across 5 Seeds](results/figures/V1_learning_curve.png)
  - [V2: Steps per Episode Compression](results/figures/V2_steps_per_episode.png)
  - [V3: Epsilon Exploration Schedule](results/figures/V3_epsilon_schedule.png)
  - [V4: Validation Reward & Convergence Marker](results/figures/V4_validation_curve.png)
  - [V5: Hyperparameter Tuning Sweeps Heatmap](results/figures/V5_tuning_heatmap.png)
  - [V6: Test Reward Distribution Histogram](results/figures/V6_test_reward_distribution.png)
  - [V7: Test Steps Distribution Histogram](results/figures/V7_test_steps_distribution.png)
  - [V8: 5x5 Spatial Policy Action Map](results/figures/V8_policy_grid.png)
  - [V9: Animated Replay GIF](results/figures/V9_replay.gif)

## Project Structure
```
.
├── config/
│   ├── default.yaml                  # Baseline hyperparameters (alpha=0.1, gamma=0.99)
│   └── best.yaml                     # Winning tuned hyperparameters (alpha=0.7, gamma=0.99)
├── src/
│   ├── __init__.py
│   ├── env_utils.py                  # make_env, state encode/decode, seeding, baseline
│   ├── agent.py                      # QLearningAgent with vectorized tie-breaking
│   ├── schedules.py                  # Multiplicative and linear decay schedules
│   ├── train.py                      # Training loop, evaluation, convergence detector
│   ├── tune.py                       # Single-factor sweep, grid search, multi-seed retrain
│   ├── evaluate.py                   # Greedy test evaluation & comparison table generator
│   └── plots.py                      # Publication visualization figures V1-V9
├── notebooks/
│   └── taxi_q_learning.ipynb         # Interactive execution & demonstration notebook
├── report/
│   └── REPORT.md                     # Comprehensive technical report
├── tests/
│   ├── test_env.py                   # Environment, config, round-trip tests
│   ├── test_agent.py                 # QLearningAgent Bellman equation & tie-breaking tests
│   ├── test_schedules.py             # Decay schedules & floor behavior tests
│   ├── test_train.py                 # Training loop and convergence detector tests
│   ├── test_tune.py                  # Tuning selection and retrain tests
│   ├── test_evaluate.py              # Test evaluation and bootstrap CI tests
│   ├── test_plots.py                 # Plotting routine smoke tests
│   └── harness/                      # Harness self-verification tests
├── results/
│   ├── baseline_random.json          # 1,000-episode random baseline metrics
│   ├── training_log.csv              # Baseline training logs
│   ├── q_table_baseline.npy          # Baseline Q-table
│   ├── tuning_results.csv            # 63 hyperparameter trial results
│   ├── final_runs/                   # 5-seed final retraining checkpoints and logs
│   ├── q_table_final.npy             # Primary winning production Q-table
│   ├── test_metrics.json             # 1,000 unseen test episodes metrics
│   ├── comparison_table.csv          # Comparative benchmarks across 4 policies
│   └── figures/                      # Publication figures V1-V8 PNGs and V9 replay GIF
└── docs/                             # PRD, operating manual, and project memory
```
