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

## Empirical Benchmarks

### Tuned Agent vs. Baseline Agent vs. Random Baseline
- Random Baseline: [results/baseline_random.json](results/baseline_random.json)
- Trained Baseline Model: [results/q_table_baseline.npy](results/q_table_baseline.npy) (trained 5,000 episodes, converged at Episode 2300)
- Winning Tuned Model: [results/q_table_final.npy](results/q_table_final.npy) (trained 10,000 episodes, converged at Episode 800 across 5 seeds)

| Metric | Random Policy | Trained Baseline ($\alpha=0.1$) | Tuned Winning Policy ($\alpha=0.7$) | Target (Tuned Greedy) |
|---|---|---|---|---|
| Validation Reward | -772.63 | **+8.34** | **+8.30** ($\pm 0.04$) | $\ge 7.5$ |
| Reward Std | 98.97 | — | 0.04 | — |
| Convergence Episode | N/A (did not converge) | 2300 | **800** ($2.8\times$ faster) | < Baseline |
| Convergence Rate | 0% | 100% | **100%** (5/5 seeds) | $\ge 80\%$ |
| Success Rate | 5.0% | **100%** | **100%** | 100% |

## Project Structure
```
.
├── config/
│   ├── default.yaml         # Baseline hyperparameters (alpha=0.1, gamma=0.99)
│   └── best.yaml            # Winning tuned hyperparameters (alpha=0.7, gamma=0.99)
├── src/
│   ├── __init__.py
│   ├── env_utils.py         # make_env, state encode/decode, seeding, baseline
│   ├── agent.py             # QLearningAgent with vectorized tie-breaking
│   ├── schedules.py         # Multiplicative and linear decay schedules
│   ├── train.py             # Training loop, evaluation, convergence detector
│   └── tune.py              # Single-factor sweep, grid search, multi-seed retrain
├── tests/
│   ├── test_env.py          # Environment, config, round-trip tests
│   ├── test_agent.py        # QLearningAgent Bellman equation & tie-breaking tests
│   ├── test_schedules.py    # Decay schedules & floor behavior tests
│   ├── test_train.py        # Training loop and convergence detector tests
│   ├── test_tune.py         # Tuning selection and retrain tests
│   └── harness/             # Harness self-verification tests
├── results/
│   ├── baseline_random.json # 1,000-episode random baseline metrics
│   ├── training_log.csv     # Baseline training logs
│   ├── q_table_baseline.npy # Baseline Q-table
│   ├── tuning_results.csv   # 63 hyperparameter trial results
│   ├── final_runs/          # 5-seed final retraining checkpoints and logs
│   └── q_table_final.npy    # Primary winning production Q-table
└── docs/                    # PRD, operating manual, and project memory
```
