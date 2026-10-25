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

## Empirical Benchmarks

### Baseline Agent vs. Random Baseline
- Random Baseline: [results/baseline_random.json](results/baseline_random.json)
- Trained Baseline Model: [results/q_table_baseline.npy](results/q_table_baseline.npy) (trained 5,000 episodes, converged at Episode 2300)

| Metric | Random Policy | Trained Baseline (5,000 eps) | Target (Tuned Greedy) |
|---|---|---|---|
| Validation Reward | -772.63 | **+8.34** | $\ge 7.5$ |
| Reward Std | 98.97 | — | — |
| Convergence Episode | N/A (did not converge) | **2300** | < Baseline |
| Success Rate | 5.0% | **100%** | 100% |

## Project Structure
```
.
├── config/
│   └── default.yaml         # Baseline hyperparameters
├── src/
│   ├── __init__.py
│   └── env_utils.py         # make_env, state encode/decode, seeding, baseline
├── tests/
│   ├── test_env.py          # Environment, config, round-trip tests
│   └── harness/             # Harness self-verification tests
├── results/
│   └── baseline_random.json # 1,000-episode random baseline metrics
└── docs/                    # PRD, operating manual, and project memory
```
