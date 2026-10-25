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

## Empirical Benchmarks

### Random Policy Baseline (1,000 Episodes)
Source: [results/baseline_random.json](results/baseline_random.json)

| Metric | Random Baseline | Target (Tuned Greedy) |
|---|---|---|
| Average Reward | -772.63 | $\ge 7.5$ |
| Reward Std | 98.97 | — |
| Average Steps | 196.56 | $\le 14$ |
| Success Rate | 5.0% | 100% |
| Truncation Rate | 95.0% | 0% |
| Illegal Actions | 64,124 | 0 |

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
