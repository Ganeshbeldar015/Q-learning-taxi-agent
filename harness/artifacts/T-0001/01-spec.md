---
status: complete
owner: po-spec
task: T-0001
inputs:
  - docs/raw.md
  - docs/harness.md
  - harness/tasks/T-0001.md
acceptance_refs:
  - AC-001-a
  - AC-001-b
  - AC-002-a
  - AC-002-b
  - AC-002-c
---
# 01-spec — T-0001: Walking Skeleton & Environment Exploration

## 1. Problem & Intent
Initialize the foundational execution environment for tabular reinforcement learning on Gymnasium Taxi. We establish configuration loading, reproducible random number generation, bidirectional state encoding/decoding, and a 1,000-episode random agent benchmark that sets the empirical performance baseline.

## 2. Scope & Boundaries
- **In Scope**:
  - `config/default.yaml` containing baseline parameters for training and evaluation.
  - Configuration loader with type and key presence validation.
  - Seeding utility returning `np.random.Generator` and seeding environment on first reset.
  - State decode `(taxi_row, taxi_col, pass_loc, dest)` and encode functions with round-trip verification across all 500 states.
  - 1,000-episode random baseline rollout saving metrics to `results/baseline_random.json`.
  - Invariant assertion: reward $= 21 - \text{steps}$ for successful episodes with zero illegal actions.
- **Out of Scope**:
  - `QLearningAgent` class, Bellman equation updates, epsilon schedules (deferred to T-0002).
  - Visualization plots and notebook UI (deferred to T-0004).

## 3. User Stories
- **As an ML Practitioner**, I want a single function `make_env(config)` that works seamlessly regardless of whether the environment is registered as Taxi-v3 or Taxi-v4.
- **As an Evaluator**, I want a seeded 1,000-episode random baseline saved to disk so I can empirically prove the learning agent outperforms chance.

## 4. Acceptance Criteria & Scenarios

### AC-001-a: Configuration Validation
- **Given** `config/default.yaml`,
- **When** loaded via `load_config(path)`,
- **Then** it validates presence and types of `env`, `seed`, `train` dictionary (episodes, alpha, gamma, epsilon_start, epsilon_min, epsilon_decay, schedule, max_steps), and `eval` dictionary (every, episodes, test_episodes, convergence_threshold).

### AC-001-b: Deterministic Seeding
- **Given** an integer seed `seed=42`,
- **When** calling `get_seeded_rng(seed)`,
- **Then** a `np.random.Generator` is returned; running two independent environments with this seed produces identical initial observation sequences.

### AC-002-a: State Encode/Decode Invariant
- **Given** discrete state integer $s \in [0, 499]$,
- **When** `decode_state(s)` returns `(row, col, pass_loc, dest)`,
- **Then** `encode_state(row, col, pass_loc, dest) == s` for all 500 valid states.

### AC-002-b: Seeded Random Policy Baseline
- **Given** a purely random agent selecting uniformly among actions $0..5$,
- **When** evaluated for 1,000 episodes using `eval.episodes` and `eval.test_episodes` conventions,
- **Then** results are saved to `results/baseline_random.json` recording `mean_reward` ($< -200$), `mean_steps`, `success_rate`, `truncation_rate`, and `illegal_action_count`.

### AC-002-c: Analytical Reward Invariant
- **Given** any episode where `terminated=True` (successful passenger drop-off) and `illegal_actions == 0`,
- **When** summing step rewards,
- **Then** total reward strictly equals $21 - \text{steps}$.

## 5. Boundary & Edge Cases
- State 0 and State 499 edge boundaries.
- Passenger in taxi (`pass_loc == 4`) vs at landmarks R (0), G (1), Y (2), B (3).
- Truncation at step 200 without drop-off.
