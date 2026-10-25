---
status: complete
owner: po-spec
task: T-0002
inputs:
  - docs/raw.md
  - docs/harness.md
  - harness/tasks/T-0002.md
acceptance_refs:
  - AC-003-a
  - AC-003-b
  - AC-003-c
  - AC-004-a
  - AC-004-b
---
# 01-spec — T-0002: Tabular Q-Learning Agent & Training Pipeline

## 1. Problem & Intent
Implement the mathematical core of tabular Q-learning to solve the Taxi environment. The agent maintains a Q-table of shape `(500, 6)` and learns optimal route policies via TD learning. An exploration schedule decays exploration over time, and a modular training loop tracks convergence through periodic greedy validation.

## 2. Scope & Boundaries
- **In Scope**:
  - `src/agent.py`: `QLearningAgent` class managing Q-table initialization, Bellman temporal-difference update, tie-breaking greedy selection, and persistence (`.npy`).
  - `src/schedules.py`: Multiplicative and linear epsilon decay schedules.
  - `src/train.py`: Training engine supporting episodic metric logging to CSV, periodic greedy validation on fixed seeds (offset 5000), convergence tracking, and model checkpointing.
  - Comprehensive unit tests covering Bellman calculations, schedule behavior, tie-breaking, and training mechanics.
- **Out of Scope**:
  - Hyperparameter sweeps across multiple parallel workers (`src/tune.py`, deferred to `T-0003`).
  - Evaluation on unseen test seeds (offset 10000) and publication figures (`src/evaluate.py`, `src/plots.py`, deferred to `T-0004`).

## 3. User Stories
- **As a Student / Researcher**, I want a transparent, from-scratch NumPy implementation of Q-learning without framework abstractions so that the Bellman equation and tie-breaking dynamics can be mathematically inspected.
- **As an RL Practitioner**, I want training to detect convergence automatically based on greedy performance on fixed validation seeds.

## 4. Acceptance Criteria & Scenarios

### AC-003-a: Bellman Equation & Termination vs Truncation
- **Given** transition $(s, a, r, s', \text{terminated}, \text{truncated})$,
- **When** `agent.update(s, a, r, s', terminated, truncated)` executes,
- **Then** the Q-table update follows:
  $$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') \cdot (1 - \text{terminated}) - Q(s, a) \right]$$
  Specifically:
  - If `terminated=True`, the target is $r$ (zero future bootstrap).
  - If `truncated=True` (and not terminated), the target is $r + \gamma \max_{a'} Q(s', a')$ (full future bootstrap).

### AC-003-b: Uniform Random Tie-Breaking
- **Given** a state $s$ where multiple actions share the maximal Q-value (including initial all-zero Q-table),
- **When** `agent.greedy_action(s, rng)` is evaluated repeatedly over 1,000 trials,
- **Then** the distribution across maximal actions is statistically uniform rather than biased toward action 0.

### AC-003-c: Epsilon Decay Schedules
- **Given** epsilon configuration parameters:
  - Multiplicative: $\epsilon(t) = \max(\epsilon_{\text{min}}, \epsilon_{\text{start}} \cdot \text{decay}^t)$
  - Linear: $\epsilon(t) = \max(\epsilon_{\text{min}}, \epsilon_{\text{start}} - t \cdot \Delta)$
- **When** querying epsilon across episodes $t \in [0, N]$,
- **Then** values decrease monotonically and strictly bound at $\epsilon_{\text{min}}$.

### AC-004-a: Training Loop & Episodic Logging
- **Given** valid training configuration from `config/default.yaml`,
- **When** calling `train(config)`,
- **Then** per-episode metrics (`episode`, `reward`, `steps`, `epsilon`, `illegal_actions`, `success`) are logged to CSV (`results/training_log.csv`).

### AC-004-b: Greedy Validation & Convergence Detector
- **Given** evaluation interval `eval.every` (100) and patience `eval.convergence_patience` (3),
- **When** greedy validation reward over `eval.episodes` (50) on validation seeds ($\ge 5000$) meets or exceeds `convergence_threshold` (7.0) for 3 consecutive intervals,
- **Then** the earliest episode achieving this threshold is logged as the convergence episode.

## 5. Boundary Conditions
- Hand-computed step updates: e.g., $s=0, a=1, r=-1, s'=5$, $\alpha=0.1, \gamma=0.99$.
- $Q(s', \cdot) = [0, 2, 5, 1, 0, 0] \implies \max = 5$.
  Target $= -1 + 0.99 \times 5 = 3.95$. New $Q(0, 1) = 0 + 0.1 \times 3.95 = 0.395$.
- Monotonicity checks at $t=0, t=1000, t=10000$.
