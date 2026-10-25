"""Tabular Q-learning agent with random tie-breaking and Bellman updates."""

from pathlib import Path

import numpy as np


class QLearningAgent:
    """Tabular Q-learning agent for discrete observation and action spaces."""

    def __init__(
        self,
        num_states: int = 500,
        num_actions: int = 6,
        alpha: float = 0.1,
        gamma: float = 0.99,
    ) -> None:
        self.num_states = int(num_states)
        self.num_actions = int(num_actions)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.q_table: np.ndarray = np.zeros(
            (self.num_states, self.num_actions), dtype=np.float64
        )

    def select_action(
        self, state: int, epsilon: float, rng: np.random.Generator
    ) -> int:
        """Select action using epsilon-greedy strategy with random tie-breaking."""
        if rng.random() < epsilon:
            return int(rng.integers(0, self.num_actions))
        return self.greedy_action(state, rng)

    def greedy_action(self, state: int, rng: np.random.Generator | None = None) -> int:
        """Select greedy action breaking ties uniformly at random."""
        q_vals = self.q_table[state]
        max_val = np.max(q_vals)
        candidates = np.flatnonzero(q_vals == max_val)
        if len(candidates) == 1 or rng is None:
            return int(candidates[0])
        return int(rng.choice(candidates))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        terminated: bool,
        truncated: bool = False,
    ) -> float:
        """Perform Bellman temporal-difference update.

        Bootstrapping is cut off strictly on terminated, NOT on truncated.
        """
        # If terminal state, future value is zero; truncation still bootstraps
        future_val = 0.0 if terminated else float(np.max(self.q_table[next_state]))
        td_target = float(reward) + self.gamma * future_val
        td_error = td_target - float(self.q_table[state, action])
        self.q_table[state, action] += self.alpha * td_error
        return float(td_error)

    def save(self, path: str | Path) -> None:
        """Persist Q-table to disk as .npy file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, self.q_table)

    def load(self, path: str | Path) -> None:
        """Load Q-table from .npy file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Q-table file not found: {path}")
        self.q_table = np.load(p)
