"""Epsilon exploration decay schedules."""

from abc import ABC, abstractmethod
from typing import Any


class BaseSchedule(ABC):
    """Abstract base class for exploration schedules."""

    @abstractmethod
    def get_epsilon(self, episode: int) -> float:
        """Compute epsilon for a given episode index."""


class MultiplicativeSchedule(BaseSchedule):
    """Multiplicative geometric decay: epsilon = max(min_eps, start * (decay ** episode))."""

    def __init__(
        self,
        start: float = 1.0,
        min_epsilon: float = 0.01,
        decay: float = 0.999,
    ) -> None:
        self.start = float(start)
        self.min_epsilon = float(min_epsilon)
        self.decay = float(decay)

    def get_epsilon(self, episode: int) -> float:
        val = self.start * (self.decay ** int(episode))
        return float(max(self.min_epsilon, val))


class LinearSchedule(BaseSchedule):
    """Linear decay from start to min_epsilon over decay_episodes."""

    def __init__(
        self,
        start: float = 1.0,
        min_epsilon: float = 0.01,
        decay_episodes: int = 5000,
    ) -> None:
        self.start = float(start)
        self.min_epsilon = float(min_epsilon)
        self.decay_episodes = max(1, int(decay_episodes))

    def get_epsilon(self, episode: int) -> float:
        if episode >= self.decay_episodes:
            return float(self.min_epsilon)
        fraction = float(episode) / float(self.decay_episodes)
        return float(self.start + fraction * (self.min_epsilon - self.start))


def get_schedule(config: dict[str, Any]) -> BaseSchedule:
    """Factory creating configured schedule from training dictionary."""
    train_cfg = config.get("train", config)
    kind = train_cfg.get("schedule", "multiplicative").lower()
    start = float(train_cfg.get("epsilon_start", 1.0))
    min_eps = float(train_cfg.get("epsilon_min", 0.01))

    if kind == "multiplicative":
        decay = float(train_cfg.get("epsilon_decay", 0.999))
        return MultiplicativeSchedule(start=start, min_epsilon=min_eps, decay=decay)
    if kind == "linear":
        decay_eps = int(
            train_cfg.get("decay_episodes", train_cfg.get("episodes", 5000))
        )
        return LinearSchedule(
            start=start, min_epsilon=min_eps, decay_episodes=decay_eps
        )
    raise ValueError(f"Unknown schedule type: '{kind}'")
