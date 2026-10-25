"""Environment utilities, configuration parsing, state transformations, and baselines."""

import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml


def load_config(path: str | Path = "config/default.yaml") -> dict[str, Any]:
    """Load and validate project configuration YAML."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with p.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise TypeError("Configuration must be a dictionary")

    # Mandatory root keys
    for k in ("seed", "env", "train", "eval"):
        if k not in data:
            raise KeyError(f"Missing required configuration key: '{k}'")

    if not isinstance(data["seed"], int):
        raise TypeError("Configuration 'seed' must be an integer")

    # Required train keys
    train = data["train"]
    if not isinstance(train, dict):
        raise TypeError("Configuration 'train' must be a dictionary")
    for tk in (
        "episodes",
        "alpha",
        "gamma",
        "epsilon_start",
        "epsilon_min",
        "epsilon_decay",
        "schedule",
        "max_steps",
    ):
        if tk not in train:
            raise KeyError(f"Missing required train key: '{tk}'")

    # Required eval keys
    evaluation = data["eval"]
    if not isinstance(evaluation, dict):
        raise TypeError("Configuration 'eval' must be a dictionary")
    for ek in (
        "every",
        "episodes",
        "test_episodes",
        "convergence_threshold",
        "convergence_patience",
    ):
        if ek not in evaluation:
            raise KeyError(f"Missing required eval key: '{ek}'")

    return data


def get_seeded_rng(seed: int = 42) -> np.random.Generator:
    """Return a seeded NumPy Generator for reproducible exploration."""
    return np.random.default_rng(seed)


def make_env(env_name: str = "Taxi-v4", render_mode: str | None = None) -> gym.Env:
    """Instantiate Taxi environment with automatic fallback for Taxi-v3 / Taxi-v4 compatibility."""
    try:
        return gym.make(env_name, render_mode=render_mode)
    except (gym.error.DeprecatedEnv, gym.error.NamespaceNotFound):
        # Fallback to Taxi-v4 if Taxi-v3 was requested on modern gymnasium
        if "Taxi" in env_name:
            return gym.make("Taxi-v4", render_mode=render_mode)
        raise


def decode_state(state: int) -> tuple[int, int, int, int]:
    """Decompose discrete state integer into (taxi_row, taxi_col, passenger_location, destination).

    Grid: 5x5. Passenger locations: 0=R, 1=G, 2=Y, 3=B, 4=in taxi. Destinations: 0=R, 1=G, 2=Y, 3=B.
    """
    dest = state % 4
    state //= 4
    pass_loc = state % 5
    state //= 5
    taxi_col = state % 5
    taxi_row = state // 5
    return (taxi_row, taxi_col, pass_loc, dest)


def encode_state(taxi_row: int, taxi_col: int, pass_loc: int, dest: int) -> int:
    """Encode state tuple (taxi_row, taxi_col, passenger_location, destination) into integer in [0, 499]."""
    return ((taxi_row * 5 + taxi_col) * 5 + pass_loc) * 4 + dest


def verify_state_roundtrip(env: gym.Env | None = None) -> bool:
    """Verify bidirectional state round-trip across all 500 Taxi states."""
    test_env = env or make_env()
    try:
        for s in range(500):
            row, col, pass_loc, dest = decode_state(s)
            s_rec = encode_state(row, col, pass_loc, dest)
            if s != s_rec:
                return False
            # Verify against gymnasium unwrapped decode if accessible
            if hasattr(test_env.unwrapped, "decode"):
                unwrapped_decoded = tuple(test_env.unwrapped.decode(s))
                if (row, col, pass_loc, dest) != unwrapped_decoded:
                    return False
        return True
    finally:
        if env is None:
            test_env.close()


def run_random_baseline(
    env: gym.Env,
    episodes: int = 1000,
    seed: int = 42,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate a purely uniform random policy over N episodes.

    Saves summary metrics to output_path if provided.
    """
    rng = get_seeded_rng(seed)
    rewards: list[float] = []
    steps_list: list[int] = []
    successes = 0
    truncations = 0
    illegal_actions = 0

    # Seed the environment on the first reset only
    _obs, _info = env.reset(seed=seed)

    for ep in range(episodes):
        if ep > 0:
            _obs, _info = env.reset()

        ep_reward = 0.0
        ep_steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = int(rng.integers(0, 6))
            _obs, reward, terminated, truncated, _info = env.step(action)
            ep_reward += float(reward)
            ep_steps += 1
            if reward == -10:
                illegal_actions += 1

        rewards.append(ep_reward)
        steps_list.append(ep_steps)
        if terminated:
            successes += 1
        if truncated:
            truncations += 1

    metrics: dict[str, Any] = {
        "policy": "random",
        "episodes": episodes,
        "seed": seed,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_steps": float(np.mean(steps_list)),
        "median_steps": float(np.median(steps_list)),
        "success_rate": float(successes / episodes),
        "truncation_rate": float(truncations / episodes),
        "illegal_actions": illegal_actions,
    }

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics
