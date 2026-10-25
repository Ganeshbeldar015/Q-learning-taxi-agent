"""Training loop, validation evaluation, and convergence detection for Q-learning."""

import csv
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from src.agent import QLearningAgent
from src.env_utils import get_seeded_rng, make_env
from src.schedules import get_schedule


def evaluate_greedy(
    agent: QLearningAgent,
    env: gym.Env,
    episodes: int = 50,
    seed_offset: int = 5000,
) -> tuple[float, float, float]:
    """Run pure greedy evaluation on fixed evaluation seeds without updating Q-table."""
    rewards: list[float] = []
    steps_list: list[int] = []
    successes = 0

    for ep in range(episodes):
        obs, _info = env.reset(seed=seed_offset + ep)
        ep_reward = 0.0
        ep_steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = agent.greedy_action(int(obs))
            obs, reward, terminated, truncated, _info = env.step(action)
            ep_reward += float(reward)
            ep_steps += 1

        rewards.append(ep_reward)
        steps_list.append(ep_steps)
        if terminated:
            successes += 1

    mean_reward = float(np.mean(rewards))
    mean_steps = float(np.mean(steps_list))
    success_rate = float(successes / max(1, episodes))
    return (mean_reward, mean_steps, success_rate)


def check_convergence(
    val_rewards: list[float],
    eval_episodes: list[int],
    threshold: float = 7.0,
    patience: int = 3,
) -> tuple[int | None, bool]:
    """Detect if greedy validation reward has sustained >= threshold for patience consecutive evaluations.

    Returns (earliest_episode_in_streak, True) if converged, else (None, False).
    """
    if len(val_rewards) < patience:
        return (None, False)

    for i in range(len(val_rewards) - patience + 1):
        window = val_rewards[i : i + patience]
        if all(r >= threshold for r in window):
            return (eval_episodes[i], True)

    return (None, False)


def train(
    config: dict[str, Any],
    env: gym.Env | None = None,
    log_csv_path: str | Path | None = "results/training_log.csv",
    q_table_save_path: str | Path | None = None,
) -> dict[str, Any]:
    """Execute training loop with periodic validation, convergence detection, and logging."""
    start_time = time.time()
    train_env = env or make_env(config.get("env", "Taxi-v4"))
    eval_env = make_env(config.get("env", "Taxi-v4"))

    seed = int(config.get("seed", 42))
    rng = get_seeded_rng(seed)

    train_cfg = config["train"]
    eval_cfg = config["eval"]

    episodes = int(train_cfg["episodes"])
    alpha = float(train_cfg["alpha"])
    gamma = float(train_cfg["gamma"])
    max_steps = int(train_cfg.get("max_steps", 200))

    eval_every = int(eval_cfg.get("every", 100))
    eval_episodes_count = int(eval_cfg.get("episodes", 50))
    val_seed_offset = int(eval_cfg.get("validation_seed_offset", 5000))
    conv_threshold = float(eval_cfg.get("convergence_threshold", 7.0))
    conv_patience = int(eval_cfg.get("convergence_patience", 3))

    agent = QLearningAgent(num_states=500, num_actions=6, alpha=alpha, gamma=gamma)
    schedule = get_schedule(config)

    logs: list[dict[str, Any]] = []
    val_rewards: list[float] = []
    val_episodes: list[int] = []
    converged_ep: int | None = None
    is_converged = False

    # Seed the training environment on the first reset only
    obs, _info = train_env.reset(seed=seed)

    try:
        for ep in range(episodes):
            if ep > 0:
                obs, _info = train_env.reset()

            eps = schedule.get_epsilon(ep)
            ep_reward = 0.0
            ep_steps = 0
            illegal_actions = 0
            terminated = False
            truncated = False

            while not (terminated or truncated) and ep_steps < max_steps:
                action = agent.select_action(int(obs), eps, rng)
                next_obs, reward, terminated, truncated, _info = train_env.step(action)

                agent.update(
                    state=int(obs),
                    action=action,
                    reward=float(reward),
                    next_state=int(next_obs),
                    terminated=terminated,
                    truncated=truncated,
                )

                obs = next_obs
                ep_reward += float(reward)
                ep_steps += 1
                if reward == -10:
                    illegal_actions += 1

            logs.append(
                {
                    "episode": ep + 1,
                    "reward": ep_reward,
                    "steps": ep_steps,
                    "epsilon": eps,
                    "illegal_actions": illegal_actions,
                    "success": int(terminated),
                }
            )

            # Periodic validation
            if (ep + 1) % eval_every == 0 or (ep + 1) == episodes:
                v_rew, _v_steps, _v_succ = evaluate_greedy(
                    agent,
                    eval_env,
                    episodes=eval_episodes_count,
                    seed_offset=val_seed_offset,
                )
                val_rewards.append(v_rew)
                val_episodes.append(ep + 1)

                if not is_converged:
                    c_ep, conv_flag = check_convergence(
                        val_rewards,
                        val_episodes,
                        threshold=conv_threshold,
                        patience=conv_patience,
                    )
                    if conv_flag:
                        converged_ep = c_ep
                        is_converged = True

            # Progress output every 500 episodes
            if (ep + 1) % 500 == 0:
                latest_val = val_rewards[-1] if val_rewards else 0.0
                print(
                    f"Episode {ep + 1}/{episodes} | Epsilon: {eps:.3f} | "
                    f"Train Rew (last 100): {np.mean([l['reward'] for l in logs[-100:]]):.2f} | "
                    f"Val Rew: {latest_val:.2f}"
                )

        wall_clock = time.time() - start_time

        # Save training logs to CSV if path specified
        if log_csv_path is not None:
            p_csv = Path(log_csv_path)
            p_csv.parent.mkdir(parents=True, exist_ok=True)
            with p_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "episode",
                        "reward",
                        "steps",
                        "epsilon",
                        "illegal_actions",
                        "success",
                    ],
                )
                writer.writeheader()
                writer.writerows(logs)

        # Save final Q-table if path specified
        if q_table_save_path is not None:
            agent.save(q_table_save_path)

        final_val = val_rewards[-1] if val_rewards else 0.0
        return {
            "episodes_trained": episodes,
            "final_reward": logs[-1]["reward"] if logs else 0.0,
            "final_validation_reward": final_val,
            "converged": is_converged,
            "convergence_episode": converged_ep,
            "wall_clock_time": wall_clock,
            "val_rewards": val_rewards,
            "val_episodes": val_episodes,
        }

    finally:
        eval_env.close()
        if env is None:
            train_env.close()
