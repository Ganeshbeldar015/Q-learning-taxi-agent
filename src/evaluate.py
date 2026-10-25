"""Evaluation pipeline and comparative policy benchmarking."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.agent import QLearningAgent
from src.env_utils import make_env


def compute_bootstrap_ci(
    data: Sequence[float],
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute non-parametric bootstrap confidence interval for the sample mean."""
    arr = np.asarray(data, dtype=np.float64)
    if len(arr) == 0:
        return (0.0, 0.0)
    if len(arr) == 1:
        return (float(arr[0]), float(arr[0]))

    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(n_bootstraps, dtype=np.float64)
    n = len(arr)
    for i in range(n_bootstraps):
        resample = rng.choice(arr, size=n, replace=True)
        bootstrap_means[i] = np.mean(resample)

    alpha = 1.0 - confidence_level
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    ci_lower = float(np.percentile(bootstrap_means, lower_pct))
    ci_upper = float(np.percentile(bootstrap_means, upper_pct))
    return ci_lower, ci_upper


def evaluate_policy(
    policy_source: Any = None,
    episodes: int = 1000,
    seed_offset: int = 10000,
    max_steps: int = 200,
    env_name: str = "Taxi-v4",
) -> dict[str, Any]:
    """Evaluate a greedy policy (or untrained/random) across deterministic test seeds.

    Args:
        policy_source: QLearningAgent instance, np.ndarray Q-table, path to .npy file, or None (untrained zeros).
        episodes: Number of evaluation episodes.
        seed_offset: Starting seed for test episodes (default: 10000).
        max_steps: Maximum steps per episode before truncation.
        env_name: Gymnasium environment name.

    Returns:
        Dictionary containing summary statistics and raw trajectories.
    """
    if isinstance(policy_source, (str, Path)):
        agent = QLearningAgent()
        agent.load(policy_source)
    elif isinstance(policy_source, np.ndarray):
        agent = QLearningAgent()
        agent.q_table = policy_source.copy()
    elif isinstance(policy_source, QLearningAgent):
        agent = policy_source
    else:
        # Untrained agent with all zeros
        agent = QLearningAgent()

    env = make_env(env_name)
    raw_rewards: list[float] = []
    raw_steps: list[int] = []
    successes: int = 0
    total_illegal_actions: int = 0

    for ep in range(episodes):
        seed = seed_offset + ep
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(seed)

        ep_reward = 0.0
        ep_steps = 0
        ep_illegal = 0
        done = False

        while not done and ep_steps < max_steps:
            action = agent.greedy_action(int(obs), rng)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += float(reward)
            ep_steps += 1

            if reward == -10:
                ep_illegal += 1

            obs = next_obs
            done = bool(terminated or truncated)
            if terminated:
                successes += 1

        raw_rewards.append(ep_reward)
        raw_steps.append(ep_steps)
        total_illegal_actions += ep_illegal

    env.close()

    arr_rewards = np.array(raw_rewards, dtype=np.float64)
    arr_steps = np.array(raw_steps, dtype=np.float64)
    ci_low, ci_high = compute_bootstrap_ci(raw_rewards, seed=seed_offset)

    return {
        "episodes": episodes,
        "mean_reward": float(np.mean(arr_rewards)),
        "std_reward": float(np.std(arr_rewards)),
        "ci_95": [ci_low, ci_high],
        "mean_steps": float(np.mean(arr_steps)),
        "std_steps": float(np.std(arr_steps)),
        "success_rate": float(successes / max(1, episodes)),
        "total_illegal_actions": total_illegal_actions,
        "mean_illegal_actions": float(total_illegal_actions / max(1, episodes)),
        "raw_rewards": raw_rewards,
        "raw_steps": raw_steps,
    }


def generate_comparison_table(
    results_dir: str = "results",
    out_csv_path: str = "results/comparison_table.csv",
    test_episodes: int = 1000,
    seed_offset: int = 10000,
) -> pd.DataFrame:
    """Benchmark 4 policies into a comparative DataFrame and CSV."""
    p_results = Path(results_dir)

    rows: list[dict[str, Any]] = []

    # 1. Random Policy from baseline_random.json
    random_json_p = p_results / "baseline_random.json"
    if random_json_p.exists():
        with open(random_json_p, encoding="utf-8") as f:
            rand_data = json.load(f)
        rows.append(
            {
                "Policy": "Random Policy",
                "Episodes": rand_data.get("episodes", 1000),
                "Mean Reward": f"{rand_data['mean_reward']:.2f}",
                "Std Reward": f"{rand_data['std_reward']:.2f}",
                "95% CI": "N/A",
                "Mean Steps": f"{rand_data['mean_steps']:.1f}",
                "Success Rate": f"{rand_data['success_rate'] * 100:.1f}%",
                "Illegal Actions": rand_data.get(
                    "illegal_actions", rand_data.get("total_illegal_actions", 0)
                ),
            }
        )

    # 2. Untrained Q-Agent (all zeros)
    untrained_metrics = evaluate_policy(
        None, episodes=test_episodes, seed_offset=seed_offset
    )
    rows.append(
        {
            "Policy": "Untrained Q-Agent",
            "Episodes": test_episodes,
            "Mean Reward": f"{untrained_metrics['mean_reward']:.2f}",
            "Std Reward": f"{untrained_metrics['std_reward']:.2f}",
            "95% CI": f"[{untrained_metrics['ci_95'][0]:.2f}, {untrained_metrics['ci_95'][1]:.2f}]",
            "Mean Steps": f"{untrained_metrics['mean_steps']:.1f}",
            "Success Rate": f"{untrained_metrics['success_rate'] * 100:.1f}%",
            "Illegal Actions": untrained_metrics["total_illegal_actions"],
        }
    )

    # 3. Baseline Trained Model
    baseline_npy = p_results / "q_table_baseline.npy"
    if baseline_npy.exists():
        base_metrics = evaluate_policy(
            baseline_npy, episodes=test_episodes, seed_offset=seed_offset
        )
        rows.append(
            {
                "Policy": "Baseline Trained Model",
                "Episodes": test_episodes,
                "Mean Reward": f"{base_metrics['mean_reward']:.2f}",
                "Std Reward": f"{base_metrics['std_reward']:.2f}",
                "95% CI": f"[{base_metrics['ci_95'][0]:.2f}, {base_metrics['ci_95'][1]:.2f}]",
                "Mean Steps": f"{base_metrics['mean_steps']:.1f}",
                "Success Rate": f"{base_metrics['success_rate'] * 100:.1f}%",
                "Illegal Actions": base_metrics["total_illegal_actions"],
            }
        )

    # 4. Tuned Final Model
    final_npy = p_results / "q_table_final.npy"
    if final_npy.exists():
        final_metrics = evaluate_policy(
            final_npy, episodes=test_episodes, seed_offset=seed_offset
        )
        rows.append(
            {
                "Policy": "Tuned Final Model",
                "Episodes": test_episodes,
                "Mean Reward": f"{final_metrics['mean_reward']:.2f}",
                "Std Reward": f"{final_metrics['std_reward']:.2f}",
                "95% CI": f"[{final_metrics['ci_95'][0]:.2f}, {final_metrics['ci_95'][1]:.2f}]",
                "Mean Steps": f"{final_metrics['mean_steps']:.1f}",
                "Success Rate": f"{final_metrics['success_rate'] * 100:.1f}%",
                "Illegal Actions": final_metrics["total_illegal_actions"],
            }
        )

    df = pd.DataFrame(rows)
    p_out = Path(out_csv_path)
    p_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p_out, index=False)
    return df


def run_evaluation_pipeline(
    q_table_path: str = "results/q_table_final.npy",
    test_metrics_path: str = "results/test_metrics.json",
    comparison_csv_path: str = "results/comparison_table.csv",
    episodes: int = 1000,
    seed_offset: int = 10000,
) -> dict[str, Any]:
    """Execute complete 1,000-episode test evaluation and comparison table generation."""
    print(
        f"Evaluating final policy from {q_table_path} on {episodes} unseen test episodes (seed {seed_offset}+)..."
    )
    metrics = evaluate_policy(q_table_path, episodes=episodes, seed_offset=seed_offset)

    p_test_json = Path(test_metrics_path)
    p_test_json.parent.mkdir(parents=True, exist_ok=True)
    with open(p_test_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved test metrics to {test_metrics_path}")
    print(
        f"Mean Reward: {metrics['mean_reward']:.2f} (95% CI: [{metrics['ci_95'][0]:.2f}, {metrics['ci_95'][1]:.2f}]) | "
        f"Mean Steps: {metrics['mean_steps']:.2f} | "
        f"Success Rate: {metrics['success_rate'] * 100:.1f}% | "
        f"Illegal Actions: {metrics['total_illegal_actions']}"
    )

    print("Generating comparative policy benchmarks table...")
    df = generate_comparison_table(
        results_dir=str(p_test_json.parent),
        out_csv_path=comparison_csv_path,
        test_episodes=episodes,
        seed_offset=seed_offset,
    )
    print(
        f"Saved comparison table to {comparison_csv_path}:\n{df.to_string(index=False)}"
    )

    return metrics


if __name__ == "__main__":
    run_evaluation_pipeline()
