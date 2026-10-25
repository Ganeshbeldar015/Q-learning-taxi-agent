"""Hyperparameter tuning and multi-seed final training for Q-learning Taxi."""

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.env_utils import load_config
from src.train import train


def run_trial(
    params: dict[str, Any],
    episodes: int = 3000,
    seed: int = 42,
    base_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute a single hyperparameter trial and return summary metrics."""
    cfg = (
        copy.deepcopy(base_config)
        if base_config
        else load_config("config/default.yaml")
    )

    alpha = float(params.get("alpha", cfg["train"]["alpha"]))
    gamma = float(params.get("gamma", cfg["train"]["gamma"]))
    decay = float(params.get("epsilon_decay", cfg["train"]["epsilon_decay"]))
    min_eps = float(params.get("epsilon_min", cfg["train"]["epsilon_min"]))

    cfg["seed"] = int(seed)
    cfg["train"]["episodes"] = int(episodes)
    cfg["train"]["alpha"] = alpha
    cfg["train"]["gamma"] = gamma
    cfg["train"]["epsilon_decay"] = decay
    cfg["train"]["epsilon_min"] = min_eps

    res = train(cfg, log_csv_path=None, q_table_save_path=None)

    return {
        "alpha": alpha,
        "gamma": gamma,
        "epsilon_decay": decay,
        "epsilon_min": min_eps,
        "seed": int(seed),
        "converged": bool(res["converged"]),
        "convergence_episode": res["convergence_episode"],
        "final_val_reward": float(res["final_validation_reward"]),
        "wall_clock_time": float(res["wall_clock_time"]),
    }


def select_best_configuration(
    trials: list[dict[str, Any]],
    threshold: float = 7.0,
    min_conv_rate: float = 0.8,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Group trials by parameter tuple and rank according to convergence speed and stability."""
    df = pd.DataFrame(trials)

    group_cols = ["alpha", "gamma", "epsilon_decay", "epsilon_min"]
    records: list[dict[str, Any]] = []

    for key, group in df.groupby(group_cols):
        alpha, gamma, decay, min_eps = key
        total = len(group)
        converged_rows = group[group["converged"]]
        conv_rate = len(converged_rows) / max(1, total)

        if len(converged_rows) > 0:
            mean_conv_ep = float(converged_rows["convergence_episode"].mean())
        else:
            mean_conv_ep = 999999.0

        mean_val_rew = float(group["final_val_reward"].mean())
        std_val_rew = float(group["final_val_reward"].std(ddof=0))

        records.append(
            {
                "alpha": alpha,
                "gamma": gamma,
                "epsilon_decay": decay,
                "epsilon_min": min_eps,
                "trials": total,
                "convergence_rate": conv_rate,
                "mean_convergence_episode": mean_conv_ep,
                "mean_validation_reward": mean_val_rew,
                "std_validation_reward": std_val_rew,
            }
        )

    summary_df = pd.DataFrame(records)

    # Filter by minimum convergence rate and reward threshold
    qualified = summary_df[
        (summary_df["convergence_rate"] >= min_conv_rate)
        & (summary_df["mean_validation_reward"] >= threshold)
    ]

    if qualified.empty:
        # Fallback to all trials with highest convergence rate and reward
        candidate_pool = summary_df
    else:
        candidate_pool = qualified

    # Sort: 1. lowest convergence episode, 2. highest reward, 3. lowest std
    sorted_df = candidate_pool.sort_values(
        by=[
            "mean_convergence_episode",
            "mean_validation_reward",
            "std_validation_reward",
        ],
        ascending=[True, False, True],
    )

    best_record = sorted_df.iloc[0].to_dict()
    return best_record, summary_df


def save_best_config(
    best_params: dict[str, Any],
    base_cfg_path: str | Path = "config/default.yaml",
    out_path: str | Path = "config/best.yaml",
) -> None:
    """Write winning parameters into config/best.yaml preserving structure."""
    base_p = Path(base_cfg_path)
    cfg = yaml.safe_load(base_p.read_text(encoding="utf-8"))

    cfg["train"]["alpha"] = float(best_params["alpha"])
    cfg["train"]["gamma"] = float(best_params["gamma"])
    cfg["train"]["epsilon_decay"] = float(best_params["epsilon_decay"])
    cfg["train"]["epsilon_min"] = float(best_params["epsilon_min"])

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(yaml.dump(cfg, sort_keys=False), encoding="utf-8")


def _seed_retrain_worker(args: tuple[dict[str, Any], int, str]) -> dict[str, Any]:
    """Helper worker for final multi-seed retraining."""
    cfg, seed, out_dir_str = args
    out_d = Path(out_dir_str)
    seed_cfg = copy.deepcopy(cfg)
    seed_cfg["seed"] = seed

    log_path = out_d / f"training_log_seed_{seed}.csv"
    q_path = out_d / f"q_table_seed_{seed}.npy"

    res = train(seed_cfg, log_csv_path=log_path, q_table_save_path=q_path)
    return {
        "seed": seed,
        "final_validation_reward": float(res["final_validation_reward"]),
        "converged": res["converged"],
        "convergence_episode": res["convergence_episode"],
        "wall_clock_time": res["wall_clock_time"],
        "q_path": str(q_path),
    }


def retrain_final_models(
    config: dict[str, Any] | str | Path,
    seeds: list[int] | None = None,
    out_dir: str | Path = "results/final_runs",
    primary_out: str | Path = "results/q_table_final.npy",
    max_workers: int = 5,
) -> dict[str, Any]:
    """Retrain winning configuration across 5 independent seeds and export best model."""
    from concurrent.futures import ProcessPoolExecutor

    if isinstance(config, (str, Path)):
        cfg = load_config(config)
    else:
        cfg = copy.deepcopy(config)

    seeds = seeds or [101, 102, 103, 104, 105]
    out_d = Path(out_dir)
    out_d.mkdir(parents=True, exist_ok=True)

    tasks = [(cfg, seed, str(out_d)) for seed in seeds]
    with ProcessPoolExecutor(max_workers=min(len(seeds), max_workers)) as executor:
        results_per_seed = list(executor.map(_seed_retrain_worker, tasks))

    # Identify primary model with maximal validation reward
    sorted_seeds = sorted(
        results_per_seed,
        key=lambda r: (
            r["final_validation_reward"],
            -(r["convergence_episode"] or 999999),
        ),
        reverse=True,
    )
    best_record = sorted_seeds[0]
    best_seed = best_record["seed"]
    best_val_rew = best_record["final_validation_reward"]
    best_q_table_path = Path(best_record["q_path"])

    # Persist primary Q-table
    if best_q_table_path.exists():
        q_table = np.load(best_q_table_path)
        p_out = Path(primary_out)
        p_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(p_out, q_table)

    return {
        "best_seed": best_seed,
        "best_validation_reward": best_val_rew,
        "seed_results": results_per_seed,
    }


def _trial_worker(task_tuple: tuple[dict[str, Any], int, int]) -> dict[str, Any]:
    """Helper worker for multiprocessing execution."""
    params, episodes, seed = task_tuple
    return run_trial(params, episodes=episodes, seed=seed)


def run_full_tuning_pipeline(
    base_cfg_path: str = "config/default.yaml",
    tuning_csv_path: str = "results/tuning_results.csv",
    best_yaml_path: str = "config/best.yaml",
    max_workers: int = 8,
) -> dict[str, Any]:
    """Execute complete Stage A single-factor sweeps and Stage B grid search."""
    from concurrent.futures import ProcessPoolExecutor

    seeds = [42, 43, 44]
    tasks: list[tuple[dict[str, Any], int, int]] = []

    # Stage A: Single-factor sweeps
    # Factor 1: Alpha sweep
    for alpha in [0.05, 0.1, 0.3, 0.5, 0.7]:
        for s in seeds:
            tasks.append(
                (
                    {
                        "alpha": alpha,
                        "gamma": 0.99,
                        "epsilon_decay": 0.999,
                        "epsilon_min": 0.01,
                    },
                    3000,
                    s,
                )
            )

    # Factor 2: Gamma sweep
    for gamma in [0.8, 0.9, 0.95, 0.99]:
        for s in seeds:
            tasks.append(
                (
                    {
                        "alpha": 0.1,
                        "gamma": gamma,
                        "epsilon_decay": 0.999,
                        "epsilon_min": 0.01,
                    },
                    3000,
                    s,
                )
            )

    # Factor 3: Epsilon Decay sweep
    for decay in [0.99, 0.995, 0.999, 0.9995]:
        for s in seeds:
            tasks.append(
                (
                    {
                        "alpha": 0.1,
                        "gamma": 0.99,
                        "epsilon_decay": decay,
                        "epsilon_min": 0.01,
                    },
                    3000,
                    s,
                )
            )

    # Stage B: Focused grid of top candidates
    # In Taxi-v4: higher alpha (0.3, 0.5) converges much faster; gamma in [0.95, 0.99]; decay in [0.995, 0.999]
    for alpha in [0.3, 0.5]:
        for gamma in [0.95, 0.99]:
            for decay in [0.995, 0.999]:
                for s in seeds:
                    tasks.append(
                        (
                            {
                                "alpha": alpha,
                                "gamma": gamma,
                                "epsilon_decay": decay,
                                "epsilon_min": 0.01,
                            },
                            3000,
                            s,
                        )
                    )

    # Execute all trials in parallel
    print(
        f"Starting tuning search: {len(tasks)} trials across {max_workers} parallel workers..."
    )
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        all_trials = list(executor.map(_trial_worker, tasks))

    # Save all trial results to CSV
    csv_p = Path(tuning_csv_path)
    csv_p.parent.mkdir(parents=True, exist_ok=True)
    df_trials = pd.DataFrame(all_trials)
    df_trials.to_csv(csv_p, index=False)
    print(f"Saved {len(all_trials)} trial results to {tuning_csv_path}")

    # Select best configuration
    best_config, _summary_df = select_best_configuration(all_trials)
    print(
        f"Selected winning config: alpha={best_config['alpha']}, gamma={best_config['gamma']}, "
        f"decay={best_config['epsilon_decay']}, min_eps={best_config['epsilon_min']} | "
        f"Mean Conv Ep: {best_config['mean_convergence_episode']} | "
        f"Mean Val Rew: {best_config['mean_validation_reward']:.2f}"
    )

    # Save to config/best.yaml
    save_best_config(best_config, base_cfg_path=base_cfg_path, out_path=best_yaml_path)
    print(f"Saved winning configuration to {best_yaml_path}")

    return best_config
