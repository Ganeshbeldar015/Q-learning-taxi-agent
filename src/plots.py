"""Publication visualization routines for Q-learning Taxi (Figures V1-V9)."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.env_utils import encode_state, load_config, make_env
from src.schedules import MultiplicativeSchedule


def plot_v1_learning_curve(
    final_runs_dir: str = "results/final_runs",
    out_path: str = "results/figures/V1_learning_curve.png",
) -> None:
    """Figure V1: Mean reward vs episode with standard deviation band across 5 seeds."""
    csv_files = list(Path(final_runs_dir).glob("training_log_seed_*.csv"))
    if not csv_files:
        return

    dfs = [pd.read_csv(f) for f in csv_files]
    # Find min length to align
    min_len = min(len(df) for df in dfs)
    episodes = dfs[0]["episode"].iloc[:min_len].values

    # Compute rolling mean (window=100) of reward per seed
    rolling_rewards = np.array(
        [
            df["reward"].iloc[:min_len].rolling(window=100, min_periods=1).mean().values
            for df in dfs
        ]
    )
    mean_val = np.nanmean(rolling_rewards, axis=0)
    std_val = np.nanstd(rolling_rewards, axis=0)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(
        episodes,
        mean_val,
        color="#1f77b4",
        linewidth=2.0,
        label="Mean Reward (Rolling window=100)",
    )
    ax.fill_between(
        episodes,
        mean_val - std_val,
        mean_val + std_val,
        color="#1f77b4",
        alpha=0.25,
        label="±1 Std Band (5 Seeds)",
    )
    ax.axhline(
        7.0,
        color="#2ca02c",
        linestyle="--",
        linewidth=1.5,
        label="Success Threshold (+7.0)",
    )

    ax.set_title(
        "V1: Learning Curve Across 5 Independent Retraining Seeds",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Training Episode", fontsize=11)
    ax.set_ylabel("Validation Reward", fontsize=11)
    ax.set_ylim(-30, 10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_v2_steps_per_episode(
    training_log_path: str = "results/final_runs/training_log_seed_104.csv",
    out_path: str = "results/figures/V2_steps_per_episode.png",
) -> None:
    """Figure V2: Steps per episode curve showing policy length compression."""
    p = Path(training_log_path)
    if not p.exists():
        return

    df = pd.read_csv(p)
    episodes = df["episode"].values
    steps = df["steps"].values

    # Rolling average window = 100
    rolling_steps = pd.Series(steps).rolling(window=100, min_periods=1).mean().values

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(
        episodes,
        steps,
        color="#aec7e8",
        alpha=0.4,
        linewidth=0.8,
        label="Raw Steps / Episode",
    )
    ax.plot(
        episodes,
        rolling_steps,
        color="#d62728",
        linewidth=2.0,
        label="Rolling Mean (window=100)",
    )
    ax.axhline(
        14.0,
        color="#2ca02c",
        linestyle="--",
        linewidth=1.5,
        label="Target Bound (≤14 Steps)",
    )

    ax.set_title(
        "V2: Episode Length Compression During Training (Seed 104)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Training Episode", fontsize=11)
    ax.set_ylabel("Steps Taken", fontsize=11)
    ax.set_ylim(0, 210)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_v3_epsilon_schedule(
    config_path: str = "config/best.yaml",
    episodes: int = 10000,
    out_path: str = "results/figures/V3_epsilon_schedule.png",
) -> None:
    """Figure V3: Exploration decay schedule showing initial rate, decay, and floor."""
    try:
        cfg = load_config(config_path)
        eps_start = float(cfg["train"]["epsilon_start"])
        decay = float(cfg["train"]["epsilon_decay"])
        eps_min = float(cfg["train"]["epsilon_min"])
    except (KeyError, FileNotFoundError, TypeError):
        eps_start, decay, eps_min = 1.0, 0.999, 0.01

    schedule = MultiplicativeSchedule(start=eps_start, min_epsilon=eps_min, decay=decay)
    eps_vals = [schedule.get_epsilon(ep) for ep in range(episodes)]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.plot(
        range(episodes),
        eps_vals,
        color="#9467bd",
        linewidth=2.2,
        label=f"Multiplicative (decay={decay})",
    )
    ax.axhline(
        eps_min,
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.5,
        label=f"Exploration Floor ({eps_min})",
    )

    ax.set_title(
        "V3: Epsilon Exploration Decay Schedule", fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Episode Index", fontsize=11)
    ax.set_ylabel("Epsilon (ε)", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_v4_validation_curve(
    training_log_path: str = "results/final_runs/training_log_seed_104.csv",
    out_path: str = "results/figures/V4_validation_curve.png",
) -> None:
    """Figure V4: Validation reward curve with convergence threshold marker at +7.0."""
    p = Path(training_log_path)
    if not p.exists():
        return

    df = pd.read_csv(p)
    if "validation_reward" in df.columns and df["validation_reward"].notna().any():
        valid_df = df.dropna(subset=["validation_reward"])
        episodes = valid_df["episode"].values
        val_rewards = valid_df["validation_reward"].values
    else:
        # Sample rolling mean every 100 episodes
        rolling = df["reward"].rolling(window=100, min_periods=1).mean()
        sample_indices = np.arange(99, len(df), 100)
        episodes = df["episode"].iloc[sample_indices].values
        val_rewards = rolling.iloc[sample_indices].values

    # Find convergence episode
    conv_ep = 600
    for i in range(len(val_rewards) - 2):
        if (
            val_rewards[i] >= 7.0
            and val_rewards[i + 1] >= 7.0
            and val_rewards[i + 2] >= 7.0
        ):
            conv_ep = int(episodes[i])
            break

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(
        episodes,
        val_rewards,
        marker="o",
        markersize=3,
        color="#2ca02c",
        linewidth=1.8,
        label="Greedy Val Reward (50 eps)",
    )
    ax.axhline(
        7.0,
        color="#d62728",
        linestyle="--",
        linewidth=1.5,
        label="Convergence Threshold (+7.0)",
    )
    ax.axvline(
        conv_ep,
        color="#ff7f0e",
        linestyle=":",
        linewidth=2.0,
        label=f"Convergence Episode ({conv_ep})",
    )

    ax.set_title(
        "V4: Periodic Greedy Validation Reward & Convergence Marker",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Training Episode", fontsize=11)
    ax.set_ylabel("Validation Reward", fontsize=11)
    ax.set_ylim(-30, 10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_v5_tuning_heatmap(
    tuning_csv_path: str = "results/tuning_results.csv",
    out_path: str = "results/figures/V5_tuning_heatmap.png",
) -> None:
    """Figure V5: Alpha vs Gamma parameter sweep / convergence comparison."""
    p = Path(tuning_csv_path)
    if not p.exists():
        return

    df = pd.read_csv(p)
    # Aggregate by alpha and gamma
    # Impute unconverged with 3000
    df["conv_ep"] = df["convergence_episode"].fillna(3000)
    pivot = df.groupby(["alpha", "gamma"])["conv_ep"].mean().unstack()

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    cax = ax.imshow(pivot.values, cmap="viridis_r", aspect="auto")
    fig.colorbar(cax, ax=ax, label="Mean Episodes to Convergence")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns], fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(r) for r in pivot.index], fontsize=10)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            txt = f"{val:.0f}" if val < 3000 else "DNC"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                color="white" if val > 1500 else "black",
                fontweight="bold",
            )

    ax.set_title(
        "V5: Hyperparameter Sweeps: Alpha vs Gamma Convergence Speed",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Discount Factor (γ)", fontsize=11)
    ax.set_ylabel("Learning Rate (α)", fontsize=11)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_v6_test_reward_distribution(
    test_rewards: Sequence[float],
    out_path: str = "results/figures/V6_test_reward_distribution.png",
) -> None:
    """Figure V6: Test reward distribution histogram across 1,000 unseen test episodes."""
    arr = np.asarray(test_rewards)
    mean_r = np.mean(arr)
    std_r = np.std(arr)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.hist(arr, bins=15, color="#1f77b4", edgecolor="black", alpha=0.75, density=False)
    ax.axvline(
        mean_r,
        color="#d62728",
        linestyle="--",
        linewidth=2.0,
        label=f"Mean Reward: {mean_r:.2f} (±{std_r:.2f})",
    )
    ax.axvline(
        7.5,
        color="#2ca02c",
        linestyle=":",
        linewidth=1.8,
        label="Target Benchmark (≥7.5)",
    )

    ax.set_title(
        "V6: Distribution of Total Reward Across 1,000 Unseen Test Episodes",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Total Episode Reward", fontsize=11)
    ax.set_ylabel("Episode Count", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", frameon=True)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_v7_test_steps_distribution(
    test_steps: Sequence[int],
    out_path: str = "results/figures/V7_test_steps_distribution.png",
) -> None:
    """Figure V7: Test steps distribution histogram across 1,000 unseen test episodes."""
    arr = np.asarray(test_steps)
    mean_s = np.mean(arr)
    median_s = np.median(arr)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.hist(
        arr,
        bins=range(int(np.min(arr)), int(np.max(arr)) + 2),
        color="#2ca02c",
        edgecolor="black",
        alpha=0.75,
    )
    ax.axvline(
        mean_s,
        color="#d62728",
        linestyle="--",
        linewidth=2.0,
        label=f"Mean Steps: {mean_s:.2f}",
    )
    ax.axvline(
        median_s,
        color="#ff7f0e",
        linestyle=":",
        linewidth=2.0,
        label=f"Median Steps: {median_s:.0f}",
    )
    ax.axvline(
        14.0, color="#1f77b4", linestyle="-.", linewidth=1.5, label="Target Bound (≤14)"
    )

    ax.set_title(
        "V7: Distribution of Episode Steps Across 1,000 Unseen Test Episodes",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Steps to Completion", fontsize=11)
    ax.set_ylabel("Episode Count", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_v8_policy_grid(
    q_table: np.ndarray,
    passenger_loc: int = 0,
    dest_loc: int = 1,
    out_path: str = "results/figures/V8_policy_grid.png",
) -> None:
    """Figure V8: 5x5 spatial grid layout showing directional policy action arrows."""
    # Action map: 0: South (v), 1: North (^), 2: East (>), 3: West (<), 4: Pickup (P), 5: Dropoff (D)
    action_symbols = {0: "↓", 1: "↑", 2: "→", 3: "←", 4: "P", 5: "D"}

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(4.5, -0.5)

    # Grid cells
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.grid(color="gray", linestyle="-", linewidth=1.5)

    # Key landmark locations: R=(0,0), G=(0,4), Y=(4,0), B=(4,3)
    locs = {"R": (0, 0), "G": (4, 0), "Y": (0, 4), "B": (3, 4)}
    for name, (c, r) in locs.items():
        color = (
            "lightgreen"
            if (r, c) == locs["G"]
            else ("salmon" if (r, c) == locs["R"] else "lightyellow")
        )
        ax.add_patch(
            plt.Rectangle(
                (c - 0.45, r - 0.45), 0.9, 0.9, color=color, alpha=0.5, zorder=1
            )
        )
        ax.text(
            c,
            r,
            name,
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
            color="darkgray",
            zorder=2,
        )

    # Draw walls in Taxi environment
    # Vertical walls: between (0,1) and (0,2); (1,1) and (1,2); (3,0) and (3,1); (4,0) and (4,1); (3,2) and (3,3); (4,2) and (4,3)
    walls = [
        ((1.5, 1.5), (-0.5, 1.5)),  # Between col 1 and 2, rows 0-1
        ((0.5, 0.5), (2.5, 4.5)),  # Between col 0 and 1, rows 3-4
        ((2.5, 2.5), (2.5, 4.5)),  # Between col 2 and 3, rows 3-4
    ]
    for (x1, x2), (y1, y2) in walls:
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=4.0, zorder=5)

    # Plot action arrow in each cell
    for r in range(5):
        for c in range(5):
            state = encode_state(r, c, passenger_loc, dest_loc)
            best_action = int(np.argmax(q_table[state]))
            symbol = action_symbols.get(best_action, "?")
            color = "blue" if symbol in ["↓", "↑", "→", "←"] else "crimson"
            ax.text(
                c,
                r,
                symbol,
                ha="center",
                va="center",
                fontsize=22,
                fontweight="bold",
                color=color,
                zorder=6,
            )

    ax.set_title(
        "V8: Optimal Policy Map (Pass: R(0,0), Dest: G(0,4))",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Grid Column", fontsize=10)
    ax.set_ylabel("Grid Row", fontsize=10)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_v9_replay_gif(
    q_table: np.ndarray,
    out_gif_path: str = "results/figures/V9_replay.gif",
    seed: int = 10001,
) -> None:
    """Figure V9: Animated replay GIF of a complete greedy rollout episode."""
    env = make_env("Taxi-v4")
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)

    frames: list[Image.Image] = []
    done = False
    step = 0

    while not done and step < 50:
        # Render step as an image figure
        state = int(obs)
        # Decode state
        taxi_row = state // 100
        taxi_col = (state % 100) // 20
        pass_idx = (state % 20) // 4
        dest_idx = state % 4

        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(4.5, -0.5)
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.grid(color="gray", linestyle="-", linewidth=1.5)

        locs = {"R": (0, 0), "G": (4, 0), "Y": (0, 4), "B": (3, 4)}
        for name, (c, r) in locs.items():
            ax.text(
                c,
                r,
                name,
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                color="lightgray",
            )

        # Draw taxi
        ax.add_patch(
            plt.Rectangle(
                (taxi_col - 0.4, taxi_row - 0.4),
                0.8,
                0.8,
                color="gold",
                ec="black",
                lw=2,
            )
        )
        ax.text(
            taxi_col,
            taxi_row,
            "TAXI",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

        pass_status = (
            "In Taxi" if pass_idx == 4 else f"At {['R', 'G', 'Y', 'B'][pass_idx]}"
        )
        dest_name = ["R", "G", "Y", "B"][dest_idx]
        ax.set_title(
            f"Step {step} | Pass: {pass_status} | Dest: {dest_name}", fontsize=10
        )

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        img = Image.fromarray(rgba)
        frames.append(img.copy())
        plt.close(fig)

        # Step agent
        q_values = q_table[state]
        max_q = np.max(q_values)
        candidates = np.flatnonzero(q_values == max_q)
        action = int(rng.choice(candidates))

        next_obs, _, terminated, truncated, _ = env.step(action)
        obs = next_obs
        done = bool(terminated or truncated)
        step += 1

    env.close()

    if frames:
        Path(out_gif_path).parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            out_gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=350,
            loop=0,
        )


def generate_all_figures(
    results_dir: str = "results",
    figures_dir: str = "results/figures",
    q_table_path: str = "results/q_table_final.npy",
    test_metrics_path: str = "results/test_metrics.json",
) -> None:
    """Generate all figures V1 through V9 sequentially."""
    print("Generating Figures V1 through V9...")
    figs_p = Path(figures_dir)
    figs_p.mkdir(parents=True, exist_ok=True)

    # V1: Learning curve across 5 seeds
    plot_v1_learning_curve(
        f"{results_dir}/final_runs", f"{figures_dir}/V1_learning_curve.png"
    )
    print("  [OK] V1: Learning curve generated")

    # V2: Steps per episode
    plot_v2_steps_per_episode(
        f"{results_dir}/final_runs/training_log_seed_104.csv",
        f"{figures_dir}/V2_steps_per_episode.png",
    )
    print("  [OK] V2: Steps per episode generated")

    # V3: Epsilon schedule
    plot_v3_epsilon_schedule(
        "config/best.yaml", 10000, f"{figures_dir}/V3_epsilon_schedule.png"
    )
    print("  [OK] V3: Epsilon schedule generated")

    # V4: Validation curve & convergence
    plot_v4_validation_curve(
        f"{results_dir}/final_runs/training_log_seed_104.csv",
        f"{figures_dir}/V4_validation_curve.png",
    )
    print("  [OK] V4: Validation curve generated")

    # V5: Tuning sweeps / heatmap
    plot_v5_tuning_heatmap(
        f"{results_dir}/tuning_results.csv", f"{figures_dir}/V5_tuning_heatmap.png"
    )
    print("  [OK] V5: Tuning heatmap generated")

    # V6 & V7: Distributions
    p_test = Path(test_metrics_path)
    if p_test.exists():
        import json

        with open(p_test, encoding="utf-8") as f:
            test_data = json.load(f)
        plot_v6_test_reward_distribution(
            test_data["raw_rewards"], f"{figures_dir}/V6_test_reward_distribution.png"
        )
        print("  [OK] V6: Test reward distribution generated")
        plot_v7_test_steps_distribution(
            test_data["raw_steps"], f"{figures_dir}/V7_test_steps_distribution.png"
        )
        print("  [OK] V7: Test steps distribution generated")

    # V8 & V9: Policy grid and Replay GIF
    p_q = Path(q_table_path)
    if p_q.exists():
        q_table = np.load(p_q)
        plot_v8_policy_grid(
            q_table,
            passenger_loc=0,
            dest_loc=1,
            out_path=f"{figures_dir}/V8_policy_grid.png",
        )
        print("  [OK] V8: Spatial policy map generated")
        plot_v9_replay_gif(
            q_table, out_gif_path=f"{figures_dir}/V9_replay.gif", seed=10001
        )
        print("  [OK] V9: Replay animated GIF generated")


if __name__ == "__main__":
    generate_all_figures()
