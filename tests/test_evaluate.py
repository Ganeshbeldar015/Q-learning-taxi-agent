"""Unit tests for policy evaluation and benchmarking."""

import numpy as np

from src.evaluate import (
    compute_bootstrap_ci,
    evaluate_policy,
    generate_comparison_table,
)


def test_compute_bootstrap_ci():
    # Synthetic normal data with mean 10.0
    rng = np.random.default_rng(42)
    data = rng.normal(loc=10.0, scale=1.0, size=500)
    ci_low, ci_high = compute_bootstrap_ci(
        data, n_bootstraps=500, confidence_level=0.95, seed=42
    )

    assert ci_low < 10.0 < ci_high
    assert 9.8 < ci_low < 10.0
    assert 10.0 < ci_high < 10.2


def test_evaluate_policy_untrained_smoke():
    # Untrained all-zeros policy evaluated over 5 episodes
    metrics = evaluate_policy(None, episodes=5, seed_offset=10000, max_steps=20)

    assert "mean_reward" in metrics
    assert "mean_steps" in metrics
    assert "success_rate" in metrics
    assert "total_illegal_actions" in metrics
    assert "ci_95" in metrics
    assert len(metrics["raw_rewards"]) == 5
    assert len(metrics["raw_steps"]) == 5


def test_evaluate_policy_loaded_q_table():
    # Synthetic Q-table of zeros
    q_table = np.zeros((500, 6), dtype=np.float64)
    metrics = evaluate_policy(q_table, episodes=3, seed_offset=10000, max_steps=20)

    assert metrics["episodes"] == 3
    assert len(metrics["raw_rewards"]) == 3


def test_generate_comparison_table(tmp_path):
    # Create mock result files
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)

    # 1. baseline_random.json
    random_json = results_dir / "baseline_random.json"
    random_json.write_text(
        '{"episodes": 100, "mean_reward": -700.0, "std_reward": 50.0, "mean_steps": 200.0, '
        '"success_rate": 0.05, "total_illegal_actions": 500, "mean_illegal_actions": 5.0}',
        encoding="utf-8",
    )

    # 2. q_table_baseline.npy & q_table_final.npy
    q_dummy = np.zeros((500, 6), dtype=np.float64)
    np.save(results_dir / "q_table_baseline.npy", q_dummy)
    np.save(results_dir / "q_table_final.npy", q_dummy)

    out_csv = results_dir / "comparison_table.csv"
    df = generate_comparison_table(
        results_dir=str(results_dir),
        out_csv_path=str(out_csv),
        test_episodes=3,
        seed_offset=10000,
    )

    assert out_csv.exists()
    assert len(df) == 4
    expected_policies = {
        "Random Policy",
        "Untrained Q-Agent",
        "Baseline Trained Model",
        "Tuned Final Model",
    }
    assert set(df["Policy"].tolist()) == expected_policies
    assert "Mean Reward" in df.columns
    assert "Success Rate" in df.columns
    assert "Illegal Actions" in df.columns
