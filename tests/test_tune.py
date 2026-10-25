from pathlib import Path

import yaml

from src.tune import (
    retrain_final_models,
    run_trial,
    save_best_config,
    select_best_configuration,
)


def test_select_best_configuration():
    # Synthetic trial results
    # Config 1: alpha=0.1, gamma=0.99, decay=0.999 -> converged at ep 2000, 2200, 2100 (mean 2100, conv rate 100%)
    # Config 2: alpha=0.3, gamma=0.99, decay=0.995 -> converged at ep 1200, 1100, 1300 (mean 1200, conv rate 100%)
    # Config 3: alpha=0.05, gamma=0.8, decay=0.99 -> only 1 of 3 converged (conv rate 33%)
    trials = [
        {
            "alpha": 0.1,
            "gamma": 0.99,
            "epsilon_decay": 0.999,
            "epsilon_min": 0.01,
            "seed": 1,
            "converged": True,
            "convergence_episode": 2000,
            "final_val_reward": 8.0,
            "wall_clock_time": 1.0,
        },
        {
            "alpha": 0.1,
            "gamma": 0.99,
            "epsilon_decay": 0.999,
            "epsilon_min": 0.01,
            "seed": 2,
            "converged": True,
            "convergence_episode": 2200,
            "final_val_reward": 8.2,
            "wall_clock_time": 1.0,
        },
        {
            "alpha": 0.1,
            "gamma": 0.99,
            "epsilon_decay": 0.999,
            "epsilon_min": 0.01,
            "seed": 3,
            "converged": True,
            "convergence_episode": 2100,
            "final_val_reward": 8.1,
            "wall_clock_time": 1.0,
        },
        {
            "alpha": 0.3,
            "gamma": 0.99,
            "epsilon_decay": 0.995,
            "epsilon_min": 0.01,
            "seed": 1,
            "converged": True,
            "convergence_episode": 1200,
            "final_val_reward": 8.3,
            "wall_clock_time": 1.0,
        },
        {
            "alpha": 0.3,
            "gamma": 0.99,
            "epsilon_decay": 0.995,
            "epsilon_min": 0.01,
            "seed": 2,
            "converged": True,
            "convergence_episode": 1100,
            "final_val_reward": 8.4,
            "wall_clock_time": 1.0,
        },
        {
            "alpha": 0.3,
            "gamma": 0.99,
            "epsilon_decay": 0.995,
            "epsilon_min": 0.01,
            "seed": 3,
            "converged": True,
            "convergence_episode": 1300,
            "final_val_reward": 8.2,
            "wall_clock_time": 1.0,
        },
        {
            "alpha": 0.05,
            "gamma": 0.8,
            "epsilon_decay": 0.99,
            "epsilon_min": 0.01,
            "seed": 1,
            "converged": True,
            "convergence_episode": 2900,
            "final_val_reward": 7.1,
            "wall_clock_time": 1.0,
        },
        {
            "alpha": 0.05,
            "gamma": 0.8,
            "epsilon_decay": 0.99,
            "epsilon_min": 0.01,
            "seed": 2,
            "converged": False,
            "convergence_episode": None,
            "final_val_reward": 2.0,
            "wall_clock_time": 1.0,
        },
        {
            "alpha": 0.05,
            "gamma": 0.8,
            "epsilon_decay": 0.99,
            "epsilon_min": 0.01,
            "seed": 3,
            "converged": False,
            "convergence_episode": None,
            "final_val_reward": 1.0,
            "wall_clock_time": 1.0,
        },
    ]

    best, _df = select_best_configuration(trials, threshold=7.0, min_conv_rate=0.8)
    assert best["alpha"] == 0.3
    assert best["gamma"] == 0.99
    assert best["epsilon_decay"] == 0.995
    assert best["mean_convergence_episode"] == 1200.0


def test_save_best_config(tmp_path):
    best_params = {
        "alpha": 0.4,
        "gamma": 0.95,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.02,
    }
    out_yaml = tmp_path / "best.yaml"
    save_best_config(
        best_params, base_cfg_path="config/default.yaml", out_path=out_yaml
    )

    assert out_yaml.exists()
    data = yaml.safe_load(out_yaml.read_text(encoding="utf-8"))
    assert data["train"]["alpha"] == 0.4
    assert data["train"]["gamma"] == 0.95
    assert data["train"]["epsilon_decay"] == 0.995
    assert data["train"]["epsilon_min"] == 0.02


def test_run_trial_smoke():
    params = {"alpha": 0.2, "gamma": 0.99, "epsilon_decay": 0.99, "epsilon_min": 0.05}
    res = run_trial(params, episodes=30, seed=42)
    assert "converged" in res
    assert "final_val_reward" in res
    assert "wall_clock_time" in res
    assert res["alpha"] == 0.2


def test_retrain_final_models_smoke(tmp_path):
    out_dir = tmp_path / "final_runs"
    primary_out = tmp_path / "q_table_final.npy"

    # Train 2 seeds for 20 episodes
    seeds = [101, 102]
    cfg = yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))
    cfg["train"]["episodes"] = 20
    cfg["eval"]["every"] = 10
    cfg["eval"]["episodes"] = 5

    res = retrain_final_models(
        cfg, seeds=seeds, out_dir=out_dir, primary_out=primary_out
    )
    assert primary_out.exists()
    assert (out_dir / "q_table_seed_101.npy").exists()
    assert (out_dir / "q_table_seed_102.npy").exists()
    assert "best_seed" in res
    assert res["best_seed"] in seeds
