"""Unit tests for visualization plotting routines."""

import numpy as np

from src.plots import (
    plot_v3_epsilon_schedule,
    plot_v6_test_reward_distribution,
    plot_v7_test_steps_distribution,
    plot_v8_policy_grid,
)


def test_plot_v3_epsilon_schedule(tmp_path):
    out_png = tmp_path / "v3.png"
    plot_v3_epsilon_schedule(
        config_path="config/default.yaml",
        episodes=100,
        out_path=str(out_png),
    )
    assert out_png.exists()
    assert out_png.stat().st_size > 1000


def test_plot_v6_and_v7_distributions(tmp_path):
    v6_png = tmp_path / "v6.png"
    v7_png = tmp_path / "v7.png"

    rewards = [8.0, 9.0, 7.5, 8.5, 9.5, 7.0, 8.0, 8.5]
    steps = [12, 13, 11, 14, 15, 12, 13, 14]

    plot_v6_test_reward_distribution(rewards, str(v6_png))
    plot_v7_test_steps_distribution(steps, str(v7_png))

    assert v6_png.exists()
    assert v6_png.stat().st_size > 1000
    assert v7_png.exists()
    assert v7_png.stat().st_size > 1000


def test_plot_v8_policy_grid(tmp_path):
    v8_png = tmp_path / "v8.png"
    q_table = np.zeros((500, 6), dtype=np.float64)

    # Set some actions
    q_table[0, 0] = 5.0  # south
    q_table[1, 1] = 5.0  # north

    plot_v8_policy_grid(q_table, passenger_loc=0, dest_loc=1, out_path=str(v8_png))
    assert v8_png.exists()
    assert v8_png.stat().st_size > 1000
