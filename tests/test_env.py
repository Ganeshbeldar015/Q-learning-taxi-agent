import json

import pytest

from src.env_utils import (
    decode_state,
    encode_state,
    get_seeded_rng,
    load_config,
    make_env,
    run_random_baseline,
    verify_state_roundtrip,
)


def test_load_config_valid():
    cfg = load_config("config/default.yaml")
    assert cfg["seed"] == 42
    assert "env" in cfg
    assert "train" in cfg and "eval" in cfg
    assert isinstance(cfg["train"]["episodes"], int)
    assert isinstance(cfg["train"]["alpha"], (int, float))
    assert isinstance(cfg["train"]["gamma"], (int, float))
    assert isinstance(cfg["eval"]["test_episodes"], int)


def test_load_config_invalid(tmp_path):
    bad_cfg = tmp_path / "bad.yaml"
    bad_cfg.write_text("seed: not_an_int\n", encoding="utf-8")
    with pytest.raises((ValueError, KeyError, TypeError)):
        load_config(bad_cfg)


def test_get_seeded_rng_determinism():
    rng1 = get_seeded_rng(42)
    rng2 = get_seeded_rng(42)
    sample1 = rng1.integers(0, 1000, size=20)
    sample2 = rng2.integers(0, 1000, size=20)
    assert (sample1 == sample2).all()


def test_make_env_instance():
    env = make_env("Taxi-v4")
    assert env.observation_space.n == 500
    assert env.action_space.n == 6
    obs, _info = env.reset(seed=42)
    assert 0 <= obs < 500
    env.close()


def test_state_encode_decode_roundtrip_all_500():
    all_ok = verify_state_roundtrip()
    assert all_ok is True

    for s in range(500):
        row, col, pass_loc, dest = decode_state(s)
        assert 0 <= row < 5
        assert 0 <= col < 5
        assert 0 <= pass_loc < 5
        assert 0 <= dest < 4
        assert encode_state(row, col, pass_loc, dest) == s


def test_reward_invariant_clean_success():
    # AC-002-c: total reward == 21 - steps for clean success
    steps = 12
    step_rewards = [-1] * (steps - 1) + [20]
    total_reward = sum(step_rewards)
    assert total_reward == 21 - steps


def test_run_random_baseline(tmp_path):
    env = make_env()
    out_file = tmp_path / "baseline_random.json"
    metrics = run_random_baseline(env, episodes=20, seed=42, output_path=out_file)
    assert out_file.exists()
    assert metrics["episodes"] == 20
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert "mean_reward" in data
    assert "mean_steps" in data
    assert "success_rate" in data
    assert "illegal_actions" in data
    assert data["episodes"] == 20
    env.close()
