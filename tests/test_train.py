import pandas as pd

from src.env_utils import load_config
from src.train import check_convergence, train


def test_train_creates_csv_and_saves_q_table(tmp_path):
    cfg = load_config("config/default.yaml")
    # Small test budget
    cfg["train"]["episodes"] = 30
    cfg["eval"]["every"] = 10
    cfg["eval"]["episodes"] = 5

    csv_path = tmp_path / "train_log.csv"
    q_table_path = tmp_path / "q_table.npy"

    results = train(cfg, log_csv_path=csv_path, q_table_save_path=q_table_path)

    assert csv_path.exists()
    assert q_table_path.exists()
    assert "episodes_trained" in results
    assert results["episodes_trained"] == 30

    df = pd.read_csv(csv_path)
    assert len(df) == 30
    for col in ("episode", "reward", "steps", "epsilon", "illegal_actions", "success"):
        assert col in df.columns


def test_convergence_detector_logic():
    # Synthetic validation rewards: threshold 7.0, patience 3
    # Case 1: never reaches 7.0
    val_rewards_1 = [1.0, 2.0, 3.0, 5.0]
    eval_episodes = [100, 200, 300, 400]
    converged_ep, is_conv = check_convergence(
        val_rewards_1, eval_episodes, threshold=7.0, patience=3
    )
    assert not is_conv
    assert converged_ep is None

    # Case 2: reaches 7.0 for 3 consecutive checks: [6.0, 7.1, 7.5, 8.0]
    val_rewards_2 = [6.0, 7.1, 7.5, 8.0]
    eval_episodes = [100, 200, 300, 400]
    converged_ep, is_conv = check_convergence(
        val_rewards_2, eval_episodes, threshold=7.0, patience=3
    )
    assert is_conv
    # Earliest episode meeting the 3-in-a-row is episode 200
    assert converged_ep == 200
