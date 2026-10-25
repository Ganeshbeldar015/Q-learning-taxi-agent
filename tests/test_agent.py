import numpy as np
import pytest

from src.agent import QLearningAgent


def test_q_table_initialization():
    agent = QLearningAgent(num_states=500, num_actions=6, alpha=0.1, gamma=0.99)
    assert agent.q_table.shape == (500, 6)
    assert np.all(agent.q_table == 0.0)


def test_bellman_update_hand_computed():
    agent = QLearningAgent(num_states=500, num_actions=6, alpha=0.1, gamma=0.99)
    # Set Q(s=5, a=2) = 5.0
    agent.q_table[5, 2] = 5.0

    # Transition: s=0, a=1, r=-1, s'=5, terminated=False, truncated=False
    # Expected target: -1 + 0.99 * 5.0 = 3.95
    # Expected new Q(0, 1): 0.0 + 0.1 * (3.95 - 0.0) = 0.395
    td_error = agent.update(
        state=0, action=1, reward=-1.0, next_state=5, terminated=False, truncated=False
    )
    assert pytest.approx(td_error, abs=1e-5) == 3.95
    assert pytest.approx(agent.q_table[0, 1], abs=1e-5) == 0.395


def test_bellman_update_terminal_no_bootstrap():
    agent = QLearningAgent(num_states=500, num_actions=6, alpha=0.2, gamma=0.99)
    agent.q_table[10, 3] = 100.0  # Large future Q-value

    # Terminal state reached: terminated=True
    # Future bootstrap must be 0! Target = 20.0
    # Expected Q: 0.0 + 0.2 * (20.0 - 0.0) = 4.0
    td_error = agent.update(
        state=0, action=4, reward=20.0, next_state=10, terminated=True
    )
    assert pytest.approx(td_error, abs=1e-5) == 20.0
    assert pytest.approx(agent.q_table[0, 4], abs=1e-5) == 4.0


def test_bellman_update_truncated_with_bootstrap():
    agent = QLearningAgent(num_states=500, num_actions=6, alpha=0.1, gamma=0.9)
    agent.q_table[20, 1] = 10.0

    # Truncated (timeout): terminated=False, truncated=True
    # Must still bootstrap from next_state! Target = -1 + 0.9 * 10 = 8.0
    td_error = agent.update(
        state=1, action=0, reward=-1.0, next_state=20, terminated=False, truncated=True
    )
    assert pytest.approx(td_error, abs=1e-5) == 8.0
    assert pytest.approx(agent.q_table[1, 0], abs=1e-5) == 0.8


def test_random_tie_breaking():
    agent = QLearningAgent(num_states=500, num_actions=6)
    rng = np.random.default_rng(42)
    # All Q-values at state 0 are 0.0
    counts = np.zeros(6, dtype=int)
    for _ in range(1200):
        action = agent.greedy_action(0, rng=rng)
        counts[action] += 1

    # In 1200 trials, each action should get roughly 200 counts (p > 0.001 that count is between 120 and 280)
    for a in range(6):
        assert 130 < counts[a] < 270, f"Action {a} had unexpected count {counts[a]}"


def test_q_table_save_and_load(tmp_path):
    agent = QLearningAgent(num_states=500, num_actions=6)
    agent.q_table[42, 3] = 7.77
    save_file = tmp_path / "test_q.npy"
    agent.save(save_file)
    assert save_file.exists()

    agent2 = QLearningAgent(num_states=500, num_actions=6)
    agent2.load(save_file)
    assert pytest.approx(agent2.q_table[42, 3]) == 7.77
    assert np.array_equal(agent.q_table, agent2.q_table)
