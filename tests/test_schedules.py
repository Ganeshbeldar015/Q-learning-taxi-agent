import pytest

from src.schedules import LinearSchedule, MultiplicativeSchedule, get_schedule


def test_multiplicative_schedule_decay_and_floor():
    sched = MultiplicativeSchedule(start=1.0, min_epsilon=0.01, decay=0.99)
    assert sched.get_epsilon(0) == 1.0
    assert pytest.approx(sched.get_epsilon(1)) == 0.99
    assert pytest.approx(sched.get_epsilon(2)) == 0.99**2
    assert sched.get_epsilon(10000) == 0.01  # strictly hits floor


def test_linear_schedule_decay_and_floor():
    sched = LinearSchedule(start=1.0, min_epsilon=0.1, decay_episodes=100)
    assert sched.get_epsilon(0) == 1.0
    assert pytest.approx(sched.get_epsilon(50)) == 0.55
    assert pytest.approx(sched.get_epsilon(100)) == 0.1
    assert sched.get_epsilon(200) == 0.1  # floor preserved


def test_get_schedule_factory():
    mult_cfg = {
        "train": {
            "schedule": "multiplicative",
            "epsilon_start": 1.0,
            "epsilon_min": 0.05,
            "epsilon_decay": 0.995,
        }
    }
    sched1 = get_schedule(mult_cfg)
    assert isinstance(sched1, MultiplicativeSchedule)

    lin_cfg = {
        "train": {
            "schedule": "linear",
            "epsilon_start": 0.8,
            "epsilon_min": 0.02,
            "episodes": 2000,
        }
    }
    sched2 = get_schedule(lin_cfg)
    assert isinstance(sched2, LinearSchedule)
