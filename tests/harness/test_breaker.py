import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import breaker as br

BUDGETS = {
    "max_stage_retries": 3,
    "max_identical_failures": 2,
    "max_diff_lines_per_lane": 400,
    "max_files_per_lane": 15,
    "max_parallel_agents": 3,
    "max_session_minutes": 90,
    "max_tool_calls_per_stage": 150,
    "no_progress_window": 3,
}


def test_normal_operating_state_does_not_trip():
    state = {
        "current_stage": "build",
        "attempts": {"build": 1},
        "breaker": {"tripped": False},
    }
    tripped, _ = br.evaluate_breaker(state, BUDGETS)
    assert not tripped


def test_trip_condition_1_stage_retries_exceeded():
    state = {"current_stage": "build", "attempts": {"build": 4}}
    tripped, reason = br.evaluate_breaker(state, BUDGETS)
    assert tripped
    assert "failed 4 times" in reason


def test_trip_condition_2_identical_failures():
    sig = br.compute_failure_signature(["test_bellman"], "AssertionError: 0 != 1")
    state = {"current_stage": "build", "failure_signatures": [sig, sig]}
    tripped, reason = br.evaluate_breaker(state, BUDGETS, event={"signature": sig})
    assert tripped
    assert "Identical failure signature" in reason


def test_trip_condition_3_diff_lines_exceeded():
    state = {"current_stage": "build"}
    tripped, reason = br.evaluate_breaker(state, BUDGETS, event={"diff_lines": 450})
    assert tripped
    assert "exceeds lane budget" in reason


def test_trip_condition_4_guard_violation():
    state = {"current_stage": "builder"}
    tripped, reason = br.evaluate_breaker(
        state, BUDGETS, event={"guard_violation": "docs/raw.md"}
    )
    assert tripped
    assert "Guard violation" in reason


def test_trip_condition_5_forbidden_tool():
    state = {"current_stage": "builder"}
    tripped, reason = br.evaluate_breaker(
        state, BUDGETS, event={"forbidden_tool": "deploy"}
    )
    assert tripped
    assert "forbidden tool" in reason


def test_trip_condition_6_no_progress():
    state = {"current_stage": "build", "failing_test_history": [5, 5, 6]}
    tripped, reason = br.evaluate_breaker(state, BUDGETS)
    assert tripped
    assert "No progress" in reason


def test_trip_condition_7_tool_calls_exhausted():
    state = {"current_stage": "build", "tool_calls_in_stage": 155}
    tripped, reason = br.evaluate_breaker(state, BUDGETS)
    assert tripped
    assert "exceeded budget" in reason


def test_trip_condition_8_malformed_artifact_streak():
    state = {"current_stage": "spec", "malformed_artifact_streak": 2}
    tripped, reason = br.evaluate_breaker(state, BUDGETS)
    assert tripped
    assert "Malformed artifact" in reason
