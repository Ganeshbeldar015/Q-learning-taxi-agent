#!/usr/bin/env python3
"""Circuit breaker for the harness. Evaluates budgets and failure patterns."""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def compute_failure_signature(test_names: list, error_text: str) -> str:
    content = "|".join(sorted(test_names)) + ":::" + error_text.strip()
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def evaluate_breaker(
    state: dict, budgets: dict, event: dict | None = None
) -> tuple[bool, str]:
    """
    Evaluates breaker conditions:
    1. Same stage failed > max_stage_retries
    2. Same failure signature repeated >= max_identical_failures
    3. Diff exceeds size limits
    4. Guard violation -> immediate
    5. Forbidden/unapproved tool -> immediate
    6. No progress in last no_progress_window
    7. Budget exhaustion
    8. Malformed artifact/handoff twice in a row
    """
    event = event or {}

    # Condition 4: Guard violation
    if event.get("guard_violation"):
        return (
            True,
            f"Immediate trip: Guard violation detected ({event.get('guard_violation')})",
        )

    # Condition 5: Forbidden tool use
    if event.get("forbidden_tool"):
        return (
            True,
            f"Immediate trip: Attempted forbidden tool '{event.get('forbidden_tool')}'",
        )

    # Condition 1: max_stage_retries
    attempts = state.get("attempts", {})
    current_stage = state.get("current_stage", "")
    max_retries = budgets.get("max_stage_retries", 3)
    if attempts.get(current_stage, 0) > max_retries:
        return (
            True,
            f"Stage '{current_stage}' failed {attempts[current_stage]} times (max {max_retries})",
        )

    # Condition 2: max_identical_failures
    sig_history = state.get("failure_signatures", [])
    max_identical = budgets.get("max_identical_failures", 2)
    current_sig = event.get("signature")
    if current_sig:
        matching = [s for s in sig_history if s == current_sig]
        if len(matching) >= max_identical:
            return (
                True,
                f"Identical failure signature '{current_sig}' repeated {len(matching)} times (max {max_identical})",
            )

    # Condition 3: Diff limits
    diff_lines = event.get("diff_lines", 0)
    max_lines = budgets.get("max_diff_lines_per_lane", 400)
    if diff_lines > max_lines:
        return True, f"Diff size {diff_lines} lines exceeds lane budget of {max_lines}"

    diff_files = event.get("diff_files", 0)
    max_files = budgets.get("max_files_per_lane", 15)
    if diff_files > max_files:
        return (
            True,
            f"Modified files count {diff_files} exceeds lane budget of {max_files}",
        )

    # Condition 6: No progress window
    fail_counts = state.get("failing_test_history", [])
    window = budgets.get("no_progress_window", 3)
    if len(fail_counts) >= window:
        recent = fail_counts[-window:]
        if all(c >= recent[0] for c in recent):
            return (
                True,
                f"No progress in reducing failing tests over last {window} iterations: {recent}",
            )

    # Condition 7: Budget exhaustion (session minutes, tool calls)
    tool_calls = state.get("tool_calls_in_stage", 0)
    max_calls = budgets.get("max_tool_calls_per_stage", 150)
    if tool_calls > max_calls:
        return True, f"Tool calls in stage ({tool_calls}) exceeded budget ({max_calls})"

    # Condition 8: Malformed artifacts/handoffs
    malformed_streak = state.get("malformed_artifact_streak", 0)
    if malformed_streak >= 2:
        return (
            True,
            f"Malformed artifact or handoff encountered {malformed_streak} times consecutively",
        )

    return False, ""


def trip_breaker(state_path: Path, reason: str):
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        data = {}
    data["breaker"] = {"tripped": True, "reason": reason}
    state_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main():
    state_file = ROOT / "harness/state/state.json"
    budget_file = ROOT / "harness/state/budgets.json"
    if not state_file.exists() or not budget_file.exists():
        print("Breaker: state.json or budgets.json missing.")
        return 0

    state = json.loads(state_file.read_text(encoding="utf-8"))
    budgets = json.loads(budget_file.read_text(encoding="utf-8"))

    tripped, reason = evaluate_breaker(state, budgets)
    if tripped:
        print(f"CIRCUIT BREAKER TRIPPED: {reason}")
        trip_breaker(state_file, reason)
        return 1
    print("CIRCUIT BREAKER: Normal operating parameters.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
