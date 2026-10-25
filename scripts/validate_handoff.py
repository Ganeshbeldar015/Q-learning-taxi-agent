#!/usr/bin/env python3
"""Validates handoff contract JSON files between pipeline roles."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ROLES_REQUIRING_CHECKS = {"builder", "test-author", "qa", "security"}


def validate_handoff(handoff_data: dict, root: Path = ROOT) -> tuple[bool, list[str]]:
    errors = []
    required_keys = [
        "task_id",
        "from",
        "to",
        "artifact",
        "status",
        "summary",
        "inputs_consumed",
        "acceptance_refs",
        "open_questions",
        "assumptions",
        "files_touched",
        "checks_run",
        "next_expected_artifact",
    ]
    for key in required_keys:
        if key not in handoff_data:
            errors.append(f"Missing required key: '{key}'")

    if handoff_data.get("status") != "complete":
        errors.append(
            f"Handoff status must be 'complete', got '{handoff_data.get('status')}'"
        )

    # Artifact must exist on disk
    artifact_rel = handoff_data.get("artifact", "")
    if not artifact_rel or not (root / artifact_rel).exists():
        errors.append(f"Artifact does not exist: '{artifact_rel}'")

    # Summary length <= 120 words
    summary = handoff_data.get("summary", "")
    if len(summary.split()) > 120:
        errors.append("Summary exceeds 120 words limit")

    # Acceptance refs must exist in harness.md
    harness_path = root / "docs/harness.md"
    if harness_path.exists():
        harness_text = harness_path.read_text(encoding="utf-8")
        for ac in handoff_data.get("acceptance_refs", []):
            if ac not in harness_text:
                errors.append(f"Acceptance ref '{ac}' not defined in docs/harness.md")

    # Checks run cannot be empty for roles requiring checks
    role_from = handoff_data.get("from")
    if role_from in ROLES_REQUIRING_CHECKS:
        checks = handoff_data.get("checks_run", [])
        if not checks:
            errors.append(
                f"Role '{role_from}' must record at least one check in 'checks_run'"
            )
        for c in checks:
            # Test Author's red test run intentionally exits non-zero
            if role_from == "test-author" and "pytest" in c.get("cmd", ""):
                continue
            if c.get("exit") != 0:
                errors.append(
                    f"Check '{c.get('cmd')}' failed with exit code {c.get('exit')}"
                )

    return len(errors) == 0, errors
