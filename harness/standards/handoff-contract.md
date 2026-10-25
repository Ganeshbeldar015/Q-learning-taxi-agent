# Handoff Contract Specification

Every stage transition produces a handoff JSON file located at `harness/artifacts/<TASK-ID>/handoff-<from>-to-<to>.json`.

## Schema Requirements
```json
{
  "task_id": "T-0001",
  "from": "tech-lead",
  "to": "test-author",
  "artifact": "harness/artifacts/T-0001/03-plan.md",
  "status": "complete",
  "summary": "Summary of stage output in 120 words or fewer.",
  "inputs_consumed": ["01-spec.md", "02-design.md"],
  "acceptance_refs": ["AC-001-a", "AC-001-b"],
  "open_questions": [],
  "assumptions": [],
  "files_touched": ["harness/artifacts/T-0001/03-plan.md"],
  "checks_run": [{"cmd": "pytest tests/ -v", "exit": 0}],
  "next_expected_artifact": "04-tests.md"
}
```

## Rejection Criteria
The Orchestrator automatically rejects a handoff if:
1. The declared artifact file does not exist on disk.
2. The `status` field is not `complete`.
3. An acceptance criterion in `acceptance_refs` is not defined in `docs/harness.md`.
4. Any modified file in `files_touched` is outside the role's allow-list.
5. `checks_run` is empty for builder, test-author, qa, or security.
