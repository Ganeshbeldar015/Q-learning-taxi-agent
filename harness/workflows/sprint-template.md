# Sprint Execution Template

## Ceremonies
1. **Sprint Planning**:
   - Select backlog items for current sprint.
   - Generate `harness/tasks/<TASK-ID>.md` and skeleton artifact directory `harness/artifacts/<TASK-ID>/`.
   - Update `harness/state/state.json`.

2. **Stage Execution**:
   - Follow linear pipeline from PO-Spec to QA Evaluator.
   - Record handoff contract JSON after each stage.
   - Enforce path allow-lists via `scripts/guard_paths.py`.

3. **Sprint Review**:
   - Verify all task criteria against `docs/harness.md §5`.
   - Record demo results in `harness/logs/decisions.log.md`.

4. **Sprint Retrospective**:
   - Librarian records anti-patterns and rules in `docs/memory/lessons.md`.
   - Reconcile harness configuration if needed.
