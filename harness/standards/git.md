# Git Standards

1. **Commit Message Format**:
   - Follow Conventional Commits: `<type>(<scope>): <short description>`.
   - Examples: `feat(agent): implement vectorized bellman update`, `test(harness): add handoff validation tests`.

2. **Branching Strategy**:
   - `master`: Stable, protected integration branch.
   - `lane/<TASK-ID>-<lane>`: Ephemeral branches for isolated builder lanes.

3. **Forbidden Actions**:
   - Never force-push (`git push --force`).
   - Never reset hard to origin without human approval.
