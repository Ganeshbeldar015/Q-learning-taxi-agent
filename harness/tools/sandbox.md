# Execution Sandbox Rules

1. **Working Directory Isolation**:
   - Work strictly within the repository workspace root (`D:\WorkSpace\ML mini project`) and temporary test directories.
   - Never write to user home directories, root system paths, or unapproved folders.

2. **Permitted Commands**:
   - Only execute commands declared in `docs/harness.md §8` plus standard inspection tools (`git status`, `git diff`, `git log`, `python`, `pytest`, `ruff`, `mypy`).

3. **Network Constraints**:
   - No arbitrary outbound network calls during training, evaluation, or testing.
   - Package installations require explicit confirmation.

4. **Resource Bounds**:
   - Default command timeout: 120 seconds (except long training runs bounded by budget).
   - Abort commands hanging without progress.

5. **Secrets Hygiene**:
   - Never log API keys, personal access tokens, or sensitive credentials in logs or output artifacts.
