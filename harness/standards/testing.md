# Testing Standards

1. **Test-First Development**:
   - Test Author writes failing tests from acceptance criteria before implementation begins.
   - Tests must fail for the correct reason and be recorded in `04-tests.md`.

2. **Test Categories**:
   - **Unit Tests**: Fast (<1s each), hermetic, deterministic, covering update math, tie-breaking, and schedule calculations.
   - **Integration Tests**: Environment resets, step transitions, baseline rollouts, and training loop progress.
   - **Invariants**: Enforce reward equation `total_reward == 21 - steps` on clean episodes; state round-trip decode/encode matching.

3. **Determinism**:
   - Explicit random seeds for all test cases.
   - Never use `time.sleep` or unseeded randomness.
