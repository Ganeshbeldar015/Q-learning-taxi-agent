# Definition of Done

A task or sprint is officially Done when all of the following conditions are satisfied:

1. **Artifact Trail**:
   - All expected stage artifacts (`01` through `08`) exist and are valid.
   - All handoff JSON contracts are validated by the Orchestrator.

2. **Quality Gates**:
   - **G0 (Harness Integrity)**: `python scripts/check_harness.py && pytest tests/harness` passes.
   - **G1 (Lane Gate)**: `ruff check`, `ruff format --check`, and unit tests pass on changed lane code.
   - **G2 (Merge Gate)**: Full test suite passes; test coverage meets the 85% floor on `src/`.
   - **G3 (Audit Gate)**: Zero blocker/major code review items, zero high/critical security issues, docs updated.
   - **G4 (QA Gate)**: All assigned acceptance criteria validated against concrete execution output.

3. **Deterministic Verification**:
   - Fixed random seeds produce bitwise reproducible results.
   - No skipped, commented out, or weakened tests.
