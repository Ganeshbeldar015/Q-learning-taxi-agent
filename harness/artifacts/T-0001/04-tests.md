---
status: complete
owner: test-author
task: T-0001
inputs:
  - harness/artifacts/T-0001/01-spec.md
  - harness/artifacts/T-0001/03-plan.md
acceptance_refs:
  - AC-001-a
  - AC-001-b
  - AC-002-a
  - AC-002-b
  - AC-002-c
---
# 04-tests — T-0001: Red Test Execution Evidence

## 1. Acceptance Criteria to Test Mapping

| Acceptance Criterion | Test Function | Description |
|---|---|---|
| `AC-001-a` | `tests/test_env.py::test_load_config_valid` | Asserts valid config loads required keys & types. |
| `AC-001-a` | `tests/test_env.py::test_load_config_invalid` | Asserts bad configs raise ValueError/KeyError. |
| `AC-001-b` | `tests/test_env.py::test_get_seeded_rng_determinism` | Asserts identical sequence from same seed. |
| `AC-001-b` | `tests/test_env.py::test_make_env_instance` | Asserts observation space 500 and action space 6. |
| `AC-002-a` | `tests/test_env.py::test_state_encode_decode_roundtrip_all_500` | Verifies identity round-trip for all 500 states. |
| `AC-002-b` | `tests/test_env.py::test_run_random_baseline` | Verifies 1,000 episode random evaluation schema. |
| `AC-002-c` | `tests/test_env.py::test_reward_invariant_clean_success` | Asserts `total_reward == 21 - steps` for clean runs. |

## 2. Red Run Captured Output
```
ImportError while importing test module 'tests/test_env.py'.
Traceback:
tests/test_env.py:6: in <module>
    from src.env_utils import (
E   ModuleNotFoundError: No module named 'src'
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
```
The test suite fails cleanly due to missing `src.env_utils` implementation. Ready for Builder.
