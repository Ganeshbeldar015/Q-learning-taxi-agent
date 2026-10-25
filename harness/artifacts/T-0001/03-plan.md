---
status: complete
owner: tech-lead
task: T-0001
inputs:
  - harness/artifacts/T-0001/01-spec.md
  - harness/artifacts/T-0001/02-design.md
acceptance_refs:
  - AC-001-a
  - AC-001-b
  - AC-002-a
  - AC-002-b
  - AC-002-c
---
# 03-plan — T-0001: Implementation Plan & Lane Decomposition

## 1. File Modification & Creation Roadmap

| Path | Action | Description |
|---|---|---|
| `requirements.txt` | Create | Pinned dependency list (`gymnasium`, `numpy`, `pandas`, `matplotlib`, `pyyaml`, `pytest`, `ruff`, `mypy`). |
| `config/default.yaml` | Create | Canonical default configuration matching raw.md §4. |
| `src/__init__.py` | Create | Package initialization. |
| `src/env_utils.py` | Create | Core environment creation, config validation, seeding, state math, and random baseline. |
| `tests/__init__.py` | Create | Test package initialization. |
| `tests/test_env.py` | Create | Unit tests verifying AC-001-a through AC-002-c. |
| `results/baseline_random.json` | Create | Generated artifact from 1,000-episode random baseline rollout. |

## 2. Lane Decomposition & Ownership Locks
- **Lane**: `lane-env`
- **Owner**: `builder`
- **Exclusive Paths**:
  - `requirements.txt`
  - `config/default.yaml`
  - `src/__init__.py`
  - `src/env_utils.py`
  - `results/baseline_random.json`

## 3. Test Strategy for Test Author
Test Author must implement `tests/test_env.py` with tests:
1. `test_load_config_valid_and_invalid`: Validates key checking and type enforcement (`AC-001-a`).
2. `test_seeded_rng_and_env_reproducibility`: Verifies determinism from `get_seeded_rng` (`AC-001-b`).
3. `test_state_encode_decode_roundtrip_all_500`: Iterates through all 500 states and asserts identity round-trip (`AC-002-a`).
4. `test_reward_invariant_clean_success`: Simulates/injects transitions to assert `reward == 21 - steps` on clean success (`AC-002-c`).
5. `test_random_baseline_execution`: Verifies that `run_random_baseline` produces correct schema and metrics (`AC-002-b`).

## 4. Rollback & Contingency Plan
If Gymnasium fails to instantiate, fallback to synthetic state transition mapping using environment attributes.
All changes are hermetic to `src/env_utils.py` and `config/default.yaml`.
