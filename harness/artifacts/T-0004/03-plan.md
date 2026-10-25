---
status: complete
owner: tech-lead
task: T-0004
inputs:
  - harness/artifacts/T-0004/02-design.md
acceptance_refs:
  - AC-007-a
  - AC-007-b
  - AC-008-a
  - AC-008-b
  - AC-009-a
  - AC-009-b
  - AC-010-a
---
# 03-plan — T-0004: Execution & Test Plan

## 1. Architectural Decomposition
- **`src/evaluate.py`**:
  - `compute_bootstrap_ci(data, n_bootstraps, confidence_level, seed)`: Generates non-parametric 95% bootstrap confidence intervals for empirical rewards.
  - `evaluate_policy(policy_source, episodes, seed_offset, max_steps, env_name)`: Runs purely greedy rollouts across deterministic unseen test seeds (10000..10999), computing reward, steps, success rate, and illegal moves.
  - `generate_comparison_table(results_dir, out_csv_path)`: Evaluates Untrained Agent, extracts Random Baseline, Evaluates Baseline Model, and Evaluates Tuned Model; produces `results/comparison_table.csv`.
  - `run_evaluation_pipeline(...)`: CLI/script orchestrator generating `results/test_metrics.json` and `results/comparison_table.csv`.
- **`src/plots.py`**:
  - Modular plotting functions for V1 through V8 using Matplotlib/Seaborn with consistent styling, DPI $\ge 150$, tight bounding boxes.
  - V9 animated GIF generation using Matplotlib frame rendering and ImageIO (or Pillow fallback).
  - `generate_all_figures(...)`: Batch runner populating `results/figures/`.
- **`notebooks/taxi_q_learning.ipynb`**:
  - Clean Jupyter notebook in `notebooks/` detailing project flow with markdown cells, embedded figures, and reproducible code blocks.
- **`report/REPORT.md`**:
  - Technical report adhering to rigorous academic standards with mathematical specifications, empirical benchmark tables, convergence charts, and error analyses.

## 2. Test Strategy (`tests/test_evaluate.py`, `tests/test_plots.py`)
- Unit test bootstrap CI calculation on synthetic distribution with known mean.
- Unit test `evaluate_policy` on short test episodes verifying metric schema and non-empty result arrays.
- Unit test `generate_comparison_table` verifying 4 rows (Random, Untrained, Baseline, Tuned) and proper column headers.
- Unit test plotting routines verifying files exist on disk with valid image headers.
- Total unit test runtime target: $< 8.0$ seconds.

## 3. Work Allocation
- `test-author`: Write `tests/test_evaluate.py` and `tests/test_plots.py`, execute red test run, write `04-tests.md`.
- `builder`: Implement `src/evaluate.py` and `src/plots.py`, run benchmarks, generate all figures, achieve green tests, write `05-build.md`.
- `reviewer` & `security`: Perform code review and static analysis in `06-review.md` and `06b-security.md`.
- `qa`: Ratify AC-007 through AC-010 in `07-qa.md`.
- `scribe`: Write `notebooks/taxi_q_learning.ipynb`, `report/REPORT.md`, update `README.md` and `CHANGELOG.md`, write `08-docs.md`.
- `librarian`: Update `docs/memory/lessons.md` and finalize handoff.
