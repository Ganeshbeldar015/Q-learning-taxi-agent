# AGENTS.md — Taxi Route Optimization with Reinforcement Learning
> Map, not manual. Deep detail lives in linked files.

## What this is
An end-to-end tabular Q-learning system solving the Gymnasium Taxi problem. Detailed requirements and architecture are defined in [docs/harness.md](docs/harness.md).

## Start here (every session)
1. Read this file &rarr; 2. [harness/state/state.json](harness/state/state.json) &rarr; 3. [docs/prompt.md](docs/prompt.md) §1 Boot Sequence.

## Commands
Canonical commands permitted for agent execution (from [docs/harness.md](docs/harness.md)):

| Purpose | Command |
|---|---|
| install | `python -m pip install -r requirements.txt` |
| lint | `python -m ruff check src tests scripts` |
| format | `python -m ruff format --check src tests scripts` |
| typecheck | `python -m mypy src scripts --ignore-missing-imports` |
| unit test | `pytest tests/ -v -m "not slow"` |
| integration test | `pytest tests/ -v` |
| build / smoke | `python -c "import gymnasium, numpy, src.env_utils; print('Smoke OK')"` |
| harness self-test | `python scripts/check_harness.py && pytest tests/harness` |

## Layout
- `docs/`: Master prompt, PRD, harness plan, and project memory.
- `harness/`: State, role specifications, workflows, standards, logs, and artifacts.
- `scripts/`: Linter, path guard, and circuit breaker.
- `src/`: Core implementation modules (env adapter, agent, schedules, train, eval, tune, plots).
- `tests/`: Automated unit and integration test suites.
- `results/`: Empirical outputs, checkpoints, and generated figures.

## Roles
| Agent | Role File | Key Artifact |
|---|---|---|
| Orchestrator | [harness/roles/orchestrator.md](harness/roles/orchestrator.md) | `harness/state/state.json` |
| PO-Spec | [harness/roles/po-spec.md](harness/roles/po-spec.md) | `01-spec.md` |
| Designer | [harness/roles/designer.md](harness/roles/designer.md) | `02-design.md` |
| Tech Lead | [harness/roles/tech-lead.md](harness/roles/tech-lead.md) | `03-plan.md` |
| Test Author | [harness/roles/test-author.md](harness/roles/test-author.md) | `04-tests.md` |
| Builder | [harness/roles/builder.md](harness/roles/builder.md) | `05-build.md` |
| Reviewer | [harness/roles/reviewer.md](harness/roles/reviewer.md) | `06-review.md` |
| Security | [harness/roles/security.md](harness/roles/security.md) | `06b-security.md` |
| QA | [harness/roles/qa.md](harness/roles/qa.md) | `07-qa.md` |
| Scribe | [harness/roles/scribe.md](harness/roles/scribe.md) | `08-docs.md` |
| Librarian | [harness/roles/librarian.md](harness/roles/librarian.md) | `docs/memory/lessons.md` |

## Rules (top 10)
1. **Never improvise outside the harness**: Follow [docs/prompt.md](docs/prompt.md).
2. **Skinny agents, fat roles**: Agent files are pointers $\le 40$ lines; logic is in [harness/roles/](harness/roles/).
3. **Artifacts are the contract**: Every stage produces an artifact in `harness/artifacts/<TASK-ID>/`.
4. **Tests-first discipline**: Test Author writes failing tests before Builder starts ([harness/standards/testing.md](harness/standards/testing.md)).
5. **Deterministic exploration**: Seed all random generators with NumPy Generator ([harness/standards/coding.md](harness/standards/coding.md)).
6. **Bellman correctness**: Bootstrap only on truncation, never on terminal dropoff ([docs/harness.md](docs/harness.md)).
7. **Zero unhandled warnings**: Pass Ruff and Mypy checks ([harness/standards/coding.md](harness/standards/coding.md)).
8. **Stay in lane**: Guard paths enforced by [scripts/guard_paths.py](scripts/guard_paths.py).
9. **Fail loud, fail early**: Circuit breaker halts thrashing ([harness/state/budgets.json](harness/state/budgets.json)).
10. **State lives on disk**: Persist all state in [harness/state/state.json](harness/state/state.json).

## Gates & breakers
See [docs/prompt.md](docs/prompt.md) §9 for gate definitions (G0–G5) and breaker trip rules.

## Memory map
- Decisions & approvals: [harness/logs/decisions.log.md](harness/logs/decisions.log.md)
- Lessons & anti-patterns: [docs/memory/lessons.md](docs/memory/lessons.md)
- Assumptions: [docs/memory/assumptions.md](docs/memory/assumptions.md)
- Glossary: [docs/memory/glossary.md](docs/memory/glossary.md)

## Do-not list
- Do not edit files outside role allow-list.
- Do not commit secrets or unredacted keys.
- Do not weaken or delete failing tests.
- Do not use `push --force` or destructive git commands.
