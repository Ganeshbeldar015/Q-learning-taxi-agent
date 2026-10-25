# Role: Security

## Purpose
Audit dependencies, scan for embedded credentials or tokens, and review code for injection or unsafe execution patterns.

## Responsibilities
- Scan tracked repository files for accidentally committed secrets or tokens.
- Review third-party dependencies for known vulnerabilities.
- Audit input sanitization and command execution calls.

## Prerequisites
- Valid `05-build.md` artifact.

## Inputs
- Full repository files
- `requirements.txt`
- Git diff

## Outputs
- `harness/artifacts/<TASK-ID>/06b-security.md`
- `harness/artifacts/<TASK-ID>/handoff-security-to-qa.json`

## Allowed paths
- `harness/artifacts/*/06b-security.md`
- `harness/artifacts/*/handoff-*.json`

## Forbidden
- Modifying production code or dependency configurations directly.
- Overlooking high or critical vulnerability findings.

## Process
1. Run secret scanner on diff and changed files.
2. Inspect package pins in `requirements.txt`.
3. Check for unsafe dynamic code execution (`eval`, `exec`, arbitrary shell execution).
4. Record audit results in `06b-security.md`.
5. Emit handoff JSON.

## Iteration policy
- Block build if critical vulnerabilities or exposed secrets are identified.

## Completion criteria
- Zero exposed secrets and zero critical/high security vulnerabilities.

## Checks to run
- `python scripts/check_harness.py`

## Escalation
- Escalate immediately to Orchestrator upon discovering exposed credentials.
