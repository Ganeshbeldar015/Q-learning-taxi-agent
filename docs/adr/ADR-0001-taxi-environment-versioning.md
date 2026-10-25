# ADR-0001: Gymnasium Taxi Environment Versioning Adapter

## Context
In `docs/raw.md`, the environment is specified as `Taxi-v3`. However, in modern Gymnasium (>= 1.0.0), `Taxi-v3` is deprecated and raising `DeprecatedEnv: Environment version v3 for Taxi is deprecated. Please use Taxi-v4 instead.`
`Taxi-v4` has identical state transition tables, 500 discrete states, 6 discrete actions, and identical reward structure (-1 step, +20 dropoff, -10 illegal move).

## Decision
Implement an adapter in `src/env_utils.py` inside `make_env(env_name="Taxi-v4")`.
If requested `Taxi-v3`, `make_env` attempts `gym.make("Taxi-v3")`, and catches `gym.error.DeprecatedEnv` to automatically fallback to `gym.make("Taxi-v4")` while logging an informational note.

## Consequences
- Full backwards compatibility with specifications naming Taxi-v3.
- Unbroken execution under current modern Gymnasium 1.4.0+.
