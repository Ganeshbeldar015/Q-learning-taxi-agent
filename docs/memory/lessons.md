# Lessons Learned

- **[L-001] DeprecatedEnv on Gymnasium 1.0+**: Calling `gym.make("Taxi-v3")` in modern Gymnasium raises `DeprecatedEnv`. Solution: Use an adapter in `src/env_utils.py` that defaults to `Taxi-v4` while honoring the exact Taxi-v3 specification.
