# Lessons Learned

- **[L-001] DeprecatedEnv on Gymnasium 1.0+**: Calling `gym.make("Taxi-v3")` in modern Gymnasium raises `DeprecatedEnv`. Solution: Use an adapter in `src/env_utils.py` that defaults to `Taxi-v4` while honoring the exact Taxi-v3 specification.
- **[L-002] Analytical Invariants**: In Gymnasium Taxi, any successful episode with zero illegal actions satisfies `total_reward == 21 - steps`. Verifying this invariant on clean traces ensures step rewards (-1 per move, +20 on dropoff) are tracked accurately without off-by-one discrepancies.
- **[L-003] Epsilon Tie-Breaking**: When the Q-table is initialized to all zeros, a standard `argmax` always selects action 0 (move south), causing severe spatial exploration bias. Vectorized random tie-breaking among all maximal values ensures uniform exploration across all available actions during early learning.
