# Parallel Lanes Isolation Protocol

## Principles
1. **Exclusive File Ownership**: No two parallel lanes may edit the same file path simultaneously.
2. **Locking Mechanism**: Tech Lead records file ownership locks in `harness/state/locks/<lane>.json`.
3. **Merge Gate (G2)**: After lane branches merge, full test suite and coverage gates execute before review.
4. **Lane Limit**: Up to `budgets.json.max_parallel_agents` concurrent builder streams.
