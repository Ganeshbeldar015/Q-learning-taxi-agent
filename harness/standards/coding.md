# Coding Standards

1. **Python Compatibility**:
   - Python 3.10+ compatible (tested under Python 3.14.7).
   - Strict typing with type annotations on all public functions, classes, and methods.

2. **Numerics & Reinforcement Learning**:
   - NumPy for all vector and matrix calculations.
   - Seed all stochastic operations using `np.random.Generator`.
   - Vectorized operations preferred over Python loops for state-action lookups.
   - Separate truncated handling (bootstrap) from terminated handling (zero bootstrap).

3. **Style & Quality**:
   - Follow PEP 8 guidelines enforced via `ruff check` and `ruff format`.
   - Docstrings for all classes and functions describing parameters, return values, and mathematical logic.
