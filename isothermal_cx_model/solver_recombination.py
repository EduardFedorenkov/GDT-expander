"""Compatibility import. The ONLY implementation lives in solver_atomic.py.

Kept temporarily so old external imports still work; not an old solver copy.
"""

from .solver_atomic import solve_trial

__all__ = ['solve_trial']
