"""Diagnostics for computed profiles, independent of the transport solver."""

import numpy as np


def classify_profile(mu):
    """Diagnostic only: never use this label as a unique solution ID."""
    mu = np.asarray(mu, dtype=float)
    differences = np.diff(mu)
    tolerance = 1e-8 * max(1.0, float(np.max(np.abs(mu))))
    signs = np.sign(differences[np.abs(differences) > tolerance])
    changes = int(np.count_nonzero(np.diff(signs))) if signs.size else 0
    if changes == 0:
        return "monotone"
    if changes == 1 and signs[0] > 0 and signs[-1] < 0:
        return "with_maximum"
    return "nonmonotone"
