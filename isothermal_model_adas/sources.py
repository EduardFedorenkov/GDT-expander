"""Local integrated-tube ADAS sources per g=z/L, not per xi.

R>=0; Q_eff is signed. PRB already includes bremsstrahlung. CX is a
separate momentum/energy sink with zero particle source. Geometry is
added by the transport layer. All sources here are per g, not per xi.
"""

import numpy as np


def coefficients_at(provider, theta, N):
    """One interface: (theta,N)->(R,Q_eff); a constant pair is for tests/off."""
    pair = provider(theta, N) if callable(provider) else provider
    if len(pair) != 2:
        raise ValueError('Need a pair (R,Q_eff).')
    R, Q = np.broadcast_arrays(np.asarray(pair[0], float), np.asarray(pair[1], float),
                               np.asarray(N, float))[:2]
    if not np.isfinite(R).all() or not np.isfinite(Q).all() or np.any(R < 0):
        raise ValueError('R must be finite and nonnegative; Q_eff finite, signed.')
    if np.ndim(N) == 0:
        if R.ndim != 0:
            raise ValueError('Scalar state needs scalar coefficients.')
        return float(R), float(Q)
    if R.shape != np.shape(N):
        raise ValueError('Coefficient arrays must match the density profile.')
    return R, Q


def adas_sources(j, N, v, area, *, theta, beta, R, Q_eff, D=0.):
    """Return (j_g, reaction part of mu_g, omega_g)."""
    phi = area*N*N
    removed = R*phi
    ion_energy = v*v/(2*beta) + 1.5*theta
    momentum = -R*j*N
    energy = -(removed*ion_energy + Q_eff*phi)
    if D != 0:
        # Same cold-stationary-neutral moment closure as the old CX model.
        momentum = -D*j + momentum
        energy = -D*area*N*ion_energy + energy
    return (-removed, momentum, energy)
