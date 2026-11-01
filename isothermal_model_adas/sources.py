"""Local integrated-tube ADAS sources per g=z/L, not per xi.

R>=0; Q_eff is signed. PRB already includes bremsstrahlung. CX is a
separate momentum/energy sink with zero particle source. Geometry is
added by the transport layer. Ionization adds particles and a binding-energy
cost, but no conservative momentum for cold stationary atoms. Per g, not xi.
PLT adds excitation radiation ONLY, separately from I*SCD and PRB.
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


def ionization_at(provider, theta, N):
    """Local (theta,N)->(D_ion,E_ion), or a constant/off pair.

    Both coefficients are nonnegative. Unlike signed recombination Q_eff,
    E_ion represents an ionization cost. Pure cooling belongs to another
    channel and is not allowed with identically zero ionization rate.
    """
    D_ion, E_ion = coefficients_at(provider, theta, N)
    if np.any(np.asarray(E_ion) < 0) or np.any((np.asarray(D_ion) == 0) & (np.asarray(E_ion) != 0)):
        raise ValueError('Need D_ion,E_ion>=0 and E_ion=0 wherever D_ion=0.')
    return D_ion, E_ion


def excitation_at(provider, theta, N):
    """Local nonnegative E_exc(theta,N), or a scalar/array constant (0=off)."""
    value = provider(theta, N) if callable(provider) else provider
    E, _ = np.broadcast_arrays(np.asarray(value, float), np.asarray(N, float))
    if not np.isfinite(E).all() or np.any(E < 0):
        raise ValueError('E_exc must be finite and nonnegative.')
    if np.ndim(N) == 0:
        if E.ndim != 0:
            raise ValueError('Scalar state needs a scalar E_exc.')
        return float(E)
    if E.shape != np.shape(N):
        raise ValueError('E_exc must match the density profile shape.')
    return E


def adas_sources(j, N, v, area, *, theta, beta, R, Q_eff, D=0., D_ion=0., E_ion=0., E_exc=0.):
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
    particles = -removed
    if np.any(np.asarray(D_ion) != 0):
        particles = particles + D_ion*area*N
    if np.any(np.asarray(E_ion) != 0):
        energy = energy - E_ion*area*N
    if np.any(np.asarray(E_exc) != 0):
        energy = energy - E_exc*area*N
    return (particles, momentum, energy)
