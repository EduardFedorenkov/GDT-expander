"""Local integrated-tube reaction sources per g=z/L, not per xi.

R>=0; Q_eff is signed. PRB already includes bremsstrahlung. CX is a
separate momentum/energy sink with zero particle source. Geometry is
added by the transport layer. Ionization adds particles and a binding-energy
cost, but no conservative momentum for cold stationary atoms. Per g, not xi.
PLT adds excitation radiation ONLY, separately from I*SCD and PRB.
AMJUEL molecular coefficients cover complete branches via H2+; their
shared ion-conversion step must not be added as a separate source.
"""

from functools import lru_cache
from math import isfinite

import numpy as np

from .molecular_amjuel import MAD_FRAGMENT_ENERGY_EV, MolecularRates
from .recombination_adas import IONIZATION_ENERGY_EV


D2_DISSOCIATION_ENERGY_EV = 4.56
MAR_ELECTRON_ENERGY_FACTOR = 0.896


class TemperatureOnly:
    """A cached theta-only coefficient compatible with local source APIs."""

    def __init__(self, evaluate):
        self.at_temperature = lru_cache(maxsize=512)(evaluate)

    def __call__(self, theta, N):
        return self.at_temperature(theta)


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


def molecular_at(provider, theta, N):
    """Local (theta,N)->(D_MAR,D_MAD,D_MAI), or a constant/off triple."""
    value = provider(theta, N) if callable(provider) else provider
    try:
        D_mar, D_mad, D_mai = value
    except (TypeError, ValueError) as error:
        raise ValueError('Need a triple (D_MAR,D_MAD,D_MAI).') from error
    density, *rates = np.broadcast_arrays(np.asarray(N, float), np.asarray(D_mar, float),
                                         np.asarray(D_mad, float), np.asarray(D_mai, float))
    if density.shape != np.shape(N):
        raise ValueError('Molecular coefficients must match the density profile shape.')
    if any(not np.isfinite(rate).all() or np.any(rate < 0) for rate in rates):
        raise ValueError('Molecular coefficients must be finite and nonnegative.')
    return MolecularRates(*(float(rate) for rate in rates)) if density.ndim == 0 else MolecularRates(*rates)


def molecular_sources(j, N, v, area, *, theta, beta, T_star_eV, molecular=(0., 0., 0.)):
    """Return molecular contributions to (j_g, mu_g, omega_g).

    No geometry term and no factor (1-alpha). H2+ is not transported.
    The first neutral takes the mean incoming ion momentum and energy in
    each branch; the MAD breakup fragment carries energy but zero mean
    axial momentum. W excludes binding energy and there is no separate
    molecular radiation term in this closure.
    """
    T_star_eV = float(T_star_eV)
    if not isfinite(T_star_eV) or T_star_eV <= 0:
        raise ValueError('T_star_eV must be finite and positive.')
    D_mar, D_mad, D_mai = molecular_at(molecular, theta, N)
    D_total = D_mar + D_mad + D_mai
    ion_energy = v*v/(2*beta) + 1.5*theta
    mad_cost = (D2_DISSOCIATION_ENERGY_EV + MAD_FRAGMENT_ENERGY_EV)/T_star_eV
    mai_cost = (D2_DISSOCIATION_ENERGY_EV + IONIZATION_ENERGY_EV)/T_star_eV
    volume_factor = area*N
    return (volume_factor*(D_mai-D_mar), -j*D_total,
            -volume_factor*(D_total*ion_energy + D_mar*(MAR_ELECTRON_ENERGY_FACTOR*theta)
                            + D_mad*mad_cost + D_mai*mai_cost))


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
