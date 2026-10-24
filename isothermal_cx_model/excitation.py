"""Optically thin atomic excitation cooling: CGS, temperatures in eV.

Hydrogen fit from Katz, Weinberg & Hernquist, Table 1:
https://arxiv.org/pdf/astro-ph/9509107
Used as an approximate D-atom coefficient, not a collisional-radiative model.
No ionization-equilibrium assumption is imposed on the prescribed densities.
"""

import numpy as np


EV_TO_K = 11604.518121550082


def L_exc(T_eV):
    """Excitation energy-loss coefficient [erg cm^3/s], scalar or array.

    P_exc = n_e * n_D0 * L_exc(T_eV). Includes excitation energy already.
    Assumes Maxwellian electrons, ground-state atoms, low-density radiative
    decay and no return of radiation energy to the plasma. Does not model
    line trapping, collisional quenching, molecules, ionization or recombination.
    """
    temperature = np.asarray(T_eV, dtype=float)
    if np.any(~np.isfinite(temperature)) or np.any(temperature < 0):
        raise ValueError("T_eV must be finite and nonnegative.")
    result = np.zeros_like(temperature)
    positive = temperature > 0
    T_K = temperature[positive] * EV_TO_K
    with np.errstate(over="ignore", under="ignore"):
        result[positive] = (
            7.5e-19 * np.exp(-118348.0 / T_K)
            / (1.0 + np.sqrt(T_K / 1e5))
        )
    return float(result) if result.ndim == 0 else result


def E_exc(theta, parameters, n0_cm3=None):
    """E(theta) = L*n0*L_exc(theta*T_star_eV)/(u_star*T_star_erg).

    No A_star and no extra excitation energy multiplier belong here.
    The neutral-density override must be the same as in make_D.
    """
    theta = float(theta)
    n0 = float(parameters["n0_cm3"] if n0_cm3 is None else n0_cm3)
    length = float(parameters["L_cm"])
    speed = float(parameters["U_star_cm_per_s"])
    energy = float(parameters["T_star_erg"])
    temperature = float(parameters["T_star_eV"])
    if (not np.isfinite([theta, n0, length, speed, energy, temperature]).all()
            or theta < 0 or n0 < 0 or min(length, speed, energy, temperature) <= 0):
        raise ValueError("Need theta,n0 >= 0 and positive finite L,u_*,T_*.")
    value = length * n0 * L_exc(theta * temperature) / (speed * energy)
    if not np.isfinite(value):
        raise ValueError("Non-finite excitation coefficient; check CGS scales.")
    return float(value)


def validate_excitation_scale(scale):
    scale = float(scale)
    if not np.isfinite(scale) or not 0 <= scale <= 1:
        raise ValueError("excitation_scale must be finite and between 0 and 1.")
    return scale


def make_E_exc(parameters, n0_cm3=None, scale=0.0):
    """Return zero (old model) or eta*E(theta), with 0 <= eta <= 1."""
    scale = validate_excitation_scale(scale)
    if scale == 0:
        return 0.0
    E_exc(0.0, parameters, n0_cm3=n0_cm3)  # Validate before a long scan.
    return lambda theta: scale * E_exc(theta, parameters, n0_cm3=n0_cm3)


def add_excitation_arguments(parser):
    parser.add_argument(
        "--excitation-scale", type=validate_excitation_scale, default=0.0,
        help="eta in [0,1]: 0 = CX only (default), 1 = excitation cooling on.",
    )
