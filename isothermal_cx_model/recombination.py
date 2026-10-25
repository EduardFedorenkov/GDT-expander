"""Case-A radiative recombination: first-step coefficients, not a solver.

Hui & Gnedin (1997), Appendix, fits to Ferland et al. (1992):
https://arxiv.org/pdf/astro-ph/9612232

Hydrogen low-density coefficients used approximately for deuterium.
Maxwellian electrons; no three-body recombination, collisional reionization
of excited atoms, radiation transport, molecules or bremsstrahlung.
No ionization-equilibrium assumption is imposed on the plasma densities.
"""

import numpy as np


EV_TO_K = 11604.518121550082
EV_TO_ERG = 1.602176634e-12
IONIZATION_ENERGY_EV = 13.6  # Approximation, not an exact isotope-specific value.
T_MIN_EV = 3.0 / EV_TO_K
T_MAX_EV = 1e9 / EV_TO_K


class RRTemperatureDomainError(ValueError):
    """A positive trial temperature is outside the published RR-fit domain."""


def rr_coefficients(T_eV):
    """Return (alpha_RR [cm^3/s], C_RR [erg cm^3/s]).

    C_RR removes only the kinetic energy of the captured electrons. It
    excludes atomic binding energy and the energy carried by new neutrals.
    Scalars return floats; arrays retain their input shape. The published
    fit domain is 3 <= T_K <= 1e9. Outside it, raise rather than impose an
    undocumented temperature floor or silently extrapolate.
    """
    temperature = np.asarray(T_eV, dtype=float)
    if (np.any(~np.isfinite(temperature))
            or np.any(temperature < T_MIN_EV)
            or np.any(temperature > T_MAX_EV)):
        raise RRTemperatureDomainError(
            "Case-A RR fits require 3 <= T_K <= 1e9 "
            f"({T_MIN_EV:.6g} <= T_eV <= {T_MAX_EV:.6g})."
        )
    temperature_K = EV_TO_K * temperature
    zeta = 315614.0 / temperature_K
    alpha = (1.269e-13 * zeta**1.503
             / (1.0 + (zeta / 0.522)**0.470)**1.923)
    cooling = (1.778e-29 * temperature_K * zeta**1.965
               / (1.0 + (zeta / 0.541)**0.502)**2.697)
    if temperature.ndim == 0:
        return float(alpha), float(cooling)
    return alpha, cooling


def alpha_rr(T_eV):
    """Particle-loss rate coefficient [cm^3/s]: S_RR = n_e*n_i*alpha_rr."""
    return rr_coefficients(T_eV)[0]


def C_rr(T_eV):
    """Electron kinetic-energy cooling [erg cm^3/s], NOT total photon power."""
    return rr_coefficients(T_eV)[1]


def mean_electron_energy_rr_eV(T_eV):
    """Mean kinetic energy of captured electrons [eV], not 3*T_e/2."""
    alpha, cooling = rr_coefficients(T_eV)
    return cooling / (alpha * EV_TO_ERG)


def L_gamma_rr(T_eV):
    """Photon energy coefficient [erg cm^3/s] after decay to ground state.

    For diagnostics only: L_gamma = I_D*alpha_RR + C_RR. Do NOT subtract
    n_e*n_i*L_gamma directly from W, which excludes ionization energy.
    """
    alpha, cooling = rr_coefficients(T_eV)
    return IONIZATION_ENERGY_EV * EV_TO_ERG * alpha + cooling


def make_rr_coefficients(parameters, scale=0.0):
    """Return (0,0) or a callable theta -> (R_RR, Q_RR).

    R=L*n_star*alpha_RR/u_star; Q=L*n_star*C_RR/(u_star*T_star_erg).
    One common eta multiplies both, with 0 <= eta <= 1. No neutral density
    appears here. The source scales are copied; later dictionary edits do
    not silently change the coefficient function. The off switch does not
    evaluate the RR fits, preserving the old model's temperature domain.
    """
    scale = float(scale)
    if not np.isfinite(scale) or not 0 <= scale <= 1:
        raise ValueError("RR scale must be finite and between 0 and 1.")
    if scale == 0:
        return (0.0, 0.0)
    keys = ('L_cm', 'n_star_cm3', 'U_star_cm_per_s',
            'T_star_erg', 'T_star_eV')
    length, density, speed, energy, temperature = (
        float(parameters[key]) for key in keys)
    if (not np.isfinite([length, density, speed, energy, temperature]).all()
            or min(length, density, speed, energy, temperature) <= 0):
        raise ValueError("RR scales must be finite and positive; use CGS.")
    prefactor = scale * length * density / speed

    def coefficients(theta):
        theta = float(theta)
        if not np.isfinite(theta) or theta <= 0:
            raise ValueError("theta must be finite and positive.")
        alpha, cooling = rr_coefficients(theta * temperature)
        return prefactor * alpha, prefactor * cooling / energy

    return coefficients
