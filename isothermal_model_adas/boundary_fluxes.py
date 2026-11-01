"""Particle, momentum and energy flows through one mirror throat."""

from math import isfinite, pi, sqrt

from .constants import EV_TO_ERG, M_D_G


def throat_fluxes(A_m, n_m, T_m, chi_E=8.0, *, m_D_g=M_D_G):
    """Calculate total throat flows in CGS, with input temperature in eV.

    A_m is the throat area [cm^2]; n_m is the physical ion/electron density
    at the throat [cm^-3], NOT the normalization of the full Maxwellian.
    T_m is the ion half-Maxwellian temperature parameter [eV], also used
    for the electron temperature. The half-Maxwellian integrates to n_m.

    Closures: p_parallel_e = n_m*T and W0 = chi_E*J0*T. Electron directed
    inertia is neglected. J0 counts ions (equivalently escaping pairs)
    per second through ONE throat; M0 and W0 include BOTH species.
    """
    values = {}
    for name, value in (
        ("A_m", A_m), ("n_m", n_m), ("T_m", T_m),
        ("chi_E", chi_E), ("m_D_g", m_D_g),
    ):
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name} must be a finite positive number.") from error
        if not isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive number.")
        values[name] = value

    temperature = values["T_m"] * EV_TO_ERG
    if not isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature in erg is outside the numerical range.")

    velocity = sqrt(2.0 * temperature / (pi * values["m_D_g"]))
    J0 = values["A_m"] * values["n_m"] * velocity
    M0 = 2.0 * values["A_m"] * values["n_m"] * temperature
    W0 = values["chi_E"] * J0 * temperature

    flows = {"J0_per_s": J0, "M0_dyn": M0, "W0_erg_per_s": W0}
    if not all(isfinite(value) and value > 0 for value in flows.values()):
        raise ValueError("Computed throat flows are outside the numerical range.")
    return flows
