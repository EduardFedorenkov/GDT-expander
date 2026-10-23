"""Collisionless ion momentum plus the approximate electron pressure.

Geometry is the existing MagneticGeometry, with coordinate g = z/L.
The inlet.py adapter supplies the prescribed chi and T_e=T_c closure.
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import erfcx


EV_TO_ERG = 1.602176634e-12


def kinetic_momentum(
    a_cm,
    geometry,
    chi,
    *,
    L_cm,
    A_m_cm2,
    n_c_cm3,
    T_i0_eV,
    T_e_eV=None,
):
    """Return M_kin(a) = A(a) [Pi_parallel_i(a) + n_i(a) T_e], dyn.

    a_cm, L_cm: position and expander length, cm; 0 <= a <= L.
    geometry: existing MagneticGeometry; A_ratio(a/L) = B_m/B(a).
    chi: e * [phi_m - phi(a)] / T_i0 >= 0 (accelerating ions).
    A_m_cm2: flux-tube area at the mirror throat, cm**2.
    n_c_cm3: FULL Maxwellian normalization, cm**-3, NOT neutral density.
        The outgoing half Maxwellian has density n_c/2 at the throat.
    T_i0_eV: source ion temperature, eV.
    T_e_eV: prescribed electron-pressure temperature at a, eV;
        defaults to T_i0_eV. This does not impose ion thermalization.

    Assumes adiabatic ion motion, no reflected ions, and a monotonic
    decreasing field / accelerating potential between the throat and a.
    Electron closure p_parallel_e = n_i T_e is an approximation.
    Endpoint values alone cannot check for intermediate orbit barriers.
    """
    if T_e_eV is None:
        T_e_eV = T_i0_eV

    a, L, area, density, Ti, Te, chi = map(
        float, (a_cm, L_cm, A_m_cm2, n_c_cm3, T_i0_eV, T_e_eV, chi)
    )
    if not np.isfinite([a, L, area, density, Ti, Te, chi]).all():
        raise ValueError("All physical parameters must be finite.")
    if min(L, area, density, Ti) <= 0 or Te < 0 or chi < 0:
        raise ValueError("Require L,A_m,n_c,T_i0 > 0 and T_e,chi >= 0.")
    if not 0 <= a <= L:
        raise ValueError("Require 0 <= a_cm <= L_cm.")

    K = float(geometry.A_ratio(a / L))
    if not np.isfinite(K) or K < 1:
        raise ValueError("This transmitted-ion formula requires K(a) >= 1.")

    c = 1.0 - 1.0 / K
    sqrt_c = np.sqrt(c)
    root_chi = np.sqrt(chi)
    # F = erfcx(sqrt(chi/c)); only sqrt(c)*F enters the result.
    F = 0.0 if c == 0 else float(erfcx(root_chi / sqrt_c))

    # H = K * N_i, where n_i = (n_c/2) * N_i.
    if chi == 0:
        H = 1.0 / (1.0 + sqrt_c)
    elif K <= 1e4:
        H = K * (float(erfcx(root_chi)) - sqrt_c * F)
    else:
        # Equivalent positive integral avoids cancellation at large K.
        H = 2.0 / np.sqrt(np.pi) * quad(
            lambda x: np.exp(-x) * x / (
                np.sqrt(x + chi) + np.sqrt(c * x + chi)
            ),
            0.0, np.inf, epsabs=0.0, epsrel=1e-10,
        )[0]

    # G_i = K*[E(chi)-c**(3/2)*E(chi/c)] + 2*sqrt(chi/pi).
    G_i = H + sqrt_c * F + 2.0 * root_chi / np.sqrt(np.pi)
    M_i0 = 0.5 * area * density * Ti * EV_TO_ERG
    return float(M_i0 * (G_i + (Te / Ti) * H))
