"""CGS source scales and prescribed-neutral parameters for CX/ionization."""

import json
from pathlib import Path
import numpy as np
from .constants import EV_TO_ERG, M_D_G
from .boundary_fluxes import throat_fluxes


def load_config(path):
    """Accept either explicit J0,M0,W0 OR the old central-source closure.

    Explicit flows are integrated tube flows, not per-area flux densities.
    The central-source approximation is n_m=n_c/2, T_m=T_c. The inherited
    chi/chi_E closures are optional physics assumptions, not ADAS data.
    """
    config = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(config, dict):
        raise ValueError('Config must be a JSON object.')
    flows = ('J0_per_s', 'M0_dyn', 'W0_erg_per_s')
    direct = any(key in config for key in flows)
    source_keys = ('n_c_cm3', 'T_c_eV', 'chi_E')
    if direct and any(key in config for key in source_keys):
        raise ValueError('Choose explicit flows OR central-source inputs, not both.')
    keys = ('A_m_cm2', 'L_cm', 'lambda_sh') + (flows if direct else source_keys)
    p = {}
    for key in keys:
        try:
            p[key] = float(config[key])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f'Missing or invalid config field: {key}') from error
    p['m_D_g'] = float(config.get('m_D_g', M_D_G))
    if not np.isfinite(list(p.values())).all() or min(p.values()) <= 0:
        raise ValueError('Source inputs must be finite and positive.')
    p['source_mode'] = 'flows' if direct else 'central'
    if not direct:
        p['chi'] = float(config.get('chi', 5.))
        if not np.isfinite(p['chi']) or p['chi'] < 0:
            raise ValueError('chi must be finite and nonnegative.')
        p['n_m_cm3'] = .5*p['n_c_cm3']
        p.update(throat_fluxes(p['A_m_cm2'], p['n_m_cm3'], p['T_c_eV'],
                              p['chi_E'], m_D_g=p['m_D_g']))
    J, M, W, m, A = (p[k] for k in (*flows, 'm_D_g', 'A_m_cm2'))
    u = M/(m*J)
    T = W/J
    p.update(A_star_cm2=A, A0_cm2=A, U_star_cm_per_s=u,
             T_star_erg=T, T_star_eV=T/EV_TO_ERG,
             n_star_cm3=J/(A*u), beta=T/(m*u*u))
    scales = [p[k] for k in ('U_star_cm_per_s', 'T_star_erg', 'T_star_eV',
                             'n_star_cm3', 'beta')]
    if not np.isfinite(scales).all() or min(scales) <= 0:
        raise ValueError('Derived scales are not representable.')
    # Backward compatibility: no n0 disables both CX and ionization.
    for key in ('n0_cm3', 'k_cx_cm3_per_s'):
        if key == 'k_cx_cm3_per_s' and key not in config:
            continue  # Required only by the explicit constant-CX mode.
        try:
            value = float(config.get(key, 0.))
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f'Invalid optional CX field: {key}') from error
        if not np.isfinite(value) or value < 0:
            raise ValueError(f'{key} must be finite and nonnegative.')
        p[key] = value
    return p
