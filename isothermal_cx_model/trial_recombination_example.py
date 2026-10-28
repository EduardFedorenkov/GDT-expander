"""One editable trial, NOT a search for a stationary boundary-value solution.

Run from project root: python -B -m isothermal_cx_model.trial_recombination_example
All source data, lambda_sh and n0 come from the existing JSON. Change only
the explicitly listed trial controls below for this small first step.
"""

from pathlib import Path

from simple_cx_model.config import load_cx_config
from .cx import make_D
from .excitation import make_E_exc
from .geometry import quadratic_area_legacy
from .inlet import make_inlet_momentum
from .recombination import make_rr_coefficients
from .solver_atomic import solve_trial


T_TRIAL_EV = 10.0
J_TARGET = 0.5
RR_SCALE = 1.0
EXCITATION_SCALE = 1.0
CX_MODE = 'temperature'


def main():
    config = Path(__file__).resolve().parents[1]/'simple_cx_model/cx_config.json'
    p = load_cx_config(config)
    geometry = quadratic_area_legacy(K=500.)
    state = solve_trial(
        T_TRIAL_EV/p['T_star_eV'], J_TARGET, p['beta'],
        make_D(p, mode=CX_MODE), p['lambda_sh'], geometry,
        E_exc=make_E_exc(p, scale=EXCITATION_SCALE),
        rr=make_rr_coefficients(p, scale=RR_SCALE),
        inlet_momentum=make_inlet_momentum(p, geometry),
    )
    print(f'TRIAL ONLY: T={T_TRIAL_EV:g} eV, j_target={J_TARGET:g}')
    print('status:', state['status'])
    print('alpha:', state['alpha'], 'a_cm:', state['alpha']*p['L_cm'])
    print('R_J:', state['R_J'], 'R_M:', state['R_M'], 'R_E:', state['R_E'])
    if 'reason' in state:
        print('reason:', state['reason'])
    print('A global solution requires BOTH R_J=0 and R_M=0.')


if __name__ == '__main__':
    main()
