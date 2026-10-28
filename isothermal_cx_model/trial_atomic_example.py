"""One (T,j_t) atomic IVP with ionization, NOT a stationary root search."""

from pathlib import Path
import argparse
import numpy as np

from simple_cx_model.config import load_cx_config
from .cx import make_D
from .excitation import make_E_exc, validate_excitation_scale
from .geometry import add_geometry_arguments, geometry_from_args
from .inlet import add_inlet_arguments, make_inlet_momentum
from .ionization import make_ionization_coefficients
from .recombination import make_rr_coefficients
from .solver_atomic import solve_trial, balance_checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=(
        Path(__file__).resolve().parents[1]/'simple_cx_model/cx_config.json'))
    parser.add_argument('--temperature', type=float, default=3.272, help='Electron/ion temperature, eV.')
    parser.add_argument('--j-target', type=float, default=5.)
    parser.add_argument('--density', type=float, help='Prescribed constant n0, cm^-3; default: JSON.')
    parser.add_argument('--lambda-sh', type=float, help='Default: JSON.')
    parser.add_argument('--ionization-scale', type=validate_excitation_scale, default=1.)
    parser.add_argument('--rr-scale', type=validate_excitation_scale, default=1.)
    parser.add_argument('--excitation-scale', type=validate_excitation_scale, default=1.)
    parser.add_argument('--cx-mode', choices=['constant', 'temperature'], default='temperature')
    parser.add_argument('--points', type=int, default=4000)
    add_geometry_arguments(parser)
    add_inlet_arguments(parser)
    args = parser.parse_args()
    try:
        if (not np.isfinite([args.temperature, args.j_target]).all()
                or min(args.temperature, args.j_target) <= 0 or args.points < 100):
            raise ValueError('Need positive finite T,j_target and points>=100.')
        p = load_cx_config(args.config)
        for key, value, zero_ok in [('n0_cm3', args.density, True),
                                    ('lambda_sh', args.lambda_sh, False)]:
            if value is not None:
                if not np.isfinite(value) or value < 0 or (value == 0 and not zero_ok):
                    raise ValueError(f'Invalid {key}.')
                p[key] = value
        geometry = geometry_from_args(args)
        state = solve_trial(args.temperature/p['T_star_eV'], args.j_target, p['beta'],
            make_D(p, args.cx_mode), p['lambda_sh'], geometry, n_points=args.points,
            E_exc=make_E_exc(p, scale=args.excitation_scale),
            rr=make_rr_coefficients(p, scale=args.rr_scale),
            ionization=make_ionization_coefficients(p, scale=args.ionization_scale),
            inlet_momentum=make_inlet_momentum(p, geometry, mode=args.inlet_momentum))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print('TRIAL ONLY: this command does not solve the full boundary-value problem.')
    print(f'T={args.temperature:g} eV, jt={args.j_target:g}, n0={p["n0_cm3"]:.6g} cm^-3, '
          f'lambda={p["lambda_sh"]:g}, ionization_scale={args.ionization_scale:g}')
    print('status:', state['status'])
    if 'reason' in state:
        print('reason:', state['reason'])
    for key in ('D', 'E_exc', 'R_rr', 'Q_rr', 'D_ion', 'E_ion'):
        if key in state:
            print(f'{key}={state[key]:.9g}')
    if 'j' in state:
        print(f'a={state["alpha"]*p["L_cm"]:.9g} cm; j_a={state["j_a"]:.9g}')
        print(f'R_J={state["R_J"]:.6e}, R_M={state["R_M"]:.6e}, R_E={state["R_E"]:.6e}')
        for key, value in balance_checks(state, p['beta']).items():
            print(f'{key}={value:.6e}')
    print('A stationary solution additionally needs BOTH R_J=0 and R_M=0.')
    return 0 if state['status'] == 'energy_matched' else 1


if __name__ == '__main__':
    raise SystemExit(main())
