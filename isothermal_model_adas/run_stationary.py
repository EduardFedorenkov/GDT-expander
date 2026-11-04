"""Explicit local solve: --guess T_eV j_target alpha. Never runs on import."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .config import load_config
from .atomic_coefficients import add_atomic_model_argument, build_atomic_coefficients, code_hashes
from .exitation_adas import validate_excitation_scale
from .geometry import add_geometry_arguments, geometry_from_args
from .inlet import make_inlet_momentum
from .stationary import solve_stationary
from .profiles import dimensional_profiles, plot_profiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    parser.add_argument('--guess', type=float, nargs=3, required=True,
                        metavar=('T_EV', 'J_TARGET', 'ALPHA'))
    parser.add_argument('--inlet-momentum', choices=('auto', 'kinetic', 'constant'), default='auto')
    add_atomic_model_argument(parser)
    parser.add_argument('--scale', type=float, default=1., help='Entire ACD/PRB channel, including bremsstrahlung.')
    parser.add_argument('--ionization-scale', type=float, default=1., help='0: off; 1: full ionization and binding-energy cost.')
    parser.add_argument('--excitation-scale', type=validate_excitation_scale, default=1., help='0: off; 1: full line cooling.')
    parser.add_argument('--density', type=float, help='Override neutral n0 [cm^-3]; 0 disables CX, ionization and excitation.')
    parser.add_argument('--j-min', type=float, default=1e-12, help='Positive numerical search lower bound.')
    parser.add_argument('--j-max', type=float, help='Optional numerical search upper bound; values >1 allowed.')
    parser.add_argument('--points', type=int, default=4000)
    parser.add_argument('--output', type=Path, required=True, help='New directory; never overwrites a previous run.')
    parser.add_argument('--plot', action='store_true', help='Save profiles.png; no interactive window.')
    add_geometry_arguments(parser)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new directory.')
    try:
        if not np.isfinite(args.ionization_scale) or not 0 <= args.ionization_scale <= 1:
            raise ValueError('ionization-scale must be finite and between 0 and 1.')
        j_bounds = (args.j_min, np.finfo(float).max if args.j_max is None else args.j_max)
        if not np.isfinite(j_bounds).all() or not 0 < j_bounds[0] < j_bounds[1]:
            raise ValueError('Need ordered positive finite j search bounds.')
        p = load_config(args.config)
        if args.density is not None:
            if not np.isfinite(args.density) or args.density < 0:
                raise ValueError('Neutral density must be finite and nonnegative.')
            p['n0_cm3'] = args.density
        geometry = geometry_from_args(args)
        mode = args.inlet_momentum
        if mode == 'auto':
            mode = 'kinetic' if p['source_mode'] == 'central' else 'constant'
        if mode == 'kinetic' and p['source_mode'] != 'central':
            raise ValueError('Kinetic inlet requires central-source inputs.')
        coefficients, processes, active, data_paths = build_atomic_coefficients(
            p, args.atomic_model, args.scale, args.ionization_scale, args.excitation_scale)
        # Use nominal printed endpoints; the reader accounts for log-grid rounding.
        # All distributed active ADAS datasets cover nominal .2..1e4 eV.
        bounds = (.2/p['T_star_eV'], 1e4/p['T_star_eV']) if active else None
        T, jt, alpha = args.guess
        outcome = solve_stationary(T/p['T_star_eV'], jt, alpha, p['beta'],
            p['lambda_sh'], coefficients, geometry, **processes,
            inlet_momentum=make_inlet_momentum(p, geometry, mode),
            theta_bounds=bounds, j_bounds=j_bounds, n_points=args.points)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in data_paths}
    report = dict(schema_version=5, model='isothermal_adas', status=outcome['status'],
        parameters=p, geometry=geometry.to_dict(), inlet_momentum=mode,
        atomic_model=args.atomic_model, code_sha256=code_hashes(args.atomic_model),
        scale=args.scale, ionization_scale=args.ionization_scale,
        excitation_scale=args.excitation_scale,
        j_min=args.j_min, j_max=args.j_max, data_sha256=hashes, guess_T_j_alpha=args.guess,
        diagnostic=outcome['diagnostic'], reason=outcome.get('reason'),
        physical_scope=('Local ACD/PRB; legacy temperature-only CX, ionization and excitation. '
                       if args.atomic_model == 'hybrid' else
                       'Local ACD/PRB, CCD(T) CX, local SCD ionization and PLT excitation. ') +
                       'Constant Ti=Te; prescribed cold stationary neutrals. '
                       'Ionization binding-energy cost and excitation photon power are distinct channels; '
                       'no radiation transport/reabsorption, molecules, impurities or neutral dynamics; PRB includes bremsstrahlung.')
    state = outcome.get('solution')
    if state is not None:
        report['solution'] = dict(theta=state['theta'], T_eV=state['theta']*p['T_star_eV'],
            alpha=state['alpha'], a_cm=state['alpha']*p['L_cm'], j_target=state['j_target'], D=state['D'],
            net_particle_change=state['j_target']-1,
            residuals={k: state[k] for k in ('R_J', 'R_M', 'R_E')},
            balances=state['balances'], verification=state['verification'])
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    if state is not None:
        profile = dimensional_profiles(state, p)
        np.savez_compressed(args.output/'profiles.npz', **profile)
        if args.plot:
            import matplotlib
            matplotlib.use('Agg')
            figure, _ = plot_profiles(profile)
            figure.savefig(args.output/'profiles.png', dpi=160)
            import matplotlib.pyplot as plt
            plt.close(figure)
        print(f"T={report['solution']['T_eV']:.9g} eV, a={report['solution']['a_cm']:.9g} cm")
    print(outcome['status'])
    print('Local search only. Failure is not a proof of nonexistence. Output:', args.output.resolve())
    return 0 if state is not None else 1


if __name__ == '__main__':
    raise SystemExit(main())
