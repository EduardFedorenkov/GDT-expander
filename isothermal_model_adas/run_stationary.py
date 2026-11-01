"""Explicit local solve: --guess T_eV j_target alpha. Never runs on import."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .config import load_config
from .geometry import add_geometry_arguments, geometry_from_args
from .inlet import make_inlet_momentum
from .recombination_adas import ADASRecombination, make_adas_coefficients
from .stationary import solve_stationary
from .profiles import dimensional_profiles, plot_profiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    parser.add_argument('--guess', type=float, nargs=3, required=True,
                        metavar=('T_EV', 'J_TARGET', 'ALPHA'))
    parser.add_argument('--inlet-momentum', choices=('auto', 'kinetic', 'constant'), default='auto')
    parser.add_argument('--scale', type=float, default=1., help='Entire ACD/PRB channel, including bremsstrahlung.')
    parser.add_argument('--points', type=int, default=4000)
    parser.add_argument('--output', type=Path, required=True, help='New directory; never overwrites a previous run.')
    parser.add_argument('--plot', action='store_true', help='Save profiles.png; no interactive window.')
    add_geometry_arguments(parser)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new directory.')
    try:
        p = load_config(args.config)
        geometry = geometry_from_args(args)
        mode = args.inlet_momentum
        if mode == 'auto':
            mode = 'kinetic' if p['source_mode'] == 'central' else 'constant'
        if mode == 'kinetic' and p['source_mode'] != 'central':
            raise ValueError('Kinetic inlet requires central-source inputs.')
        tables = ADASRecombination()
        coefficients = make_adas_coefficients(p, args.scale, data=tables)
        # Use nominal printed endpoints; the reader accounts for log-grid rounding.
        bounds = None if args.scale == 0 else (.2/p['T_star_eV'], 1e4/p['T_star_eV'])
        T, jt, alpha = args.guess
        outcome = solve_stationary(T/p['T_star_eV'], jt, alpha, p['beta'],
            p['lambda_sh'], coefficients, geometry,
            inlet_momentum=make_inlet_momentum(p, geometry, mode),
            theta_bounds=bounds, n_points=args.points)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (tables.acd_path, tables.prb_path)}
    report = dict(schema_version=1, model='isothermal_adas', status=outcome['status'],
        parameters=p, geometry=geometry.to_dict(), inlet_momentum=mode,
        scale=args.scale, data_sha256=hashes, guess_T_j_alpha=args.guess,
        diagnostic=outcome['diagnostic'], reason=outcome.get('reason'),
        physical_scope='ACD/PRB only; common constant Ti=Te; no CX, excitation, '
                       'ionization, molecules, impurities or neutral dynamics; PRB includes bremsstrahlung.')
    state = outcome.get('solution')
    if state is not None:
        report['solution'] = dict(theta=state['theta'], T_eV=state['theta']*p['T_star_eV'],
            alpha=state['alpha'], a_cm=state['alpha']*p['L_cm'], j_target=state['j_target'],
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
