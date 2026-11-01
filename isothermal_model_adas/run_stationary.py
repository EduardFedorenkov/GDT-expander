"""Explicit local solve: --guess T_eV j_target alpha. Never runs on import."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .config import load_config
from .cx_adas import ADASCX, make_D
from .geometry import add_geometry_arguments, geometry_from_args
from .inlet import make_inlet_momentum
from .ionization_adas import ADASIonization, make_ionization_coefficients
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
    parser.add_argument('--ionization-scale', type=float, default=1., help='0: off; 1: full local SCD and I*SCD cost.')
    parser.add_argument('--cx-mode', choices=('temperature', 'constant'), default='temperature',
                        help='temperature: CCD96_D(theta); constant: explicit k_cx from config.')
    parser.add_argument('--density', type=float, help='Override neutral n0 [cm^-3]; 0 disables CX and ionization.')
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
        tables = ADASRecombination() if args.scale != 0 else None
        coefficients = make_adas_coefficients(p, args.scale, data=tables)
        cx_tables = ADASCX() if p['n0_cm3'] > 0 and args.cx_mode == 'temperature' else None
        D = make_D(p, args.cx_mode, data=cx_tables)
        ion_tables = ADASIonization() if p['n0_cm3'] > 0 and args.ionization_scale > 0 else None
        ionization = make_ionization_coefficients(p, args.ionization_scale, data=ion_tables)
        # Use nominal printed endpoints; the reader accounts for log-grid rounding.
        # All distributed active ADAS datasets cover nominal .2..1e4 eV.
        bounds = (.2/p['T_star_eV'], 1e4/p['T_star_eV']) if any(
            item is not None for item in (tables, cx_tables, ion_tables)) else None
        T, jt, alpha = args.guess
        outcome = solve_stationary(T/p['T_star_eV'], jt, alpha, p['beta'],
            p['lambda_sh'], coefficients, geometry, D=D, ionization=ionization,
            inlet_momentum=make_inlet_momentum(p, geometry, mode),
            theta_bounds=bounds, j_bounds=j_bounds, n_points=args.points)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    data_paths = [] if tables is None else [tables.acd_path, tables.prb_path]
    if cx_tables is not None:
        data_paths.append(cx_tables.path)
    if ion_tables is not None:
        data_paths.append(ion_tables.path)
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in data_paths}
    report = dict(schema_version=3, model='isothermal_adas', status=outcome['status'],
        parameters=p, geometry=geometry.to_dict(), inlet_momentum=mode,
        scale=args.scale, ionization_scale=args.ionization_scale, cx_mode=args.cx_mode,
        j_min=args.j_min, j_max=args.j_max, data_sha256=hashes, guess_T_j_alpha=args.guess,
        diagnostic=outcome['diagnostic'], reason=outcome.get('reason'),
        physical_scope='ACD/PRB, optional CX and local effective SCD ionization; constant Ti=Te; prescribed cold stationary '
                       'neutral moment closure with k_model(T)=CCD(T) approximation in temperature mode; '
                       'SCD energy source is I*SCD only, no PLT excitation radiation; '
                       'no molecules, impurities or neutral dynamics; PRB includes bremsstrahlung.')
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
