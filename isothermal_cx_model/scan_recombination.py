"""Automatic 2D (T,j_target) survey, verified refinement and saved figure.

No required --guess. A finite grid plus local correction cannot prove that
all roots have been found. Source JSON and previous outputs are never edited.
"""

from collections import Counter
from pathlib import Path
import argparse
import json
import numpy as np

from simple_cx_model.config import load_cx_config
from .cx import make_D
from .excitation import make_E_exc, validate_excitation_scale
from .geometry import add_geometry_arguments, geometry_from_args
from .inlet import add_inlet_arguments, make_inlet_momentum
from .ionization import make_ionization_coefficients
from .recombination import make_rr_coefficients
from .recombination_map import make_j_grid, scan_residuals, choose_starts, plot_residuals
from .stationary_recombination import find_stationary_solutions
from .profiles_recombination import compute_stationary_profiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=(
        Path(__file__).resolve().parents[1]/'simple_cx_model/cx_config.json'))
    parser.add_argument('--density', type=float)
    parser.add_argument('--lambda-sh', type=float)
    parser.add_argument('--temperature-min', type=float, default=.001)
    parser.add_argument('--temperature-max', type=float, default=100.)
    parser.add_argument('--temperature-count', type=int, default=45)
    parser.add_argument('--j-min', type=float, default=1e-6)
    parser.add_argument('--j-max', type=float,
                        help='Map bound only. Default: 1 without ionization, 10 with it.')
    parser.add_argument('--j-count', type=int, default=48)
    parser.add_argument('--j-grid', choices=['mixed', 'linear', 'log'], default='mixed')
    parser.add_argument('--j-axis', choices=['loss', 'linear', 'log'], default='loss')
    parser.add_argument('--max-starts', type=int, default=16,
                        help='Maximum automatic starts; optional manual guesses are additional.')
    parser.add_argument('--guess', type=float, nargs=2, action='append', default=[],
                        metavar=('T_EV', 'J_TARGET'), help='Optional extra start; not required.')
    parser.add_argument('--map-only', action='store_true', help='Skip root refinement.')
    parser.add_argument('--cx-mode', choices=['constant', 'temperature'], default='temperature')
    parser.add_argument('--rr-scale', type=validate_excitation_scale, default=1.)
    parser.add_argument('--excitation-scale', type=validate_excitation_scale, default=1.)
    parser.add_argument('--ionization-scale', type=validate_excitation_scale, default=1.,
                        help='0: previous physics (default); 1: full atomic ionization.')
    parser.add_argument('--iterations', type=int, default=40)
    parser.add_argument('--points', type=int, default=4000, help='Verified root profile samples.')
    parser.add_argument('--residual-tol', type=float, default=1e-9)
    parser.add_argument('--balance-tol', type=float, default=5e-6)
    parser.add_argument('--output', type=Path, default=Path('rr_2d_scan'))
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--no-paths', action='store_true')
    add_geometry_arguments(parser)
    add_inlet_arguments(parser)
    args = parser.parse_args()
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error('Output must be absent or empty; choose a new --output.')
    try:
        if (not np.isfinite([args.temperature_min, args.temperature_max]).all()
                or not 0 < args.temperature_min < args.temperature_max
                or args.temperature_count < 2 or args.max_starts < 1 or args.points < 100
                or args.iterations < 1
                or not np.isfinite([args.residual_tol, args.balance_tol]).all()
                or min(args.residual_tol, args.balance_tol) <= 0):
            raise ValueError('Check temperature bounds, counts, iterations and positive tolerances.')
        p = load_cx_config(args.config)
        for key, value, zero_ok in [('n0_cm3', args.density, True),
                                    ('lambda_sh', args.lambda_sh, False)]:
            if value is not None:
                if not np.isfinite(value) or value < 0 or (value == 0 and not zero_ok):
                    raise ValueError(f'Invalid {key}.')
                p[key] = value
        p['nu_cx_per_s'] = p['n0_cm3']*p['k_cx_cm3_per_s']
        p['D'] = p['L_cm']*p['nu_cx_per_s']/p['U_star_cm_per_s']
        p.pop('theta_exists', None)
        geometry = geometry_from_args(args)
        D = make_D(p, args.cx_mode)
        E = make_E_exc(p, scale=args.excitation_scale)
        rr = make_rr_coefficients(p, scale=args.rr_scale)
        ionization = make_ionization_coefficients(p, scale=args.ionization_scale)
        inlet = make_inlet_momentum(p, geometry, mode=args.inlet_momentum)
        T = np.geomspace(args.temperature_min, args.temperature_max, args.temperature_count)
        j_max = args.j_max if args.j_max is not None else (10. if args.ionization_scale > 0 else 1.)
        js = make_j_grid(args.j_min, j_max, args.j_count, args.j_grid)
        manual = []
        for temperature, jt in args.guess:
            if (not np.isfinite([temperature, jt]).all()
                    or not T[0] <= temperature <= T[-1]
                    or not args.j_min <= jt <= j_max):
                raise ValueError('Manual starts need T and j_t inside the scan range.')
            manual.append(dict(theta=temperature/p['T_star_eV'], j_target=jt, source='manual'))
        def progress(done, total):
            if done == 1 or done == total or done % 5 == 0:
                print(f'Temperature columns: {done}/{total}', flush=True)
        scan = scan_residuals(T/p['T_star_eV'], js, p['beta'], D, p['lambda_sh'],
            geometry, E_exc=E, rr=rr, ionization=ionization,
            inlet_momentum=inlet, progress=progress)
        starts = choose_starts(scan, max_starts=args.max_starts)+manual
        for index, start in enumerate(starts, start=1):
            start.update(start_id=f'start_{index:03d}', T_eV=start['theta']*p['T_star_eV'])
    except (ValueError, OSError) as error:
        parser.error(str(error))
    # Save the expensive survey BEFORE starting any nonlinear correction.
    args.output.mkdir(parents=True, exist_ok=True)
    arrays = {key: value for key, value in scan.items() if isinstance(value, np.ndarray)}
    np.savez_compressed(args.output/'residuals.npz', T_eV=T,
                        ionization_scale=args.ionization_scale, format_version=2, **arrays)
    controls = dict(max_starts=args.max_starts, max_iterations=args.iterations,
        n_points=args.points, residual_tol=args.residual_tol, balance_tol=args.balance_tol,
        theta_bounds=(T[[0, -1]]/p['T_star_eV']).tolist(), j_min=args.j_min,
        j_max=j_max, refinement_j_max=None, j_grid=args.j_grid,
        rtol=1e-9, atol=1e-11, max_step=.02)
    report = dict(format_version=2, stage='map_complete', source_config=str(args.config.resolve()),
        parameters=p, geometry=geometry.to_dict(), cx_mode=args.cx_mode,
        rr_scale=args.rr_scale, excitation_scale=args.excitation_scale,
        ionization_scale=args.ionization_scale,
        inlet_momentum=args.inlet_momentum, scan_shape=list(scan['valid'].shape),
        controls=controls, summary=scan['summary'], starts=starts,
        solutions=[], diagnostic=None,
        physical_scope='Isothermal atomic CX, excitation, Case-A RR and optional direct ionization; '
                       'prescribed constant neutrals; no three-body recombination or molecules.')
    def save_report():
        (args.output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False),
                                               encoding='utf-8')
    save_report()
    if args.no_show:
        import matplotlib
        matplotlib.use('Agg')
    # An image remains available even if refinement is interrupted or fails.
    title = (f"n0={p['n0_cm3']:.3g} cm^-3, lambda={p['lambda_sh']:g}; "
             f"RR={args.rr_scale:g}, excitation={args.excitation_scale:g}, "
             f"ionization={args.ionization_scale:g}")
    fig, _ = plot_residuals(scan, p['T_star_eV'], starts=starts, title=title, j_axis=args.j_axis)
    fig.savefig(args.output/'residuals.png', dpi=170)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"Joint valid trials: {scan['summary']['n_valid']}/{scan['summary']['n_trials']}; "
          f"candidate cells: {scan['summary']['candidate_cell_count']}; starts: {len(starts)}", flush=True)
    states, diagnostic = [], dict(attempts=[], n_solutions=0)
    if not args.map_only and starts:
        print('Refining automatic/manual starts with the checked stationary solver...', flush=True)
        states, diagnostic = find_stationary_solutions(
            [(s['theta'], s['j_target']) for s in starts], p['beta'], D, p['lambda_sh'],
            geometry, E_exc=E, rr=rr, ionization=ionization, inlet_momentum=inlet,
            theta_bounds=controls['theta_bounds'], j_min=args.j_min,
            max_iterations=args.iterations, n_points=args.points,
            residual_tol=args.residual_tol, balance_tol=args.balance_tol)
    for start, attempt in zip(starts, diagnostic['attempts']):
        attempt['start_id'] = start['start_id']
        attempt['converged_to'] = None
        if attempt['status'] == 'converged':
            last = attempt['last_parameters']
            for state in states:
                difference = np.log([last['theta']/state['theta'],
                                     last['j_target']/state['j_target']])
                if np.max(np.abs(difference)) <= 1e-8:
                    attempt['converged_to'] = state['solution_id']
                    break
    for state in states:
        profile = compute_stationary_profiles(state, p)
        report['solutions'].append(dict(solution_id=state['solution_id'],
            theta=state['theta'], T_eV=profile['T_eV'], j_target=state['j_target'],
            alpha=state['alpha'], a_cm=profile['a_cm'],
            outside_scan_j_range=not args.j_min <= state['j_target'] <= j_max,
            net_particle_loss_fraction=1-state['j_target'],
            recombination_count_per_inlet=state['balances']['particle_recombination_integral'],
            ionization_count_per_inlet=state['balances']['particle_ionization_integral'],
            R_J=state['R_J'], R_M=state['R_M'], R_E=state['R_E'],
            mu_target=state['mu_target'],
            coefficients={k: state[k] for k in ('D', 'E_exc', 'R_rr', 'Q_rr', 'D_ion', 'E_ion')},
            verification=state['verification'], balances=state['balances']))
        np.savez_compressed(args.output/f"{state['solution_id']}_profiles.npz",
            **{k: v for k, v in profile.items() if isinstance(v, np.ndarray)},
            theta=state['theta'], alpha=state['alpha'], j_target=state['j_target'],
            ionization_scale=args.ionization_scale, D_ion=state['D_ion'], E_ion=state['E_ion'])
        print(f"{state['solution_id']}: T={profile['T_eV']:.9g} eV, a={profile['a_cm']:.9g} cm, "
              f"jt={state['j_target']:.12g}, R_J={state['R_J']:.2e}, R_M={state['R_M']:.2e}", flush=True)
    report.update(stage='map_only' if args.map_only else 'refinement_complete', diagnostic=diagnostic)
    save_report()
    fig, _ = plot_residuals(scan, p['T_star_eV'], solutions=states, starts=starts,
        attempts=diagnostic['attempts'], title=title, j_axis=args.j_axis, show_paths=not args.no_paths)
    fig.savefig(args.output/'residuals.png', dpi=170)
    if args.no_show:
        plt.close(fig)
    else:
        plt.show()
    print('Attempt statuses:', dict(Counter(a['status'] for a in diagnostic['attempts'])))
    print('Finite grid + local corrections: NOT a proof of completeness or nonexistence.')
    print('Output:', args.output.resolve())
    return 0 if args.map_only or states else 1


if __name__ == '__main__':
    raise SystemExit(main())
