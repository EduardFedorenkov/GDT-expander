"""Automatic ADAS (T,j_t,alpha) map, multistart correction and saved profiles.

No required --guess. --dry-run validates setup without IVPs or output files.
Finite search failures never establish physical nonexistence of solutions.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy

from .config import load_config
from .cx_adas import ADASCX, make_D
from .exitation_adas import ADASExcitation, make_E_exc, validate_excitation_scale
from .geometry import add_geometry_arguments, geometry_from_args
from .inlet import make_inlet_momentum
from .ionization_adas import ADASIonization, make_ionization_coefficients
from .profiles import dimensional_profiles, plot_profiles
from .recombination_adas import ADASRecombination, make_adas_coefficients
from .residual_map import scan_residuals, choose_starts, save_residuals, plot_residuals
from .stationary import find_stationary_solutions


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    parser.add_argument('--density', type=float, help='Override neutral n0 [cm^-3].')
    parser.add_argument('--lambda-sh', type=float)
    parser.add_argument('--cx-mode', choices=('constant', 'temperature'), default='temperature')
    parser.add_argument('--rr-scale', '--scale', dest='rr_scale', type=validate_excitation_scale, default=1.,
                        help='Entire ACD/PRB channel, including bremsstrahlung.')
    parser.add_argument('--ionization-scale', type=validate_excitation_scale, default=1.)
    parser.add_argument('--excitation-scale', type=validate_excitation_scale, default=1.)
    parser.add_argument('--inlet-momentum', choices=('auto', 'kinetic', 'constant'), default='auto')
    parser.add_argument('--temperature-min', type=float, default=.2, help='eV, not theta.')
    parser.add_argument('--temperature-max', type=float, default=100.)
    parser.add_argument('--temperature-count', type=int, default=16)
    parser.add_argument('--j-min', type=float, default=1e-6, help='Map lower bound only.')
    parser.add_argument('--j-max', type=float, default=10., help='Map upper bound only; j_t>1 is allowed.')
    parser.add_argument('--j-count', type=int, default=16)
    parser.add_argument('--j-grid', choices=('log', 'linear'), default='log')
    parser.add_argument('--alpha-min', type=float, default=.05)
    parser.add_argument('--alpha-max', type=float, default=.995)
    parser.add_argument('--alpha-count', type=int, default=7)
    parser.add_argument('--max-starts', type=int, default=16)
    parser.add_argument('--separation', type=int, default=2, help='Grid-index separation of automatic starts.')
    parser.add_argument('--guess', type=float, nargs=3, action='append', default=[],
                        metavar=('T_EV', 'J_TARGET', 'ALPHA'), help='Optional extra start; repeatable.')
    parser.add_argument('--iterations', type=int, default=40)
    parser.add_argument('--points', type=int, default=4000, help='Verified profile samples.')
    parser.add_argument('--map-points', type=int, default=100)
    parser.add_argument('--residual-tol', type=float, default=1e-9)
    parser.add_argument('--balance-tol', type=float, default=5e-6)
    parser.add_argument('--dedup-tol', type=float, default=1e-7)
    parser.add_argument('--rtol', type=float, default=1e-9)
    parser.add_argument('--atol', type=float, default=1e-11)
    parser.add_argument('--max-step', type=float, default=.02, help='ODE step bound in xi.')
    parser.add_argument('--map-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--j-axis', choices=('log', 'linear', 'loss'), default='log')
    parser.add_argument('--alpha-slice', type=int, help='Also save this alpha-index slice (zero based).')
    parser.add_argument('--no-show', action='store_true', help='Save plots without an interactive window.')
    parser.add_argument('--no-plot', action='store_true', help='Skip all PNGs and interactive plots.')
    parser.add_argument('--output', type=Path, default=Path('adas_stationary_scan'))
    add_geometry_arguments(parser)
    return parser


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save_json(path, payload):
    """Atomic checkpoint; validate serialisation before touching the file."""
    text = json.dumps(payload, indent=2, allow_nan=False)
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    # The output directory is exclusively owned by this run.
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(text+'\n')
    temporary.replace(path)


def _prepare(args):
    """Load exactly the active tables; never integrate the plasma here."""
    for name, minimum in (('temperature_count', 2), ('j_count', 2), ('alpha_count', 2),
                          ('max_starts', 1), ('separation', 1), ('iterations', 1),
                          ('points', 100), ('map_points', 2)):
        if getattr(args, name) < minimum:
            raise ValueError(f'{name} must be >= {minimum}.')
    controls = [getattr(args, key) for key in ('residual_tol', 'balance_tol', 'dedup_tol',
                                               'rtol', 'atol', 'max_step')]
    if not np.isfinite(controls).all() or min(controls) <= 0:
        raise ValueError('All numerical tolerances/step bounds must be positive finite.')
    for lower, upper, name in ((args.temperature_min, args.temperature_max, 'T'),
                               (args.j_min, args.j_max, 'j_t')):
        if not np.isfinite([lower, upper]).all() or not 0 < lower < upper:
            raise ValueError(f'Need 0 < {name}_min < {name}_max, both finite.')
    p = load_config(args.config)
    for key, value, allow_zero in (('n0_cm3', args.density, True),
                                   ('lambda_sh', args.lambda_sh, False)):
        if value is not None:
            if not np.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
                raise ValueError(f'Invalid override for {key}.')
            p[key] = value
    geometry = geometry_from_args(args)
    if (not np.isfinite([args.alpha_min, args.alpha_max]).all()
            or not geometry.g_min < args.alpha_min < args.alpha_max < 1):
        raise ValueError('Require g_min < alpha_min < alpha_max < 1.')
    if args.alpha_slice is not None and not 0 <= args.alpha_slice < args.alpha_count:
        raise ValueError('alpha-slice is a zero-based index into the alpha grid.')
    mode = args.inlet_momentum
    if mode == 'auto':
        mode = 'kinetic' if p['source_mode'] == 'central' else 'constant'
    if mode == 'kinetic' and p['source_mode'] != 'central':
        raise ValueError('Kinetic inlet requires central-source config.')
    inlet = make_inlet_momentum(p, geometry, mode)
    rr = ADASRecombination() if args.rr_scale else None
    cx = ADASCX() if p['n0_cm3'] > 0 and args.cx_mode == 'temperature' else None
    ion = ADASIonization() if p['n0_cm3'] > 0 and args.ionization_scale else None
    exc = ADASExcitation() if p['n0_cm3'] > 0 and args.excitation_scale else None
    active = [table for table in (rr, cx, ion, exc) if table is not None]
    if active:
        log_lower = max(np.log10(table.T_bounds_eV[0]) for table in active)
        log_upper = min(np.log10(table.T_bounds_eV[1]) for table in active)
        # Match only the readers' endpoint-rounding tolerance; no extrapolation.
        if (np.log10(args.temperature_min) < log_lower-1e-8
                or np.log10(args.temperature_max) > log_upper+1e-8):
            raise ValueError(f'Temperature grid exceeds active ADAS domain '
                             f'[{10.**log_lower:g}, {10.**log_upper:g}] eV.')
    coefficients = make_adas_coefficients(p, args.rr_scale, data=rr)
    processes = dict(D=make_D(p, args.cx_mode, data=cx),
        ionization=make_ionization_coefficients(p, args.ionization_scale, data=ion),
        excitation=make_E_exc(p, scale=args.excitation_scale, data=exc), inlet_momentum=inlet)
    paths = [] if rr is None else [rr.acd_path, rr.prb_path]
    paths.extend(table.path for table in (cx, ion, exc) if table is not None)
    provenance = dict(source_config=str(args.config.resolve()), source_config_sha256=_sha256(args.config),
        data_sha256={path.name: _sha256(path) for path in paths},
        code_sha256={path.name: _sha256(path) for path in sorted(Path(__file__).parent.glob('*.py'))},
        runtime=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__))
    temperatures = np.geomspace(args.temperature_min, args.temperature_max, args.temperature_count)
    j_grid = np.geomspace if args.j_grid == 'log' else np.linspace
    js = j_grid(args.j_min, args.j_max, args.j_count)
    alphas = np.linspace(args.alpha_min, args.alpha_max, args.alpha_count)
    manual = []
    for T, jt, alpha in args.guess:
        if (not np.isfinite([T, jt, alpha]).all()
                or not temperatures[0] <= T <= temperatures[-1]
                or not args.j_min <= jt <= args.j_max
                or not args.alpha_min <= alpha <= args.alpha_max):
            raise ValueError('Manual starts must lie inside all three map ranges.')
        manual.append(dict(theta=T/p['T_star_eV'], j_target=jt, alpha=alpha, source='manual'))
    return p, geometry, coefficients, processes, mode, provenance, temperatures, js, alphas, manual


def _solution_record(state, p, args):
    """Guard exported roots; Q_eff is signed, j_t may exceed one."""
    if not state.get('stationary_verified', False):
        raise RuntimeError('Refusing to export an unverified solution.')
    target = float(state['mu_target'])
    residuals = np.array([state[k] for k in ('R_J', 'R_M', 'R_E')], float)
    balances = state['balances']
    errors = [balances[k] for k in ('particle_abs', 'momentum_abs', 'energy_abs')]
    if (not np.isfinite([target, *residuals, *errors]).all() or target <= 0
            or min(errors) < 0 or max(errors) > args.balance_tol
            or abs(residuals[0]) > args.residual_tol or abs(residuals[2]) > args.residual_tol
            or abs(residuals[1]) > args.residual_tol*(1+abs(target))):
        raise RuntimeError('Solver result failed export residual/balance checks.')
    ranges = {}
    for key in ('R_profile', 'Q_eff_profile', 'D_ion_profile', 'E_ion_profile', 'E_exc_profile'):
        values = np.asarray(state[key])
        if not np.isfinite(values).all():
            raise RuntimeError(f'Nonfinite {key} in verified result.')
        ranges[key] = [float(values.min()), float(values.max())]
    return dict(solution_id=state['solution_id'], stationary_verified=True,
        theta=state['theta'], T_eV=state['theta']*p['T_star_eV'],
        j_target=state['j_target'], alpha=state['alpha'], a_cm=state['alpha']*p['L_cm'],
        mu_target=target, R_J=float(residuals[0]), R_M=float(residuals[1]), R_E=float(residuals[2]),
        outside_scan_j_range=not args.j_min <= state['j_target'] <= args.j_max,
        outside_scan_alpha_range=not args.alpha_min <= state['alpha'] <= args.alpha_max,
        net_particle_change=state['j_target']-1, coefficients=dict(D=state['D'], local_ranges=ranges),
        verification=state['verification'], balances=balances,
        profile_file=f"{state['solution_id']}_profiles.npz")


def run_scan(args):
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise ValueError('Output must be absent or empty; existing results are never overwritten.')
    p, geometry, coefficients, processes, mode, provenance, T, js, alphas, manual = _prepare(args)
    total = len(T)*len(js)*len(alphas)
    print(f'Grid: T={len(T)}, j_t={len(js)}, alpha={len(alphas)}; trial IVPs: {total}', flush=True)
    print(f'Maximum automatic starts: {args.max_starts}; additional manual starts: {len(manual)}', flush=True)
    print('The number of corrector IVPs is additional and is not known in advance.', flush=True)
    if args.dry_run:
        print('Dry run: no plasma integration and no output files.')
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    options = dict(cx_mode=args.cx_mode, rr_scale=args.rr_scale,
                   ionization_scale=args.ionization_scale, excitation_scale=args.excitation_scale,
                   inlet_momentum=mode)
    controls = {key: getattr(args, key) for key in ('iterations', 'points', 'map_points',
        'residual_tol', 'balance_tol', 'dedup_tol', 'rtol', 'atol', 'max_step',
        'max_starts', 'separation', 'j_grid')}
    controls.update(theta_bounds=(T[[0, -1]]/p['T_star_eV']).tolist(),
        refinement_j_bounds=[1e-12, None], refinement_alpha_bounds=[geometry.g_min, 1.])
    report = dict(format_version=1, model='isothermal_adas', kind='stationary_scan_3d',
        stage='mapping', parameters=p, geometry=geometry.to_dict(), model_options=options,
        provenance=provenance, controls=controls,
        grid=dict(T_eV=T.tolist(), j_target=js.tolist(), alpha=alphas.tolist()),
        starts=[], solutions=[], diagnostic=dict(attempts=[], n_solutions=0, duplicate_convergences=0),
        warning='Finite grid and multistart correction are not a completeness/nonexistence proof. '
                'Root IDs are discoveries, not physical branches. PRB includes bremsstrahlung.')
    _save_json(args.output/'source_config.json', json.loads(args.config.read_text(encoding='utf-8-sig')))
    _save_json(args.output/'geometry.json', geometry.to_dict())
    _save_json(args.output/'summary.json', report)
    scan = None

    def save_report():
        _save_json(args.output/'summary.json', report)

    def plot_map(states=(), *, final=False):
        if args.no_plot:
            return
        import matplotlib.pyplot as plt
        title = f"n0={p['n0_cm3']:.3g} cm^-3, lambda={p['lambda_sh']:g}; ADAS"
        fig, _ = plot_residuals(scan, p['T_star_eV'], starts=report['starts'],
                               solutions=states, title=title, j_axis=args.j_axis)
        fig.savefig(args.output/'residuals.png', dpi=150)
        if final and not args.no_show:
            plt.show()
        plt.close(fig)
        if args.alpha_slice is not None:
            fig, _ = plot_residuals(scan, p['T_star_eV'], starts=report['starts'],
                solutions=states, title=title, j_axis=args.j_axis, alpha_index=args.alpha_slice)
            fig.savefig(args.output/f'alpha_slice_{args.alpha_slice:03d}.png', dpi=150)
            plt.close(fig)

    try:
        if not args.no_plot and args.no_show:
            import matplotlib
            matplotlib.use('Agg')

        def map_progress(done, size):
            if done == 1 or done == size or done % max(1, size//20) == 0:
                print(f'Map trials: {done}/{size}', flush=True)

        scan = scan_residuals(T/p['T_star_eV'], js, alphas, p['beta'], p['lambda_sh'],
            coefficients, geometry, **processes, n_points=args.map_points,
            rtol=args.rtol, atol=args.atol, max_step=args.max_step, progress=map_progress)
        starts = choose_starts(scan, max_starts=args.max_starts, separation=args.separation)+manual
        # Exact repeated starts add no information; retain distinct alpha values.
        unique = {}
        for start in starts:
            unique.setdefault(tuple(start[k] for k in ('theta', 'j_target', 'alpha')), start)
        starts = list(unique.values())
        for index, start in enumerate(starts, start=1):
            start.update(start_id=f'start_{index:03d}', T_eV=start['theta']*p['T_star_eV'])
        report.update(stage='map_complete', starts=starts, summary=scan['summary'])
        save_residuals(args.output/'residuals.npz', scan,
                       metadata=dict(parameters=p, model_options=options, provenance=provenance))
        save_report()
        plot_map()
        print(f"Valid trials: {scan['summary']['n_valid']}/{total}; starts: {len(starts)}", flush=True)
        if args.map_only:
            report.update(stage='map_only')
            save_report()
            print('Map only; no stationary correction performed. Output:', args.output.resolve())
            return 0
        report['stage'] = 'refining'
        save_report()

        def refined(done, size, attempt, state):
            attempt['start_id'] = starts[done-1]['start_id']
            report['diagnostic']['attempts'].append(attempt)
            if state is not None:
                record = _solution_record(state, p, args)
                profile = dimensional_profiles(state, p)
                with (args.output/record['profile_file']).open('xb') as stream:
                    np.savez_compressed(stream, **profile, theta=state['theta'], alpha=state['alpha'],
                                        j_target=state['j_target'], T_eV=record['T_eV'])
                report['solutions'].append(record)
                if not args.no_plot:
                    import matplotlib.pyplot as plt
                    fig, _ = plot_profiles(profile)
                    fig.savefig(args.output/f"{state['solution_id']}_profiles.png", dpi=150)
                    plt.close(fig)
                print(f"{state['solution_id']}: T={record['T_eV']:.8g} eV, "
                      f"a={record['a_cm']:.8g} cm, jt={record['j_target']:.10g}", flush=True)
            diagnostic = report['diagnostic']
            diagnostic['n_solutions'] = len(report['solutions'])
            diagnostic['duplicate_convergences'] = sum(a['status'] == 'converged'
                for a in diagnostic['attempts'])-len(report['solutions'])
            save_report()
            print(f"Refinement: {done}/{size}: {attempt['status']}", flush=True)

        states, diagnostic = find_stationary_solutions(
            [(s['theta'], s['j_target'], s['alpha']) for s in starts], p['beta'], p['lambda_sh'],
            coefficients, geometry, **processes, theta_bounds=controls['theta_bounds'],
            max_iterations=args.iterations, n_points=args.points, residual_tol=args.residual_tol,
            balance_tol=args.balance_tol, rtol=args.rtol, atol=args.atol, max_step=args.max_step,
            dedup_tol=args.dedup_tol, progress=refined)
        report.update(stage='refinement_complete', diagnostic=diagnostic)
        save_report()
        plot_map(states, final=True)
        print('Attempt statuses:', dict(Counter(a['status'] for a in diagnostic['attempts'])))
        print('No roots found in this finite search.' if not states else f'Verified distinct roots: {len(states)}')
        print(report['warning'])
        print('Output:', args.output.resolve())
        return 0 if states else 1
    except (Exception, KeyboardInterrupt) as error:
        previous_stage = report['stage']
        report.update(stage='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                      last_completed_or_active_stage=previous_stage,
                      error=dict(type=type(error).__name__, message=str(error)))
        save_report()
        raise


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_scan(args)
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))
    except KeyboardInterrupt:
        print('Interrupted; inspect summary.json and any saved map/roots before restarting.')
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
