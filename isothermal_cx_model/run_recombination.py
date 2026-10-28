"""Find verified stationary atomic solutions from explicit initial guesses.

Each --guess supplies T_eV and j_target. This is a local/multistart search,
not a proof that all solutions have been found. Does not need the map patch.
"""

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
from .recombination import make_rr_coefficients, T_MIN_EV, T_MAX_EV
from .stationary_recombination import find_stationary_solutions
from .profiles_recombination import compute_stationary_profiles, plot_stationary_profiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=(
        Path(__file__).resolve().parents[1]/'simple_cx_model/cx_config.json'))
    parser.add_argument('--guess', type=float, nargs=2, action='append', required=True,
                        metavar=('T_EV', 'J_TARGET'))
    parser.add_argument('--density', type=float)
    parser.add_argument('--lambda-sh', type=float)
    parser.add_argument('--cx-mode', choices=['constant', 'temperature'], default='temperature')
    parser.add_argument('--rr-scale', type=validate_excitation_scale, default=1.)
    parser.add_argument('--excitation-scale', type=validate_excitation_scale, default=1.)
    parser.add_argument('--ionization-scale', type=validate_excitation_scale, default=0.,
                        help='0: previous physics (default); 1: full atomic ionization.')
    parser.add_argument('--temperature-min', type=float)
    parser.add_argument('--temperature-max', type=float)
    parser.add_argument('--j-min', type=float, default=1e-12)
    parser.add_argument('--j-max', type=float,
                        help='Optional numerical search bound, not a physical limit; j_t>1 allowed.')
    parser.add_argument('--iterations', type=int, default=40)
    parser.add_argument('--points', type=int, default=4000)
    parser.add_argument('--residual-tol', type=float, default=1e-9)
    parser.add_argument('--balance-tol', type=float, default=5e-6)
    parser.add_argument('--output', type=Path, default=Path('rr_stationary_results'))
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--no-plot', action='store_true')
    add_geometry_arguments(parser)
    add_inlet_arguments(parser)
    args = parser.parse_args()
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error('Output must be absent or empty; choose a new --output to preserve old results.')
    try:
        p = load_cx_config(args.config)
        for key, value, allow_zero in [('n0_cm3', args.density, True),
                                       ('lambda_sh', args.lambda_sh, False)]:
            if value is not None:
                if not np.isfinite(value) or value < 0 or (not allow_zero and value == 0):
                    raise ValueError(f'Invalid {key}.')
                p[key] = value
        p['nu_cx_per_s'] = p['n0_cm3']*p['k_cx_cm3_per_s']
        p['D'] = p['L_cm']*p['nu_cx_per_s']/p['U_star_cm_per_s']
        p.pop('theta_exists', None)  # Old constant-area compatibility flag, not used here.
        geometry = geometry_from_args(args)
        bounds = None
        if args.temperature_min is not None or args.temperature_max is not None:
            if args.temperature_min is None or args.temperature_max is None:
                raise ValueError('Supply both temperature bounds or neither.')
            bounds = np.asarray([args.temperature_min, args.temperature_max], dtype=float)
        elif args.rr_scale > 0:
            bounds = np.array([T_MIN_EV, T_MAX_EV])
        theta_bounds = None if bounds is None else bounds/p['T_star_eV']
        guesses = np.asarray(args.guess, dtype=float)
        guesses[:, 0] /= p['T_star_eV']
        states, diagnostic = find_stationary_solutions(guesses, p['beta'],
            make_D(p, args.cx_mode), p['lambda_sh'], geometry,
            E_exc=make_E_exc(p, scale=args.excitation_scale),
            rr=make_rr_coefficients(p, scale=args.rr_scale),
            ionization=make_ionization_coefficients(p, scale=args.ionization_scale),
            inlet_momentum=make_inlet_momentum(p, geometry, mode=args.inlet_momentum),
            theta_bounds=theta_bounds, j_min=args.j_min, j_max=args.j_max,
            max_iterations=args.iterations, n_points=args.points,
            residual_tol=args.residual_tol, balance_tol=args.balance_tol)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    profiles = [compute_stationary_profiles(state, p) for state in states]
    summaries = []
    for state, profile in zip(states, profiles):
        summaries.append(dict(
            solution_id=state['solution_id'], T_eV=profile['T_eV'],
            theta=state['theta'], j_target=state['j_target'], alpha=state['alpha'],
            a_cm=profile['a_cm'], net_particle_loss_fraction=1-state['j_target'],
            recombination_count_per_inlet=state['balances']['particle_recombination_integral'],
            ionization_count_per_inlet=state['balances']['particle_ionization_integral'],
            R_J=state['R_J'], R_M=state['R_M'], R_E=state['R_E'],
            Jtarget_per_s=float(profile['J_per_s'][-1]),
            Wtarget_erg_per_s=float(profile['W_erg_per_s'][-1]),
            mu_a=state['mu_a'], mu_target=state['mu_target'],
            coefficients={key: state[key] for key in ('D', 'E_exc', 'R_rr', 'Q_rr',
                                                     'D_ion', 'E_ion')},
            verification=state['verification'], balances=state['balances'],
            delta_theta=state['delta_theta'], delta_j_target=state['delta_j_target']))
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(source_config=str(args.config.resolve()), parameters=p, geometry=geometry.to_dict(),
        cx_mode=args.cx_mode, excitation_scale=args.excitation_scale, rr_scale=args.rr_scale,
        ionization_scale=args.ionization_scale, schema_version=2,
        inlet_momentum=args.inlet_momentum, guesses_T_eV_j_target=args.guess,
        temperature_bounds_eV=None if bounds is None else bounds.tolist(),
        j_min=args.j_min, j_max=args.j_max, solutions=summaries, diagnostic=diagnostic,
        solver_controls=dict(max_iterations=args.iterations, n_points=args.points,
                             residual_tol=args.residual_tol, balance_tol=args.balance_tol),
        physical_scope='Isothermal atomic CX, excitation, Case-A RR and optional direct ionization; '
                       'prescribed constant neutrals; no three-body recombination or molecules.')
    (args.output/'solutions.json').write_text(json.dumps(report, indent=2, allow_nan=False),
                                            encoding='utf-8')
    for state, profile in zip(states, profiles):
        arrays = {key: value for key, value in profile.items() if isinstance(value, np.ndarray)}
        arrays.update(theta=state['theta'], alpha=state['alpha'], j_target=state['j_target'],
                      ionization_scale=args.ionization_scale, D_ion=state['D_ion'], E_ion=state['E_ion'])
        np.savez_compressed(args.output/f"{state['solution_id']}_profiles.npz", **arrays)
    for item in summaries:
        print(f"{item['solution_id']}: T={item['T_eV']:.9g} eV, "
              f"a={item['a_cm']:.9g} cm, jt={item['j_target']:.12g}, "
              f"net_loss={item['net_particle_loss_fraction']:.5g}, "
              f"R_J={item['R_J']:.2e}, R_M={item['R_M']:.2e}", flush=True)
    if profiles and not args.no_plot:
        if args.no_show:
            import matplotlib
            matplotlib.use('Agg')
        figure, _ = plot_stationary_profiles(profiles,
            title=f"n0={p['n0_cm3']:.3g} cm^-3, lambda={p['lambda_sh']:g}, "
                  f"RR={args.rr_scale:g}, excitation={args.excitation_scale:g}, "
                  f"ionization={args.ionization_scale:g}")
        figure.savefig(args.output/'profiles.png', dpi=150)
        import matplotlib.pyplot as plt
        if args.no_show:
            plt.close(figure)
        else:
            plt.show()
    if not states:
        print('No verified root from these guesses. This is NOT a nonexistence proof.')
    print(diagnostic['warning'])
    print('Output:', args.output.resolve())
    return 0 if states else 1


if __name__ == '__main__':
    raise SystemExit(main())
