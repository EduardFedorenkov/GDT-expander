"""Scan lambda_sh and n0 with a fresh automatic atomic (T,j_t) survey per pair."""

from pathlib import Path
import argparse
import numpy as np

from simple_cx_model.config import load_cx_config
from .geometry import add_geometry_arguments, geometry_from_args
from .inlet import add_inlet_arguments
from .excitation import validate_excitation_scale
from .recombination_map import make_j_grid
from .recombination_batch import scan_lambda_neutral
from .scan_lambda import print_source_summary, positive_grid, plot_grid


def _range(values, low, high, count, name, logarithmic=False):
    if values is not None:
        return positive_grid(values, name)
    if (count < 1 or not np.isfinite([low, high]).all() or not 0 < low <= high
            or (count == 1 and low != high)):
        raise ValueError(f'{name}: need positive ordered bounds; count=1 requires equal bounds or an explicit list.')
    return positive_grid(np.geomspace(low, high, count) if logarithmic
                         else np.linspace(low, high, count), name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=(
        Path(__file__).resolve().parents[1]/'simple_cx_model/cx_config.json'))
    parser.add_argument('--lambdas', type=float, nargs='+')
    parser.add_argument('--lambda-min', type=float, default=5.)
    parser.add_argument('--lambda-max', type=float, default=9.)
    parser.add_argument('--lambda-count', type=int, default=5)
    parser.add_argument('--densities', type=float, nargs='+')
    parser.add_argument('--n0-min', type=float, default=1e12)
    parser.add_argument('--n0-max', type=float, default=1e15)
    parser.add_argument('--n0-count', type=int, default=7)
    parser.add_argument('--temperature-min', type=float, default=.001)
    parser.add_argument('--temperature-max', type=float, default=100.)
    parser.add_argument('--temperature-count', type=int, default=45)
    parser.add_argument('--j-min', type=float, default=1e-6)
    parser.add_argument('--j-max', type=float,
                        help='Map bound only. Default: 1 without ionization, 10 with it.')
    parser.add_argument('--j-count', type=int, default=48)
    parser.add_argument('--j-grid', choices=['mixed', 'linear', 'log'], default='mixed')
    parser.add_argument('--j-axis', choices=['loss', 'linear', 'log'], default='loss')
    parser.add_argument('--max-starts', type=int, default=16)
    parser.add_argument('--iterations', type=int, default=40)
    parser.add_argument('--points', type=int, default=4000)
    parser.add_argument('--residual-tol', type=float, default=1e-9)
    parser.add_argument('--balance-tol', type=float, default=5e-6)
    parser.add_argument('--cx-mode', choices=['constant', 'temperature'], default='temperature')
    parser.add_argument('--rr-scale', type=validate_excitation_scale, default=1.)
    parser.add_argument('--excitation-scale', type=validate_excitation_scale, default=1.)
    parser.add_argument('--ionization-scale', type=validate_excitation_scale, default=1.,
                        help='0: previous physics (default); 1: full atomic ionization.')
    parser.add_argument('--output', type=Path, default=Path('rr_lambda_scan_results'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--no-plot', action='store_true', help='Skip aggregate plots; per-pair maps remain saved.')
    add_geometry_arguments(parser)
    add_inlet_arguments(parser)
    args = parser.parse_args()
    try:
        lambdas = _range(args.lambdas, args.lambda_min, args.lambda_max, args.lambda_count, 'lambda')
        densities = _range(args.densities, args.n0_min, args.n0_max, args.n0_count, 'n0', True)
        if (not np.isfinite([args.temperature_min, args.temperature_max]).all()
                or not 0 < args.temperature_min < args.temperature_max
                or args.temperature_count < 2 or args.max_starts < 1 or args.iterations < 1
                or args.points < 100
                or not np.isfinite([args.residual_tol, args.balance_tol]).all()
                or min(args.residual_tol, args.balance_tol) <= 0):
            raise ValueError('Check the inner grids, iteration count and positive tolerances.')
        if args.j_max is None:
            args.j_max = 10. if args.ionization_scale > 0 else 1.
        js = make_j_grid(args.j_min, args.j_max, args.j_count, args.j_grid)
        geometry = geometry_from_args(args)
        p = load_cx_config(args.config)
        inner_names = ('temperature_min', 'temperature_max', 'temperature_count',
            'j_min', 'j_max', 'j_count', 'j_grid', 'j_axis', 'max_starts', 'iterations',
            'points', 'residual_tol', 'balance_tol', 'cx_mode', 'rr_scale',
            'excitation_scale', 'inlet_momentum')
        inner = []
        for name in inner_names:
            inner.extend(['--'+name.replace('_', '-'), str(getattr(args, name))])
        print_source_summary(p)
        pairs = len(lambdas)*len(densities)
        print(f'Physical pairs: {pairs}; inner grid: {len(js)} x {args.temperature_count}; '
              f'about {pairs*len(js)*args.temperature_count} IVP trials plus root refinement.', flush=True)
        roots, grid = scan_lambda_neutral(lambdas, densities, args.config, args.output,
            geometry=geometry, scan_arguments=inner, ionization_scale=args.ionization_scale,
            resume=args.resume)
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))
    except KeyboardInterrupt:
        print('\nInterrupted. Completed pairs are saved; repeat with --resume.', flush=True)
        return 130
    if not args.no_plot:
        if args.no_show:
            import matplotlib
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axis = plot_grid(grid)
        axis.set_title(f'Verified roots found; ionization={args.ionization_scale:g}\n'
                       '(finite search, not an existence boundary)')
        fig.savefig(args.output/'root_count.png', dpi=170, bbox_inches='tight')
        if roots:
            from .plot_lambda_results import plot_roots
            plot, _ = plot_roots(roots)
            plot.savefig(args.output/f'roots_{args.cx_mode}_recombination.png',
                         dpi=170, bbox_inches='tight')
        if not args.no_show:
            plt.show()
        plt.close('all')
    print(f'Completed {len(grid)} pairs; retained {len(roots)} verified roots.')
    print('A zero count is NOT proof of nonexistence; local IDs are not continuous physical branches.')
    print('Output:', args.output.resolve())
    return 0  # Scan completed; even when all found counts are zero.


if __name__ == '__main__':
    raise SystemExit(main())
