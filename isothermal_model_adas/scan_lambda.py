"""Independent ADAS stationary searches over lambda_sh and neutral n0.

All plasma work is delegated to scan_stationary. --dry-run never integrates.
Zero found roots is not a proof of physical nonexistence or a detachment boundary.
"""

import argparse
from pathlib import Path

import numpy as np

from .exitation_adas import validate_excitation_scale
from .geometry import add_geometry_arguments
from .parameter_scan import CHOICES, parameter_grid, scan_defaults, scan_lambda_neutral


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    parser.add_argument('--lambdas', type=float, nargs='+')
    parser.add_argument('--lambda-min', type=float, default=5.)
    parser.add_argument('--lambda-max', type=float, default=9.)
    parser.add_argument('--lambda-count', type=int, default=5)
    parser.add_argument('--densities', type=float, nargs='+', help='Neutral densities [cm^-3]; zero is a control case.')
    parser.add_argument('--n0-min', type=float, default=1e12)
    parser.add_argument('--n0-max', type=float, default=1e15)
    parser.add_argument('--n0-count', type=int, default=7)
    for key, default in scan_defaults().items():
        flag = '--'+key.replace('_', '-')
        if key == 'no_plot':
            parser.add_argument('--no-case-plots', action='store_true', help='Skip per-pair maps/profile PNGs.')
        elif key == 'guess':
            parser.add_argument(flag, type=float, nargs=3, action='append', default=[],
                metavar=('T_EV', 'J_TARGET', 'ALPHA'), help='Extra start repeated for every pair, not required.')
        elif key == 'alpha_slice':
            parser.add_argument(flag, type=int, help='Additional alpha-index slice in every pair.')
        else:
            options = dict(default=default)
            if key in CHOICES:
                options['choices'] = CHOICES[key]
            else:
                options['type'] = validate_excitation_scale if key.endswith('_scale') else type(default)
            parser.add_argument(flag, **options)
    parser.add_argument('--output', type=Path, default=Path('adas_lambda_scan_results'))
    parser.add_argument('--resume', action='store_true', help='Reuse only completed, unchanged pairs.')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--no-show', action='store_true', help='No interactive aggregate plots.')
    parser.add_argument('--no-plot', action='store_true', help='Skip aggregate plots; see also --no-case-plots.')
    add_geometry_arguments(parser)
    return parser


def _range(explicit, low, high, count, name, *, logarithmic=False):
    if explicit is not None:
        return parameter_grid(explicit, name, allow_zero=logarithmic)
    if (count < 1 or not np.isfinite([low, high]).all() or not 0 < low <= high
            or (count == 1 and low != high)):
        raise ValueError(f'{name}: positive ordered bounds required; count=1 needs equal bounds or an explicit list.')
    values = (np.geomspace if logarithmic else np.linspace)(low, high, count)
    return parameter_grid(values, name)


def plot_results(roots, grid):
    """Sampled root counts and scatter T(n0),a(n0); no inferred branch links."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm

    def density_axis(axis):
        densities = np.array([row['n0_cm3'] for row in grid])
        if np.any(densities == 0):
            positive = densities[densities > 0]
            axis.set_xscale('symlog', linthresh=float(positive.min()/10) if positive.size else 1.)
        else:
            axis.set_xscale('log')
        axis.set_xlabel(r'$n_0$ [cm$^{-3}$]')
        axis.grid(alpha=.2)

    figure, axis = plt.subplots(figsize=(8, 5), layout='constrained')
    maximum = max([1, *[row['n_roots'] for row in grid]])
    cmap = plt.get_cmap('viridis', maximum+1)
    points = axis.scatter([r['n0_cm3'] for r in grid], [r['lambda_sh'] for r in grid],
        c=[r['n_roots'] for r in grid], s=65, cmap=cmap,
        norm=BoundaryNorm(np.arange(-.5, maximum+1.5), cmap.N))
    figure.colorbar(points, ax=axis, ticks=range(maximum+1), label='Verified roots found')
    density_axis(axis)
    axis.set_ylabel(r'$\lambda_{\mathrm{sh}}$')
    axis.set_title('Completed finite searches; not an existence/detachment boundary')

    profiles, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    lambdas = sorted({row['lambda_sh'] for row in grid})
    colors = plt.get_cmap('tab10')
    for index, lam in enumerate(lambdas):
        rows = [row for row in roots if row['lambda_sh'] == lam]
        for ax, field in zip(axes, ('T_eV', 'a_cm')):
            ax.scatter([r['n0_cm3'] for r in rows], [r[field] for r in rows],
                       s=32, color=colors(index % 10), label=f'lambda={lam:g}')
    for ax, label in zip(axes, ('T [eV]', 'a [cm]')):
        density_axis(ax)
        ax.set_ylabel(label)
    if roots:
        axes[0].set_yscale('log')
        axes[0].legend(fontsize=8)
    else:
        for ax in axes:
            ax.text(.5, .5, 'No roots found in these finite searches', transform=ax.transAxes,
                    ha='center', va='center', fontsize=9)
    profiles.suptitle('Verified roots; points are not assigned to continuous physical branches')
    return figure, profiles


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        lambdas = _range(args.lambdas, args.lambda_min, args.lambda_max, args.lambda_count, 'lambda_sh')
        densities = _range(args.densities, args.n0_min, args.n0_max, args.n0_count, 'n0', logarithmic=True)
        options = {key: getattr(args, key) for key in scan_defaults() if key != 'no_plot'}
        options['no_plot'] = args.no_case_plots
        roots, grid = scan_lambda_neutral(lambdas, densities, args.config, args.output,
            geometry_path=args.geometry, K=args.K, scan_options=options,
            resume=args.resume, dry_run=args.dry_run)
        if args.dry_run:
            return 0
        if not args.no_plot:
            if args.no_show:
                import matplotlib
                matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            figures = plot_results(roots, grid)
            for fig, name in zip(figures, ('root_count.png', 'roots.png')):
                fig.savefig(args.output/name, dpi=160)
            if not args.no_show:
                plt.show()
            for fig in figures:
                plt.close(fig)
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        parser.error(str(error))
    except KeyboardInterrupt:
        print('Interrupted. Completed pairs are checkpointed; repeat the same command with --resume.')
        return 130
    print(f'Completed {len(grid)} pairs; {len(roots)} verified roots saved.')
    print('Zero found roots does not prove nonexistence; no physical branch/stability assignment.')
    print('Output:', args.output.resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
