"""Plot saved ADAS roots.csv without importing or running the plasma solver.

Three panels: T(n0), a(n0), j_t(n0). Multiple roots remain separate samples.
Optional temperature-ranked curves are visual guides, not physical branches.
"""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np


_CORE = ('lambda_sh', 'n0_cm3', 'T_eV', 'a_cm', 'j_target')
_SCALES = ('rr_scale', 'ionization_scale', 'excitation_scale')


def _read_csv(path, required):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream, delimiter=';')
        names = reader.fieldnames or []
        missing = set(required)-set(names)
        if missing or len(names) != len(set(names)):
            raise ValueError(f'{path}: missing columns {sorted(missing)} or duplicate headers.')
        rows = []
        for line, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f'{path}: malformed CSV row {line}.')
            rows.append(row)
    return rows


def _roots(rows):
    result, seen = [], set()
    for line, source in enumerate(rows, start=2):
        row = dict(source)
        try:
            for key in (*_CORE, *[name for name in _SCALES if name in row]):
                row[key] = float(row[key])
            if (not np.isfinite([row[key] for key in _CORE]).all()
                    or row['n0_cm3'] < 0 or min(row[key] for key in _CORE if key != 'n0_cm3') <= 0
                    or any(not np.isfinite(row[key]) or not 0 <= row[key] <= 1 for key in _SCALES if key in row)):
                raise ValueError('Invalid root coordinates or process scale.')
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f'Invalid root data at CSV row {line}: {error}') from error
        identity = tuple(row[key] for key in _CORE)+tuple(row.get(key) for key in (*_SCALES, 'inlet_momentum'))
        if identity in seen:
            raise ValueError(f'Duplicate root sample at CSV row {line}.')
        seen.add(identity)
        result.append(row)
    return result


def load_roots_csv(path):
    """Read semicolon/UTF-8(-BOM) ADAS CSV. No missing j_t->1 substitution.

    The plotter checks table consistency, not the BVP or integral budgets.
    Extra columns are retained; optional process metadata separates groups.
    """
    return _roots(_read_csv(path, _CORE))


def _grid(rows):
    result, seen = [], set()
    for line, source in enumerate(rows, start=2):
        row = dict(source)
        try:
            lam, density, count = (float(row[key]) for key in ('lambda_sh', 'n0_cm3', 'n_roots'))
            if (not np.isfinite([lam, density, count]).all() or lam <= 0 or density < 0
                    or count < 0 or not count.is_integer()):
                raise ValueError('Invalid parameter or root count.')
            row.update(lambda_sh=lam, n0_cm3=density, n_roots=int(count))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f'Invalid grid data at CSV row {line}: {error}') from error
        expected = 'roots_found' if count else 'no_root_found_in_finite_search'
        if 'search_status' in row and row['search_status'] != expected:
            raise ValueError(f'Unfinished or inconsistent search status at CSV row {line}.')
        pair = (lam, density)
        if pair in seen:
            raise ValueError(f'Duplicate physical pair in grid row {line}.')
        seen.add(pair)
        result.append(row)
    return result


def load_grid_csv(path):
    return _grid(_read_csv(path, ('lambda_sh', 'n0_cm3', 'n_roots')))


def load_density_plan(path):
    """Only read the neutral-density plan, not solver inputs or ADAS files."""
    manifest = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if (manifest.get('model') != 'isothermal_adas' or manifest.get('kind') != 'lambda_neutral_scan'
            or manifest.get('format_version') != 1):
        raise ValueError('Not an ADAS lambda-neutral scan manifest.')
    return _density_plan(manifest['signature']['densities'])


def _density_plan(values):
    values = np.asarray(values, float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Density plan must be a nonempty finite nonnegative 1D grid.')
    return np.unique(values)


def _check_grid(rows, grid):
    lookup = {(row['lambda_sh'], row['n0_cm3']): row for row in grid}
    counts = Counter((row['lambda_sh'], row['n0_cm3']) for row in rows)
    for root in rows:
        pair = (root['lambda_sh'], root['n0_cm3'])
        if pair not in lookup:
            raise ValueError('A root has no matching completed pair in grid.csv.')
        case = lookup[pair]
        if root.get('case_id') and case.get('case_id') and root['case_id'] != case['case_id']:
            raise ValueError('Root/grid case IDs disagree.')
    if any(counts.get(pair, 0) != case['n_roots'] for pair, case in lookup.items()):
        raise ValueError('Root counts disagree between roots.csv and grid.csv; use a coherent snapshot.')


def _temperature_tracks(rows, density_plan, completed):
    """Rank-only guides; break at zero/unknown densities, ties or count changes."""
    by_density = {}
    for row in rows:
        by_density.setdefault(row['n0_cm3'], []).append(row)
    tracks, active = [], []
    for density in density_plan:
        if density <= 0 or density not in completed:
            active = []
            continue
        samples = sorted(by_density.get(density, []), key=lambda row: row['T_eV'])
        temperatures = np.array([row['T_eV'] for row in samples])
        if len(samples) > 1 and np.any(np.diff(temperatures) <= 1e-12*np.max(temperatures)):
            active = []  # Equal temperatures do not define a unique ordering.
            continue
        if len(samples) != len(active):
            active = [[] for _ in samples]
            tracks.extend(active)
        for track, row in zip(active, samples):
            track.append(row)
    return tracks


def plot_roots(rows, lambdas=None, *, connect=False, grid=None, density_values=None,
               temperature_scale='linear'):
    """Return (figure, axes[3]). No file access, solver, show or save here.

    connect needs completed-case data. Missing planned cases break guides.
    Without a density plan, only gaps known from the supplied grid can be
    detected. Curves are PCHIP in log10(n0), with no extrapolation.
    """
    rows = _roots(rows)
    grid = None if grid is None else _grid(grid)
    if temperature_scale not in ('linear', 'log'):
        raise ValueError('temperature_scale must be linear or log.')
    if connect and grid is None:
        raise ValueError('Connecting guides requires grid.csv to distinguish gaps from roots.')
    if lambdas is not None:
        requested = np.asarray(lambdas, float)
        if requested.ndim != 1 or not requested.size or not np.isfinite(requested).all() or np.any(requested <= 0):
            raise ValueError('Need a nonempty list of positive finite lambdas.')
        available = {row['lambda_sh'] for row in [*rows, *(grid or [])]}
        if any(not any(np.isclose(lam, known, rtol=1e-9, atol=0) for known in available) for lam in requested):
            raise ValueError('No saved completed data for at least one requested lambda.')
        def selected(row):
            return any(np.isclose(row['lambda_sh'], lam, rtol=1e-9, atol=0) for lam in requested)
        rows = [row for row in rows if selected(row)]
        grid = None if grid is None else [row for row in grid if selected(row)]
    if grid is not None:
        _check_grid(rows, grid)
    all_densities = sorted({row['n0_cm3'] for row in [*rows, *(grid or [])]})
    plan = _density_plan(density_values) if density_values is not None else np.array(all_densities)
    if any(density not in plan for density in all_densities):
        raise ValueError('A saved density lies outside the supplied scan plan.')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True, layout='constrained')
    columns = ('T_eV', 'a_cm', 'j_target')
    group_fields = ('lambda_sh', *_SCALES, 'inlet_momentum')
    def group(row):
        return tuple(row.get(key) for key in group_fields)
    groups = sorted({group(row) for row in rows}, key=lambda g: (g[0], *map(str, g[1:])))
    mixed_physics = len({g[1:] for g in groups}) > 1
    colors = plt.get_cmap('tab10')
    for index, key in enumerate(groups):
        samples = [row for row in rows if group(row) == key]
        label = rf'$\lambda_{{\mathrm{{sh}}}}={key[0]:g}$'
        if mixed_physics:
            label += '\n'+', '.join(f'{name}={value if value is not None else "?"}'
                for name, value in zip(('RR', 'ion', 'exc', 'inlet'), key[1:]))
        color = colors(index % 10)
        if connect:
            from scipy.interpolate import PchipInterpolator
            completed = {row['n0_cm3'] for row in grid if row['lambda_sh'] == key[0]}
            for track in _temperature_tracks(samples, plan, completed):
                if len(track) < 2:
                    continue
                knots = np.log10([row['n0_cm3'] for row in track])
                smooth = np.unique(np.r_[knots, np.linspace(knots[0], knots[-1], 200)])
                for axis, column in zip(axes, columns):
                    values = PchipInterpolator(knots, [row[column] for row in track], extrapolate=False)(smooth)
                    axis.plot(10.**smooth, values, color=color, linewidth=.9, alpha=.7, linestyle=':')
        for axis, column in zip(axes, columns):
            axis.scatter([row['n0_cm3'] for row in samples], [row[column] for row in samples],
                         s=22, color=color, label=label, zorder=3)
    axes[0].set_yscale(temperature_scale)
    axes[2].axhline(1., color='0.5', linestyle='--', linewidth=.8, zorder=0)
    extent = plan.tolist() if plan.size else all_densities
    for axis, label in zip(axes, ('T [eV]', 'a [cm]', r'$j_t=J(L)/J_0$')):
        axis.set_ylabel(label)
        if any(density == 0 for density in extent):
            positive = [density for density in extent if density > 0]
            axis.set_xscale('symlog', linthresh=min(positive)/10 if positive else 1.)
        else:
            axis.set_xscale('log')
        if extent:
            low, high = min(extent), max(extent)
            if high == 0:
                axis.set_xlim(0., 1.)
            elif low == high:
                axis.set_xlim(low/1.2, high*1.2)
            else:
                axis.set_xlim(low*.95, high*1.05)
        axis.grid(alpha=.2)
        if not rows:
            axis.text(.5, .5, 'No saved roots in the selected finite searches',
                      transform=axis.transAxes, ha='center', va='center', fontsize=9)
    axes[-1].set_xlabel(r'$n_0$ [cm$^{-3}$]')
    if rows:
        axes[0].legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1))
    figure.suptitle('Saved roots; lines are temperature-ranked visual guides, NOT physical branches'
                   if connect else 'Saved root samples; no physical branch or stability assignment', fontsize=10)
    return figure, axes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv_path', type=Path, help='roots.csv or a batch output directory.')
    parser.add_argument('--lambdas', type=float, nargs='+')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--grid', type=Path, help='Default: sibling grid.csv if present.')
    group.add_argument('--no-grid', action='store_true', help='Do not read a sibling grid.')
    parser.add_argument('--manifest', type=Path, help='Default: sibling manifest.json if present.')
    parser.add_argument('--save', type=Path)
    parser.add_argument('--overwrite', action='store_true', help='Allow replacing the requested image, never CSV.')
    parser.add_argument('--no-show', action='store_true')
    lines = parser.add_mutually_exclusive_group()
    lines.add_argument('--connect', action='store_true', help='Ranked PCHIP guides; needs grid.csv.')
    lines.add_argument('--points-only', action='store_true', help='Computed points only (default).')
    parser.add_argument('--temperature-scale', choices=('linear', 'log'), default='linear')
    args = parser.parse_args(argv)
    try:
        path = args.csv_path/'roots.csv' if args.csv_path.is_dir() else args.csv_path
        grid_path = args.grid or path.with_name('grid.csv')
        manifest_path = args.manifest or path.with_name('manifest.json')
        rows = load_roots_csv(path)
        grid = load_grid_csv(grid_path) if not args.no_grid and (args.grid or grid_path.exists()) else None
        plan = load_density_plan(manifest_path) if args.manifest or manifest_path.exists() else None
        if args.save and args.save.resolve() in {path.resolve(), grid_path.resolve(), manifest_path.resolve()}:
            raise ValueError('The image path must not overwrite an input file.')
        if args.save and args.save.exists() and not args.overwrite:
            raise ValueError('Image already exists; choose another --save path or use --overwrite.')
        if args.no_show:
            import matplotlib
            matplotlib.use('Agg')
        figure, _ = plot_roots(rows, args.lambdas, connect=args.connect, grid=grid,
                              density_values=plan, temperature_scale=args.temperature_scale)
        import matplotlib.pyplot as plt
        if args.save:
            figure.savefig(args.save, dpi=180)
            print('Saved:', args.save.resolve())
        if not args.no_show:
            plt.show()
        plt.close(figure)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
