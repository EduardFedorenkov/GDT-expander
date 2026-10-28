"""A finite 2D trial map, automatic starts and a static diagnostic figure.

No interpolation across undefined trials. A candidate cell or a small residual
at a grid point is NOT a verified stationary solution.
"""

from collections import Counter
import numpy as np

from .geometry import quadratic_area_legacy
from .solver_atomic import solve_trial


def _axis(values, name, maximum=None):
    values = np.asarray(values, dtype=float)
    if (values.ndim != 1 or values.size < 2 or not np.isfinite(values).all()
            or np.any(values <= 0) or np.any(np.diff(values) <= 0)
            or (maximum is not None and np.any(values > maximum))):
        raise ValueError(f'{name}: need >=2 ordered, distinct, positive finite values.')
    return values.copy()


def make_j_grid(j_min=1e-6, j_max=1., count=48, mode='mixed'):
    """Mixed: broad logarithmic j_t plus logarithmic losses down to 1e-12.

    Includes j_t=1 if requested. The actual count can differ after merging.
    For ranges below .9 the mixed grid reduces to a logarithmic grid.
    """
    if (not isinstance(count, (int, np.integer)) or count < 4
            or not np.isfinite([j_min, j_max]).all() or not 0 < j_min < j_max <= 1
            or mode not in ('mixed', 'linear', 'log')):
        raise ValueError('Need 0<j_min<j_max<=1, integer count>=4 and a known grid mode.')
    if mode == 'linear':
        return np.linspace(j_min, j_max, count)
    if mode == 'log' or j_max <= .9:
        return np.geomspace(j_min, j_max, count)
    broad = np.geomspace(j_min, j_max, (count+1)//2)
    min_gap = max(1-j_max, min(1e-12, .1*(1-j_min)))
    max_gap = min(.5, 1-j_min)
    near_one = 1-np.geomspace(min_gap, max_gap, count//2)
    near_one = near_one[(near_one >= j_min) & (near_one <= j_max)]
    return np.unique(np.r_[broad, near_one])


def scan_residuals(theta_values, j_targets, beta, D, lambda_sh, geometry=None,
                   *, E_exc=0., rr=(0., 0.), inlet_momentum=None,
                   n_points=100, rtol=1e-9, atol=1e-11, max_step=.02, progress=None):
    """Scalar arrays have shape (n_j, n_theta), not (n_theta, n_j).

    'valid' means both inlet residuals exist, NOT that a BVP is solved.
    Keep off-root j_a>1 and negative target sonic margins. Errors in input
    or coefficient code propagate; explicit invalid-IVP statuses are stored.
    """
    theta_values = _axis(theta_values, 'theta')
    j_targets = _axis(j_targets, 'j_target', maximum=1.)
    geometry = quadratic_area_legacy() if geometry is None else geometry
    shape = (len(j_targets), len(theta_values))
    fields = ('R_J', 'R_M', 'R_J_scaled', 'R_M_scaled', 'alpha', 'R_E',
              'wall_slope', 'j_a', 'mu_a', 'mu_target')
    data = {key: np.full(shape, np.nan) for key in fields}
    status = np.full(shape, '', dtype='U64')
    reasons = Counter()
    for ti, theta in enumerate(theta_values):
        for ji, jt in enumerate(j_targets):
            trial = solve_trial(theta, jt, beta, D, lambda_sh, geometry,
                E_exc=E_exc, rr=rr, inlet_momentum=inlet_momentum,
                n_points=n_points, rtol=rtol, atol=atol, max_step=max_step)
            status[ji, ti] = trial['status']
            for key in fields:
                data[key][ji, ti] = trial.get(key, np.nan)
            if 'reason' in trial:
                reasons[str(trial['reason'])] += 1
        if progress is not None:
            progress(ti+1, len(theta_values))
    valid = status == 'energy_matched'
    for key in ('R_J', 'R_M', 'R_J_scaled', 'R_M_scaled', 'alpha',
                'j_a', 'mu_a', 'mu_target'):
        valid &= np.isfinite(data[key])
    valid &= (data['j_a'] > 0) & (data['mu_a'] > 0) & (data['mu_target'] > 0)
    norm = np.full(shape, np.nan)
    # Exactly the residual norm used by the local stationary corrector.
    norm[valid] = np.maximum(np.abs(np.log(data['j_a'][valid])),
        np.abs(np.log(data['mu_a'][valid]/data['mu_target'][valid])))
    data.update(theta=theta_values, j_target=j_targets, status=status,
                valid=valid, log_residual_norm=norm)
    summary = dict(n_trials=int(status.size), n_valid=int(valid.sum()),
        statuses=dict(Counter(status.ravel())), reason_counts=dict(reasons),
        unresolved_numerical_count=int(sum(np.count_nonzero(status == item) for item in (
            'integration_failed', 'critical_wall_unresolved',
            'sonic_resolution_limited', 'profile_validation_failed'))),
        warning='Finite grid, not a proof of existence, absence or completeness of roots.')
    if valid.any():
        ji, ti = np.unravel_index(np.nanargmin(norm), shape)
        summary['nearest_sample_not_root'] = dict(theta=float(theta_values[ti]),
            j_target=float(j_targets[ji]), log_residual_norm=float(norm[ji, ti]),
            R_J=float(data['R_J'][ji, ti]), R_M=float(data['R_M'][ji, ti]))
    else:
        summary['nearest_sample_not_root'] = None
    data['summary'] = summary
    summary['candidate_cell_count'] = len(candidate_cells(data))
    return data


def candidate_cells(scan):
    """Both residual ranges contain zero at FOUR valid corners.

    Only a heuristic screen: parallel zero curves can satisfy it; tangencies,
    narrow domains, or multiple crossings inside a cell can be missed.
    """
    good = scan['valid']
    eligible = good[:-1, :-1] & good[1:, :-1] & good[:-1, 1:] & good[1:, 1:]
    for key in ('R_J', 'R_M'):
        v = scan[key]
        corners = np.stack((v[:-1, :-1], v[1:, :-1], v[:-1, 1:], v[1:, 1:]))
        eligible &= (np.isfinite(corners).all(axis=0)
                     & (corners.min(axis=0) <= 0) & (corners.max(axis=0) >= 0))
    return [(int(ji), int(ti)) for ji, ti in np.argwhere(eligible)]


def choose_starts(scan, max_starts=16, separation=2):
    """Valid grid samples: candidate-cell corners, local minima, then low norms.

    Separation is in grid indices, not a physical root-deduplication rule.
    Nothing here asserts a root, a bracket, or convergence of a chosen start.
    """
    if (not isinstance(max_starts, (int, np.integer)) or max_starts < 1
            or not isinstance(separation, (int, np.integer)) or separation < 1):
        raise ValueError('Need positive integer max_starts and separation.')
    valid, norm = scan['valid'], scan['log_residual_norm']
    ranked = []
    for ji, ti in candidate_cells(scan):
        corners = [(ji+dji, ti+dti) for dji in (0, 1) for dti in (0, 1)]
        j, t = min(corners, key=lambda index: norm[index])
        ranked.append((0, float(norm[j, t]), j, t, 'candidate_cell_corner'))
    for ji, ti in np.argwhere(valid):
        neighbors = norm[max(0, ji-1):ji+2, max(0, ti-1):ti+2]
        local = norm[ji, ti] <= np.nanmin(neighbors)
        ranked.append((1 if local else 2, float(norm[ji, ti]), int(ji), int(ti),
                       'grid_local_minimum' if local else 'low_residual_sample'))
    starts = []
    for _, score, ji, ti, source in sorted(ranked):
        if any(max(abs(ji-s['j_index']), abs(ti-s['theta_index'])) < separation
               for s in starts):
            continue
        starts.append(dict(theta=float(scan['theta'][ti]),
            j_target=float(scan['j_target'][ji]), j_index=ji, theta_index=ti,
            source=source, grid_log_residual_norm=score))
        if len(starts) == max_starts:
            break
    return starts


def plot_residuals(scan, T_star_eV=1., *, solutions=(), starts=(), attempts=(),
                   title=None, j_axis='loss', show_paths=True):
    """Signed residuals and their joint norm; roots/seeds are separate symbols.

    'loss' shows 1-j_t on a symlog axis (includes the exact endpoint j_t=1).
    Both contours share one validity mask; corner_mask=False forbids bridging
    a cell with an invalid corner. Paths show accepted corrector iterates,
    not streamlines, plasma trajectories or a basin-of-attraction map.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, SymLogNorm
    from matplotlib.lines import Line2D
    if not np.isfinite(T_star_eV) or T_star_eV <= 0 or j_axis not in ('loss', 'linear', 'log'):
        raise ValueError('Need T_star_eV>0 and j_axis=loss,linear,log.')
    x = scan['theta']*T_star_eV
    y = 1-scan['j_target'] if j_axis == 'loss' else scan['j_target']
    order = np.argsort(y)
    y = y[order]
    valid = scan['valid'][order]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8), sharex=True, sharey=True,
                             layout='constrained')
    signed_norm = SymLogNorm(linthresh=1e-4, vmin=-1, vmax=1)
    norm_field = np.log10(np.maximum(scan['log_residual_norm'], 1e-12))
    panels = [('R_J_scaled', r'$R_J$ (scaled)', signed_norm, 'coolwarm'),
              ('R_M_scaled', r'$R_M$ (scaled)', signed_norm, 'coolwarm'),
              (None, r'$\log_{10}\max(|\ln j_a|,|\ln(\mu_a/\mu_{in})|)$',
               Normalize(vmin=-6, vmax=1), 'viridis_r')]
    def ypos(jt):
        return 1-np.asarray(jt) if j_axis == 'loss' else np.asarray(jt)
    for axis, (key, label, normalization, cmap) in zip(axes, panels):
        axis.set_facecolor('#b0b0b0')
        values = (norm_field if key is None else scan[key])[order]
        # Nearest shading is a display of sampled nodes, NOT interpolation.
        mesh = axis.pcolormesh(x, y, np.ma.masked_where(~valid, values),
                              shading='nearest', cmap=cmap, norm=normalization)
        fig.colorbar(mesh, ax=axis, shrink=.8)
        for field, color, style in [('R_J', 'black', '-'), ('R_M', '#d97706', '--')]:
            z = np.ma.masked_where(~valid, scan[field][order])
            finite = z.compressed()
            if finite.size and finite.min() <= 0 <= finite.max() and finite.min() < finite.max():
                axis.contour(x, y, z, levels=[0], colors=[color], linestyles=style,
                             linewidths=1.5, corner_mask=False)
        if starts:
            axis.scatter([s['theta']*T_star_eV for s in starts],
                ypos([s['j_target'] for s in starts]), s=40, marker='o',
                facecolors='none', edgecolors='#db2777', linewidths=1.1, zorder=4)
        for root in solutions:
            axis.scatter(root['theta']*T_star_eV, ypos(root['j_target']), marker='*',
                         s=170, c='#22c55e', edgecolors='black', linewidths=.8, zorder=6)
        axis.set_xscale('log')
        if j_axis == 'loss':
            axis.set_yscale('symlog', linthresh=1e-12, linscale=.5)
        else:
            axis.set_yscale(j_axis)
        axis.set_xlim(x[0], x[-1])
        axis.set_ylim(y[0], y[-1])
        axis.set_title(label, fontsize=11)
        axis.set_xlabel(r'$T$ [eV]')
        axis.grid(alpha=.15)
        if not valid.any():
            axis.text(.5, .5, 'No joint valid trials', ha='center', va='center',
                      transform=axis.transAxes)
    if show_paths:
        for attempt in attempts:
            # One uninterrupted line per correction phase; do not draw a fake
            # path between unrelated search and verification evaluations.
            for phase in ('search', 'verification'):
                h = [item for item in attempt.get('history', []) if item['phase'] == phase]
                if len(h) > 1:
                    axes[2].plot([s['theta']*T_star_eV for s in h],
                        ypos([s['j_target'] for s in h]), color='#e11d48',
                        linewidth=.7, alpha=.5, zorder=3)
    for root in solutions:
        root_x = root['theta']*T_star_eV
        near_right = np.log(root_x/x[0])/np.log(x[-1]/x[0]) > .75
        axes[2].annotate(root.get('solution_id', 'root'),
            (root_x, ypos(root['j_target'])),
            xytext=(-7 if near_right else 6, 7), textcoords='offset points',
            ha='right' if near_right else 'left', fontsize=8, zorder=7, clip_on=True)
    axes[0].set_ylabel(r'$1-j_t$ (particle loss at a root)' if j_axis == 'loss'
                       else r'$j_t=J(L)/J_0$')
    legend = [Line2D([], [], color='black', label=r'$R_J=0$'),
              Line2D([], [], color='#d97706', linestyle='--', label=r'$R_M=0$'),
              Line2D([], [], marker='o', color='#db2777', markerfacecolor='none',
                     linestyle='', label='Selected starts'),
              Line2D([], [], marker='*', color='#22c55e', markeredgecolor='black',
                     markersize=12, linestyle='', label='Verified roots')]
    axes[0].legend(handles=legend, loc='lower left', fontsize=8)
    fig.suptitle((title+'\n' if title else '')+
        'Grey: undefined trials. Finite grid; zero-curve crossings need verification. '
        'Red paths: accepted numerical iterates.', fontsize=10)
    return fig, axes
