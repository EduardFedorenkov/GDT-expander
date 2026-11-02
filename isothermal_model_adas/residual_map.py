"""Finite (theta, j_t, alpha) trial maps, not stationary solutions.

Arrays are indexed [alpha_index, j_index, theta_index]. No IVP runs on
import. Coefficients and inlet closure have the same interface as transport.
Undefined trials remain masked; coefficient/programming errors propagate.
"""

from collections import Counter
import json
from pathlib import Path

import numpy as np

from .geometry import quadratic_area_legacy
from .transport import solve_trial


_RESIDUALS = ('F_J', 'F_M', 'F_E')
_INLET_FIELDS = ('R_J', 'R_M', 'R_E', *_RESIDUALS, 'residual_norm',
                 'j_a', 'mu_a', 'omega_a', 'mu_target')
_FIELDS = (*_INLET_FIELDS, 'D', 'wall_slope_per_j')
_WARNING = ('Finite grid only: samples, sign-changing cells and selected starts '
            'do not prove existence, absence, uniqueness or completeness of roots.')


def _axis(values, name, *, positive=True):
    axis = np.asarray(values, dtype=float)
    if (axis.ndim != 1 or not axis.size or not np.isfinite(axis).all()
            or np.any(np.diff(axis) <= 0) or (positive and np.any(axis <= 0))):
        raise ValueError(f'{name}: need an increasing, nonempty finite 1D grid'
                         + (' with positive values.' if positive else '.'))
    return axis.copy()


def scan_residuals(theta_values, j_targets, alpha_values, beta, lambda_sh,
                   recombination, geometry=None, *, D=0., ionization=(0., 0.),
                   excitation=0., inlet_momentum=None, n_points=100,
                   rtol=1e-9, atol=1e-11, max_step=.02, progress=None):
    """Evaluate three inlet residuals without running a root corrector.

    F = (log(j_a), log(mu_a / mu_target), omega_a - 1), exactly the
    residual convention of stationary.solve_stationary. R_* retain the
    unscaled inlet mismatches. An integrated IVP is NOT a verified BVP.
    Singleton axes are allowed for slices. alpha must be inside the open
    geometry interval. No assumed energy monotonicity, Q_eff sign or j_t<=1.
    progress(done, total) is called after each trial; its errors propagate.
    """
    theta = _axis(theta_values, 'theta')
    jt = _axis(j_targets, 'j_target')
    alpha = _axis(alpha_values, 'alpha', positive=False)
    geometry = quadratic_area_legacy() if geometry is None else geometry
    if np.any(alpha <= geometry.g_min) or np.any(alpha >= 1):
        raise ValueError('Require geometry.g_min < every alpha < 1.')
    if (not np.isfinite([beta, lambda_sh, rtol, atol, max_step]).all()
            or min(beta, lambda_sh, rtol, atol, max_step) <= 0):
        raise ValueError('Positive finite beta, lambda_sh and ODE controls required.')
    if not isinstance(n_points, (int, np.integer)) or n_points < 2:
        raise ValueError('n_points must be an integer >=2.')
    shape = (alpha.size, jt.size, theta.size)
    data = {key: np.full(shape, np.nan) for key in _FIELDS}
    valid = np.zeros(shape, dtype=bool)
    statuses, reasons = [], Counter()
    total = int(np.prod(shape))
    for ai, ji, ti in np.ndindex(shape):
        trial = solve_trial(float(theta[ti]), float(jt[ji]), float(alpha[ai]),
            beta, lambda_sh, recombination, geometry, D=D, ionization=ionization,
            excitation=excitation, inlet_momentum=inlet_momentum,
            n_points=n_points, rtol=rtol, atol=atol, max_step=max_step)
        status = trial['status']
        for key in ('D', 'wall_slope_per_j'):
            data[key][ai, ji, ti] = trial.get(key, np.nan)
        if 'reason' in trial:
            reasons[str(trial['reason'])] += 1
        if status == 'integrated':
            ja, ma, wa, target = (float(trial['j'][0]), float(trial['mu'][0]),
                                 float(trial['omega'][0]), float(trial['mu_target']))
            if np.isfinite([ja, ma, wa, target]).all() and min(ja, ma, target) > 0:
                with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
                    f = np.array([np.log(ja), np.log(ma/target), wa-1])
                if np.isfinite(f).all():
                    valid[ai, ji, ti] = True
                    values = (ja-1, ma-target, wa-1, *f, np.max(abs(f)),
                              ja, ma, wa, target)
                    for key, value in zip(_INLET_FIELDS, values):
                        data[key][ai, ji, ti] = value
                else:
                    status = 'invalid_inlet_residual'
            else:
                status = 'invalid_inlet_residual'
        statuses.append(str(status))
        if progress is not None:
            progress(len(statuses), total)
    # Allocate from the collected strings: no object dtype or silent truncation.
    status = np.asarray(statuses, dtype=str).reshape(shape)
    data.update(theta=theta, j_target=jt, alpha=alpha, status=status, valid=valid)
    data['settings'] = dict(beta=float(beta), lambda_sh=float(lambda_sh),
        geometry=geometry.to_dict(), n_points=int(n_points), rtol=float(rtol),
        atol=float(atol), max_step=float(max_step))
    summary = dict(n_trials=total, n_valid=int(valid.sum()),
        statuses=dict(Counter(statuses)), reason_counts=dict(reasons), warning=_WARNING,
        unresolved_numerical_count=sum(statuses.count(item) for item in (
            'integration_failed', 'critical_wall_unresolved', 'sonic_resolution_limited',
            'profile_validation_failed', 'invalid_inlet_residual')),
        nearest_sample_not_root=None)
    if valid.any():
        index = np.unravel_index(np.nanargmin(data['residual_norm']), shape)
        ai, ji, ti = index
        summary['nearest_sample_not_root'] = dict(theta=float(theta[ti]),
            j_target=float(jt[ji]), alpha=float(alpha[ai]),
            residual_norm=float(data['residual_norm'][index]),
            **{key: float(data[key][index]) for key in (*_RESIDUALS, 'R_J', 'R_M', 'R_E')})
    summary['candidate_cell_count'] = len(candidate_cells(data))
    data['summary'] = summary
    return data


def candidate_cells(scan):
    """Lower-corner (alpha,j,theta) indices of heuristic candidate cubes.

    All EIGHT corners must be valid, and each residual range must include
    zero. This is NOT a multidimensional root bracket: even parallel zero
    surfaces can pass. Tangent roots and narrow valid regions may be missed.
    """
    shape = scan['valid'].shape
    if min(shape) < 2:
        return []
    slices = [tuple(slice(d, n-1+d) for d, n in zip(offset, shape))
              for offset in np.ndindex(2, 2, 2)]
    eligible = np.logical_and.reduce([scan['valid'][s] for s in slices])
    for key in _RESIDUALS:
        corners = np.stack([scan[key][s] for s in slices])
        eligible &= (np.isfinite(corners).all(axis=0)
                     & (corners.min(axis=0) <= 0) & (corners.max(axis=0) >= 0))
    return [tuple(map(int, index)) for index in np.argwhere(eligible)]


def choose_starts(scan, max_starts=16, separation=2):
    """Choose valid sampled triples, not roots or interpolated cell centres.

    Priorities: candidate-cube best corners, local minima in 26-neighbourhoods,
    then low norms. separation is Chebyshev distance in GRID INDICES only,
    including alpha. It is not physical root deduplication or a completeness
    criterion. Use separation=1 to allow adjacent starts on a coarse grid.
    """
    if any(not isinstance(v, (int, np.integer)) or v < 1
           for v in (max_starts, separation)):
        raise ValueError('max_starts and separation must be positive integers.')
    norm, valid = scan['residual_norm'], scan['valid']
    ranked = []
    for cell in candidate_cells(scan):
        corners = [tuple(i+d for i, d in zip(cell, offset))
                   for offset in np.ndindex(2, 2, 2)]
        index = min(corners, key=lambda k: norm[k])
        ranked.append((0, float(norm[index]), index, 'candidate_cell_corner'))
    for raw in np.argwhere(valid):
        index = tuple(map(int, raw))
        region = tuple(slice(max(0, i-1), i+2) for i in index)
        local = norm[index] <= np.nanmin(norm[region])
        ranked.append((1 if local else 2, float(norm[index]), index,
                       'grid_local_minimum' if local else 'low_residual_sample'))
    starts = []
    selected_indices = []
    for _, score, index, source in sorted(ranked):
        if any(max(abs(i-j) for i, j in zip(index, old)) < separation
               for old in selected_indices):
            continue
        ai, ji, ti = index
        starts.append(dict(theta=float(scan['theta'][ti]),
            j_target=float(scan['j_target'][ji]), alpha=float(scan['alpha'][ai]),
            alpha_index=ai, j_index=ji, theta_index=ti, source=source,
            grid_residual_norm=score))
        selected_indices.append(index)
        if len(starts) == max_starts:
            break
    return starts


def project_residuals(scan, *, alpha_index=None):
    """A fixed-alpha slice, or a sampled minimum-norm projection over alpha.

    Projection selects ONE alpha at each (j,theta) sample using the FULL
    three-residual norm; all displayed residuals come from that same trial.
    Ties select the lowest alpha index. No interpolation/minimisation solver.
    This is not elimination via omega=1, nor a root/basin-of-attraction map.
    Invalid columns retain index -1 and NaNs, never a fictitious minimum.
    """
    valid = scan['valid']
    shape = valid.shape[1:]
    if alpha_index is None:
        selected = np.argmin(np.where(valid, scan['residual_norm'], np.inf), axis=0)
        mask = valid.any(axis=0)
        mode = 'minimum_over_sampled_alpha'
    else:
        if (not isinstance(alpha_index, (int, np.integer))
                or not 0 <= alpha_index < valid.shape[0]):
            raise ValueError('alpha_index must index the alpha grid.')
        selected = np.full(shape, alpha_index, dtype=int)
        mask = valid[alpha_index].copy()
        mode = 'fixed_alpha_slice'
    jj, tt = np.indices(shape)
    result = {key: np.where(mask, scan[key][selected, jj, tt], np.nan)
              for key in _FIELDS}
    result.update(theta=scan['theta'].copy(), j_target=scan['j_target'].copy(),
        alpha=np.where(mask, scan['alpha'][selected], np.nan), valid=mask,
        alpha_index=np.where(mask, selected, -1),
        valid_alpha_count=valid.sum(axis=0), mode=mode,
        fixed_alpha=None if alpha_index is None else float(scan['alpha'][alpha_index]))
    return result


def save_residuals(path, scan, *, metadata=None):
    """Save the full cube as NPZ readable with allow_pickle=False.

    Never overwrites a file. Parent directory must already exist. Optional
    JSON metadata should identify the config, coefficient data and closures:
    settings alone do NOT fully describe callable coefficient providers.
    """
    arrays = {key: scan[key] for key in (*_FIELDS, 'theta', 'j_target', 'alpha',
                                        'status', 'valid')}
    payload = dict(arrays, format_version=np.array(1),
        settings_json=np.array(json.dumps(scan['settings'], allow_nan=False)),
        summary_json=np.array(json.dumps(scan['summary'], allow_nan=False)),
        metadata_json=np.array(json.dumps({} if metadata is None else metadata, allow_nan=False)))
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **payload)


def plot_residuals(scan, T_star_eV=1., *, alpha_index=None, starts=(),
                   solutions=(), title=None, j_axis='log'):
    """Six panels, no interpolation or zero contours across alpha switches.

    Returns (figure, axes[2,3]); never shows/saves a window on its own.
    Projection markers discard alpha visually; fixed slices show only markers
    at that alpha. solutions must explicitly have stationary_verified=True.
    Grey denotes undefined trials, not a physically excluded parameter region.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, SymLogNorm
    from matplotlib.lines import Line2D

    if not np.isfinite(T_star_eV) or T_star_eV <= 0 or j_axis not in ('log', 'linear', 'loss'):
        raise ValueError('Need T_star_eV>0 and j_axis=log,linear,loss.')
    starts, solutions = list(starts), list(solutions)
    if any(not root.get('stationary_verified', False) for root in solutions):
        raise ValueError('Only stationary_verified solutions may be marked as roots.')
    view = project_residuals(scan, alpha_index=alpha_index)
    x = view['theta']*T_star_eV
    y = 1-view['j_target'] if j_axis == 'loss' else view['j_target']
    order = np.argsort(y)
    y = y[order]
    xx, yy = np.meshgrid(x, y)
    mask = view['valid'][order]
    signed = SymLogNorm(linthresh=1e-4, vmin=-1, vmax=1)
    a_min, a_max = float(scan['alpha'][0]), float(scan['alpha'][-1])
    if a_min == a_max:
        a_min, a_max = a_min-.001, a_max+.001
    with np.errstate(invalid='ignore', divide='ignore'):
        lognorm = np.log10(np.maximum(view['residual_norm'], 1e-12))
    panels = [(lognorm, r'$\log_{10}\rho$, $\rho=\max_i|F_i|$', Normalize(-6, 1), 'viridis_r'),
        (view['F_J'], r'$F_J=\ln j_a$', signed, 'coolwarm'),
        (view['F_M'], r'$F_M=\ln(\mu_a/\mu_{in})$', signed, 'coolwarm'),
        (view['F_E'], r'$F_E=\omega_a-1$', signed, 'coolwarm'),
        (view['alpha'], r'Selected $\alpha$', Normalize(a_min, a_max), 'viridis'),
        (view['valid_alpha_count'], 'Valid alpha samples (entire cube)',
         Normalize(0, max(1, len(scan['alpha']))), 'cividis')]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True,
                             layout='constrained')
    for index, (axis, (values, label, normalization, cmap)) in enumerate(zip(axes.flat, panels)):
        axis.set_facecolor('#b0b0b0')
        # The count panel intentionally includes zero-valid columns.
        values = np.ma.masked_where(~mask, values[order]) if index < 5 else values[order]
        if len(x) > 1 and len(y) > 1:
            artist = axis.pcolormesh(xx, yy, values, shading='nearest', norm=normalization, cmap=cmap)
        else:
            artist = axis.scatter(xx.ravel(), yy.ravel(), c=values.ravel(),
                                  s=90, marker='s', norm=normalization, cmap=cmap)
        fig.colorbar(artist, ax=axis, shrink=.8)
        axis.set_xscale('log')
        if j_axis == 'loss':
            axis.set_yscale('symlog', linthresh=1e-8)
        else:
            axis.set_yscale(j_axis)
        axis.set_title(label, fontsize=10)
        axis.set_xlabel('T [eV]')
        axis.set_ylabel('1 - j_t' if j_axis == 'loss' else 'j_t')
        axis.grid(alpha=.15)
        axis.set_xlim((x[0], x[-1]) if len(x) > 1 else (x[0]/1.1, x[0]*1.1))
        if len(y) > 1:
            axis.set_ylim(y[0], y[-1])
        elif j_axis == 'log':
            axis.set_ylim(y[0]/1.1, y[0]*1.1)
        else:
            pad = .05*max(1., abs(y[0]))
            axis.set_ylim(y[0]-pad, y[0]+pad)
        for points, marker, color, size in ((starts, 'o', '#db2777', 40),
                                             (solutions, '*', '#22c55e', 140)):
            visible = [p for p in points if alpha_index is None or np.isclose(
                p['alpha'], view['fixed_alpha'], rtol=0., atol=1e-12)]
            if visible:
                yp = np.array([p['j_target'] for p in visible])
                axis.scatter([p['theta']*T_star_eV for p in visible],
                    1-yp if j_axis == 'loss' else yp, marker=marker, s=size,
                    facecolors='none' if marker == 'o' else color,
                    edgecolors=color if marker == 'o' else 'black', linewidths=.9, zorder=4)
        if index < 5 and not mask.any():
            axis.text(.5, .5, 'No valid trials', transform=axis.transAxes,
                      ha='center', va='center')
    legend = []
    if starts:
        legend.append(Line2D([], [], marker='o', color='#db2777', markerfacecolor='none',
                             linestyle='', label='Selected starts (not roots)'))
    if solutions:
        legend.append(Line2D([], [], marker='*', color='#22c55e', markeredgecolor='black',
                             linestyle='', label='Verified roots'))
    if legend:
        axes[0, 0].legend(handles=legend, fontsize=8)
    mode = ('Minimum FULL residual norm over sampled alpha; markers projected to (T,j_t)'
            if alpha_index is None else f"Fixed alpha = {view['fixed_alpha']:.6g}")
    fig.suptitle((title+'\n' if title else '')+mode+'\n'
                 'Grey: undefined trials. Finite grid; no root/completeness guarantee.', fontsize=10)
    return fig, axes
