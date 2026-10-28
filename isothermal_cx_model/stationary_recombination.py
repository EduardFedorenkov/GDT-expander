"""Local and multistart stationary RR shooting. No claim of global completeness.

The checked IVP is unchanged. Unknowns are log(theta), log(j_t), not logit(j_t):
the second derivative stays resolvable near j_t=1, including the zero-RR limit.
Invalid IVPs never become artificial residuals or optimizer penalties.
"""

from collections import Counter
import numpy as np

from .geometry import quadratic_area_legacy
from .solver_atomic import solve_trial, balance_checks


def solve_stationary(theta_guess, j_guess, beta, D, lambda_sh, geometry=None,
                     *, E_exc=0., rr=(0., 0.), inlet_momentum=None,
                     theta_bounds=None, j_min=1e-12, residual_tol=1e-9,
                     balance_tol=5e-6, max_iterations=40, n_points=4000,
                     rtol=1e-9, atol=1e-11, max_step=.02):
    """Safeguarded Newton in a valid IVP domain, followed by tighter correction.

    A result has status='converged' and a 'solution' ONLY after re-correction,
    a new high-resolution IVP, both inlet checks, sonic admissibility and
    integral balances. Failures report diagnostics, never 'no physical root'.
    theta_bounds and j_min are numerical search windows, not physical cutoffs.
    Input/programming errors still raise exceptions.
    """
    geometry = quadratic_area_legacy() if geometry is None else geometry
    values = np.asarray([theta_guess, j_guess, beta, lambda_sh, residual_tol,
                         balance_tol, rtol, atol, max_step], dtype=float)
    if (not np.isfinite(values).all() or np.any(values <= 0)
            or not 0 < j_min < 1 or j_guess > 1):
        raise ValueError('Positive finite inputs required, 0<j_min<1 and 0<j_guess<=1.')
    if (not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1
            or not isinstance(n_points, (int, np.integer)) or n_points < 100):
        raise ValueError('Need integer max_iterations>=1 and n_points>=100.')
    if theta_bounds is None:
        lower_theta, upper_theta = np.finfo(float).tiny, np.finfo(float).max
    else:
        lower_theta, upper_theta = map(float, theta_bounds)
        if (not np.isfinite([lower_theta, upper_theta]).all()
                or not 0 < lower_theta < upper_theta):
            raise ValueError('theta_bounds must be ordered positive finite values.')
    lower = np.log([lower_theta, j_min])
    upper = np.log([upper_theta, 1.])
    x = np.log([theta_guess, j_guess])
    diagnostics = dict(initial_guess=[float(theta_guess), float(j_guess)],
                       evaluations=0, invalid_trials={}, history=[])
    invalid = Counter()

    def finish(status, reason=None):
        diagnostics['invalid_trials'] = dict(invalid)
        diagnostics['last_parameters'] = dict(theta=float(np.exp(x[0])),
                                               j_target=float(np.exp(x[1])))
        if reason is not None:
            diagnostics['reason'] = str(reason)
        return dict(status=status, diagnostics=diagnostics)

    if np.any(x < lower) or np.any(x > upper):
        return finish('initial_outside_search_bounds')

    def correct(start, settings, tolerance, phase):
        current = start.copy()
        cache = {}

        def evaluate(point):
            if (not np.isfinite(point).all() or np.any(point < lower)
                    or np.any(point > upper)):
                return None
            key = tuple(point)
            if key not in cache:
                theta, jt = np.exp(point)
                state = solve_trial(theta, jt, beta, D, lambda_sh, geometry,
                    E_exc=E_exc, rr=rr, inlet_momentum=inlet_momentum,
                    n_points=100, **settings)
                diagnostics['evaluations'] += 1
                if state['status'] != 'energy_matched':
                    invalid[state['status']] += 1
                    cache[key] = None
                else:
                    f = np.array([np.log(state['j_a']),
                                  np.log(state['mu_a']/state['mu_target'])])
                    if not np.isfinite(f).all():
                        invalid['nonfinite_log_residual'] += 1
                        cache[key] = None
                    else:
                        cache[key] = (f, state)
            return cache[key]

        for iteration in range(max_iterations+1):
            evaluated = evaluate(current)
            if evaluated is None:
                return current, 'invalid_initial_trial', None
            f, state = evaluated
            norm = float(np.max(np.abs(f)))
            diagnostics['history'].append(dict(phase=phase, iteration=iteration,
                theta=float(np.exp(current[0])), j_target=float(np.exp(current[1])),
                norm=norm, R_J=state['R_J'], R_M=state['R_M']))
            if norm <= tolerance:
                return current, 'corrected', state
            if iteration == max_iterations:
                return current, 'iteration_limit', state
            jacobian = np.empty((2, 2))
            for column in range(2):
                h = 1e-4*max(1., abs(current[column]))
                derivative = None
                for _ in range(6):
                    shift = np.zeros(2)
                    shift[column] = h
                    plus, minus = evaluate(current+shift), evaluate(current-shift)
                    if plus is not None and minus is not None:
                        derivative = (plus[0]-minus[0])/(2*h)
                    elif plus is not None:
                        derivative = (plus[0]-f)/h
                    elif minus is not None:
                        derivative = (f-minus[0])/h
                    if derivative is not None:
                        break
                    h *= .5
                if derivative is None:
                    return current, 'jacobian_unavailable', state
                jacobian[:, column] = derivative
            condition = float(np.linalg.cond(jacobian))
            diagnostics['last_jacobian_condition'] = condition if np.isfinite(condition) else None
            diagnostics['last_jacobian'] = jacobian.tolist()
            if not np.isfinite(condition) or condition > 1e12:
                return current, 'ill_conditioned_jacobian', state
            directions = [np.linalg.solve(jacobian, -f), -jacobian.T @ f]
            accepted = False
            for direction in directions:
                length = float(np.max(np.abs(direction)))
                if not np.isfinite(length) or length == 0:
                    continue
                direction = direction/max(1., length)  # Trust radius in log coordinates.
                damping = 1.
                for _ in range(24):
                    proposal = np.clip(current+damping*direction, lower, upper)
                    if np.array_equal(proposal, current):
                        damping *= .5
                        continue
                    candidate = evaluate(proposal)
                    if candidate is not None:
                        next_norm = float(np.max(np.abs(candidate[0])))
                        if next_norm <= tolerance or next_norm < norm*(1-1e-4*damping):
                            current = proposal
                            accepted = True
                            break
                    damping *= .5
                if accepted:
                    break
            if not accepted:
                return current, 'line_search_failed', state
        raise AssertionError('Unreachable correction state.')

    coarse = dict(rtol=rtol, atol=atol, max_step=max_step)
    x, status, initial = correct(x, coarse, residual_tol, 'search')
    if status != 'corrected':
        return finish(status)
    coarse_parameters = np.exp(x)
    tight = dict(rtol=min(1e-11, rtol*.01), atol=min(1e-13, atol*.01),
                 max_step=max_step*.5)
    verified_tol = min(1e-11, residual_tol*.1)
    x, status, _ = correct(x, tight, verified_tol, 'verification')
    if status != 'corrected':
        return finish('verification_'+status)
    theta, jt = np.exp(x)
    checked = solve_trial(theta, jt, beta, D, lambda_sh, geometry,
        E_exc=E_exc, rr=rr, inlet_momentum=inlet_momentum,
        n_points=n_points, **tight)
    diagnostics['evaluations'] += 1
    if checked['status'] != 'energy_matched':
        return finish('verification_failed', checked['status'])
    if (abs(checked['R_J']) > residual_tol
            or abs(checked['R_M']) > residual_tol*(1+abs(checked['mu_target']))
            or abs(checked['R_E']) > residual_tol):
        return finish('verification_failed', 'Refined inlet residuals exceed tolerance.')
    if checked['inlet_target_sonic_margin'] <= 0:
        return finish('inlet_not_strictly_subsonic')
    balances = balance_checks(checked, beta)
    diagnostics['balances'] = balances
    if max(balances[k] for k in ('particle_abs', 'momentum_abs', 'energy_abs')) > balance_tol:
        return finish('balance_check_failed', 'Increase output sampling and/or ODE accuracy.')
    checked.update(stationary_verified=True, beta=float(beta),
        verification=dict(residual_tol=residual_tol, refined_tol=verified_tol,
            balance_tol=balance_tol, n_points=n_points, **tight),
        balances=balances, delta_theta=float(theta-coarse_parameters[0]),
        delta_j_target=float(jt-coarse_parameters[1]))
    outcome = finish('converged')
    outcome['solution'] = checked
    return outcome


def find_stationary_solutions(guesses, beta, D, lambda_sh, geometry=None,
                              *, dedup_tol=1e-8, **kwargs):
    """Verified multistart local search. Guesses are (theta,j_t), NOT eV.

    Returns (solutions, diagnostic). No ordering or identification by the
    'monotone' label. A failed/empty local search is not a nonexistence proof.
    """
    guesses = np.asarray(guesses, dtype=float)
    if (guesses.ndim != 2 or guesses.shape[1] != 2 or guesses.shape[0] == 0
            or not np.isfinite(guesses).all() or np.any(guesses <= 0)
            or np.any(guesses[:, 1] > 1) or not np.isfinite(dedup_tol) or dedup_tol <= 0):
        raise ValueError('Need nonempty positive pairs (theta,j_t), j_t<=1 and dedup_tol>0.')
    solutions, attempts = [], []
    duplicates = 0
    for guess in guesses:
        outcome = solve_stationary(*guess, beta, D, lambda_sh, geometry, **kwargs)
        attempts.append(dict(status=outcome['status'], **outcome['diagnostics']))
        if outcome['status'] != 'converged':
            continue
        state = outcome['solution']
        point = np.log([state['theta'], state['j_target']])
        if any(np.max(np.abs(point-np.log([old['theta'], old['j_target']]))) <= dedup_tol
               and abs(state['alpha']-old['alpha']) <= dedup_tol for old in solutions):
            duplicates += 1
        else:
            solutions.append(state)
    solutions.sort(key=lambda state: (state['theta'], state['j_target']))
    for index, state in enumerate(solutions, start=1):
        state['solution_id'] = f'root_{index:03d}'
    return solutions, dict(attempts=attempts, duplicate_convergences=duplicates,
        n_solutions=len(solutions),
        warning='Local/multistart results only; not a proof of completeness, uniqueness or stability.')
