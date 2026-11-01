"""Local three-parameter wall shooting; no first-energy-crossing assumption.

Unknowns: theta,j_target,alpha. Residuals: j(a)-1, mu(a)-G(alpha), w(a)-1.
Safeguarded Newton uses only admissible IVPs, never artificial penalties.
This is a local algorithm, not a completeness/uniqueness/stability proof.
"""

from collections import Counter
import numpy as np
from scipy.special import expit, logit
from .geometry import quadratic_area_legacy
from .transport import solve_trial, balance_checks


def solve_stationary(theta_guess, j_guess, alpha_guess, beta, lambda_sh,
                     recombination, geometry=None, *, inlet_momentum=None,
                     theta_bounds=None, j_bounds=(1e-12, 1.),
                     max_iterations=40, residual_tol=1e-9, balance_tol=5e-6,
                     n_points=4000, rtol=1e-9, atol=1e-11, max_step=.02):
    """Return a verified solution only if all BCs AND budgets pass.

    j_target<=1 follows from recombination only and j(a)=1; the numerical
    j_bounds may be widened for off-root exploration or future processes.
    theta_bounds should be the intersection of active coefficient domains.
    The caller must explicitly choose a new alpha start for multiple roots.
    """
    geometry = quadratic_area_legacy() if geometry is None else geometry
    values = (theta_guess, j_guess, beta, lambda_sh, residual_tol, balance_tol,
              rtol, atol, max_step)
    if not np.isfinite(values).all() or min(values) <= 0:
        raise ValueError('Positive finite inputs required.')
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError('max_iterations must be an integer >=1.')
    if not isinstance(n_points, int) or n_points < 100:
        raise ValueError('n_points must be an integer >=100.')
    if not np.isfinite(alpha_guess) or not geometry.g_min < alpha_guess < 1:
        raise ValueError('Initial alpha outside the open geometry interval.')
    tb = (np.finfo(float).tiny, np.finfo(float).max) if theta_bounds is None else theta_bounds
    tb, jb = np.asarray(tb, float), np.asarray(j_bounds, float)
    if any(bound.shape != (2,) or not np.isfinite(bound).all()
           or not 0 < bound[0] < bound[1] for bound in (tb, jb)):
        raise ValueError('Ordered positive finite theta/j bounds required.')
    lower = np.array([np.log(tb[0]), np.log(jb[0]), -np.inf])
    upper = np.array([np.log(tb[1]), np.log(jb[1]), np.inf])
    x = np.array([np.log(theta_guess), np.log(j_guess),
                  logit((alpha_guess-geometry.g_min)/(1-geometry.g_min))])
    invalid = Counter()
    diagnostic = dict(evaluations=0, history=[], initial_guess=[
        float(theta_guess), float(j_guess), float(alpha_guess)])

    def unpack(point):
        theta, jt = np.exp(point[:2])
        alpha = geometry.g_min+(1-geometry.g_min)*expit(point[2])
        return float(theta), float(jt), float(alpha)

    def finish(status, reason=None, solution=None):
        diagnostic['invalid_trials'] = dict(invalid)
        diagnostic['last_parameters'] = dict(zip(('theta', 'j_target', 'alpha'), unpack(x)))
        result = dict(status=status, diagnostic=diagnostic)
        if reason is not None:
            result['reason'] = str(reason)
        if solution is not None:
            result['solution'] = solution
        return result

    def trial(point, settings, points=100):
        if not np.isfinite(point).all() or np.any(point < lower) or np.any(point > upper):
            return None
        theta, jt, alpha = unpack(point)
        if not geometry.g_min < alpha < 1:
            return None
        state = solve_trial(theta, jt, alpha, beta, lambda_sh, recombination,
            geometry, inlet_momentum=inlet_momentum, n_points=points, **settings)
        diagnostic['evaluations'] += 1
        if state['status'] != 'integrated':
            invalid[state['status']] += 1
            return None
        # Log residuals scale the two positive flow conditions. Energy flux
        # is signed, so its residual is linear, never log-transformed.
        f = np.array([np.log(state['j'][0]),
                      np.log(state['mu'][0]/state['mu_target']), state['R_E']])
        if not np.isfinite(f).all():
            invalid['nonfinite_residual'] += 1
            return None
        return f, state

    def correct(start, settings, tolerance, phase):
        point = start.copy()
        cache = {}

        def evaluate(y):
            key = tuple(y)
            if key not in cache:
                cache[key] = trial(y, settings)
            return cache[key]

        for iteration in range(max_iterations+1):
            current = evaluate(point)
            if current is None:
                return point, 'invalid_initial_trial'
            f, _ = current
            norm = float(np.max(abs(f)))
            diagnostic['history'].append(dict(phase=phase, iteration=iteration,
                norm=norm, parameters=list(unpack(point))))
            if norm <= tolerance:
                return point, 'corrected'
            if iteration == max_iterations:
                return point, 'iteration_limit'
            jac = np.empty((3, 3))
            for column in range(3):
                h = 1e-4*max(1., abs(point[column]))
                derivative = None
                for _ in range(8):
                    step = np.zeros(3)
                    step[column] = h
                    plus, minus = evaluate(point+step), evaluate(point-step)
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
                    return point, 'jacobian_unavailable'
                jac[:, column] = derivative
            condition = float(np.linalg.cond(jac))
            if not np.isfinite(condition) or condition > 1e12:
                return point, 'ill_conditioned_jacobian'
            accepted = False
            for direction in (np.linalg.solve(jac, -f), -jac.T@f):
                length = float(np.max(abs(direction)))
                if not np.isfinite(length) or length == 0:
                    continue
                direction /= max(1., length)
                damping = 1.
                for _ in range(24):
                    proposal = np.clip(point+damping*direction, lower, upper)
                    candidate = evaluate(proposal)
                    if candidate is not None:
                        next_norm = float(np.max(abs(candidate[0])))
                        if next_norm <= tolerance or next_norm < norm*(1-1e-4*damping):
                            point, accepted = proposal, True
                            break
                    damping *= .5
                if accepted:
                    break
            if not accepted:
                return point, 'line_search_failed'
        raise AssertionError('Unreachable Newton state.')

    if np.any(x < lower) or np.any(x > upper):
        return finish('initial_outside_search_bounds')
    settings = dict(rtol=rtol, atol=atol, max_step=max_step)
    x, status = correct(x, settings, residual_tol, 'search')
    if status != 'corrected':
        return finish(status)
    tight = dict(rtol=min(1e-11, rtol*.01), atol=min(1e-13, atol*.01), max_step=max_step*.5)
    x, status = correct(x, tight, min(1e-11, residual_tol*.1), 'verification')
    if status != 'corrected':
        return finish('verification_'+status)
    checked = trial(x, tight, points=n_points)
    if checked is None:
        return finish('verification_failed')
    _, state = checked
    if (abs(state['R_J']) > residual_tol or abs(state['R_E']) > residual_tol
            or abs(state['R_M']) > residual_tol*(1+abs(state['mu_target']))):
        return finish('verification_failed', 'Inlet BC mismatch.')
    balances = balance_checks(state, recombination)
    if max(balances[k] for k in ('particle_abs', 'momentum_abs', 'energy_abs')) > balance_tol:
        return finish('balance_check_failed', 'Refine output sampling/ODE accuracy.')
    state.update(stationary_verified=True, balances=balances,
                 verification=dict(residual_tol=residual_tol, balance_tol=balance_tol,
                                   n_points=n_points, **tight))
    return finish('converged', solution=state)
