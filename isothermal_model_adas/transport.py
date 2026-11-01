"""Backward wall shooting on fixed xi in [0,1], with alpha an input.

Unlike the old first-omega=1 event, this integrates to a prescribed alpha.
The stationary solver varies theta,j_t,alpha to satisfy all three inlet
conditions. Signed Q_eff and nonmonotone omega need no special branch rule.
"""

from math import isfinite, sqrt
import numpy as np
from scipy.integrate import solve_ivp, simpson
from .geometry import quadratic_area_legacy
from .inlet import evaluate_inlet, KineticInletDomainError
from .recombination_adas import ADASDomainError
from .sources import coefficients_at, adas_sources


class StateDomainError(ValueError):
    """A provisional IVP state cannot be reconstructed as positive plasma."""


def reconstruct(g, state, theta, beta, geometry):
    j, mu = state[:2]
    b = sqrt(8*beta*theta)
    # Clipping ONLY permits provisional RK stages around the sonic endpoint.
    # Accepted profiles are checked independently below.
    radical = np.sqrt(np.maximum((mu-b*j)*(mu+b*j), 0.))
    H = mu+radical
    area = geometry.A_ratio(g)
    if np.any(np.asarray(j) <= 0) or np.any(H <= 0):
        raise StateDomainError('Nonpositive particle flux or reconstructed H.')
    N = H/(4*beta*theta*area)
    v = 4*beta*theta*j/H
    if not np.isfinite(N).all() or not np.isfinite(v).all():
        raise StateDomainError('Nonfinite reconstructed plasma state.')
    return H, area, N, v


def cx_coefficient(D, theta):
    """Evaluate a nonnegative scalar D or D(theta), once per isothermal trial."""
    value = float(D(theta) if callable(D) else D)
    if not isfinite(value) or value < 0:
        raise ValueError('D must give one finite nonnegative scalar.')
    return value


def wall_slope_per_j(theta, jt, beta, R_t, D, geometry):
    """F_g(1)/j_t for F=mu-sqrt(8*beta*theta)*j; coefficients already evaluated."""
    b = sqrt(8*beta*theta)
    geom = -.5*b*float(geometry.dlnB_dg(1.))
    rec = R_t*b*jt/(4*beta*theta*float(geometry.A_ratio(1.)))
    slope = geom-D+rec
    if not isfinite(slope):
        raise ValueError('Unrepresentable wall derivative.')
    tolerance = 64*np.finfo(float).eps*max(1., abs(geom), D, abs(rec))
    return slope, tolerance


def rhs_g(g, state, *, theta, beta, recombination, geometry, D=0.):
    """Right-hand side per g. D is the frozen nonnegative CX coefficient."""
    H, area, N, v = reconstruct(g, state, theta, beta, geometry)
    R, Q = coefficients_at(recombination, theta, N)
    particles, momentum, energy = adas_sources(state[0], N, v, area,
        theta=theta, beta=beta, R=R, Q_eff=Q, D=D)
    return np.array([particles, -.5*geometry.dlnB_dg(g)*H+momentum, energy])


def solve_trial(theta, j_target, alpha, beta, lambda_sh, recombination,
                geometry=None, *, D=0., inlet_momentum=None, n_points=1200,
                rtol=1e-9, atol=1e-11, max_step=.02):
    """An IVP, NOT a BVP solution. Return status and all three inlet residuals.

    No energy monotonicity, wall_energy<1, or positive-Q assumption.
    max_step is in xi. ADAS domain failures are explicit invalid trials,
    not proof of physical nonexistence. Malformed data/program errors raise.
    D is a scalar or D(theta), evaluated once because the layer is isothermal.
    Default D=0 retains the former recombination-only equations.
    """
    theta, jt, alpha, beta, lam, rtol, atol, max_step = map(float,
        (theta, j_target, alpha, beta, lambda_sh, rtol, atol, max_step))
    if not all(isfinite(x) and x > 0 for x in (theta, jt, beta, lam, rtol, atol, max_step)):
        raise ValueError('Positive finite theta,j_target,beta,lambda,tolerances required.')
    if not isinstance(n_points, (int, np.integer)) or n_points < 2:
        raise ValueError('n_points must be an integer >=2.')
    geometry = quadratic_area_legacy() if geometry is None else geometry
    if not isfinite(alpha) or not geometry.g_min < alpha < 1:
        raise ValueError('Require geometry.g_min < alpha < 1.')
    b = sqrt(8*beta*theta)
    wall = np.array([jt, b*jt, lam*theta*jt])
    if not np.isfinite(wall).all():
        raise ValueError('Unrepresentable wall state.')
    result = dict(model='isothermal_adas', theta=theta, j_target=jt, alpha=alpha,
                  beta=beta, b=b, geometry=geometry, lambda_sh=lam,
                  R_J=np.nan, R_M=np.nan, R_E=np.nan)

    def stop(status, reason=None):
        result['status'] = status
        if reason is not None:
            result['reason'] = str(reason)
        return result

    # F=mu-b*j >=0. A backward subsonic launch requires F_g(1)<0.
    wall_area = float(geometry.A_ratio(1.))
    N_t = b*jt/(4*beta*theta*wall_area)
    try:
        D_value = cx_coefficient(D, theta)
        R_t, Q_t = coefficients_at(recombination, theta, N_t)
    except ADASDomainError as error:
        return stop('adas_outside_domain', error)
    slope, tolerance = wall_slope_per_j(theta, jt, beta, R_t, D_value, geometry)
    result.update(D=D_value, wall_slope_per_j=slope, R_wall=R_t, Q_eff_wall=Q_t,
                  N_wall=N_t)
    if slope > tolerance:
        return stop('inadmissible_wall', 'Geometry, recombination and CX cannot launch a backward subsonic branch.')
    if abs(slope) <= tolerance:
        return stop('critical_wall_unresolved', 'First wall derivative is zero within numerical tolerance.')

    length = 1-alpha

    def rhs(xi, state):
        return length*rhs_g(alpha+length*xi, state, theta=theta, beta=beta,
                            recombination=recombination, geometry=geometry, D=D_value)

    def sonic_return(xi, state):
        return state[1]-b*state[0]
    sonic_return.terminal = True
    sonic_return.direction = -1

    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            integration = solve_ivp(rhs, (1., 0.), wall, method='DOP853',
                events=sonic_return, dense_output=True, rtol=rtol, atol=atol,
                max_step=max_step, first_step=min(1e-4, max_step))
    except ADASDomainError as error:
        return stop('adas_outside_domain', error)
    except StateDomainError as error:
        return stop('state_outside_domain', error)
    except (FloatingPointError, OverflowError) as error:
        return stop('integration_failed', error)
    result.update(ode_steps=len(integration.t), xi_stop=float(integration.t[-1]))
    if not integration.success:
        return stop('integration_failed', integration.message)
    if integration.t_events[0].size:
        return stop('sonic_return')
    xi = np.linspace(0., 1., n_points)
    g = alpha+length*xi
    j, mu, omega = integration.sol(xi)
    if not np.isfinite([j, mu, omega]).all() or np.any(j <= 0):
        return stop('profile_validation_failed', 'Nonfinite or nonpositive particle profile.')
    gap = mu-b*j
    tol = 50*(atol+rtol*(1+float(np.max(abs(mu)))))
    if np.any(gap[:-1] <= 0) or abs(gap[-1]) > tol:
        return stop('sonic_resolution_limited')
    # Recombination removes particles; CX has zero net particle source.
    if np.any(np.diff(j) > 50*(atol+rtol*(1+float(np.max(j))))):
        return stop('profile_validation_failed', 'Particle flux increases without ionization.')
    if not np.allclose([j[-1], mu[-1], omega[-1]], wall, rtol=10*rtol, atol=10*atol):
        return stop('profile_validation_failed', 'Wall boundary mismatch.')
    try:
        H, area, N, v = reconstruct(g, (j, mu, omega), theta, beta, geometry)
        R, Q = coefficients_at(recombination, theta, N)
        inlet = evaluate_inlet(alpha, inlet_momentum)
    except ADASDomainError as error:
        return stop('adas_outside_domain', error)
    except KineticInletDomainError as error:
        return stop('kinetic_inlet_outside_domain', error)
    result.update(xi=xi, g=g, j=j, mu=mu, omega=omega, H=H,
        A_over_A_star=area, n_over_n_star=N, u_over_u_star=v,
        u_over_cs=2*v/b, R_profile=R, Q_eff_profile=Q,
        Aq_e_over_W0=omega-j*(v*v/(2*beta)+5*theta),
        R_J=float(j[0]-1), R_M=float(mu[0]-inlet['mu_target']),
        R_E=float(omega[0]-1), **inlet)
    return stop('integrated')


def balance_checks(state, recombination):
    """Independent dimensional-form budgets, not a call to adas_sources.

    Re-evaluates local coefficients; no frozen wall/inlet values. W+I*J
    conservation offers an additional photon/chemical-energy audit in tests.
    """
    g, j, mu, w, N, v, A, H = (np.asarray(state[k]) for k in
        ('g', 'j', 'mu', 'omega', 'n_over_n_star', 'u_over_u_star', 'A_over_A_star', 'H'))
    theta, beta = state['theta'], state['beta']
    R, Q = coefficients_at(recombination, theta, N)
    removed = R*A*N*N
    D = cx_coefficient(state.get('D', 0.), theta)
    momentum = -.5*state['geometry'].dlnB_dg(g)*H-D*j-R*j*N
    ion_energy = v*v/(2*beta)+1.5*theta
    ion_loss, effective_loss = removed*ion_energy, Q*A*N*N
    cx_loss = D*A*N*ion_energy
    lost = float(simpson(removed, x=g))
    energy_loss = float(simpson(cx_loss+ion_loss+effective_loss, x=g))
    return dict(particle_abs=abs(float(j[-1]-j[0])+lost),
        momentum_abs=abs(float(mu[-1]-mu[0])-float(simpson(momentum, x=g))),
        energy_abs=abs(float(w[0]-w[-1])-energy_loss),
        particle_recombination_integral=lost, energy_loss_integral=energy_loss,
        ion_energy_loss_integral=float(simpson(ion_loss, x=g)),
        effective_energy_loss_integral=float(simpson(effective_loss, x=g)),
        cx_energy_loss_integral=float(simpson(cx_loss, x=g)),
        cx_momentum_loss_integral=float(simpson(D*j, x=g)))
