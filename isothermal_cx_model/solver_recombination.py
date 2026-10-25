"""One isothermal (theta, j_target) trial. No parameter search is performed.

Integrate j, mu, omega in g=z/L from the sonic plate to omega=1. The event
defines alpha=a/L. Both inlet residuals remain unchecked trial outputs.
Case-A RR; prescribed neutrals; no ionization/three-body recombination.
"""

from math import isfinite, sqrt
import numpy as np
from scipy.integrate import solve_ivp

from .geometry import quadratic_area_legacy
from .inlet import evaluate_inlet, KineticInletDomainError
from .recombination import RRTemperatureDomainError
from .solver_general import classify_profile


def _nonnegative(value, theta, name):
    value = float(value(theta) if callable(value) else value)
    if not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return value


def solve_trial(theta, j_target, beta, D, lambda_sh, geometry=None,
                n_points=1600, *, E_exc=0.0, rr=(0.0, 0.0),
                inlet_momentum=None, rtol=1e-9, atol=1e-11, max_step=0.02):
    """Return a status dictionary; arrays are present after energy matching.

    D and E_exc: numbers or functions of theta. rr: a pair (R,Q), or one
    function theta -> (R,Q), e.g. make_rr_coefficients(parameters, scale=1).
    R,Q include their common RR multiplier already. They are evaluated once.

    status='energy_matched' means ONLY omega(alpha)=1. A global solution
    additionally needs R_J=0 and R_M=0. j(alpha)>1 and a negative inlet
    target sonic margin remain valid off-root residuals, never clipped.

    Input/programming errors raise ValueError. Numerical/physical stopping
    conditions return explicit statuses with NaN residuals where undefined.
    Integration failure is not evidence that no physical solution exists.
    """
    theta, j_target, beta, lambda_sh, rtol, atol, max_step = map(
        float, (theta, j_target, beta, lambda_sh, rtol, atol, max_step))
    if (not all(isfinite(v) and v > 0 for v in
                (theta, beta, lambda_sh, rtol, atol, max_step))
            or not isfinite(j_target) or not 0 < j_target <= 1):
        raise ValueError("Need positive finite theta,beta,lambda,tolerances; 0<j_target<=1.")
    if not isinstance(n_points, (int, np.integer)) or n_points < 2:
        raise ValueError("n_points must be an integer >=2.")
    geometry = quadratic_area_legacy() if geometry is None else geometry
    result = dict(
        model='rr_trial', theta=theta, j_target=j_target, geometry=geometry,
        status=None, alpha=np.nan, R_J=np.nan, R_M=np.nan, R_E=np.nan,
    )

    def stop(status, reason=None):
        result['status'] = status
        if reason is not None:
            result['reason'] = str(reason)
        return result

    D_value = _nonnegative(D, theta, 'D')
    E_value = _nonnegative(E_exc, theta, 'E_exc')
    try:
        rr_values = np.asarray(rr(theta) if callable(rr) else rr, dtype=float)
    except RRTemperatureDomainError as error:
        return stop('recombination_temperature_out_of_range', error)
    if rr_values.shape != (2,) or not np.isfinite(rr_values).all() or np.any(rr_values < 0):
        raise ValueError("rr must give one finite nonnegative pair (R,Q).")
    R, Q = map(float, rr_values)
    if R == 0 and Q != 0:
        raise ValueError("Pure RR cannot have Q>0 with R=0; use a consistent pair.")
    b = sqrt(8 * beta * theta)
    wall_energy = lambda_sh * theta * j_target
    if not isfinite(b) or b <= 0 or not isfinite(wall_energy) or wall_energy <= 0:
        raise ValueError("Wall state is not representable; check scales.")
    result.update(b=b, D=D_value, E_exc=E_value, R_rr=R, Q_rr=Q)
    if wall_energy >= 1:
        return stop('energy_temperature_limit')

    field_gradient = float(geometry.dlnB_dg(1.0))
    wall_area = float(geometry.A_ratio(1.0))
    geometric_term = -0.5 * b * field_gradient
    rec_term = R * b * j_target / (4 * beta * theta * wall_area)
    slope_per_j = geometric_term - D_value + rec_term
    if not isfinite(slope_per_j):
        raise ValueError("Wall derivative is not representable; check scales.")
    tolerance = 64 * np.finfo(float).eps * max(1., abs(geometric_term), D_value, rec_term)
    result.update(wall_slope=j_target*slope_per_j, wall_slope_per_j=slope_per_j)
    if slope_per_j > tolerance:
        return stop('inadmissible_wall')
    if abs(slope_per_j) <= tolerance:
        return stop('critical_wall_unresolved')

    def rhs(g, state):
        j, mu, _ = state
        # Clipping is only for provisional stages across the sonic event.
        # Accepted profiles below are checked against mu-b*j, not mu-b.
        radical = sqrt(max((mu-b*j)*(mu+b*j), 0.0))
        H = mu + radical
        area = geometry.A_ratio(g)
        phi = H*H / (16 * beta*beta * theta*theta * area)
        j_rhs = -R * phi
        mu_rhs = (-0.5 * geometry.dlnB_dg(g) * H - D_value*j
                  - R*j*H/(4*beta*theta*area))
        energy_rhs = -D_value*(5*mu+radical)/(8*beta)
        if E_value != 0:
            energy_rhs -= E_value*H/(4*beta*theta)
        if R != 0:
            energy_rhs -= R*j*j/(2*beta*area) + (1.5*R*theta+Q)*phi
        return [j_rhs, mu_rhs, energy_rhs]

    def inlet_energy(g, state):
        return state[2]-1.0
    inlet_energy.terminal = True
    inlet_energy.direction = 1

    def return_to_sonic(g, state):
        return state[1]-b*state[0]
    return_to_sonic.terminal = True
    return_to_sonic.direction = -1

    def particle_zero(g, state):
        return state[0]
    particle_zero.terminal = True
    particle_zero.direction = -1

    step_limit = min(max_step, 0.2/max(1., abs(field_gradient)))
    # Explicitly bound the first probe as well: some SciPy versions sample
    # outside a short geometry interval while estimating their first step.
    first_step = min(step_limit, 0.1*(1-geometry.g_min), 1e-4)
    initial = [j_target, b*j_target, wall_energy]
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        try:
            integration = solve_ivp(
                rhs, (1., geometry.g_min), initial, method='DOP853',
                events=(inlet_energy, return_to_sonic, particle_zero),
                dense_output=True, rtol=rtol, atol=atol, max_step=step_limit,
                first_step=first_step,
            )
        except (FloatingPointError, OverflowError) as error:
            return stop('integration_failed', error)
    result.update(ode_steps=len(integration.t), max_step=step_limit,
                  g_stop=float(integration.t[-1]),
                  last_state=np.asarray(integration.y[:, -1]).copy())
    if not integration.success:
        return stop('integration_failed', integration.message)
    if not integration.t_events[0].size:
        if integration.t_events[1].size:
            return stop('sonic_return')
        if integration.t_events[2].size:
            return stop('nonpositive_particle_flux')
        return stop('geometry_limit')

    alpha = float(integration.t_events[0][0])
    result['alpha'] = alpha
    if not geometry.g_min < alpha < 1:
        return stop('boundary_outside_open_domain')
    xi = np.linspace(0., 1., n_points)
    g = alpha + (1-alpha)*xi
    j, mu, omega = integration.sol(g)
    if not np.isfinite([j, mu, omega]).all():
        return stop('profile_validation_failed', 'Non-finite arrays.')
    scale_j = 1 + float(np.max(np.abs(j)))
    scale_mu = 1 + float(np.max(np.abs(mu)))
    tol_j, tol_mu, tol_w = (50*(atol+rtol*scale_j),
                           50*(atol+rtol*scale_mu), 50*(atol+rtol))
    if np.any(j <= 0) or np.any(np.diff(j) > tol_j):
        return stop('profile_validation_failed', 'Particle flux is nonpositive or increasing.')
    gap = mu-b*j
    if np.any(gap < -tol_mu) or np.any(gap[:-1] <= 0):
        return stop('sonic_resolution_limited', 'Profile is not strictly subsonic inside.')
    if (not np.isclose(j[-1], j_target, rtol=10*rtol, atol=10*atol)
            or abs(gap[-1]) > tol_mu
            or abs(omega[0]-1) > tol_w
            or abs(omega[-1]-wall_energy) > tol_w
            or np.any(omega <= 0) or np.any(np.diff(omega) > tol_w)):
        return stop('profile_validation_failed', 'Boundary or energy monotonicity check failed.')
    radical = np.sqrt(np.maximum((mu-b*j)*(mu+b*j), 0.))
    H = mu+radical
    area = geometry.A_ratio(g)
    velocity = 4*beta*theta*j/H
    density = H/(4*beta*theta*area)
    result.update(
        xi=xi, g=g, j=j, mu=mu, omega=omega, H=H,
        A_over_A_star=area, u_over_u_star=velocity,
        u_over_cs=2*velocity/b, n_over_n_star=density,
        Aq_e_over_W0=omega-j*(velocity**2/(2*beta)+5*theta),
        branch=classify_profile(mu), R_J=float(j[0]-1), R_E=float(omega[0]-1),
        j_a=float(j[0]), mu_a=float(mu[0]), inlet_boundary=inlet_momentum,
    )
    # Preserve the energy-matched profile and R_J even if the kinetic inlet
    # is out of domain. R_M is genuinely undefined in that case.
    try:
        inlet = evaluate_inlet(alpha, inlet_momentum)
    except KineticInletDomainError as error:
        return stop('kinetic_inlet_outside_domain', error)
    target = inlet['mu_target']
    momentum_scale = 1+max(abs(float(mu[0])), abs(target))
    particle_scale = 1+max(abs(float(j[0])), 1.)
    result.update(
        **inlet, R_M=float(mu[0]-target),
        R_J_scaled=result['R_J']/particle_scale,
        R_M_scaled=float(mu[0]-target)/momentum_scale,
        particle_scale=particle_scale, momentum_scale=momentum_scale,
        inlet_target_sonic_margin=float(target-b*j[0]),
    )
    return stop('energy_matched')
