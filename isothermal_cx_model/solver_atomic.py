"""Canonical isothermal atomic trial: CX, excitation, Case-A RR, ionization.

Integrate j, mu, omega in g=z/L from the sonic plate to omega=1. The event
defines alpha=a/L. Both inlet residuals remain unchecked trial outputs.
Prescribed constant neutrals per trial; no ionization equilibrium imposed.
The stationary finder and 2D map accept the same ionization argument.
No three-body recombination, molecules or new temperature evolution.
"""

from math import isfinite, sqrt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import simpson

from .geometry import quadratic_area_legacy
from .inlet import evaluate_inlet, KineticInletDomainError
from .recombination import RRTemperatureDomainError
from .solver_general import classify_profile


def _nonnegative(value, theta, name):
    value = float(value(theta) if callable(value) else value)
    if not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return value


def atomic_rhs(g, state, *, theta, beta, D, E_exc, R, Q,
               D_ion, E_ion, geometry):
    """RHS in g=z/L, with already evaluated constant coefficients.

    The radical clipping is ONLY for provisional integrator stages; accepted
    profiles are independently checked to lie on the subsonic branch.
    Ionization adds no conservative momentum source for stationary atoms.
    """
    j, mu, _ = state
    b = sqrt(8*beta*theta)
    radical = sqrt(max((mu-b*j)*(mu+b*j), 0.0))
    H = mu+radical
    area = geometry.A_ratio(g)
    phi = H*H/(16*beta*beta*theta*theta*area)
    j_rhs = -R*phi
    if D_ion != 0:
        j_rhs += D_ion*H/(4*beta*theta)
    mu_rhs = -.5*geometry.dlnB_dg(g)*H-D*j-R*j*H/(4*beta*theta*area)
    energy_rhs = -D*(5*mu+radical)/(8*beta)
    if E_exc != 0:
        energy_rhs -= E_exc*H/(4*beta*theta)
    if E_ion != 0:
        energy_rhs -= E_ion*H/(4*beta*theta)
    if R != 0:
        energy_rhs -= R*j*j/(2*beta*area)+(1.5*R*theta+Q)*phi
    return [j_rhs, mu_rhs, energy_rhs]


def balance_checks(state, beta):
    """Independent quadratures of a trial profile; NOT a BVP acceptance test.

    Errors normalized by J0,M0,W0. Particle source is ionization minus RR.
    Power is evaluated using N and v, separately from the expanded RHS.
    particle_loss_integral is retained as a legacy alias for RR ONLY.
    """
    g, j, mu, omega, H, A, N, v = [np.asarray(state[k]) for k in
        ('g', 'j', 'mu', 'omega', 'H', 'A_over_A_star', 'n_over_n_star', 'u_over_u_star')]
    theta, D, E, R, Q = [state[k] for k in ('theta', 'D', 'E_exc', 'R_rr', 'Q_rr')]
    D_ion, E_ion = state.get('D_ion', 0.), state.get('E_ion', 0.)
    ionization = D_ion*A*N
    recombination = R*A*N*N
    momentum_rhs = -.5*state['geometry'].dlnB_dg(g)*H-D*j-R*j*N
    ion_energy = v*v/(2*beta)+1.5*theta
    power_loss = (D*A*N*ion_energy+E*A*N+E_ion*A*N
                  +recombination*ion_energy+Q*A*N*N)
    born = float(simpson(ionization, x=g))
    lost = float(simpson(recombination, x=g))
    return dict(
        particle_abs=abs(float(j[-1]-j[0])-(born-lost)),
        momentum_abs=abs(float(mu[-1]-mu[0])-float(simpson(momentum_rhs, x=g))),
        energy_abs=abs(float(omega[0]-omega[-1])-float(simpson(power_loss, x=g))),
        particle_loss_integral=lost, particle_recombination_integral=lost,
        particle_ionization_integral=born, particle_net_source_integral=born-lost,
        energy_ionization_integral=float(simpson(E_ion*A*N, x=g)),
    )


def solve_trial(theta, j_target, beta, D, lambda_sh, geometry=None,
                n_points=1600, *, E_exc=0.0, rr=(0.0, 0.0),
                ionization=(0.0, 0.0), inlet_momentum=None,
                rtol=1e-9, atol=1e-11, max_step=0.02):
    """Return a status dictionary; arrays are present after energy matching.

    D and E_exc: numbers or functions of theta. rr: a pair (R,Q), or one
    function theta -> (R,Q), e.g. make_rr_coefficients(parameters, scale=1).
    R,Q include their common RR multiplier already. They are evaluated once.
    ionization is (D_ion,E_ion) or theta -> that pair, evaluated once.
    make_ionization_coefficients supplies the consistent E_ion=(I/T_star)*D_ion.
    The default (0,0) preserves the former RR-only physics. j_target>1 is
    permitted, including off-root trials; the wall energy bound still applies.

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
            or not isfinite(j_target) or not j_target > 0):
        raise ValueError("Need positive finite theta,beta,lambda,tolerances and j_target.")
    if not isinstance(n_points, (int, np.integer)) or n_points < 2:
        raise ValueError("n_points must be an integer >=2.")
    geometry = quadratic_area_legacy() if geometry is None else geometry
    result = dict(
        model='atomic_trial', theta=theta, j_target=j_target, geometry=geometry,
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
    ion_values = np.asarray(ionization(theta) if callable(ionization) else ionization, dtype=float)
    if ion_values.shape != (2,) or not np.isfinite(ion_values).all() or np.any(ion_values < 0):
        raise ValueError('ionization must give one finite nonnegative pair (D_ion,E_ion).')
    D_ion, E_ion = map(float, ion_values)
    if D_ion == 0 and E_ion != 0:
        raise ValueError('Ionization cannot have E_ion>0 with D_ion=0.')
    b = sqrt(8 * beta * theta)
    wall_energy = lambda_sh * theta * j_target
    if not isfinite(b) or b <= 0 or not isfinite(wall_energy) or wall_energy <= 0:
        raise ValueError("Wall state is not representable; check scales.")
    result.update(b=b, D=D_value, E_exc=E_value, R_rr=R, Q_rr=Q,
                  D_ion=D_ion, E_ion=E_ion)
    if wall_energy >= 1:
        return stop('energy_temperature_limit')

    field_gradient = float(geometry.dlnB_dg(1.0))
    wall_area = float(geometry.A_ratio(1.0))
    geometric_term = -0.5 * b * field_gradient
    rec_term = R * b * j_target / (4 * beta * theta * wall_area)
    slope_per_j = geometric_term - D_value - 2*D_ion + rec_term
    if not isfinite(slope_per_j):
        raise ValueError("Wall derivative is not representable; check scales.")
    tolerance = 64 * np.finfo(float).eps * max(1., abs(geometric_term), D_value, 2*D_ion, rec_term)
    result.update(wall_slope=j_target*slope_per_j, wall_slope_per_j=slope_per_j)
    if slope_per_j > tolerance:
        return stop('inadmissible_wall')
    if abs(slope_per_j) <= tolerance:
        return stop('critical_wall_unresolved')

    def rhs(g, state):
        return atomic_rhs(g, state, theta=theta, beta=beta, D=D_value,
            E_exc=E_value, R=R, Q=Q, D_ion=D_ion, E_ion=E_ion, geometry=geometry)

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
    if np.any(j <= 0):
        return stop('profile_validation_failed', 'Particle flux is nonpositive.')
    if D_ion == 0 and np.any(np.diff(j) > tol_j):
        return stop('profile_validation_failed', 'Particle flux increases with no ionization.')
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
