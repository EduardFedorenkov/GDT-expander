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
from .molecular_amjuel import AMJUELDomainError, MAD_FRAGMENT_ENERGY_EV
from .recombination_adas import ADASDomainError, IONIZATION_ENERGY_EV
from .sources import (D2_DISSOCIATION_ENERGY_EV, MAR_ELECTRON_ENERGY_FACTOR,
    TemperatureOnly, adas_sources, coefficients_at, excitation_at, ionization_at,
    molecular_at, molecular_sources)


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


def wall_slope_per_j(theta, jt, beta, R_t, D, geometry, *, D_ion_t=0., molecular_t=(0., 0., 0.)):
    """F_g(1)/j_t for F=mu-sqrt(8*beta*theta)*j; coefficients already evaluated."""
    b = sqrt(8*beta*theta)
    geom = -.5*b*float(geometry.dlnB_dg(1.))
    rec = R_t*b*jt/(4*beta*theta*float(geometry.A_ratio(1.)))
    D_mar, D_mad, D_mai = molecular_t
    # At the wall b*A*N/j_t = 2. Combine -D_sum from mu_g with
    # -2*(D_MAI-D_MAR) from -b*j_g/j_t.
    slope = geom-D-2*D_ion_t+rec+D_mar-D_mad-3*D_mai
    if not isfinite(slope):
        raise ValueError('Unrepresentable wall derivative.')
    tolerance = 64*np.finfo(float).eps*max(
        1., abs(geom), D, 2*abs(D_ion_t), abs(rec), abs(D_mar), abs(D_mad), 3*abs(D_mai))
    return slope, tolerance


def rhs_g(g, state, *, theta, beta, recombination, geometry, D=0.,
          ionization=(0., 0.), excitation=0., molecular=None, T_star_eV=None):
    """Right-hand side per g. D is the frozen nonnegative CX coefficient."""
    H, area, N, v = reconstruct(g, state, theta, beta, geometry)
    R, Q = coefficients_at(recombination, theta, N)
    D_ion, E_ion = ionization_at(ionization, theta, N)
    E_exc = excitation_at(excitation, theta, N)
    particles, momentum, energy = adas_sources(state[0], N, v, area,
        theta=theta, beta=beta, R=R, Q_eff=Q, D=D, D_ion=D_ion, E_ion=E_ion, E_exc=E_exc)
    if molecular is not None:
        if T_star_eV is None:
            raise ValueError('T_star_eV is required for molecular sources.')
        j_mol, mu_mol, omega_mol = molecular_sources(state[0], N, v, area,
            theta=theta, beta=beta, T_star_eV=T_star_eV, molecular=molecular)
        particles, momentum, energy = particles+j_mol, momentum+mu_mol, energy+omega_mol
    return np.array([particles, -.5*geometry.dlnB_dg(g)*H+momentum, energy])


def solve_trial(theta, j_target, alpha, beta, lambda_sh, recombination,
                geometry=None, *, D=0., ionization=(0., 0.), excitation=0.,
                molecular=None, T_star_eV=None, inlet_momentum=None, n_points=1200,
                rtol=1e-9, atol=1e-11, max_step=.02):
    """An IVP, NOT a BVP solution. Return status and all three inlet residuals.

    No energy monotonicity, wall_energy<1, or positive-Q assumption.
    max_step is in xi. ADAS domain failures are explicit invalid trials,
    not proof of physical nonexistence. Malformed data/program errors raise.
    D is a scalar or D(theta), evaluated once because the layer is isothermal.
    ionization is a pair or (theta,N)->(D_ion,E_ion), evaluated LOCALLY.
    excitation is a nonnegative E_exc or local (theta,N)->E_exc from PLT.
    molecular is None (off), a triple or a local (theta,N)->(D_MAR,D_MAD,D_MAI).
    A specified molecular provider requires a positive T_star_eV for energy costs.
    TemperatureOnly providers are evaluated before entering the ODE loop.
    Default off preserves the old calls. Positive j_target>1 is permitted.
    """
    theta, jt, alpha, beta, lam, rtol, atol, max_step = map(float,
        (theta, j_target, alpha, beta, lambda_sh, rtol, atol, max_step))
    if not all(isfinite(x) and x > 0 for x in (theta, jt, beta, lam, rtol, atol, max_step)):
        raise ValueError('Positive finite theta,j_target,beta,lambda,tolerances required.')
    if not isinstance(n_points, (int, np.integer)) or n_points < 2:
        raise ValueError('n_points must be an integer >=2.')
    if molecular is not None:
        if T_star_eV is None or not isfinite(float(T_star_eV)) or float(T_star_eV) <= 0:
            raise ValueError('Molecular sources require a finite positive T_star_eV.')
        T_star_eV = float(T_star_eV)
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
        if isinstance(ionization, TemperatureOnly):
            ionization = ionization.at_temperature(theta)
        if isinstance(excitation, TemperatureOnly):
            excitation = excitation.at_temperature(theta)
        R_t, Q_t = coefficients_at(recombination, theta, N_t)
        D_ion_t, E_ion_t = ionization_at(ionization, theta, N_t)
        E_exc_t = excitation_at(excitation, theta, N_t)
        molecular_t = (0., 0., 0.) if molecular is None else molecular_at(molecular, theta, N_t)
    except ADASDomainError as error:
        return stop('adas_outside_domain', error)
    except AMJUELDomainError as error:
        return stop('amjuel_outside_domain', error)
    slope, tolerance = wall_slope_per_j(theta, jt, beta, R_t, D_value, geometry,
                                       D_ion_t=D_ion_t, molecular_t=molecular_t)
    result.update(D=D_value, wall_slope_per_j=slope, R_wall=R_t, Q_eff_wall=Q_t,
                  N_wall=N_t, D_ion_wall=D_ion_t, E_ion_wall=E_ion_t, E_exc_wall=E_exc_t)
    if molecular is not None:
        result.update(T_star_eV=T_star_eV, D_mar_wall=molecular_t[0],
                      D_mad_wall=molecular_t[1], D_mai_wall=molecular_t[2])
    if slope > tolerance:
        return stop('inadmissible_wall', 'Geometry and active sources cannot launch a backward subsonic branch.')
    if abs(slope) <= tolerance:
        return stop('critical_wall_unresolved', 'First wall derivative is zero within numerical tolerance.')

    length = 1-alpha

    def rhs(xi, state):
        return length*rhs_g(alpha+length*xi, state, theta=theta, beta=beta,
                            recombination=recombination, geometry=geometry, D=D_value,
                            ionization=ionization, excitation=excitation,
                            molecular=molecular, T_star_eV=T_star_eV)

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
    except AMJUELDomainError as error:
        return stop('amjuel_outside_domain', error)
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
    if not np.allclose([j[-1], mu[-1], omega[-1]], wall, rtol=10*rtol, atol=10*atol):
        return stop('profile_validation_failed', 'Wall boundary mismatch.')
    try:
        H, area, N, v = reconstruct(g, (j, mu, omega), theta, beta, geometry)
        R, Q = coefficients_at(recombination, theta, N)
        D_ion, E_ion = ionization_at(ionization, theta, N)
        E_exc = excitation_at(excitation, theta, N)
        D_mai = 0.
        if molecular is not None:
            D_mar, D_mad, D_mai = molecular_at(molecular, theta, N)
            result.update(D_mar_profile=D_mar, D_mad_profile=D_mad, D_mai_profile=D_mai)
        inlet = evaluate_inlet(alpha, inlet_momentum)
    except ADASDomainError as error:
        return stop('adas_outside_domain', error)
    except AMJUELDomainError as error:
        return stop('amjuel_outside_domain', error)
    except KineticInletDomainError as error:
        return stop('kinetic_inlet_outside_domain', error)
    # MAI can increase J even if the atomic ionization channel is off.
    if (not np.any(D_ion > 0) and not np.any(D_mai > 0)
            and np.any(np.diff(j) > 50*(atol+rtol*(1+float(np.max(j)))))):
        return stop('profile_validation_failed', 'Particle flux increases without ionization or MAI.')
    result.update(xi=xi, g=g, j=j, mu=mu, omega=omega, H=H,
        A_over_A_star=area, n_over_n_star=N, u_over_u_star=v,
        u_over_cs=2*v/b, R_profile=R, Q_eff_profile=Q,
        D_ion_profile=D_ion, E_ion_profile=E_ion, E_exc_profile=E_exc,
        Aq_e_over_W0=omega-j*(v*v/(2*beta)+5*theta),
        R_J=float(j[0]-1), R_M=float(mu[0]-inlet['mu_target']),
        R_E=float(omega[0]-1), **inlet)
    return stop('integrated')


def balance_checks(state, recombination, ionization=None, *, excitation=None, molecular=None):
    """Independent dimensional-form budgets, not a call to the local source assemblers.

    Re-evaluates local coefficients; no frozen wall/inlet values. W+I*J
    conservation offers an additional photon/chemical-energy audit in tests.
    Pass the active ionization provider explicitly. Omitting it is allowed
    only for old/off solutions, never silently drops a stored active source.
    The same rule applies to the keyword-only excitation and molecular providers.
    """
    g, j, mu, w, N, v, A, H = (np.asarray(state[k]) for k in
        ('g', 'j', 'mu', 'omega', 'n_over_n_star', 'u_over_u_star', 'A_over_A_star', 'H'))
    theta, beta = state['theta'], state['beta']
    R, Q = coefficients_at(recombination, theta, N)
    if ionization is None:
        if (np.any(np.asarray(state.get('D_ion_profile', 0.)) != 0)
                or np.any(np.asarray(state.get('E_ion_profile', 0.)) != 0)):
            raise ValueError('Pass the local ionization provider to balance_checks for an active solution.')
        ionization = (0., 0.)
    D_ion, E_ion = ionization_at(ionization, theta, N)
    if excitation is None:
        if np.any(np.asarray(state.get('E_exc_profile', 0.)) != 0):
            raise ValueError('Pass the local excitation provider to balance_checks for an active solution.')
        excitation = 0.
    E_exc = excitation_at(excitation, theta, N)
    created = D_ion*A*N
    removed = R*A*N*N
    D = cx_coefficient(state.get('D', 0.), theta)
    momentum = -.5*state['geometry'].dlnB_dg(g)*H-D*j-R*j*N
    ion_energy = v*v/(2*beta)+1.5*theta
    ion_loss, effective_loss = removed*ion_energy, Q*A*N*N
    cx_loss = D*A*N*ion_energy
    ionization_loss = E_ion*A*N
    excitation_loss = E_exc*A*N
    lost = float(simpson(removed, x=g))
    born = float(simpson(created, x=g))
    power_loss = cx_loss+ion_loss+effective_loss+ionization_loss+excitation_loss
    net_source = born-lost
    molecular_integrals = {}
    if molecular is None:
        if any(np.any(np.asarray(state.get(f'D_{branch}_profile', 0.)) != 0)
               for branch in ('mar', 'mad', 'mai')):
            raise ValueError('Pass the local molecular provider to balance_checks for an active solution.')
    else:
        D_mar, D_mad, D_mai = molecular_at(molecular, theta, N)
        if 'T_star_eV' not in state:
            raise ValueError('Molecular balance checks require T_star_eV in the trial state.')
        T_star_eV = state['T_star_eV']
        mar_events, mad_events, mai_events = D_mar*A*N, D_mad*A*N, D_mai*A*N
        lost_mar, born_mai = float(simpson(mar_events, x=g)), float(simpson(mai_events, x=g))
        net_source += born_mai-lost_mar
        mar_loss = mar_events*(ion_energy+MAR_ELECTRON_ENERGY_FACTOR*theta)
        mad_loss = mad_events*(ion_energy+(D2_DISSOCIATION_ENERGY_EV+MAD_FRAGMENT_ENERGY_EV)/T_star_eV)
        mai_loss = mai_events*(ion_energy+(D2_DISSOCIATION_ENERGY_EV+IONIZATION_ENERGY_EV)/T_star_eV)
        power_loss = power_loss+mar_loss+mad_loss+mai_loss
        molecular_drag = j*(D_mar+D_mad+D_mai)
        momentum = momentum-molecular_drag
        molecular_integrals = dict(particle_mar_integral=lost_mar, particle_mai_integral=born_mai,
            mad_event_integral=float(simpson(mad_events, x=g)),
            mar_energy_loss_integral=float(simpson(mar_loss, x=g)),
            mad_energy_loss_integral=float(simpson(mad_loss, x=g)),
            mai_energy_loss_integral=float(simpson(mai_loss, x=g)),
            molecular_momentum_loss_integral=float(simpson(molecular_drag, x=g)))
    energy_loss = float(simpson(power_loss, x=g))
    return dict(particle_abs=abs(float(j[-1]-j[0])-net_source),
        momentum_abs=abs(float(mu[-1]-mu[0])-float(simpson(momentum, x=g))),
        energy_abs=abs(float(w[0]-w[-1])-energy_loss),
        particle_recombination_integral=lost, energy_loss_integral=energy_loss,
        particle_ionization_integral=born, particle_net_source_integral=net_source,
        ion_energy_loss_integral=float(simpson(ion_loss, x=g)),
        effective_energy_loss_integral=float(simpson(effective_loss, x=g)),
        cx_energy_loss_integral=float(simpson(cx_loss, x=g)),
        cx_momentum_loss_integral=float(simpson(D*j, x=g)),
        ionization_energy_loss_integral=float(simpson(ionization_loss, x=g)),
        excitation_energy_loss_integral=float(simpson(excitation_loss, x=g)), **molecular_integrals)
