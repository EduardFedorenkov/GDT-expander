"""Energy-first shooting for the isothermal CX model and general B(g).

omega(alpha)=1 sets the inlet position; mu(alpha)=G(alpha) selects temperature.
Pass inlet_momentum for the kinetic closure; None retains the old G=1 API.
"""

from math import isfinite, sqrt
import numpy as np
from scipy.integrate import solve_ivp

from .geometry import quadratic_area_legacy
from .inlet import evaluate_inlet, KineticInletDomainError


def classify_profile(mu):
    """Diagnostic only: never use this label as a unique solution ID."""
    mu = np.asarray(mu, dtype=float)
    differences = np.diff(mu)
    tolerance = 1e-8 * max(1.0, float(np.max(np.abs(mu))))
    signs = np.sign(differences[np.abs(differences) > tolerance])
    changes = int(np.count_nonzero(np.diff(signs))) if signs.size else 0
    if changes == 0:
        return "monotone"
    if changes == 1 and signs[0] > 0 and signs[-1] < 0:
        return "with_maximum"
    return "nonmonotone"


def solve_at_theta(theta, beta, D, lambda_sh, geometry=None, n_points=1600,
                   *, inlet_momentum=None, rtol=1e-9, atol=1e-11, max_step=0.02):
    """Return a trial dictionary with explicit status and, on success, arrays.

    status='energy_matched' means omega(alpha)=1; R_M can still be nonzero.
    Other physical statuses do NOT mean that a global solution is impossible.
    Numerical integration failures raise RuntimeError.
    D=0 and critical sonic-wall initial data are explicitly unresolved.
    """
    geometry = quadratic_area_legacy() if geometry is None else geometry
    theta, beta, lambda_sh = map(float, (theta, beta, lambda_sh))
    D_value = float(D(theta) if callable(D) else D)
    if not all(isfinite(v) and v > 0
               for v in (theta, beta, lambda_sh, rtol, atol, max_step)):
        raise ValueError("theta, beta, lambda_sh and numerical tolerances must be >0.")
    if not isfinite(D_value) or D_value < 0:
        raise ValueError("D must be finite and nonnegative.")
    if not isinstance(n_points, (int, np.integer)) or n_points < 2:
        raise ValueError("n_points must be an integer >=2.")

    b = sqrt(8 * beta * theta)
    result = {
        "theta": theta, "b": b, "D": D_value, "geometry": geometry,
        "status": None, "alpha": np.nan, "R_M": np.nan, "R_E": np.nan,
    }

    def stop(status):
        result["status"] = status
        return result

    if inlet_momentum is None and b >= 1:
        return stop("inlet_sonic_limit")
    if D_value == 0:
        return stop("zero_cx_degenerate")
    if lambda_sh * theta >= 1:
        return stop("energy_temperature_limit")

    wall_coefficient = -0.5 * geometry.dlnB_dg(1.0)
    wall_slope = b * wall_coefficient - D_value
    result["wall_slope"] = wall_slope
    critical_tolerance = 64 * np.finfo(float).eps * max(
        1.0, abs(b * wall_coefficient), D_value,
    )
    if wall_slope > critical_tolerance:
        return stop("inadmissible_wall")
    if abs(wall_slope) <= critical_tolerance:
        return stop("critical_wall_unresolved")

    def rhs(g, state):
        mu = float(state[0])
        # Clipping is ONLY for provisional integrator stages near sonic events.
        radical = sqrt(max((mu - b) * (mu + b), 0.0))
        H = mu + radical
        return [
            -0.5 * geometry.dlnB_dg(g) * H - D_value,
            -D_value * (5 * mu + radical) / (8 * beta),
        ]

    def inlet_energy(g, state):
        return state[1] - 1.0
    inlet_energy.terminal = True
    inlet_energy.direction = 1

    def return_to_sonic(g, state):
        return state[0] - b
    return_to_sonic.terminal = True
    return_to_sonic.direction = -1

    # Resolve a large logarithmic gradient near the wall. solve_ivp may take
    # still smaller steps according to its error estimate.
    wall_scale = max(1.0, abs(geometry.dlnB_dg(1.0)))
    step_limit = min(max_step, 0.2 / wall_scale)
    integration = solve_ivp(
        rhs, (1.0, geometry.g_min), [b, lambda_sh * theta],
        method="DOP853", events=(inlet_energy, return_to_sonic),
        dense_output=True, rtol=rtol, atol=atol, max_step=step_limit,
    )
    if not integration.success:
        raise RuntimeError(f"ODE integration failed: {integration.message}")
    if not integration.t_events[0].size:
        return stop("sonic_return" if integration.t_events[1].size
                    else "geometry_limit")

    alpha = float(integration.t_events[0][0])
    if not geometry.g_min < alpha < 1:
        return stop("boundary_outside_open_domain")
    xi = np.linspace(0.0, 1.0, n_points)
    g = alpha + (1 - alpha) * xi
    mu, omega = integration.sol(g)

    tolerance = 50 * (atol + rtol * max(1.0, float(np.max(np.abs(mu)))))
    if not np.isfinite(mu).all() or not np.isfinite(omega).all():
        raise RuntimeError("Non-finite solution arrays.")
    if np.any(mu < b - tolerance) or np.any(mu[:-1] <= b):
        raise RuntimeError("Accepted profile is not strictly subsonic in its interior.")
    if not np.isclose(mu[-1], b, rtol=rtol*10, atol=atol*10):
        raise RuntimeError("Sonic wall condition was lost.")
    if abs(omega[0] - 1) > 50 * (atol + rtol):
        raise RuntimeError("Energy event was not located accurately.")
    if not np.isclose(omega[-1], lambda_sh*theta, rtol=rtol*10, atol=atol*10):
        raise RuntimeError("Wall energy condition was lost.")
    if np.any(omega <= 0) or np.any(np.diff(omega) > 50*(atol+rtol)):
        raise RuntimeError("Accepted energy profile is nonpositive or increasing.")

    try:
        inlet = evaluate_inlet(alpha, inlet_momentum)
    except KineticInletDomainError as error:
        result.update(alpha=alpha, kinetic_domain_reason=str(error))
        return stop("kinetic_inlet_outside_domain")
    target = inlet["mu_target"]
    residual = float(mu[0] - target)
    # abs + rel tolerance with equal coefficients: eps*(1+max(|mu|,|G|)).
    momentum_scale = 1.0 + max(abs(float(mu[0])), abs(target))

    radical = np.sqrt(np.maximum((mu - b) * (mu + b), 0.0))
    result.update(
        status="energy_matched", alpha=alpha, xi=xi, g=g, mu=mu, omega=omega,
        u_over_cs=b / (mu + radical),
        branch=classify_profile(mu),
        R_M=residual, R_E=float(omega[0] - 1), mu_a=float(mu[0]),
        R_M_scaled=residual/momentum_scale, momentum_scale=momentum_scale,
        inlet_residual=residual,
        inlet_target_sonic_margin=float(target - b),
        inlet_boundary=inlet_momentum,
        ode_steps=len(integration.t), max_step=step_limit,
        **inlet,
    )
    return result
