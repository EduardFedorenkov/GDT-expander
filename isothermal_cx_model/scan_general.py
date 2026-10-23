"""Find temperature roots without identifying solutions by profile shape."""

from collections import Counter
import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .geometry import quadratic_area_legacy
from .solver_general import solve_at_theta


def find_temperature_solutions(beta, D, lambda_sh, geometry=None,
                               n_points=1600, b_points=100,
                               *, residual_tol=1e-7,
                               boundary_rtol=1e-6, boundary_max_steps=40,
                               inlet_momentum=None):
    """Return (verified states, diagnostics).

    Searches a finite mixed b-grid, refines validity boundaries, extrema
    and |R_M| minima. A finite point next to an undefined residual is not
    necessarily close enough to the boundary to bracket a nearby root.
    This is a numerical search, not a proof that all roots were found.
    Undefined residual intervals are never bracketed across.
    """
    beta, lambda_sh = float(beta), float(lambda_sh)
    if not np.isfinite([beta, lambda_sh, residual_tol, boundary_rtol]).all() or min(
            beta, lambda_sh, residual_tol, boundary_rtol) <= 0:
        raise ValueError("beta, lambda_sh and tolerances must be positive.")
    if (not isinstance(boundary_max_steps, (int, np.integer))
            or boundary_max_steps < 1):
        raise ValueError("boundary_max_steps must be an integer >=1.")
    if b_points < 20:
        raise ValueError("b_points must be >=20.")
    geometry = quadratic_area_legacy() if geometry is None else geometry
    b_limit = np.sqrt(8 * beta / lambda_sh)
    if inlet_momentum is None:
        b_limit = min(1.0, b_limit)
    b_hi = b_limit * (1 - 1e-8)
    b_lo = b_hi * 1e-7
    cache = {}

    def evaluate(b):
        b = float(b)
        if b not in cache:
            cache[b] = solve_at_theta(
                b*b/(8*beta), beta, D, lambda_sh, geometry, n_points,
                inlet_momentum=inlet_momentum,
            )
        return cache[b]

    def residual(b):
        return evaluate(b)["R_M"]

    def tolerance(state):
        return residual_tol * state.get("momentum_scale", 1.0)

    boundary_intervals = []

    def refine_domain_boundaries(points):
        """Bisect every detected finite/undefined transition in log(b).

        Keep both sides and every tested undefined point. Do not invent a
        residual outside the physical domain or bracket across its gaps.
        Entire valid islands with two undefined endpoints can still be
        missed by a finite seed grid; no completeness claim is made.
        """
        points = np.unique(np.r_[points, list(cache)])
        values = np.array([residual(b) for b in points])
        for left, right, f_left, f_right in zip(
                points[:-1], points[1:], values[:-1], values[1:]):
            left_valid = bool(np.isfinite(f_left))
            if left_valid == bool(np.isfinite(f_right)):
                continue
            initial_left, initial_right = float(left), float(right)
            steps = 0
            while right-left > boundary_rtol*max(abs(left), abs(right)):
                if steps >= boundary_max_steps:
                    break
                # Positive b: use the logarithmic midpoint so the operation
                # remains scale-aware for very cold roots.
                middle = float(np.sqrt(left)*np.sqrt(right))
                if not left < middle < right:
                    break
                middle_valid = bool(np.isfinite(residual(middle)))
                if middle_valid == left_valid:
                    left = middle
                else:
                    right = middle
                steps += 1
            boundary_intervals.append({
                "initial_left": initial_left, "initial_right": initial_right,
                "left": float(left), "right": float(right), "steps": steps,
                "resolved": bool(right-left <= boundary_rtol*max(
                    abs(left), abs(right))),
            })
        return np.unique(np.r_[points, list(cache)])

    nodes = np.unique(np.r_[
        np.geomspace(b_lo, b_hi, max(10, b_points//2)),
        np.linspace(b_lo, b_hi, b_points),
    ])
    nodes = refine_domain_boundaries(nodes)
    values = np.array([residual(b) for b in nodes])
    extra = []
    extrema_windows = []

    # Refine both signed extrema (a pair of roots can hide in a local
    # minimum OR maximum) and possible tangencies of the residual.
    for i in range(1, len(nodes)-1):
        triple = values[i-1:i+2]
        if not np.isfinite(triple).all():
            continue
        objectives = []
        if triple[1] <= min(triple[0], triple[2]):
            objectives.append(lambda x: residual(x))
        if triple[1] >= max(triple[0], triple[2]):
            objectives.append(lambda x: -residual(x))
        if abs(triple[1]) <= min(abs(triple[0]), abs(triple[2])):
            objectives.append(lambda x: abs(residual(x)))
        for objective in objectives:
            def safe_objective(x):
                value = objective(x)
                return value if np.isfinite(value) else np.inf
            found = minimize_scalar(
                safe_objective, bounds=(nodes[i-1], nodes[i+1]),
                method="bounded",
                options={"xatol": max(1e-14, b_hi*1e-10)},
            )
            if found.success and np.isfinite(found.fun):
                extra.append(float(found.x))
                extrema_windows.append((float(found.x), nodes[i-1], nodes[i+1]))

    # Preserve every tested invalid point as a gap.
    nodes = refine_domain_boundaries(np.r_[nodes, extra])
    values = np.array([residual(b) for b in nodes])
    candidates = []
    discontinuous_brackets = 0
    for left, right, f_left, f_right in zip(
            nodes[:-1], nodes[1:], values[:-1], values[1:]):
        if not np.isfinite([f_left, f_right]).all() or f_left*f_right >= 0:
            continue
        try:
            def bracket_residual(b):
                value = residual(b)
                if not np.isfinite(value):
                    raise ValueError("Undefined residual inside bracket.")
                return value
            b_root = brentq(
                bracket_residual, left, right,
                xtol=max(1e-14, b_hi*1e-12), rtol=1e-11,
            )
        except ValueError:
            discontinuous_brackets += 1
            continue
        candidates.append((float(b_root), "sign_change"))

    # A small residual on many optimizer samples must NOT create many roots.
    # Bracketed roots take precedence. Add only optimized near-zero extrema
    # in windows containing no bracketed root (possible tangencies).
    for b, value in zip(nodes, values):
        if value == 0:
            candidates.append((float(b), "exact_sample"))
    tangent_windows = []
    for b, left, right in extrema_windows:
        if abs(residual(b)) > tolerance(evaluate(b)):
            continue
        if any(left <= root <= right for root, _ in candidates):
            continue
        overlapping = [w for w in tangent_windows if left <= w[2] and right >= w[1]]
        if overlapping:
            previous = min(overlapping, key=lambda w: abs(residual(w[0])))
            if abs(residual(b)) < abs(residual(previous[0])):
                tangent_windows.remove(previous)
                tangent_windows.append((b, left, right))
        else:
            tangent_windows.append((b, left, right))
    candidates.extend((b, "possible_tangency") for b, _, _ in tangent_windows)

    # Tighter ODE tolerances AND a smaller integration step, not merely
    # more output samples. A failed verification is reported, never hidden.
    solutions = []
    priority = {"sign_change": 0, "exact_sample": 1, "possible_tangency": 2}
    for b_root, root_kind in sorted(candidates, key=lambda item: priority[item[1]]):
        if any(abs(b_root-s["b"]) <= max(1e-12, 1e-6*abs(b_root))
               for s in solutions):
            continue
        trial = evaluate(b_root)
        checked = solve_at_theta(
            trial["theta"], beta, trial["D"], lambda_sh,
            geometry, 2*n_points, rtol=1e-10, atol=1e-12,
            max_step=trial.get("max_step", 0.02)/2,
            inlet_momentum=inlet_momentum,
        )
        if checked["status"] != "energy_matched":
            raise RuntimeError("A candidate disappeared under ODE refinement.")
        if abs(checked["R_M"]) > 5*tolerance(checked):
            raise RuntimeError("Momentum verification failed; refine the search.")
        # G<b is valid for an off-root residual (needed for bracketing),
        # but cannot be accepted as a strictly subsonic matched inlet.
        if checked.get("inlet_target_sonic_margin", 1.0) <= 0:
            raise RuntimeError("Matched inlet target is not strictly subsonic.")
        checked.update(
            root_kind=root_kind,
            R_M_check=checked["R_M"], R_E_check=checked["R_E"],
            delta_alpha=checked["alpha"]-trial["alpha"],
            R_M_tolerance=5*tolerance(checked),
        )
        solutions.append(checked)
    solutions.sort(key=lambda state: state["theta"])

    finite = np.array([s["R_M"] for s in cache.values()
                       if np.isfinite(s["R_M"])])
    statuses = dict(Counter(s["status"] for s in cache.values()))
    diagnostic = {
        "min_R_M": float(finite.min()) if finite.size else np.nan,
        "max_R_M": float(finite.max()) if finite.size else np.nan,
        "min_abs_R_M": float(np.abs(finite).min()) if finite.size else np.nan,
        "statuses": statuses,
        "discontinuous_brackets": discontinuous_brackets,
        "n_evaluations": len(cache),
        "b_min": b_lo, "b_max": b_hi,
        "inlet_momentum": "constant" if inlet_momentum is None else "variable",
        "kinetic_domain_reasons": dict(Counter(
            s["kinetic_domain_reason"] for s in cache.values()
            if "kinetic_domain_reason" in s)),
        "kinetic_path_incomplete": bool(inlet_momentum is not None
                                        and geometry.g_min > 0),
        "boundary_intervals": boundary_intervals,
        "boundary_rtol": boundary_rtol,
        "boundary_refinements": sum(item["steps"] > 0 for item in boundary_intervals),
        "boundary_evaluations": sum(item["steps"] for item in boundary_intervals),
        "boundary_resolution_limited": any(
            not item["resolved"] for item in boundary_intervals),
    }
    if statuses.get("zero_cx_degenerate"):
        raise NotImplementedError(
            "D=0: energy does not determine alpha; use a separate zero-CX problem."
        )
    diagnostic["has_unresolved_points"] = bool(
        statuses.get("critical_wall_unresolved") or discontinuous_brackets
        or diagnostic["boundary_resolution_limited"]
    )
    return solutions, diagnostic
