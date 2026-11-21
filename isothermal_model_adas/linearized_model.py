"""Dimensionless normal-mode problem about one SAVED isothermal solution.

No stationary root search is performed. Scales stay frozen at their saved
values. The time variable is tau=t*U_star/L; sigma=s*L/U_star.
The kinetic source is frozen, chi fixed, electron kinetic energy=(3/2)n_i*T_c.
AMJUEL intermediates and neutral gas do not acquire independent dynamics.
"""

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad, simpson, solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import root

from .atomic_coefficients import build_atomic_coefficients
from .geometry import load_geometry
from .inlet import make_inlet_momentum
from .sources import (adas_sources, coefficients_at, excitation_at,
                      ionization_at, molecular_sources)
from .transport import balance_checks, solve_trial


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def saved_cases(directory, n0, lambda_sh):
    """Load all verified saved roots at an exact requested physical case."""
    directory = Path(directory).resolve()
    manifest = read_json(directory / "manifest.json")
    results = []
    for case_id, entry in sorted(read_json(directory / "checkpoint.json")["completed"].items()):
        report_path = (directory / entry["report"]).resolve()
        if directory not in report_path.parents:
            raise ValueError("Checkpoint report is outside the supplied scan directory.")
        report = read_json(report_path)
        p = report["parameters"]
        if not (np.isclose(p["n0_cm3"], n0, rtol=1e-12, atol=0) and
                np.isclose(p["lambda_sh"], lambda_sh, rtol=1e-12, atol=0)):
            continue
        for index, seed in enumerate(report["solutions"]):
            if seed.get("stationary_verified") is True:
                results.append(SavedBackground(report_path, report, seed, case_id, index, manifest))
    return results


@dataclass
class SavedBackground:
    report_path: Path
    report: dict
    seed: dict
    case_id: str
    root_index: int
    manifest: dict

    def __post_init__(self):
        self.p = self.report["parameters"]
        opts = self.report["model_options"]
        self.geometry = load_geometry(self.report_path.with_name("geometry.json"))
        mode = opts.get("atomic_model", "full-adas")
        if mode != "full-adas" or opts["inlet_momentum"] != "kinetic":
            raise ValueError("This stability closure requires full-adas and the kinetic inlet.")
        self.rr, self.processes, _, paths = build_atomic_coefficients(
            self.p, mode, opts["rr_scale"], opts["ionization_scale"], opts["excitation_scale"])
        expected = self.report.get("provenance", {}).get(
            "data_sha256", self.manifest["signature"]["data_sha256"])
        actual = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        if actual != expected:
            raise ValueError("Active reaction tables differ from the saved stationary calculation.")
        self.inlet = make_inlet_momentum(self.p, self.geometry, "kinetic")
        self.theta = float(self.seed["T_eV"] / self.p["T_star_eV"])
        self.jt = float(self.seed["j_target"])
        self.alpha = float(self.seed["alpha"])
        self.beta = self.p["beta"]
        self.lam = self.p["lambda_sh"]
        self.time_star = self.p["L_cm"] / self.p["U_star_cm_per_s"]
        self.b = np.sqrt(8 * self.beta * self.theta)
        if not self.geometry.g_min < self.alpha < 1:
            raise ValueError("Saved inlet is outside the open geometry interval.")

    def trial(self, theta=None, jt=None, alpha=None, points=4001):
        """Reconstruct a profile using fixed parameters, not solve_stationary."""
        return solve_trial(
            self.theta if theta is None else theta,
            self.jt if jt is None else jt,
            self.alpha if alpha is None else alpha,
            self.beta, self.lam, self.rr, self.geometry,
            **self.processes, inlet_momentum=self.inlet, n_points=points,
            rtol=1e-11, atol=1e-13, max_step=.02)

    def verify(self):
        state = self.trial()
        if state["status"] != "integrated":
            raise ValueError(f"Saved profile cannot be reproduced: {state['status']}")
        residual = max(abs(state[key]) for key in ("R_J", "R_M", "R_E"))
        budget = balance_checks(
            state, self.rr, self.processes["ionization"],
            excitation=self.processes["excitation"], molecular=self.processes.get("molecular"))
        max_budget = max(budget[key] for key in ("particle_abs", "momentum_abs", "energy_abs"))
        if residual > 2e-7 or max_budget > 5e-6:
            raise ValueError(f"Background failed validation: residual={residual:g}, budget={max_budget:g}")
        return dict(max_stationary_residual=residual, max_budget_error=max_budget)

    def local(self, g, j, r, theta):
        """RHS for (j,F,omega) per g, F=mu-b*j, r=sqrt(F).

        A signed r is allowed ONLY in differentiation: this is a smooth local
        extension across the sonic endpoint, not a physical supersonic branch.
        """
        b = np.sqrt(8 * self.beta * theta)
        H = b*j + r*r + r*np.sqrt(2*b*j + r*r)
        area = self.geometry.A_ratio(g)
        N, v = H/(4*self.beta*theta*area), 4*self.beta*theta*j/H
        R, Q = coefficients_at(self.rr, theta, N)
        Di, Ei = ionization_at(self.processes["ionization"], theta, N)
        Ex = excitation_at(self.processes["excitation"], theta, N)
        provider = self.processes["D"]
        D = provider(theta) if callable(provider) else provider
        # adas_sources has a scalar branch on D, so add its vectorized CX term here.
        source = np.array(adas_sources(j, N, v, area, theta=theta, beta=self.beta,
                                      R=R, Q_eff=Q, D_ion=Di, E_ion=Ei, E_exc=Ex))
        source[1] -= .5*self.geometry.dlnB_dg(g)*H + D*j
        source[2] -= D*area*N*(v*v/(2*self.beta) + 1.5*theta)
        if "molecular" in self.processes:
            source += np.array(molecular_sources(
                j, N, v, area, theta=theta, beta=self.beta,
                T_star_eV=self.p["T_star_eV"], molecular=self.processes["molecular"]))
        source[1] -= b*source[0]
        return source

    def storage(self, j, r, theta):
        """(AN, momentum stock j, energy stock) in frozen dimensionless scales."""
        b = np.sqrt(8*self.beta*theta)
        H = b*j+r*r+r*np.sqrt(2*b*j+r*r)
        AN = H/(4*self.beta*theta)
        v = j/AN
        return np.array([AN, j, AN*(3*theta+v*v/(2*self.beta))])

    def kinetic_storage(self):
        K = self.geometry.A_ratio(self.alpha)
        c, chi = 1-1/K, self.p["chi"]
        def kernel(x):
            return 2/np.sqrt(np.pi)*x*np.exp(-x)/(np.sqrt(x+chi)+np.sqrt(c*x+chi))
        H = quad(kernel, 0, np.inf, epsabs=1e-12)[0]
        I = quad(lambda x: kernel(x)*(x+chi), 0, np.inf, epsabs=1e-12)[0]
        scale = .5*self.p["n_c_cm3"]/self.p["n_star_cm3"]
        return np.array([scale*H, 1.,
            scale*self.p["T_c_eV"]/self.p["T_star_eV"]*(I+1.5*H)])


@dataclass(frozen=True)
class LinearControls:
    points: int = 2049
    derivative_step: float = 1e-5
    rtol: float = 2e-10
    max_step: float = .025

    def __post_init__(self):
        if self.points < 129 or not isinstance(self.points, int):
            raise ValueError("At least 129 coefficient points required.")
        if not 0 < self.derivative_step < .01 or not 0 < self.rtol < .01 or not 0 < self.max_step <= 1:
            raise ValueError("Invalid linear solver controls.")

    def refined(self):
        return LinearControls(2*self.points-1, self.derivative_step/3,
                              self.rtol/5, self.max_step/2)


class LinearizedModel:
    """Boundary matrix in columns (delta log j_t, delta alpha, delta log theta).

    Rows are particle, momentum, energy matching. Physical s is in s^-1.
    For the sonic endpoint use x=sqrt((1-g)/(1-alpha)); F~launch*x^2.
    Coefficients are real; complex arithmetic is used only for normal modes.
    """

    def __init__(self, background, controls=LinearControls()):
        self.bg, self.controls = background, controls
        bg, cfg = background, controls
        ell = 1-bg.alpha
        self.launch = -ell*bg.local(1., bg.jt, 0., bg.theta)[1]
        if self.launch <= 0:
            raise ValueError("No regular backward subsonic launch.")
        def steady(x, y):
            F = y[1] if y[1] >= 0 else self.launch*x*x
            return -2*ell*x*bg.local(1-ell*x*x, y[0], np.sqrt(F), bg.theta)
        self.base = solve_ivp(
            steady, (0, 1), [bg.jt, 0., bg.lam*bg.theta*bg.jt], method="DOP853",
            rtol=cfg.rtol*.1, atol=cfg.rtol*.001,
            first_step=1e-4, max_step=cfg.max_step, dense_output=True)
        if not self.base.success or np.any(self.base.y[1, 1:] <= 0):
            raise ValueError("Failed to reconstruct a subsonic spectral base.")
        x = np.linspace(0, 1, cfg.points)
        g = 1-ell*x*x
        j, F, w = self.base.sol(x)
        r = np.sqrt(F)
        x_over_2r = np.divide(x, 2*r, out=np.full_like(x, .5/np.sqrt(self.launch)), where=r>0)
        eps = cfg.derivative_step
        dj, dr = eps*j, eps*np.maximum(r, 1e-3)
        jj = j+np.array([dj, -dj, 0*dj, 0*dj, 0*dj, 0*dj])
        rr = r+np.array([0*dr, 0*dr, dr, -dr, 0*dr, 0*dr])
        tt = bg.theta*(1+eps*np.array([0, 0, 0, 0, 1, -1]))[:, None]
        sources, stocks = bg.local(g, jj, rr, tt), bg.storage(jj, rr, tt)
        # F_g = mu_g-b*j_g, hence its accumulation is delta j-b*delta(AN).
        stocks[1] -= bg.b*stocks[0]
        def jacobian(values):
            return np.stack((
                -ell*x*(values[:, 0]-values[:, 1])/dj,
                -ell*x_over_2r*(values[:, 2]-values[:, 3])/dr,
                -ell*x*(values[:, 4]-values[:, 5])/eps), axis=1).transpose(2, 0, 1)
        self.coefficients = CubicSpline(x, np.stack((jacobian(sources), jacobian(stocks)), axis=1))
        self.end_flux = np.array([j[-1], F[-1]+bg.b*j[-1], w[-1]])
        target = np.array([1., bg.inlet.evaluate(bg.alpha)["mu_target"], 1.])
        self.base_residual = float(max(abs(self.end_flux-target)))
        if self.base_residual > 3e-7:
            raise ValueError(f"Spectral base does not reproduce the saved root: {self.base_residual:g}")
        self.jump = bg.storage(j[-1], r[-1], bg.theta)-bg.kinetic_storage()
        self.jump[1] = 0.
        self.gradient = bg.local(bg.alpha, j[-1], r[-1], bg.theta)
        self.gradient[1] += bg.b*self.gradient[0]
        step = min(1e-5, (1-bg.alpha)/20, (bg.alpha-bg.geometry.g_min)/20)
        mu = lambda a: bg.inlet.evaluate(a)["mu_target"]
        self.gradient[1] -= (mu(bg.alpha+step)-mu(bg.alpha-step))/(2*step)

    @lru_cache(maxsize=4096)
    def boundary_matrix(self, s):
        """Full B(s); mechanical submatrix is NEVER divided out here."""
        bg, cfg = self.bg, self.controls
        sigma = complex(s)*bg.time_star
        initial = np.array([[bg.jt, 0.], [0., 0.],
                            [bg.lam*bg.theta*bg.jt, bg.lam*bg.theta*bg.jt]], complex)
        def rhs(x, flat):
            source, stock = self.coefficients(x)
            c = source-sigma*stock
            dy = c[:, :2]@flat.reshape(3, 2)[:2]
            dy[:, 1] += c[:, 2]
            return dy.ravel()
        sol = solve_ivp(rhs, (0, 1), initial.ravel(), method="DOP853",
                        rtol=cfg.rtol, atol=cfg.rtol*.01, max_step=cfg.max_step)
        if not sol.success:
            raise ValueError(f"Linear integration failed: {sol.message}")
        endpoint = sol.y[:, -1].reshape(3, 2).copy()
        endpoint[1] += bg.b*endpoint[0]
        endpoint[1, 1] += .5*bg.b*self.end_flux[0]
        matrix = np.column_stack((endpoint[:, 0],
                                  self.gradient-sigma*self.jump, endpoint[:, 1]))
        if not np.isfinite(matrix).all():
            raise ValueError("Nonfinite B(s): spectral search is unresolved.")
        matrix.setflags(write=False)
        return matrix

    def static_response(self):
        matrix = self.boundary_matrix(0).real
        mechanical = np.linalg.solve(matrix[:2, :2], -matrix[:2, 2])
        power = matrix[2, 2]+matrix[2, :2]@mechanical
        bg = self.bg
        return np.array([bg.jt*mechanical[0], bg.p["L_cm"]*mechanical[1],
                         bg.p["W0_erg_per_s"]*1e-10*power])/bg.seed["T_eV"]

    def mode_diagnostics(self, s, amplitude, points=6001):
        """Audit a mode by direct source/storage directional differences.

        This does not differentiate the interpolated coefficient arrays again.
        All THREE integrated physical-flux balances are tested independently.
        """
        bg, cfg = self.bg, self.controls
        amplitude = np.asarray(amplitude,complex)
        a_j, _, a_T = amplitude
        initial = np.array([bg.jt*a_j, 0., bg.lam*bg.theta*bg.jt*(a_j+a_T)],complex)
        sigma = complex(s)*bg.time_star
        def rhs(x,y):
            source, stock = self.coefficients(x)
            c = source-sigma*stock
            return c[:,:2]@y[:2]+c[:,2]*a_T
        ode = solve_ivp(rhs,(0,1),initial,method="DOP853",rtol=cfg.rtol,
                        atol=cfg.rtol*.01,max_step=cfg.max_step,dense_output=True)
        if not ode.success:
            raise ValueError(ode.message)
        x = np.linspace(0,1,points)
        ell = 1-bg.alpha
        g = 1-ell*x*x
        j,F,_ = self.base.sol(x)
        r = np.sqrt(F)
        mode = ode.sol(x)
        dr = np.divide(mode[1],2*r,out=np.zeros_like(mode[1]),where=r>0)
        dstock = np.zeros((3,points),complex)
        dsource = np.zeros_like(dstock)
        # Reaction tables accept real inputs only. Differentiate separately in
        # real and imaginary directions; linearity then recovers the complex mode.
        for part, unit in ((np.real,1.),(np.imag,1j)):
            dj, rr, tt = part(mode[0]),part(dr),float(part(a_T))
            relative = max(float(max(abs(dj)/j)),
                           float(max(abs(rr)/np.maximum(r,1e-3))),abs(tt),1.)
            h = 3e-6/relative
            plus = (j+h*dj, r+h*rr, bg.theta*(1+h*tt))
            minus = (j-h*dj, r-h*rr, bg.theta*(1-h*tt))
            dstock += unit*(bg.storage(*plus)-bg.storage(*minus))/(2*h)
            fp,fm = bg.local(g,*plus),bg.local(g,*minus)
            fp[1] += np.sqrt(8*bg.beta*plus[2])*fp[0]
            fm[1] += np.sqrt(8*bg.beta*minus[2])*fm[0]
            dsource += unit*(fp-fm)/(2*h)
        flux = mode.copy()
        flux[1] += bg.b*mode[0]+.5*bg.b*j*a_T
        integral = simpson((dsource-sigma*dstock)*(2*ell*x),x=x,axis=-1)
        difference = flux[:,0]-flux[:,-1]
        errors = abs(difference-integral)/np.maximum(np.maximum(abs(difference),abs(integral)),1.)
        return dict(integrated_balance_relative_errors=errors.tolist(),
                    max_integrated_balance_error=float(max(errors)),
                    passed=bool(max(errors)<5e-5))

    def check_static(self, relative_step=.001):
        """Only perturb each saved branch locally; do not discover new branches."""
        bg = self.bg
        matrix = self.boundary_matrix(0).real
        prediction = np.linalg.solve(matrix[:2, :2], -matrix[:2, 2])
        def solve_at(factor):
            theta = bg.theta*factor
            guess = np.array([np.log(bg.jt), bg.alpha])+prediction*np.log(factor)
            def trial(x):
                state = bg.trial(theta, np.exp(x[0]), x[1], points=16)
                if state["status"] != "integrated":
                    raise ValueError(f"Static reference left admissible domain: {state['status']}")
                return state
            def residual(x):
                state = trial(x)
                return np.array([state["R_J"], state["R_M"]])
            fit = root(residual, guess, tol=1e-9)
            state = trial(fit.x)
            if max(abs(residual(fit.x))) > 3e-8:
                raise ValueError("Local static reference failed to match particle/momentum BC.")
            return np.array([state["j_target"], state["alpha"], state["omega"][0]])
        refs = []
        for h in (relative_step, relative_step/2):
            refs.append((solve_at(1+h)-solve_at(1-h))/(2*h*bg.seed["T_eV"]) *
                        np.array([1., bg.p["L_cm"], bg.p["W0_erg_per_s"]*1e-10]))
        actual = self.static_response()
        refined = LinearizedModel(bg, self.controls.refined()).static_response()
        scale = np.maximum(abs(refined), 1e-9)
        errors = [float(max(abs(a-refined)/scale)) for a in (actual, refs[-1], refs[0])]
        return dict(linear=refined.tolist(), nonlinear=refs[-1].tolist(),
                    coarse_nonlinear=refs[0].tolist(),
                    relative_errors=dict(linear_refinement=errors[0],
                                         nonlinear_reference=errors[1], coarse_reference=errors[2]),
                    passed=all(error < .002 for error in errors))
