"""Bounded numerical normal-mode search, NOT a global stability proof."""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares
from .adaptive_contour import contour_count_adaptive


class SearchUnresolved(RuntimeError):
    pass


@dataclass(frozen=True)
class SpectralBox:
    real_min: float
    real_max: float
    imag_min: float
    imag_max: float

    def __post_init__(self):
        if (not np.isfinite([self.real_min, self.real_max, self.imag_min, self.imag_max]).all()
                or not self.real_min < self.real_max or not self.imag_min < self.imag_max):
            raise ValueError("Spectral bounds must be finite and ordered.")

    def contains(self, s):
        return self.real_min < s.real < self.real_max and self.imag_min < s.imag < self.imag_max

    def corners(self):
        return [complex(self.real_min, self.imag_min), complex(self.real_max, self.imag_min),
                complex(self.real_max, self.imag_max), complex(self.real_min, self.imag_max)]

    def split(self):
        # Do not repeatedly put the real axis on an internal contour.
        re = self.real_min+.473*(self.real_max-self.real_min)
        im = self.imag_min+.527*(self.imag_max-self.imag_min)
        return [SpectralBox(a,b,c,d) for a,b in ((self.real_min,re),(re,self.real_max))
                for c,d in ((self.imag_min,im),(im,self.imag_max))]


def equilibrate(matrix):
    """Positive scaling preserves zeros and phase; no mechanical elimination."""
    row = np.maximum(np.max(abs(matrix), axis=1), np.finfo(float).tiny)
    normalized = matrix/row[:, None]
    col = np.maximum(np.max(abs(normalized), axis=0), np.finfo(float).tiny)
    return normalized/col, row, col


def determinant(matrix):
    # A single scalar scale cannot normalize a vanishing row/column back to 1.
    # Per-row scaling at every Newton iterate would turn diag(s-s0,1,1)
    # into a pure phase and destroy the root-finding residual.
    scale = max(np.max(abs(matrix)),np.finfo(float).tiny)
    return np.linalg.det(matrix/scale)


def contour_count(matrix_at, box, start_points=16, max_points=1024, progress=None):
    """Argument-principle count with sampling refinement and phase-step guard.

    This is a convergence check, not interval-arithmetic certification.
    An unresolved count MUST NOT be interpreted as zero.
    """
    previous, history = None, []
    n = start_points
    while n <= max_points:
        corners = box.corners()
        path = np.concatenate([np.linspace(a,b,n,endpoint=False) for a,b in
                               zip(corners,corners[1:]+corners[:1])] + [[corners[0]]])
        phases = []
        for s in path:
            sign, log_abs = np.linalg.slogdet(equilibrate(matrix_at(complex(s)))[0])
            if abs(sign) == 0 or not np.isfinite(log_abs):
                raise SearchUnresolved(f"Unresolved determinant on contour at s={complex(s)}: "
                    f"sign={sign}, log_abs={log_abs}. This can be a contour root OR numerical loss of rank.")
            phases.append(sign)
        increments = np.angle(np.array(phases[1:])/phases[:-1])
        raw = float(sum(increments)/(2*np.pi))
        count = int(round(raw))
        step = float(max(abs(increments)))
        history.append(dict(points_per_edge=n, count=count, max_phase_step=step))
        if progress is not None:
            progress(f"contour: {n} points/edge, count={count}, max phase step={step:.4g}")
        if count == previous and step < .65 and abs(raw-count) < 1e-6 and count >= 0:
            return count, history
        previous, n = count, n*2
    raise SearchUnresolved(f"Contour count did not converge: {history}")


def find_modes(model, box, max_depth=7, growth_tolerance=1e-6, method="direct", progress=None,
               contour_method="adaptive",contour_options=None):
    """Search a finite box and repeat counts/eigenpairs on refined coefficients.

    B columns correspond to (delta log j_t, delta alpha, delta log theta).
    This accepts temperature-free modes too.
    """
    if method not in ("direct","qr"):
        raise ValueError("Spectral transport must be direct or qr.")
    if contour_method not in ("adaptive","uniform"):
        raise ValueError("Contour method must be adaptive or uniform.")
    refined = type(model)(model.bg, model.controls.refined())
    matrix = model.qr_boundary_matrix if method == "qr" else model.boundary_matrix
    matrix_ref = refined.qr_boundary_matrix if method == "qr" else refined.boundary_matrix
    options = {} if contour_options is None else dict(contour_options)
    # Share phase evaluations between outer contour and subrectangles, but
    # NEVER between initial and refined coefficient grids.
    phase_cache,phase_cache_ref = {},{}
    def count_in(provider,region,log=None):
        if contour_method == "uniform":
            return contour_count(provider,region,progress=log)
        cache = phase_cache_ref if provider == matrix_ref else phase_cache
        return contour_count_adaptive(provider,region,progress=log,cache=cache,**options)
    _, fixed_row, fixed_col = equilibrate(matrix_ref(0.))
    if progress is not None:
        progress(f"{method}: counting on the initial coefficient grid")
    total, history = count_in(matrix,box,progress)
    if progress is not None:
        progress(f"{method}: counting on the refined coefficient grid")
    total_ref, refined_history = count_in(matrix_ref,box,progress)
    if total != total_ref:
        raise SearchUnresolved("Contour counts disagree between coefficient grids.")
    scale = max(box.real_max-box.real_min, box.imag_max-box.imag_min, 1.)
    found = []

    def refine_root(region, guess=None, solver=matrix):
        centre = complex((region.real_min+region.real_max)/2,
                         (region.imag_min+region.imag_max)/2)
        # Freeze the residual preconditioner at the initial guess. Dividing by
        # an s-dependent large matrix norm can make a nonzero determinant tiny
        # far from a root, especially when the interface column grows as s.
        initial_guess = guess if guess is not None else centre
        B0 = solver(initial_guess)
        _, row0, _ = equilibrate(B0)
        row0 = np.maximum(row0,1e-12*max(float(max(row0)),1.))
        col0 = np.maximum(np.max(abs(B0/row0[:,None]),axis=0),1e-12)
        def residual(y):
            value = np.linalg.det(solver(complex(*y)*scale)/row0[:,None]/col0)
            return [value.real, value.imag]
        # A local candidate is sought ONLY in its contour-counted rectangle.
        # An unconstrained Newton step could request a huge s far outside it,
        # overflowing direct shooting or exhausting the QR interval budget.
        fit = least_squares(residual,[(initial_guess/scale).real,
                                     (initial_guess/scale).imag],
            bounds=([region.real_min/scale,region.imag_min/scale],
                    [region.real_max/scale,region.imag_max/scale]),
            xtol=1e-11,ftol=1e-11,gtol=1e-11,max_nfev=80)
        candidate = complex(*fit.x)*scale
        if not region.contains(candidate) or np.linalg.norm(fit.fun) > 1e-8:
            return None
        # real coefficients: remove a numerical imaginary remainder only if tested.
        if abs(candidate.imag) < 1e-8*max(1.,abs(candidate.real)):
            real_candidate = complex(candidate.real)
            if np.linalg.norm(residual([real_candidate.real/scale,0.])) < 1e-8:
                candidate = real_candidate
        return candidate

    def visit(region, expected, depth):
        if expected == 0:
            return
        if expected == 1:
            candidate = refine_root(region)
            if candidate is not None:
                verified = refine_root(region, candidate, matrix_ref)
                if verified is not None:
                    found.append((candidate,verified))
                    return
        if depth >= max_depth:
            raise SearchUnresolved("Root localization reached maximum subdivision depth.")
        children = region.split()
        counts = [count_in(matrix,child,progress)[0] for child in children]
        if sum(counts) != expected:
            raise SearchUnresolved("Subcontour counts do not sum to the parent count.")
        for child, count in zip(children,counts):
            visit(child,count,depth+1)

    visit(box,total,0)
    if len(found) != total:
        raise SearchUnresolved("Not all contour-counted eigenvalues were located.")
    modes = []
    for coarse, s in sorted(found,key=lambda pair:(pair[1].real,pair[1].imag)):
        if any(abs(s-complex(*r["s_per_s"])) < 1e-6*max(abs(s),1.) for r in modes):
            raise SearchUnresolved("Duplicate root localization; completeness is unresolved.")
        B = matrix_ref(s)
        normalized, col = B/fixed_row[:,None]/fixed_col, fixed_col
        _, sv, vh = np.linalg.svd(normalized)
        amplitude = vh[-1].conj()/col
        amplitude /= max(abs(amplitude))
        pivot = np.argmax(abs(amplitude))
        amplitude *= np.exp(-1j*np.angle(amplitude[pivot]))
        residual = float(sv[-1]/sv[0])
        change = float(abs(s-coarse)/max(abs(s),1.))
        if residual > 1e-7 or change > .002:
            raise SearchUnresolved(f"Eigenpair refinement failed: residual={residual:g}, change={change:g}")
        # Direct BC residual, after the same row scaling, is independently reported.
        bc_residual = float(np.linalg.norm(normalized@(amplitude*col)) /
                            max(np.linalg.norm(amplitude*col),np.finfo(float).tiny))
        qr_coordinates = amplitude.copy() if method == "qr" else None
        if method == "qr":
            amplitude,audit = refined.qr_mode_diagnostics(s,amplitude)
            rotation = np.exp(-1j*np.angle(amplitude[np.argmax(abs(amplitude))]))
            amplitude *= rotation
            qr_coordinates *= rotation
        else:
            audit = refined.mode_diagnostics(s,amplitude)
        if not audit["passed"]:
            raise SearchUnresolved(f"Mode failed independent integrated balances: {audit}")
        mode = dict(s_per_s=[float(s.real),float(s.imag)],
            sigma=[float(s.real*model.bg.time_star),float(s.imag*model.bg.time_star)],
            growth_time_s=None if s.real <= 0 else float(1/s.real),
            frequency_Hz=float(abs(s.imag)/(2*np.pi)),
            singular_ratio=residual, boundary_residual=bc_residual,
            relative_refinement=change, independent_audit=audit,
            amplitudes=[[float(a.real),float(a.imag)] for a in amplitude])
        if qr_coordinates is not None:
            mode["qr_coordinates"] = [[float(a.real),float(a.imag)] for a in qr_coordinates]
        modes.append(mode)
    growing = any(r["s_per_s"][0] > growth_tolerance for r in modes)
    neutral = any(abs(r["s_per_s"][0]) <= growth_tolerance for r in modes)
    status = ("unstable" if growing else "near_neutral_unresolved" if neutral
              else "no_growing_modes_found_in_box")
    # Cross-check real-coefficient conjugation at a generic point.
    probe = complex(.37*box.real_max+.63*box.real_min,
                    .31*box.imag_max+.69*box.imag_min)
    symmetry = float(np.linalg.norm(matrix_ref(probe.conjugate())-matrix_ref(probe).conjugate()) /
                     max(np.linalg.norm(matrix_ref(probe)),1e-100))
    if symmetry > 1e-8:
        raise SearchUnresolved("Complex conjugation check failed.")
    return dict(status=status, transport=method, contour_method=contour_method,
                box_s_per_s=vars(box), root_count=total_ref, modes=modes,
                contour=history, refined_contour=refined_history,
                conjugation_error=symmetry, global_stability_proven=False)
