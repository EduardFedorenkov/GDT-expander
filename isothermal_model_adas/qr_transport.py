"""Short-interval propagation of a two-dimensional complex solution space.

The QR diagonal is real positive. Consequently det(R_total)>0: removing its
magnitude preserves BOTH zeros and phase of the boundary determinant.
We store log(det R_total), never the exponentially large product itself.
"""

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class QRTransfer:
    basis: np.ndarray
    log_volume: float
    segments: int
    max_condition: float
    orthogonality_error: float
    history: dict | None = None


def positive_qr(values):
    """Reduced complex QR with positive diagonal; no discarded complex phase."""
    Q,R = np.linalg.qr(values,mode="reduced")
    diagonal = np.diag(R)
    magnitude = abs(diagonal)
    if not np.isfinite(values).all() or np.any(magnitude == 0):
        raise ValueError("QR basis is nonfinite or has lost rank.")
    phase = diagonal/magnitude
    Q = Q*phase
    R = phase.conjugate()[:,None]*R
    R[np.diag_indices_from(R)] = magnitude
    return Q,R


def propagate_basis(matrix_at, initial, *, rtol=2e-10, max_step=.025,
                    max_growth=2., max_segments=20000, retain_history=False):
    """Propagate V'=A(x)V on [0,1], reorthogonalizing at every interval.

    max_growth limits the estimated |eigenvalue(A)|*interval. It is not a
    rigorous norm bound: integration and the actual local R conditioning are
    checked too. Failed intervals are retried at smaller length.
    """
    if not (0 < rtol < .01 and 0 < max_step <= 1 and 0 < max_growth <= 10
            and isinstance(max_segments,int) and max_segments > 0):
        raise ValueError("Invalid QR integration controls.")
    initial = np.asarray(initial,complex)
    if initial.ndim != 2 or initial.shape[1] != 2 or initial.shape[0] < 2:
        raise ValueError("QR transport expects two independent augmented columns.")
    Q,R0 = positive_qr(initial)
    log_volume = float(sum(np.log(abs(np.diag(R0)))))
    x, count = 0.,0
    max_condition = float(np.linalg.cond(R0))
    orth_error = float(np.linalg.norm(Q.conjugate().T@Q-np.eye(2)))
    history = dict(nodes=[0.],bases=[Q.copy()],triangles=[],solves=[]) if retain_history else None
    while x < 1:
        if count >= max_segments:
            raise RuntimeError(f"QR segment budget exceeded at x={x:.8g}; increase --qr-max-segments or reduce the box.")
        step = min(max_step,1-x)
        for _ in range(2):
            radius = max(float(max(abs(np.linalg.eigvals(matrix_at(point)))))
                         for point in (x,x+step/2,x+step))
            if not np.isfinite(radius):
                raise ValueError(f"Nonfinite QR generator at x={x:.8g}.")
            if radius > 0:
                step = min(step,max_growth/radius)
        accepted = False
        for attempt in range(18):
            if step < 32*np.finfo(float).eps and 1-x > 64*np.finfo(float).eps:
                raise RuntimeError(f"QR interval became too small at x={x:.8g}.")
            end = min(1.,x+step)
            if 1-end < 64*np.finfo(float).eps:
                end = 1.
            def rhs(point, flat):
                return (matrix_at(point)@flat.reshape(initial.shape)).ravel()
            try:
                with warnings.catch_warnings(), np.errstate(over="raise",invalid="raise",divide="raise"):
                    warnings.simplefilter("error",RuntimeWarning)
                    solve = solve_ivp(rhs,(x,end),Q.ravel(),method="DOP853",
                        rtol=rtol,atol=rtol*.01,max_step=step,
                        dense_output=retain_history)
                if not solve.success or not np.isfinite(solve.y[:,-1]).all():
                    raise ValueError("Local QR integration failed.")
                next_Q,R = positive_qr(solve.y[:,-1].reshape(initial.shape))
                condition = float(np.linalg.cond(R))
                if not np.isfinite(condition) or condition > 1e4:
                    raise ValueError("Local QR columns are too ill-conditioned.")
                accepted = True
                break
            except (ValueError,FloatingPointError,RuntimeWarning):
                step /= 2
        if not accepted:
            raise RuntimeError(f"Could not resolve a QR interval at x={x:.8g}.")
        log_volume += float(sum(np.log(abs(np.diag(R)))))
        max_condition = max(max_condition,condition)
        orth_error = max(orth_error,float(np.linalg.norm(next_Q.conjugate().T@next_Q-np.eye(2))))
        count += 1
        x,Q = end,next_Q
        if history is not None:
            history["nodes"].append(x)
            history["bases"].append(Q.copy())
            history["triangles"].append(R)
            history["solves"].append(solve.sol)
    return QRTransfer(Q,float(log_volume),count,max_condition,orth_error,history)


def reconstruct_mode(transfer, inlet_coefficients, eta, grid):
    """Back-substitute short R factors; never reconstruct their full product.

    Returns a bounded augmented mode and the consistently rescaled eta.
    Values exponentially smaller than the peak may underflow to zero; the
    independent mode audit must still pass before accepting a root.
    """
    hist = transfer.history
    if hist is None:
        raise ValueError("Mode reconstruction requires retained QR intervals.")
    count = transfer.segments
    unit = np.asarray(inlet_coefficients,complex)
    norm = float(np.linalg.norm(unit))
    if norm == 0 or not np.isfinite(norm):
        raise ValueError("Invalid null-vector coefficients.")
    directions = [None]*(count+1)
    logs = np.empty(count+1)
    directions[-1],logs[-1] = unit/norm,np.log(norm)
    for k in range(count-1,-1,-1):
        vector = np.linalg.solve(hist["triangles"][k],directions[k+1])
        norm = float(np.linalg.norm(vector))
        if norm == 0 or not np.isfinite(norm):
            raise ValueError("QR mode back-substitution lost representable precision.")
        directions[k],logs[k] = vector/norm,logs[k+1]+np.log(norm)
    offset = max(float(max(logs)),float(np.log(abs(eta))) if abs(eta)>0 else -np.inf,0.)
    grid = np.asarray(grid,float)
    result = np.zeros((transfer.basis.shape[0],len(grid)),complex)
    nodes = np.asarray(hist["nodes"])
    indices = np.clip(np.searchsorted(nodes,grid,side="right")-1,0,count-1)
    for k in np.unique(indices):
        selected = np.flatnonzero(indices == k)
        coefficients = directions[k]*np.exp(logs[k]-offset)
        local = hist["solves"][k](grid[selected]).reshape(transfer.basis.shape[0],2,-1)
        result[:,selected] = np.einsum("ijk,j->ik",local,coefficients)
    return result,complex(eta)*np.exp(-offset)
