"""Adaptive argument-principle sampling on a rectangular complex contour.

This is a sampled convergence test, NOT a certified exclusion of hidden
winding between samples. It does not silently move the physical search box.
"""

import numpy as np


class AdaptiveContourError(RuntimeError):
    def __init__(self,message,diagnostic):
        super().__init__(message)
        self.diagnostic = diagnostic


def contour_count_adaptive(matrix_at,box,*,start_points=16,max_points=8192,
        max_depth=20,phase_limit=.65,curvature_limit=.25,log_limit=2.,
        max_rounds=3,progress=None,cache=None):
    """Refine only difficult intervals, then repeat with stricter criteria.

    Each interval is probed at 1/4,1/2,3/4 before acceptance. Rejection tests:
    phase step, phase curvature, missed winding vs the endpoint shortcut,
    and balanced log-modulus variation. A second pass doubles the initial
    sampling and halves all thresholds. Counts must agree.

    max_points is a per-contour budget of distinct sampled s, across passes.
    The optional phase cache may be shared with subcontours of the SAME
    matrix provider. Never share it between different coefficient grids.
    """
    if (not isinstance(start_points,int) or start_points < 2
            or not isinstance(max_points,int) or max_points < 16
            or not isinstance(max_depth,int) or max_depth < 1
            or not isinstance(max_rounds,int) or max_rounds < 2
            or not np.isfinite([phase_limit,curvature_limit,log_limit]).all()
            or not 0 < phase_limit < np.pi/2 or curvature_limit <= 0 or log_limit <= 0):
        raise ValueError("Invalid adaptive contour controls.")
    cache = {} if cache is None else cache
    visited = set()
    history = []
    evaluations = 0
    worst_location = None
    smallest_log = (np.inf,None)

    def coordinates(s):
        return [float(s.real),float(s.imag)]

    def fail(message,a=None,b=None,depth=None,**extra):
        detail = dict(method="adaptive",history=history.copy(),
            distinct_points=len(visited),new_matrix_evaluations=evaluations,
            interval=None if a is None else [coordinates(a),coordinates(b)],
            depth=depth,min_balanced_log_abs=None if smallest_log[1] is None else float(smallest_log[0]),
            min_log_location=None if smallest_log[1] is None else coordinates(smallest_log[1]),
            **extra)
        raise AdaptiveContourError(message,detail)

    def sample(s,a,b,depth):
        nonlocal evaluations,smallest_log
        s = complex(s)
        if s not in visited:
            if len(visited) >= max_points:
                fail(f"Adaptive contour point budget {max_points} exceeded near s={s}.",
                     a,b,depth,failed_point=coordinates(s))
            visited.add(s)
        if s not in cache:
            try:
                matrix = np.asarray(matrix_at(s),complex)
                if matrix.shape != (3,3) or not np.isfinite(matrix).all():
                    raise ValueError("Expected a finite 3x3 boundary matrix.")
                row = np.maximum(np.max(abs(matrix),axis=1),np.finfo(float).tiny)
                normalized = matrix/row[:,None]
                col = np.maximum(np.max(abs(normalized),axis=0),np.finfo(float).tiny)
                sign,log_abs = np.linalg.slogdet(normalized/col)
                if abs(sign) == 0 or not np.isfinite(log_abs):
                    fail(f"Zero/nonfinite determinant on adaptive contour at s={s}.",
                         a,b,depth,failed_point=coordinates(s))
            except AdaptiveContourError:
                raise
            except (ValueError,RuntimeError,FloatingPointError,OverflowError) as error:
                fail(f"Adaptive contour evaluation failed at s={s}: {error}",
                     a,b,depth,failed_point=coordinates(s))
            cache[s] = (complex(sign),float(log_abs))
            evaluations += 1
            if progress is not None and evaluations % 256 == 0:
                progress(f"adaptive: {len(visited)} distinct points, {evaluations} new matrix evaluations")
        value = cache[s]
        if value[1] < smallest_log[0]:
            smallest_log = (value[1],s)
        return value

    previous = None
    for iteration in range(max_rounds):
        n = start_points*2**iteration
        limit = phase_limit/2**iteration
        bend = curvature_limit/2**iteration
        modulus_limit = log_limit/2**iteration
        accepted = []
        deepest = 0
        worst_step = 0.

        def refine(a,b,depth):
            nonlocal deepest,worst_step,worst_location
            deepest = max(deepest,depth)
            points = [a,a+(b-a)/4,(a+b)/2,a+3*(b-a)/4,b]
            values = [sample(s,a,b,depth) for s in points]
            signs = np.array([v[0] for v in values])
            logs = np.array([v[1] for v in values])
            delta = np.angle(signs[1:]/signs[:-1])
            largest = float(max(abs(delta)))
            curvature = float(max(abs(np.diff(delta))))
            shortcut = float(np.angle(signs[-1]/signs[0]))
            hidden_turn = abs(float(sum(delta))-shortcut) > 1e-6
            modulus_jump = float(max(abs(np.diff(logs))))
            modulus_bend = float(max(abs(np.diff(logs,n=2))))
            split = (largest > limit or curvature > bend or hidden_turn
                     or modulus_jump > modulus_limit or modulus_bend > modulus_limit)
            if split:
                if depth >= max_depth:
                    fail(f"Adaptive contour depth {max_depth} exhausted on [{a}, {b}].",
                         a,b,depth,max_phase_step=largest,
                         phase_curvature=curvature,hidden_turn=hidden_turn)
                middle = points[2]
                if middle == a or middle == b:
                    fail("Adaptive contour interval is below floating-point resolution.",a,b,depth)
                refine(a,middle,depth+1)
                refine(middle,b,depth+1)
            else:
                accepted.extend(delta.tolist())
                if largest >= worst_step:
                    worst_step = largest
                    worst_location = [coordinates(a),coordinates(b)]

        corners = box.corners()
        if progress is not None:
            progress(f"adaptive pass {iteration+1}: seed={n}/edge, phase limit={limit:.4g}")
        for a,b in zip(corners,corners[1:]+corners[:1]):
            points = np.linspace(a,b,n+1)
            for left,right in zip(points[:-1],points[1:]):
                refine(complex(left),complex(right),0)
        raw = float(sum(accepted)/(2*np.pi))
        count = int(round(raw))
        row = dict(pass_index=iteration+1,initial_points_per_edge=n,count=count,
            raw_winding=raw,integer_error=abs(raw-count),
            phase_limit=limit,max_phase_step=worst_step,max_depth_used=deepest,
            accepted_segments=len(accepted),distinct_points=len(visited),
            new_matrix_evaluations=evaluations,worst_accepted_interval=worst_location,
            min_balanced_log_abs=float(smallest_log[0]),
            min_log_location=coordinates(smallest_log[1]))
        history.append(row)
        if progress is not None:
            progress(f"adaptive pass {iteration+1}: count={count}, points={len(visited)}, "
                     f"max phase step={worst_step:.4g}, depth={deepest}")
        if count == previous and count >= 0 and abs(raw-count) < 1e-6:
            return count,history
        previous = count
    fail(f"Adaptive contour count did not stabilize after {max_rounds} passes.")
