"""Magnetic geometry. Coordinates are dimensionless: g = z/L."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import argparse
import json

import numpy as np
from scipy.interpolate import PchipInterpolator


@dataclass(frozen=True)
class MagneticGeometry:
    """Positive B/B_m and its logarithmic derivative on [g_min, 1].

    Callables must accept numpy arrays or scalars. A/A_m is always
    calculated from B/B_m, never specified independently.
    """
    name: str
    field_function: Callable
    derivative_function: Callable
    g_min: float = 0.0
    specification: dict = field(default_factory=dict)

    def __post_init__(self):
        if not np.isfinite(self.g_min) or not 0 <= self.g_min < 1:
            raise ValueError("Geometry requires 0 <= g_min < 1.")
        self.B_ratio(np.linspace(self.g_min, 1.0, 257))
        self.dlnB_dg(np.linspace(self.g_min, 1.0, 257))
        self.A_ratio(np.linspace(self.g_min, 1.0, 257))

    def _evaluate(self, function, g, positive=False):
        points = np.asarray(g, dtype=float)
        tolerance = 32 * np.finfo(float).eps
        if (not np.isfinite(points).all()
                or np.any(points < self.g_min - tolerance)
                or np.any(points > 1 + tolerance)):
            raise ValueError("Requested point is outside the geometry domain.")
        points = np.clip(points, self.g_min, 1.0)
        values = np.asarray(function(points), dtype=float)
        values = np.broadcast_to(values, points.shape)
        if not np.isfinite(values).all() or (positive and np.any(values <= 0)):
            raise ValueError("Geometry returned a non-finite or non-positive field.")
        return float(values) if values.ndim == 0 else values.copy()

    def B_ratio(self, g):
        return self._evaluate(self.field_function, g, positive=True)

    def dlnB_dg(self, g):
        return self._evaluate(self.derivative_function, g)

    def A_ratio(self, g):
        with np.errstate(over="ignore", divide="ignore"):
            ratio = 1.0 / self.B_ratio(g)
        if not np.isfinite(ratio).all():
            raise ValueError("Reciprocal field produced a non-finite area.")
        return ratio

    def to_dict(self):
        return dict(self.specification or {
            "type": "callable", "name": self.name, "g_min": self.g_min,
        })


def linear_B(K=500.0):
    K = float(K)
    if not np.isfinite(K) or K <= 0:
        raise ValueError("K must be finite and positive.")
    epsilon = 1.0 - 1.0 / K
    return MagneticGeometry(
        "linear_B",
        lambda g: 1.0 - epsilon * g,
        lambda g: -epsilon / (1.0 - epsilon * g),
        specification={"type": "linear_B", "K": K},
    )


def constant_B():
    return MagneticGeometry(
        "constant_B", lambda g: np.ones_like(g), lambda g: np.zeros_like(g),
        specification={"type": "constant_B"},
    )


def quadratic_area_legacy(K=500.0, g_min=1e-6):
    """Legacy A/A_m=K*g**2, NOT a finite throat profile at g=0."""
    K, g_min = float(K), float(g_min)
    if not np.isfinite(K) or K <= 0 or not 0 < g_min < 1:
        raise ValueError("Legacy geometry requires K>0 and 0<g_min<1.")
    return MagneticGeometry(
        "quadratic_area_legacy",
        lambda g: 1.0 / (K * g**2), lambda g: -2.0 / g,
        g_min=g_min,
        specification={"type": "quadratic_area_legacy", "K": K,
                       "g_min": g_min},
    )


def tabulated_B(g, B_ratio):
    g, values = np.asarray(g, dtype=float), np.asarray(B_ratio, dtype=float)
    if (g.ndim != 1 or values.shape != g.shape or g.size < 2
            or not np.isfinite(g).all() or not np.isfinite(values).all()
            or np.any(np.diff(g) <= 0) or np.any(values <= 0)
            or g[0] < 0 or g[0] >= 1 or g[-1] != 1):
        raise ValueError("Table requires ordered g in [0,1], ending at 1, and B>0.")
    interpolant = PchipInterpolator(g, np.log(values), extrapolate=False)
    derivative = interpolant.derivative()
    return MagneticGeometry(
        "tabulated_B", lambda x: np.exp(interpolant(x)), derivative,
        g_min=float(g[0]),
        specification={"type": "tabulated_B", "g": g.tolist(),
                       "B_ratio": values.tolist()},
    )


def load_geometry(path):
    """Read an explicit JSON specification. No eval and no extrapolation."""
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("Geometry JSON must contain an object.")
    kind = data.get("type")
    if kind == "linear_B":
        return linear_B(data.get("K", 500.0))
    if kind == "constant_B":
        return constant_B()
    if kind == "quadratic_area_legacy":
        return quadratic_area_legacy(data.get("K", 500.0), data.get("g_min", 1e-6))
    if kind == "tabulated_B":
        return tabulated_B(data["g"], data["B_ratio"])
    raise ValueError(f"Unknown geometry type: {kind!r}")


def add_geometry_arguments(parser):
    parser.add_argument("--geometry", type=Path,
                        help="Geometry JSON; omitted: legacy quadratic area.")
    parser.add_argument("--K", type=float, default=None,
                        help="Legacy geometry only; incompatible with --geometry.")


def geometry_from_args(args):
    if args.geometry is not None:
        if args.K is not None:
            raise ValueError("Use either --geometry or --K, not both.")
        return load_geometry(args.geometry)
    return quadratic_area_legacy(500.0 if args.K is None else args.K)
