"""Public API for the simple one-dimensional charge-exchange model."""

from .config import load_cx_config
from .plotting import plot_cx_profiles, plot_solution_region
from .profiles import cx_profiles
from .solver import find_alpha, find_theta, has_theta_solution

__all__ = [
    "cx_profiles",
    "find_alpha",
    "find_theta",
    "has_theta_solution",
    "load_cx_config",
    "plot_cx_profiles",
    "plot_solution_region",
]
