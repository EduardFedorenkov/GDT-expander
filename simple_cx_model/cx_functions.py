"""Backward-compatible imports for the original single-module API."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simple_cx_model.config import load_cx_config
from simple_cx_model.plotting import plot_cx_profiles, plot_solution_region
from simple_cx_model.profiles import cx_profiles
from simple_cx_model.solver import find_alpha, find_theta, has_theta_solution

__all__ = [
    "cx_profiles",
    "find_alpha",
    "find_theta",
    "has_theta_solution",
    "load_cx_config",
    "plot_cx_profiles",
    "plot_solution_region",
]
