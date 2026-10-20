"""Regression tests for roots next to finite/undefined residual boundaries."""

import unittest
from unittest.mock import patch
import numpy as np

from isothermal_cx_model.geometry import constant_B, quadratic_area_legacy
from isothermal_cx_model.scan_general import find_temperature_solutions
from isothermal_cx_model.cx import make_D


def fake_solver(function, invalid):
    def solve(theta, beta, D, lam, geometry=None, n_points=100, **kwargs):
        b = np.sqrt(8*beta*theta)
        valid = not invalid(b)
        return dict(theta=theta, b=b, D=4.0, alpha=.8,
                    R_M=float(function(b)) if valid else np.nan, R_E=0.0,
                    status="energy_matched" if valid else "sonic_return",
                    branch="monotone", max_step=.02)
    return solve


class DomainBoundaryTests(unittest.TestCase):
    def run_fake(self, function, invalid, **kwargs):
        with patch("isothermal_cx_model.scan_general.solve_at_theta",
                   fake_solver(function, invalid)):
            return find_temperature_solutions(
                .5, 4, 5, constant_B(), n_points=100, b_points=30, **kwargs)

    def test_root_next_to_lower_validity_edge(self):
        roots, diagnostic = self.run_fake(
            lambda b: b-.4, lambda b: b < .397)
        self.assertEqual(len(roots), 1)
        self.assertAlmostEqual(roots[0]["b"], .4, delta=1e-9)
        self.assertGreater(diagnostic["boundary_refinements"], 0)
        self.assertFalse(diagnostic["boundary_resolution_limited"])

    def test_root_next_to_upper_validity_edge(self):
        roots, _ = self.run_fake(lambda b: b-.4, lambda b: b > .403)
        self.assertEqual(len(roots), 1)
        self.assertAlmostEqual(roots[0]["b"], .4, delta=1e-9)

    def test_very_cold_boundary_uses_relative_resolution(self):
        roots, _ = self.run_fake(
            lambda b: (b-4e-5)/4e-5, lambda b: b < 3.97e-5)
        self.assertEqual(len(roots), 1)
        self.assertAlmostEqual(roots[0]["b"], 4e-5, delta=1e-12)

    def test_invalid_gap_is_not_removed(self):
        roots, diagnostic = self.run_fake(
            lambda b: b-.4, lambda b: .35 < b < .45)
        self.assertEqual(roots, [])
        self.assertIn("sonic_return", diagnostic["statuses"])

    def test_budget_exhaustion_is_visible(self):
        roots, diagnostic = self.run_fake(
            lambda b: b-.4, lambda b: b < .397, boundary_max_steps=1)
        self.assertTrue(diagnostic["boundary_resolution_limited"])
        self.assertTrue(diagnostic["has_unresolved_points"])

    def test_all_invalid_is_not_a_sign_change(self):
        roots, diagnostic = self.run_fake(
            lambda b: b-.4, lambda b: True)
        self.assertEqual(roots, [])
        self.assertEqual(diagnostic["boundary_evaluations"], 0)

    def test_boundary_option_validation(self):
        for kwargs in [
            {"boundary_rtol": 0}, {"boundary_rtol": np.nan},
            {"boundary_max_steps": 0}, {"boundary_max_steps": 2.5},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.run_fake(lambda b: b-.4, lambda b: False, **kwargs)


class ActualMasterParametersTests(unittest.TestCase):
    @staticmethod
    def parameters():
        J, M, W, mass = 4.5e21, 1.85e5, 1.4e13, 3.3435837724e-24
        u = M/(mass*J)
        return dict(
            beta=(W/J)/(mass*u*u), T_star_eV=(W/J)/1.602176634e-12,
            U_star_cm_per_s=u, L_cm=184.5, n0_cm3=1e14,
            k_cx_cm3_per_s=1e-8,
        )

    def check_density(self, n0, expected_T, expected_a):
        p = self.parameters()
        D = make_D(p, mode="temperature", n0_cm3=n0)
        roots, diagnostic = find_temperature_solutions(
            p["beta"], D, 5, quadratic_area_legacy(),
            n_points=120, b_points=100)
        self.assertEqual(len(roots), 1)
        self.assertAlmostEqual(roots[0]["theta"]*p["T_star_eV"],
                               expected_T, delta=max(2e-8, 2e-6*expected_T))
        self.assertAlmostEqual(roots[0]["alpha"]*p["L_cm"],
                               expected_a, delta=2e-4)
        self.assertEqual(roots[0]["branch"], "with_maximum")
        self.assertLess(abs(roots[0]["R_M_check"]), 1e-7)
        self.assertFalse(diagnostic["has_unresolved_points"])

    def test_missing_root_at_1_584893e14(self):
        self.check_density(1.5848931924611108e14, .229688, 30.6009)

    def test_missing_root_at_1e15(self):
        self.check_density(1e15, .00501291, 30.3526)


if __name__ == "__main__":
    unittest.main()
