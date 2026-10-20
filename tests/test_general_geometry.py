"""Regression and interface tests. Run: python -B -m unittest discover -s tests -v."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import brentq

from isothermal_cx_model.geometry import (
    MagneticGeometry, linear_B, constant_B, quadratic_area_legacy,
    tabulated_B, load_geometry,
)
from isothermal_cx_model.solver_general import solve_at_theta, classify_profile
from isothermal_cx_model.scan_general import find_temperature_solutions
from isothermal_cx_model.scan_neutral import find_density_solutions
from isothermal_cx_model.profiles import compute_profiles


def reference_parameters():
    J, M, W, mass = 1e20, 3000.0, 1.4e10, 3.3435837724e-24
    u = M/(mass*J)
    area = 38.48451000647496
    return dict(
        J0_per_s=J, M0_dyn=M, W0_erg_per_s=W, m_D_g=mass,
        U_star_cm_per_s=u, T_star_erg=W/J, T_star_eV=W/J/1.602176634e-12,
        beta=(W/J)/(mass*u*u), L_cm=184.5, n0_cm3=2.8e13,
        k_cx_cm3_per_s=1e-8, lambda_sh=5.0, A0_cm2=area,
        n_star_cm3=J/(area*u),
    )


class GeometryTests(unittest.TestCase):
    def test_linear_endpoints_and_derivative(self):
        geometry = linear_B(500)
        self.assertAlmostEqual(geometry.B_ratio(0), 1)
        self.assertAlmostEqual(geometry.B_ratio(1), 1/500)
        self.assertAlmostEqual(geometry.A_ratio(1), 500)
        self.assertAlmostEqual(geometry.dlnB_dg(1), -499)
        g = np.linspace(0, 1, 11)
        np.testing.assert_allclose(geometry.B_ratio(g)*geometry.A_ratio(g), 1)

    def test_legacy_geometry_not_extended_to_throat(self):
        geometry = quadratic_area_legacy(500, 1e-4)
        self.assertAlmostEqual(geometry.A_ratio(.7), 500*.7**2)
        self.assertAlmostEqual(geometry.dlnB_dg(.7), -2/.7)
        with self.assertRaises(ValueError):
            geometry.B_ratio(0)

    def test_constant_geometry(self):
        geometry = constant_B()
        np.testing.assert_allclose(geometry.B_ratio(np.linspace(0,1,7)), 1)
        self.assertEqual(geometry.dlnB_dg(.3), 0)

    def test_tabulated_log_interpolation(self):
        g = [0, .2, .8, 1]
        values = [1, .7, .03, .002]
        geometry = tabulated_B(g, values)
        np.testing.assert_allclose(geometry.B_ratio(g), values)
        self.assertTrue(np.all(geometry.B_ratio(np.linspace(0,1,101)) > 0))
        self.assertTrue(np.isfinite(geometry.dlnB_dg(np.linspace(0,1,101))).all())

    def test_table_invalid(self):
        for g, values in [([0,1],[1,0]), ([0,.5],[1,.1]),
                          ([0,.5,.4,1],[1,.8,.7,.1])]:
            with self.assertRaises(ValueError):
                tabulated_B(g, values)

    def test_no_extrapolation(self):
        with self.assertRaises(ValueError):
            linear_B().B_ratio(1.01)

    def test_bad_K(self):
        for K in [0, -1, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                linear_B(K)

    def test_custom_callable(self):
        geometry = MagneticGeometry(
            "exponential", lambda g: np.exp(-2*g),
            lambda g: -2*np.ones_like(g),
        )
        self.assertAlmostEqual(geometry.A_ratio(.5), np.e)

    def test_json_round_trip(self):
        for geometry in [constant_B(), linear_B(), quadratic_area_legacy(),
                         tabulated_B([0,.5,1], [1,.1,.002])]:
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/"geometry.json"
                path.write_text(json.dumps(geometry.to_dict()), encoding="utf-8")
                restored = load_geometry(path)
                g = np.linspace(max(geometry.g_min, 1e-5),1,50)
                np.testing.assert_allclose(restored.B_ratio(g), geometry.B_ratio(g))


class SolverTests(unittest.TestCase):
    def test_constant_area_analytic_mu(self):
        state = solve_at_theta(.1, .5, 4, 5, constant_B(), n_points=250)
        self.assertEqual(state["status"], "energy_matched")
        expected = state["b"] + 4*(1-state["g"])
        np.testing.assert_allclose(state["mu"], expected, atol=1e-9)
        self.assertAlmostEqual(state["omega"][0], 1)
        self.assertAlmostEqual(state["omega"][-1], .5)
        self.assertGreater(abs(state["R_M"]), 1e-3)

    def test_constant_area_full_analytic_root(self):
        beta, D, lam = .5, 4.0, 5.0
        def energy_residual(b):
            radical = np.sqrt(1-b*b)
            integral = 2.5*(1-b*b) + .5*(
                radical-b*b*np.log((1+radical)/b))
            return lam*b*b/(8*beta) + integral/(8*beta)-1
        expected_b = brentq(energy_residual, .1, .99)
        states, _ = find_temperature_solutions(
            beta, D, lam, constant_B(), n_points=100, b_points=25)
        self.assertEqual(len(states), 1)
        self.assertAlmostEqual(states[0]["b"], expected_b, delta=1e-7)
        self.assertAlmostEqual(states[0]["alpha"],
                               1-(1-expected_b)/D, delta=1e-7)

    def test_linear_field_full_root(self):
        geometry = linear_B()
        states, diagnostic = find_temperature_solutions(
            .6, 10000, 5, geometry, n_points=120, b_points=25)
        self.assertGreaterEqual(len(states), 1)
        for state in states:
            self.assertLess(abs(state["R_M"]), 1e-6)
            self.assertLess(abs(state["R_E"]), 1e-6)
            self.assertLess(state["alpha"], 1)
            self.assertGreater(state["alpha"], 0)

    def test_legacy_reference_two_roots(self):
        parameters = reference_parameters()
        D = parameters["L_cm"]*2.8e13*1e-8/parameters["U_star_cm_per_s"]
        states, _ = find_temperature_solutions(
            parameters["beta"], D, 5, quadratic_area_legacy(),
            n_points=150, b_points=35)
        self.assertEqual(len(states), 2)
        np.testing.assert_allclose(
            [s["alpha"]*184.5 for s in states],
            [149.435, 163.950], atol=.002)
        with contextlib.redirect_stdout(io.StringIO()):
            from isothermal_cx_model.solver import solve_mu, compute_omega
        for state in states:
            old = solve_mu(state["theta"], parameters["beta"], D,
                           n_points=5000, rtol=1e-10, atol=1e-12)
            nearest = min(old, key=lambda x: abs(x["alpha"]-state["alpha"]))
            self.assertAlmostEqual(nearest["alpha"], state["alpha"], delta=1e-7)
            _, residual = compute_omega(nearest, parameters["beta"], 5)
            self.assertLess(abs(residual), 1e-6)

    def test_linear_wall_rejected(self):
        state = solve_at_theta(.01, .5, 1, 5, linear_B())
        self.assertEqual(state["status"], "inadmissible_wall")

    def test_critical_wall_explicit(self):
        b = np.sqrt(8*.5*.01)
        state = solve_at_theta(.01, .5, 249.5*b, 5, linear_B())
        self.assertEqual(state["status"], "critical_wall_unresolved")

    def test_zero_cx_explicit(self):
        state = solve_at_theta(.01, .5, 0, 5, constant_B())
        self.assertEqual(state["status"], "zero_cx_degenerate")
        with self.assertRaises(NotImplementedError):
            find_temperature_solutions(.5, 0, 5, constant_B(), b_points=20)

    def test_boundary_limits(self):
        self.assertEqual(solve_at_theta(.3,.5,4,5,constant_B())["status"],
                         "inlet_sonic_limit")
        self.assertEqual(solve_at_theta(.2,.5,4,5,constant_B())["status"],
                         "energy_temperature_limit")

    def test_geometry_limit(self):
        geometry = MagneticGeometry(
            "short_domain", lambda g: np.ones_like(g),
            lambda g: np.zeros_like(g), g_min=.999)
        state = solve_at_theta(.01,.5,4,5,geometry)
        self.assertEqual(state["status"], "geometry_limit")

    def test_D_function_evaluated_once(self):
        calls = []
        state = solve_at_theta(.1,.5,lambda t: calls.append(t) or 4,
                               5, constant_B())
        self.assertEqual(calls, [.1])
        self.assertEqual(state["D"], 4)

    def test_profiles_use_solver_geometry(self):
        parameters = reference_parameters()
        geometry = linear_B(2)
        state = solve_at_theta(.01, parameters["beta"], 5, 5,
                               geometry, n_points=100)
        self.assertEqual(state["status"], "energy_matched")
        profile = compute_profiles(state, parameters)
        np.testing.assert_allclose(
            profile["A_cm2"], parameters["A0_cm2"]*geometry.A_ratio(state["g"]))
        density = profile["n_over_n_star"]*parameters["n_star_cm3"]
        velocity = profile["u_over_u_star"]*parameters["U_star_cm_per_s"]
        np.testing.assert_allclose(
            density*velocity*profile["A_cm2"], parameters["J0_per_s"])
        with self.assertRaises(ValueError):
            compute_profiles(state, parameters, geometry=constant_B())

    def test_cx_temperature_mode_end_to_end(self):
        parameters = reference_parameters()
        for mode in ["constant", "temperature"]:
            roots, diagnostic = find_density_solutions(
                2.8e13, parameters, n_points=100, b_points=35, cx_mode=mode)
            self.assertEqual(len(roots), 2)
            self.assertTrue(all(abs(r["R_M_check"]) < 1e-6 for r in roots))
        self.assertNotAlmostEqual(roots[0]["D"], roots[1]["D"])

    def test_profile_labels_are_diagnostic(self):
        self.assertEqual(classify_profile([1, .8, .5]), "monotone")
        self.assertEqual(classify_profile([1, 1.2, .5]), "with_maximum")
        self.assertEqual(classify_profile([1,1.2,.8,1.1,.5]), "nonmonotone")


class SearchTests(unittest.TestCase):
    def fake_solver(self, function, invalid=None):
        def solve(theta, beta, D, lam, geometry=None, n_points=100, **kwargs):
            b = np.sqrt(8*beta*theta)
            ok = invalid is None or not invalid(b)
            return dict(theta=theta, b=b, D=4, alpha=.8,
                        R_M=function(b) if ok else np.nan, R_E=0.,
                        status="energy_matched" if ok else "sonic_return",
                        branch="monotone", max_step=.02)
        return solve

    def test_two_roots_same_shape_not_overwritten(self):
        with patch("isothermal_cx_model.scan_general.solve_at_theta",
                   self.fake_solver(lambda b: (b-.3)*(b-.65))):
            states, _ = find_temperature_solutions(.5,4,5,constant_B(),
                                                    n_points=100,b_points=30)
        self.assertEqual(len(states), 2)
        np.testing.assert_allclose([s["b"] for s in states], [.3,.65], atol=1e-8)

    def test_tangent_candidate(self):
        with patch("isothermal_cx_model.scan_general.solve_at_theta",
                   self.fake_solver(lambda b: (b-.4)**2)):
            states, _ = find_temperature_solutions(.5,4,5,constant_B(),
                                                    n_points=100,b_points=30)
        self.assertEqual(len(states), 1)
        self.assertIn(states[0]["root_kind"], ("possible_tangency", "exact_sample"))
        self.assertAlmostEqual(states[0]["b"], .4, delta=1e-6)

    def test_no_bracketing_across_gaps(self):
        with patch("isothermal_cx_model.scan_general.solve_at_theta",
                   self.fake_solver(lambda b: b-.4,
                                    invalid=lambda b: .35 < b < .45)):
            states, diagnostic = find_temperature_solutions(
                .5,4,5,constant_B(),n_points=100,b_points=30)
        self.assertEqual(states, [])
        self.assertIn("sonic_return", diagnostic["statuses"])


class CLITests(unittest.TestCase):
    def test_five_profile_panels_and_energy_identity(self):
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from isothermal_cx_model.plot_profiles import plot_profiles
        parameters = reference_parameters()
        state = solve_at_theta(.01, parameters["beta"], 5, 5, linear_B(2),
                               n_points=100)
        profile = compute_profiles(state, parameters)
        velocity = profile["u_over_u_star"]*parameters["U_star_cm_per_s"]
        temperature = state["theta"]*parameters["T_star_erg"]
        recovered = parameters["J0_per_s"]*(
            parameters["m_D_g"]*velocity**2/2 + 5*temperature
        ) + profile["A_cm2"]*profile["q_e_erg_cm2_s"]
        np.testing.assert_allclose(recovered,
            parameters["W0_erg_per_s"]*profile["omega"], rtol=1e-12)
        figure, axes = plot_profiles([profile])
        figure.canvas.draw()
        self.assertEqual(len(axes), 5)
        self.assertTrue(all(len(axis.lines) >= 1 for axis in axes))
        plt.close(figure)

    def test_cli_help(self):
        for module in ["scan", "scan_neutral", "scan_lambda", "plot_profiles"]:
            process = subprocess.run(
                [sys.executable, "-B", "-m", "isothermal_cx_model."+module, "--help"],
                capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn("--geometry", process.stdout)
            self.assertIn("--cx-mode", process.stdout)


if __name__ == "__main__":
    unittest.main()
