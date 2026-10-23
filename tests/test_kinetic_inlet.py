"""Kinetic inlet matching, independent analytic oracle, and old-BC regression."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import brentq

from simple_cx_model.config import load_cx_config
from isothermal_cx_model.geometry import (
    constant_B, linear_B, quadratic_area_legacy, MagneticGeometry,
)
from isothermal_cx_model.inlet import (
    make_inlet_momentum, KineticInletDomainError,
)
from isothermal_cx_model.kinetic_momentum import kinetic_momentum
from isothermal_cx_model.solver_general import solve_at_theta
from isothermal_cx_model.scan_general import find_temperature_solutions
from isothermal_cx_model.scan_neutral import find_density_solutions
from isothermal_cx_model.scan_lambda import scan_lambda_neutral, save_csv, ROOT_FIELDS
from isothermal_cx_model.profiles import compute_profiles


CONFIG = Path(__file__).resolve().parents[1] / "simple_cx_model/cx_config.json"


def parameters():
    return load_cx_config(CONFIG)


class KineticConfigTests(unittest.TestCase):
    def load_chi(self, value=None, omit=False):
        data = json.loads(CONFIG.read_text(encoding="utf-8-sig"))
        if omit:
            data.pop("chi", None)
        else:
            data["chi"] = value
        with patch("pathlib.Path.read_text", return_value=json.dumps(data)):
            return load_cx_config("in_memory.json")

    def test_sample_and_missing_chi_default_to_five(self):
        self.assertEqual(parameters()["chi"], 5)
        self.assertEqual(self.load_chi(omit=True)["chi"], 5)

    def test_zero_allowed_and_bad_chi_rejected(self):
        self.assertEqual(self.load_chi(0)["chi"], 0)
        for value in (-1, np.nan, np.inf, "bad", None):
            with self.subTest(chi=value), self.assertRaisesRegex(ValueError, "chi"):
                self.load_chi(value)

    def test_chi_does_not_renormalize_throat_fluxes_or_scales(self):
        first, second = self.load_chi(0), self.load_chi(5)
        for key in ("J0_per_s", "M0_dyn", "W0_erg_per_s", "beta", "D",
                    "U_star_cm_per_s", "T_star_eV", "chi_E"):
            self.assertEqual(first[key], second[key], key)


class KineticBoundaryTests(unittest.TestCase):
    def test_target_matches_standalone_model(self):
        p, geo, alpha = parameters(), linear_B(), .7
        boundary = make_inlet_momentum(p, geo)
        result = boundary.evaluate(alpha)
        expected = kinetic_momentum(
            alpha*p["L_cm"], geo, 5, L_cm=p["L_cm"],
            A_m_cm2=p["A_m_cm2"], n_c_cm3=p["n_c_cm3"],
            T_i0_eV=p["T_c_eV"], T_e_eV=p["T_c_eV"],
        )
        self.assertEqual(result["M_kin_dyn"], expected)
        self.assertEqual(result["mu_target"], expected/p["M0_dyn"])
        self.assertEqual(result["T_kin_eV"], p["T_c_eV"])

    def test_old_mode_is_explicit(self):
        self.assertIsNone(make_inlet_momentum({}, constant_B(), "constant"))
        with self.assertRaises(ValueError):
            make_inlet_momentum({}, constant_B(), "bad")

    def test_legacy_geometry_endpoint_and_missing_prefix(self):
        p, geo = parameters(), quadratic_area_legacy()
        boundary = make_inlet_momentum(p, geo)
        with self.assertRaises(KineticInletDomainError):
            boundary.evaluate(.02)
        self.assertEqual(boundary.evaluate(.5)["kinetic_path_check"],
                         "sampled_partial_prefix")

    def test_increasing_field_prefix_rejected(self):
        geo = MagneticGeometry(
            "nonmonotone", lambda g: np.exp(-g + .2*np.sin(10*g)),
            lambda g: -1 + 2*np.cos(10*g),
        )
        with self.assertRaisesRegex(KineticInletDomainError, "increasing"):
            make_inlet_momentum(parameters(), geo).evaluate(.9)

    def test_wrong_throat_normalization_rejected(self):
        geo = MagneticGeometry("wrong", lambda g: .5*np.ones_like(g),
                               lambda g: np.zeros_like(g))
        with self.assertRaisesRegex(ValueError, "B_m"):
            make_inlet_momentum(parameters(), geo)


class KineticSolverTests(unittest.TestCase):
    def test_constant_area_independent_analytic_solution_above_b_one(self):
        p, geo, D = parameters(), constant_B(), 4.0
        boundary = make_inlet_momentum(p, geo)
        target = boundary.evaluate(.5)["mu_target"]
        beta, lam = p["beta"], p["lambda_sh"]

        def analytic_energy(b):
            radical = np.sqrt(target*target - b*b)
            integral = 2.5*(target*target-b*b) + .5*(
                target*radical - b*b*np.log((target+radical)/b))
            return lam*b*b/(8*beta) + integral/(8*beta) - 1

        b_max = np.sqrt(8*beta/lam)*(1-1e-8)
        expected = brentq(analytic_energy, 1.0, b_max)
        states, diagnostic = find_temperature_solutions(
            beta, D, lam, geo, n_points=100, b_points=30,
            inlet_momentum=boundary,
        )
        self.assertEqual(len(states), 1)
        state = states[0]
        self.assertGreater(diagnostic["b_max"], 1)
        self.assertGreater(state["b"], 1)
        self.assertAlmostEqual(state["b"], expected, delta=1e-7)
        self.assertAlmostEqual(state["alpha"], 1-(target-expected)/D, delta=1e-8)
        np.testing.assert_allclose(state["mu"], state["b"]+D*(1-state["g"]), atol=1e-9)
        self.assertLess(abs(state["R_M_check"]), state["R_M_tolerance"])
        self.assertAlmostEqual(state["mu_a"], target, delta=1e-7)

    def test_small_off_root_target_does_not_destroy_valid_bracket(self):
        p = parameters()
        state = solve_at_theta(1.1**2/(8*p["beta"]), p["beta"], 4, 5,
                               constant_B(), inlet_momentum=lambda alpha: .5)
        self.assertEqual(state["status"], "energy_matched")
        self.assertLess(state["inlet_target_sonic_margin"], 0)
        self.assertTrue(np.isfinite(state["R_M"]))

    def test_kinetic_domain_failure_is_not_a_fake_residual(self):
        p, geo = parameters(), quadratic_area_legacy(.5)
        state = solve_at_theta(.1, .5, 10, 5, geo,
                               inlet_momentum=make_inlet_momentum(p, geo))
        self.assertEqual(state["status"], "kinetic_inlet_outside_domain")
        self.assertTrue(np.isnan(state["R_M"]))

    def test_refined_solve_recomputes_target_at_its_own_alpha(self):
        p, geo, seen = parameters(), constant_B(), []

        def target(alpha):
            seen.append(alpha)
            return 1.4 + .05*alpha

        states, _ = find_temperature_solutions(p["beta"], 4, 5, geo,
            n_points=100, b_points=25, inlet_momentum=target)
        self.assertEqual(len(states), 1)
        state = states[0]
        self.assertIn(state["alpha"], seen)
        self.assertEqual(state["mu_target"], target(state["alpha"]))
        self.assertLess(abs(state["mu_a"]-target(state["alpha"])),
                        state["R_M_tolerance"])

    def test_full_density_scan_profiles_and_exports(self):
        p, geo = parameters(), constant_B()
        p["k_cx_cm3_per_s"] = 4*p["U_star_cm_per_s"]/(p["L_cm"]*p["n0_cm3"])
        rows, _ = find_density_solutions(p["n0_cm3"], p, n_points=100,
                                          b_points=25, geometry=geo)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        state = solve_at_theta(row["theta"], p["beta"], row["D"], 5, geo,
                               inlet_momentum=make_inlet_momentum(p, geo),
                               rtol=1e-10, atol=1e-12)
        profile = compute_profiles(state, p)
        self.assertAlmostEqual(profile["R_M"], state["R_M"])
        self.assertGreater(abs(state["mu_a"]-1), .1)
        np.testing.assert_allclose(profile["n_cm3"]*profile["u_cm_s"]*
                                   profile["A_cm2"], p["J0_per_s"])
        with self.assertRaisesRegex(ValueError, "config"):
            compute_profiles(state, dict(p, chi=4))
        with contextlib.redirect_stdout(io.StringIO()):
            roots, grid = scan_lambda_neutral([5], [p["n0_cm3"]], p,
                n_points=100, b_points=25, geometry=geo)
        self.assertEqual(grid[0]["n_roots"], 1)
        self.assertEqual(roots[0]["chi"], 5)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"roots.csv"
            save_csv(path, roots, ROOT_FIELDS)
            self.assertIn("mu_target", path.read_text(encoding="utf-8-sig"))


if __name__ == "__main__":
    unittest.main()
