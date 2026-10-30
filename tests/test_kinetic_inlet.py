"""Kinetic inlet matching, independent analytic oracle, and old-BC regression."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from simple_cx_model.config import load_cx_config
from isothermal_cx_model.geometry import (
    constant_B, linear_B, quadratic_area_legacy, MagneticGeometry,
)
from isothermal_cx_model.inlet import (
    make_inlet_momentum, KineticInletDomainError,
)
from isothermal_cx_model.kinetic_momentum import kinetic_momentum


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


if __name__ == "__main__":
    unittest.main()
