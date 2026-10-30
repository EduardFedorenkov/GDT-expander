"""Regression and interface tests. Run: python -B -m unittest discover -s tests -v."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from isothermal_cx_model.geometry import (
    MagneticGeometry, linear_B, constant_B, quadratic_area_legacy,
    tabulated_B, load_geometry,
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


if __name__ == "__main__":
    unittest.main()
