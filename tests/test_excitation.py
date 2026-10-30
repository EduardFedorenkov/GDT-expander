"""Independent excitation, normalization, energy-balance and plumbing checks."""

import argparse
import contextlib
import copy
import io
from pathlib import Path
import unittest

import numpy as np

from simple_cx_model.config import load_cx_config
from isothermal_cx_model.excitation import (
    L_exc, E_exc, make_E_exc, add_excitation_arguments,
)


CONFIG = Path(__file__).resolve().parents[1] / "simple_cx_model/cx_config.json"


class CoefficientTests(unittest.TestCase):
    def test_scalar_array_and_zero(self):
        self.assertIsInstance(L_exc(10), float)
        values = L_exc(np.array([0., 7.438392538197074, 47.703016514101265]))
        np.testing.assert_allclose(values, [0, 9.868919925645329e-20,
                                            1.806361405265463e-19], rtol=1e-12)

    def test_invalid_temperature(self):
        for value in [-1, np.nan, np.inf, [1, -2]]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                L_exc(value)

    def test_cgs_normalization_and_density_override(self):
        p = load_cx_config(CONFIG)
        original = copy.deepcopy(p)
        theta = .02
        expected = (p['L_cm']*p['n0_cm3']*L_exc(theta*p['T_star_eV'])
                    /(p['U_star_cm_per_s']*p['T_star_erg']))
        self.assertEqual(E_exc(theta, p), expected)
        self.assertAlmostEqual(E_exc(theta, p, 2*p['n0_cm3']), 2*expected)
        self.assertEqual(E_exc(0, p), 0)
        self.assertEqual(E_exc(theta, p, 0), 0)
        self.assertEqual(p, original)

    def test_off_and_fractional_scale(self):
        self.assertEqual(make_E_exc({}, scale=0), 0)
        p = load_cx_config(CONFIG)
        self.assertEqual(make_E_exc(p, scale=.25)(.02), .25*E_exc(.02, p))

    def test_bad_parameters_and_scale(self):
        p = load_cx_config(CONFIG)
        for scale in [-1, 1.01, np.nan, np.inf]:
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                make_E_exc(p, scale=scale)
        for key in ['L_cm', 'U_star_cm_per_s', 'T_star_eV', 'T_star_erg']:
            with self.subTest(key=key), self.assertRaises(ValueError):
                E_exc(.1, dict(p, **{key: 0}))
        for theta in [-1, np.inf, np.nan]:
            with self.subTest(theta=theta), self.assertRaises(ValueError):
                E_exc(theta, p)

    def test_parser(self):
        parser = argparse.ArgumentParser()
        add_excitation_arguments(parser)
        self.assertEqual(parser.parse_args([]).excitation_scale, 0)
        self.assertEqual(parser.parse_args(['--excitation-scale', '.5']).excitation_scale, .5)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(['--excitation-scale', 'nan'])


if __name__ == '__main__':
    unittest.main()
