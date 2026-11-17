"""Checks for the AMJUEL H.4 coefficient layer (no plasma solve)."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from isothermal_model_adas import molecular_amjuel as amjuel


class TestAMJUELMolecular(unittest.TestCase):
    def setUp(self):
        self.data = amjuel.AMJUELMolecular()
        self.parameters = {'L_cm': 100., 'U_star_cm_per_s': 1e6,
                           'T_star_eV': 10., 'n_star_cm3': 1e13,
                           'n_d2_v0_cm3': 2e13}

    def test_reference_values_from_pdf_coefficients(self):
        rates = self.data.evaluate(5., 1e13)
        np.testing.assert_allclose(
            rates, (4.481723721e-11, 4.506080519e-10, 1.350423620e-12),
            rtol=2e-10, atol=0.)
        self.assertTrue(all(isinstance(value, float) for value in rates))
        self.assertEqual(amjuel.k_mar(5., 1e13, data=self.data), rates.mar)
        self.assertEqual(amjuel.k_mad(5., 1e13, data=self.data), rates.mad)
        self.assertEqual(amjuel.k_mai(5., 1e13, data=self.data), rates.mai)

    def test_broadcasting(self):
        temperatures = np.array([[1.], [5.]])
        densities = [1e8, 1e13, 1e16]
        rates = self.data.evaluate(temperatures, densities)
        for branch in rates:
            self.assertEqual(branch.shape, (2, 3))
        for i, temperature in enumerate(temperatures[:, 0]):
            for j, density in enumerate(densities):
                np.testing.assert_allclose(
                    [branch[i, j] for branch in rates], self.data.evaluate(temperature, density),
                    rtol=1e-13, atol=0.)

    def test_density_range_and_invalid_inputs(self):
        for density in (1e7, 1e17):
            with self.assertRaises(amjuel.AMJUELDomainError):
                self.data.evaluate(5., density)
        for temperature, density in ((0., 1e13), (float('nan'), 1e13), (5., -1.)):
            with self.assertRaises(ValueError):
                self.data.evaluate(temperature, density)

    def test_dimensionless_coefficients(self):
        coeffs = amjuel.make_molecular_coefficients(self.parameters, scale=.5, data=self.data)
        expected = .5 * 100. * 2e13 / 1e6 * np.array(self.data.evaluate(5., 1e13))
        np.testing.assert_allclose(coeffs(.5, 1.), expected, rtol=1e-13, atol=0.)
        half_density = amjuel.make_molecular_coefficients(
            self.parameters, scale=.5, n_d2_v0_cm3=1e13, data=self.data)
        np.testing.assert_allclose(half_density(.5, 1.), expected / 2, rtol=1e-13, atol=0.)

    def test_disabled_does_not_need_data_or_scales(self):
        self.assertEqual(amjuel.make_molecular_coefficients({}), (0., 0., 0.))
        self.assertEqual(amjuel.make_molecular_coefficients(
            {}, scale=1., n_d2_v0_cm3=0.), (0., 0., 0.))
        with self.assertRaisesRegex(ValueError, 'n_d2_v0_cm3'):
            amjuel.make_molecular_coefficients({}, scale=1.)

    def test_invalid_matrix_and_explicit_temperature_range(self):
        doc = json.loads(amjuel.AMJUEL_PATH.read_text(encoding='utf-8-sig'))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'fit.json'
            doc['fit']['temperature_range_eV'] = [1., 20.]
            path.write_text(json.dumps(doc), encoding='utf-8')
            bounded = amjuel.AMJUELMolecular(path)
            with self.assertRaises(amjuel.AMJUELDomainError):
                bounded.evaluate(.5, 1e13)
            doc['reactions']['MAD']['b'] = [[1.]]
            path.write_text(json.dumps(doc), encoding='utf-8')
            with self.assertRaisesRegex(amjuel.AMJUELDataError, 'MAD.*9 by 9'):
                amjuel.AMJUELMolecular(path)


if __name__ == '__main__':
    unittest.main()
