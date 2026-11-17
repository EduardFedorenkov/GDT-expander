"""Checks for the AMJUEL H.4 coefficient layer (no plasma solve)."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from isothermal_model_adas import molecular_amjuel as amjuel
from isothermal_model_adas.atomic_coefficients import build_atomic_coefficients
from isothermal_model_adas.config import load_config, neutral_densities


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

    def test_config_split_and_density_override(self):
        example = Path(amjuel.__file__).with_name('config.json')
        config = json.loads(example.read_text(encoding='utf-8-sig'))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.json'
            config.pop('molecular_fraction', None)
            path.write_text(json.dumps(config), encoding='utf-8')
            p = load_config(path)
            self.assertEqual(neutral_densities(p), (p['n0_cm3'], 0.))
            config.update(n0_cm3=1e13, molecular_fraction=.3)
            path.write_text(json.dumps(config), encoding='utf-8')
            p = load_config(path)
            np.testing.assert_allclose(neutral_densities(p), (7e12, 3e12), rtol=1e-14)
            p['n0_cm3'] = 2e13
            np.testing.assert_allclose(neutral_densities(p), (1.4e13, 6e12), rtol=1e-14)

    def test_fraction_validation_and_boundaries(self):
        for fraction in (-.1, 1.1, float('nan'), float('inf'), None):
            with self.subTest(fraction=fraction), self.assertRaisesRegex(ValueError, 'molecular_fraction'):
                neutral_densities(dict(n0_cm3=1e13, molecular_fraction=fraction))
        self.assertEqual(neutral_densities(dict(n0_cm3=1e13, molecular_fraction=1.)), (0., 1e13))
        self.assertEqual(neutral_densities(dict(n0_cm3=0., molecular_fraction=.3)), (0., 0.))

    def test_molecular_provider_from_fraction_and_fragment_energy(self):
        parameters = dict(self.parameters, n0_cm3=1e13, molecular_fraction=.3)
        del parameters['n_d2_v0_cm3']
        provider = amjuel.make_molecular_coefficients(parameters, scale=1., data=self.data)
        explicit = amjuel.make_molecular_coefficients(
            parameters, scale=1., n_d2_v0_cm3=3e12, data=self.data)
        np.testing.assert_allclose(provider(.5, 1.), explicit(.5, 1.), rtol=1e-14)
        self.assertAlmostEqual(amjuel.MAD_FRAGMENT_ENERGY_EV, 2.9)

    def test_stationary_pipeline_splits_atomic_and_molecular_densities(self):
        config = Path(amjuel.__file__).with_name('config.json')
        p = dict(load_config(config), n0_cm3=1e13, molecular_fraction=.3)
        theta, N = 5./p['T_star_eV'], 1.
        expected = amjuel.make_molecular_coefficients(p, scale=1., n_d2_v0_cm3=3e12, data=self.data)
        for mode in ('hybrid', 'full-adas'):
            with self.subTest(mode=mode):
                _, base, _, _ = build_atomic_coefficients(dict(p, molecular_fraction=0.), mode, 0., 1., 1.)
                _, mixed, tables, paths = build_atomic_coefficients(p, mode, 0., 1., 1.)
                np.testing.assert_allclose(mixed['D'](theta), .7*base['D'](theta), rtol=1e-14)
                for key in ('ionization', 'excitation'):
                    np.testing.assert_allclose(mixed[key](theta, N),
                        .7*np.asarray(base[key](theta, N)), rtol=1e-14, atol=0.)
                np.testing.assert_allclose(mixed['molecular'](theta, N), expected(theta, N))
                self.assertIsInstance(tables[-1], amjuel.AMJUELMolecular)
                self.assertIn(amjuel.AMJUEL_PATH, paths)
                _, only_mol, _, mol_paths = build_atomic_coefficients(
                    dict(p, molecular_fraction=1.), mode, 0., 1., 1.)
                self.assertEqual(mol_paths, [amjuel.AMJUEL_PATH])
                self.assertEqual(only_mol['D'], 0.)
                self.assertEqual(only_mol['ionization'], (0., 0.))
                self.assertEqual(only_mol['excitation'], 0.)

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
