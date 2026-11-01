"""SCD interpolation and normalization only: no IVP/BVP/scan imports."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from isothermal_model_adas import ionization_adas as ion


def table_text(T=(0., 2.), n=(10., 12., 14.)):
    values = -12+.25*np.array(T)[:, None]+.1*(np.array(n)[None, :]-10)
    lines = [f'1 {len(n)} {len(T)} 1 1 /HYDROGEN /SYNTHETIC TEST', '------',
        ' '.join(map(str, n)), ' '.join(map(str, T)),
        '----/ IPRT=1 / IGRD=1 / Z1=1 / DATE=01/11/26']
    lines.extend(' '.join(map(str, row)) for row in values)
    return '\n'.join(lines+['------', 'C RATE COEFFICIENTS IN UNITS OF CM**3 S-1'])


class CoefficientTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'scd.dat'
        self.path.write_text(table_text(), encoding='utf-8')
        self.data = ion.ADASIonization(self.path)
        self.p = dict(L_cm=100., n_star_cm3=1e12, U_star_cm_per_s=1e7,
            T_star_eV=100., T_star_erg=100*ion.EV_TO_ERG, n0_cm3=1e13)

    def test_non_square_orientation_nodes_and_log_bilinear_midpoints(self):
        self.assertEqual(self.data.log_T.shape, (2,))
        self.assertEqual(self.data.log_n.shape, (3,))
        for lt, ln in ((0, 10), (2, 14), (1, 11), (.3, 12.7)):
            expected = 10.**(-12+.25*lt+.1*(ln-10))
            actual = ion.k_ion(10.**lt, 10.**ln, data=self.data)
            self.assertAlmostEqual(actual/expected, 1., places=13)

    def test_arrays_broadcast_scalar_and_no_mutation(self):
        T = np.array([[1.], [10.], [100.]])
        n = np.array([1e10, 1e12, 1e14])
        original = T.copy()
        actual = ion.k_ion(T, n, data=self.data)
        expected = 1e-12*T**.25*(n/1e10)**.1
        np.testing.assert_allclose(actual, expected, rtol=1e-13)
        self.assertEqual(actual.shape, (3, 3))
        np.testing.assert_array_equal(T, original)
        self.assertIsInstance(ion.k_ion(10., 1e12, data=self.data), float)
        self.assertFalse(self.data.log_T.flags.writeable)
        self.assertFalse(self.data.log_n.flags.writeable)

    def test_energy_units_helpers_and_natural_log(self):
        k, L = ion.ionization_coefficients(10., 1e12, data=self.data)
        self.assertAlmostEqual(L/k, 13.6*ion.EV_TO_ERG, places=23)
        self.assertEqual(ion.L_ion(10., 1e12, data=self.data), L)
        self.assertAlmostEqual(ion.log_k_ion(10., 1e12, data=self.data), np.log(k), places=13)

    def test_one_query_for_dimensional_pair(self):
        with patch.object(self.data, 'evaluate', wraps=self.data.evaluate) as reader:
            ion.ionization_coefficients(10., 1e12, data=self.data)
        reader.assert_called_once()

    def test_density_is_required_and_really_changes_the_rate(self):
        with self.assertRaises(TypeError):
            ion.k_ion(10., data=self.data)
        low = ion.k_ion(10., 1e10, data=self.data)
        high = ion.k_ion(10., 1e14, data=self.data)
        self.assertAlmostEqual(high/low, 10**.4, places=13)

    def test_no_extrapolation_or_clamping(self):
        for T, n in ((.99, 1e12), (101., 1e12), (10., 1e9), (10., 1e15)):
            with self.subTest(T=T, n=n), self.assertRaises(ion.ADASDomainError):
                ion.k_ion(T, n, data=self.data)

    def test_invalid_inputs_and_incompatible_shapes(self):
        for value in (0., -1., np.nan, np.inf):
            with self.assertRaises(ValueError):
                ion.k_ion(value, 1e12, data=self.data)
            with self.assertRaises(ValueError):
                ion.k_ion(10., value, data=self.data)
        with self.assertRaises(ValueError):
            ion.k_ion([1., 10.], [1e10, 1e12, 1e14], data=self.data)

    def test_factory_dimensions_not_extra_density_factor(self):
        callback = ion.make_ionization_coefficients(self.p, scale=1., data=self.data)
        D, E = callback(.1, 1.)
        expected = self.p['L_cm']*self.p['n0_cm3']/self.p['U_star_cm_per_s']*self.data.evaluate(10., 1e12)
        self.assertAlmostEqual(D/expected, 1., places=13)
        self.assertAlmostEqual(E/D, 13.6/100, places=13)
        # n_star maps N to n_e; it is NOT another multiplicative prefactor.
        other = dict(self.p, n_star_cm3=2e12)
        alternative = ion.make_ionization_coefficients(other, scale=1., data=self.data)
        np.testing.assert_allclose(alternative(.1, .5), (D, E), rtol=1e-13)

    def test_factory_arrays(self):
        callback = ion.make_ionization_coefficients(self.p, scale=1., data=self.data)
        theta = np.array([[.01], [.1], [1.]])
        N = np.array([.01, 1., 100.])
        D, E = callback(theta, N)
        self.assertEqual(D.shape, (3, 3))
        np.testing.assert_allclose(E, 13.6/100*D, rtol=1e-13)

    def test_scale_and_neutral_override(self):
        full = ion.make_ionization_coefficients(self.p, scale=1., data=self.data)(.1, 1.)
        half = ion.make_ionization_coefficients(self.p, scale=.5, data=self.data)(.1, 1.)
        double = ion.make_ionization_coefficients(self.p, scale=1., n0_cm3=2e13, data=self.data)(.1, 1.)
        np.testing.assert_allclose(half, .5*np.array(full), rtol=1e-13)
        np.testing.assert_allclose(double, 2*np.array(full), rtol=1e-13)

    def test_factory_snapshots_parameters(self):
        callback = ion.make_ionization_coefficients(self.p, scale=1., data=self.data)
        before = callback(.1, 1.)
        self.p.update(L_cm=1., n_star_cm3=1., n0_cm3=1., T_star_eV=1.)
        self.assertEqual(callback(.1, 1.), before)

    def test_off_switches_need_no_scales_or_table(self):
        with patch.object(ion, '_default_data', side_effect=AssertionError('Do not load data')):
            self.assertEqual(ion.make_ionization_coefficients(None), (0., 0.))
            self.assertEqual(ion.make_ionization_coefficients(None, scale=1., n0_cm3=0.), (0., 0.))
            self.assertEqual(ion.make_ionization_coefficients({'n0_cm3': 0.}, scale=1.), (0., 0.))

    def test_factory_invalid_scales_parameters_and_query(self):
        for scale in (-1., 1.01, np.nan, np.inf):
            with self.assertRaises(ValueError):
                ion.make_ionization_coefficients(self.p, scale=scale, data=self.data)
        for key in ('L_cm', 'n_star_cm3', 'U_star_cm_per_s', 'T_star_eV', 'T_star_erg', 'n0_cm3'):
            bad = dict(self.p, **{key: -1.})
            with self.subTest(key=key), self.assertRaises(ValueError):
                ion.make_ionization_coefficients(bad, scale=1., data=self.data)
        with self.assertRaises(ValueError):
            ion.make_ionization_coefficients(dict(self.p, T_star_erg=1.), scale=1., data=self.data)
        with self.assertRaises(ValueError):
            ion.make_ionization_coefficients({}, scale=1., data=self.data)
        callback = ion.make_ionization_coefficients(self.p, scale=1., data=self.data)
        with self.assertRaises(ion.ADASDomainError):
            callback(.001, 1.)

    def test_bad_format_element_units_stages_and_multiple_blocks(self):
        original = table_text()
        cases = (original.replace('HYDROGEN', 'DEUTERIUM'),
                 original.replace('CM**3 S-1', 'W CM**3'),
                 original.replace('1 3 2 1 1', '1 3 3 1 1'),
                 original.replace('Z1=1', 'Z1=2'),
                 original.replace('10.0 12.0 14.0', '10.0 10.0 14.0'),
                 original+'\nIPRT=1 / IGRD=1 / Z1=1', '')
        for text in cases:
            self.path.write_text(text, encoding='utf-8')
            with self.assertRaises(ion.ADASDataError):
                ion.ADASIonization(self.path)

    def test_fortran_exponents_and_missing_file(self):
        text = table_text().replace('10.0 12.0 14.0', '1.0D+01 1.2D+01 1.4D+01')
        self.path.write_text(text, encoding='utf-8')
        np.testing.assert_allclose(ion.ADASIonization(self.path).evaluate(10., 1e12), self.data.evaluate(10., 1e12))
        with self.assertRaises(OSError):
            ion.ADASIonization(self.path.with_name('missing.dat'))


@unittest.skipUnless(ion.SCD_PATH.exists(), 'Real SCD data is not distributed with code/tests.')
class RealDataTests(unittest.TestCase):
    def test_known_nodes_and_rounded_endpoints(self):
        data = ion.ADASIonization()
        self.assertEqual(data.log_T.shape, (29,))
        self.assertEqual(data.log_n.shape, (24,))
        for T, n, expected in ((.2, 5e7, 2.34261004e-38),
                               (10., 5e7, 5.43312879e-9),
                               (10., 1e13, 7.71969248e-9),
                               (10., 2e15, 1.67243790e-8)):
            self.assertAlmostEqual(data.evaluate(T, n)/expected, 1., places=7)
        self.assertGreater(data.evaluate(.2, 5e7), 0.)
        with self.assertRaises(ion.ADASDomainError):
            data.evaluate(.1999, 1e13)

    def test_full_grid_positive_and_finite(self):
        data = ion.ADASIonization()
        rates = data.evaluate(10.**data.log_T[:, None], 10.**data.log_n[None, :])
        self.assertEqual(rates.shape, (29, 24))
        self.assertTrue(np.isfinite(rates).all())
        self.assertTrue(np.all(rates > 0))

    def test_CLI_only_outputs_coefficients(self):
        command = [sys.executable, '-B', '-m', 'isothermal_model_adas.ionization_adas']
        result = subprocess.run(command+['--temperatures', '.2', '1', '10', '--density', '1e13'],
                                text=True, capture_output=True, encoding='utf-8')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Coefficients ONLY', result.stdout)
        self.assertIn('excludes PLT', result.stdout)
        bad = subprocess.run(command+['--temperatures', '.1'], capture_output=True)
        self.assertNotEqual(bad.returncode, 0)


if __name__ == '__main__':
    unittest.main()
