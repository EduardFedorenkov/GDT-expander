"""Coefficient-only tests: no plasma ODE, stationary solver or scans."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from isothermal_model_adas import cx_adas as cx


def table_text(values=None):
    if values is None:
        values = [[-9., -9., -9.], [-8., -8., -8.]]
    lines = ['1 3 2 1 1 /DEUTERIUM /SYNTHETIC TEST', '-----',
             '8 10 12', '0 2',
             '----/ IPRT=1 / IGRD=1 / MH=2.01 / Z1=1 / DATE=01/11/26']
    lines.extend(' '.join(map(str, row)) for row in values)
    return '\n'.join(lines+['------', 'C RATE COEFFICIENTS IN UNITS OF CM**3 S-1'])


class CoefficientTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'ccd.dat'
        self.path.write_text(table_text(), encoding='utf-8')
        self.data = cx.ADASCX(self.path)
        self.p = dict(L_cm=100., U_star_cm_per_s=1e7, T_star_eV=100.,
                      n0_cm3=1e13, k_cx_cm3_per_s=2e-8)

    def test_nodes_and_log_midpoint(self):
        for T, expected in ((1., 1e-9), (10., 10**-8.5), (100., 1e-8)):
            self.assertAlmostEqual(cx.k_cx(T, data=self.data)/expected, 1., places=13)

    def test_scalar_arrays_and_no_mutation(self):
        temperatures = np.array([[1., 10.], [100., 10.]])
        original = temperatures.copy()
        result = cx.k_cx(temperatures, data=self.data)
        np.testing.assert_allclose(result, 1e-9*np.sqrt(temperatures), rtol=1e-13)
        np.testing.assert_array_equal(temperatures, original)
        self.assertIsInstance(cx.k_cx(10., 0., data=self.data), float)
        self.assertFalse(self.data.log_k.flags.writeable)

    def test_domain_and_invalid_temperature(self):
        for T in (.5, 101.):
            with self.subTest(T=T), self.assertRaises(cx.ADASDomainError):
                cx.k_cx(T, data=self.data)
        for T in (0., -1., np.nan, np.inf):
            with self.subTest(T=T), self.assertRaises(ValueError):
                cx.k_cx(T, data=self.data)

    def test_nonzero_or_array_drift_is_rejected(self):
        for u in (1., -1., 1e-30, np.nan, np.inf, [0.]):
            with self.subTest(u=u), self.assertRaises(ValueError):
                cx.k_cx(10., u, data=self.data)

    def test_dimensions(self):
        theta = .1
        expected = self.p['L_cm']*self.p['n0_cm3']/self.p['U_star_cm_per_s']*10**-8.5
        self.assertAlmostEqual(cx.D_cx(theta, self.p, data=self.data)/expected, 1., places=13)
        self.assertAlmostEqual(cx.D_cx(theta, self.p, n0_cm3=2e13, data=self.data)/expected, 2., places=13)
        self.assertIsInstance(cx.D_cx(theta, self.p, data=self.data), float)

    def test_D_arrays(self):
        theta = np.array([.01, .1, 1.])
        actual = cx.D_cx(theta, self.p, data=self.data)
        expected = self.p['L_cm']*self.p['n0_cm3']/self.p['U_star_cm_per_s']*self.data.evaluate(theta*100)
        np.testing.assert_allclose(actual, expected)

    def test_factory_snapshot_and_default_mode(self):
        coefficient = cx.make_D(self.p, data=self.data)
        before = coefficient(.1)
        self.p.update(L_cm=1., T_star_eV=1., n0_cm3=1.)
        self.assertEqual(coefficient(.1), before)

    def test_constant_mode_does_not_load_ADAS(self):
        with patch.object(cx, '_default_data', side_effect=AssertionError('No ADAS needed')):
            self.assertAlmostEqual(cx.make_D(self.p, 'constant'), 2.)

    def test_off_needs_no_table_or_temperature_scale(self):
        self.p.pop('T_star_eV')
        with patch.object(cx, '_default_data', side_effect=AssertionError('No ADAS needed')):
            self.assertEqual(cx.make_D(self.p, n0_cm3=0.), 0.)
            self.assertEqual(cx.D_cx(0., self.p, n0_cm3=0.), 0.)
            np.testing.assert_array_equal(cx.D_cx([0., 1.], self.p, n0_cm3=0.), [0., 0.])

    def test_zero_constant_coefficient(self):
        self.p['k_cx_cm3_per_s'] = 0.
        self.assertEqual(cx.make_D(self.p, 'constant'), 0.)

    def test_invalid_parameters_and_mode(self):
        for key in ('L_cm', 'U_star_cm_per_s', 'T_star_eV', 'n0_cm3'):
            bad = dict(self.p)
            bad[key] = -1.
            with self.subTest(key=key), self.assertRaises(ValueError):
                cx.make_D(bad, data=self.data)
        for theta in (-1., np.nan, np.inf):
            with self.assertRaises(ValueError):
                cx.D_cx(theta, self.p, data=self.data)
        with self.assertRaises(ValueError):
            cx.make_D(self.p, 'unknown', n0_cm3=0.)
        with self.assertRaisesRegex(ValueError, 'n0_cm3'):
            cx.make_D({k:v for k,v in self.p.items() if k != 'n0_cm3'}, data=self.data)

    def test_no_electron_density_parameter(self):
        first = cx.D_cx(.1, self.p, data=self.data)
        self.p['n_star_cm3'] = 1e99
        self.assertEqual(cx.D_cx(.1, self.p, data=self.data), first)

    def test_rejects_density_dependent_table(self):
        self.path.write_text(table_text([[-9., -9., -9.], [-8., -7., -8.]]), encoding='utf-8')
        with self.assertRaisesRegex(cx.ADASDataError, 'Density-dependent'):
            cx.ADASCX(self.path)

    def test_rejects_bad_units_element_grids_counts_and_multiple_blocks(self):
        original = table_text()
        for bad in (original.replace('CM**3 S-1', 'W CM**3'),
                    original.replace('DEUTERIUM', 'HYDROGEN'),
                    original.replace('8 10 12', '8 8 12'),
                    original.replace('0 2\n', '2 0\n'),
                    original.replace('-9.0 -9.0 -9.0', '-9.0 -9.0', 1),
                    original+'\nIPRT=1 / IGRD=1 / Z1=1'):
            self.path.write_text(bad, encoding='utf-8')
            with self.assertRaises(cx.ADASDataError):
                cx.ADASCX(self.path)

    def test_missing_file(self):
        with self.assertRaises(OSError):
            cx.ADASCX(self.path.with_name('missing.dat'))


@unittest.skipUnless(cx.CCD_PATH.exists(), 'Real ADAS table is not distributed with code/tests.')
class RealDataTests(unittest.TestCase):
    def test_real_nodes_and_rounded_endpoint(self):
        data = cx.ADASCX()
        self.assertEqual(data.log_T.shape, (29,))
        self.assertEqual(data.log_n.shape, (24,))
        np.testing.assert_allclose(data.evaluate(10.**data.log_T), 10.**data.log_k, rtol=1e-13)
        for T, log_rate in ((.2, -8.59260), (1., -8.24311), (10., -7.81632)):
            self.assertAlmostEqual(np.log10(data.evaluate(T)), log_rate, places=5)
        with self.assertRaises(cx.ADASDomainError):
            data.evaluate(.1999)

    def test_CLI_only_prints_coefficients(self):
        command = [sys.executable, '-B', '-m', 'isothermal_model_adas.cx_adas']
        result = subprocess.run(command+['--temperatures', '.2', '1', '10'],
                                text=True, capture_output=True, encoding='utf-8')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Coefficients only', result.stdout)
        bad = subprocess.run(command+['--temperatures', '.1'], capture_output=True)
        self.assertNotEqual(bad.returncode, 0)


if __name__ == '__main__':
    unittest.main()
