"""ADF11 layout, log interpolation, signed energy bookkeeping and CGS scales."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from isothermal_model_adas import recombination_adas as adas


def write_table(path, kind, T, n, values):
    """Small synthetic single-H block; never redistribute licensed raw data."""
    unit = 'CM**3 S-1' if kind == 'ACD' else 'W CM**3'
    text = [f'1 {len(n)} {len(T)} 1 1 /HYDROGEN /SYNTHETIC TEST', '------',
            ' '.join(map(str, n)), ' '.join(map(str, T)),
            '----/ IPRT=1 / IGRD=1 / Z1=1 / DATE=01/11/26']
    text.extend(' '.join(map(str, row)) for row in values)
    text.extend(['------', 'C UNITS OF '+unit])
    path.write_text('\n'.join(text), encoding='utf-8')


class SyntheticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.acd, self.prb = Path(self.temp.name)/'acd.dat', Path(self.temp.name)/'prb.dat'
        self.T, self.n = np.array([0., 2.]), np.array([10., 12., 14.])
        a = -12+.2*self.T[:, None]+.05*(self.n[None, :]-10)
        b = -29+.3*self.T[:, None]+.02*(self.n[None, :]-10)
        write_table(self.acd, 'ACD', self.T, self.n, a)
        write_table(self.prb, 'PRB', self.T, self.n, b)
        self.data = adas.ADASRecombination(self.acd, self.prb)

    def test_non_square_orientation_nodes_and_log_midpoint(self):
        for lt, ln in ((0, 10), (2, 14), (1, 11), (.4, 13.3)):
            a, b, c = self.data.evaluate(10.**lt, 10.**ln)
            self.assertAlmostEqual(a/10.**(-12+.2*lt+.05*(ln-10)), 1., places=13)
            self.assertAlmostEqual(b/(1e7*10.**(-29+.3*lt+.02*(ln-10))), 1., places=13)
            self.assertAlmostEqual((c+adas.IONIZATION_ENERGY_ERG*a)/b, 1., places=13)
        self.assertEqual(self.data.log_T.shape, (2,))
        self.assertEqual(self.data.log_n.shape, (3,))

    def test_scalar_array_broadcast_and_no_input_mutation(self):
        T, n = np.array([[1.], [100.]]), np.array([[1e10, 1e12, 1e14]])
        before_T, before_n = T.copy(), n.copy()
        values = self.data.evaluate(T, n)
        for value in values:
            self.assertEqual(value.shape, (2, 3))
        np.testing.assert_array_equal(T, before_T)
        np.testing.assert_array_equal(n, before_n)
        self.assertTrue(all(isinstance(v, float) for v in self.data.evaluate(10., 1e12)))
        for i in range(2):
            for j in range(3):
                np.testing.assert_allclose([v[i, j] for v in values], self.data.evaluate(T[i, 0], n[0, j]))

    def test_public_helpers_and_energy_identity(self):
        a, c = adas.adas_coefficients(10., 1e12, data=self.data)
        self.assertEqual(a, adas.alpha_adas(10., 1e12, data=self.data))
        self.assertEqual(c, adas.C_eff(10., 1e12, data=self.data))
        gamma = adas.L_gamma_adas(10., 1e12, data=self.data)
        self.assertAlmostEqual((c+adas.IONIZATION_ENERGY_ERG*a)/gamma, 1., places=14)

    def test_no_extrapolation_and_no_silent_clamping(self):
        for T, n in ((.99, 1e12), (101, 1e12), (10, 9.9e9), (10, 1.01e14)):
            with self.subTest(T=T, n=n), self.assertRaises(adas.ADASDomainError):
                self.data.evaluate(T, n)

    def test_invalid_inputs_and_shapes(self):
        for value in (0., -1., float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.data.evaluate(value, 1e12)
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.data.evaluate(10., value)
        with self.assertRaises(ValueError):
            self.data.evaluate(np.ones(2), np.ones(3))

    def test_printed_endpoint_rounding_not_physical_extrapolation(self):
        T, n = np.array([-.69897, 4.]), np.array([7.69897, 15.30103])
        write_table(self.acd, 'ACD', T, n, np.full((2, 2), -12.))
        write_table(self.prb, 'PRB', T, n, np.full((2, 2), -29.))
        data = adas.ADASRecombination(self.acd, self.prb)
        self.assertEqual(data.evaluate(.2, 5e7)[0], 1e-12)
        self.assertEqual(data.evaluate(1e4, 2e15)[0], 1e-12)
        with self.assertRaises(adas.ADASDomainError):
            data.evaluate(.1999, 5e7)

    def test_bad_layout_units_stage_and_grid_rejected(self):
        original = self.acd.read_text()
        for text in (original.replace('1 3 2 1 1', '2 3 2 1 1'),
                     original.replace('HYDROGEN', 'HELIUM'),
                     original.replace('IPRT=1', 'IPRT=2'),
                     original.replace('CM**3 S-1', 'M**3 S-1'),
                     original.replace('10.0 12.0 14.0', '10.0 10.0 14.0'),
                     original.replace('-12.0 -11.9 -11.8', '-12.0 -11.9')):
            self.acd.write_text(text)
            with self.subTest(text=text), self.assertRaises(adas.ADASDataError):
                adas.ADASRecombination(self.acd, self.prb)
        self.acd.write_text(original)

    def test_mismatched_pair_missing_file_and_extra_block(self):
        write_table(self.prb, 'PRB', self.T, self.n+1, np.zeros((2, 3)))
        with self.assertRaises(adas.ADASDataError):
            adas.ADASRecombination(self.acd, self.prb)
        with self.assertRaises(FileNotFoundError):
            adas.ADASRecombination(self.acd.parent/'missing.dat', self.prb)
        self.acd.write_text(self.acd.read_text()+'\n---/ IPRT=1 / IGRD=1 / Z1=1')
        with self.assertRaises(adas.ADASDataError):
            adas.ADASRecombination(self.acd, self.prb)

    def parameters(self):
        return dict(L_cm=184.5, n_star_cm3=1e12, U_star_cm_per_s=2e7,
                    T_star_eV=1000., T_star_erg=1000*adas.EV_TO_ERG)

    def test_dimensionless_factory_density_dependence_and_scale(self):
        p = self.parameters()
        function = adas.make_adas_coefficients(p, scale=1., data=self.data)
        a, c = self.data.evaluate(10., 1e12)[::2]
        factor = p['L_cm']*p['n_star_cm3']/p['U_star_cm_per_s']
        R, Q = function(.01, 1.)
        self.assertAlmostEqual(R/(factor*a), 1., places=14)
        self.assertAlmostEqual(Q/(factor*c/p['T_star_erg']), 1., places=14)
        self.assertNotEqual(function(.01, .1)[0], function(.01, 10.)[0])
        half = adas.make_adas_coefficients(p, scale=.5, data=self.data)
        np.testing.assert_allclose(half(.01, 1.), .5*np.array([R, Q]), rtol=1e-14)
        N = np.array([.1, 1., 10.])
        self.assertEqual(function(.01, N)[0].shape, N.shape)

    def test_source_scales_copied_and_neutrals_not_used(self):
        p = self.parameters()
        function = adas.make_adas_coefficients(p, scale=1., data=self.data)
        before = function(.01, 1.)
        for key in p:
            p[key] *= 2
        p['n0_cm3'] = 1e99
        self.assertEqual(function(.01, 1.), before)

    def test_off_switch_does_not_read_data_or_scales(self):
        with patch.object(adas, '_default_data', side_effect=AssertionError('read data')):
            self.assertEqual(adas.make_adas_coefficients({}, scale=0), (0., 0.))

    def test_bad_factory_inputs(self):
        for scale in (-1, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                adas.make_adas_coefficients(self.parameters(), scale=scale, data=self.data)
        for key in self.parameters():
            p = self.parameters()
            p[key] = -1
            with self.assertRaises(ValueError):
                adas.make_adas_coefficients(p, scale=1., data=self.data)
        p = self.parameters()
        p['T_star_erg'] *= 2
        with self.assertRaises(ValueError):
            adas.make_adas_coefficients(p, scale=1., data=self.data)


@unittest.skipUnless(adas.ACD_PATH.is_file() and adas.PRB_PATH.is_file(),
                     'Download the local ACD12/PRB12 files; not redistributed with tests.')
class RealDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = adas.ADASRecombination()

    def test_known_cold_node_negative_effective_energy_and_units(self):
        a, b, c = self.data.evaluate(.2, 2e15)
        self.assertAlmostEqual(a/(10.**-8.48235), 1., places=13)
        self.assertAlmostEqual(b/(1e7*10.**-26.28251), 1., places=13)
        self.assertLess(c, 0)
        self.assertAlmostEqual(c/(-1.9584478925195872e-20), 1., places=13)
        self.assertEqual(adas.C_eff(.2, 2e15), c)

    def test_known_hot_node_and_complete_grid(self):
        a, b, _ = self.data.evaluate(1e4, 5e7)
        self.assertAlmostEqual(a/10.**-16.81661, 1., places=13)
        self.assertAlmostEqual(b/(1e7*10.**-29.75473), 1., places=13)
        T, n = 10.**self.data.log_T[:, None], 10.**self.data.log_n[None, :]
        a, b, c = self.data.evaluate(T, n)
        self.assertEqual(a.shape, (29, 24))
        self.assertTrue(np.all(a > 0) and np.all(b > 0))
        np.testing.assert_allclose(c+adas.IONIZATION_ENERGY_ERG*a, b, rtol=1e-13, atol=0)

    def test_factory_preserves_negative_q_and_epsilon_relation(self):
        p = SyntheticTests.parameters(self)
        function = adas.make_adas_coefficients(p, scale=1., data=self.data)
        R, Q = function(.2/p['T_star_eV'], 2e15/p['n_star_cm3'])
        a, b, _ = self.data.evaluate(.2, 2e15)
        epsilon = adas.IONIZATION_ENERGY_ERG/p['T_star_erg']
        factor = p['L_cm']*p['n_star_cm3']/p['U_star_cm_per_s']
        self.assertGreater(R, 0)
        self.assertLess(Q, 0)
        self.assertAlmostEqual((Q+epsilon*R)/(factor*b/p['T_star_erg']), 1., places=13)

    def test_cli_and_cold_out_of_domain(self):
        command = [sys.executable, '-B', '-m', 'isothermal_model_adas.recombination_adas']
        good = subprocess.run(command+['--temperatures', '.2', '1', '--density', '2e15'],
                              capture_output=True, text=True)
        self.assertEqual(good.returncode, 0, good.stdout+good.stderr)
        self.assertIn('Coefficients ONLY', good.stdout)
        bad = subprocess.run(command+['--temperatures', '.119194929'],
                             capture_output=True, text=True)
        self.assertEqual(bad.returncode, 2)
        self.assertIn('No extrapolation', bad.stderr)


if __name__ == '__main__':
    unittest.main()
