"""Cross-section transcription, Maxwell normalization and cold-tail tests."""

import math
import subprocess
import sys
import unittest
from unittest.mock import patch
import numpy as np
from scipy.integrate import quad

from isothermal_cx_model import ionization as ion


class IonizationTests(unittest.TestCase):
    def setUp(self):
        ion._log_k_single.cache_clear()

    def test_cross_section_reference_values(self):
        # Eq. (14)/Table 3 independently evaluated with Decimal precision=50.
        np.testing.assert_allclose(ion.sigma_ion_D([20., 27.2, 100., 1000.]),
            [2.9934835297548225e-17, 4.711502617141812e-17,
             5.570618966254601e-17, 1.1443481758395653e-17], rtol=2e-14)

    def test_exact_threshold_and_stable_near_threshold(self):
        I = ion.IONIZATION_ENERGY_EV
        np.testing.assert_array_equal(ion.sigma_ion_D([0., 1., I/2, I]), 0.)
        self.assertGreater(ion.sigma_ion_D(np.nextafter(I, np.inf)), 0.)
        self.assertTrue(np.all(ion.sigma_ion_D(np.geomspace(I*1.00001, 1e5, 200)) > 0))

    def test_rate_reference_values(self):
        temperatures = [.1, .119194929, 1., 3.272164647, 14.93634817, 77.8218403]
        expected = [1.493883404e-68, 5.293160869e-59, 7.006602545e-15,
                    1.718068943e-10, 9.021229221e-9, 2.685361971e-8]
        np.testing.assert_allclose(ion.k_ion(temperatures), expected, rtol=2e-9, atol=0)

    def test_independent_velocity_space_average(self):
        # Different integration variable/kernel than production code. No finite
        # velocity cutoff; this checks m_e, eV->erg, 4/sqrt(pi), and threshold.
        for T in [.03, .1192, 1., 10., 100.]:
            with self.subTest(T=T):
                vth = math.sqrt(2*1.602176634e-12*T/9.1093837139e-28)
                lower = math.sqrt(13.6/T)
                integral, _ = quad(lambda x: ion.sigma_ion_D(T*x*x)*x**3*math.exp(-x*x),
                    lower, np.inf, epsabs=0., epsrel=2e-10, limit=200)
                expected = 4*vth/math.sqrt(math.pi)*integral
                self.assertAlmostEqual(ion.k_ion(T)/expected, 1., delta=2e-8)

    def test_no_one_ev_cutoff_or_artificial_cold_floor(self):
        values = ion.k_ion([.03, .1, .3, .5, .999, 1., 1.001])
        self.assertTrue(np.all(values > 0))
        self.assertTrue(np.all(np.diff(values) > 0))
        self.assertLess(values[0], 1e-200)
        self.assertLess(abs(values[-1]/values[-2]-1), .02)

    def test_logarithmic_cold_tail_and_natural_underflow(self):
        with np.errstate(all='raise'):
            self.assertEqual(ion.k_ion(.001), 0.)
            self.assertTrue(np.isfinite(ion.log_k_ion(.001)))
            self.assertLess(ion.log_k_ion(.001), -13000.)
        # Leading threshold asymptotic: k~C*sqrt(T)*exp(-I/T).
        T = 1e-4
        coefficient = (math.sqrt(8*ion.EV_TO_ERG/(math.pi*ion.M_E_G))
                       *1e-13*(.18450-.032226)/13.6**2)
        asymptotic = math.log(coefficient)+.5*math.log(T)-13.6/T
        self.assertAlmostEqual(ion.log_k_ion(T), asymptotic, delta=1e-3)

    def test_zero_temperature_limit_does_not_integrate(self):
        with patch('isothermal_cx_model.ionization.quad') as integrate:
            self.assertEqual(ion.k_ion(0.), 0.)
            self.assertEqual(ion.L_ion(0.), 0.)
            self.assertEqual(ion.log_k_ion(0.), -math.inf)
        integrate.assert_not_called()

    def test_scalar_array_empty_shape_and_input_not_modified(self):
        temperatures = np.array([[.1, 1.], [10., 100.]])
        before = temperatures.copy()
        for function in (ion.sigma_ion_D, ion.k_ion, ion.log_k_ion, ion.L_ion):
            with self.subTest(function=function.__name__):
                self.assertIsInstance(function(1.), float)
                self.assertEqual(function(temperatures).shape, temperatures.shape)
                self.assertEqual(function(np.array([])).shape, (0,))
        np.testing.assert_array_equal(temperatures, before)

    def test_invalid_inputs_not_clipped(self):
        for function in (ion.sigma_ion_D, ion.k_ion, ion.log_k_ion, ion.L_ion):
            for value in (-1., np.nan, np.inf, -np.inf, [1., -1.]):
                with self.subTest(function=function.__name__, value=value), self.assertRaises(ValueError):
                    function(value)

    def test_potential_energy_coefficient_only(self):
        temperatures = np.array([.03, .1, 1., 10., 100.])
        np.testing.assert_allclose(ion.L_ion(temperatures),
            13.6*1.602176634e-12*ion.k_ion(temperatures), rtol=1e-15, atol=0)

    def test_logarithm_rate_consistency_and_cache(self):
        temperatures = np.geomspace(.03, 100, 30)
        np.testing.assert_allclose(np.log(ion.k_ion(temperatures)), ion.log_k_ion(temperatures),
                                   rtol=1e-14, atol=0)
        before = ion._log_k_single.cache_info().hits
        ion.k_ion(temperatures)
        self.assertEqual(ion._log_k_single.cache_info().hits-before, len(temperatures))

    def test_tighter_quadrature_gives_same_rate(self):
        temperatures = [.1192, 1., 10., 100.]
        normal = ion.k_ion(temperatures)
        ion._log_k_single.cache_clear()
        with patch('isothermal_cx_model.ionization._QUAD_RTOL', 1e-11):
            refined = ion.k_ion(temperatures)
        np.testing.assert_allclose(normal, refined, rtol=1e-10, atol=0)

    def test_numerical_failure_not_changed_into_zero(self):
        with patch('isothermal_cx_model.ionization.quad', return_value=(0., 1., {})):
            with self.assertRaisesRegex(RuntimeError, 'quadrature failed'):
                ion.k_ion(2.)
        with patch('isothermal_cx_model.ionization.quad', return_value=(1., 1., {}, 'test failure')):
            with self.assertRaisesRegex(RuntimeError, 'test failure'):
                ion.k_ion(2.)

    def test_command_line_table_and_invalid_temperature(self):
        command = [sys.executable, '-B', '-m', 'isothermal_cx_model.ionization']
        result = subprocess.run(command+['--temperatures', '.001', '.1192', '1', '10'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('cm^3/s', result.stdout)
        self.assertIn('ln(k_ion)', result.stdout)
        self.assertIn('no solver changes', result.stdout)
        invalid = subprocess.run(command+['--temperatures', '-1'], capture_output=True, text=True)
        self.assertEqual(invalid.returncode, 2)


if __name__ == '__main__':
    unittest.main()
