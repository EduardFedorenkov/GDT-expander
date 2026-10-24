"""Run: python -B -m unittest discover -s tests -p test_recombination_coefficients.py -v."""

import unittest
import numpy as np

from isothermal_cx_model.recombination import (
    rr_coefficients, alpha_rr, C_rr, mean_electron_energy_rr_eV, L_gamma_rr,
    EV_TO_ERG, IONIZATION_ENERGY_EV, T_MIN_EV, T_MAX_EV,
)


class RecombinationCoefficientTests(unittest.TestCase):
    def test_reference_values(self):
        # Independently evaluated reference points from the published fits.
        temperature = np.array([1., 5., 10., 50.])
        alpha, cooling = rr_coefficients(temperature)
        np.testing.assert_allclose(alpha, [
            3.8617975480089997e-13, 1.1204426854160366e-13,
            6.20027988458935e-14, 1.3152346535968613e-14,
        ], rtol=1e-12)
        np.testing.assert_allclose(cooling, [
            4.7487395193447475e-25, 6.053875225520784e-25,
            6.138636305933396e-25, 4.79098414448819e-25,
        ], rtol=1e-12)

    def test_scalar_results(self):
        for function in [alpha_rr, C_rr, mean_electron_energy_rr_eV, L_gamma_rr]:
            self.assertIsInstance(function(1.), float)

    def test_array_shape_and_no_input_mutation(self):
        temperature = np.array([[.03, .125], [1., 50.]])
        original = temperature.copy()
        for function in [alpha_rr, C_rr, mean_electron_energy_rr_eV, L_gamma_rr]:
            self.assertEqual(function(temperature).shape, temperature.shape)
        np.testing.assert_array_equal(temperature, original)

    def test_invalid_temperature_not_silently_clipped(self):
        for value in [0., -1., np.nan, np.inf, -np.inf,
                      T_MIN_EV*.5, T_MAX_EV*2, [1., 0.]]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                rr_coefficients(value)

    def test_fit_endpoints(self):
        for temperature in [T_MIN_EV, T_MAX_EV]:
            values = rr_coefficients(temperature)
            self.assertTrue(np.isfinite(values).all())
            self.assertTrue(all(value > 0 for value in values))

    def test_positive_over_domain_and_decreasing_particle_rate(self):
        temperature = np.geomspace(T_MIN_EV, T_MAX_EV, 200)
        alpha, cooling = rr_coefficients(temperature)
        self.assertTrue(np.all(alpha > 0))
        self.assertTrue(np.all(cooling > 0))
        self.assertTrue(np.all(np.diff(alpha) < 0))

    def test_energy_bookkeeping(self):
        temperature = np.geomspace(.03, 100, 50)
        alpha, cooling = rr_coefficients(temperature)
        mean_energy = mean_electron_energy_rr_eV(temperature)
        np.testing.assert_allclose(cooling, alpha*mean_energy*EV_TO_ERG)
        photons = L_gamma_rr(temperature)
        np.testing.assert_allclose(
            photons, cooling+IONIZATION_ENERGY_EV*EV_TO_ERG*alpha)
        self.assertTrue(np.all(photons > cooling))

    def test_captured_electrons_not_assigned_average_maxwellian_energy(self):
        temperature = np.array([.03, .125, 1., 10., 50.])
        captured = mean_electron_energy_rr_eV(temperature)
        self.assertTrue(np.all(captured > 0))
        self.assertTrue(np.all(captured < 1.5*temperature))
        self.assertAlmostEqual(mean_electron_energy_rr_eV(1.), .7675001196663355)


if __name__ == '__main__':
    unittest.main()
