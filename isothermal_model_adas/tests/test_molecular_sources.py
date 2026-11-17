"""Molecular balance terms per g, independent of the transport solver."""

import unittest

import numpy as np

from isothermal_model_adas.sources import molecular_at, molecular_sources


class TestMolecularSources(unittest.TestCase):
    def source(self, molecular=(0., 0., 0.), **scales):
        parameters = dict(theta=.75, beta=.25, T_star_eV=10.)
        parameters.update(scales)
        # A*N = 12, j = A*N*v = 6, ion kinetic + thermal energy = 16.25 eV.
        return molecular_sources(6., 3., .5, 4., molecular=molecular, **parameters)

    def test_individual_branches(self):
        for branch, rates, sign, extra_ev in (
            ('MAR', (.4, 0., 0.), -1., 6.72),
            ('MAD', (0., .4, 0.), 0., 7.46),
            ('MAI', (0., 0., .4), 1., 18.16),
        ):
            with self.subTest(branch=branch):
                expected = (sign*4.8, -2.4, -4.8*(16.25+extra_ev)/10.)
                np.testing.assert_allclose(self.source(rates), expected, rtol=1e-14, atol=0.)

    def test_shared_step_is_counted_once_and_particle_net_can_cancel(self):
        mar, mad, mai = .2, .4, .2
        combined = np.asarray(self.source((mar, mad, mai)))
        sum_of_branches = sum((np.asarray(self.source(rates)) for rates in
                               ((mar, 0., 0.), (0., mad, 0.), (0., 0., mai))))
        np.testing.assert_allclose(combined, sum_of_branches, rtol=1e-14, atol=0.)
        self.assertEqual(combined[0], 0.)
        self.assertLess(combined[1], 0.)
        self.assertLess(combined[2], 0.)

    def test_disabled(self):
        self.assertEqual(self.source(), (0., 0., 0.))

    def test_profile_with_local_provider(self):
        N = np.array([2., 3.])
        j = 4.*N*.5

        def provider(theta, local_N):
            return .1*local_N, .2, theta*.1

        rates = molecular_sources(j, N, .5, 4., theta=.75, beta=.25,
                                  T_star_eV=10., molecular=provider)
        for i, local_N in enumerate(N):
            scalar = molecular_sources(j[i], local_N, .5, 4., theta=.75,
                                       beta=.25, T_star_eV=10., molecular=provider)
            np.testing.assert_allclose([value[i] for value in rates], scalar,
                                       rtol=1e-14, atol=0.)

    def test_invalid_coefficients_and_energy_scale(self):
        for rates in ((1., 2.), (1., -1., 2.), (1., float('nan'), 2.)):
            with self.subTest(rates=rates), self.assertRaises(ValueError):
                molecular_at(rates, .75, 3.)
        with self.assertRaisesRegex(ValueError, 'shape'):
            molecular_at(([.1, .2], 0., 0.), .75, 3.)
        with self.assertRaisesRegex(ValueError, 'T_star_eV'):
            self.source((0., .4, 0.), T_star_eV=0.)


if __name__ == '__main__':
    unittest.main()
