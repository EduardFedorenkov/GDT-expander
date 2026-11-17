"""Molecular source plumbing and sonic-wall checks for individual trials."""

import unittest
from unittest.mock import patch

import numpy as np

from isothermal_model_adas import transport as tr
from isothermal_model_adas.geometry import constant_B, quadratic_area_legacy
from isothermal_model_adas.molecular_amjuel import AMJUELDomainError, make_molecular_coefficients


class TestMolecularTransport(unittest.TestCase):
    def trial(self, **options):
        return tr.solve_trial(.25, 1., .85, 1., 8., (0., 0.), constant_B(), **options)

    def test_wall_formula_matches_rhs_for_each_branch(self):
        theta, beta, jt = .25, 1.2, .7
        b = np.sqrt(8*beta*theta)
        geom = quadratic_area_legacy()
        R, D, Di = .1, 1.2, .14
        before, _ = tr.wall_slope_per_j(theta, jt, beta, R, D, geom, D_ion_t=Di)
        for rates, correction in (((.2, 0., 0.), .2), ((0., .2, 0.), -.2),
                                  ((0., 0., .2), -.6), ((.2, .3, .4), -1.3)):
            with self.subTest(rates=rates):
                slope, _ = tr.wall_slope_per_j(theta, jt, beta, R, D, geom,
                                               D_ion_t=Di, molecular_t=rates)
                dg = tr.rhs_g(1., (jt, b*jt, 2.), theta=theta, beta=beta,
                    recombination=(R, 0.), D=D, ionization=(Di, .01), geometry=geom,
                    molecular=rates, T_star_eV=20.)
                self.assertAlmostEqual(slope, (dg[1]-b*dg[0])/jt)
                self.assertAlmostEqual(slope-before, correction)

    def test_molecules_off_preserves_rhs_and_trial(self):
        kwargs = dict(theta=.25, beta=1., recombination=(0., 0.), D=.5,
                      geometry=constant_B())
        args = (.9, (1., 2., 2.))
        np.testing.assert_array_equal(
            tr.rhs_g(*args, **kwargs),
            tr.rhs_g(*args, **kwargs, molecular=(0., 0., 0.), T_star_eV=20.))
        old = self.trial(D=.5)
        off = self.trial(D=.5, molecular=(0., 0., 0.), T_star_eV=20.)
        self.assertEqual(old['status'], 'integrated')
        self.assertEqual(off['status'], old['status'])
        for key in ('j', 'mu', 'omega'):
            np.testing.assert_array_equal(off[key], old[key])

    def test_positive_and_critical_sonic_slopes_reject_before_integrating(self):
        with patch.object(tr, 'solve_ivp', side_effect=AssertionError('Must not integrate')):
            mar = self.trial(molecular=(.2, 0., 0.), T_star_eV=20.)
            critical = self.trial(molecular=(.2, .2, 0.), T_star_eV=20.)
        self.assertEqual(mar['status'], 'inadmissible_wall')
        self.assertEqual(critical['status'], 'critical_wall_unresolved')

    def test_mai_can_increase_j_without_atomic_ionization(self):
        state = self.trial(molecular=(0., 0., .2), T_star_eV=20., n_points=101)
        self.assertEqual(state['status'], 'integrated')
        self.assertTrue(np.all(np.diff(state['j']) > 0.))
        self.assertEqual(state['D_mai_wall'], .2)
        np.testing.assert_array_equal(state['D_mai_profile'], np.full(101, .2))
        self.assertEqual(state['D_ion_wall'], 0.)

    def test_real_amjuel_provider_and_independent_balances(self):
        provider = make_molecular_coefficients(
            dict(L_cm=100., U_star_cm_per_s=1e6, T_star_eV=20., n_star_cm3=1e13,
                 n0_cm3=1e13, molecular_fraction=.3), scale=1.)
        state = self.trial(molecular=provider, T_star_eV=20., n_points=1501)
        self.assertEqual(state['status'], 'integrated')
        self.assertGreater(np.ptp(state['D_mad_profile']), 0.)
        with self.assertRaisesRegex(ValueError, 'molecular provider'):
            tr.balance_checks(state, (0., 0.))
        with patch.object(tr, 'molecular_sources', side_effect=AssertionError('Independent budget required')):
            checks = tr.balance_checks(state, (0., 0.), molecular=provider)
        for key in ('particle_abs', 'momentum_abs', 'energy_abs'):
            self.assertLess(checks[key], 2e-7)
        self.assertAlmostEqual(checks['particle_net_source_integral'],
                              checks['particle_mai_integral']-checks['particle_mar_integral'])
        self.assertAlmostEqual(checks['energy_loss_integral'],
             sum(checks[f'{branch}_energy_loss_integral'] for branch in ('mar', 'mad', 'mai')))
        self.assertGreater(checks['mad_event_integral'], 0.)

    def test_amjuel_outside_domain_at_wall_rhs_and_profile(self):
        for where in ('wall', 'rhs', 'profile'):
            calls = []

            def provider(theta, N):
                calls.append(np.ndim(N))
                if (where == 'wall' or (where == 'rhs' and len(calls) > 1)
                        or (where == 'profile' and np.ndim(N) > 0)):
                    raise AMJUELDomainError(where)
                return (0., .2, 0.)

            with self.subTest(where=where):
                state = self.trial(molecular=provider, T_star_eV=20.)
                self.assertEqual(state['status'], 'amjuel_outside_domain')
                self.assertEqual(state['reason'], where)

    def test_energy_scale_is_required_for_molecular_trial(self):
        with self.assertRaisesRegex(ValueError, 'T_star_eV'):
            self.trial(molecular=(.1, .2, 0.))


if __name__ == '__main__':
    unittest.main()
