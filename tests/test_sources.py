"""Source bookkeeping and frozen pre-refactor RHS regression checks."""

from math import sqrt
import unittest
from unittest.mock import patch

import numpy as np

from isothermal_cx_model.geometry import constant_B, quadratic_area_legacy
from isothermal_cx_model.sources import atomic_sources, SourceTerms
from isothermal_cx_model.transport import atomic_rhs, solve_trial, balance_checks


def previous_rhs(g, state, *, theta, beta, D, E_exc, R, Q,
                 D_ion, E_ion, geometry):
    """Frozen old formula, deliberately independent of the source assembler."""
    j, mu, _ = state
    b = sqrt(8*beta*theta)
    radical = sqrt(max((mu-b*j)*(mu+b*j), 0.0))
    H = mu+radical
    area = geometry.A_ratio(g)
    phi = H*H/(16*beta*beta*theta*theta*area)
    j_rhs = -R*phi
    if D_ion != 0:
        j_rhs += D_ion*H/(4*beta*theta)
    mu_rhs = -.5*geometry.dlnB_dg(g)*H-D*j-R*j*H/(4*beta*theta*area)
    energy_rhs = -D*(5*mu+radical)/(8*beta)
    if E_exc != 0:
        energy_rhs -= E_exc*H/(4*beta*theta)
    if E_ion != 0:
        energy_rhs -= E_ion*H/(4*beta*theta)
    if R != 0:
        energy_rhs -= R*j*j/(2*beta*area)+(1.5*R*theta+Q)*phi
    return [j_rhs, mu_rhs, energy_rhs]


class SourceTests(unittest.TestCase):
    def local(self, **overrides):
        beta, theta, j, mu, area = .5, .1, 1.2, 2., 3.
        radical = sqrt(mu*mu-8*beta*theta*j*j)
        args = dict(theta=theta, beta=beta, D=4., E_exc=.03,
                    R=.02, Q=.001, D_ion=.1, E_ion=.02)
        args.update(overrides)
        return atomic_sources(j, mu, radical, mu+radical, area, **args)

    def test_sources_agree_with_independent_density_velocity_form(self):
        terms = self.local()
        beta, theta, j, mu, A = .5, .1, 1.2, 2., 3.
        H = mu+sqrt(mu*mu-8*beta*theta*j*j)
        N = H/(4*beta*theta*A)
        v = j/(A*N)
        ion_energy = v*v/(2*beta)+1.5*theta
        expected = {
            'cx': (0., -4*j, 4*A*N*ion_energy),
            'excitation': (0., 0., .03*A*N),
            'ionization': (.1*A*N, 0., .02*A*N),
            'recombination': (-.02*A*N*N, -.02*j*N,
                              .02*A*N*N*ion_energy+.001*A*N*N),
        }
        actual = terms.by_process()
        self.assertEqual(set(actual), set(expected))
        for name in expected:
            np.testing.assert_allclose(actual[name], expected[name], rtol=3e-15)
        np.testing.assert_allclose(terms.totals, np.sum(list(expected.values()), axis=0),
                                   rtol=3e-15)

    def test_empty_channels_and_geometry_only(self):
        terms = SourceTerms((), (), ())
        self.assertEqual(terms.totals, (0., 0., 0.))
        self.assertEqual(terms.by_process(), {})
        self.assertEqual(terms.rhs(2.), [0., 2., 0.])

    def test_disabled_processes_have_zero_rates(self):
        terms = self.local(D=0., E_exc=0., R=0., Q=0., D_ion=0., E_ion=0.)
        np.testing.assert_array_equal(terms.totals, 0.)
        np.testing.assert_array_equal(terms.rhs(3.), [0., 3., 0.])

    def test_energy_sign_and_no_ionization_momentum(self):
        rates = self.local().by_process()
        self.assertGreater(rates['ionization'].particles, 0)
        self.assertEqual(rates['ionization'].momentum, 0)
        self.assertLess(rates['recombination'].particles, 0)
        self.assertTrue(all(rate.power_loss >= 0 for rate in rates.values()))
        self.assertLess(self.local().rhs(0.)[2], 0)

    def test_rhs_matches_previous_operation_order_bitwise(self):
        rng = np.random.default_rng(42803)
        for index in range(300):
            beta, theta, j = np.exp(rng.uniform(-4, 3, 3))
            b = sqrt(8*beta*theta)
            mu = b*j*(1+rng.uniform(0, 5))
            args = dict(theta=theta, beta=beta, D=float(rng.uniform(0, 50)),
                E_exc=float(rng.uniform(0, 1)), R=float(rng.uniform(0, 1)),
                Q=float(rng.uniform(0, 1)), D_ion=float(rng.uniform(0, 50)),
                E_ion=float(rng.uniform(0, 1)),
                geometry=constant_B() if index%2 else quadratic_area_legacy())
            if index%5 == 0:
                for key in ('D', 'E_exc', 'R', 'Q', 'D_ion', 'E_ion'):
                    args[key] = 0.
            old = np.asarray(previous_rhs(.8, [j, mu, .7], **args))
            new = np.asarray(atomic_rhs(.8, [j, mu, .7], **args))
            self.assertEqual(new.tobytes(), old.tobytes())

    def test_provisional_stage_radical_clipping_unchanged(self):
        args = dict(theta=.1, beta=.5, D=4., E_exc=.03, R=.02,
                    Q=.001, D_ion=.1, E_ion=.02, geometry=constant_B())
        for mu in (sqrt(.4), sqrt(.4)*(1-1e-10)):
            np.testing.assert_array_equal(atomic_rhs(.8, [1., mu, .7], **args),
                previous_rhs(.8, [1., mu, .7], **args))

    def test_integral_checks_remain_independent(self):
        state = solve_trial(.1, 1.2, .5, 4., 5., constant_B(),
            E_exc=.03, rr=(.02, .001), ionization=(.1, .02), n_points=4000)
        self.assertEqual(state['status'], 'energy_matched')
        with patch('isothermal_cx_model.transport.atomic_sources',
                   side_effect=AssertionError('integral check reused RHS sources')):
            errors = balance_checks(state, .5)
        self.assertLess(max(errors[k] for k in
            ('particle_abs', 'momentum_abs', 'energy_abs')), 5e-6)


if __name__ == '__main__':
    unittest.main()
