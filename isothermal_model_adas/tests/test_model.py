"""Model checks prepared for a later explicit numerical validation step.

Some tests integrate IVPs; they have NOT been run during the code transfer.
Synthetic providers isolate transport from the licensed ADAS tables.
"""

import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from isothermal_model_adas.config import load_config
from isothermal_model_adas.geometry import constant_B, linear_B, quadratic_area_legacy
from isothermal_model_adas.sources import coefficients_at, adas_sources
from isothermal_model_adas.transport import rhs_g, solve_trial, balance_checks
from isothermal_model_adas.stationary import solve_stationary
from isothermal_model_adas.recombination_adas import ADASDomainError


class AlgebraTests(unittest.TestCase):
    def test_particle_momentum_energy(self):
        j, N, v, A, theta, beta, R, Q = .8, 2., .2, 2., .1, 1., .03, -.01
        Jg, Mg, Wg = adas_sources(j, N, v, A, theta=theta, beta=beta, R=R, Q_eff=Q)
        self.assertAlmostEqual(Jg, -R*A*N*N)
        self.assertAlmostEqual(Mg, -R*j*N)
        self.assertAlmostEqual(Wg, -A*N*N*(R*(v*v/(2*beta)+1.5*theta)+Q))

    def test_signed_q_and_pure_prb_limit(self):
        self.assertEqual(coefficients_at((0., -.2), .1, 1.), (0., -.2))
        self.assertEqual(adas_sources(1., 2., .1, 3., theta=.1, beta=1., R=0., Q_eff=.5),
                         (0., 0., -6.))
        with self.assertRaises(ValueError):
            coefficients_at((-.1, 0.), .1, 1.)

    def test_local_density_and_arrays(self):
        provider = lambda theta, N: (.1*np.asarray(N), -.2*np.asarray(N))
        N = np.array([1., 2., 3.])
        R, Q = coefficients_at(provider, .1, N)
        np.testing.assert_allclose(R, .1*N)
        np.testing.assert_allclose(Q, -.2*N)

    def test_expanded_and_primitive_energy_agree(self):
        theta, beta, j, mu, A, R, Q = .03, 1.2, .7, 1.1, 12., .02, -.01
        radical = np.sqrt(mu*mu-8*beta*theta*j*j)
        H = mu+radical
        N, v = H/(4*beta*theta*A), 4*beta*theta*j/H
        actual = adas_sources(j, N, v, A, theta=theta, beta=beta, R=R, Q_eff=Q)[2]
        expected = -R*j*j/(2*beta*A)-(1.5*R*theta+Q)*A*N*N
        self.assertAlmostEqual(actual, expected, places=13)

    def test_chemical_energy_identity(self):
        # Q=P-epsilon_I*R: (w+epsilon_I*j)'=-Phi*(R*e_i+P).
        theta, beta, j, N, v, A, R, P, I = .02, 1., .7, 3., .1, 2., .01, .003, .1
        Jg, _, Wg = adas_sources(j, N, v, A, theta=theta, beta=beta, R=R, Q_eff=P-I*R)
        self.assertAlmostEqual(Wg+I*Jg, -A*N*N*(R*(v*v/(2*beta)+1.5*theta)+P))

    def test_rhs_frozen_equals_local_constant_provider(self):
        kw = dict(theta=.05, beta=1., geometry=linear_B(.5))
        state = [.8, 1., 1.]
        np.testing.assert_allclose(rhs_g(.8, state, recombination=(.001, -.01), **kw),
            rhs_g(.8, state, recombination=lambda t, n: (.001, -.01), **kw))

    def test_config_explicit_flows(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'config.json'
            path.write_text(json.dumps(dict(A_m_cm2=2., L_cm=10., lambda_sh=5.,
                J0_per_s=1e20, M0_dyn=2e4, W0_erg_per_s=1e11)), encoding='utf-8')
            p = load_config(path)
        self.assertEqual(p['source_mode'], 'flows')
        self.assertAlmostEqual(p['T_star_erg'], p['W0_erg_per_s']/p['J0_per_s'])
        self.assertAlmostEqual(p['n_star_cm3']*p['A_star_cm2']*p['U_star_cm_per_s']/p['J0_per_s'], 1.)
        self.assertEqual(p['n0_cm3'], 0.)

    def test_config_central_same_reference_scales(self):
        p = load_config(Path(__file__).parents[1]/'config.json')
        self.assertAlmostEqual(p['T_star_eV'], 1200.)
        self.assertAlmostEqual(p['beta'], 1.2732395447351628)


class DeferredNumericalTests(unittest.TestCase):
    def test_expanding_wall_is_inadmissible(self):
        for geometry in (linear_B(500), quadratic_area_legacy(), constant_B()):
            state = solve_trial(.05, .8, .7, 1., 8., (.001, -.01), geometry)
            self.assertEqual(state['status'], 'inadmissible_wall')

    def test_domain_is_reported_not_clipped(self):
        def unavailable(theta, N):
            raise ADASDomainError('synthetic outside-table state')
        state = solve_trial(.05, .8, .7, 1., 8., unavailable, linear_B(.5))
        self.assertEqual(state['status'], 'adas_outside_domain')

    def test_signed_energy_no_cooling_or_wall_energy_cutoff(self):
        # Contracting geometry is a numerical control, NOT the GDL expander.
        state = solve_trial(.05, .8, .7, 1., 30., (.001, -.1), linear_B(.5))
        self.assertEqual(state['status'], 'integrated')
        self.assertGreater(state['omega'][-1], 1.)
        self.assertGreater(state['omega'][-1], state['omega'][0])
        self.assertGreater(state['j'][0], state['j'][-1])
        budgets = balance_checks(state, (.001, -.1))
        for key in ('particle_abs', 'momentum_abs', 'energy_abs'):
            self.assertLess(budgets[key], 1e-5)

    def test_local_coefficients_not_frozen_at_wall(self):
        def provider(theta, N):
            return .0001*np.asarray(N), -.001*np.asarray(N)
        state = solve_trial(.05, .8, .7, 1., 8., provider, linear_B(.5))
        self.assertEqual(state['status'], 'integrated')
        self.assertGreater(np.ptp(state['R_profile']), 0.)
        np.testing.assert_allclose(state['R_profile'], .0001*state['n_over_n_star'])

    def test_stationary_failure_does_not_become_a_solution(self):
        result = solve_stationary(.05, .8, .7, 1., 8., (.001, 0.), quadratic_area_legacy())
        self.assertEqual(result['status'], 'invalid_initial_trial')
        self.assertNotIn('solution', result)
        self.assertIn('inadmissible_wall', result['diagnostic']['invalid_trials'])


if __name__ == '__main__':
    unittest.main()
