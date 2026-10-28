"""Independent trial-IVP checks, not tests of a full (theta,j_t) search."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.integrate import simpson

from simple_cx_model.config import load_cx_config
from isothermal_cx_model.geometry import (
    MagneticGeometry, constant_B, linear_B, quadratic_area_legacy,
)
from isothermal_cx_model.inlet import KineticInletDomainError, make_inlet_momentum
from isothermal_cx_model.recombination import make_rr_coefficients, rr_coefficients
from isothermal_cx_model.solver_general import solve_at_theta
from isothermal_cx_model.solver_recombination import solve_trial
from isothermal_cx_model.profiles import compute_profiles
from isothermal_cx_model.cx import make_D
from isothermal_cx_model.excitation import make_E_exc


CONFIG = Path(__file__).resolve().parents[1]/'simple_cx_model/cx_config.json'


def trial(**kwargs):
    options = dict(theta=.1, j_target=.7, beta=.5, D=4., lambda_sh=5.,
                   geometry=constant_B(), n_points=500)
    options.update(kwargs)
    return solve_trial(**options)


class CoefficientFactoryTests(unittest.TestCase):
    def test_normalization_and_common_multiplier(self):
        p = load_cx_config(CONFIG)
        theta = .01
        alpha, cooling = rr_coefficients(theta*p['T_star_eV'])
        R, Q = make_rr_coefficients(p, scale=.25)(theta)
        self.assertAlmostEqual(R, .25*p['L_cm']*p['n_star_cm3']*alpha/p['U_star_cm_per_s'])
        self.assertAlmostEqual(Q/R, cooling/(alpha*p['T_star_erg']))
        np.testing.assert_allclose(make_rr_coefficients(p, scale=1)(theta),
                                   np.array([R, Q])*4, rtol=1e-14)

    def test_off_does_not_require_parameters_or_evaluate_fit(self):
        self.assertEqual(make_rr_coefficients({}, scale=0), (0., 0.))
        state = trial(theta=1e-10, rr=make_rr_coefficients({}, scale=0))
        self.assertNotEqual(state['status'], 'recombination_temperature_out_of_range')

    def test_copied_scales_and_no_neutral_density_dependency(self):
        p = load_cx_config(CONFIG)
        rates = make_rr_coefficients(p, scale=1)
        before = rates(.01)
        p['n0_cm3'] *= 20
        np.testing.assert_array_equal(rates(.01), before)
        np.testing.assert_array_equal(make_rr_coefficients(p, scale=1)(.01), before)
        p['n_star_cm3'] *= 2
        np.testing.assert_array_equal(rates(.01), before)
        np.testing.assert_allclose(make_rr_coefficients(p, scale=1)(.01), np.array(before)*2)

    def test_invalid_factory_inputs(self):
        for scale in [-1, 2, np.nan, np.inf]:
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                make_rr_coefficients({}, scale=scale)
        p = load_cx_config(CONFIG)
        for key in ['L_cm', 'n_star_cm3', 'U_star_cm_per_s', 'T_star_erg', 'T_star_eV']:
            with self.subTest(key=key), self.assertRaises(ValueError):
                make_rr_coefficients(dict(p, **{key: 0}), scale=1)


class TrialEquationTests(unittest.TestCase):
    def test_zero_rr_reproduces_old_solver(self):
        for geometry in [constant_B(), linear_B(2), quadratic_area_legacy()]:
            with self.subTest(geometry=geometry.name):
                old = solve_at_theta(.1, .5, 4, 5, geometry, E_exc=.02, n_points=500,
                                     rtol=1e-11, atol=1e-13)
                new = solve_trial(.1, 1., .5, 4, 5, geometry, E_exc=.02, n_points=500,
                                  rtol=1e-11, atol=1e-13)
                self.assertEqual(new['status'], old['status'])
                if new['status'] == 'energy_matched':
                    self.assertAlmostEqual(new['alpha'], old['alpha'], delta=2e-8)
                    np.testing.assert_allclose(new['mu'], old['mu'], atol=2e-7)
                    np.testing.assert_allclose(new['omega'], old['omega'], atol=2e-7)
                    np.testing.assert_array_equal(new['j'], 1.)
                    self.assertEqual(new['R_J'], 0)

    def test_constant_area_independent_no_rr_analytic_solution(self):
        # j_t != 1 also verifies the j factors in CX and excitation cooling.
        beta, theta, D, lam, E, jt = .5, .1, 4., 5., .02, .6
        state = solve_trial(theta, jt, beta, D, lam, constant_B(), E_exc=E,
                            rtol=1e-11, atol=1e-13, n_points=1000)
        self.assertEqual(state['status'], 'energy_matched')
        b = np.sqrt(8*beta*theta)
        mu = b*jt + D*jt*(1-state['g'])
        lower = b*jt
        rad = np.sqrt(np.maximum(mu*mu-lower*lower, 0))
        integral_rad = .5*(mu*rad-lower*lower*np.log((mu+rad)/lower))
        integral_mu = .5*(mu*mu-lower*lower)
        expected = (lam*theta*jt + (5*integral_mu+integral_rad)/(8*beta*jt)
                    + E*(integral_mu+integral_rad)/(4*beta*theta*D*jt))
        np.testing.assert_allclose(state['mu'], mu, atol=1e-10)
        np.testing.assert_allclose(state['omega'], expected, atol=3e-9)
        self.assertAlmostEqual(state['R_J'], jt-1)

    def test_rr_changes_particle_flux_but_does_not_reject_positive_RJ(self):
        state = trial(j_target=1., rr=(.02, .001), E_exc=.03)
        self.assertEqual(state['status'], 'energy_matched')
        self.assertGreater(state['R_J'], 0)
        self.assertTrue(np.all(np.diff(state['j']) < 0))
        self.assertTrue(np.isfinite(state['R_M']))
        self.assertAlmostEqual(state['j'][-1], 1.)

    def test_variable_j_algebra_and_sonic_wall(self):
        state = trial(rr=(.02, .001), E_exc=.03)
        self.assertEqual(state['status'], 'energy_matched')
        j, mu, N, v, A = [state[key] for key in
                          ['j', 'mu', 'n_over_n_star', 'u_over_u_star', 'A_over_A_star']]
        np.testing.assert_allclose(A*N*v, j, rtol=1e-13)
        np.testing.assert_allclose(j*(v+2*.5*.1/v), mu, rtol=1e-13)
        np.testing.assert_allclose(state['omega'],
            j*(v*v/(2*.5)+5*.1)+state['Aq_e_over_W0'], rtol=1e-13)
        self.assertAlmostEqual(state['u_over_cs'][-1], 1)
        self.assertTrue(np.all(state['u_over_cs'][:-1] < 1))
        self.assertAlmostEqual(state['mu'][-1], state['b']*state['j_target'])
        self.assertAlmostEqual(state['omega'][-1], 5*.1*.7)

    def test_inlet_target_below_sonic_margin_still_gives_residual(self):
        state = trial(rr=(.02, .001), inlet_momentum=lambda alpha: .01)
        self.assertEqual(state['status'], 'energy_matched')
        self.assertLess(state['inlet_target_sonic_margin'], 0)
        self.assertTrue(np.isfinite(state['R_M']))

    def test_energy_domain_uses_theta_times_target_flux(self):
        state = trial(theta=.3, j_target=.2)
        self.assertEqual(state['status'], 'energy_matched')
        self.assertGreater(5*.3, 1)
        self.assertLess(5*.3*.2, 1)
        self.assertEqual(trial(theta=.3, j_target=.8)['status'], 'energy_temperature_limit')

    def test_recombination_wall_condition_replaces_old_condition(self):
        b = np.sqrt(8*.5*.1)
        R, jt = .2, .7
        critical_D = R*b*jt/(4*.5*.1)
        # Without RR every positive D is wall-admissible for constant area.
        self.assertEqual(trial(D=.5*critical_D, rr=(R, .001))['status'], 'inadmissible_wall')
        self.assertEqual(trial(D=critical_D, rr=(R, .001))['status'], 'critical_wall_unresolved')
        accepted = trial(D=4., rr=(R, .001))
        self.assertEqual(accepted['status'], 'energy_matched')
        self.assertAlmostEqual(accepted['wall_slope'], jt*(critical_D-4))

    def test_coefficients_evaluated_once(self):
        calls = []
        state = trial(D=lambda t: calls.append(('D', t)) or 4.,
            E_exc=lambda t: calls.append(('E', t)) or .03,
            rr=lambda t: calls.append(('RR', t)) or (.02, .001))
        self.assertEqual(state['status'], 'energy_matched')
        self.assertEqual(calls, [('D', .1), ('E', .1), ('RR', .1)])

    def test_dimensional_particle_momentum_energy_integrals(self):
        p = load_cx_config(CONFIG)
        theta, jt = .001, .8
        D = make_D(p, mode='temperature')
        E = make_E_exc(p, scale=.25)
        rr = make_rr_coefficients(p, scale=1)
        state = solve_trial(theta, jt, p['beta'], D, p['lambda_sh'],
            constant_B(), E_exc=E, rr=rr, n_points=8000,
            rtol=1e-11, atol=1e-13)
        self.assertEqual(state['status'], 'energy_matched')
        z = p['L_cm']*state['g']
        area = p['A0_cm2']*state['A_over_A_star']
        density = p['n_star_cm3']*state['n_over_n_star']
        velocity = p['U_star_cm_per_s']*state['u_over_u_star']
        energy = theta*p['T_star_erg']
        alpha, cooling = rr_coefficients(theta*p['T_star_eV'])
        S = alpha*density**2
        nu = state['D']*p['U_star_cm_per_s']/p['L_cm']
        exc_coef = state['E_exc']*p['U_star_cm_per_s']*p['T_star_erg']/(p['L_cm']*p['n0_cm3'])
        kinetic = .5*p['m_D_g']*velocity**2
        losses = (density*nu*(kinetic+1.5*energy)
                  + density*p['n0_cm3']*exc_coef
                  + S*(kinetic+1.5*energy)+cooling*density**2)
        particle = simpson(area*S, x=z)/p['J0_per_s']
        momentum = simpson(area*p['m_D_g']*velocity*(density*nu+S), x=z)/p['M0_dyn']
        power = simpson(area*losses, x=z)/p['W0_erg_per_s']
        self.assertAlmostEqual(particle/(state['j'][0]-jt), 1., delta=2e-6)
        self.assertAlmostEqual(momentum/(state['mu'][0]-state['mu'][-1]), 1., delta=2e-6)
        self.assertAlmostEqual(power/(1-state['omega'][-1]), 1., delta=2e-6)


class StopStatusTests(unittest.TestCase):
    def test_invalid_inputs(self):
        for kwargs in [dict(j_target=0), dict(j_target=np.nan),
                       dict(theta=0), dict(beta=-1), dict(lambda_sh=np.inf),
                       dict(D=-1), dict(E_exc=np.nan), dict(rr=(.1, -.2)),
                       dict(rr=(.1,)), dict(rr=(0, .1)), dict(n_points=1),
                       dict(rtol=0), dict(max_step=0)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                trial(**kwargs)

    def test_temperature_domain_is_explicit_status(self):
        p = load_cx_config(CONFIG)
        state = trial(theta=1e-12, rr=make_rr_coefficients(p, scale=1))
        self.assertEqual(state['status'], 'recombination_temperature_out_of_range')
        self.assertTrue(np.isnan(state['R_J']))
        self.assertTrue(np.isnan(state['R_M']))

    def test_geometry_limit_not_misreported_as_a_solution(self):
        geometry = MagneticGeometry('short', lambda g: np.ones_like(g),
                                    lambda g: np.zeros_like(g), g_min=.999)
        state = trial(geometry=geometry, rr=(.02, .001))
        self.assertEqual(state['status'], 'geometry_limit')
        self.assertTrue(np.isnan(state['R_J']))
        self.assertEqual(state['g_stop'], .999)

    def test_sonic_return(self):
        state = trial(D=.7, geometry=quadratic_area_legacy(), j_target=1.)
        self.assertEqual(state['status'], 'sonic_return')
        self.assertTrue(np.isnan(state['R_M']))

    def test_kinetic_domain_failure_preserves_energy_profile(self):
        def boundary(alpha):
            raise KineticInletDomainError('test inlet outside domain')
        state = trial(inlet_momentum=boundary, rr=(.02, .001))
        self.assertEqual(state['status'], 'kinetic_inlet_outside_domain')
        self.assertIn('j', state)
        self.assertTrue(np.isfinite(state['R_J']))
        self.assertTrue(np.isnan(state['R_M']))

    def test_numerical_failure_is_an_explicit_status(self):
        failed = SimpleNamespace(success=False, message='test integrator failure',
                                 t=np.array([1.]), y=np.array([[.7], [.5], [.35]]))
        with patch('isothermal_cx_model.solver_atomic.solve_ivp', return_value=failed):
            state = trial()
        self.assertEqual(state['status'], 'integration_failed')
        self.assertIn('test integrator failure', state['reason'])
        self.assertTrue(np.isnan(state['R_J']))

    def test_old_constant_J_profiles_are_guarded(self):
        state = trial(rr=(.02, .001))
        with self.assertRaisesRegex(ValueError, 'assumes J=J0'):
            compute_profiles(state, {})

    def test_actual_kinetic_inlet_and_cold_case(self):
        p = load_cx_config(CONFIG)
        p.update(n0_cm3=1e15, lambda_sh=7.)
        geometry = quadratic_area_legacy()
        theta = .02964039754150312/p['T_star_eV']
        state = solve_trial(theta, .5, p['beta'], make_D(p, 'temperature'),
            7., geometry, E_exc=make_E_exc(p, scale=1),
            rr=make_rr_coefficients(p, scale=1),
            inlet_momentum=make_inlet_momentum(p, geometry), n_points=1000)
        self.assertEqual(state['status'], 'energy_matched')
        self.assertEqual(state['inlet_momentum'], 'kinetic')
        self.assertTrue(np.isfinite([state['R_J'], state['R_M']]).all())
        self.assertTrue(np.all(state['j'] > 0))
        refined = solve_trial(theta, .5, p['beta'], make_D(p, 'temperature'),
            7., geometry, E_exc=make_E_exc(p, scale=1),
            rr=make_rr_coefficients(p, scale=1),
            inlet_momentum=make_inlet_momentum(p, geometry), n_points=2000,
            rtol=1e-11, atol=1e-13, max_step=.01)
        self.assertEqual(refined['status'], 'energy_matched')
        self.assertAlmostEqual(refined['alpha'], state['alpha'], delta=2e-7)
        np.testing.assert_allclose([refined['R_J'], refined['R_M']],
                                   [state['R_J'], state['R_M']], rtol=1e-6, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
