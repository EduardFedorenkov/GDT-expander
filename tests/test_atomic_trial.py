"""Stage 1: coefficients and IVPs; no automatic ionizing BVP search."""

from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
import numpy as np
from scipy.integrate import simpson

from simple_cx_model.config import load_cx_config
from isothermal_cx_model.ionization import make_ionization_coefficients, k_ion, IONIZATION_ENERGY_ERG
from isothermal_cx_model.geometry import constant_B, quadratic_area_legacy
from isothermal_cx_model.transport import solve_trial, atomic_rhs, balance_checks
from isothermal_cx_model.cx import make_D
from isothermal_cx_model.excitation import make_E_exc, L_exc
from isothermal_cx_model.recombination import make_rr_coefficients, rr_coefficients
from isothermal_cx_model.inlet import make_inlet_momentum

CONFIG = Path(__file__).resolve().parents[1]/'simple_cx_model/cx_config.json'


def trial(**kwargs):
    options = dict(theta=.1, j_target=1.2, beta=.5, D=4., lambda_sh=5.,
        geometry=constant_B(), rr=(.02, .001), ionization=(.1, .02),
        n_points=4000, rtol=1e-11, atol=1e-13)
    options.update(kwargs)
    return solve_trial(**options)


class FactoryTests(unittest.TestCase):
    def test_normalization_and_linked_energy_cost(self):
        p = load_cx_config(CONFIG)
        theta = 3/p['T_star_eV']
        D, E = make_ionization_coefficients(p, scale=.25)(theta)
        self.assertAlmostEqual(D, .25*p['L_cm']*p['n0_cm3']*k_ion(3)/p['U_star_cm_per_s'])
        self.assertAlmostEqual(E/D, IONIZATION_ENERGY_ERG/p['T_star_erg'])
        np.testing.assert_allclose(make_ionization_coefficients(p, scale=1)(theta),
                                   4*np.array([D, E]), rtol=1e-14)

    def test_off_and_no_neutrals_do_not_evaluate_atomic_integral(self):
        p = load_cx_config(CONFIG)
        with patch('isothermal_cx_model.ionization.k_ion', side_effect=AssertionError('must not run')):
            self.assertEqual(make_ionization_coefficients({}, scale=0), (0., 0.))
            self.assertEqual(make_ionization_coefficients(p, scale=1, n0_cm3=0), (0., 0.))

    def test_source_copied_and_neutral_override(self):
        p = load_cx_config(CONFIG)
        theta = .01
        coefficients = make_ionization_coefficients(p, scale=1)
        before = np.array(coefficients(theta))
        np.testing.assert_allclose(make_ionization_coefficients(p, scale=1,
            n0_cm3=2*p['n0_cm3'])(theta), 2*before)
        p['n_star_cm3'] *= 3
        np.testing.assert_array_equal(make_ionization_coefficients(p, scale=1)(theta), before)
        p['n0_cm3'] *= 5
        p['T_star_eV'] *= 2
        np.testing.assert_array_equal(coefficients(theta), before)

    def test_invalid_factory_inputs_and_temperature(self):
        for scale in (-1, 2, np.nan, np.inf):
            with self.assertRaises(ValueError):
                make_ionization_coefficients({}, scale=scale)
        p = load_cx_config(CONFIG)
        for key in ('L_cm', 'U_star_cm_per_s', 'T_star_erg', 'T_star_eV'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                make_ionization_coefficients(dict(p, **{key: 0}), scale=1)
        with self.assertRaises(ValueError):
            make_ionization_coefficients(p, scale=1, n0_cm3=-1)
        with self.assertRaises(ValueError):
            make_ionization_coefficients(dict(p, T_star_erg=p['T_star_erg']*2), scale=1)
        coefficients = make_ionization_coefficients(p, scale=1)
        self.assertEqual(coefficients(0), (0., 0.))
        for theta in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                coefficients(theta)


class AtomicEquationTests(unittest.TestCase):
    def test_off_reproduces_frozen_pre_ionization_reference(self):
        s = trial(j_target=.7, E_exc=.03, ionization=(0., 0.))
        self.assertEqual(s['status'], 'energy_matched')
        # Captured from the former solver before changing the equations.
        np.testing.assert_allclose([s['alpha'], s['R_J'], s['R_M']],
            [.8436705717496279, -.19007443739364616, -.08207431516460151], atol=3e-11, rtol=0)
        default = solve_trial(.1, .7, .5, 4, 5, constant_B(), E_exc=.03,
            rr=(.02, .001), n_points=4000, rtol=1e-11, atol=1e-13)
        np.testing.assert_array_equal(s['j'], default['j'])
        np.testing.assert_array_equal(s['omega'], default['omega'])

    def test_no_explicit_ionization_momentum_source(self):
        options = dict(theta=.1, beta=.5, D=4., E_exc=.03, R=.02, Q=.001,
                       geometry=constant_B())
        without = np.array(atomic_rhs(.8, [1.2, 2., .5], D_ion=0, E_ion=0, **options))
        with_ion = np.array(atomic_rhs(.8, [1.2, 2., .5], D_ion=.3, E_ion=.06, **options))
        self.assertEqual(with_ion[1], without[1])
        self.assertGreater(with_ion[0], without[0])
        self.assertAlmostEqual((with_ion[2]-without[2])/(with_ion[0]-without[0]), -.2)

    def test_pure_ionization_invariants_and_analytic_free_boundary(self):
        # Constant area and mu=1. With j_a=1, j_t=1/b the energy BC gives
        # lambda*b^2/(8*beta)-(1+epsilon)*b+epsilon=0. Both roots are tested.
        beta, lam, epsilon, Di = .5, 5., .2, 1.
        for b in np.sort(np.roots([lam/(8*beta), -(1+epsilon), epsilon])):
            theta, jt = b*b/(8*beta), 1/b
            with self.subTest(theta=theta):
                state = trial(theta=theta, j_target=jt, D=0, rr=(0., 0.),
                              ionization=(Di, epsilon*Di))
                self.assertEqual(state['status'], 'energy_matched')
                self.assertGreater(jt, 1)
                np.testing.assert_allclose(state['mu'], 1., rtol=0, atol=3e-13)
                np.testing.assert_allclose(state['omega']+epsilon*state['j'], 1+epsilon,
                                           rtol=0, atol=3e-12)
                F = lambda y: np.arcsin(y)-y/(1+np.sqrt(max(1-y*y, 0)))
                alpha = 1-b/(2*Di)*(F(1)-F(b))
                self.assertAlmostEqual(state['alpha'], alpha, delta=1e-9)
                self.assertAlmostEqual(state['R_J'], 0., delta=1e-10)
                self.assertAlmostEqual(state['R_M'], 0., delta=1e-10)
                self.assertTrue(np.all(np.diff(state['j']) > 0))
                errors = balance_checks(state, beta)
                self.assertLess(max(errors[k] for k in ('particle_abs', 'momentum_abs', 'energy_abs')), 5e-6)

    def test_mixed_reactions_allow_nonmonotone_particle_flux(self):
        state = trial()
        self.assertEqual(state['status'], 'energy_matched')
        self.assertGreater(state['j_target'], 1)
        difference = np.diff(state['j'])
        self.assertTrue(np.any(difference > 1e-9))
        self.assertTrue(np.any(difference < -1e-9))
        self.assertTrue(np.all(state['j'] > 0))
        errors = balance_checks(state, .5)
        self.assertGreater(errors['particle_ionization_integral'], 0)
        self.assertGreater(errors['particle_recombination_integral'], 0)
        self.assertAlmostEqual(errors['particle_net_source_integral'],
                               state['j'][-1]-state['j'][0], delta=1e-6)

    def test_new_wall_criterion_and_critical_case(self):
        theta, beta, R, jt, Di = .1, .5, .2, 1.2, .1
        b = np.sqrt(8*beta*theta)
        critical_D = R*b*jt/(4*beta*theta)-2*Di
        for D, expected in [(critical_D*.5, 'inadmissible_wall'),
                            (critical_D, 'critical_wall_unresolved')]:
            state = trial(D=D, rr=(R, .001), ionization=(Di, .02))
            self.assertEqual(state['status'], expected)
        state = trial(D=4, rr=(R, .001), ionization=(Di, .02))
        self.assertAlmostEqual(state['wall_slope'], jt*(critical_D-4))

    def test_coefficients_evaluated_once(self):
        calls = []
        state = trial(ionization=lambda theta: calls.append(theta) or (.1, .02))
        self.assertEqual(state['status'], 'energy_matched')
        self.assertEqual(calls, [.1])
        self.assertEqual((state['D_ion'], state['E_ion']), (.1, .02))

    def test_zero_particle_event_is_not_a_fake_root(self):
        state = trial(j_target=.5, D=0., rr=(0., 0.), ionization=(1., .2))
        self.assertEqual(state['status'], 'nonpositive_particle_flux')
        self.assertTrue(np.isnan(state['R_J']))
        self.assertTrue(np.isnan(state['R_M']))

    def test_energy_limit_and_invalid_ionization_pair(self):
        self.assertEqual(trial(j_target=3.)['status'], 'energy_temperature_limit')
        for values in ((-1., .1), (.1, np.nan), (.1,), (0, .1)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                trial(ionization=values)

    def test_numerical_coefficient_errors_are_not_physical_stops(self):
        def fail(theta):
            raise RuntimeError('test atomic quadrature failure')
        with self.assertRaisesRegex(RuntimeError, 'quadrature failure'):
            trial(ionization=fail)

    def test_all_dimensional_integral_balances_on_physical_trial(self):
        p = load_cx_config(CONFIG)
        p.update(n0_cm3=1e15, lambda_sh=7.)
        geometry = quadratic_area_legacy()
        temperature = 3.272
        state = solve_trial(temperature/p['T_star_eV'], 5, p['beta'], make_D(p, 'temperature'),
            7, geometry, E_exc=make_E_exc(p, scale=1), rr=make_rr_coefficients(p, scale=1),
            ionization=make_ionization_coefficients(p, scale=1),
            inlet_momentum=make_inlet_momentum(p, geometry),
            n_points=8000, rtol=1e-11, atol=1e-13)
        self.assertEqual(state['status'], 'energy_matched')
        A = p['A0_cm2']*state['A_over_A_star']
        n = p['n_star_cm3']*state['n_over_n_star']
        u = p['U_star_cm_per_s']*state['u_over_u_star']
        T = state['theta']*p['T_star_erg']
        z = p['L_cm']*state['g']
        alpha, cooling = rr_coefficients(temperature)
        ion = n*p['n0_cm3']*k_ion(temperature)
        rec = n*n*alpha
        nu = state['D']*p['U_star_cm_per_s']/p['L_cm']
        momentum_rhs = (2*n*T*A*(-geometry.dlnB_dg(state['g']))/p['L_cm']
                        -A*p['m_D_g']*u*(n*nu+rec))
        ion_energy = .5*p['m_D_g']*u*u+1.5*T
        loss = A*(n*nu*ion_energy+n*p['n0_cm3']*L_exc(temperature)
                  +rec*ion_energy+n*n*cooling+IONIZATION_ENERGY_ERG*ion)
        particle = simpson(A*(ion-rec), x=z)/p['J0_per_s']
        momentum = simpson(momentum_rhs, x=z)/p['M0_dyn']
        energy = simpson(loss, x=z)/p['W0_erg_per_s']
        self.assertAlmostEqual(state['j'][-1]-state['j'][0], particle, delta=2e-6)
        self.assertAlmostEqual(state['mu'][-1]-state['mu'][0], momentum, delta=2e-6)
        self.assertAlmostEqual(state['omega'][0]-state['omega'][-1], energy, delta=2e-6)
        errors = balance_checks(state, p['beta'])
        self.assertLess(max(errors[k] for k in ('particle_abs', 'momentum_abs', 'energy_abs')), 2e-6)


    def test_trial_command_line_does_not_claim_stationarity(self):
        command = [sys.executable, '-B', '-m', 'isothermal_cx_model.trial_example']
        result = subprocess.run(command+['--temperature', '3.272', '--j-target', '5',
            '--density', '1e15', '--lambda-sh', '7'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertIn('TRIAL ONLY', result.stdout)
        self.assertIn('energy_matched', result.stdout)
        self.assertIn('particle_ionization_integral', result.stdout)
        self.assertIn('BOTH R_J=0 and R_M=0', result.stdout)
        invalid = subprocess.run(command+['--ionization-scale', '2'], capture_output=True, text=True)
        self.assertEqual(invalid.returncode, 2)


if __name__ == '__main__':
    unittest.main()
