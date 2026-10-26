"""Verified stationary roots, failures, multistart and variable-J output."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import brentq

from simple_cx_model.config import load_cx_config
from isothermal_cx_model.geometry import constant_B, quadratic_area_legacy
from isothermal_cx_model.cx import make_D
from isothermal_cx_model.excitation import make_E_exc
from isothermal_cx_model.inlet import make_inlet_momentum
from isothermal_cx_model.recombination import make_rr_coefficients
from isothermal_cx_model.stationary_recombination import (
    solve_stationary, find_stationary_solutions,
)
from isothermal_cx_model.profiles_recombination import (
    compute_stationary_profiles, plot_stationary_profiles,
)


CONFIG = Path(__file__).resolve().parents[1]/'simple_cx_model/cx_config.json'


def cold_problem():
    p = load_cx_config(CONFIG)
    p.update(n0_cm3=1e15, lambda_sh=7.)
    g = quadratic_area_legacy()
    options = dict(E_exc=make_E_exc(p, scale=1), rr=make_rr_coefficients(p, scale=1),
                   inlet_momentum=make_inlet_momentum(p, g), n_points=2000)
    return p, g, make_D(p, 'temperature'), options


def fake_trial(theta, jt, *args, **kwargs):
    # Two separate roots with the SAME monotone label, no physical oracle intended.
    x = np.log(theta)
    fJ = np.log(jt/.7)
    fM = (x-np.log(.1))*(x-np.log(.2))
    ja, mua = np.exp(fJ), np.exp(fM)
    return dict(status='energy_matched', theta=float(theta), j_target=float(jt),
                alpha=.6+theta*.1, j_a=float(ja), mu_a=float(mua), mu_target=1.,
                R_J=float(ja-1), R_M=float(mua-1), R_E=0.,
                inlet_target_sonic_margin=1., branch='monotone')


class NumericalRootTests(unittest.TestCase):
    def test_cold_reference_and_refined_balances(self):
        p, g, D, options = cold_problem()
        result = solve_stationary(.13/p['T_star_eV'], .8, p['beta'], D, 7, g, **options)
        self.assertEqual(result['status'], 'converged')
        s = result['solution']
        self.assertTrue(s['stationary_verified'])
        self.assertAlmostEqual(s['theta']*p['T_star_eV'], .11919492896752329, delta=2e-8)
        self.assertAlmostEqual(s['j_target'], .7755613756347011, delta=2e-8)
        self.assertAlmostEqual(s['alpha']*p['L_cm'], 164.99343626332248, delta=2e-6)
        self.assertLess(max(abs(s['R_J']), abs(s['R_M'])), 1e-9)
        self.assertLess(max(s['balances'][k] for k in ('particle_abs', 'momentum_abs', 'energy_abs')), 5e-6)
        self.assertIn('verification', {h['phase'] for h in result['diagnostics']['history']})
        self.assertLess(s['verification']['rtol'], 1e-9)
        self.assertGreater(s['inlet_target_sonic_margin'], 0)

    def test_zero_rr_boundary_j_equal_one_and_analytic_oracle(self):
        beta, D, lam = .5, 4., 5.
        def residual(b):
            radical = np.sqrt(1-b*b)
            integral = 2.5*(1-b*b)+.5*(radical-b*b*np.log((1+radical)/b))
            return lam*b*b/(8*beta)+integral/(8*beta)-1
        expected_b = brentq(residual, .1, .89)
        result = solve_stationary(.1, 1, beta, D, lam, constant_B(), n_points=1500)
        self.assertEqual(result['status'], 'converged')
        state = result['solution']
        self.assertAlmostEqual(state['theta'], expected_b**2/(8*beta), delta=1e-8)
        self.assertEqual(state['j_target'], 1.)
        self.assertEqual(state['R_J'], 0.)

    def test_real_warm_root_resolves_particle_loss_near_one(self):
        p, g, D, options = cold_problem()
        result = solve_stationary(15/p['T_star_eV'], .99999997,
                                  p['beta'], D, 7, g, **options)
        self.assertEqual(result['status'], 'converged')
        s = result['solution']
        self.assertAlmostEqual(s['theta']*p['T_star_eV'], 14.93634817097698, delta=2e-6)
        self.assertGreater(1-s['j_target'], 0)
        self.assertAlmostEqual(1-s['j_target'], 3.391836955746186e-8, delta=2e-11)

    def test_real_multistart_keeps_three_verified_roots(self):
        p, g, D, options = cold_problem()
        guesses = np.array([[.13, .8], [15, .99999997], [80, .9999999999]])
        guesses[:, 0] /= p['T_star_eV']
        states, _ = find_stationary_solutions(guesses, p['beta'], D, 7, g, **options)
        self.assertEqual(len(states), 3)
        np.testing.assert_allclose([s['theta']*p['T_star_eV'] for s in states],
            [.11919492896752329, 14.93634817097698, 77.82184032733745], rtol=1e-6)
        self.assertTrue(all(s['stationary_verified'] for s in states))

    def test_invalid_guess_and_search_bounds(self):
        result = solve_stationary(.3, .8, .5, 4, 5, constant_B())
        self.assertEqual(result['status'], 'invalid_initial_trial')
        self.assertNotIn('solution', result)
        self.assertIn('energy_temperature_limit', result['diagnostics']['invalid_trials'])
        result = solve_stationary(.1, .5, .5, 4, 5, theta_bounds=(.2, .4))
        self.assertEqual(result['status'], 'initial_outside_search_bounds')

    def test_input_validation(self):
        for kwargs in [dict(j_min=0), dict(j_min=1), dict(residual_tol=0),
                       dict(balance_tol=np.nan), dict(max_iterations=0),
                       dict(n_points=20), dict(theta_bounds=(.2, .1))]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                solve_stationary(.1, .8, .5, 4, 5, **kwargs)

    def test_iteration_limit_is_not_root(self):
        result = solve_stationary(.08, .8, .5, 4, 5, constant_B(), max_iterations=1)
        self.assertEqual(result['status'], 'iteration_limit')
        self.assertNotIn('solution', result)

    def test_new_solver_does_not_silence_programming_errors(self):
        with patch('isothermal_cx_model.stationary_recombination.solve_trial',
                   side_effect=ValueError('bad coefficient')):
            with self.assertRaisesRegex(ValueError, 'bad coefficient'):
                solve_stationary(.1, .8, .5, 4, 5)


class VerificationAndMultistartTests(unittest.TestCase):
    def oracle(self):
        return patch('isothermal_cx_model.stationary_recombination.solve_trial', fake_trial)

    def balances(self, **overrides):
        values = dict(particle_abs=0., momentum_abs=0., energy_abs=0., particle_loss_integral=.3)
        values.update(overrides)
        return patch('isothermal_cx_model.stationary_recombination.balance_checks', return_value=values)

    def test_two_roots_same_shape_and_duplicate_seed(self):
        with self.oracle(), self.balances():
            states, diagnostic = find_stationary_solutions(
                [[.095, .68], [.205, .72], [.102, .7]], .5, 4, 5, constant_B())
        self.assertEqual(len(states), 2)
        np.testing.assert_allclose([s['theta'] for s in states], [.1, .2], atol=1e-8)
        self.assertEqual([s['solution_id'] for s in states], ['root_001', 'root_002'])
        self.assertEqual(diagnostic['duplicate_convergences'], 1)
        json.dumps(diagnostic, allow_nan=False)

    def test_bad_integral_cannot_be_accepted_as_stationary_solution(self):
        with self.oracle(), self.balances(energy_abs=.1):
            result = solve_stationary(.1, .7, .5, 4, 5)
        self.assertEqual(result['status'], 'balance_check_failed')
        self.assertNotIn('solution', result)

    def test_verification_failure_is_visible(self):
        def changes_with_accuracy(theta, jt, *args, **kwargs):
            if kwargs['rtol'] < 1e-10:
                return dict(status='integration_failed')
            return fake_trial(theta, jt)
        with patch('isothermal_cx_model.stationary_recombination.solve_trial',
                   changes_with_accuracy):
            result = solve_stationary(.1, .7, .5, 4, 5)
        self.assertEqual(result['status'], 'verification_invalid_initial_trial')
        self.assertNotIn('solution', result)

    def test_flat_jacobian_and_false_success_are_rejected(self):
        def flat(theta, jt, *args, **kwargs):
            state = fake_trial(.11, .8)
            state.update(theta=theta, j_target=jt)
            return state
        with patch('isothermal_cx_model.stationary_recombination.solve_trial', flat):
            result = solve_stationary(.11, .8, .5, 4, 5)
        self.assertEqual(result['status'], 'ill_conditioned_jacobian')
        self.assertNotIn('solution', result)
        json.dumps(result, allow_nan=False)

    def test_no_initial_guess_set_claims_no_global_completeness(self):
        with patch('isothermal_cx_model.stationary_recombination.solve_trial',
                   return_value=dict(status='sonic_return')):
            states, diagnostic = find_stationary_solutions([[.1, .7]], .5, 4, 5)
        self.assertEqual(states, [])
        self.assertIn('not a proof', diagnostic['warning'])


class ProfileAndCLITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p, geometry, D, options = cold_problem()
        cls.state = solve_stationary(.13/cls.p['T_star_eV'], .8, cls.p['beta'], D,
            7, geometry, **options)['solution']
        cls.state['solution_id'] = 'root_001'

    def test_variable_particle_and_energy_flux_identities(self):
        f = compute_stationary_profiles(self.state, self.p)
        np.testing.assert_allclose(f['A_cm2']*f['n_cm3']*f['u_cm_s'], f['J_per_s'], rtol=1e-12)
        thermal = self.state['theta']*self.p['T_star_erg']
        recovered = f['J_per_s']*(.5*self.p['m_D_g']*f['u_cm_s']**2+5*thermal)
        recovered += f['A_cm2']*f['q_e_erg_cm2_s']
        np.testing.assert_allclose(recovered, f['W_erg_per_s'], rtol=1e-12)
        self.assertAlmostEqual(f['J_per_s'][-1]/self.p['J0_per_s'], .7755613756347011, delta=2e-8)

    def test_unverified_and_mismatched_source_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'verified'):
            compute_stationary_profiles(dict(self.state, stationary_verified=False), self.p)
        with self.assertRaisesRegex(ValueError, 'normalization'):
            compute_stationary_profiles(self.state, dict(self.p, n_star_cm3=self.p['n_star_cm3']*2))

    def test_five_profile_panels(self):
        import matplotlib
        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt
        figure, axes = plot_stationary_profiles([compute_stationary_profiles(self.state, self.p)])
        figure.canvas.draw()
        self.assertEqual(len(axes), 5)
        plt.close(figure)

    def test_cli_from_config_and_safe_output(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'results'
            args = [sys.executable, '-B', '-m', 'isothermal_cx_model.run_recombination',
                '--config', str(CONFIG), '--density', '1e15', '--lambda-sh', '7',
                '--guess', '.13', '.8', '--points', '1500', '--no-plot', '--output', str(output)]
            process = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            report = json.loads((output/'solutions.json').read_text(encoding='utf-8'))
            self.assertEqual(len(report['solutions']), 1)
            self.assertAlmostEqual(report['solutions'][0]['T_eV'], .11919492896752329, delta=2e-8)
            self.assertEqual(report['guesses_T_eV_j_target'], [[.13, .8]])
            with np.load(output/'root_001_profiles.npz', allow_pickle=False) as arrays:
                self.assertIn('J_per_s', arrays)
                self.assertIn('q_e_erg_cm2_s', arrays)
            again = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(again.returncode, 2)
            self.assertIn('preserve old results', again.stderr)

    def test_cli_help(self):
        result = subprocess.run([sys.executable, '-B', '-m',
            'isothermal_cx_model.run_recombination', '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--guess', result.stdout)


if __name__ == '__main__':
    unittest.main()
