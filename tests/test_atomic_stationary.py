"""Stage 2: actual stationary correction with ionization, not just IVPs."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from simple_cx_model.config import load_cx_config
from isothermal_cx_model.geometry import constant_B, quadratic_area_legacy
from isothermal_cx_model.cx import make_D
from isothermal_cx_model.excitation import make_E_exc
from isothermal_cx_model.ionization import make_ionization_coefficients
from isothermal_cx_model.recombination import make_rr_coefficients
from isothermal_cx_model.inlet import make_inlet_momentum
from isothermal_cx_model.solver_atomic import solve_trial
from isothermal_cx_model.stationary_recombination import (
    solve_stationary, find_stationary_solutions,
)
from isothermal_cx_model.profiles_recombination import (
    compute_stationary_profiles, plot_stationary_profiles,
)


CONFIG = Path(__file__).resolve().parents[1]/'simple_cx_model/cx_config.json'


def analytic_roots():
    # A=const, only ionization, mu_a=1, j_a=omega_a=1. This derivation
    # uses conservation laws, not the shooting routine under test.
    beta, lam, epsilon, Di = .5, 5., .2, 1.
    roots = []
    for b in np.sort(np.roots([lam/(8*beta), -(1+epsilon), epsilon])):
        F = lambda y: np.arcsin(y)-y/(1+np.sqrt(max(1-y*y, 0)))
        roots.append((b*b/(8*beta), 1/b, 1-b/(2*Di)*(F(1)-F(b))))
    return roots


def pure_ionization(theta, jt, **kwargs):
    options = dict(ionization=(1., .2), n_points=4000)
    options.update(kwargs)
    return solve_stationary(theta, jt, .5, 0., 5., constant_B(), **options)


class AtomicStationaryTests(unittest.TestCase):
    def test_both_independent_analytic_roots_and_all_boundary_conditions(self):
        for seed, expected in zip([(.012, 4.5), (.135, 1.35)], analytic_roots()):
            with self.subTest(seed=seed):
                outcome = pure_ionization(*seed)
                self.assertEqual(outcome['status'], 'converged')
                s = outcome['solution']
                np.testing.assert_allclose([s['theta'], s['j_target'], s['alpha']],
                                           expected, rtol=0, atol=1e-9)
                self.assertTrue(s['stationary_verified'])
                self.assertGreater(s['j_target'], 1)
                self.assertLess(max(abs(s[k]) for k in ('R_J', 'R_M', 'R_E')), 1e-9)
                np.testing.assert_allclose(s['mu'], 1, atol=1e-10, rtol=0)
                np.testing.assert_allclose(s['omega']+.2*s['j'], 1.2, atol=1e-10, rtol=0)
                self.assertAlmostEqual(s['mu'][-1], s['b']*s['j_target'], delta=1e-12)
                self.assertAlmostEqual(s['omega'][-1], 5*s['theta']*s['j_target'], delta=1e-12)
                self.assertGreater(s['inlet_target_sonic_margin'], 0)
                self.assertAlmostEqual(s['balances']['particle_ionization_integral'],
                                       s['j_target']-1, delta=2e-6)
                self.assertEqual(s['balances']['particle_recombination_integral'], 0)
                self.assertLess(max(s['balances'][k] for k in
                    ('particle_abs', 'momentum_abs', 'energy_abs')), 5e-6)

    def test_newton_can_cross_j_one_and_use_window_above_one(self):
        r = pure_ionization(.18, .99)
        self.assertEqual(r['status'], 'converged')
        self.assertGreater(r['solution']['j_target'], 1)
        r = pure_ionization(.135, 1.35, j_min=1.1, j_max=2)
        self.assertEqual(r['status'], 'converged')

    def test_multistart_above_one_keeps_both_roots_and_deduplicates(self):
        states, diagnostic = find_stationary_solutions(
            [[.012, 4.5], [.135, 1.35], [.18, .99]], .5, 0, 5, constant_B(),
            ionization=(1., .2), n_points=4000)
        self.assertEqual(len(states), 2)
        np.testing.assert_allclose([s['theta'] for s in states],
                                   [r[0] for r in analytic_roots()], atol=1e-9)
        self.assertEqual(diagnostic['duplicate_convergences'], 1)
        json.dumps(diagnostic, allow_nan=False)

    def test_coefficients_forwarded_to_every_trial_including_fresh_verification(self):
        ionization = lambda theta: (1., .2)
        with patch('isothermal_cx_model.stationary_recombination.solve_trial',
                   wraps=solve_trial) as calls:
            outcome = pure_ionization(.135, 1.35, ionization=ionization)
        self.assertEqual(outcome['status'], 'converged')
        self.assertGreater(calls.call_count, 3)
        self.assertTrue(all(call.kwargs['ionization'] is ionization for call in calls.call_args_list))
        last = calls.call_args_list[-1]
        self.assertEqual(last.kwargs['n_points'], 4000)
        self.assertLess(last.kwargs['rtol'], 1e-9)
        self.assertIn('verification', {h['phase'] for h in outcome['diagnostics']['history']})

    def test_invalid_windows_and_outside_seed_are_not_physical_nonexistence(self):
        for options in [dict(j_min=np.nan), dict(j_min=np.inf), dict(j_min=0),
                        dict(j_max=np.nan), dict(j_max=np.inf), dict(j_max=0),
                        dict(j_min=2, j_max=2)]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                pure_ionization(.135, 1.35, **options)
        outcome = pure_ionization(.012, 4.5, j_max=2)
        self.assertEqual(outcome['status'], 'initial_outside_search_bounds')
        self.assertNotIn('solution', outcome)
        self.assertEqual(outcome['diagnostics']['search_bounds']['j_max'], 2)

    def test_zero_ionization_is_exact_backward_compatibility(self):
        args = (.1, 1., .5, 4., 5., constant_B())
        previous = solve_stationary(*args, n_points=1500)
        zero = solve_stationary(*args, n_points=1500, ionization=(0., 0.))
        self.assertEqual(previous['status'], 'converged')
        self.assertEqual(zero['status'], 'converged')
        for key in ('theta', 'j_target', 'alpha', 'j', 'mu', 'omega'):
            np.testing.assert_array_equal(previous['solution'][key], zero['solution'][key])

    def test_invalid_ivp_does_not_become_penalty_root(self):
        r = pure_ionization(.1, .5)
        self.assertEqual(r['status'], 'invalid_initial_trial')
        self.assertNotIn('solution', r)
        self.assertIn('nonpositive_particle_flux', r['diagnostics']['invalid_trials'])
        r = pure_ionization(.15, 2.)
        self.assertEqual(r['status'], 'invalid_initial_trial')
        self.assertIn('energy_temperature_limit', r['diagnostics']['invalid_trials'])

    def test_failed_refinement_with_ionization_is_not_accepted(self):
        def fail_refinement(*args, **kwargs):
            if kwargs['rtol'] < 1e-10:
                return dict(status='integration_failed')
            return solve_trial(*args, **kwargs)
        with patch('isothermal_cx_model.stationary_recombination.solve_trial', fail_refinement):
            r = pure_ionization(.135, 1.35)
        self.assertEqual(r['status'], 'verification_invalid_initial_trial')
        self.assertNotIn('solution', r)


class PhysicalAtomicAndCLITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = load_cx_config(CONFIG)
        cls.p.update(n0_cm3=1e15, lambda_sh=7.)
        g = quadratic_area_legacy()
        p = cls.p
        cls.outcome = solve_stationary(20/p['T_star_eV'], 8., p['beta'],
            make_D(p, 'temperature'), 7, g, E_exc=make_E_exc(p, scale=1),
            rr=make_rr_coefficients(p, scale=1),
            ionization=make_ionization_coefficients(p, scale=1),
            inlet_momentum=make_inlet_momentum(p, g), n_points=8000)

    def test_all_physical_coefficient_factories_and_separate_particle_counts(self):
        self.assertEqual(self.outcome['status'], 'converged')
        s = self.outcome['solution']
        self.assertAlmostEqual(s['theta']*self.p['T_star_eV'], 96.1704570832, delta=2e-6)
        self.assertAlmostEqual(s['j_target'], 1.60179494169, delta=2e-7)
        self.assertAlmostEqual(s['alpha']*self.p['L_cm'], 184.422155247, delta=2e-6)
        for key in ('D', 'E_exc', 'R_rr', 'Q_rr', 'D_ion', 'E_ion'):
            self.assertGreater(s[key], 0)
        balances = s['balances']
        self.assertGreater(balances['particle_recombination_integral'], 0)
        self.assertGreater(balances['particle_ionization_integral'], .1)
        self.assertAlmostEqual(s['j_target']-1,
            balances['particle_ionization_integral']-balances['particle_recombination_integral'],
            delta=5e-6)
        self.assertAlmostEqual(balances['energy_ionization_integral'],
            (13.6/self.p['T_star_eV'])*balances['particle_ionization_integral'], delta=1e-12)

    def test_dimensional_identities_and_plot_does_not_clip_j_above_one(self):
        profile = compute_stationary_profiles(self.outcome['solution'], self.p)
        np.testing.assert_allclose(profile['A_cm2']*profile['n_cm3']*profile['u_cm_s'],
                                   profile['J_per_s'], rtol=1e-12)
        s = self.outcome['solution']
        thermal = s['theta']*self.p['T_star_erg']
        reconstructed = (profile['J_per_s']*(.5*self.p['m_D_g']*profile['u_cm_s']**2+5*thermal)
                         +profile['A_cm2']*profile['q_e_erg_cm2_s'])
        np.testing.assert_allclose(reconstructed, profile['W_erg_per_s'], rtol=1e-12)
        import matplotlib
        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt
        figure, axes = plot_stationary_profiles([profile])
        figure.canvas.draw()
        self.assertGreater(axes[0].get_ylim()[1], np.max(profile['j']))
        plt.close(figure)

    def test_cli_ionization_metadata_profiles_and_safe_output(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'atomic'
            command = [sys.executable, '-B', '-m', 'isothermal_cx_model.run_recombination',
                '--config', str(CONFIG), '--density', '1e15', '--lambda-sh', '7',
                '--ionization-scale', '1', '--guess', '20', '8',
                '--points', '4000', '--j-max', '10', '--no-show', '--output', str(output)]
            process = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stdout+process.stderr)
            report = json.loads((output/'solutions.json').read_text(encoding='utf-8'))
            self.assertEqual(report['schema_version'], 2)
            self.assertEqual(report['ionization_scale'], 1)
            self.assertEqual(report['j_max'], 10)
            root = report['solutions'][0]
            self.assertNotIn('recombined_fraction', root)
            self.assertLess(root['net_particle_loss_fraction'], 0)
            self.assertGreater(root['recombination_count_per_inlet'], 0)
            self.assertGreater(root['ionization_count_per_inlet'], .1)
            self.assertGreater(root['coefficients']['D_ion'], 0)
            with np.load(output/'root_001_profiles.npz', allow_pickle=False) as profile:
                self.assertAlmostEqual(float(profile['ionization_scale']), 1)
                self.assertGreater(float(profile['D_ion']), 0)
                self.assertGreater(np.max(profile['j']), 1.1)
            self.assertGreater((output/'profiles.png').stat().st_size, 1000)
            again = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(again.returncode, 2)
            self.assertIn('preserve old results', again.stderr)

    def test_cli_off_by_default_and_explicit_bad_scale(self):
        help_result = subprocess.run([sys.executable, '-B', '-m',
            'isothermal_cx_model.run_recombination', '--help'], capture_output=True, text=True)
        self.assertEqual(help_result.returncode, 0)
        self.assertIn('--ionization-scale', help_result.stdout)
        self.assertIn('--j-max', help_result.stdout)
        result = subprocess.run([sys.executable, '-B', '-m',
            'isothermal_cx_model.run_recombination', '--guess', '1', '1.2',
            '--ionization-scale', '2'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)


if __name__ == '__main__':
    unittest.main()
