"""Independent excitation, normalization, energy-balance and plumbing checks."""

import argparse
import contextlib
import copy
import io
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy.integrate import simpson
from scipy.optimize import brentq

from simple_cx_model.config import load_cx_config
from isothermal_cx_model.excitation import (
    L_exc, E_exc, make_E_exc, add_excitation_arguments,
)
from isothermal_cx_model.geometry import constant_B, quadratic_area_legacy
from isothermal_cx_model.solver_general import solve_at_theta
from isothermal_cx_model.scan_general import find_temperature_solutions
from isothermal_cx_model.scan_neutral import find_density_solutions
from isothermal_cx_model.scan_lambda import scan_lambda_neutral, ROOT_FIELDS, GRID_FIELDS
from isothermal_cx_model.profiles import compute_profiles
from isothermal_cx_model.cx import make_D
from isothermal_cx_model.inlet import make_inlet_momentum


CONFIG = Path(__file__).resolve().parents[1] / "simple_cx_model/cx_config.json"


def analytic_omega(mu, b, beta, D, lam, excitation):
    """Constant area: integrate d(omega)/d(mu) independently of ODE code."""
    theta = b*b/(8*beta)
    mu = np.asarray(mu)
    radical = np.sqrt(np.maximum(mu*mu-b*b, 0))
    integral_radical = .5*(mu*radical-b*b*np.log((mu+radical)/b))
    integral_mu = .5*(mu*mu-b*b)
    return (lam*theta + (5*integral_mu+integral_radical)/(8*beta)
            + excitation*(integral_mu+integral_radical)/(4*beta*theta*D))


class CoefficientTests(unittest.TestCase):
    def test_scalar_array_and_zero(self):
        self.assertIsInstance(L_exc(10), float)
        values = L_exc(np.array([0., 7.438392538197074, 47.703016514101265]))
        np.testing.assert_allclose(values, [0, 9.868919925645329e-20,
                                            1.806361405265463e-19], rtol=1e-12)

    def test_invalid_temperature(self):
        for value in [-1, np.nan, np.inf, [1, -2]]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                L_exc(value)

    def test_cgs_normalization_and_density_override(self):
        p = load_cx_config(CONFIG)
        original = copy.deepcopy(p)
        theta = .02
        expected = (p['L_cm']*p['n0_cm3']*L_exc(theta*p['T_star_eV'])
                    /(p['U_star_cm_per_s']*p['T_star_erg']))
        self.assertEqual(E_exc(theta, p), expected)
        self.assertAlmostEqual(E_exc(theta, p, 2*p['n0_cm3']), 2*expected)
        self.assertEqual(E_exc(0, p), 0)
        self.assertEqual(E_exc(theta, p, 0), 0)
        self.assertEqual(p, original)

    def test_off_and_fractional_scale(self):
        self.assertEqual(make_E_exc({}, scale=0), 0)
        p = load_cx_config(CONFIG)
        self.assertEqual(make_E_exc(p, scale=.25)(.02), .25*E_exc(.02, p))

    def test_bad_parameters_and_scale(self):
        p = load_cx_config(CONFIG)
        for scale in [-1, 1.01, np.nan, np.inf]:
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                make_E_exc(p, scale=scale)
        for key in ['L_cm', 'U_star_cm_per_s', 'T_star_eV', 'T_star_erg']:
            with self.subTest(key=key), self.assertRaises(ValueError):
                E_exc(.1, dict(p, **{key: 0}))
        for theta in [-1, np.inf, np.nan]:
            with self.subTest(theta=theta), self.assertRaises(ValueError):
                E_exc(theta, p)

    def test_parser(self):
        parser = argparse.ArgumentParser()
        add_excitation_arguments(parser)
        self.assertEqual(parser.parse_args([]).excitation_scale, 0)
        self.assertEqual(parser.parse_args(['--excitation-scale', '.5']).excitation_scale, .5)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(['--excitation-scale', 'nan'])


class EquationTests(unittest.TestCase):
    def test_off_is_exact_old_rhs(self):
        default = solve_at_theta(.1, .5, 4, 5, constant_B(), n_points=300)
        disabled = solve_at_theta(.1, .5, 4, 5, constant_B(), n_points=300,
                                  E_exc=lambda theta: 0)
        self.assertEqual(default['alpha'], disabled['alpha'])
        for key in ['mu', 'omega', 'u_over_cs']:
            np.testing.assert_array_equal(default[key], disabled[key])

    def test_constant_area_independent_energy_integral(self):
        theta, beta, D, lam, excitation = .1, .5, 4., 5., .02
        state = solve_at_theta(theta, beta, D, lam, constant_B(),
                               E_exc=excitation, n_points=800,
                               rtol=1e-11, atol=1e-13)
        self.assertEqual(state['status'], 'energy_matched')
        expected_mu = state['b'] + D*(1-state['g'])
        np.testing.assert_allclose(state['mu'], expected_mu, atol=1e-10)
        np.testing.assert_allclose(state['omega'], analytic_omega(
            expected_mu, state['b'], beta, D, lam, excitation), atol=2e-9)
        inlet_mu = brentq(lambda mu: analytic_omega(
            mu, state['b'], beta, D, lam, excitation)-1, state['b'], 3)
        self.assertAlmostEqual(state['alpha'], 1-(inlet_mu-state['b'])/D,
                               delta=2e-9)

    def test_fixed_temperature_front_moves_towards_wall(self):
        states = [solve_at_theta(.1, .5, 4, 5, constant_B(), E_exc=value)
                  for value in [0, .01, .02]]
        self.assertTrue(all(s['status'] == 'energy_matched' for s in states))
        self.assertTrue(np.all(np.diff([s['alpha'] for s in states]) > 0))

    def test_callable_evaluated_once(self):
        calls = []
        state = solve_at_theta(.1, .5, 4, 5, constant_B(),
                               E_exc=lambda t: calls.append(t) or .02)
        self.assertEqual(calls, [.1])
        self.assertEqual(state['E_exc'], .02)

    def test_invalid_coefficient(self):
        for value in [-1, np.nan, np.inf]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                solve_at_theta(.1, .5, 4, 5, constant_B(), E_exc=value)

    def test_zero_cx_not_silently_treated_as_supported(self):
        state = solve_at_theta(.1, .5, 0, 5, constant_B(), E_exc=.02)
        self.assertEqual(state['status'], 'zero_cx_with_excitation_unresolved')
        with self.assertRaises(NotImplementedError):
            find_temperature_solutions(.5, 0, 5, constant_B(), E_exc=.02,
                                       b_points=20)

    def test_root_and_refinement_keep_temperature_dependent_coefficient(self):
        coefficient = lambda theta: .03*theta
        expected_b = brentq(lambda b: analytic_omega(
            1, b, .5, 4, 5, coefficient(b*b/4))-1, .1, .89)
        states, _ = find_temperature_solutions(.5, 4, 5, constant_B(),
            E_exc=coefficient, n_points=100, b_points=25)
        self.assertEqual(len(states), 1)
        self.assertAlmostEqual(states[0]['b'], expected_b, delta=1e-7)
        self.assertEqual(states[0]['E_exc'], coefficient(states[0]['theta']))
        self.assertLess(abs(states[0]['R_M_check']), states[0]['R_M_tolerance'])

    def test_dimensional_energy_balance_and_excitation_integral(self):
        p = load_cx_config(CONFIG)
        geometry = quadratic_area_legacy()
        D = make_D(p, mode='temperature')
        E = make_E_exc(p, scale=1)
        state = solve_at_theta(7.438392538197074/p['T_star_eV'], p['beta'], D,
            p['lambda_sh'], geometry, E_exc=E, n_points=12800,
            inlet_momentum=make_inlet_momentum(p, geometry),
            rtol=1e-11, atol=1e-13)
        self.assertEqual(state['status'], 'energy_matched')  # Off momentum root.
        f = compute_profiles(state, p)
        z, n, u, area = [f[k] for k in ['z_cm', 'n_cm3', 'u_cm_s', 'A_cm2']]
        energy = state['theta']*p['T_star_erg']
        rate = state['D']*p['U_star_cm_per_s']/(p['L_cm']*p['n0_cm3'])
        cx = area*n*p['n0_cm3']*rate*(p['m_D_g']*u*u/2+1.5*energy)
        exc = area*n*p['n0_cm3']*L_exc(f['T_eV'])
        total = simpson(cx+exc, x=z)
        expected = p['W0_erg_per_s']*(1-p['lambda_sh']*state['theta'])
        self.assertAlmostEqual(total/expected, 1, delta=2e-7)
        integral_exc = simpson(exc, x=z)
        alternative = p['J0_per_s']*p['n0_cm3']*L_exc(f['T_eV'])*simpson(1/u, x=z)
        self.assertAlmostEqual(integral_exc/alternative, 1, delta=1e-12)
        H = state['mu']+np.sqrt(np.maximum(state['mu']**2-state['b']**2, 0))
        dimensionless = ((1-state['alpha'])*state['E_exc']
                         * H/(4*p['beta']*state['theta']))
        self.assertAlmostEqual(simpson(dimensionless, x=state['xi']),
                               integral_exc/p['W0_erg_per_s'], delta=1e-12)
        self.assertEqual(f['E_exc'], state['E_exc'])
        legacy = dict(state)
        legacy.pop('omega')
        with self.assertRaisesRegex(ValueError, 'solver_general'):
            compute_profiles(legacy, p)


class IntegrationTests(unittest.TestCase):
    def test_density_override_is_used_by_both_reaction_coefficients(self):
        p = load_cx_config(CONFIG)
        original = copy.deepcopy(p)
        for n0 in [p['n0_cm3'], 2*p['n0_cm3']]:
            with patch('isothermal_cx_model.scan_neutral.find_temperature_solutions',
                       return_value=([], {})) as search:
                _, diagnostic = find_density_solutions(n0, p, excitation_scale=.5,
                                                       cx_mode='temperature')
            args, kwargs = search.call_args
            theta = .02
            self.assertAlmostEqual(args[1](theta), make_D(p, 'temperature', n0)(theta))
            self.assertAlmostEqual(kwargs['E_exc'](theta), .5*E_exc(theta, p, n0))
            self.assertEqual(diagnostic['excitation_scale'], .5)
        self.assertEqual(p, original)

    def test_csv_metadata_and_plot_reconstruction(self):
        p = load_cx_config(CONFIG)
        # Constant-area toy matching isolates the plumbing from real-branch topology.
        p['beta'] = .5
        p['U_star_cm_per_s'] = np.sqrt(p['T_star_erg']/(p['beta']*p['m_D_g']))
        p['M0_dyn'] = p['m_D_g']*p['J0_per_s']*p['U_star_cm_per_s']
        p['n_star_cm3'] = p['J0_per_s']/(p['A0_cm2']*p['U_star_cm_per_s'])
        with contextlib.redirect_stdout(io.StringIO()):
            roots, grid = scan_lambda_neutral([5], [p['n0_cm3']], p,
                geometry=constant_B(), inlet_mode='constant', excitation_scale=.25,
                n_points=120, b_points=25, cx_mode='temperature')
        self.assertGreater(len(roots), 0)
        for row in roots:
            self.assertEqual(row['excitation_scale'], .25)
            self.assertAlmostEqual(row['E_exc'], .25*E_exc(row['theta'], p))
            self.assertFalse(set(row)-set(ROOT_FIELDS))
            state = solve_at_theta(row['theta'], p['beta'], row['D'], 5,
                constant_B(), E_exc=row['E_exc'], rtol=1e-10, atol=1e-12)
            self.assertLess(abs(state['R_M']), row['R_M_tolerance'])
        self.assertEqual(grid[0]['excitation_scale'], .25)
        self.assertFalse(set(grid[0])-set(GRID_FIELDS))

    def test_cli_help_exposes_same_switch_everywhere(self):
        for module in ['scan', 'scan_neutral', 'scan_lambda', 'plot_profiles']:
            with self.subTest(module=module):
                result = subprocess.run([sys.executable, '-B', '-m',
                    'isothermal_cx_model.'+module, '--help'], capture_output=True,
                    text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('--excitation-scale', result.stdout)


if __name__ == '__main__':
    unittest.main()
