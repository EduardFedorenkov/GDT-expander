"""CX plumbing/algebra only. Every IVP and stationary launch is mocked.

No solve_ivp integration, Newton search, physical profile or scan is run.
The independent old source assembler is compared when available locally.
"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.integrate import simpson
from isothermal_model_adas import transport as tr
from isothermal_model_adas import run_stationary as cli
from isothermal_model_adas.config import load_config
from isothermal_model_adas.cx_adas import make_D, ADASCX, CCD_PATH
from isothermal_model_adas.sources import adas_sources
from isothermal_model_adas.geometry import constant_B, quadratic_area_legacy
from isothermal_model_adas.recombination_adas import ADASDomainError


PACKAGE = Path(__file__).parents[1]
OLD_SOURCES = PACKAGE.parent/'isothermal_cx_model'/'sources.py'


class SourceAlgebraTests(unittest.TestCase):
    def test_cx_adds_no_particles(self):
        args = (.8, 2., .2, 3.)
        kwargs = dict(theta=.05, beta=1., R=.01, Q_eff=-.001)
        before = adas_sources(*args, **kwargs)
        after = adas_sources(*args, **kwargs, D=.4)
        self.assertEqual(after[0], before[0])
        self.assertAlmostEqual(after[1]-before[1], -.4*.8)
        self.assertAlmostEqual(after[2]-before[2], -.4*3*2*(.2**2/2+1.5*.05))

    def test_zero_D_preserves_sources_exactly(self):
        args = (.8, 2., .2, 3.)
        kwargs = dict(theta=.05, beta=1., R=.01, Q_eff=-.001)
        self.assertEqual(adas_sources(*args, **kwargs), adas_sources(*args, **kwargs, D=0.))

    @unittest.skipUnless(OLD_SOURCES.exists(), 'Old source assembler not installed beside the new package.')
    def test_matches_old_source_assembler_for_100_states(self):
        spec = importlib.util.spec_from_file_location('legacy_source_comparison', OLD_SOURCES)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        rng = np.random.default_rng(42)
        for _ in range(100):
            beta, theta = 10**rng.uniform(-1, 1), 10**rng.uniform(-3, -1)
            j, A = rng.uniform(.01, 1.), 10**rng.uniform(0, 3)
            v = rng.uniform(.05, .95)*np.sqrt(2*beta*theta)
            mu = j*(v+2*beta*theta/v)
            radical = np.sqrt(mu*mu-8*beta*theta*j*j)
            H = mu+radical
            N = H/(4*beta*theta*A)
            D, R, Q = rng.uniform(.01, 3.), rng.uniform(1e-4, .1), rng.uniform(-.01, .01)
            expected = old.atomic_sources(j, mu, radical, H, A, theta=theta, beta=beta,
                D=D, R=R, Q=Q, E_exc=0., D_ion=0., E_ion=0.).rhs(0.)
            actual = adas_sources(j, N, v, A, theta=theta, beta=beta, D=D, R=R, Q_eff=Q)
            np.testing.assert_allclose(actual, expected, rtol=5e-13, atol=1e-14)

    def test_wall_derivative_matches_direct_rhs(self):
        theta, beta, jt, R, D = .05, 1.2, .8, .01, 2.
        geometry = quadratic_area_legacy()
        b = np.sqrt(8*beta*theta)
        derivative = tr.rhs_g(1., [jt, b*jt, .4], theta=theta, beta=beta,
            recombination=(R, -.001), geometry=geometry, D=D)
        actual, _ = tr.wall_slope_per_j(theta, jt, beta, R, D, geometry)
        self.assertAlmostEqual(actual, (derivative[1]-b*derivative[0])/jt)

    def test_cx_can_change_wall_admissibility(self):
        geometry = quadratic_area_legacy()
        before, _ = tr.wall_slope_per_j(.05, .8, 1., .01, 0., geometry)
        after, _ = tr.wall_slope_per_j(.05, .8, 1., .01, 2., geometry)
        self.assertGreater(before, 0.)
        self.assertLess(after, 0.)
        self.assertAlmostEqual(after-before, -2.)

    def test_independent_budgets_include_cx(self):
        g = np.linspace(.4, 1., 101)
        H = 3-.3*g
        N = H/(4*.05)
        state = dict(g=g, j=1-.1*g, mu=2-.2*g, omega=1-.3*g, H=H,
            A_over_A_star=np.ones_like(g), n_over_n_star=N,
            u_over_u_star=.2*np.ones_like(g), theta=.05, beta=1., geometry=constant_B())
        with patch.object(tr, 'adas_sources', side_effect=AssertionError('Budget must remain independent')):
            before = tr.balance_checks(state, (.01, -.001))
            state['D'] = .4
            after = tr.balance_checks(state, (.01, -.001))
        expected = float(simpson(.4*N*(.2**2/2+1.5*.05), x=g))
        self.assertEqual(after['particle_abs'], before['particle_abs'])
        self.assertAlmostEqual(after['energy_loss_integral']-before['energy_loss_integral'], expected)
        self.assertAlmostEqual(after['cx_energy_loss_integral'], expected)
        self.assertAlmostEqual(after['cx_momentum_loss_integral'], float(simpson(.4*state['j'], x=g)))


class PlumbingTests(unittest.TestCase):
    def test_D_evaluated_once_and_xi_factor_used(self):
        calls, captured = [], {}

        def D(theta):
            calls.append(theta)
            return 2.

        def not_an_integrator(rhs, interval, initial, **kwargs):
            captured['derivative'] = rhs(.8, initial)
            captured['initial'] = initial
            return SimpleNamespace(t=[1.], success=False, message='Mock: no integration performed')

        geometry = quadratic_area_legacy()
        with patch.object(tr, 'solve_ivp', side_effect=not_an_integrator):
            result = tr.solve_trial(.05, .8, .7, 1., 8., (.001, -.01), geometry, D=D)
        self.assertEqual(calls, [.05])
        self.assertEqual(result['D'], 2.)
        self.assertEqual(result['status'], 'integration_failed')
        expected = .3*tr.rhs_g(.7+.3*.8, captured['initial'], theta=.05, beta=1.,
            recombination=(.001, -.01), geometry=geometry, D=2.)
        np.testing.assert_allclose(captured['derivative'], expected, rtol=1e-13)

    def test_bad_D_and_domain_status_before_integration(self):
        with patch.object(tr, 'solve_ivp', side_effect=AssertionError('Must not integrate')):
            for D in (-1., np.nan, np.inf):
                with self.assertRaises(ValueError):
                    tr.solve_trial(.05, .8, .7, 1., 8., (0., 0.), D=D)

            def outside(theta):
                raise ADASDomainError('CCD temperature outside table')

            result = tr.solve_trial(.05, .8, .7, 1., 8., (0., 0.), D=outside)
        self.assertEqual(result['status'], 'adas_outside_domain')
        self.assertIn('CCD', result['reason'])

    def test_stationary_passes_D_to_all_trials_without_solving(self):
        # Inspect the forwarding contract, do not execute Newton/IVP.
        import ast
        tree = ast.parse((PACKAGE/'stationary.py').read_text(encoding='utf-8-sig'))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == 'solve_trial']
        self.assertTrue(calls)
        for call in calls:
            forwarded = [key.value for key in call.keywords if key.arg == 'D']
            self.assertEqual(len(forwarded), 1)
            self.assertIsInstance(forwarded[0], ast.Name)
            self.assertEqual(forwarded[0].id, 'D')

    def test_optional_config_defaults_and_validation(self):
        config = json.loads((PACKAGE/'config.json').read_text())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'config.json'
            config.pop('n0_cm3', None)
            config.pop('k_cx_cm3_per_s', None)
            path.write_text(json.dumps(config))
            self.assertEqual(load_config(path)['n0_cm3'], 0.)
            for value in (-1., float('inf'), 'invalid', None):
                config['n0_cm3'] = value
                path.write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    load_config(path)

    @unittest.skipUnless(CCD_PATH.exists(), 'CCD not distributed with code/tests.')
    def test_real_D_factory_without_integrating(self):
        p = load_config(PACKAGE/'config.json')
        data = ADASCX()
        provider = make_D(p, data=data)
        theta = 5./p['T_star_eV']
        expected = p['L_cm']*p['n0_cm3']/p['U_star_cm_per_s']*data.evaluate(5.)
        self.assertAlmostEqual(provider(theta)/expected, 1., places=13)
        with patch.object(tr, 'solve_ivp', return_value=SimpleNamespace(
                t=[1.], success=False, message='Mock: no integration')):
            trial = tr.solve_trial(theta, .8, .8, p['beta'], p['lambda_sh'], (0., 0.), D=provider)
        self.assertEqual(trial['status'], 'integration_failed')
        self.assertEqual(trial['D'], provider(theta))


class CommandTests(unittest.TestCase):
    def mocked_run(self, extra):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        output = Path(temp.name)/'report'
        outcome = dict(status='mock_not_solved', diagnostic=dict(evaluations=0))
        argv = ['run_stationary', '--config', str(PACKAGE/'config.json'), '--guess',
                '5', '.8', '.8', '--scale', '0', '--output', str(output), *extra]
        with patch.object(sys, 'argv', argv), patch.object(cli, 'solve_stationary', return_value=outcome) as solver:
            with patch.object(cli, 'ADASRecombination', side_effect=AssertionError('Recombination disabled')):
                self.assertEqual(cli.main(), 1)
        return solver, json.loads((output/'report.json').read_text())

    @unittest.skipUnless(CCD_PATH.exists(), 'CCD not distributed with code/tests.')
    def test_cli_defaults_ADAS_and_records_ccd_hash(self):
        solver, report = self.mocked_run([])
        self.assertEqual(report['cx_mode'], 'temperature')
        self.assertEqual(set(report['data_sha256']), {'ccd96_d.dat'})
        self.assertTrue(callable(solver.call_args.kwargs['D']))
        self.assertEqual(report['schema_version'], 2)

    def test_cli_constant_mode_uses_explicit_k_and_no_ccd(self):
        with patch.object(cli, 'ADASCX', side_effect=AssertionError('No CCD in constant mode')):
            solver, report = self.mocked_run(['--cx-mode', 'constant'])
        p = report['parameters']
        expected = p['L_cm']*p['n0_cm3']/p['U_star_cm_per_s']*p['k_cx_cm3_per_s']
        self.assertEqual(solver.call_args.kwargs['D'], expected)
        self.assertEqual(report['data_sha256'], {})
        self.assertIsNone(solver.call_args.kwargs['theta_bounds'])

    def test_cli_zero_density_disables_CCD_loading(self):
        with patch.object(cli, 'ADASCX', side_effect=AssertionError('CX disabled')):
            solver, report = self.mocked_run(['--density', '0'])
        self.assertEqual(solver.call_args.kwargs['D'], 0.)
        self.assertEqual(report['parameters']['n0_cm3'], 0.)
        self.assertEqual(report['data_sha256'], {})
        self.assertIsNone(solver.call_args.kwargs['theta_bounds'])


if __name__ == '__main__':
    unittest.main()
