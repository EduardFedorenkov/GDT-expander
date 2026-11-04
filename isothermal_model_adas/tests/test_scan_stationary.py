"""Mocked end-to-end scan and multistart tests, never a physical scan."""

from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from isothermal_model_adas import atomic_coefficients as atomic
from isothermal_model_adas import scan_stationary as cli
from isothermal_model_adas import stationary
from isothermal_model_adas import residual_map
from isothermal_model_adas.config import load_config
from isothermal_model_adas.cx_adas import ADASCX
from isothermal_model_adas.geometry import constant_B
from isothermal_model_adas.tests.test_residual_map import manufactured_trial


PACKAGE = Path(__file__).parents[1]


def fabricated_solution(theta=.02, jt=1.2, alpha=.6):
    """Self-consistent export shapes, NOT a solution of the plasma equations."""
    p = load_config(PACKAGE/'config.json')
    xi = np.linspace(0., 1., 101)
    g = alpha+(1-alpha)*xi
    return dict(theta=theta, j_target=jt, alpha=alpha, beta=p['beta'],
        stationary_verified=True, j=1+(jt-1)*xi, mu=np.ones_like(xi),
        omega=np.ones_like(xi), xi=xi, g=g, A_over_A_star=np.ones_like(xi),
        Aq_e_over_W0=np.zeros_like(xi), u_over_u_star=np.ones_like(xi),
        u_over_cs=np.ones_like(xi), n_over_n_star=np.ones_like(xi),
        R_profile=np.full_like(xi, .1), Q_eff_profile=np.full_like(xi, -.01),
        D_ion_profile=np.full_like(xi, .2), E_ion_profile=np.full_like(xi, .3),
        E_exc_profile=np.full_like(xi, .4), D=.5, mu_target=1., R_J=0., R_M=0., R_E=0.,
        verification=dict(residual_tol=1e-9, balance_tol=5e-6),
        balances=dict(particle_abs=0., momentum_abs=0., energy_abs=0.,
            particle_recombination_integral=.1, particle_ionization_integral=.3))


def converged(state):
    return dict(status='converged', solution=state, diagnostic=dict(
        last_parameters={k: state[k] for k in ('theta', 'j_target', 'alpha')}, history=[]))


class MultistartTests(unittest.TestCase):
    def test_alpha_is_not_ignored_and_duplicates_are_recorded(self):
        states = [fabricated_solution(alpha=.4), fabricated_solution(alpha=.8),
                  fabricated_solution(alpha=.4+1e-10)]
        progress = []
        with patch.object(stationary, 'solve_stationary', side_effect=[converged(s) for s in states]):
            roots, diagnostic = stationary.find_stationary_solutions(
                [(s['theta'], s['j_target'], s['alpha']) for s in states], 1., 5., (0., 0.),
                constant_B(), progress=lambda *data: progress.append(data))
        self.assertEqual(len(roots), 2)
        self.assertEqual([r['solution_id'] for r in roots], ['root_001', 'root_002'])
        self.assertEqual(diagnostic['duplicate_convergences'], 1)
        self.assertEqual([a['converged_to'] for a in diagnostic['attempts']],
                         ['root_001', 'root_002', 'root_001'])
        self.assertEqual(len(progress), 3)
        self.assertIsNone(progress[-1][-1])

    def test_failures_are_attempts_not_roots(self):
        outcome = dict(status='invalid_initial_trial', diagnostic={'invalid_trials': {'adas_outside_domain': 1}})
        with patch.object(stationary, 'solve_stationary', return_value=outcome):
            roots, diagnostic = stationary.find_stationary_solutions([(.1, 1., .5)], 1., 5., (0., 0.))
        self.assertEqual(roots, [])
        self.assertEqual(diagnostic['attempts'][0]['status'], 'invalid_initial_trial')

    def test_unverified_convergence_is_an_error(self):
        state = fabricated_solution()
        state['stationary_verified'] = False
        with patch.object(stationary, 'solve_stationary', return_value=converged(state)):
            with self.assertRaises(RuntimeError):
                stationary.find_stationary_solutions([(.1, 1., .5)], 1., 5., (0., 0.))

    def test_setup_errors_are_not_silently_zero_roots(self):
        with patch.object(stationary, 'solve_stationary', side_effect=ValueError('bad table')):
            with self.assertRaisesRegex(ValueError, 'bad table'):
                stationary.find_stationary_solutions([(.1, 1., .5)], 1., 5., (0., 0.))
        for guesses in ([(0., 1., .5)], [(1., 1., 1.)], [(1., 1.)]):
            with self.assertRaises(ValueError):
                stationary.find_stationary_solutions(guesses, 1., 5., (0., 0.))

    def test_empty_start_list_does_not_launch_solver(self):
        with patch.object(stationary, 'solve_stationary') as solver:
            roots, diagnostic = stationary.find_stationary_solutions([], 1., 5., (0., 0.))
        solver.assert_not_called()
        self.assertEqual(roots, [])
        self.assertEqual(diagnostic['n_solutions'], 0)

    def test_real_corrector_on_manufactured_residuals_without_ivps(self):
        def trial(theta, jt, alpha, *args, **kwargs):
            state = manufactured_trial(theta, jt, alpha, *args, **kwargs)
            state.update(theta=theta, j_target=jt, alpha=alpha,
                R_J=state['j'][0]-1, R_M=state['mu'][0]-state['mu_target'],
                R_E=state['omega'][0]-1)
            return state
        with patch.object(stationary, 'solve_trial', side_effect=trial), patch.object(
                stationary, 'balance_checks', return_value=dict(particle_abs=0., momentum_abs=0., energy_abs=0.)):
            roots, _ = stationary.find_stationary_solutions([(.15, .9, .4), (.3, 1.5, .7)],
                1., 5., (0., 0.), constant_B(), theta_bounds=(.05, .5), n_points=100)
        self.assertEqual(len(roots), 1)
        np.testing.assert_allclose([roots[0][k] for k in ('theta', 'j_target', 'alpha')],
                                   [.2, 1.2, .5], rtol=1e-8, atol=1e-10)


class ScanCLITests(unittest.TestCase):
    def arguments(self, output, *extra):
        return ['--config', str(PACKAGE/'config.json'), '--output', str(output),
                '--temperature-count', '2', '--j-count', '2', '--alpha-count', '2',
                '--points', '100', '--no-show', '--no-plot', *extra]

    def execute(self, output, *extra):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return cli.main(self.arguments(output, *extra))

    def test_dry_run_reads_active_data_but_creates_nothing_and_calls_no_ivp(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(cli, 'scan_residuals') as scan:
            output = Path(folder)/'dry'
            self.assertEqual(self.execute(output, '--dry-run'), 0)
            self.assertFalse(output.exists())
            scan.assert_not_called()

    def test_setup_tracks_all_active_data_and_disabled_channels_need_no_tables(self):
        with tempfile.TemporaryDirectory() as folder:
            args = cli.build_parser().parse_args(self.arguments(Path(folder)/'out', '--atomic-model', 'full-adas'))
            prepared = cli._prepare(args)
            self.assertEqual(set(prepared[5]['data_sha256']),
                {'acd12_h.dat', 'prb12_h.dat', 'ccd96_d.dat', 'scd12_h.dat', 'plt12_h.dat'})
            self.assertTrue(callable(prepared[3]['D']))
            args = cli.build_parser().parse_args(self.arguments(Path(folder)/'off',
                '--rr-scale', '0', '--density', '0', '--temperature-min', '.001', '--atomic-model', 'full-adas'))
            with patch.object(atomic, 'ADASRecombination', side_effect=AssertionError('loaded disabled table')), \
                 patch.object(atomic, 'ADASCX', side_effect=AssertionError('loaded disabled table')), \
                 patch.object(atomic, 'ADASIonization', side_effect=AssertionError('loaded disabled table')), \
                 patch.object(atomic, 'ADASExcitation', side_effect=AssertionError('loaded disabled table')):
                prepared = cli._prepare(args)
            self.assertEqual(prepared[5]['data_sha256'], {})
            self.assertEqual(prepared[3]['D'], 0.)

    def test_density_override_uses_CCD_not_a_constant(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(atomic, 'ADASCX', wraps=ADASCX) as reader:
            args = cli.build_parser().parse_args(self.arguments(Path(folder)/'out',
                '--density', '2e13', '--lambda-sh', '5', '--atomic-model', 'full-adas'))
            p, _, _, processes, _, provenance, *_ = cli._prepare(args)
        reader.assert_called_once()
        self.assertIn('ccd96_d.dat', provenance['data_sha256'])
        self.assertEqual(p['lambda_sh'], 5.)
        self.assertTrue(callable(processes['D']))
        for temperature in (1., 5.):
            expected = p['L_cm']*2e13/p['U_star_cm_per_s']*ADASCX().evaluate(temperature)
            self.assertAlmostEqual(processes['D'](temperature/p['T_star_eV'])/expected, 1., places=13)

    def test_obsolete_cx_option_rejected_even_for_dry_run(self):
        with tempfile.TemporaryDirectory() as folder:
            for mode in ('constant', 'temperature'):
                with self.subTest(mode=mode), self.assertRaises(SystemExit) as error:
                    self.execute(Path(folder)/'unused', '--dry-run', '--cx-mode', mode)
                self.assertEqual(error.exception.code, 2)
                self.assertFalse((Path(folder)/'unused').exists())

    def test_geometry_file_and_flow_source_auto_inlet(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            geometry = root/'geometry.json'
            geometry.write_text(json.dumps({'type': 'constant_B'}), encoding='utf-8')
            config = root/'flows.json'
            config.write_text(json.dumps(dict(A_m_cm2=2., L_cm=10., lambda_sh=5.,
                J0_per_s=1e20, M0_dyn=2e4, W0_erg_per_s=1e11)), encoding='utf-8')
            args = cli.build_parser().parse_args(self.arguments(root/'out',
                '--config', str(config), '--geometry', str(geometry)))
            prepared = cli._prepare(args)
            self.assertEqual(prepared[1].name, 'constant_B')
            self.assertEqual(prepared[4], 'constant')
            args.inlet_momentum = 'kinetic'
            with self.assertRaisesRegex(ValueError, 'central-source'):
                cli._prepare(args)

    def test_solution_outside_map_j_and_alpha_range_is_labelled_not_discarded(self):
        with tempfile.TemporaryDirectory() as folder:
            args = cli.build_parser().parse_args(self.arguments(Path(folder)/'out'))
            state = fabricated_solution(jt=20., alpha=.999)
            state['solution_id'] = 'root_001'
            record = cli._solution_record(state, load_config(PACKAGE/'config.json'), args)
            self.assertTrue(record['outside_scan_j_range'])
            self.assertTrue(record['outside_scan_alpha_range'])

    def test_invalid_inputs_rejected_before_map_and_without_output(self):
        errors = [('--temperature-min', '.001'), ('--alpha-min', '0'), ('--points', '10'),
                  ('--alpha-slice', '2'), ('--guess', '5', '1', '0'),
                  ('--density', '-1'), ('--balance-tol', 'nan'), ('--alpha-count', '1')]
        with tempfile.TemporaryDirectory() as folder, patch.object(cli, 'scan_residuals') as scan:
            for index, extra in enumerate(errors):
                output = Path(folder)/str(index)
                with self.subTest(extra=extra), self.assertRaises(SystemExit) as error:
                    self.execute(output, *extra)
                self.assertEqual(error.exception.code, 2)
                self.assertFalse(output.exists())
        scan.assert_not_called()

    def test_existing_output_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            marker = output/'keep.txt'
            marker.write_text('user data', encoding='utf-8')
            with self.assertRaises(SystemExit):
                self.execute(output)
            self.assertEqual(marker.read_text(encoding='utf-8'), 'user data')

    def test_map_only_checkpoints_full_cube_and_skips_corrector(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(residual_map, 'solve_trial',
                side_effect=manufactured_trial), patch.object(stationary, 'solve_stationary') as solver:
            output = Path(folder)/'map'
            self.assertEqual(self.execute(output, '--map-only'), 0)
            report = json.loads((output/'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(report['stage'], 'map_only')
            self.assertEqual(report['summary']['n_trials'], 8)
            with np.load(output/'residuals.npz', allow_pickle=False) as cube:
                self.assertEqual(cube['F_E'].shape, (2, 2, 2))
                metadata = json.loads(str(cube['metadata_json']))
                self.assertEqual(metadata['model_options'], report['model_options'])
            self.assertEqual(report['kind'], 'stationary_scan_3d')
            self.assertEqual(report['format_version'], 2)
            self.assertEqual(set(report['model_options']),
                {'atomic_model', 'rr_scale', 'ionization_scale', 'excitation_scale', 'inlet_momentum'})
            self.assertEqual(report['solutions'], [])
            self.assertTrue((output/'source_config.json').exists())
            self.assertTrue((output/'geometry.json').exists())
            solver.assert_not_called()

    def test_automatic_and_manual_starts_export_one_verified_root(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(residual_map, 'solve_trial',
                side_effect=manufactured_trial):
            output = Path(folder)/'roots'

            def solve(*args, **kwargs):
                checkpoint = json.loads((output/'summary.json').read_text(encoding='utf-8'))
                self.assertEqual(checkpoint['stage'], 'refining')
                self.assertTrue((output/'residuals.npz').exists())
                self.assertIn('ionization', kwargs)
                self.assertIn('excitation', kwargs)
                return converged(fabricated_solution())

            with patch.object(stationary, 'solve_stationary', side_effect=solve) as solver:
                result = self.execute(output, '--max-starts', '2', '--separation', '1',
                                      '--guess', '5', '1', '.6')
            self.assertEqual(result, 0)
            self.assertEqual(solver.call_count, 3)
            report = json.loads((output/'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(report['stage'], 'refinement_complete')
            self.assertEqual(len(report['solutions']), 1)
            self.assertEqual(report['diagnostic']['duplicate_convergences'], 2)
            root = report['solutions'][0]
            self.assertGreater(root['j_target'], 1.)
            self.assertLess(root['coefficients']['local_ranges']['Q_eff_profile'][0], 0.)
            self.assertTrue(root['stationary_verified'])
            with np.load(output/root['profile_file'], allow_pickle=False) as profile:
                self.assertIn('q_e_erg_cm2_s', profile.files)
                self.assertIn('T_eV', profile.files)
            self.assertTrue(all(a['start_id'] for a in report['diagnostic']['attempts']))

    def test_no_roots_returns_one_but_completed_search_not_nonexistence(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(residual_map, 'solve_trial',
                side_effect=manufactured_trial), patch.object(stationary, 'solve_stationary',
                return_value=dict(status='line_search_failed', diagnostic={})):
            output = Path(folder)/'none'
            self.assertEqual(self.execute(output), 1)
            report = json.loads((output/'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(report['stage'], 'refinement_complete')
            self.assertEqual(report['solutions'], [])
            self.assertIn('nonexistence', report['warning'])

    def test_all_invalid_map_does_not_force_a_fake_start(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(residual_map, 'solve_trial',
                return_value=dict(status='adas_outside_domain')), patch.object(
                stationary, 'solve_stationary') as solver:
            output = Path(folder)/'invalid'
            self.assertEqual(self.execute(output), 1)
            report = json.loads((output/'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(report['starts'], [])
            self.assertEqual(report['summary']['n_valid'], 0)
            solver.assert_not_called()

    def test_interruption_preserves_finished_attempt_and_root(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(residual_map, 'solve_trial',
                side_effect=manufactured_trial), patch.object(stationary, 'solve_stationary',
                side_effect=[converged(fabricated_solution()), KeyboardInterrupt()]):
            output = Path(folder)/'interrupted'
            self.assertEqual(self.execute(output, '--separation', '1', '--max-starts', '2'), 130)
            report = json.loads((output/'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(report['stage'], 'interrupted')
            self.assertEqual(len(report['solutions']), 1)
            self.assertEqual(len(report['diagnostic']['attempts']), 1)
            self.assertTrue((output/'root_001_profiles.npz').exists())

    def test_coefficient_bug_records_failed_not_zero_roots(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(cli, 'scan_residuals',
                side_effect=ValueError('coefficient implementation bug')):
            output = Path(folder)/'error'
            with self.assertRaises(SystemExit) as error:
                self.execute(output)
            self.assertEqual(error.exception.code, 2)
            report = json.loads((output/'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(report['stage'], 'failed')
            self.assertEqual(report['error']['type'], 'ValueError')

    def test_interruption_during_mapping_does_not_claim_a_saved_cube(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(cli, 'scan_residuals',
                side_effect=KeyboardInterrupt()):
            output = Path(folder)/'mapping_interrupted'
            self.assertEqual(self.execute(output), 130)
            report = json.loads((output/'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(report['stage'], 'interrupted')
            self.assertEqual(report['last_completed_or_active_stage'], 'mapping')
            self.assertFalse((output/'residuals.npz').exists())

    def test_export_checks_independent_balance_even_when_flag_is_set(self):
        with tempfile.TemporaryDirectory() as folder:
            args = cli.build_parser().parse_args(self.arguments(Path(folder)/'out'))
            state = fabricated_solution()
            state['solution_id'] = 'root_001'
            for error in (.01, np.nan):
                state['balances']['energy_abs'] = error
                with self.subTest(error=error), self.assertRaises(RuntimeError):
                    cli._solution_record(state, load_config(PACKAGE/'config.json'), args)

    def test_bad_export_is_not_accepted_even_if_verified_flag_is_set(self):
        state = fabricated_solution()
        state['R_E'] = .1
        with tempfile.TemporaryDirectory() as folder, patch.object(residual_map, 'solve_trial',
                side_effect=manufactured_trial), patch.object(stationary, 'solve_stationary',
                return_value=converged(state)):
            output = Path(folder)/'bad_export'
            with self.assertRaises(SystemExit):
                self.execute(output)
            report = json.loads((output/'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(report['stage'], 'failed')
            self.assertEqual(report['solutions'], [])

    def test_atomic_json_rejects_nan_without_corrupting_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'summary.json'
            cli._save_json(path, {'good': 1})
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                cli._save_json(path, {'bad': np.nan})
            self.assertEqual(path.read_bytes(), before)

    def test_plotting_writes_map_and_profile_pngs_with_fake_solver(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(residual_map, 'solve_trial',
                side_effect=manufactured_trial), patch.object(stationary, 'solve_stationary',
                return_value=converged(fabricated_solution())):
            output = Path(folder)/'plots'
            arguments = self.arguments(output, '--alpha-slice', '0', '--max-starts', '1')
            arguments.remove('--no-plot')
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(arguments), 0)
            for filename in ('residuals.png', 'alpha_slice_000.png', 'root_001_profiles.png'):
                self.assertGreater((output/filename).stat().st_size, 5000)


if __name__ == '__main__':
    unittest.main()
