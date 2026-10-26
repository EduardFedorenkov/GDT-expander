"""Batch state, scientific provenance, root retention and real CLI checks."""

import contextlib
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from simple_cx_model.config import load_cx_config
from isothermal_cx_model.geometry import quadratic_area_legacy
from isothermal_cx_model.recombination_batch import scan_lambda_neutral
from isothermal_cx_model.scan_lambda_recombination import _range


def source_file(folder):
    path = Path(folder)/'source.json'
    data = dict(A_m_cm2=38.48451000647496, n_c_cm3=3e13, T_c_eV=150.,
        chi_E=8., chi=5., k_cx_cm3_per_s=1e-8, lambda_sh=8., L_cm=184.5,
        n0_cm3=1e14, m_D_g=3.3435837724e-24)
    path.write_text(json.dumps(data), encoding='utf-8')
    return path


def flag(command, key):
    return command[command.index(key)+1]


def fake_scan(command, cwd, log_path, empty=False):
    """Synthetic child report, not a numerical physics test."""
    out = Path(flag(command, '--output'))
    out.mkdir(parents=True)
    Path(log_path).write_text('synthetic child\n', encoding='utf-8')
    p = load_cx_config(flag(command, '--config'))
    p.update(lambda_sh=float(flag(command, '--lambda-sh')), n0_cm3=float(flag(command, '--density')))
    D = p['L_cm']*p['n0_cm3']*p['k_cx_cm3_per_s']/p['U_star_cm_per_s']
    solutions = []
    if not empty:
        for i, temperature in enumerate((.1, 3.), start=1):
            solutions.append(dict(solution_id=f'root_{i:03d}', theta=temperature/p['T_star_eV'],
                T_eV=temperature, alpha=.8, a_cm=.8*p['L_cm'], j_target=.8+i*.01,
                R_J=0., R_M=0., R_E=0., outside_scan_j_range=False,
                coefficients=dict(D=D, E_exc=.1, R_rr=.03, Q_rr=.01),
                verification=dict(balance_tol=5e-6),
                balances=dict(particle_abs=1e-10, momentum_abs=2e-9, energy_abs=3e-9)))
    report = dict(stage='refinement_complete', parameters=p,
        geometry=json.loads(Path(flag(command, '--geometry')).read_text()),
        cx_mode='temperature', rr_scale=1., excitation_scale=1., inlet_momentum='kinetic',
        solutions=solutions, starts=[dict(theta=.1, j_target=.8)]*len(solutions),
        diagnostic=dict(attempts=[dict(status='converged')]*len(solutions)),
        summary=dict(n_trials=12, n_valid=8, candidate_cell_count=2,
            nearest_sample_not_root=dict(log_residual_norm=.01),
            unresolved_numerical_count=0, statuses=dict(energy_matched=8, sonic_return=4)))
    (out/'summary.json').write_text(json.dumps(report), encoding='utf-8')
    return 1 if empty else 0


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.config = source_file(self.base)
        self.output = self.base/'results'

    def run_scan(self, lambdas=(7., 8.), densities=(1e14, 1e15), **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return scan_lambda_neutral(lambdas, densities, self.config, self.output,
                                       geometry=quadratic_area_legacy(), **kwargs)

    def test_every_pair_independent_all_roots_retained_and_coefficients_evaluated(self):
        before = self.config.read_bytes()
        with patch('isothermal_cx_model.recombination_batch._run_scan', side_effect=fake_scan) as runner:
            roots, grid = self.run_scan(lambdas=(8., 7., 7.))
        self.assertEqual(runner.call_count, 4)
        self.assertEqual(len(roots), 8)
        self.assertEqual(len(grid), 4)
        self.assertEqual(len({r['solution_id'] for r in roots}), 8)
        self.assertEqual([r['n_roots'] for r in grid], [2]*4)
        self.assertEqual(self.config.read_bytes(), before)
        self.assertAlmostEqual(roots[2]['D']/roots[0]['D'], 10.)
        for call in runner.call_args_list:
            cmd = call.args[0]
            self.assertNotIn('--guess', cmd)
            self.assertEqual(Path(flag(cmd, '--config')), self.output/'source_config.json')
            self.assertIn('--no-show', cmd)
        with (self.output/'roots_temperature_recombination.csv').open(encoding='utf-8-sig', newline='') as stream:
            self.assertEqual(len(list(csv.DictReader(stream, delimiter=';'))), 8)

    def test_completed_resume_does_not_rerun_children(self):
        with patch('isothermal_cx_model.recombination_batch._run_scan', side_effect=fake_scan):
            expected = self.run_scan()
        before = (self.output/'roots_temperature_recombination.csv').read_bytes()
        with patch('isothermal_cx_model.recombination_batch._run_scan') as runner:
            actual = self.run_scan(resume=True)
        runner.assert_not_called()
        self.assertEqual(actual, expected)
        self.assertEqual((self.output/'roots_temperature_recombination.csv').read_bytes(), before)

    def test_interruption_preserves_prior_pair_and_retries_in_new_directory(self):
        def interrupted(command, cwd, log_path):
            if float(flag(command, '--density')) == 1e15:
                Path(log_path).write_text('partial log', encoding='utf-8')
                raise KeyboardInterrupt
            return fake_scan(command, cwd, log_path)
        with patch('isothermal_cx_model.recombination_batch._run_scan', side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                self.run_scan(lambdas=(7.,))
        progress = json.loads((self.output/'progress.json').read_text())
        self.assertEqual(len(progress['completed']), 1)
        with patch('isothermal_cx_model.recombination_batch._run_scan', side_effect=fake_scan) as runner:
            roots, grid = self.run_scan(lambdas=(7.,), resume=True)
        self.assertEqual(runner.call_count, 1)
        self.assertEqual(len(roots), 4)
        self.assertEqual(len(grid), 2)
        parent = self.output/'cases/pair_00002'
        self.assertEqual((parent/'attempt_001/scan.log').read_text(), 'partial log')
        self.assertTrue((parent/'attempt_002/result/summary.json').exists())

    def test_no_root_child_exit_is_a_completed_pair_not_nonexistence_proof(self):
        def no_roots(command, cwd, log_path):
            return fake_scan(command, cwd, log_path, empty=True)
        with patch('isothermal_cx_model.recombination_batch._run_scan', side_effect=no_roots):
            roots, grid = self.run_scan(lambdas=(7.,), densities=(1e14,))
        self.assertEqual(roots, [])
        self.assertEqual(grid[0]['n_roots'], 0)
        self.assertIn('from_this_search', grid[0]['search_status'])
        self.assertEqual(json.loads((self.output/'progress.json').read_text())['stage'], 'complete')

    def test_failure_is_not_recorded_as_zero_roots_and_previous_pairs_survive(self):
        def fail_second(command, cwd, log_path):
            if float(flag(command, '--density')) == 1e15:
                return 2
            return fake_scan(command, cwd, log_path)
        with patch('isothermal_cx_model.recombination_batch._run_scan', side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, 'pair_00002 failed'):
                self.run_scan(lambdas=(7.,))
        progress = json.loads((self.output/'progress.json').read_text())
        self.assertEqual(progress['stage'], 'failed')
        self.assertEqual(len(progress['completed']), 1)
        with (self.output/'grid_temperature_recombination.csv').open(encoding='utf-8-sig') as stream:
            self.assertEqual(len(list(csv.DictReader(stream, delimiter=';'))), 1)

    def test_exit_one_without_completed_report_is_not_no_root_result(self):
        with patch('isothermal_cx_model.recombination_batch._run_scan', return_value=1):
            with self.assertRaises(RuntimeError):
                self.run_scan(lambdas=(7.,), densities=(1e14,))
        self.assertEqual(json.loads((self.output/'progress.json').read_text())['completed'], {})

    def test_bad_balance_and_inconsistent_child_exit_rejected(self):
        for mode in ('bad_balance', 'bad_exit'):
            with self.subTest(mode=mode):
                self.output = self.base/mode
                def bad(command, cwd, log_path):
                    code = fake_scan(command, cwd, log_path)
                    if mode == 'bad_exit':
                        return 1
                    file = Path(flag(command, '--output'))/'summary.json'
                    report = json.loads(file.read_text())
                    report['solutions'][0]['balances']['energy_abs'] = .1
                    file.write_text(json.dumps(report), encoding='utf-8')
                    return code
                with patch('isothermal_cx_model.recombination_batch._run_scan', side_effect=bad):
                    with self.assertRaises(RuntimeError):
                        self.run_scan(lambdas=(7.,), densities=(1e14,))

    def test_resume_refuses_changed_source_settings_and_code(self):
        with patch('isothermal_cx_model.recombination_batch._run_scan', side_effect=fake_scan):
            self.run_scan()
        with self.assertRaisesRegex(ValueError, 'Resume requires'):
            self.run_scan(lambdas=(7.,), resume=True)
        with patch('isothermal_cx_model.recombination_batch.code_fingerprint', return_value='changed'):
            with self.assertRaisesRegex(ValueError, 'Resume requires'):
                self.run_scan(resume=True)
        data = json.loads(self.config.read_text())
        data['n_c_cm3'] *= 2
        self.config.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'Resume requires'):
            self.run_scan(resume=True)

    def test_existing_output_and_reserved_arguments_never_overwritten(self):
        self.output.mkdir()
        note = self.output/'user_note.txt'
        note.write_text('keep')
        with self.assertRaisesRegex(ValueError, 'Output must'):
            self.run_scan()
        self.assertEqual(note.read_text(), 'keep')
        for flag_name in ('--map-only', '--density', '--config', '--output=elsewhere'):
            with self.subTest(flag_name=flag_name), self.assertRaises(ValueError):
                self.run_scan(scan_arguments=[flag_name])

    def test_resume_path_cannot_escape_output(self):
        with patch('isothermal_cx_model.recombination_batch._run_scan', side_effect=fake_scan):
            self.run_scan(lambdas=(7.,), densities=(1e14,))
        file = self.output/'progress.json'
        progress = json.loads(file.read_text())
        progress['completed']['pair_00001']['report'] = '../outside.json'
        file.write_text(json.dumps(progress))
        with self.assertRaisesRegex(ValueError, 'escapes'):
            self.run_scan(lambdas=(7.,), densities=(1e14,), resume=True)

    def test_parameter_grids_and_plot_compatibility(self):
        np.testing.assert_array_equal(_range([8, 7, 7], 0, 0, 0, 'lambda'), [7, 8])
        with self.assertRaises(ValueError):
            _range(None, 5, 9, 1, 'lambda')
        with patch('isothermal_cx_model.recombination_batch._run_scan', side_effect=fake_scan):
            self.run_scan()
        import matplotlib
        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt
        from isothermal_cx_model.plot_lambda_results import load_roots_csv, plot_roots
        rows = load_roots_csv(self.output/'roots_temperature_recombination.csv')
        self.assertEqual(len(rows), 8)
        figure, axes = plot_roots(rows)
        figure.canvas.draw()
        self.assertEqual(len(axes), 2)
        plt.close(figure)

    def test_children_use_frozen_source_when_original_is_edited(self):
        initial = load_cx_config(self.config)
        def edit_original(command, cwd, log_path):
            code = fake_scan(command, cwd, log_path)
            data = json.loads(self.config.read_text())
            data['T_c_eV'] *= 2
            self.config.write_text(json.dumps(data))
            return code
        with patch('isothermal_cx_model.recombination_batch._run_scan', side_effect=edit_original):
            roots, grid = self.run_scan(lambdas=(7.,))
        self.assertEqual(len(grid), 2)
        self.assertEqual(load_cx_config(self.output/'source_config.json'), initial)
        self.assertEqual(len(roots), 4)

    def test_code_change_during_child_prevents_accepting_pair(self):
        with patch('isothermal_cx_model.recombination_batch.code_fingerprint',
                   side_effect=['same', 'same', 'changed']):
            with patch('isothermal_cx_model.recombination_batch._run_scan', side_effect=fake_scan):
                with self.assertRaises(RuntimeError):
                    self.run_scan(lambdas=(7.,), densities=(1e14,))
        progress = json.loads((self.output/'progress.json').read_text())
        self.assertEqual(progress['stage'], 'failed')
        self.assertEqual(progress['completed'], {})


class CLITests(unittest.TestCase):
    def command(self, *args):
        return subprocess.run([sys.executable, '-B', '-m',
            'isothermal_cx_model.scan_lambda_recombination', *map(str, args)],
            capture_output=True, text=True)

    def test_help_and_bad_input(self):
        result = self.command('--help')
        self.assertEqual(result.returncode, 0, result.stderr)
        for flag_name in ('--lambdas', '--densities', '--resume', '--cx-mode', '--rr-scale'):
            self.assertIn(flag_name, result.stdout)
        with tempfile.TemporaryDirectory() as folder:
            result = self.command('--lambdas', '0', '--output', Path(folder)/'results')
            self.assertEqual(result.returncode, 2)
            self.assertFalse((Path(folder)/'results').exists())

    def test_actual_two_pairs_and_resume_without_manual_guess(self):
        with tempfile.TemporaryDirectory() as folder:
            config = source_file(folder)
            output = Path(folder)/'results'
            args = ['--config', config, '--lambdas', '7', '7.1', '--densities', '1e15',
                '--temperature-min', '.1', '--temperature-max', '.15',
                '--temperature-count', '3', '--j-min', '.7', '--j-max', '.85',
                '--j-count', '4', '--j-grid', 'linear', '--max-starts', '1',
                '--no-show', '--output', output]
            result = self.command(*args)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            with (output/'roots_temperature_recombination.csv').open(encoding='utf-8-sig') as stream:
                roots = list(csv.DictReader(stream, delimiter=';'))
            self.assertEqual(len(roots), 2)
            self.assertAlmostEqual(float(roots[0]['T_eV']), .119194929, places=6)
            self.assertEqual(len({r['solution_id'] for r in roots}), 2)
            for row in roots:
                report = output/row['case_report']
                self.assertTrue(report.exists())
                self.assertTrue((report.parent/'residuals.npz').exists())
                self.assertTrue((report.parent/'residuals.png').exists())
            self.assertTrue((output/'root_count.png').exists())
            self.assertTrue((output/'roots_temperature_recombination.png').exists())
            before = (output/'roots_temperature_recombination.csv').read_bytes()
            repeated = self.command(*args, '--resume')
            self.assertEqual(repeated.returncode, 0, repeated.stdout+repeated.stderr)
            self.assertIn('Reusing completed pair.', repeated.stdout)
            self.assertEqual((output/'roots_temperature_recombination.csv').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
