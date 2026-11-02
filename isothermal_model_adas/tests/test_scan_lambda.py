"""Batch/CLI tests with fabricated IVPs and roots, NOT physical searches."""

from contextlib import redirect_stderr, redirect_stdout
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

from isothermal_model_adas import parameter_scan as batch
from isothermal_model_adas import scan_lambda as cli
from isothermal_model_adas import scan_stationary as single
from isothermal_model_adas import stationary, residual_map
from isothermal_model_adas.transport import reconstruct
from isothermal_model_adas.recombination_adas import IONIZATION_ENERGY_ERG
from isothermal_model_adas.tests.test_residual_map import manufactured_trial
from isothermal_model_adas.tests.test_scan_stationary import fabricated_solution, converged


PACKAGE = Path(__file__).parents[1]
DATA_AVAILABLE = all((PACKAGE/'processes_data'/name).exists() for name in
    ('acd12_h.dat', 'prb12_h.dat', 'ccd96_d.dat', 'scd12_h.dat', 'plt12_h.dat'))
SMALL = dict(temperature_count=2, j_count=2, alpha_count=2, points=100,
             max_starts=1, separation=1, no_plot=True)


def synthetic_child(command, cwd, log_path, *, no_roots=False):
    """Exercise the actual single-case CLI, replacing all numerical physics.

    Fabricated arrays obey export normalisations and bookkeeping, not the ODE.
    Nothing in these tests establishes existence of a plasma solution.
    """
    argv = command[command.index('isothermal_model_adas.scan_stationary')+1:]
    args = single.build_parser().parse_args(argv)
    p, geometry, _, providers, *_ = single._prepare(args)
    if no_roots:
        outcome = dict(status='line_search_failed', diagnostic={})
    else:
        ion_active = p['n0_cm3'] > 0 and args.ionization_scale > 0
        state = fabricated_solution(jt=1.2 if ion_active else .9)
        theta, jt, alpha = (state[key] for key in ('theta', 'j_target', 'alpha'))
        xi = state['xi']
        inlet = providers['inlet_momentum']
        target = 1. if inlet is None else inlet.evaluate(alpha)['mu_target']
        b = np.sqrt(8*p['beta']*theta)
        state['mu_target'] = target
        state['mu'] = (1-xi)*target+xi*b*jt
        state['omega'] = 1-xi+xi*p['lambda_sh']*theta*jt
        H, area, N, v = reconstruct(state['g'], (state['j'], state['mu'], state['omega']),
                                    theta, p['beta'], geometry)
        state.update(A_over_A_star=area, n_over_n_star=N, u_over_u_star=v,
            u_over_cs=2*v/b, Aq_e_over_W0=state['omega']-state['j']*(v*v/(2*p['beta'])+5*theta),
            D=providers['D'](theta) if callable(providers['D']) else providers['D'])
        born, lost = (.3, .1) if ion_active else (0., .1)
        total = 1-state['omega'][-1]
        ion_energy = IONIZATION_ENERGY_ERG/p['T_star_erg']*born
        cx_loss = total+.2 if p['n0_cm3'] > 0 else 0.
        excitation_loss = .05 if p['n0_cm3'] > 0 else 0.
        state['balances'].update(particle_ionization_integral=born,
            particle_recombination_integral=lost, particle_net_source_integral=born-lost,
            energy_loss_integral=total, cx_energy_loss_integral=cx_loss,
            excitation_energy_loss_integral=excitation_loss,
            ionization_energy_loss_integral=ion_energy,
            ion_energy_loss_integral=.1, effective_energy_loss_integral=total-cx_loss-excitation_loss-ion_energy-.1)
        outcome = converged(state)
    with patch.object(residual_map, 'solve_trial', side_effect=manufactured_trial), \
            patch.object(stationary, 'solve_stationary', return_value=outcome):
        code = single.main(argv)
    Path(log_path).write_text('SYNTHETIC TEST CHILD: no plasma calculation\n', encoding='utf-8')
    return code


class UtilityTests(unittest.TestCase):
    def test_ranges_and_defaults(self):
        np.testing.assert_array_equal(cli._range([8, 5, 8], 1., 2., 3, 'lambda'), [5., 8.])
        np.testing.assert_allclose(cli._range(None, 1e12, 1e14, 3, 'n0', logarithmic=True), [1e12, 1e13, 1e14])
        np.testing.assert_array_equal(cli._range([0., 1e13], 1., 2., 3, 'n0', logarithmic=True), [0., 1e13])
        self.assertEqual(batch.scan_defaults()['temperature_max'], 150.)
        for values in ([], [np.nan], [-1.]):
            with self.assertRaises(ValueError):
                batch.parameter_grid(values, 'lambda')
        with self.assertRaises(ValueError):
            cli._range(None, 5., 9., 1, 'lambda')

    def test_options_are_typed_and_batch_flags_cannot_be_injected(self):
        for options in ({'config': 'other.json'}, {'density': 1e13}, {'map_only': True},
                        {'temperature_count': 2.5}, {'no_plot': 'yes'}, {'guess': [1., 2.]},
                        {'inlet_momentum': 'unsupported'}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                batch._options(options)

    def test_report_paths_cannot_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                batch._inside(Path(folder), '../other/summary.json')

    def test_real_child_process_only_prints_a_marker_and_saves_log(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            log = Path(folder)/'child.log'
            result = batch._run_scan([sys.executable, '-B', '-u', '-c', "print('child marker')"],
                                     Path(folder), log)
            self.assertEqual(result, 0)
            self.assertIn('child marker', log.read_text(encoding='utf-8'))


@unittest.skipUnless(DATA_AVAILABLE, 'Full ADAS dataset is not distributed with code.')
class BatchTests(unittest.TestCase):
    def run_batch(self, output, **kwargs):
        lambdas = kwargs.pop('lambdas', [7., 8.])
        densities = kwargs.pop('densities', [1e13, 2e13])
        options = dict(SMALL, **kwargs.pop('options', {}))
        with redirect_stdout(io.StringIO()):
            return batch.scan_lambda_neutral(lambdas, densities, PACKAGE/'config.json', output,
                                              scan_options=options, **kwargs)

    def test_full_batch_csv_checkpoint_and_resume_do_not_rerun_completed_pairs(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(batch, '_run_scan', side_effect=synthetic_child) as runner:
            output = Path(folder)/'batch'
            roots, grid = self.run_batch(output)
            self.assertEqual(runner.call_count, 4)
            self.assertEqual(len(roots), 4)
            self.assertEqual(len({r['solution_id'] for r in roots}), 4)
            self.assertTrue(all(r['j_target'] > 1 for r in roots))
            self.assertTrue(all(r['effective_energy_loss_fraction'] < 0 for r in roots))
            self.assertEqual([r['n_roots'] for r in grid], [1]*4)
            with (output/'roots.csv').open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream, delimiter=';'))
            self.assertEqual(len(rows), 4)
            self.assertIn('case_report', rows[0])
            self.assertEqual(json.loads((output/'status.json').read_text())['stage'], 'complete')
            first = (output/'source_config.json').read_bytes()
            self.assertEqual(first, (PACKAGE/'config.json').read_bytes())
            # Aggregate CSVs are rebuildable; checkpoint/reports are authoritative.
            (output/'roots.csv').write_text('broken aggregate', encoding='utf-8')
            again, again_grid = self.run_batch(output, resume=True)
            self.assertEqual(runner.call_count, 4)
            self.assertEqual(again, roots)
            self.assertEqual(again_grid, grid)
            command = runner.call_args.args[0]
            self.assertIn('--no-show', command)
            self.assertIn('--alpha-count', command)
            self.assertNotIn('--map-only', command)

    def test_no_roots_is_a_completed_finite_search_not_a_failure(self):
        def no_roots(*args):
            return synthetic_child(*args, no_roots=True)
        with tempfile.TemporaryDirectory() as folder, patch.object(batch, '_run_scan', side_effect=no_roots):
            output = Path(folder)/'empty'
            roots, grid = self.run_batch(output, lambdas=[8.], densities=[1e13])
            self.assertEqual(roots, [])
            self.assertEqual(grid[0]['n_roots'], 0)
            self.assertEqual(grid[0]['search_status'], 'no_root_found_in_finite_search')
            self.assertEqual(len(json.loads((output/'checkpoint.json').read_text())['completed']), 1)

    def test_interrupted_case_is_preserved_and_retried_in_new_attempt_directory(self):
        calls = []

        def interrupt_second(command, cwd, log_path):
            calls.append(command)
            if len(calls) == 2:
                output = Path(command[command.index('--output')+1])
                output.mkdir()
                (output/'partial.json').write_text('{}', encoding='utf-8')
                raise KeyboardInterrupt()
            return synthetic_child(command, cwd, log_path)

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'interrupted'
            with patch.object(batch, '_run_scan', side_effect=interrupt_second):
                with self.assertRaises(KeyboardInterrupt):
                    self.run_batch(output, lambdas=[8.])
            checkpoint = json.loads((output/'checkpoint.json').read_text())
            self.assertEqual(len(checkpoint['completed']), 1)
            self.assertEqual(json.loads((output/'status.json').read_text())['stage'], 'interrupted')
            with patch.object(batch, '_run_scan', side_effect=synthetic_child) as runner:
                roots, grid = self.run_batch(output, lambdas=[8.], resume=True)
            self.assertEqual(runner.call_count, 1)
            self.assertEqual(len(grid), 2)
            self.assertTrue((output/'cases/L000_N001/attempt_001/partial.json').exists())
            self.assertIn('attempt_002', grid[-1]['case_report'])

    def test_exit_one_without_completed_report_does_not_become_zero_roots(self):
        def failed(command, cwd, log_path):
            output = Path(command[command.index('--output')+1])
            output.mkdir()
            (output/'summary.json').write_text(json.dumps({'stage': 'failed'}), encoding='utf-8')
            return 1
        with tempfile.TemporaryDirectory() as folder, patch.object(batch, '_run_scan', side_effect=failed):
            output = Path(folder)/'failed'
            with self.assertRaises(ValueError):
                self.run_batch(output, lambdas=[8.], densities=[1e13])
            self.assertEqual(json.loads((output/'checkpoint.json').read_text())['completed'], {})
            self.assertEqual(json.loads((output/'status.json').read_text())['stage'], 'failed')

    def test_unexpected_child_exit_is_not_checkpointed(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(batch, '_run_scan', return_value=2):
            output = Path(folder)/'exit2'
            with self.assertRaises(RuntimeError):
                self.run_batch(output, lambdas=[8.], densities=[1e13])
            self.assertEqual(json.loads((output/'checkpoint.json').read_text())['completed'], {})

    def test_child_exit_must_agree_with_the_verified_report(self):
        def wrong_exit(*args):
            synthetic_child(*args)
            return 1  # The report actually contains a root.
        with tempfile.TemporaryDirectory() as folder, patch.object(batch, '_run_scan', side_effect=wrong_exit):
            output = Path(folder)/'wrong_exit'
            with self.assertRaisesRegex(ValueError, 'exit code'):
                self.run_batch(output, lambdas=[8.], densities=[1e13])
            self.assertEqual(json.loads((output/'checkpoint.json').read_text())['completed'], {})

    def test_zero_neutral_density_has_only_recombination_table_provenance(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(batch, '_run_scan', side_effect=synthetic_child):
            output = Path(folder)/'zero_control'
            roots, grid = self.run_batch(output, lambdas=[8.], densities=[0., 1e13])
            self.assertEqual(len(grid), 2)
            zero = json.loads((output/grid[0]['case_report']).read_text())
            self.assertEqual(set(zero['provenance']['data_sha256']), {'acd12_h.dat', 'prb12_h.dat'})
            self.assertEqual(roots[0]['D'], 0.)

    def test_changed_frozen_input_mid_batch_stops_before_the_next_pair(self):
        def change_frozen(command, cwd, log_path):
            code = synthetic_child(command, cwd, log_path)
            frozen = Path(command[command.index('--config')+1])
            frozen.write_bytes(frozen.read_bytes()+b' ')
            return code
        with tempfile.TemporaryDirectory() as folder, patch.object(batch, '_run_scan', side_effect=change_frozen) as runner:
            output = Path(folder)/'changed'
            with self.assertRaisesRegex(ValueError, 'changed during'):
                self.run_batch(output, lambdas=[8.])
            self.assertEqual(runner.call_count, 1)
            self.assertEqual(len(json.loads((output/'checkpoint.json').read_text())['completed']), 1)

    def test_nonfinite_profile_is_rejected_before_checkpoint(self):
        def bad_profile(command, cwd, log_path):
            code = synthetic_child(command, cwd, log_path)
            path = Path(command[command.index('--output')+1])/'root_001_profiles.npz'
            with np.load(path, allow_pickle=False) as source:
                arrays = {key: source[key] for key in source.files}
            arrays['omega'][1] = np.nan
            np.savez_compressed(path, **arrays)
            return code
        with tempfile.TemporaryDirectory() as folder, patch.object(batch, '_run_scan', side_effect=bad_profile):
            output = Path(folder)/'nan_profile'
            with self.assertRaisesRegex(ValueError, 'Nonfinite saved profile'):
                self.run_batch(output, lambdas=[8.], densities=[1e13])
            self.assertEqual(json.loads((output/'checkpoint.json').read_text())['completed'], {})

    def test_completed_checkpoint_path_cannot_escape_the_batch_directory(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(batch, '_run_scan', side_effect=synthetic_child):
            output = Path(folder)/'escape'
            self.run_batch(output, lambdas=[8.], densities=[1e13])
            path = output/'checkpoint.json'
            checkpoint = json.loads(path.read_text())
            checkpoint['completed']['L000_N000']['report'] = '../outside.json'
            path.write_text(json.dumps(checkpoint), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'escapes'):
                self.run_batch(output, lambdas=[8.], densities=[1e13], resume=True)

    def test_resume_rejects_changed_settings_code_and_data(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(batch, '_run_scan', side_effect=synthetic_child):
            output = Path(folder)/'resume'
            self.run_batch(output, lambdas=[8.], densities=[1e13])
            with self.assertRaisesRegex(ValueError, 'Resume mismatch'):
                self.run_batch(output, lambdas=[8.], densities=[1e13], resume=True, options={'temperature_count': 3})
            path = output/'manifest.json'
            original = path.read_bytes()
            for field in ('code_sha256', 'data_sha256'):
                manifest = json.loads(original)
                key = next(iter(manifest['signature'][field]))
                manifest['signature'][field][key] = 'changed'
                path.write_text(json.dumps(manifest), encoding='utf-8')
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Resume mismatch'):
                    self.run_batch(output, lambdas=[8.], densities=[1e13], resume=True)
                path.write_bytes(original)

    def test_modified_completed_report_profile_or_frozen_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(batch, '_run_scan', side_effect=synthetic_child):
            output = Path(folder)/'modified'
            self.run_batch(output, lambdas=[8.], densities=[1e13])
            checkpoint = json.loads((output/'checkpoint.json').read_text())
            report = output/next(iter(checkpoint['completed'].values()))['report']
            for path in (report, report.parent/'root_001_profiles.npz', output/'source_config.json'):
                original = path.read_bytes()
                path.write_bytes(original+b' ')
                with self.subTest(path=path), self.assertRaises((ValueError, OSError)):
                    self.run_batch(output, lambdas=[8.], densities=[1e13], resume=True)
                path.write_bytes(original)

    def test_wrong_density_or_negative_budget_cannot_enter_csv(self):
        for mutation in ('density', 'budget', 'escape'):
            def bad_child(command, cwd, log_path):
                result = synthetic_child(command, cwd, log_path)
                path = Path(command[command.index('--output')+1])/'summary.json'
                report = json.loads(path.read_text())
                if mutation == 'density':
                    report['parameters']['n0_cm3'] *= 2
                elif mutation == 'budget':
                    report['solutions'][0]['balances']['particle_abs'] = -.1
                else:
                    report['solutions'][0]['profile_file'] = '../escape.npz'
                path.write_text(json.dumps(report), encoding='utf-8')
                return result
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder, \
                    patch.object(batch, '_run_scan', side_effect=bad_child):
                output = Path(folder)/'bad'
                with self.assertRaises(ValueError):
                    self.run_batch(output, lambdas=[8.], densities=[1e13])
                self.assertEqual(json.loads((output/'checkpoint.json').read_text())['completed'], {})

    def test_dry_run_and_existing_output_safety(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(batch, '_run_scan') as runner:
            output = Path(folder)/'dry'
            self.assertEqual(self.run_batch(output, dry_run=True), ([], []))
            self.assertFalse(output.exists())
            output.mkdir()
            marker = output/'keep.txt'
            marker.write_text('user data', encoding='utf-8')
            with self.assertRaises(ValueError):
                self.run_batch(output)
            self.assertEqual(marker.read_text(), 'user data')
            runner.assert_not_called()

    def test_cli_runs_mocked_batch_and_saves_two_figures(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(batch, '_run_scan', side_effect=synthetic_child):
            output = Path(folder)/'cli'
            argv = ['--lambdas', '7', '8', '--densities', '1e13', '--temperature-count', '2',
                '--j-count', '2', '--alpha-count', '2', '--points', '100', '--max-starts', '1',
                '--no-case-plots', '--no-show', '--output', str(output)]
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(argv), 0)
            for name in ('root_count.png', 'roots.png'):
                self.assertGreater((output/name).stat().st_size, 5000)


class PlotTests(unittest.TestCase):
    def test_empty_and_zero_density_plots_render(self):
        import matplotlib
        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt
        for grid in ([dict(n0_cm3=1e13, lambda_sh=8., n_roots=0)],
                     [dict(n0_cm3=0., lambda_sh=8., n_roots=0)]):
            figures = cli.plot_results([], grid)
            for figure in figures:
                figure.canvas.draw()
                plt.close(figure)


if __name__ == '__main__':
    unittest.main()
