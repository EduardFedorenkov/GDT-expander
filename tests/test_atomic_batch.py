"""Atomic batch provenance, positive-j roots, CSV schema and real child runs."""

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

from isothermal_cx_model.geometry import quadratic_area_legacy
from isothermal_cx_model.ionization import k_ion
from isothermal_cx_model.parameter_scan import scan_lambda_neutral
from isothermal_cx_model.scan_lambda import main
from test_parameter_scan import fake_scan, flag, source_file


class AtomicBatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.config = source_file(self.base)
        self.output = self.base/'results'

    def run_scan(self, **kwargs):
        options = dict(ionization_scale=1.)
        options.update(kwargs)
        with contextlib.redirect_stdout(io.StringIO()):
            return scan_lambda_neutral([7.], [9e14, 1e15], self.config, self.output,
                                       geometry=quadratic_area_legacy(), **options)

    def modified_child(self, change):
        def child(command, cwd, log_path):
            code = fake_scan(command, cwd, log_path)
            file = Path(flag(command, '--output'))/'summary.json'
            report = json.loads(file.read_text())
            change(report)
            file.write_text(json.dumps(report), encoding='utf-8')
            return code
        return child

    def test_scale_forwarded_once_manifest_and_counts_written(self):
        with patch('isothermal_cx_model.parameter_scan._run_scan', side_effect=fake_scan) as child:
            roots, grid = self.run_scan()
        self.assertEqual(child.call_count, 2)
        self.assertEqual(len(roots), 4)  # Synthetic multi-root reports, no branch selection.
        self.assertEqual(len(grid), 2)
        for call in child.call_args_list:
            command = call.args[0]
            self.assertEqual(command.count('--ionization-scale'), 1)
            self.assertEqual(float(flag(command, '--ionization-scale')), 1.)
        manifest = json.loads((self.output/'manifest.json').read_text())
        self.assertEqual(manifest['format_version'], 2)
        self.assertEqual(manifest['ionization_scale'], 1.)
        for row in roots:
            self.assertNotIn('recombined_fraction', row)
            self.assertEqual(row['root_kind'], 'verified_stationary_atomic')
            self.assertEqual(row['ionization_scale'], 1.)
            self.assertEqual(row['format_version'], 2)
            self.assertAlmostEqual(row['j_target']-1,
                row['ionization_count_per_inlet']-row['recombination_count_per_inlet'])
        # Same T/source in the two synthetic reports: only prescribed n0 differs.
        self.assertAlmostEqual(roots[3]['D_ion']/roots[1]['D_ion'], 1e15/9e14)
        with (self.output/'roots_temperature_recombination.csv').open(encoding='utf-8-sig') as stream:
            fields = csv.DictReader(stream, delimiter=';').fieldnames
        for key in ('ionization_scale', 'D_ion', 'E_ion', 'net_particle_loss_fraction',
                    'ionization_count_per_inlet', 'recombination_count_per_inlet', 'energy_ionization_fraction'):
            self.assertIn(key, fields)
        self.assertNotIn('recombined_fraction', fields)
        self.assertTrue(all(row['ionization_scale'] == 1 for row in grid))

    def test_root_above_one_is_accepted_when_counts_agree(self):
        def above_one(report):
            root = report['solutions'][0]
            root.update(j_target=1.2, net_particle_loss_fraction=-.2,
                        recombination_count_per_inlet=.2)
            root['balances']['particle_recombination_integral'] = .2
        with patch('isothermal_cx_model.parameter_scan._run_scan',
                   side_effect=self.modified_child(above_one)):
            roots, _ = self.run_scan()
        self.assertGreater(roots[0]['j_target'], 1)
        self.assertLess(roots[0]['net_particle_loss_fraction'], 0)

    def test_resume_reuses_same_physics_and_rejects_changed_scale_or_schema(self):
        with patch('isothermal_cx_model.parameter_scan._run_scan', side_effect=fake_scan):
            before = self.run_scan()
        with patch('isothermal_cx_model.parameter_scan._run_scan') as child:
            self.assertEqual(self.run_scan(resume=True), before)
        child.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'Resume requires'):
            self.run_scan(ionization_scale=.5, resume=True)
        manifest_path = self.output/'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['format_version'] = 1
        manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Resume requires'):
            self.run_scan(resume=True)

    def test_invalid_scale_and_override_aliases_rejected_before_output(self):
        for scale in (-1, 2, np.nan, np.inf):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                self.run_scan(ionization_scale=scale)
            self.assertFalse(self.output.exists())
        for flag_name in ('--ionization-scale=0', '--ionization-scale', '--ionization-sca=0'):
            with self.subTest(flag_name=flag_name), self.assertRaises(ValueError):
                self.run_scan(scan_arguments=[flag_name])
            self.assertFalse(self.output.exists())

    def test_inconsistent_reports_are_not_committed_as_completed_pairs(self):
        def wrong_coefficient(report):
            report['solutions'][1]['coefficients']['D_ion'] *= 2
        def wrong_energy(report):
            report['solutions'][1]['balances']['energy_ionization_integral'] *= 2
        def wrong_count(report):
            report['solutions'][0]['balances']['particle_ionization_integral'] += .1
        def bad_residual(report):
            report['solutions'][0]['R_J'] = .01
        def bad_j(report):
            report['solutions'][0]['j_target'] = 0
        changes = [lambda r: r.update(ionization_scale=0), wrong_coefficient,
                   wrong_energy, wrong_count, bad_residual, bad_j]
        for index, change in enumerate(changes):
            self.output = self.base/f'bad_{index}'
            with self.subTest(index=index), patch('isothermal_cx_model.parameter_scan._run_scan',
                side_effect=self.modified_child(change)), self.assertRaises(RuntimeError):
                self.run_scan()
            progress = json.loads((self.output/'progress.json').read_text())
            self.assertEqual(progress['completed'], {})
            self.assertEqual(progress['stage'], 'failed')
            self.assertFalse((self.output/'roots_temperature_recombination.csv').exists())

    def test_resume_revalidates_stored_case_physics(self):
        with patch('isothermal_cx_model.parameter_scan._run_scan', side_effect=fake_scan):
            self.run_scan()
        report_path = self.output/'cases/pair_00001/attempt_001/result/summary.json'
        report = json.loads(report_path.read_text())
        report['ionization_scale'] = 0
        report_path.write_text(json.dumps(report), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Mismatched ionization'):
            self.run_scan(resume=True)

    def test_active_empty_search_is_completed_without_claiming_nonexistence(self):
        def empty(command, cwd, log_path):
            return fake_scan(command, cwd, log_path, empty=True)
        with patch('isothermal_cx_model.parameter_scan._run_scan', side_effect=empty):
            roots, grid = self.run_scan()
        self.assertEqual(roots, [])
        self.assertTrue(all(r['n_roots'] == 0 and r['ionization_scale'] == 1 for r in grid))
        self.assertTrue(all(r['search_status'] == 'no_root_found_from_this_search' for r in grid))
        self.assertEqual(json.loads((self.output/'progress.json').read_text())['stage'], 'complete')

    def test_active_interruption_preserves_completed_pair_and_retries_new_attempt(self):
        def interrupt(command, cwd, log_path):
            if float(flag(command, '--density')) == 1e15:
                Path(log_path).write_text('partial ionization run', encoding='utf-8')
                raise KeyboardInterrupt
            return fake_scan(command, cwd, log_path)
        with patch('isothermal_cx_model.parameter_scan._run_scan', side_effect=interrupt), \
                self.assertRaises(KeyboardInterrupt):
            self.run_scan()
        self.assertEqual(len(json.loads((self.output/'progress.json').read_text())['completed']), 1)
        with patch('isothermal_cx_model.parameter_scan._run_scan', side_effect=fake_scan) as child:
            roots, grid = self.run_scan(resume=True)
        self.assertEqual(child.call_count, 1)
        self.assertEqual(len(roots), 4)
        self.assertEqual(len(grid), 2)
        self.assertEqual((self.output/'cases/pair_00002/attempt_001/scan.log').read_text(),
                         'partial ionization run')
        self.assertTrue((self.output/'cases/pair_00002/attempt_002/result/summary.json').exists())

    def test_new_plot_separates_ionization_settings_and_loads_old_csv(self):
        import matplotlib
        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt
        from isothermal_cx_model.plot_lambda_results import load_roots_csv, plot_roots
        with patch('isothermal_cx_model.parameter_scan._run_scan', side_effect=fake_scan):
            self.run_scan()
        rows = load_roots_csv(self.output/'roots_temperature_recombination.csv')
        figure, axes = plot_roots(rows+[dict(rows[0], ionization_scale=0.)])
        figure.canvas.draw()
        labels = axes[0].get_legend_handles_labels()[1]
        self.assertTrue(any('ionization=0' in label for label in labels))
        self.assertTrue(any('ionization=1' in label for label in labels))
        plt.close(figure)
        legacy = self.base/'old.csv'
        legacy.write_text('lambda_sh;n0_cm3;T_eV;a_cm\n7;1e15;1;180\n', encoding='utf-8-sig')
        figure, _ = plot_roots(load_roots_csv(legacy))
        figure.canvas.draw()
        plt.close(figure)

    def test_cli_effective_default_grid_is_consistent_with_child(self):
        for scale, upper in [(0, 1.), (1, 10.)]:
            with self.subTest(scale=scale), patch('sys.argv', ['scan_lambda',
                '--ionization-scale', str(scale), '--lambdas', '7', '--densities', '1e15',
                '--no-plot', '--output', str(self.output)]), contextlib.redirect_stdout(io.StringIO()), \
                patch('isothermal_cx_model.scan_lambda.scan_lambda_neutral',
                      return_value=([], [])) as batch:
                self.assertEqual(main(), 0)
            self.assertEqual(batch.call_args.kwargs['ionization_scale'], scale)
            inner = batch.call_args.kwargs['scan_arguments']
            self.assertEqual(float(flag(inner, '--j-max')), upper)
            self.assertNotIn('--ionization-scale', inner)  # Controlled separately.


class ActualAtomicBatchCLITests(unittest.TestCase):
    def command(self, module, args):
        return subprocess.run([sys.executable, '-B', '-m', module, *map(str, args)],
                              capture_output=True, text=True)

    def test_actual_density_pairs_resume_and_agreement_with_single_map(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            config = source_file(base)
            before = config.read_bytes()
            output = base/'batch'
            inner = ['--config', config, '--ionization-scale', '1',
                '--temperature-min', '80', '--temperature-max', '110', '--temperature-count', '7',
                '--j-min', '1.35', '--j-max', '1.9', '--j-count', '7', '--j-grid', 'linear',
                '--j-axis', 'linear', '--max-starts', '2', '--points', '4000', '--no-show']
            args = inner+['--lambdas', '7', '--densities', '9e14', '1e15', '--output', output]
            result = self.command('isothermal_cx_model.scan_lambda', args)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            csv_path = output/'roots_temperature_recombination.csv'
            with csv_path.open(encoding='utf-8-sig') as stream:
                roots = list(csv.DictReader(stream, delimiter=';'))
            self.assertEqual(len(roots), 2)
            for row in roots:
                self.assertGreater(float(row['j_target']), 1)
                self.assertEqual(float(row['ionization_scale']), 1)
                report = json.loads((output/row['case_report']).read_text())
                root = report['solutions'][0]
                p = report['parameters']
                expected = p['L_cm']*float(row['n0_cm3'])*k_ion(float(row['T_eV']))/p['U_star_cm_per_s']
                self.assertAlmostEqual(float(row['D_ion'])/expected, 1., delta=1e-12)
                self.assertGreater(float(row['ionization_count_per_inlet']), .5)
                self.assertLess(float(row['net_particle_loss_fraction']), 0)
                self.assertGreater(root['balances']['particle_ionization_integral'], .5)
            single = self.command('isothermal_cx_model.scan_stationary', inner+
                ['--density', '1e15', '--lambda-sh', '7', '--output', base/'single'])
            self.assertEqual(single.returncode, 0, single.stdout+single.stderr)
            root = json.loads((base/'single/summary.json').read_text())['solutions'][0]
            for key in ('T_eV', 'alpha', 'j_target'):
                self.assertAlmostEqual(float(roots[1][key]), root[key], delta=1e-8)
            csv_before = csv_path.read_bytes()
            repeated = self.command('isothermal_cx_model.scan_lambda', args+['--resume'])
            self.assertEqual(repeated.returncode, 0, repeated.stdout+repeated.stderr)
            self.assertEqual(repeated.stdout.count('Reusing completed pair.'), 2)
            self.assertNotIn('Temperature columns:', repeated.stdout)
            self.assertEqual(csv_path.read_bytes(), csv_before)
            self.assertEqual(config.read_bytes(), before)
            changed = args.copy()
            changed[changed.index('--ionization-scale')+1] = '0'
            result = self.command('isothermal_cx_model.scan_lambda', changed+['--resume'])
            self.assertEqual(result.returncode, 2)
            self.assertIn('Resume requires', result.stderr)
            self.assertEqual(csv_path.read_bytes(), csv_before)
            for name in ('root_count.png', 'roots_temperature_recombination.png'):
                self.assertGreater((output/name).stat().st_size, 1000)


if __name__ == '__main__':
    unittest.main()
