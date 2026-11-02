"""CSV/plot tests only; no ADAS tables, IVPs, Newton searches or scans."""

from contextlib import redirect_stdout, redirect_stderr
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from isothermal_model_adas import plot_lambda_results as plots


FIELDS = ('case_id', 'solution_id', 'lambda_sh', 'n0_cm3', 'T_eV', 'a_cm', 'j_target',
          'rr_scale', 'ionization_scale', 'excitation_scale', 'inlet_momentum')


def samples():
    rows = []
    for index, density in enumerate((1e13, 2e13, 3e13)):
        for name, T, a, jt in (('cold', 5.+index, 100.+10*index, 8.-index),
                                ('hot', 60.+2*index, 170.+index, 2.+.1*index)):
            rows.append(dict(case_id=f'N{index}', solution_id=f'N{index}_{name}',
                lambda_sh=8., n0_cm3=density, T_eV=T, a_cm=a, j_target=jt,
                rr_scale=1., ionization_scale=1., excitation_scale=1., inlet_momentum='kinetic'))
    return rows


def cases(rows):
    result = {}
    for root in rows:
        pair = (root['lambda_sh'], root['n0_cm3'])
        if pair not in result:
            result[pair] = dict(case_id=root['case_id'], lambda_sh=pair[0], n0_cm3=pair[1],
                                n_roots=0, search_status='roots_found')
        result[pair]['n_roots'] += 1
    return list(result.values())


def write_csv(path, rows, fields):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter=';')
        writer.writeheader()
        writer.writerows(rows)


def fixture(directory, rows=None, grid=None, plan=None):
    directory = Path(directory)
    rows = samples() if rows is None else rows
    write_csv(directory/'roots.csv', rows, FIELDS)
    grid = cases(rows) if grid is None else grid
    write_csv(directory/'grid.csv', grid, ('case_id', 'lambda_sh', 'n0_cm3', 'n_roots', 'search_status'))
    if plan is not None:
        (directory/'manifest.json').write_text(json.dumps(dict(model='isothermal_adas',
            kind='lambda_neutral_scan', format_version=1, signature={'densities': plan})), encoding='utf-8')
    return directory/'roots.csv'


class CSVTests(unittest.TestCase):
    def test_bom_semicolon_and_multiple_roots(self):
        with tempfile.TemporaryDirectory() as folder:
            path = fixture(folder)
            rows = plots.load_roots_csv(path)
            grid = plots.load_grid_csv(path.with_name('grid.csv'))
        self.assertEqual(len(rows), 6)
        self.assertEqual([row['n_roots'] for row in grid], [2, 2, 2])
        self.assertEqual(rows[0]['j_target'], 8.)
        self.assertEqual(rows[0]['rr_scale'], 1.)

    def test_missing_current_column_is_not_silently_filled(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'roots.csv'
            path.write_text('lambda_sh;n0_cm3;T_eV;a_cm\n8;1e13;5;100\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'j_target'):
                plots.load_roots_csv(path)

    def test_duplicate_headers_and_ragged_rows_rejected(self):
        header = ';'.join(FIELDS)
        row = ';'.join(map(str, samples()[0].values()))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'bad.csv'
            for text in (header+';T_eV\n'+row+';5\n', header+'\n'+row+';extra\n', header+'\n8;1e13\n'):
                path.write_text(text, encoding='utf-8')
                with self.assertRaises(ValueError):
                    plots.load_roots_csv(path)

    def test_nonfinite_negative_and_invalid_scales_rejected(self):
        for key, value in (('T_eV', np.nan), ('a_cm', -1.), ('j_target', 0.),
                           ('n0_cm3', -1.), ('lambda_sh', np.inf), ('rr_scale', 2.)):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as folder:
                rows = samples()
                rows[0][key] = value
                path = fixture(folder, rows)
                with self.assertRaisesRegex(ValueError, 'row 2'):
                    plots.load_roots_csv(path)

    def test_exact_duplicate_sample_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = samples()
            path = fixture(folder, rows+[dict(rows[0])])
            with self.assertRaisesRegex(ValueError, 'Duplicate root'):
                plots.load_roots_csv(path)

    def test_grid_count_and_status_validation(self):
        for value in (-1, 1.5, np.nan):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as folder:
                grid = cases(samples())
                grid[0]['n_roots'] = value
                path = fixture(folder, grid=grid)
                with self.assertRaises(ValueError):
                    plots.load_grid_csv(path.with_name('grid.csv'))
        with tempfile.TemporaryDirectory() as folder:
            grid = cases(samples())
            grid[0]['search_status'] = 'failed'
            path = fixture(folder, grid=grid)
            with self.assertRaisesRegex(ValueError, 'Unfinished'):
                plots.load_grid_csv(path.with_name('grid.csv'))

    def test_manifest_plan_and_foreign_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            path = fixture(folder, plan=[1e13, 2e13, 3e13]).with_name('manifest.json')
            np.testing.assert_array_equal(plots.load_density_plan(path), [1e13, 2e13, 3e13])
            path.write_text(json.dumps({'kind': 'old_scan'}), encoding='utf-8')
            with self.assertRaises(ValueError):
                plots.load_density_plan(path)

    def test_import_does_not_load_the_solver_or_matplotlib(self):
        code = ('import sys; import isothermal_model_adas.plot_lambda_results; '
                'assert "isothermal_model_adas.transport" not in sys.modules; '
                'assert "isothermal_model_adas.parameter_scan" not in sys.modules; '
                'assert "matplotlib.pyplot" not in sys.modules')
        result = subprocess.run([sys.executable, '-B', '-c', code],
            cwd=Path(__file__).parents[2], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class PlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import matplotlib
        matplotlib.use('Agg', force=True)

    def tearDown(self):
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_default_three_panels_have_all_roots_and_no_guides(self):
        rows = samples()
        before = [dict(row) for row in rows]
        fig, axes = plots.plot_roots(rows, grid=cases(rows))
        self.assertEqual(len(axes), 3)
        self.assertEqual(sum(len(c.get_offsets()) for c in axes[0].collections), 6)
        self.assertEqual(len(axes[0].lines), 0)
        self.assertEqual(len(axes[2].lines), 1)  # j_t=1 reference only.
        self.assertEqual(rows, before)
        fig.canvas.draw()

    def test_optional_guides_include_knots_and_do_not_extrapolate(self):
        rows = samples()
        _, axes = plots.plot_roots(rows, grid=cases(rows), connect=True)
        self.assertEqual(len(axes[0].lines), 2)
        for line in axes[0].lines:
            x, y = line.get_data()
            self.assertGreaterEqual(x.min(), 1e13)
            self.assertLessEqual(x.max(), 3e13*(1+1e-14))
            for knot in (1e13, 2e13, 3e13):
                self.assertTrue(np.any(np.isclose(x, knot, rtol=1e-14)))
            self.assertTrue(np.isfinite(y).all())

    def test_zero_root_and_unknown_planned_gaps_break_guides(self):
        rows = [row for row in samples() if row['n0_cm3'] != 2e13]
        grid = cases(rows)
        _, axes = plots.plot_roots(rows, grid=grid, connect=True, density_values=[1e13, 2e13, 3e13])
        self.assertEqual(len(axes[0].lines), 0)
        grid.append(dict(lambda_sh=8., n0_cm3=2e13, n_roots=0, search_status='no_root_found_in_finite_search'))
        _, axes = plots.plot_roots(rows, grid=grid, connect=True)
        self.assertEqual(len(axes[0].lines), 0)

    def test_count_changes_restart_tracks(self):
        rows = [row for row in samples() if row['n0_cm3'] == 1e13 or row['T_eV'] > 50]
        _, axes = plots.plot_roots(rows, grid=cases(rows), connect=True)
        self.assertEqual(len(axes[0].lines), 1)
        np.testing.assert_allclose([axes[0].lines[0].get_xdata().min(),
                                    axes[0].lines[0].get_xdata().max()], [2e13, 3e13])

    def test_temperature_ties_are_not_assigned_a_false_order(self):
        rows = samples()
        for row in rows:
            if row['n0_cm3'] == 2e13:
                row['T_eV'] = 20.
        _, axes = plots.plot_roots(rows, grid=cases(rows), connect=True)
        self.assertEqual(len(axes[0].lines), 0)

    def test_different_process_settings_are_separate_groups(self):
        rows = samples()
        more = [dict(row, rr_scale=0.) for row in rows]
        fig, axes = plots.plot_roots(rows+more)
        self.assertEqual(len(axes[0].collections), 2)
        labels = [text.get_text() for text in axes[0].get_legend().get_texts()]
        self.assertTrue(all('RR=' in label for label in labels))
        fig.canvas.draw()

    def test_zero_density_control_is_shown_but_not_interpolated(self):
        rows = samples()
        rows.append(dict(rows[0], case_id='zero', solution_id='zero', n0_cm3=0.))
        fig, axes = plots.plot_roots(rows, grid=cases(rows), connect=True)
        self.assertEqual(axes[0].get_xscale(), 'symlog')
        self.assertTrue(all(line.get_xdata().min() > 0 for line in axes[0].lines))
        fig.canvas.draw()

    def test_lambda_filter_allows_completed_zero_root_lambda(self):
        rows = samples()
        grid = cases(rows)+[dict(lambda_sh=7., n0_cm3=1e13, n_roots=0,
                                 search_status='no_root_found_in_finite_search')]
        fig, axes = plots.plot_roots(rows, lambdas=[7.], grid=grid, temperature_scale='log')
        self.assertEqual(len(axes[0].collections), 0)
        self.assertTrue(axes[0].texts)
        self.assertGreater(axes[0].get_xlim()[0], 1e12)
        fig.canvas.draw()
        with self.assertRaises(ValueError):
            plots.plot_roots(rows, lambdas=[99.], grid=grid)

    def test_mismatched_snapshot_counts_and_case_ids_are_rejected(self):
        rows = samples()
        grid = cases(rows)
        grid[0]['n_roots'] = 1
        with self.assertRaises(ValueError):
            plots.plot_roots(rows, grid=grid)
        grid = cases(rows)
        grid[0]['case_id'] = 'different'
        with self.assertRaisesRegex(ValueError, 'IDs'):
            plots.plot_roots(rows, grid=grid)

    def test_guides_require_grid_and_plan_must_contain_saved_densities(self):
        with self.assertRaisesRegex(ValueError, 'grid.csv'):
            plots.plot_roots(samples(), connect=True)
        with self.assertRaisesRegex(ValueError, 'outside'):
            plots.plot_roots(samples(), density_values=[1e13])


class CommandTests(unittest.TestCase):
    def execute(self, arguments):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return plots.main(arguments)

    def test_cli_folder_autoloads_sidecars_and_does_not_change_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            path = fixture(folder, plan=[1e13, 2e13, 3e13])
            before = path.read_bytes()
            output = Path(folder)/'figure.png'
            self.assertEqual(self.execute([folder, '--connect', '--temperature-scale', 'log',
                                          '--save', str(output), '--no-show']), 0)
            self.assertGreater(output.stat().st_size, 5000)
            self.assertEqual(path.read_bytes(), before)

    def test_cli_explicit_file_and_filter(self):
        with tempfile.TemporaryDirectory() as folder:
            path = fixture(folder)
            output = Path(folder)/'selected.png'
            self.assertEqual(self.execute([str(path), '--lambdas', '8', '--points-only',
                                          '--save', str(output), '--no-show']), 0)
            self.assertTrue(output.exists())

    def test_existing_image_requires_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path = fixture(folder)
            image = Path(folder)/'existing.png'
            image.write_bytes(b'user image')
            argv = [str(path), '--save', str(image), '--no-show']
            with self.assertRaises(SystemExit) as error:
                self.execute(argv)
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(image.read_bytes(), b'user image')
            self.assertEqual(self.execute([*argv, '--overwrite']), 0)

    def test_input_files_cannot_be_overwritten_even_explicitly(self):
        with tempfile.TemporaryDirectory() as folder:
            path = fixture(folder)
            before = path.read_bytes()
            with self.assertRaises(SystemExit):
                self.execute([str(path), '--save', str(path), '--overwrite', '--no-show'])
            self.assertEqual(path.read_bytes(), before)

    def test_empty_root_table_with_completed_zero_cases_renders(self):
        with tempfile.TemporaryDirectory() as folder:
            path = fixture(folder, rows=[], grid=[dict(case_id='N0', lambda_sh=8., n0_cm3=1e13,
                n_roots=0, search_status='no_root_found_in_finite_search')])
            output = Path(folder)/'empty.png'
            self.assertEqual(self.execute([str(path), '--save', str(output), '--no-show']), 0)
            self.assertGreater(output.stat().st_size, 5000)


if __name__ == '__main__':
    unittest.main()
