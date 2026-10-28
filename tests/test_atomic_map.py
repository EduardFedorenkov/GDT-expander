"""Ionization in the 2D map, automatic starts and independently checked roots."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from isothermal_cx_model.geometry import constant_B, quadratic_area_legacy
from isothermal_cx_model.recombination_map import (
    make_j_grid, scan_residuals, candidate_cells, choose_starts, plot_residuals,
)
from isothermal_cx_model.solver_atomic import solve_trial
from isothermal_cx_model.stationary_recombination import find_stationary_solutions
from isothermal_cx_model.recombination_batch import scan_lambda_neutral, _read_case


CONFIG = Path(__file__).resolve().parents[1]/'simple_cx_model/cx_config.json'


class AtomicMapTests(unittest.TestCase):
    def test_mixed_grid_resolves_both_sides_and_preserves_endpoints(self):
        js = make_j_grid(1e-6, 10, 48)
        self.assertEqual(js[0], 1e-6)
        self.assertEqual(js[-1], 10)
        self.assertTrue(np.all(np.diff(js) > 0))
        self.assertIn(1., js)
        self.assertTrue(np.any((js < 1) & (js > 1-1e-10)))
        self.assertTrue(np.any((js > 1) & (js < 1+1e-10)))
        for low, high in [(1, 10), (1.01, 4), (2, 10), (.99, 1.01)]:
            for mode in ('linear', 'log', 'mixed'):
                with self.subTest(bounds=(low, high), mode=mode):
                    grid = make_j_grid(low, high, 12, mode)
                    self.assertEqual(grid[0], low)
                    self.assertEqual(grid[-1], high)
                    self.assertTrue(np.all(np.diff(grid) > 0))

    def test_legacy_mixed_grid_is_unchanged(self):
        for low, high in [(1e-6, 1), (.999, 1), (.1, .99)]:
            broad = np.geomspace(low, high, 24)
            min_gap = max(1-high, min(1e-12, .1*(1-low)))
            near = 1-np.geomspace(min_gap, min(.5, 1-low), 24)
            near = near[(near >= low) & (near <= high)]
            np.testing.assert_array_equal(make_j_grid(low, high, 48), np.unique(np.r_[broad, near]))

    def test_ionization_is_forwarded_and_coefficients_are_saved_per_node(self):
        ion = lambda theta: (1+theta, .2*(1+theta))
        theta, js = [.13, .14], [1.3, 1.4]
        with patch('isothermal_cx_model.recombination_map.solve_trial', wraps=solve_trial) as calls:
            scan = scan_residuals(theta, js, .5, 0., 5., constant_B(), ionization=ion)
        self.assertTrue(scan['valid'].all())
        self.assertTrue(all(call.kwargs['ionization'] is ion for call in calls.call_args_list))
        self.assertEqual(calls.call_count, 4)
        for ti, t in enumerate(theta):
            np.testing.assert_array_equal(scan['D_ion'][:, ti], 1+t)
            np.testing.assert_array_equal(scan['E_ion'][:, ti], .2*(1+t))
            for ji, j in enumerate(js):
                direct = solve_trial(t, j, .5, 0., 5., constant_B(), ionization=ion, n_points=100)
                self.assertEqual(scan['status'][ji, ti], direct['status'])
                self.assertEqual(scan['R_J'][ji, ti], direct['R_J'])
                self.assertEqual(scan['R_M'][ji, ti], direct['R_M'])

    def test_zero_ionization_reproduces_previous_map_exactly(self):
        args = ([.05, .1], [.7, 1.], .5, 4., 5., constant_B())
        default = scan_residuals(*args)
        zero = scan_residuals(*args, ionization=(0., 0.))
        for key in default:
            if isinstance(default[key], np.ndarray):
                np.testing.assert_array_equal(default[key], zero[key])

    def test_invalid_ivps_stay_masked_and_no_false_candidate_bridges(self):
        scan = scan_residuals([.1, .15], [.5, 2], .5, 0., 5., constant_B(),
                              ionization=(1., .2))
        self.assertFalse(scan['valid'].any())
        self.assertTrue(np.isnan(scan['log_residual_norm']).all())
        self.assertEqual(candidate_cells(scan), [])
        self.assertEqual(choose_starts(scan), [])
        self.assertIn('energy_temperature_limit', scan['summary']['statuses'])
        self.assertIn('nonpositive_particle_flux', scan['summary']['statuses'])

    def test_automatic_starts_recover_both_pure_ionization_analytic_roots(self):
        theta = [.01, .0115, .013, .13, .139, .145]
        js = [1.28, 1.34, 1.42, 4.4, 4.65, 4.9]
        scan = scan_residuals(theta, js, .5, 0., 5., constant_B(), ionization=(1., .2))
        starts = choose_starts(scan, max_starts=6, separation=1)
        self.assertTrue(all(s['j_target'] > 1 for s in starts))
        roots, diagnostic = find_stationary_solutions(
            [(s['theta'], s['j_target']) for s in starts], .5, 0., 5., constant_B(),
            ionization=(1., .2), theta_bounds=(theta[0], theta[-1]), n_points=4000)
        expected_b = np.sort(np.roots([5/4, -1.2, .2]))
        self.assertEqual(len(roots), 2)
        np.testing.assert_allclose([s['theta'] for s in roots], expected_b**2/4, atol=1e-9)
        np.testing.assert_allclose([s['j_target'] for s in roots], 1/expected_b, atol=1e-8)
        self.assertTrue(all(s['stationary_verified'] for s in roots))
        self.assertIn('not a proof', diagnostic['warning'])

    def test_signed_loss_plot_shows_gain_and_both_sides_without_invalid_contours(self):
        import matplotlib
        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt
        scan = scan_residuals([.12, .14], [.9, 1., 1.3, 1.4], .5, 0., 5., constant_B(),
                              ionization=(1., .2))
        for mode in ('loss', 'linear', 'log'):
            fig, axes = plot_residuals(scan, j_axis=mode, starts=choose_starts(scan))
            fig.canvas.draw()
            if mode == 'loss':
                self.assertLess(axes[0].get_ylim()[0], 0)
                self.assertGreater(axes[0].get_ylim()[1], 0)
                self.assertIn('net gain', axes[0].get_ylabel())
                self.assertEqual(axes[0].get_yscale(), 'symlog')
            plt.close(fig)


class AtomicMapCLITests(unittest.TestCase):
    def command(self, *args):
        return subprocess.run([sys.executable, '-B', '-m',
            'isothermal_cx_model.scan_recombination', *map(str, args)],
            capture_output=True, text=True)

    def test_actual_auto_root_without_manual_guess_and_complete_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)/'hot'
            source_before = CONFIG.read_bytes()
            args = ['--density', '1e15', '--lambda-sh', '7', '--ionization-scale', '1',
                '--temperature-min', '80', '--temperature-max', '110', '--temperature-count', '7',
                '--j-min', '1.35', '--j-max', '1.9', '--j-count', '7', '--j-grid', 'linear',
                '--j-axis', 'linear', '--max-starts', '2', '--points', '4000',
                '--no-show', '--output', out]
            result = self.command(*args)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            report = json.loads((out/'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(report['format_version'], 2)
            self.assertEqual(report['ionization_scale'], 1)
            self.assertEqual(len(report['solutions']), 1)
            self.assertTrue(all(s['source'] != 'manual' for s in report['starts']))
            root = report['solutions'][0]
            self.assertAlmostEqual(root['T_eV'], 96.1704570832, delta=2e-6)
            self.assertAlmostEqual(root['j_target'], 1.60179494169, delta=2e-7)
            self.assertGreater(root['coefficients']['D_ion'], 0)
            self.assertLess(root['net_particle_loss_fraction'], 0)
            self.assertGreater(root['ionization_count_per_inlet'], .5)
            self.assertGreater(root['recombination_count_per_inlet'], 0)
            self.assertTrue(all(a['converged_to'] == 'root_001'
                                for a in report['diagnostic']['attempts']))
            with np.load(out/'residuals.npz', allow_pickle=False) as data:
                self.assertEqual(data['valid'].shape, (7, 7))
                self.assertEqual(float(data['ionization_scale']), 1)
                self.assertGreater(np.nanmax(data['D_ion']), 0)
                self.assertGreater(np.min(data['j_target']), 1)
                np.testing.assert_array_equal(data['D_ion'][0], data['D_ion'][-1])
            with np.load(out/'root_001_profiles.npz', allow_pickle=False) as data:
                self.assertEqual(float(data['ionization_scale']), 1)
                self.assertGreater(float(data['D_ion']), 0)
                self.assertGreater(float(data['E_ion']), 0)
            self.assertGreater((out/'residuals.png').stat().st_size, 1000)
            self.assertEqual(CONFIG.read_bytes(), source_before)
            report_before = (out/'summary.json').read_bytes()
            self.assertEqual(self.command(*args).returncode, 2)
            self.assertEqual((out/'summary.json').read_bytes(), report_before)

    def test_map_only_defaults_expand_with_ionization_and_remain_legacy_when_off(self):
        with tempfile.TemporaryDirectory() as folder:
            for scale, upper in [(0, 1.), (1, 10.)]:
                out = Path(folder)/str(scale)
                result = self.command('--ionization-scale', scale,
                    '--temperature-min', '80', '--temperature-max', '100',
                    '--temperature-count', '2', '--j-count', '4',
                    '--map-only', '--no-show', '--output', out)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                report = json.loads((out/'summary.json').read_text(encoding='utf-8'))
                self.assertEqual(report['controls']['j_max'], upper)
                self.assertEqual(report['controls']['refinement_j_max'], None)
                self.assertEqual(report['stage'], 'map_only')
                self.assertEqual(report['solutions'], [])
                with np.load(out/'residuals.npz', allow_pickle=False) as data:
                    self.assertEqual(np.max(data['j_target']), upper)
                    self.assertEqual(float(data['ionization_scale']), scale)
                    if scale == 0:
                        np.testing.assert_array_equal(data['D_ion'], 0.)

    def test_invalid_ionization_scale_rejected_before_creating_output(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)/'bad'
            self.assertEqual(self.command('--ionization-scale', '2', '--output', out).returncode, 2)
            self.assertFalse(out.exists())


class BatchBoundaryTests(unittest.TestCase):
    def test_batch_scale_is_explicit_and_mismatched_report_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                scan_lambda_neutral([7], [1e15], CONFIG, Path(folder)/'out',
                    geometry=quadratic_area_legacy(), scan_arguments=['--ionization-scale=1'])
            report_path = Path(folder)/'ionizing.json'
            report_path.write_text(json.dumps(dict(ionization_scale=1)), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Mismatched ionization scale'):
                _read_case(report_path, 7, 1e15)


if __name__ == '__main__':
    unittest.main()
