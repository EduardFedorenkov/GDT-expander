"""Finite maps preserve invalid domains; starts and roots remain distinct."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from isothermal_cx_model.geometry import constant_B
from isothermal_cx_model.recombination_map import (
    make_j_grid, scan_residuals, candidate_cells, choose_starts, plot_residuals,
)


def fake_trial(theta, jt, *args, **kwargs):
    RJ, RM = jt-.5, theta-.2
    return dict(status='energy_matched', R_J=RJ, R_M=RM,
        R_J_scaled=RJ/(2+abs(RJ)), R_M_scaled=RM/(2+abs(RM)),
        j_a=1+RJ, mu_a=1+RM, mu_target=1., alpha=.7, R_E=0.,
        inlet_target_sonic_margin=-1.)


class MapTests(unittest.TestCase):
    def scan(self, function=fake_trial, **kwargs):
        with patch('isothermal_cx_model.recombination_map.solve_trial', function):
            return scan_residuals([.1, .3], [.3, .8], .5, 4., 5., constant_B(), **kwargs)

    def test_grid_and_near_one_resolution(self):
        js = make_j_grid()
        self.assertEqual(js[0], 1e-6)
        self.assertEqual(js[-1], 1.)
        self.assertTrue(np.all(np.diff(js) > 0))
        self.assertTrue(np.any((js < 1) & (js > 1-1e-10)))
        for mode in ('mixed', 'linear', 'log'):
            zoom = make_j_grid(.999, 1, 20, mode)
            self.assertTrue(np.all(zoom >= .999))
            self.assertEqual(zoom[-1], 1.)

    def test_bad_axes_grid_and_start_controls(self):
        for args in [(0, 1, 10), (.1, 1.1, 10), (.1, 1, 2), (.1, .1, 10),
                     (.1, 1, 10, 'unknown')]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                make_j_grid(*args)
        for theta, js in [([.1], [.2, .3]), ([.1, .1], [.2, .3]),
                          ([.1, .2], [0, 1]), ([.1, .2], [.5, 2])]:
            with self.assertRaises(ValueError):
                scan_residuals(theta, js, .5, 4, 5)
        for control in [dict(max_starts=0), dict(separation=0)]:
            with self.assertRaises(ValueError):
                choose_starts(self.scan(), **control)

    def test_shape_signs_and_actual_corrector_norm(self):
        scan = self.scan()
        np.testing.assert_allclose(scan['R_J'], [[-.2, -.2], [.3, .3]])
        np.testing.assert_allclose(scan['R_M'], [[-.1, .1], [-.1, .1]])
        self.assertTrue(scan['valid'].all())  # Do NOT reject negative target margin.
        self.assertEqual(candidate_cells(scan), [(0, 0)])
        np.testing.assert_allclose(scan['log_residual_norm'], np.maximum(
            abs(np.log(scan['j_a'])), abs(np.log(scan['mu_a']/scan['mu_target']))))

    def test_invalid_corner_not_bridged(self):
        def hole(theta, jt, *args, **kwargs):
            if theta == .1 and jt == .3:
                return dict(status='sonic_return')
            return fake_trial(theta, jt)
        scan = self.scan(hole)
        self.assertEqual(candidate_cells(scan), [])
        self.assertTrue(np.isnan(scan['log_residual_norm'][0, 0]))
        self.assertTrue(all((s['j_index'], s['theta_index']) != (0, 0)
                            for s in choose_starts(scan)))

    def test_no_joint_domain_keeps_partial_residual_and_status(self):
        scan = self.scan(lambda *a, **k: dict(status='kinetic_inlet_outside_domain',
                                             R_J=2, alpha=.5))
        np.testing.assert_array_equal(scan['R_J'], 2)
        self.assertFalse(scan['valid'].any())
        self.assertIsNone(scan['summary']['nearest_sample_not_root'])
        self.assertEqual(choose_starts(scan), [])

    def test_numerical_failures_visible_and_programming_error_propagates(self):
        scan = self.scan(lambda *a, **k: dict(status='integration_failed', reason='test failure'))
        self.assertEqual(scan['summary']['unresolved_numerical_count'], 4)
        self.assertEqual(scan['summary']['reason_counts'], {'test failure': 4})
        def bad(*args, **kwargs):
            raise ValueError('bad coefficient')
        with self.assertRaisesRegex(ValueError, 'bad coefficient'):
            self.scan(bad)

    def test_parallel_contours_not_a_root_proof(self):
        def parallel(theta, jt, *args, **kwargs):
            result = fake_trial(theta, jt)
            result.update(R_J=theta-.15, R_M=theta-.25)
            return result
        scan = self.scan(parallel)
        self.assertEqual(len(candidate_cells(scan)), 1)
        self.assertIn('not a proof', scan['summary']['warning'])

    def test_starts_are_valid_deterministic_and_prioritized(self):
        scan = self.scan()
        starts = choose_starts(scan, max_starts=2, separation=1)
        self.assertEqual(len(starts), 2)
        self.assertEqual(starts, choose_starts(scan, max_starts=2, separation=1))
        self.assertEqual(starts[0]['source'], 'candidate_cell_corner')
        self.assertEqual(len({(s['j_index'], s['theta_index']) for s in starts}), 2)

    def test_progress_controls_and_no_pickle(self):
        progress, calls = [], []
        def record(*args, **kwargs):
            calls.append(kwargs)
            return fake_trial(*args, **kwargs)
        scan = self.scan(record, rr=(.1, .01), E_exc=.03, rtol=1e-10,
                         progress=lambda done, total: progress.append((done, total)))
        self.assertEqual(progress, [(1, 2), (2, 2)])
        self.assertTrue(all(k['rr'] == (.1, .01) and k['rtol'] == 1e-10 for k in calls))
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder)/'map.npz'
            np.savez_compressed(file, **{k: v for k, v in scan.items() if isinstance(v, np.ndarray)})
            with np.load(file, allow_pickle=False) as loaded:
                np.testing.assert_array_equal(loaded['valid'], scan['valid'])
                np.testing.assert_array_equal(loaded['status'], scan['status'])
        json.dumps(scan['summary'], allow_nan=False)

    def test_actual_zero_rr_map(self):
        scan = scan_residuals([.05, .1], [.7, 1.], .5, 4, 5,
                             constant_B(), n_points=60)
        self.assertTrue(scan['valid'].all())
        np.testing.assert_allclose(scan['R_J'], [[-.3, -.3], [0, 0]], atol=1e-14)

    def test_plot_all_axes_masks_roots_starts_and_paths(self):
        import matplotlib
        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt
        for scan in [self.scan(), self.scan(lambda *a, **k: dict(status='sonic_return'))]:
            for mode in ('loss', 'log', 'linear'):
                states = [dict(theta=.2, j_target=.5, solution_id='root_001')]
                attempts = [dict(history=[dict(theta=.1, j_target=.3, phase='search'),
                                          dict(theta=.2, j_target=.5, phase='search')])]
                fig, axes = plot_residuals(scan, 100, j_axis=mode,
                    solutions=states, starts=choose_starts(scan), attempts=attempts)
                fig.canvas.draw()
                self.assertEqual(len(axes), 3)
                self.assertEqual(axes[0].get_xscale(), 'log')
                self.assertEqual(axes[0].get_yscale(), 'symlog' if mode == 'loss' else mode)
                plt.close(fig)


class CLITests(unittest.TestCase):
    def write_source(self, folder):
        data = dict(A_m_cm2=38.48451000647496, n_c_cm3=3e13, T_c_eV=150.,
            chi_E=8., chi=5., k_cx_cm3_per_s=1e-8, lambda_sh=8.,
            L_cm=184.5, n0_cm3=1e14, m_D_g=3.3435837724e-24)
        path = Path(folder)/'source.json'
        path.write_text(json.dumps(data), encoding='utf-8')
        return path, data

    def command(self, *args):
        return subprocess.run([sys.executable, '-B', '-m',
            'isothermal_cx_model.scan_recombination', *map(str, args)],
            capture_output=True, text=True)

    def test_help_guess_optional(self):
        result = self.command('--help')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('[--guess T_EV J_TARGET]', result.stdout)
        for flag in ('--map-only', '--max-starts', '--j-axis', '--temperature-count'):
            self.assertIn(flag, result.stdout)

    def test_cli_automatic_scan_without_guess_and_preserve_old_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source, config = self.write_source(folder)
            output = folder/'results'
            args = ['--config', source, '--density', '1e15', '--lambda-sh', '7',
                '--temperature-min', '.1', '--temperature-max', '.15',
                '--temperature-count', '3', '--j-min', '.7', '--j-max', '.85',
                '--j-count', '4', '--j-grid', 'linear', '--max-starts', '1',
                '--no-show', '--output', output]
            result = self.command(*args)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            report = json.loads((output/'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(report['stage'], 'refinement_complete')
            self.assertEqual(report['starts'][0]['source'], 'candidate_cell_corner')
            self.assertEqual(len(report['solutions']), 1)
            self.assertEqual(report['diagnostic']['attempts'][0]['converged_to'], 'root_001')
            self.assertEqual(report['starts'][0]['start_id'], 'start_001')
            self.assertAlmostEqual(report['solutions'][0]['T_eV'], .119194929, places=6)
            self.assertTrue((output/'residuals.png').exists())
            with np.load(output/'residuals.npz', allow_pickle=False) as data:
                self.assertEqual(data['valid'].shape, (4, 3))
            self.assertEqual(json.loads(source.read_text()), config)
            before = (output/'summary.json').read_bytes()
            repeated = self.command(*args)
            self.assertEqual(repeated.returncode, 2)
            self.assertEqual(before, (output/'summary.json').read_bytes())

    def test_map_only_saves_without_claiming_roots(self):
        with tempfile.TemporaryDirectory() as folder:
            result = self.command('--temperature-min', '.1', '--temperature-max', '.2',
                '--temperature-count', '2', '--j-count', '4', '--map-only', '--no-show',
                '--output', Path(folder)/'results')
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            report = json.loads((Path(folder)/'results/summary.json').read_text(encoding='utf-8'))
            self.assertEqual(report['stage'], 'map_only')
            self.assertEqual(report['solutions'], [])
            self.assertTrue((Path(folder)/'results/residuals.png').exists())

    def test_intermediate_root_found_without_manual_guess(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'results'
            source, _ = self.write_source(folder)
            result = self.command('--config', source, '--density', '1e15', '--lambda-sh', '7',
                '--temperature-min', '2.8', '--temperature-max', '3.8',
                '--temperature-count', '5', '--j-min', '.999992', '--j-max', '.999998',
                '--j-count', '5', '--j-grid', 'linear', '--max-starts', '1',
                '--no-show', '--output', output)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            report = json.loads((output/'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(len(report['solutions']), 1)
            self.assertAlmostEqual(report['solutions'][0]['T_eV'], 3.272164647, places=6)
            self.assertAlmostEqual(report['solutions'][0]['j_target'], .999994997394, places=10)

    def test_bad_input_rejected_before_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            result = self.command('--temperature-min', '10', '--temperature-max', '1',
                                  '--output', Path(folder)/'results')
            self.assertEqual(result.returncode, 2)
            self.assertFalse((Path(folder)/'results').exists())


if __name__ == '__main__':
    unittest.main()
