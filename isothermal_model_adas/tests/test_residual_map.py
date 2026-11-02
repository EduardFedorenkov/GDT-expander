"""Synthetic residual-map checks; no physical IVP or root search is run."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from isothermal_model_adas.geometry import constant_B, quadratic_area_legacy
from isothermal_model_adas import residual_map as maps


def manufactured_trial(theta, jt, alpha, beta, lam, recombination, geometry, **kwargs):
    """Known root at (.2,1.2,.5), independent of the plasma RHS."""
    f = (np.log(theta/.2), np.log(jt/1.2), alpha-.5)
    target = 2+.5*alpha
    return dict(status='integrated', j=np.array([np.exp(f[0]), jt]),
        mu=np.array([target*np.exp(f[1]), 1.]), omega=np.array([1+f[2], 0.]),
        mu_target=target, D=.3, wall_slope_per_j=-.1)


class ResidualMapTests(unittest.TestCase):
    def scan(self, theta=(.1, .2, .4), jt=(.6, 1.2, 2.4), alpha=(.2, .5, .8),
             provider=manufactured_trial, **kwargs):
        with patch.object(maps, 'solve_trial', side_effect=provider):
            return maps.scan_residuals(theta, jt, alpha, 1., 5., (.1, -.2),
                                       constant_B(), **kwargs)

    def test_shape_order_and_all_three_residuals(self):
        scan = self.scan()
        self.assertEqual(scan['valid'].shape, (3, 3, 3))
        self.assertTrue(scan['valid'].all())
        np.testing.assert_allclose(scan['F_J'][0, 2], np.log(np.array([.1, .2, .4])/.2))
        np.testing.assert_allclose(scan['F_M'][0, :, 1], np.log(np.array([.6, 1.2, 2.4])/1.2))
        np.testing.assert_allclose(scan['F_E'][:, 1, 1], [-.3, 0., .3])
        np.testing.assert_allclose(scan['residual_norm'],
            np.max(np.abs([scan[k] for k in maps._RESIDUALS]), axis=0))
        self.assertEqual(scan['summary']['nearest_sample_not_root']['alpha'], .5)
        self.assertEqual(scan['summary']['n_trials'], 27)
        self.assertEqual(scan['summary']['nearest_sample_not_root']['residual_norm'], 0.)
        self.assertNotIn('stationary_verified', scan)

    def test_raw_residuals_and_nonconstant_inlet_target(self):
        scan = self.scan()
        np.testing.assert_allclose(scan['R_J'], scan['j_a']-1)
        np.testing.assert_allclose(scan['R_M'], scan['mu_a']-scan['mu_target'])
        np.testing.assert_allclose(scan['R_E'], scan['omega_a']-1)
        self.assertNotEqual(scan['mu_target'][0, 0, 0], scan['mu_target'][-1, 0, 0])

    def test_process_providers_and_controls_forwarded_unchanged(self):
        cx = lambda t: .5*t
        ion = lambda t, n: (.1, .2)
        exc = lambda t, n: .3
        inlet = lambda a: 1.
        progress = []
        with patch.object(maps, 'solve_trial', side_effect=manufactured_trial) as trial:
            scan = maps.scan_residuals([.2], [1.2], [.5], 1., 5., (.1, -.2),
                constant_B(), D=cx, ionization=ion, excitation=exc, inlet_momentum=inlet,
                n_points=12, rtol=1e-8, atol=1e-10, max_step=.01,
                progress=lambda done, total: progress.append((done, total)))
        self.assertTrue(scan['valid'].all())
        args = trial.call_args.kwargs
        for key, expected in dict(D=cx, ionization=ion, excitation=exc,
                                  inlet_momentum=inlet).items():
            self.assertIs(args[key], expected)
        self.assertEqual(args['n_points'], 12)
        self.assertEqual(args['rtol'], 1e-8)
        self.assertEqual(progress, [(1, 1)])

    def test_invalid_trials_remain_undefined_and_reasons_counted(self):
        def invalid(*args, **kwargs):
            return dict(status='adas_outside_domain', reason='outside density range')
        scan = self.scan(provider=invalid)
        self.assertFalse(scan['valid'].any())
        self.assertTrue(np.isnan(scan['residual_norm']).all())
        self.assertIsNone(scan['summary']['nearest_sample_not_root'])
        self.assertEqual(scan['summary']['statuses'], {'adas_outside_domain': 27})
        self.assertEqual(scan['summary']['reason_counts'], {'outside density range': 27})
        self.assertEqual(maps.choose_starts(scan), [])

    def test_programming_and_malformed_data_errors_propagate(self):
        for error in (ValueError('bad coefficient'), FileNotFoundError('missing table')):
            with self.subTest(error=error), patch.object(maps, 'solve_trial', side_effect=error):
                with self.assertRaises(type(error)):
                    maps.scan_residuals([.2], [1.2], [.5], 1., 5., (0., 0.))

    def test_invalid_integrated_endpoints_not_mistaken_for_good_trials(self):
        for key, value in [('j', [0.]), ('mu', [-1.]), ('omega', [np.nan]),
                           ('mu_target', np.inf)]:
            def bad(*args, **kwargs):
                result = manufactured_trial(*args, **kwargs)
                result[key] = value
                return result
            with self.subTest(key=key):
                scan = self.scan(provider=bad)
                self.assertFalse(scan['valid'].any())
                self.assertEqual(scan['summary']['statuses'], {'invalid_inlet_residual': 27})
                self.assertEqual(scan['summary']['unresolved_numerical_count'], 27)

    def test_negative_energy_and_particle_gain_are_not_filtered(self):
        def signed(*args, **kwargs):
            trial = manufactured_trial(*args, **kwargs)
            trial['omega'][0] = -2.
            return trial
        scan = self.scan(provider=signed)
        self.assertTrue(scan['valid'].all())
        self.assertTrue(np.any(scan['j_target'] > 1))
        np.testing.assert_allclose(scan['F_E'], -3.)

    def test_axis_and_control_validation_before_any_trial(self):
        bad_grids = [([], [1.], [.5]), ([.2, .1], [1.], [.5]),
                     ([np.nan], [1.], [.5]), ([0.], [1.], [.5]),
                     ([.2], [-1.], [.5]), ([.2], [1.], [0.]),
                     ([.2], [1.], [1.]), ([.2], [1.], [.5, .5])]
        with patch.object(maps, 'solve_trial') as trial:
            for theta, jt, alpha in bad_grids:
                with self.subTest(grids=(theta, jt, alpha)), self.assertRaises(ValueError):
                    maps.scan_residuals(theta, jt, alpha, 1., 5., (0., 0.), constant_B())
            for controls in ({'n_points': 1}, {'n_points': 2.5}, {'max_step': 0.},
                             {'rtol': np.nan}, {'atol': -1.}):
                with self.subTest(controls=controls), self.assertRaises(ValueError):
                    maps.scan_residuals([.2], [1.], [.5], 1., 5., (0., 0.), **controls)
            for beta, lam in [(0., 5.), (1., -1.), (np.nan, 5.)]:
                with self.assertRaises(ValueError):
                    maps.scan_residuals([.2], [1.], [.5], beta, lam, (0., 0.))
            trial.assert_not_called()

    def test_geometry_lower_limit_is_respected(self):
        with patch.object(maps, 'solve_trial') as trial, self.assertRaises(ValueError):
            maps.scan_residuals([.2], [1.], [.1], 1., 5., (0., 0.),
                                quadratic_area_legacy(g_min=.2))
        trial.assert_not_called()

    def test_candidate_cube_requires_all_eight_corners_and_energy_sign(self):
        scan = self.scan(theta=(.1, .4), jt=(.6, 2.4), alpha=(.2, .8))
        self.assertEqual(maps.candidate_cells(scan), [(0, 0, 0)])
        scan['valid'][1, 1, 1] = False
        self.assertEqual(maps.candidate_cells(scan), [])
        scan['valid'][1, 1, 1] = True
        scan['F_E'][:] = 1.
        self.assertEqual(maps.candidate_cells(scan), [])

    def test_singleton_axes_have_no_candidate_cubes(self):
        scan = self.scan(alpha=(.5,))
        self.assertEqual(maps.candidate_cells(scan), [])
        self.assertTrue(maps.choose_starts(scan))

    def test_starts_are_valid_grid_points_and_do_not_merge_alpha_roots(self):
        def two_roots(*args, **kwargs):
            trial = manufactured_trial(*args, **kwargs)
            alpha = args[2]
            trial['omega'][0] = 1+(alpha-.2)*(alpha-.8)
            return trial
        scan = self.scan(provider=two_roots)
        starts = maps.choose_starts(scan, max_starts=2, separation=1)
        self.assertEqual({s['alpha'] for s in starts}, {.2, .8})
        for start in starts:
            self.assertEqual(start['theta'], .2)
            self.assertEqual(start['j_target'], 1.2)
            index = (start['alpha_index'], start['j_index'], start['theta_index'])
            self.assertTrue(scan['valid'][index])

    def test_start_separation_uses_all_three_grid_indices(self):
        scan = self.scan()
        starts = maps.choose_starts(scan, max_starts=20, separation=2)
        indices = [(s['alpha_index'], s['j_index'], s['theta_index']) for s in starts]
        self.assertEqual(len(indices), len(set(indices)))
        for k, index in enumerate(indices):
            for other in indices[:k]:
                self.assertGreaterEqual(max(abs(i-j) for i, j in zip(index, other)), 2)
        for controls in ({'max_starts': 0}, {'separation': 0}, {'max_starts': 1.5}):
            with self.assertRaises(ValueError):
                maps.choose_starts(scan, **controls)

    def test_projection_uses_full_norm_and_same_alpha_for_all_fields(self):
        scan = self.scan(alpha=(.2, .8))
        for key in maps._RESIDUALS:
            scan[key][:] = 0.
        scan['F_M'][0] = .4
        scan['F_J'][1] = .3
        scan['residual_norm'][0] = .4
        scan['residual_norm'][1] = .3
        view = maps.project_residuals(scan)
        np.testing.assert_array_equal(view['alpha_index'], 1)
        np.testing.assert_allclose(view['F_J'], .3)
        np.testing.assert_allclose(view['F_M'], 0.)
        np.testing.assert_allclose(view['residual_norm'], .3)
        np.testing.assert_allclose(view['alpha'], .8)
        np.testing.assert_array_equal(view['valid_alpha_count'], 2)

    def test_projection_invalid_columns_and_ties(self):
        scan = self.scan(alpha=(.2, .8))
        scan['valid'][:, 0, 0] = False
        scan['residual_norm'][:] = .4
        view = maps.project_residuals(scan)
        self.assertEqual(view['alpha_index'][0, 0], -1)
        self.assertTrue(np.isnan(view['alpha'][0, 0]))
        self.assertTrue(np.isnan(view['residual_norm'][0, 0]))
        self.assertEqual(view['alpha_index'][1, 1], 0)

    def test_fixed_slice_does_not_substitute_another_alpha(self):
        scan = self.scan()
        scan['valid'][0, 1, 1] = False
        view = maps.project_residuals(scan, alpha_index=0)
        self.assertEqual(view['mode'], 'fixed_alpha_slice')
        self.assertEqual(view['fixed_alpha'], .2)
        self.assertFalse(view['valid'][1, 1])
        np.testing.assert_allclose(view['F_E'][view['valid']], -.3)
        for index in (-1, 3, .5):
            with self.assertRaises(ValueError):
                maps.project_residuals(scan, alpha_index=index)

    def test_npz_full_cube_roundtrip_without_pickle_and_no_overwrite(self):
        scan = self.scan()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'map.npz'
            maps.save_residuals(path, scan, metadata={'test_only': True})
            with np.load(path, allow_pickle=False) as saved:
                for key in (*maps._FIELDS, 'theta', 'j_target', 'alpha', 'valid', 'status'):
                    np.testing.assert_array_equal(saved[key], scan[key])
                self.assertEqual(int(saved['format_version']), 1)
                self.assertEqual(json.loads(str(saved['summary_json'])), scan['summary'])
                self.assertEqual(json.loads(str(saved['settings_json'])), scan['settings'])
                self.assertEqual(json.loads(str(saved['metadata_json'])), {'test_only': True})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                maps.save_residuals(path, scan)
            self.assertEqual(path.read_bytes(), before)

    def test_non_json_metadata_fails_before_creating_file(self):
        scan = self.scan()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'map.npz'
            with self.assertRaises(ValueError):
                maps.save_residuals(path, scan, metadata={'bad': np.nan})
            self.assertFalse(path.exists())

    def test_long_status_not_truncated(self):
        status = 'unresolved_'+'x'*100
        scan = self.scan(provider=lambda *a, **kw: dict(status=status))
        self.assertEqual(scan['status'][0, 0, 0], status)

    def test_real_transport_interface_with_fabricated_integrator(self):
        # Exercise the actual solve_trial wrapper without taking an ODE step.
        from isothermal_model_adas import transport
        from isothermal_model_adas.tests.test_ionization_integration import fabricated_integrator
        with patch.object(transport, 'solve_ivp', side_effect=fabricated_integrator) as ivp:
            scan = maps.scan_residuals([.03, .05], [.6, .8], [.3, .7],
                1., 5., (0., 0.), constant_B(), D=1., n_points=12)
        self.assertEqual(ivp.call_count, 8)
        self.assertEqual(scan['summary']['statuses'], {'integrated': 8})
        self.assertTrue(scan['valid'].all())
        np.testing.assert_allclose(scan['F_J'], 0.)
        np.testing.assert_allclose(scan['F_E'], 0.)
        self.assertEqual(scan['settings']['n_points'], 12)
        # Having two zero residuals does NOT mark the trial as a solution.
        self.assertTrue(np.any(scan['residual_norm'] > 0))


class ResidualPlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import matplotlib
        matplotlib.use('Agg', force=True)

    def test_projection_slice_and_singleton_render(self):
        import matplotlib.pyplot as plt
        with patch.object(maps, 'solve_trial', side_effect=manufactured_trial):
            scan = maps.scan_residuals([.1, .2, .4], [.6, 1., 1.2, 2.4], [.2, .5, .8],
                                       1., 5., (0., 0.), constant_B())
        with tempfile.TemporaryDirectory() as folder:
            for alpha_index, j_axis in [(None, 'log'), (1, 'loss'), (0, 'linear')]:
                fig, axes = maps.plot_residuals(scan, 100., alpha_index=alpha_index,
                    j_axis=j_axis, starts=maps.choose_starts(scan, max_starts=2),
                    solutions=[dict(theta=.2, j_target=1.2, alpha=.5, stationary_verified=True)])
                self.assertEqual(axes.shape, (2, 3))
                path = Path(folder)/f'{j_axis}.png'
                fig.savefig(path, dpi=50)
                self.assertGreater(path.stat().st_size, 5000)
                plt.close(fig)
            with patch.object(maps, 'solve_trial', side_effect=manufactured_trial):
                single = maps.scan_residuals([.2], [1.2], [.5], 1., 5., (0., 0.), constant_B())
            fig, _ = maps.plot_residuals(single)
            fig.canvas.draw()
            plt.close(fig)

    def test_all_invalid_map_renders_without_fake_roots(self):
        import matplotlib.pyplot as plt
        with patch.object(maps, 'solve_trial', return_value=dict(status='adas_outside_domain')):
            scan = maps.scan_residuals([.1, .2], [.5, 1.], [.2, .8], 1., 5., (0., 0.))
        fig, _ = maps.plot_residuals(scan)
        fig.canvas.draw()
        plt.close(fig)
        with self.assertRaises(ValueError):
            maps.plot_residuals(scan, solutions=[dict(theta=.1, j_target=1., alpha=.5)])


if __name__ == '__main__':
    unittest.main()
