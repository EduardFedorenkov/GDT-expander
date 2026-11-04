"""Hybrid configuration uses local ACD/PRB and freezes legacy neutral rates."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from isothermal_cx_model import cx, excitation, ionization
from isothermal_model_adas import atomic_coefficients as atomic
from isothermal_model_adas import scan_lambda, scan_stationary, transport
from isothermal_model_adas.config import load_config
from isothermal_model_adas.recombination_adas import ACD_PATH, PRB_PATH
from isothermal_model_adas.sources import TemperatureOnly


class HybridTests(unittest.TestCase):
    def setUp(self):
        self.p = load_config(Path(atomic.__file__).with_name('config.json'))
        self.theta = .5/self.p['T_star_eV']

    def test_default_mode_and_code_fingerprint(self):
        for cli in (scan_stationary, scan_lambda):
            self.assertEqual(cli.build_parser().parse_args([]).atomic_model, 'hybrid')
        files = set(atomic.code_hashes('hybrid')) - set(atomic.code_hashes('full-adas'))
        self.assertEqual(files, {f'isothermal_cx_model/{name}.py'
                                 for name in ('cx', 'ionization', 'excitation')})

    @unittest.skipUnless(ACD_PATH.exists() and PRB_PATH.exists(), 'Requires ACD and PRB.')
    def test_hybrid_reads_only_acd_prb_and_reuses_legacy_coefficients(self):
        with patch.multiple(atomic, ADASCX=object(), ADASIonization=object(), ADASExcitation=object()):
            rr, processes, tables, paths = atomic.build_atomic_coefficients(self.p, 'hybrid', 1., .7, .8)
        self.assertEqual(len(tables), 1)
        self.assertEqual(paths, [ACD_PATH, PRB_PATH])
        T = self.theta*self.p['T_star_eV']
        ratio = self.p['L_cm']*self.p['n0_cm3']/self.p['U_star_cm_per_s']
        self.assertAlmostEqual(processes['D'](self.theta)/(ratio*cx.k_cx(T, 0.)), 1.)
        for N in (.1, 1., 10.):
            self.assertAlmostEqual(processes['ionization'](self.theta, N)[0] /
                                   (.7*ratio*ionization.k_ion(T)), 1.)
            self.assertAlmostEqual(processes['excitation'](self.theta, N) /
                                   (.8*ratio*excitation.L_exc(T)/self.p['T_star_erg']), 1.)
        R, Q = rr(self.theta, np.array([.1, 10.]))
        self.assertGreater(R[1], R[0])
        self.assertNotEqual(Q[0], Q[1])

    def test_neutral_control_without_tables(self):
        rr, providers, tables, paths = atomic.build_atomic_coefficients(
            dict(self.p, n0_cm3=0.), 'hybrid', 0., 1., 1.)
        self.assertEqual((rr, providers, tables, paths),
                         ((0., 0.), dict(D=0., ionization=(0., 0.), excitation=0.), [], []))

    def test_trial_freezes_ionization_and_excitation_before_integration(self):
        calls = []

        def evaluate(value):
            def provider(theta):
                calls.append(theta)
                return value
            return provider

        def integrate(rhs, span, wall, **kwargs):
            # A real wall RHS, called twice; density-dependent lookups here would fail.
            rhs(1., wall)
            rhs(1., wall)
            return SimpleNamespace(t=[1.], success=False, message='test stop')

        ion = TemperatureOnly(evaluate((.001, .0001)))
        exc = TemperatureOnly(evaluate(.0001))
        with patch.object(TemperatureOnly, '__call__', side_effect=AssertionError('not frozen')), \
             patch.object(transport, 'solve_ivp', side_effect=integrate):
            result = transport.solve_trial(.05, 1., .8, 1., 7., (0., 0.), D=1.,
                                           ionization=ion, excitation=exc)
        self.assertEqual(result['status'], 'integration_failed')
        self.assertEqual(calls, [.05, .05])
