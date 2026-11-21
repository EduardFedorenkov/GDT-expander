"""Checks of units, per-profile maxima, default bounds, and CLI dispatch."""

from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from isothermal_model_adas.reference_frequency import reference_frequency
from isothermal_model_adas import check_stability as cli


def sample():
    p = dict(L_cm=100.,U_star_cm_per_s=10.,beta=1.,molecular_fraction=.3)
    state = dict(status="integrated",alpha=.5,theta=.1,g=np.array([.5,1.]),
        n_over_n_star=np.array([1.,4.]),D=2.,D_ion_profile=np.array([.3,.5]),
        R_profile=np.array([.4,.3]),D_mar_profile=np.array([.2,.3]),
        D_mad_profile=np.array([3.,4.]),D_mai_profile=np.array([.1,.2]))
    return p,state


class FrequencyTests(unittest.TestCase):
    def test_reaction_units_maximum_and_molecular_dominance(self):
        p,state = sample()
        out = reference_frequency(p,state)
        self.assertAlmostEqual(out["rates"]["cx"]["max_per_s"],.2)
        self.assertAlmostEqual(out["rates"]["recombination"]["max_per_s"],.12)
        self.assertEqual(out["rates"]["recombination"]["z_at_max_cm"],100.)
        self.assertEqual(out["dominant_process"],"mad")
        self.assertAlmostEqual(out["nu_ref_per_s"],.4)
        self.assertAlmostEqual(out["s_real_max_per_s"],4.)
        self.assertTrue(out["heuristic_not_proven_spectral_bound"])

    def test_acoustic_scale_and_no_double_count_of_composition(self):
        p,state = sample()
        out = reference_frequency(p,state)
        self.assertAlmostEqual(out["c_s_cm_per_s"],10*np.sqrt(.2))
        self.assertAlmostEqual(out["rates"]["acoustic"]["max_per_s"],10*np.sqrt(.2)/50)
        p["molecular_fraction"] = .9
        self.assertEqual(reference_frequency(p,state)["nu_ref_per_s"],out["nu_ref_per_s"])

    def test_no_molecules_is_valid_and_factor_configurable(self):
        p,state = sample()
        p["molecular_fraction"] = 0.
        for key in ("D_mar_profile","D_mad_profile","D_mai_profile"):
            state.pop(key)
        out = reference_frequency(p,state,5.)
        self.assertEqual(out["dominant_process"],"cx")
        self.assertEqual(out["rates"]["mar"]["max_per_s"],0.)
        self.assertAlmostEqual(out["s_real_max_per_s"],1.)

    def test_acoustic_dominance(self):
        p,state = sample()
        for key in ("D","D_ion_profile","R_profile","D_mar_profile","D_mad_profile","D_mai_profile"):
            state[key] = 0.
        self.assertEqual(reference_frequency(p,state)["dominant_process"],"acoustic")

    def test_rejects_missing_molecular_channel(self):
        p,state = sample()
        del state["D_mad_profile"]
        with self.assertRaises(ValueError):
            reference_frequency(p,state)

    def test_rejects_bad_profile_and_multiplier(self):
        for factor in (0.,-1.,float("nan"),float("inf")):
            with self.subTest(factor=factor):
                with self.assertRaises(ValueError):
                    reference_frequency(*sample(),factor)
        p,state = sample()
        state["n_over_n_star"][0] = 0
        with self.assertRaises(ValueError):
            reference_frequency(p,state)

    def test_manual_and_automatic_bounds(self):
        freq = {"s_real_max_per_s":4.}
        box = cli.select_box(0,None,None,freq)
        self.assertEqual(vars(box),dict(real_min=0,real_max=4.,imag_min=-4.,imag_max=4.))
        box = cli.select_box(-1,3,2,freq)
        self.assertEqual(vars(box),dict(real_min=-1,real_max=3,imag_min=-2,imag_max=2))
        self.assertEqual(cli.select_box(0,None,7,freq).real_max,4.)

    def test_frequency_only_never_builds_linear_or_spectral_model(self):
        p,state = sample()
        p.update(molecular_fraction=.3)
        bg = SimpleNamespace(
            p=p,case_id="test",root_index=0,seed={"T_eV":2.},
            alpha=.5,report_path="synthetic.json",verify=lambda:{},trial=lambda:state)
        stdout = StringIO()
        with patch.object(cli,"saved_cases",return_value=[bg]), \
             patch.object(cli,"LinearizedModel",side_effect=AssertionError("No linear solve")), \
             patch.object(cli,"find_modes",side_effect=AssertionError("No spectral solve")), \
             redirect_stdout(stdout):
            code = cli.main(["--scan","unused","--n0","1e14","--lambda-sh","7","--frequency-only"])
        self.assertEqual(code,0)
        self.assertIn("frequency_computed_spectrum_not_searched",stdout.getvalue())

    def test_default_dispatch_passes_frequency_box_to_search(self):
        p,state = sample()
        bg = SimpleNamespace(
            p=p,case_id="test",root_index=0,seed={"T_eV":2.},
            alpha=.5,report_path="synthetic.json",verify=lambda:{},trial=lambda:state)
        fake_model = SimpleNamespace(check_static=lambda:dict(passed=True))
        with patch.object(cli,"saved_cases",return_value=[bg]), \
             patch.object(cli,"LinearizedModel",return_value=fake_model), \
             patch.object(cli,"find_modes",return_value=dict(status="no_growing_modes_found_in_box")) as search, \
             redirect_stdout(StringIO()):
            code = cli.main(["--scan","unused","--n0","1e14","--lambda-sh","7"])
        self.assertEqual(code,0)
        box = search.call_args.args[1]
        self.assertEqual(box.real_max,4.)
        self.assertEqual(box.imag_max,4.)

    def test_static_only_still_checks_linearization_without_spectrum(self):
        p,state = sample()
        bg = SimpleNamespace(
            p=p,case_id="test",root_index=0,seed={"T_eV":2.},
            alpha=.5,report_path="synthetic.json",verify=lambda:{},trial=lambda:state)
        static = unittest.mock.Mock(return_value=dict(passed=True))
        with patch.object(cli,"saved_cases",return_value=[bg]), \
             patch.object(cli,"LinearizedModel",return_value=SimpleNamespace(check_static=static)), \
             patch.object(cli,"find_modes",side_effect=AssertionError("No spectrum")), \
             redirect_stdout(StringIO()):
            code = cli.main(["--scan","unused","--n0","1e14","--lambda-sh","7","--static-only"])
        self.assertEqual(code,0)
        static.assert_called_once()


if __name__ == "__main__":
    unittest.main()
