"""Fast synthetic tests. Saved ADAS integrations are explicit CLI checks."""

from types import SimpleNamespace
import unittest
import numpy as np

from isothermal_model_adas.geometry import quadratic_area_legacy
from isothermal_model_adas.linearized_model import SavedBackground, LinearControls, LinearizedModel
from isothermal_model_adas.spectrum import SpectralBox, contour_count, find_modes
from isothermal_model_adas.transport import rhs_g


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.bg = object.__new__(SavedBackground)
        self.bg.beta = 1.2
        self.bg.geometry = quadratic_area_legacy()
        self.bg.p = dict(T_star_eV=40.)
        self.bg.rr = lambda t,n:(.03*np.asarray(n)**.2*(1+t), -.002+0*np.asarray(n))
        self.bg.processes = dict(
            D=lambda t: .5*np.asarray(t)**.2,
            ionization=lambda t,n:(.1*np.asarray(n)**.1*np.asarray(t)**.3,
                                   .034*np.asarray(n)**.1*np.asarray(t)**.3),
            excitation=lambda t,n: .02*np.asarray(t)**.4+0*np.asarray(n),
            molecular=lambda t,n:tuple(k*np.asarray(t)**p*np.asarray(n)**q for k,p,q in
                                       ((.1,.2,.1),(.2,.1,.3),(.3,.4,.2))),
            T_star_eV=40.)

    def test_atomic_and_molecular_rhs_agree_with_existing_transport(self):
        bg = self.bg
        theta,j,r = .03,.7,.2
        b = np.sqrt(8*bg.beta*theta)
        mu = b*j+r*r
        for molecular in (False,True):
            with self.subTest(molecular=molecular):
                processes = dict(bg.processes)
                if not molecular:
                    processes.pop("molecular")
                    bg.processes = processes
                expected = rhs_g(.8,(j,mu,.9),theta=theta,beta=bg.beta,
                    recombination=bg.rr,geometry=bg.geometry,
                    **dict(processes,D=processes["D"](theta)))
                expected[1] -= b*expected[0]
                np.testing.assert_allclose(bg.local(.8,j,r,theta),expected,rtol=1e-13)
                if not molecular:
                    self.setUp()
                    bg = self.bg

    def test_vectorized_sources_equal_scalar_calls(self):
        bg = self.bg
        g,j,r,t = np.array([.7,.8,.9]),np.array([.4,.6,.8]),np.array([.1,.2,.3]),np.array([.02,.03,.04])
        vector = bg.local(g,j,r,t)
        scalar = np.column_stack([bg.local(*args) for args in zip(g,j,r,t)])
        np.testing.assert_allclose(vector,scalar,rtol=1e-13)

    def test_storage_includes_ion_and_electron_internal_energy(self):
        j,r,t,beta = .8,.2,.03,self.bg.beta
        stock = self.bg.storage(j,r,t)
        self.assertAlmostEqual(stock[1],j)
        v = j/stock[0]
        self.assertAlmostEqual(stock[2],stock[0]*(3*t+v*v/(2*beta)))

    def test_kinetic_throat_normalization(self):
        bg = self.bg
        from isothermal_model_adas.geometry import constant_B
        bg.geometry = constant_B()
        bg.alpha = .5
        bg.p.update(chi=0.,n_c_cm3=10.,n_star_cm3=2.,T_c_eV=20.,T_star_eV=40.)
        np.testing.assert_allclose(bg.kinetic_storage(),[2.5,1.,3.75],rtol=1e-10)

    def test_controls_validate(self):
        for kwargs in (dict(points=5),dict(derivative_step=0),dict(max_step=2)):
            with self.assertRaises(ValueError):
                LinearControls(**kwargs)

    def test_static_check_refines_locally_without_stationary_search(self):
        # This module deliberately never imports solve_stationary.
        self.assertNotIn("solve_stationary", LinearizedModel.check_static.__code__.co_names)


class FakeControls:
    def refined(self):
        return self


class AnalyticModel:
    def __init__(self,bg,controls=None):
        self.bg,self.controls = bg,controls or FakeControls()

    def boundary_matrix(self,s):
        # Contains modes whose temperature component is identically zero.
        value = np.prod([s-r for r in self.bg.roots])
        return np.diag([value,1.,1.]).astype(complex)

    def mode_diagnostics(self,s,amplitude):
        return dict(passed=True, analytic_test=True)


class SpectrumTests(unittest.TestCase):
    def model(self,roots):
        return AnalyticModel(SimpleNamespace(roots=roots,time_star=.1))

    def test_count_and_positive_real_mode_with_zero_temperature(self):
        model = self.model([.3])
        box = SpectralBox(0,1,-1,1)
        self.assertEqual(contour_count(model.boundary_matrix,box)[0],1)
        result = find_modes(model,box)
        self.assertEqual(result["status"],"unstable")
        self.assertAlmostEqual(result["modes"][0]["s_per_s"][0],.3,places=7)
        self.assertAlmostEqual(result["modes"][0]["amplitudes"][2][0],0)
        self.assertFalse(result["global_stability_proven"])

    def test_zero_count_is_not_global_stability(self):
        result = find_modes(self.model([-.3]),SpectralBox(0,1,-1,1))
        self.assertEqual(result["status"],"no_growing_modes_found_in_box")
        self.assertEqual(result["root_count"],0)

    def test_complex_conjugate_pair(self):
        result = find_modes(self.model([.3+.4j,.3-.4j]),SpectralBox(0,1,-1,1))
        self.assertEqual(result["root_count"],2)
        actual = sorted(complex(*r["s_per_s"]).imag for r in result["modes"])
        np.testing.assert_allclose(actual,[-.4,.4],atol=1e-7)

    def test_invalid_box(self):
        with self.assertRaises(ValueError):
            SpectralBox(1,0,-1,1)


if __name__ == "__main__":
    unittest.main()
