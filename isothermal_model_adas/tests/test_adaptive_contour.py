"""Analytic contour counts, narrow boundary features, caching and safe failures."""

import unittest
import numpy as np

from isothermal_model_adas.adaptive_contour import contour_count_adaptive,AdaptiveContourError
from isothermal_model_adas.spectrum import SpectralBox,contour_count,SearchUnresolved


def polynomial_matrix(roots):
    def at(s):
        return np.diag([np.prod([s-r for r in roots]),1.,1.]).astype(complex)
    return at


class AdaptiveTests(unittest.TestCase):
    def setUp(self):
        self.box=SpectralBox(0,1,-1,1)

    def test_no_roots_and_two_confirming_passes(self):
        count,h=contour_count_adaptive(polynomial_matrix([-.3]),self.box)
        self.assertEqual(count,0)
        self.assertGreaterEqual(len(h),2)
        self.assertLess(h[-1]["phase_limit"],h[0]["phase_limit"])
        self.assertLessEqual(h[-1]["max_phase_step"],h[-1]["phase_limit"])

    def test_real_and_complex_roots(self):
        for roots in ([.3],[.3+.4j,.3-.4j],[.2,.5,.7+.3j,.7-.3j]):
            with self.subTest(roots=roots):
                count,h=contour_count_adaptive(polynomial_matrix(roots),self.box)
                self.assertEqual(count,len(roots))

    def test_multiplicity_is_counted(self):
        self.assertEqual(contour_count_adaptive(polynomial_matrix([.3,.3]),self.box)[0],2)

    def test_root_very_close_to_boundary_requires_only_local_refinement(self):
        at=polynomial_matrix([1e-6+.317915j])
        with self.assertRaises(SearchUnresolved):
            contour_count(at,self.box,max_points=1024)
        count,h=contour_count_adaptive(at,self.box)
        self.assertEqual(count,1)
        self.assertGreater(h[-1]["max_depth_used"],5)
        self.assertLess(h[-1]["distinct_points"],4096)

    def test_shared_cache_reuses_every_matrix_evaluation(self):
        calls=[]
        at=polynomial_matrix([.3])
        def record(s):
            calls.append(s)
            return at(s)
        cache={}
        first=contour_count_adaptive(record,self.box,cache=cache)
        old=len(calls)
        second=contour_count_adaptive(record,self.box,cache=cache)
        self.assertEqual(first[0],second[0])
        self.assertEqual(len(calls),old)
        self.assertEqual(second[1][-1]["new_matrix_evaluations"],0)

    def test_root_on_contour_does_not_become_a_zero_count(self):
        with self.assertRaises(AdaptiveContourError) as ctx:
            contour_count_adaptive(polynomial_matrix([0j]),self.box)
        self.assertEqual(ctx.exception.diagnostic["failed_point"],[0.,0.])

    def test_budget_failure_has_coordinates(self):
        with self.assertRaises(AdaptiveContourError) as ctx:
            contour_count_adaptive(polynomial_matrix([.3]),self.box,max_points=16)
        detail=ctx.exception.diagnostic
        self.assertEqual(detail["distinct_points"],16)
        self.assertIsNotNone(detail["interval"])

    def test_depth_failure_does_not_claim_absence(self):
        with self.assertRaises(AdaptiveContourError) as ctx:
            contour_count_adaptive(polynomial_matrix([1e-9+.317915j]),self.box,max_depth=1)
        self.assertIn("max_phase_step",ctx.exception.diagnostic)

    def test_high_frequency_nonzero_exponential_factor_keeps_zero_count(self):
        def at(s):
            phase=np.exp(30j*s.imag)  # phase of the nonzero analytic factor exp(30*s)
            return np.diag([phase*(s-.3),1.,1.]).astype(complex)
        count,h=contour_count_adaptive(at,self.box)
        self.assertEqual(count,1)

    def test_nonfinite_matrix_failure_is_not_a_result(self):
        with self.assertRaises(AdaptiveContourError) as ctx:
            contour_count_adaptive(lambda s:np.full((3,3),np.nan),self.box)
        self.assertIn("failed_point",ctx.exception.diagnostic)

    def test_invalid_controls(self):
        for opts in (dict(max_points=2),dict(phase_limit=2),dict(max_rounds=1)):
            with self.assertRaises(ValueError):
                contour_count_adaptive(polynomial_matrix([.3]),self.box,**opts)


if __name__ == "__main__":
    unittest.main()
