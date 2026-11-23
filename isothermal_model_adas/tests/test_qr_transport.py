"""QR phase, determinant, reconstruction and overflow-regression checks."""

import unittest
from types import SimpleNamespace
import numpy as np
from scipy.linalg import expm

from isothermal_model_adas.qr_transport import positive_qr,propagate_basis,reconstruct_mode
from isothermal_model_adas.linearized_model import LinearControls
from isothermal_model_adas.spectrum import SpectralBox,find_modes


class QRTests(unittest.TestCase):
    def test_qr_diagonal_is_positive_and_factorization_preserved(self):
        V = np.array([[1+2j,3j],[.2-1j,.5],[1.,2.],[0.,1.]])
        Q,R = positive_qr(V)
        np.testing.assert_allclose(Q@R,V,atol=2e-15)
        np.testing.assert_allclose(Q.conjugate().T@Q,np.eye(2),atol=1e-15)
        self.assertTrue(np.all(np.diag(R).real>0))
        np.testing.assert_allclose(np.diag(R).imag,0)

    def moderate_problem(self,retain=False):
        A = np.array([[2+.3j,.1,.5,.2],[.2,-1+.1j,.3,.1],
                      [.4,.2,.1,.6],[0,0,0,0]],complex)
        V = np.array([[1,0],[0,0],[1,1],[0,1]],complex)
        return A,V,propagate_basis(lambda x:A,V,max_step=.05,retain_history=retain)

    def test_determinant_magnitude_and_phase_match_exact_transfer(self):
        A,V,result = self.moderate_problem()
        C = np.array([[1,0,0,0],[.5,1,0,.3],[0,0,1,0]])
        d = np.array([.2,-.1,.4])
        exact = C@expm(A)@V
        qr = C@result.basis
        direct_det = np.linalg.det(np.column_stack((exact[:,0],d,exact[:,1])))
        qr_det = np.linalg.det(np.column_stack((qr[:,0],d,qr[:,1])))
        np.testing.assert_allclose(qr_det*np.exp(result.log_volume),direct_det,rtol=3e-10)
        self.assertLess(result.orthogonality_error,1e-13)

    def test_exponential_overflow_is_not_reconstructed(self):
        A = np.diag([1000.+40j,500.-7j,0.,0.])
        V = np.eye(4,2,dtype=complex)
        result = propagate_basis(lambda x:A,V)
        self.assertTrue(np.isfinite(result.basis).all())
        self.assertAlmostEqual(result.log_volume,1500.,places=5)
        np.testing.assert_allclose(result.basis[:2,:],np.diag(np.exp(1j*np.array([40.,-7.]))),atol=2e-8)
        self.assertGreater(result.segments,100)

    def test_independent_columns_do_not_merge_under_differential_growth(self):
        A = np.diag([1000.,500.,0.,0.])
        V = np.array([[1,1],[1,-1],[0,0],[0,0]],complex)
        result = propagate_basis(lambda x:A,V)
        np.testing.assert_allclose(result.basis.conjugate().T@result.basis,np.eye(2),atol=1e-13)
        self.assertAlmostEqual(result.log_volume,1500.+np.log(2.),places=5)

    def test_backward_reconstruction_agrees_with_exact_solution(self):
        A,V,result = self.moderate_problem(retain=True)
        coeff = np.array([.3+.1j,.7-.2j])
        eta = .2
        grid = np.linspace(0,1,151)
        mode,new_eta = reconstruct_mode(result,coeff,eta,grid)
        initial_coeff = np.linalg.lstsq(expm(A)@V,result.basis@coeff,rcond=None)[0]
        expected = np.column_stack([expm(A*x)@V@initial_coeff for x in grid])
        np.testing.assert_allclose(mode,expected*(new_eta/eta),rtol=2e-9,atol=2e-11)
        np.testing.assert_allclose(mode[3],mode[3,0],atol=1e-11)

    def test_segment_budget_has_explicit_failure(self):
        with self.assertRaisesRegex(RuntimeError,"segment budget"):
            propagate_basis(lambda x:np.eye(4),np.eye(4,2),max_step=.01,max_segments=1)

    def test_last_roundoff_sized_interval_finishes(self):
        result=propagate_basis(lambda x:np.diag([.7,1.1,0.,0.]),
                               np.eye(4,2),max_step=.0125)
        self.assertAlmostEqual(result.log_volume,1.8,places=10)

    def test_controls_refine_qr_and_keep_budget(self):
        control = LinearControls(qr_max_step=.04,qr_max_growth=3.,qr_max_segments=100)
        refined = control.refined()
        self.assertEqual(refined.qr_max_step,.02)
        self.assertEqual(refined.qr_max_growth,1.5)
        self.assertEqual(refined.qr_max_segments,200)
        with self.assertRaises(ValueError):
            LinearControls(qr_max_growth=0)

    def test_rank_zero_column_rejected(self):
        with self.assertRaises(ValueError):
            positive_qr(np.zeros((4,2)))

    def test_qr_backend_keeps_temperature_free_modes(self):
        class Controls:
            def refined(self): return self
        class Model:
            def __init__(self,bg,controls=None):
                self.bg,self.controls=bg,controls or Controls()
            def qr_boundary_matrix(self,s):
                return np.diag([s-.3,1.,1.]).astype(complex)
            def qr_mode_diagnostics(self,s,coordinates):
                return coordinates,dict(passed=True)
        result=find_modes(Model(SimpleNamespace(time_star=1.)),SpectralBox(0,1,-1,1),method="qr")
        self.assertEqual(result["transport"],"qr")
        self.assertEqual(result["status"],"unstable")
        self.assertAlmostEqual(result["modes"][0]["s_per_s"][0],.3)
        self.assertAlmostEqual(result["modes"][0]["amplitudes"][2][0],0.)


if __name__ == "__main__":
    unittest.main()
