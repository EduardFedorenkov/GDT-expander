"""Numerical verification of stationary and mapping APIs with AMJUEL sources."""

import unittest

import numpy as np
from scipy.optimize import brentq

from isothermal_model_adas.constants import EV_TO_ERG, M_D_G
from isothermal_model_adas.geometry import constant_B
from isothermal_model_adas.molecular_amjuel import make_molecular_coefficients
from isothermal_model_adas.profiles import dimensional_profiles
from isothermal_model_adas.residual_map import scan_residuals
from isothermal_model_adas.stationary import solve_stationary
from isothermal_model_adas.transport import solve_trial


class MolecularStationaryTests(unittest.TestCase):
    def test_real_amjuel_map_and_verified_constructed_root(self):
        energy = 20.*EV_TO_ERG
        u_star = np.sqrt(energy/M_D_G)  # beta=1
        J0 = 1e13*u_star  # A_star=1 cm^2
        p = dict(T_star_eV=20., L_cm=100., U_star_cm_per_s=u_star, n_star_cm3=1e13,
                 n0_cm3=1e13, molecular_fraction=.3, beta=1., A_star_cm2=1.,
                 J0_per_s=J0, M0_dyn=M_D_G*u_star*J0, W0_erg_per_s=energy*J0)
        molecular = make_molecular_coefficients(p, scale=1.)
        theta, alpha, beta = .25, .85, p['beta']
        geometry = constant_B()
        sources = dict(molecular=molecular, T_star_eV=p['T_star_eV'])
        rr = (0., 0.)

        def fixed_trial(jt):
            result = solve_trial(theta, jt, alpha, beta, 8., rr, geometry,
                rtol=1e-12, atol=1e-14, max_step=.005, **sources)
            self.assertEqual(result['status'], 'integrated')
            return result

        jt = brentq(lambda jt: fixed_trial(jt)['R_J'], .9, 1., xtol=1e-13)
        reference = fixed_trial(jt)
        # J and M do not depend on W; changing lambda shifts W by a constant.
        lambda_sh = 8.-reference['R_E']/(theta*jt)
        inlet = lambda _alpha: reference['mu'][0]
        scan = scan_residuals([theta], [jt], [alpha], beta, lambda_sh, rr, geometry,
            inlet_momentum=inlet, **sources)
        self.assertTrue(scan['valid'].item())
        self.assertLess(scan['residual_norm'].item(), 1e-7)
        outcome = solve_stationary(theta, jt, alpha, beta, lambda_sh, rr, geometry,
            inlet_momentum=inlet, n_points=1501, **sources)
        self.assertEqual(outcome['status'], 'converged', outcome['diagnostic'])
        solution = outcome['solution']
        self.assertTrue(solution['stationary_verified'])
        self.assertGreater(solution['balances']['mad_energy_loss_integral'], 0.)
        self.assertLess(solution['balances']['energy_abs'], 1e-7)
        profile = dimensional_profiles(solution, p)
        self.assertGreater(np.ptp(profile['D_mad']), 0.)
        self.assertEqual(profile['D_mar'].shape, profile['n_cm3'].shape)
        self.assertEqual(profile['D_mai'].shape, profile['n_cm3'].shape)


if __name__ == '__main__':
    unittest.main()
