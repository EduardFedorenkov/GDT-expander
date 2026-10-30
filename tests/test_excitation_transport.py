"""Independent CX/excitation limits of the current variable-J transport solver."""

import unittest
import numpy as np
from scipy.optimize import brentq

from isothermal_cx_model.geometry import constant_B
from isothermal_cx_model.transport import solve_trial
from isothermal_cx_model.stationary import solve_stationary


def analytic_omega(mu, b, beta, D, lam, excitation):
    """Constant area: integrate d(omega)/d(mu) independently of ODE code."""
    theta = b*b/(8*beta)
    mu = np.asarray(mu)
    radical = np.sqrt(np.maximum(mu*mu-b*b, 0))
    integral_radical = .5*(mu*radical-b*b*np.log((mu+radical)/b))
    integral_mu = .5*(mu*mu-b*b)
    return (lam*theta + (5*integral_mu+integral_radical)/(8*beta)
            + excitation*(integral_mu+integral_radical)/(4*beta*theta*D))


class ExcitationTransportTests(unittest.TestCase):
    def test_constant_area_independent_energy_integral(self):
        theta, beta, D, lam, E = .1, .5, 4., 5., .02
        state = solve_trial(theta, 1., beta, D, lam, constant_B(),
                            E_exc=E, n_points=1000, rtol=1e-11, atol=1e-13)
        self.assertEqual(state["status"], "energy_matched")
        mu = state["b"] + D*(1-state["g"])
        np.testing.assert_allclose(state["j"], 1., atol=1e-14)
        np.testing.assert_allclose(state["mu"], mu, atol=1e-10)
        np.testing.assert_allclose(state["omega"],
            analytic_omega(mu, state["b"], beta, D, lam, E), atol=3e-9)

    def test_excitation_moves_energy_matching_front_towards_wall(self):
        states = [solve_trial(.1, 1., .5, 4., 5., constant_B(), E_exc=E)
                  for E in (0., .01, .02)]
        self.assertTrue(all(s["status"] == "energy_matched" for s in states))
        self.assertTrue(np.all(np.diff([s["alpha"] for s in states]) > 0))

    def test_coefficient_is_evaluated_once_per_trial(self):
        calls = []
        state = solve_trial(.1, 1., .5, 4., 5., constant_B(),
                            E_exc=lambda t: calls.append(t) or .02)
        self.assertEqual(state["status"], "energy_matched")
        self.assertEqual(calls, [.1])
        self.assertEqual(state["E_exc"], .02)

    def test_stationary_root_matches_independent_energy_integral(self):
        beta, D, lam = .5, 4., 5.
        E = lambda theta: .03*theta
        expected_b = brentq(lambda b:
            analytic_omega(1., b, beta, D, lam, E(b*b/(8*beta)))-1, .1, .89)
        result = solve_stationary(expected_b**2/(8*beta)*1.01, .999,
            beta, D, lam, constant_B(), E_exc=E, n_points=4000)
        self.assertEqual(result["status"], "converged")
        state = result["solution"]
        self.assertAlmostEqual(state["b"], expected_b, delta=1e-7)
        self.assertAlmostEqual(state["j_target"], 1., delta=1e-9)
        self.assertEqual(state["E_exc"], E(state["theta"]))


if __name__ == "__main__":
    unittest.main()
