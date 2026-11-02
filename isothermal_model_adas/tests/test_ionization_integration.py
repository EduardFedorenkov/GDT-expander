"""Algebra and mocked plumbing ONLY; no physical IVP/BVP or scan is run."""

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.integrate import simpson
from isothermal_model_adas import transport as tr
from isothermal_model_adas import stationary as stationary
from isothermal_model_adas import run_stationary as cli
from isothermal_model_adas.geometry import constant_B, quadratic_area_legacy
from isothermal_model_adas.sources import adas_sources, ionization_at
from isothermal_model_adas.profiles import dimensional_profiles
from isothermal_model_adas.recombination_adas import ADASDomainError
from isothermal_model_adas.ionization_adas import SCD_PATH
from isothermal_model_adas.cx_adas import CCD_PATH


PACKAGE = Path(__file__).parents[1]
OLD_SOURCES = PACKAGE.parent/'isothermal_cx_model'/'sources.py'


def fabricated_integrator(rhs, interval, initial, **kwargs):
    """Fabricated arrays for forwarding/validation tests, NOT an ODE solution."""
    jt, mu_t, wt = initial
    b = mu_t/jt

    def profile(xi):
        xi = np.asarray(xi)
        j = 1+(jt-1)*xi
        mu = b*j+.5*(1-xi)
        w = 1+(wt-1)*xi
        return np.array([j, mu, w])

    # Evaluate one RHS to verify local coefficients and xi scaling, no step.
    rhs(.5, profile(.5))
    return SimpleNamespace(t=[1., 0.], success=True, t_events=[np.array([])], sol=profile)


class SourceTests(unittest.TestCase):
    def test_particles_and_energy_but_no_momentum(self):
        args = (.8, 3., .2, 2.)
        kw = dict(theta=.05, beta=1., R=.01, Q_eff=-.001, D=.4)
        before = np.array(adas_sources(*args, **kw))
        after = np.array(adas_sources(*args, **kw, D_ion=.3, E_ion=.04))
        np.testing.assert_allclose(after-before, [.3*2*3, 0., -.04*2*3], atol=1e-14)

    def test_off_preserves_exact_sources(self):
        args = (.8, 3., .2, 2.)
        kw = dict(theta=.05, beta=1., R=.01, Q_eff=-.001, D=.4)
        self.assertEqual(adas_sources(*args, **kw), adas_sources(*args, **kw, D_ion=0., E_ion=0.))

    def test_validates_ionization_pair(self):
        for pair in ((-1., 0.), (1., -.1), (0., 1.), (np.nan, 0.)):
            with self.assertRaises(ValueError):
                ionization_at(pair, .05, 1.)
        D, E = ionization_at(lambda t,n: (.1*np.asarray(n), .01*np.asarray(n)), .05, np.array([1.,2.]))
        np.testing.assert_allclose(D, [.1,.2])
        np.testing.assert_allclose(E, [.01,.02])

    def test_chemical_energy_identity_with_both_reactions(self):
        R, P, I, D_ion = .02, .003, .1, .5
        j, N, v, A, theta, beta = .8, 2., .2, 3., .05, 1.
        jg, _, wg = adas_sources(j, N, v, A, theta=theta, beta=beta,
            R=R, Q_eff=P-I*R, D_ion=D_ion, E_ion=I*D_ion)
        self.assertAlmostEqual(wg+I*jg, -A*N*N*(R*(v*v/(2*beta)+1.5*theta)+P))

    @unittest.skipUnless(OLD_SOURCES.exists(), 'Old source assembler unavailable.')
    def test_matches_old_algebra_for_100_states(self):
        spec = importlib.util.spec_from_file_location('old_sources_with_ionization', OLD_SOURCES)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        rng = np.random.default_rng(19)
        for _ in range(100):
            beta, theta = 10**rng.uniform(-1,1), 10**rng.uniform(-3,-1)
            j, A = rng.uniform(.01,3.), 10**rng.uniform(0,3)
            v = rng.uniform(.05,.95)*np.sqrt(2*beta*theta)
            mu = j*(v+2*beta*theta/v)
            radical = np.sqrt(mu*mu-8*beta*theta*j*j)
            H, N = mu+radical, (mu+radical)/(4*beta*theta*A)
            D, R, Q, Di, Ei = 1.2, .02, -.001, .3, .04
            expected = old.atomic_sources(j,mu,radical,H,A,theta=theta,beta=beta,
                D=D,R=R,Q=Q,E_exc=0.,D_ion=Di,E_ion=Ei).rhs(0.)
            actual = adas_sources(j,N,v,A,theta=theta,beta=beta,D=D,R=R,Q_eff=Q,D_ion=Di,E_ion=Ei)
            np.testing.assert_allclose(actual,expected,rtol=5e-13,atol=1e-13)

    def test_wall_minus_two_Dion_matches_RHS(self):
        theta, beta, jt, R, D, Di = .05, 1.2, 2., .01, 2., .3
        b = np.sqrt(8*beta*theta)
        geometry = quadratic_area_legacy()
        derivative = tr.rhs_g(1., [jt,b*jt,.4],theta=theta,beta=beta,
            recombination=(R,-.001),geometry=geometry,D=D,ionization=(Di,.04))
        actual,_ = tr.wall_slope_per_j(theta,jt,beta,R,D,geometry,D_ion_t=Di)
        before,_ = tr.wall_slope_per_j(theta,jt,beta,R,D,geometry)
        self.assertAlmostEqual(actual,(derivative[1]-b*derivative[0])/jt)
        self.assertAlmostEqual(actual-before,-2*Di)

    def test_independent_budgets(self):
        g = np.linspace(.4,1.,101)
        H = 3-.3*g
        N = H/(4*.05)
        state = dict(g=g,j=1-.1*g,mu=2-.2*g,omega=1-.3*g,H=H,
            A_over_A_star=np.ones_like(g),n_over_n_star=N,u_over_u_star=.2*np.ones_like(g),
            theta=.05,beta=1.,geometry=constant_B(),D=.4)
        with patch.object(tr,'adas_sources',side_effect=AssertionError('Independent budgets required')):
            before = tr.balance_checks(state,(.01,-.001))
            after = tr.balance_checks(state,(.01,-.001),(.3,.04))
        self.assertEqual(after['momentum_abs'],before['momentum_abs'])
        self.assertAlmostEqual(after['particle_ionization_integral'],float(simpson(.3*N,x=g)))
        self.assertAlmostEqual(after['ionization_energy_loss_integral'],float(simpson(.04*N,x=g)))
        self.assertAlmostEqual(after['energy_loss_integral']-before['energy_loss_integral'],float(simpson(.04*N,x=g)))
        state['D_ion_profile'] = np.ones_like(g)
        with self.assertRaisesRegex(ValueError,'ionization provider'):
            tr.balance_checks(state,(.01,-.001))


class PlumbingTests(unittest.TestCase):
    def test_increasing_J_and_target_above_one_not_rejected(self):
        with patch.object(tr,'solve_ivp',side_effect=fabricated_integrator):
            state = tr.solve_trial(.05,2.,.7,1.,8.,(0.,0.),constant_B(),ionization=(.5,.068),n_points=101)
        self.assertEqual(state['status'],'integrated')
        self.assertTrue(np.all(np.diff(state['j'])>0))
        self.assertEqual(state['j_target'],2.)
        self.assertEqual(state['D_ion_profile'].shape,(101,))

    def test_old_off_case_still_rejects_fabricated_J_growth(self):
        with patch.object(tr,'solve_ivp',side_effect=fabricated_integrator):
            state = tr.solve_trial(.05,2.,.7,1.,8.,(0.,0.),constant_B(),D=2.,n_points=101)
        self.assertEqual(state['status'],'profile_validation_failed')

    def test_local_callback_called_at_wall_RHS_and_profile(self):
        densities=[]
        def provider(theta,N):
            densities.append(np.asarray(N).copy())
            return .01*np.asarray(N),.001*np.asarray(N)
        with patch.object(tr,'solve_ivp',side_effect=fabricated_integrator):
            state=tr.solve_trial(.05,2.,.7,1.,8.,(0.,0.),constant_B(),ionization=provider,n_points=101)
        self.assertEqual(state['status'],'integrated')
        self.assertTrue(any(n.ndim==0 for n in densities))
        self.assertTrue(any(n.shape==(101,) for n in densities))
        self.assertGreater(np.ptp(state['D_ion_profile']),0.)
        np.testing.assert_allclose(state['D_ion_profile'],.01*state['n_over_n_star'])

    def test_SCD_domain_failures_at_wall_and_inside_RHS(self):
        def unavailable(theta,N):
            raise ADASDomainError('SCD outside table')
        with patch.object(tr,'solve_ivp',side_effect=AssertionError('Must not integrate')):
            result=tr.solve_trial(.05,2.,.7,1.,8.,(0.,0.),constant_B(),ionization=unavailable)
        self.assertEqual(result['status'],'adas_outside_domain')
        count=[0]
        def later_unavailable(theta,N):
            count[0]+=1
            if count[0]>1:
                raise ADASDomainError('SCD provisional stage outside table')
            return .5,.068
        with patch.object(tr,'solve_ivp',side_effect=fabricated_integrator):
            result=tr.solve_trial(.05,2.,.7,1.,8.,(0.,0.),constant_B(),ionization=later_unavailable)
        self.assertEqual(result['status'],'adas_outside_domain')

    def test_no_artificial_default_j_upper_bound(self):
        with patch.object(stationary,'solve_trial',return_value={'status':'mock_not_integrated'}) as trial:
            result=stationary.solve_stationary(.05,2.,.7,1.,8.,(0.,0.),ionization=(.5,.068))
        self.assertEqual(result['status'],'invalid_initial_trial')
        self.assertEqual(trial.call_count,1)
        self.assertEqual(trial.call_args.kwargs['ionization'],(.5,.068))

    def test_explicit_search_upper_bound_still_honored(self):
        with patch.object(stationary,'solve_trial',side_effect=AssertionError('Outside bounds')):
            result=stationary.solve_stationary(.05,2.,.7,1.,8.,(0.,0.),j_bounds=(1e-12,1.))
        self.assertEqual(result['status'],'initial_outside_search_bounds')

    def test_stationary_forwards_provider_to_verification_and_budgets(self):
        tree=ast.parse((PACKAGE/'stationary.py').read_text(encoding='utf-8-sig'))
        calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)]
        for call in [node for node in calls if node.func.id=='solve_trial']:
            self.assertTrue(any(kw.arg=='ionization' and isinstance(kw.value,ast.Name)
                                and kw.value.id=='ionization' for kw in call.keywords))
        budgets=[node for node in calls if node.func.id=='balance_checks']
        self.assertEqual(len(budgets),1)
        self.assertEqual(budgets[0].args[2].id,'ionization')

    def test_profile_exports_local_coefficients(self):
        with patch.object(tr,'solve_ivp',side_effect=fabricated_integrator):
            state=tr.solve_trial(.05,2.,.7,1.,8.,(0.,0.),constant_B(),ionization=(.5,.068),n_points=101)
        # Mark ONLY to exercise exporter with fabricated arrays; not a verified physical solution.
        state['stationary_verified']=True
        p=dict(beta=1.,A_star_cm2=2.,L_cm=100.,J0_per_s=1e20,M0_dyn=1.,W0_erg_per_s=1.,
               n_star_cm3=1e12,U_star_cm_per_s=1e7)
        output=dimensional_profiles(state,p)
        np.testing.assert_array_equal(output['D_ion'],state['D_ion_profile'])
        np.testing.assert_array_equal(output['E_ion'],state['E_ion_profile'])


@unittest.skipUnless(CCD_PATH.exists(), 'Active CX requires the CCD table.')
class CommandTests(unittest.TestCase):
    def mocked_run(self,extra):
        temp=tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        output=Path(temp.name)/'output'
        argv=['run_stationary','--config',str(PACKAGE/'config.json'),'--guess','5','2','.8',
              '--scale','0','--excitation-scale','0','--output',str(output),*extra]
        outcome=dict(status='mock_not_solved',diagnostic=dict(evaluations=0))
        with patch.object(sys,'argv',argv),patch.object(cli,'solve_stationary',return_value=outcome) as solver:
            with patch.object(cli,'ADASRecombination',side_effect=AssertionError('ACD/PRB off')):
                self.assertEqual(cli.main(),1)
        return solver,json.loads((output/'report.json').read_text())

    @unittest.skipUnless(SCD_PATH.exists(),'SCD not distributed with code.')
    def test_default_enabled_local_SCD_and_hash(self):
        solver,report=self.mocked_run([])
        self.assertEqual(report['schema_version'],5)
        self.assertEqual(report['ionization_scale'],1.)
        self.assertEqual(set(report['data_sha256']),{'ccd96_d.dat','scd12_h.dat'})
        provider=solver.call_args.kwargs['ionization']
        self.assertTrue(callable(provider))
        theta=5/report['parameters']['T_star_eV']
        self.assertNotEqual(provider(theta,1.)[0],provider(theta,2.)[0])
        self.assertGreater(solver.call_args.kwargs['j_bounds'][1],1.)

    def test_off_scale_does_not_load_SCD(self):
        with patch.object(cli,'ADASIonization',side_effect=AssertionError('SCD off')):
            solver,report=self.mocked_run(['--ionization-scale','0'])
        self.assertEqual(solver.call_args.kwargs['ionization'],(0.,0.))
        self.assertEqual(set(report['data_sha256']),{'ccd96_d.dat'})
        self.assertIsNotNone(solver.call_args.kwargs['theta_bounds'])

    def test_zero_neutrals_disable_both_neutral_channels(self):
        with patch.object(cli,'ADASIonization',side_effect=AssertionError('No SCD')):
            solver,report=self.mocked_run(['--density','0'])
        self.assertEqual(solver.call_args.kwargs['D'],0.)
        self.assertEqual(solver.call_args.kwargs['ionization'],(0.,0.))
        self.assertEqual(report['data_sha256'],{})

    def test_explicit_j_window_passed(self):
        solver,_=self.mocked_run(['--ionization-scale','0','--j-min','.01','--j-max','10'])
        self.assertEqual(solver.call_args.kwargs['j_bounds'],(.01,10.))


if __name__=='__main__':
    unittest.main()
