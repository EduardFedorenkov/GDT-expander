"""PLT algebra and mocked plumbing only; no plasma integration or scan."""

import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.integrate import simpson
from isothermal_model_adas import transport as tr
from isothermal_model_adas import stationary
from isothermal_model_adas import run_stationary as cli
from isothermal_model_adas.sources import adas_sources, excitation_at
from isothermal_model_adas.geometry import constant_B
from isothermal_model_adas.profiles import dimensional_profiles
from isothermal_model_adas.config import load_config
from isothermal_model_adas.exitation_adas import PLT_PATH, make_E_exc, ADASExcitation
from isothermal_model_adas.cx_adas import CCD_PATH
from isothermal_model_adas.recombination_adas import ADASDomainError
from isothermal_model_adas.tests.test_ionization_integration import fabricated_integrator


PACKAGE=Path(__file__).parents[1]
OLD_SOURCES=PACKAGE.parent/'isothermal_cx_model'/'sources.py'


class SourceTests(unittest.TestCase):
    def test_PLT_changes_only_energy(self):
        args=(.8,3.,.2,2.)
        kw=dict(theta=.05,beta=1.,R=.01,Q_eff=-.001,D=.4,D_ion=.3,E_ion=.04)
        before=np.array(adas_sources(*args,**kw))
        after=np.array(adas_sources(*args,**kw,E_exc=.2))
        np.testing.assert_allclose(after-before,[0.,0.,-.2*2*3],atol=1e-14)

    def test_off_preserves_sources_exactly(self):
        args=(.8,3.,.2,2.)
        kw=dict(theta=.05,beta=1.,R=.01,Q_eff=-.001,D=.4,D_ion=.3,E_ion=.04)
        self.assertEqual(adas_sources(*args,**kw),adas_sources(*args,**kw,E_exc=0.))

    def test_local_scalar_and_array_validation(self):
        self.assertEqual(excitation_at(.2,.05,1.),.2)
        np.testing.assert_allclose(excitation_at(lambda t,n:.1*np.asarray(n),.05,np.array([1.,2.])),[.1,.2])
        for value in (-1.,np.nan,np.inf,[1.]):
            with self.assertRaises(ValueError):
                excitation_at(value,.05,1.)
        with self.assertRaises(ValueError):
            excitation_at(np.ones((2,1)),.05,np.array([1.,2.]))

    def test_RHS_and_wall_momentum_are_unchanged(self):
        theta,beta,j=.05,1.,2.
        b=np.sqrt(8*beta*theta)
        args=dict(theta=theta,beta=beta,recombination=(.01,-.001),geometry=constant_B(),D=1.,ionization=(.3,.04))
        state=[j,b*j,.4]
        before=tr.rhs_g(1.,state,**args)
        after=tr.rhs_g(1.,state,**args,excitation=.2)
        np.testing.assert_array_equal(before[:2],after[:2])
        self.assertEqual(before[1]-b*before[0],after[1]-b*after[0])
        self.assertLess(after[2],before[2])

    def test_binding_energy_identity_with_PLT(self):
        j,N,v,A,theta,beta=.8,2.,.2,3.,.05,1.
        R,P,I,Di,E=.02,.003,.1,.5,.04
        jg,_,wg=adas_sources(j,N,v,A,theta=theta,beta=beta,R=R,Q_eff=P-I*R,
            D_ion=Di,E_ion=I*Di,E_exc=E)
        expected=-A*N*N*(R*(v*v/(2*beta)+1.5*theta)+P)-E*A*N
        self.assertAlmostEqual(wg+I*jg,expected)

    @unittest.skipUnless(OLD_SOURCES.exists(),'Old assembler not installed beside new package.')
    def test_matches_old_excitation_channel_for_100_states(self):
        spec=importlib.util.spec_from_file_location('legacy_sources_PLT_comparison',OLD_SOURCES)
        old=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        rng=np.random.default_rng(87)
        for _ in range(100):
            beta,theta=10**rng.uniform(-1,1),10**rng.uniform(-3,-1)
            j,A=rng.uniform(.01,3.),10**rng.uniform(0,3)
            v=rng.uniform(.05,.95)*np.sqrt(2*beta*theta)
            mu=j*(v+2*beta*theta/v)
            radical=np.sqrt(mu*mu-8*beta*theta*j*j)
            H=mu+radical
            N=H/(4*beta*theta*A)
            D,R,Q,Di,Ei,E=1.2,.02,-.001,.3,.04,.2
            expected=old.atomic_sources(j,mu,radical,H,A,theta=theta,beta=beta,
                D=D,R=R,Q=Q,E_exc=E,D_ion=Di,E_ion=Ei).rhs(0.)
            actual=adas_sources(j,N,v,A,theta=theta,beta=beta,D=D,R=R,Q_eff=Q,D_ion=Di,E_ion=Ei,E_exc=E)
            np.testing.assert_allclose(actual,expected,rtol=5e-13,atol=1e-13)

    def test_independent_energy_budget_and_missing_provider_guard(self):
        g=np.linspace(.4,1.,101)
        H=3-.3*g
        N=H/(4*.05)
        state=dict(g=g,j=1-.1*g,mu=2-.2*g,omega=1-.3*g,H=H,
            A_over_A_star=np.ones_like(g),n_over_n_star=N,u_over_u_star=.2*np.ones_like(g),
            theta=.05,beta=1.,geometry=constant_B(),D=.4)
        provider=lambda t,n:.01*np.asarray(n)
        with patch.object(tr,'adas_sources',side_effect=AssertionError('Independent budgets required')):
            before=tr.balance_checks(state,(.01,-.001),(.3,.04))
            after=tr.balance_checks(state,(.01,-.001),(.3,.04),excitation=provider)
        expected=float(simpson(.01*N*N,x=g))
        self.assertEqual(after['particle_abs'],before['particle_abs'])
        self.assertEqual(after['momentum_abs'],before['momentum_abs'])
        self.assertEqual(after['ionization_energy_loss_integral'],before['ionization_energy_loss_integral'])
        self.assertAlmostEqual(after['energy_loss_integral']-before['energy_loss_integral'],expected)
        self.assertAlmostEqual(after['excitation_energy_loss_integral'],expected)
        state['E_exc_profile']=provider(.05,N)
        with self.assertRaisesRegex(ValueError,'excitation provider'):
            tr.balance_checks(state,(.01,-.001),(.3,.04))


class PlumbingTests(unittest.TestCase):
    def test_local_queries_wall_RHS_profile_and_export(self):
        calls=[]
        def provider(theta,N):
            calls.append(np.asarray(N).copy())
            return .01*np.asarray(N)
        with patch.object(tr,'solve_ivp',side_effect=fabricated_integrator):
            result=tr.solve_trial(.05,2.,.7,1.,8.,(0.,0.),constant_B(),ionization=(.5,.068),
                                  excitation=provider,n_points=101)
        self.assertEqual(result['status'],'integrated')
        self.assertTrue(any(n.ndim==0 for n in calls))
        self.assertTrue(any(n.shape==(101,) for n in calls))
        self.assertGreater(np.ptp(result['E_exc_profile']),0.)
        np.testing.assert_allclose(result['E_exc_profile'],.01*result['n_over_n_star'])
        # Fabricated arrays, never a physical/verified solution: flag is for exporter test ONLY.
        result['stationary_verified']=True
        p=dict(beta=1.,A_star_cm2=2.,L_cm=100.,J0_per_s=1e20,M0_dyn=1.,W0_erg_per_s=1.,
               n_star_cm3=1e12,U_star_cm_per_s=1e7)
        output=dimensional_profiles(result,p)
        np.testing.assert_array_equal(output['E_exc'],result['E_exc_profile'])

    def test_trial_wall_check_is_identical_with_PLT(self):
        kw=dict(ionization=(.5,.068),n_points=101)
        with patch.object(tr,'solve_ivp',side_effect=fabricated_integrator):
            before=tr.solve_trial(.05,2.,.7,1.,8.,(0.,0.),constant_B(),**kw)
            after=tr.solve_trial(.05,2.,.7,1.,8.,(0.,0.),constant_B(),excitation=.2,**kw)
        self.assertEqual(before['wall_slope_per_j'],after['wall_slope_per_j'])
        np.testing.assert_array_equal(before['j'],after['j'])
        np.testing.assert_array_equal(before['mu'],after['mu'])

    def test_domain_errors_at_wall_provisional_RHS_and_profile(self):
        for limit in (1,2,3):
            count=[0]
            def provider(theta,N):
                count[0]+=1
                if count[0]>=limit:
                    raise ADASDomainError('PLT outside table')
                return .01*np.asarray(N)
            with self.subTest(limit=limit),patch.object(tr,'solve_ivp',side_effect=fabricated_integrator):
                result=tr.solve_trial(.05,2.,.7,1.,8.,(0.,0.),constant_B(),ionization=(.5,.068),excitation=provider)
            self.assertEqual(result['status'],'adas_outside_domain')

    def test_negative_coefficient_does_not_reach_integrator(self):
        with patch.object(tr,'solve_ivp',side_effect=AssertionError('Must not integrate')):
            with self.assertRaises(ValueError):
                tr.solve_trial(.05,2.,.7,1.,8.,(0.,0.),constant_B(),ionization=(.5,.068),excitation=-1.)

    def test_fixed_interval_xi_factor(self):
        captured={}
        def fake(rhs,interval,initial,**kwargs):
            captured['xi_derivative']=rhs(.8,initial)
            captured['initial']=initial
            return fabricated_integrator(rhs,interval,initial,**kwargs)
        with patch.object(tr,'solve_ivp',side_effect=fake):
            tr.solve_trial(.05,2.,.7,1.,8.,(0.,0.),constant_B(),ionization=(.5,.068),excitation=.2)
        expected=.3*tr.rhs_g(.7+.3*.8,captured['initial'],theta=.05,beta=1.,
            recombination=(0.,0.),geometry=constant_B(),ionization=(.5,.068),excitation=.2)
        np.testing.assert_allclose(captured['xi_derivative'],expected,rtol=1e-13)

    def test_stationary_forwards_to_trials_and_budgets(self):
        with patch.object(stationary,'solve_trial',return_value={'status':'mock_not_integrated'}) as trial:
            result=stationary.solve_stationary(.05,2.,.7,1.,8.,(0.,0.),excitation=.2)
        self.assertEqual(result['status'],'invalid_initial_trial')
        self.assertEqual(trial.call_args.kwargs['excitation'],.2)
        tree=ast.parse((PACKAGE/'stationary.py').read_text(encoding='utf-8-sig'))
        names=('solve_trial','balance_checks')
        calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call)
               and isinstance(node.func,ast.Name) and node.func.id in names]
        self.assertEqual({node.func.id for node in calls},set(names))
        for node in calls:
            self.assertTrue(any(kw.arg=='excitation' and isinstance(kw.value,ast.Name)
                                and kw.value.id=='excitation' for kw in node.keywords))

    @unittest.skipUnless(PLT_PATH.exists(),'PLT not distributed with code.')
    def test_real_factory_without_integrating(self):
        p=load_config(PACKAGE/'config.json')
        data=ADASExcitation()
        provider=make_E_exc(p,scale=1.,data=data)
        theta=10./p['T_star_eV']
        N=1e13/p['n_star_cm3']
        expected=p['L_cm']*p['n0_cm3']/(p['U_star_cm_per_s']*p['T_star_erg'])*data.evaluate(10.,1e13)
        self.assertAlmostEqual(provider(theta,N)/expected,1.,places=13)


@unittest.skipUnless(CCD_PATH.exists(), 'Active CX requires the CCD table.')
class CommandTests(unittest.TestCase):
    def mocked_run(self,extra):
        temp=tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        output=Path(temp.name)/'output'
        argv=['run_stationary','--config',str(PACKAGE/'config.json'),'--guess','5','2','.8',
              '--scale','0','--ionization-scale','0','--output',str(output),*extra]
        outcome=dict(status='mock_not_solved',diagnostic=dict(evaluations=0))
        with patch.object(sys,'argv',argv),patch.object(cli,'solve_stationary',return_value=outcome) as solver:
            with patch('builtins.print'):
                self.assertEqual(cli.main(),1)
        return solver,json.loads((output/'report.json').read_text())

    @unittest.skipUnless(PLT_PATH.exists(),'PLT not distributed with code.')
    def test_default_PLT_enabled_and_hash_recorded(self):
        with (patch.object(cli,'ADASRecombination',side_effect=AssertionError('Recombination off')),
              patch.object(cli,'ADASIonization',side_effect=AssertionError('Ionization off'))):
            solver,report=self.mocked_run([])
        self.assertEqual(report['schema_version'],5)
        self.assertEqual(report['excitation_scale'],1.)
        self.assertEqual(set(report['data_sha256']),{'ccd96_d.dat','plt12_h.dat'})
        provider=solver.call_args.kwargs['excitation']
        self.assertTrue(callable(provider))
        theta=5/report['parameters']['T_star_eV']
        self.assertNotEqual(provider(theta,1.),provider(theta,2.))
        self.assertIsNotNone(solver.call_args.kwargs['theta_bounds'])

    def test_off_no_PLT_loading_but_CCD_temperature_limit_remains(self):
        with patch.object(cli,'ADASExcitation',side_effect=AssertionError('PLT off')):
            solver,report=self.mocked_run(['--excitation-scale','0'])
        self.assertEqual(solver.call_args.kwargs['excitation'],0.)
        self.assertEqual(set(report['data_sha256']),{'ccd96_d.dat'})
        self.assertIsNotNone(solver.call_args.kwargs['theta_bounds'])

    def test_zero_neutrals_disable_all_neutral_channels(self):
        with patch.object(cli,'ADASExcitation',side_effect=AssertionError('No PLT')):
            solver,report=self.mocked_run(['--density','0'])
        self.assertEqual(solver.call_args.kwargs['D'],0.)
        self.assertEqual(solver.call_args.kwargs['ionization'],(0.,0.))
        self.assertEqual(solver.call_args.kwargs['excitation'],0.)
        self.assertEqual(report['data_sha256'],{})

    @unittest.skipUnless(all((PACKAGE/'processes_data'/name).exists() for name in
        ('acd12_h.dat','prb12_h.dat','ccd96_d.dat','scd12_h.dat','plt12_h.dat')),'Full ADAS set not distributed.')
    def test_all_five_active_files_in_report(self):
        solver,report=self.mocked_run(['--scale','1','--ionization-scale','1'])
        self.assertEqual(set(report['data_sha256']),{'acd12_h.dat','prb12_h.dat','ccd96_d.dat','scd12_h.dat','plt12_h.dat'})
        self.assertTrue(callable(solver.call_args.kwargs['ionization']))
        self.assertTrue(callable(solver.call_args.kwargs['excitation']))


if __name__=='__main__':
    unittest.main()
