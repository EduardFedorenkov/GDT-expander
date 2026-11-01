"""PLT coefficients, format tags and normalization only; no plasma solver."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from isothermal_model_adas import exitation_adas as exc


def table_text(T=(0.,2.), n=(10.,12.,14.)):
    values = -30+.4*np.array(T)[:,None]-.15*(np.array(n)[None,:]-10)
    lines = [f'1 {len(n)} {len(T)} 1 1 /HYDROGEN /SYNTHETIC TEST', '------',
        ' '.join(map(str,n)), ' '.join(map(str,T)),
        '----/ IPRT=1 / IGRD=0 / Z1=1 / DATE=01/11/26']
    lines.extend(' '.join(map(str,row)) for row in values)
    return '\n'.join(lines+['------','C RADIATED POWERS IN UNITS OF W CM**3'])


class CoefficientTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'plt.dat'
        self.path.write_text(table_text(),encoding='utf-8')
        self.data = exc.ADASExcitation(self.path)
        self.p = dict(L_cm=100., n_star_cm3=1e12, U_star_cm_per_s=1e7,
            T_star_eV=100., T_star_erg=100*exc.EV_TO_ERG, n0_cm3=1e13)

    def test_non_square_nodes_and_log_bilinear_interpolation(self):
        self.assertEqual(self.data.log_T.shape,(2,))
        self.assertEqual(self.data.log_n.shape,(3,))
        for lt,ln in ((0,10),(2,14),(1,11),(.4,12.7)):
            expected = 1e7*10.**(-30+.4*lt-.15*(ln-10))
            actual = exc.L_exc(10.**lt,10.**ln,data=self.data)
            self.assertAlmostEqual(actual/expected,1.,places=13)

    def test_watt_conversion_once_and_no_photon_energy_multiplier(self):
        self.assertAlmostEqual(exc.L_exc(1.,1e10,data=self.data)/1e-23,1.,places=13)
        self.assertEqual(exc.WATT_TO_ERG_PER_S,1e7)

    def test_arrays_broadcast_no_mutation_and_scalar_return(self):
        T = np.array([[1.],[10.],[100.]])
        n = np.array([1e10,1e12,1e14])
        original = T.copy()
        loss = exc.L_exc(T,n,data=self.data)
        np.testing.assert_allclose(loss,1e-23*T**.4*(n/1e10)**-.15,rtol=1e-13)
        self.assertEqual(loss.shape,(3,3))
        np.testing.assert_array_equal(T,original)
        self.assertIsInstance(exc.L_exc(10.,1e12,data=self.data),float)
        self.assertFalse(self.data.log_T.flags.writeable)
        self.assertFalse(self.data.log_n.flags.writeable)

    def test_density_required_and_not_discarded(self):
        with self.assertRaises(TypeError):
            exc.L_exc(10.,data=self.data)
        low = exc.L_exc(10.,1e10,data=self.data)
        high = exc.L_exc(10.,1e14,data=self.data)
        self.assertAlmostEqual(high/low,10**-.6,places=13)

    def test_no_extrapolation(self):
        for T,n in ((.99,1e12),(101.,1e12),(10.,1e9),(10.,1e15)):
            with self.subTest(T=T,n=n), self.assertRaises(exc.ADASDomainError):
                exc.L_exc(T,n,data=self.data)

    def test_invalid_values_and_shapes(self):
        for value in (0.,-1.,np.nan,np.inf):
            with self.assertRaises(ValueError):
                exc.L_exc(value,1e12,data=self.data)
            with self.assertRaises(ValueError):
                exc.L_exc(10.,value,data=self.data)
        with self.assertRaises(ValueError):
            exc.L_exc([1.,10.],[1e10,1e12,1e14],data=self.data)

    def test_factory_units_and_one_shot_E_exc(self):
        provider = exc.make_E_exc(self.p,scale=1.,data=self.data)
        expected = self.p['L_cm']*self.p['n0_cm3']/(self.p['U_star_cm_per_s']*self.p['T_star_erg'])*self.data.evaluate(10.,1e12)
        self.assertAlmostEqual(provider(.1,1.)/expected,1.,places=13)
        self.assertAlmostEqual(exc.E_exc(.1,1.,self.p,data=self.data)/expected,1.,places=13)
        other = dict(self.p,n_star_cm3=2e12)
        # No extra n_star factor in the prefactor: same physical n_e -> same E.
        self.assertAlmostEqual(exc.make_E_exc(other,scale=1.,data=self.data)(.1,.5)/expected,1.,places=13)

    def test_scale_neutral_override_and_arrays(self):
        full = exc.make_E_exc(self.p,scale=1.,data=self.data)
        half = exc.make_E_exc(self.p,scale=.5,data=self.data)
        double = exc.make_E_exc(self.p,n0_cm3=2e13,scale=1.,data=self.data)
        theta = np.array([[.01],[.1],[1.]])
        N = np.array([.01,1.,100.])
        self.assertEqual(full(theta,N).shape,(3,3))
        np.testing.assert_allclose(half(theta,N),.5*full(theta,N),rtol=1e-13)
        np.testing.assert_allclose(double(theta,N),2*full(theta,N),rtol=1e-13)
        np.testing.assert_allclose(exc.E_exc(theta,N,self.p,data=self.data),full(theta,N),rtol=1e-13)

    def test_snapshot_of_source_parameters(self):
        provider = exc.make_E_exc(self.p,scale=1.,data=self.data)
        before = provider(.1,1.)
        self.p.update(L_cm=1.,n_star_cm3=1.,n0_cm3=1.,T_star_eV=1.)
        self.assertEqual(provider(.1,1.),before)

    def test_off_modes_do_not_load_tables_or_scales(self):
        with patch.object(exc,'_default_data',side_effect=AssertionError('No table needed')):
            self.assertEqual(exc.make_E_exc(None),0.)
            self.assertEqual(exc.make_E_exc(None,n0_cm3=0.,scale=1.),0.)
            self.assertEqual(exc.E_exc(0.,0.,None,n0_cm3=0.),0.)
            np.testing.assert_array_equal(exc.E_exc([0.,1.],[0.,1.],None,n0_cm3=0.),[0.,0.])

    def test_invalid_scales_parameters_and_query(self):
        for scale in (-1.,1.01,np.nan,np.inf,None):
            with self.assertRaises(ValueError):
                exc.make_E_exc(self.p,scale=scale,data=self.data)
        for key in self.p:
            bad = dict(self.p,**{key:-1.})
            with self.subTest(key=key),self.assertRaises(ValueError):
                exc.make_E_exc(bad,scale=1.,data=self.data)
        with self.assertRaises(ValueError):
            exc.make_E_exc(dict(self.p,T_star_erg=1.),scale=1.,data=self.data)
        with self.assertRaises(ValueError):
            exc.make_E_exc({},scale=1.,data=self.data)
        with self.assertRaises(exc.ADASDomainError):
            exc.make_E_exc(self.p,scale=1.,data=self.data)(.001,1.)
        with self.assertRaises(ValueError):
            exc.E_exc(-1.,1.,self.p,data=self.data)

    def test_IGRD_zero_is_accepted_but_other_tag_conventions_rejected(self):
        for before,after in (('IGRD=0','IGRD=1'),('IPRT=1','IPRT=0'),('Z1=1','Z1=2')):
            self.path.write_text(table_text().replace(before,after),encoding='utf-8')
            with self.assertRaises(exc.ADASDataError):
                exc.ADASExcitation(self.path)

    def test_bad_units_element_counts_grids_and_extra_blocks(self):
        original = table_text()
        for text in (original.replace('W CM**3','CM**3 S-1'),
                     original.replace('HYDROGEN','DEUTERIUM'),
                     original.replace('1 3 2 1 1','1 3 3 1 1'),
                     original.replace('10.0 12.0 14.0','10.0 10.0 14.0'),
                     original+'\nIPRT=1 / IGRD=0 / Z1=1',''):
            self.path.write_text(text,encoding='utf-8')
            with self.assertRaises(exc.ADASDataError):
                exc.ADASExcitation(self.path)

    def test_fortran_exponents_and_missing_file(self):
        text = table_text().replace('10.0 12.0 14.0','1.0D+01 1.2D+01 1.4D+01')
        self.path.write_text(text,encoding='utf-8')
        self.assertEqual(exc.ADASExcitation(self.path).evaluate(10.,1e12),self.data.evaluate(10.,1e12))
        with self.assertRaises(OSError):
            exc.ADASExcitation(self.path.with_name('missing.dat'))


@unittest.skipUnless(exc.PLT_PATH.exists(),'Real PLT data is not redistributed with code/tests.')
class RealDataTests(unittest.TestCase):
    def test_known_nodes_and_rounded_boundaries(self):
        data = exc.ADASExcitation()
        self.assertEqual(data.log_T.shape,(29,))
        self.assertEqual(data.log_n.shape,(24,))
        for T,n,expected in ((1.,1e13,2.113245729e-23),(10.,1e13,2.008861521e-19),
                             (10.,5e7,2.47793551e-19),(10.,2e15,4.60967174e-20)):
            self.assertAlmostEqual(data.evaluate(T,n)/expected,1.,places=7)
        self.assertGreater(data.evaluate(.2,5e7),0.)
        with self.assertRaises(exc.ADASDomainError):
            data.evaluate(.1999,1e13)

    def test_full_grid_positive_and_finite(self):
        data = exc.ADASExcitation()
        values = data.evaluate(10.**data.log_T[:,None],10.**data.log_n[None,:])
        self.assertEqual(values.shape,(29,24))
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue(np.all(values>0))

    def test_CLI_outputs_coefficients_only(self):
        command = [sys.executable,'-B','-m','isothermal_model_adas.exitation_adas']
        result = subprocess.run(command+['--temperatures','.2','1','10','--density','1e13'],
                                text=True,capture_output=True,encoding='utf-8')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('Coefficients ONLY',result.stdout)
        self.assertIn('no I*SCD or PRB',result.stdout)
        bad = subprocess.run(command+['--temperatures','.1'],capture_output=True)
        self.assertNotEqual(bad.returncode,0)


if __name__=='__main__':
    unittest.main()
