"""CGS profiles; q_e is the residual conductive flux in the kappa->infinity limit."""

import numpy as np


def dimensional_profiles(solution, parameters):
    if not solution.get('stationary_verified', False):
        raise ValueError('Only a verified BVP solution may be exported as a solution.')
    p, s = parameters, solution
    if not np.isclose(p['beta'], s['beta'], rtol=1e-10, atol=0):
        raise ValueError('Source/solution beta mismatch.')
    A = p['A_star_cm2']*s['A_over_A_star']
    return dict(z_cm=p['L_cm']*s['g'], xi=s['xi'],
        J_per_s=p['J0_per_s']*s['j'], M_dyn=p['M0_dyn']*s['mu'],
        W_erg_per_s=p['W0_erg_per_s']*s['omega'], omega=s['omega'],
        q_e_erg_cm2_s=p['W0_erg_per_s']*s['Aq_e_over_W0']/A,
        u_over_u_star=s['u_over_u_star'], u_over_cs=s['u_over_cs'],
        n_over_n_star=s['n_over_n_star'], n_cm3=p['n_star_cm3']*s['n_over_n_star'],
        u_cm_s=p['U_star_cm_per_s']*s['u_over_u_star'], A_cm2=A,
        R=s['R_profile'], Q_eff=s['Q_eff_profile'],
        D_cx=np.full_like(s['g'], float(s.get('D', 0.))),
        D_ion=np.asarray(s.get('D_ion_profile', np.zeros_like(s['g']))),
        E_ion=np.asarray(s.get('E_ion_profile', np.zeros_like(s['g']))))


def plot_profiles(profile):
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True, layout='constrained')
    for axis, key, label in zip(axes,
            ('omega', 'q_e_erg_cm2_s', 'u_over_u_star', 'n_over_n_star'),
            (r'$W/W_0$', r'$q_e$ [erg cm$^{-2}$ s$^{-1}$]', r'$u/u_*$', r'$n/n_*$')):
        axis.plot(profile['z_cm'], profile[key])
        axis.set_ylabel(label)
        axis.grid(alpha=.25)
    axes[-1].set_xlabel('z [cm]')
    return figure, axes
