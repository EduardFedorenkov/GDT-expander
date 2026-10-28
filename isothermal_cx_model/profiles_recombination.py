"""Dimensional profiles for a verified variable-J stationary solution."""

import numpy as np


def compute_stationary_profiles(solution, parameters):
    if not solution.get('stationary_verified', False):
        raise ValueError('A verified stationary solution is required, not an IVP trial.')
    p = parameters
    keys = ('J0_per_s', 'M0_dyn', 'W0_erg_per_s', 'L_cm', 'A0_cm2',
            'U_star_cm_per_s', 'n_star_cm3', 'T_star_erg', 'T_star_eV', 'm_D_g')
    if not np.isfinite([p[key] for key in keys]).all() or min(p[key] for key in keys) <= 0:
        raise ValueError('Source scales must be finite and positive.')
    expected = [p['m_D_g']*p['J0_per_s']*p['U_star_cm_per_s'],
                p['J0_per_s']*p['T_star_erg'],
                p['J0_per_s']/(p['A0_cm2']*p['U_star_cm_per_s']),
                p['T_star_eV']*1.602176634e-12]
    actual = [p['M0_dyn'], p['W0_erg_per_s'], p['n_star_cm3'], p['T_star_erg']]
    beta = p['T_star_erg']/(p['m_D_g']*p['U_star_cm_per_s']**2)
    if (not np.allclose(actual, expected, rtol=1e-10, atol=0)
            or not np.isclose(beta, solution['beta'], rtol=1e-10, atol=0)):
        raise ValueError('Profile/source normalization does not match the solution.')
    g, j, mu, omega, N, v, A, mach = [np.asarray(solution[key], dtype=float) for key in (
        'g', 'j', 'mu', 'omega', 'n_over_n_star', 'u_over_u_star',
        'A_over_A_star', 'u_over_cs')]
    if (g.ndim != 1 or g.size < 2 or np.any(np.diff(g) <= 0)
            or any(array.shape != g.shape or not np.isfinite(array).all()
                   for array in (g, j, mu, omega, N, v, A, mach))):
        raise ValueError('Invalid solution arrays.')
    area = p['A0_cm2']*A
    velocity = p['U_star_cm_per_s']*v
    particles = p['J0_per_s']*j
    energy = p['W0_erg_per_s']*omega
    T_erg = p['T_star_erg']*solution['theta']
    heat_flux = (energy-particles*(.5*p['m_D_g']*velocity**2+5*T_erg))/area
    return dict(
        z_cm=p['L_cm']*g, xi=np.asarray(solution['xi']), j=j.copy(),
        omega=omega.copy(), A_cm2=area, n_cm3=p['n_star_cm3']*N,
        u_cm_s=velocity, J_per_s=particles, M_dyn=p['M0_dyn']*mu,
        W_erg_per_s=energy, q_e_erg_cm2_s=heat_flux,
        u_over_cs=mach.copy(), u_over_u_star=v.copy(), n_over_n_star=N.copy(),
        T_eV=solution['theta']*p['T_star_eV'], a_cm=solution['alpha']*p['L_cm'],
        j_target=solution['j_target'], solution_id=solution.get('solution_id', 'root'),
    )


def plot_stationary_profiles(profiles, title=None):
    import matplotlib.pyplot as plt
    profiles = list(profiles)
    if not profiles:
        raise ValueError('No verified profiles to plot.')
    figure, axes = plt.subplots(5, 1, figsize=(8.5, 12), sharex=True,
                                layout='constrained')
    keys = ('j', 'omega', 'q_e_erg_cm2_s', 'u_over_cs', 'n_over_n_star')
    labels = (r'$J/J_0$', r'$W/W_0$', r'$q_e$ [erg cm$^{-2}$ s$^{-1}$]',
              r'$u/c_s$', r'$n/n_*$')
    for profile in profiles:
        label = (f"{profile['solution_id']}: T={profile['T_eV']:.5g} eV, "
                 f"a={profile['a_cm']:.4g} cm, jt={profile['j_target']:.5g}")
        for axis, key in zip(axes, keys):
            axis.plot(profile['z_cm'], profile[key], label=label)
    for axis, label in zip(axes, labels):
        axis.set_ylabel(label)
        axis.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0, 1.05*max(1., max(float(np.max(p['j'])) for p in profiles)))
    axes[2].axhline(0, color='black', linestyle='--', linewidth=.7)
    axes[-1].set_xlabel('z [cm]')
    if title:
        axes[0].set_title(title)
    return figure, axes
