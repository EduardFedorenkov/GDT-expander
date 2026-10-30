"""Local reaction sources for the dimensionless isothermal equations.

These are contributions to derivatives in g=z/L, NOT dimensional volumetric
S_n, R_p and P_loss. Particle/momentum sources are signed; power losses are
positive for cooling. Magnetic geometry is not a reaction source.

Ordered terms retain the previous arithmetic order, including adding the
geometric momentum term before the reaction terms. This matters for local
root corrections near very narrow valid domains. No registry or new physics.
"""

from typing import NamedTuple


class SourceRates(NamedTuple):
    particles: float
    momentum: float
    power_loss: float


def _total(terms):
    if not terms:
        return 0.0
    value = terms[0][1]
    for _, term in terms[1:]:
        value += term
    return value


class SourceTerms(NamedTuple):
    """Ordered (process_name, value) terms in each conservation channel."""

    particle_terms: tuple
    momentum_terms: tuple
    power_loss_terms: tuple

    @property
    def totals(self):
        return SourceRates(_total(self.particle_terms),
                           _total(self.momentum_terms),
                           _total(self.power_loss_terms))

    def by_process(self):
        """Optional diagnostic decomposition; not used by balance_checks."""
        result = {}
        for channel, terms in enumerate(self):
            for name, value in terms:
                result.setdefault(name, [0.0, 0.0, 0.0])[channel] += value
        return {name: SourceRates(*values) for name, values in result.items()}

    def rhs(self, geometric_momentum):
        """Assemble (j_g, mu_g, omega_g) without reassociating old sums."""
        momentum = geometric_momentum
        for _, value in self.momentum_terms:
            momentum += value
        if self.power_loss_terms:
            energy = -self.power_loss_terms[0][1]
            for _, value in self.power_loss_terms[1:]:
                energy -= value
        else:
            energy = 0.0
        return [_total(self.particle_terms), momentum, energy]


def atomic_sources(j, mu, radical, H, area, *, theta, beta,
                   D, E_exc, R, Q, D_ion, E_ion):
    """Frozen-coefficient CX, excitation, Case-A RR and ionization sources.

    The caller reconstructs the local state and validates coefficients.
    H=mu+radical and radical=sqrt(mu**2-8*beta*theta*j**2) on the subsonic
    branch; provisional integrator stages use the existing clipped radical.
    For RR the ion kinetic/internal loss and electron cooling stay grouped.
    The binding energy is NOT added to RR cooling a second time.
    """
    phi = H*H/(16*beta*beta*theta*theta*area)
    particles = (("recombination", -R*phi),)
    if D_ion != 0:
        particles += (("ionization", D_ion*H/(4*beta*theta)),)
    momentum = (("cx", -D*j),
                ("recombination", -R*j*H/(4*beta*theta*area)))
    power = (("cx", D*(5*mu+radical)/(8*beta)),)
    if E_exc != 0:
        power += (("excitation", E_exc*H/(4*beta*theta)),)
    if E_ion != 0:
        power += (("ionization", E_ion*H/(4*beta*theta)),)
    if R != 0:
        power += (("recombination", R*j*j/(2*beta*area)+(1.5*R*theta+Q)*phi),)
    return SourceTerms(particles, momentum, power)
