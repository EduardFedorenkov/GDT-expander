"""Momentum matching at a: a prescribed kinetic closure, not a new ODE."""

from dataclasses import dataclass
import numpy as np

from .kinetic_momentum import kinetic_momentum


class KineticInletDomainError(ValueError):
    """A hydrodynamic trial reached an inlet outside the kinetic model domain."""


@dataclass(frozen=True)
class KineticInletMomentum:
    geometry: object
    L_cm: float
    A_m_cm2: float
    n_c_cm3: float
    T_c_eV: float
    M0_dyn: float
    chi: float = 5.0

    def __post_init__(self):
        values = (self.L_cm, self.A_m_cm2, self.n_c_cm3,
                  self.T_c_eV, self.M0_dyn, self.chi)
        if not np.isfinite(values).all() or min(values[:-1]) <= 0 or self.chi < 0:
            raise ValueError("Kinetic inlet: positive finite source inputs; chi >=0.")
        if self.geometry.g_min == 0 and not np.isclose(
                self.geometry.B_ratio(0), 1.0, rtol=1e-10, atol=1e-12):
            raise ValueError("Kinetic inlet requires B(0)/B_m=1 at the throat.")

    def evaluate(self, alpha):
        """Return G(alpha) and metadata; the hydro temperature is NOT an input.

        chi is the entire prescribed potential drop reached by a. A sampled
        field check is a screen, not a proof of collisionless orbit access.
        For a geometry missing the throat, the omitted prefix remains an
        explicit physical assumption (not silently reconstructed).
        """
        alpha = float(alpha)
        if not self.geometry.g_min <= alpha <= 1:
            raise KineticInletDomainError("Inlet outside the supplied geometry.")
        K_a = float(self.geometry.A_ratio(alpha))
        if K_a < 1:
            raise KineticInletDomainError("Transmitted-ion model requires A(a)/A_m >=1.")
        prefix = np.linspace(self.geometry.g_min, alpha, 257)
        if np.any(self.geometry.dlnB_dg(prefix) > 1e-10):
            raise KineticInletDomainError(
                "The supplied kinetic segment contains an increasing magnetic field.")
        momentum = kinetic_momentum(
            self.L_cm * alpha, self.geometry, self.chi,
            L_cm=self.L_cm, A_m_cm2=self.A_m_cm2, n_c_cm3=self.n_c_cm3,
            T_i0_eV=self.T_c_eV, T_e_eV=self.T_c_eV,
        )
        return dict(
            mu_target=momentum / self.M0_dyn, M_kin_dyn=momentum,
            inlet_momentum="kinetic", chi=self.chi, T_kin_eV=self.T_c_eV,
            K_a=K_a, kinetic_path_check=("sampled_full_prefix"
                if self.geometry.g_min == 0 else "sampled_partial_prefix"),
        )


def make_inlet_momentum(parameters, geometry, mode="kinetic"):
    """None explicitly denotes the old mu(alpha)=1 boundary condition."""
    if mode == "constant":
        return None
    if mode != "kinetic":
        raise ValueError("inlet momentum mode must be 'kinetic' or 'constant'.")
    return KineticInletMomentum(
        geometry=geometry, L_cm=parameters["L_cm"],
        A_m_cm2=parameters["A_m_cm2"], n_c_cm3=parameters["n_c_cm3"],
        T_c_eV=parameters["T_c_eV"], M0_dyn=parameters["M0_dyn"],
        chi=parameters["chi"],
    )


def evaluate_inlet(alpha, inlet_momentum=None):
    """Also accept a positive scalar callback G(alpha) for analytic tests."""
    if inlet_momentum is None:
        data = dict(mu_target=1.0, inlet_momentum="constant")
    elif hasattr(inlet_momentum, "evaluate"):
        data = inlet_momentum.evaluate(alpha)
    else:
        data = dict(mu_target=float(inlet_momentum(alpha)), inlet_momentum="custom")
    target = float(data["mu_target"])
    if not np.isfinite(target) or target <= 0:
        raise ValueError("Inlet momentum target must be positive and finite.")
    return dict(data, mu_target=target)


def add_inlet_arguments(parser):
    parser.add_argument(
        "--inlet-momentum", choices=("kinetic", "constant"), default="kinetic",
        help="kinetic: M_kin(a)/M0 with chi from config; constant: old mu(a)=1.",
    )
