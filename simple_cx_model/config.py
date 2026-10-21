"""Configuration loading and dimensional scale calculation."""

import json
from math import isfinite
from pathlib import Path

from .boundary_fluxes import throat_fluxes
from .constants import EV_TO_ERG, M_D_G
from .solver import has_theta_solution


def load_cx_config(filename):
    """Load central-source inputs and derive throat flows and model scales.

    The source approximation is n_m = n_c/2 and T_m = T_c, NOT density
    scaling with magnetic field. J0, M0, W0 refer to the mirror throat.
    Matching M_kin(a) at the hydrodynamic inlet is a separate task.
    """
    config = json.loads(Path(filename).read_text(encoding="utf-8-sig"))

    if not isinstance(config, dict):
        raise ValueError("Корневой элемент JSON должен быть объектом.")

    derived_inputs = {"J0_per_s", "M0_dyn", "W0_erg_per_s", "A0_cm2"}
    supplied = sorted(derived_inputs.intersection(config))
    if supplied:
        raise ValueError(
            "Удалите вычисляемые поля из JSON: " + ", ".join(supplied)
            + ". Задайте A_m_cm2, n_c_cm3, T_c_eV и chi_E."
        )

    config.setdefault("m_D_g", M_D_G)

    fields = (
        "A_m_cm2",
        "n_c_cm3",
        "T_c_eV",
        "chi_E",
        "k_cx_cm3_per_s",
        "lambda_sh",
        "L_cm",
        "n0_cm3",
        "m_D_g",
    )

    parameters = {}

    for name in fields:
        if name not in config:
            raise ValueError(f"В конфиге отсутствует параметр: {name}")

        try:
            value = float(config[name])
        except (ValueError, TypeError) as error:
            raise ValueError(
                f"Параметр {name} должен быть числом."
            ) from error

        if not isfinite(value) or value <= 0:
            raise ValueError(
                f"Параметр {name} должен быть положительным и конечным."
            )

        parameters[name] = value

    # The full source Maxwellian is normalized to n_c; its outgoing half
    # has physical density n_m = n_c/2. Do not apply B_m/B_c here.
    parameters.update(
        A0_cm2=parameters["A_m_cm2"],
        n_m_cm3=parameters["n_c_cm3"] / 2.0,
        T_m_eV=parameters["T_c_eV"],
    )
    parameters.update(throat_fluxes(
        parameters["A_m_cm2"],
        parameters["n_m_cm3"],
        parameters["T_m_eV"],
        parameters["chi_E"],
        m_D_g=parameters["m_D_g"],
    ))

    particle_flux = parameters["J0_per_s"]
    momentum_flux = parameters["M0_dyn"]
    energy_flux = parameters["W0_erg_per_s"]
    length = parameters["L_cm"]
    area = parameters["A0_cm2"]
    neutral_density = parameters["n0_cm3"]
    particle_mass = parameters["m_D_g"]
    cx_rate = parameters["k_cx_cm3_per_s"]

    velocity_scale = momentum_flux / (particle_mass * particle_flux)
    temperature_scale = energy_flux / particle_flux
    density_scale = particle_flux / (area * velocity_scale)
    cx_frequency = cx_rate * neutral_density

    parameters.update(
        U_star_cm_per_s=velocity_scale,
        T_star_erg=temperature_scale,
        T_star_eV=temperature_scale / EV_TO_ERG,
        n_star_cm3=density_scale,
        A_star_cm2=area,
        L_star_cm=length,
        J_star_per_s=particle_flux,
        M_star_dyn=momentum_flux,
        W_star_erg_per_s=energy_flux,
        nu_cx_per_s=cx_frequency,
        beta=temperature_scale / (particle_mass * velocity_scale**2),
        D=length * cx_frequency / velocity_scale,
    )

    derived = (
        "U_star_cm_per_s",
        "T_star_erg",
        "T_star_eV",
        "n_star_cm3",
        "nu_cx_per_s",
        "beta",
        "D",
    )

    if not all(
        isfinite(parameters[key]) and parameters[key] > 0
        for key in derived
    ):
        raise ValueError(
            "Некорректные вычисленные масштабы: проверьте единицы."
        )

    parameters["theta_exists"] = has_theta_solution(
        parameters["beta"], parameters["lambda_sh"]
    )

    return parameters
