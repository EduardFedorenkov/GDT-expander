"""Configuration loading and dimensional scale calculation."""

import json
from math import isfinite
from pathlib import Path

from .constants import EV_TO_ERG, M_D_G
from .solver import has_theta_solution


def load_cx_config(filename):
    """Load a JSON config and calculate the model's derived parameters."""
    config = json.loads(Path(filename).read_text(encoding="utf-8-sig"))

    if not isinstance(config, dict):
        raise ValueError("Корневой элемент JSON должен быть объектом.")

    config.setdefault("m_D_g", M_D_G)

    fields = (
        "J0_per_s",
        "M0_dyn",
        "W0_erg_per_s",
        "k_cx_cm3_per_s",
        "lambda_sh",
        "L_cm",
        "n0_cm3",
        "A0_cm2",
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
