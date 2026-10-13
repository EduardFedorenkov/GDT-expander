"""Восстановление профилей: A(z) = A_m*K*(z/L)**2."""

import contextlib
import io

import numpy as np

# Не выводим демонстрационный пример из текущего solver.py.
with contextlib.redirect_stdout(io.StringIO()):
    from .solver import compute_omega


def compute_profiles(solution, parameters, K=500.0):
    """Вернуть профили на сетке одного кандидата из solve_mu.

    Температуру и границу a здесь не подбираем.
    R_E показывает, выполнено ли входное условие энергии.
    Размерные величины возвращаются в СГС, температура — в eV.
    """
    K = float(K)
    if not np.isfinite(K) or K <= 0:
        raise ValueError("K должен быть положительным и конечным.")

    beta = parameters["beta"]
    theta = float(solution["theta"])
    b = float(solution["b"])
    alpha = float(solution["alpha"])
    xi = np.asarray(solution["xi"], dtype=float)

    omega, residual = compute_omega(
        solution, beta, parameters["lambda_sh"]
    )

    mach = np.asarray(solution["u_over_cs"], dtype=float)
    if (
        mach.shape != xi.shape
        or not np.isfinite(mach).all()
        or np.any(mach <= 0)
        or np.any(mach > 1 + 1e-8)
    ):
        raise ValueError("Некорректный дозвуковой профиль u/c_s.")

    z_over_L = alpha + (1 - alpha) * xi
    z_cm = parameters["L_cm"] * z_over_L

    # A0_cm2 здесь означает площадь в пробке A_m,
    # а не значение квадратичной аппроксимации при z=0.
    area = parameters["A0_cm2"] * K * z_over_L**2

    # c_s/u_* = b/2; НЕ смешиваем u/c_s и u/u_*.
    velocity_ratio = (b / 2) * mach
    velocity = parameters["U_star_cm_per_s"] * velocity_ratio

    density = parameters["J0_per_s"] / (area * velocity)
    density_ratio = density / parameters["n_star_cm3"]

    temperature_erg = theta * parameters["T_star_erg"]
    energy = parameters["W0_erg_per_s"] * omega

    # W = J*(m_D*u**2/2 + 5*T) + A*q_e.
    # q_e — плотность теплопроводного потока, не полный W/A.
    convective_energy = parameters["J0_per_s"] * (
        parameters["m_D_g"] * velocity**2 / 2
        + 5 * temperature_erg
    )
    heat_flux = (energy - convective_energy) / area

    # Aq_e/(J_0*T) = omega/theta - 5 - (u/c_s)**2
    heat_flux_normalized = omega / theta - 5.0 - mach**2

    # Электронная тепловая скорость: T задана в эргах.
    electron_mass_g = 9.1093837139e-28
    electron_thermal_speed = np.sqrt(
        temperature_erg / electron_mass_g
    )

    # Отношение модуля теплопроводного потока
    # к кинетическому масштабу n*T*v_Te.
    chi_q = np.abs(heat_flux) / (
        density * temperature_erg * electron_thermal_speed
    )

    return {
        "xi": xi.copy(),
        "z_cm": z_cm,
        "A_cm2": area,
        "omega": omega,
        "q_e_erg_cm2_s": heat_flux,
        "Aq_e_over_J0T": heat_flux_normalized,
        "chi_q": chi_q,
        "u_over_u_star": velocity_ratio,
        "n_over_n_star": density_ratio,
        "u_cm_s": velocity,
        "n_cm3": density,
        "T_eV": theta * parameters["T_star_eV"],
        "a_cm": alpha * parameters["L_cm"],
        "branch": solution["branch"],
        "R_E": residual,
    }
