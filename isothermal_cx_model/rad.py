import numpy as np


def L_rad(T_eV):
    """
    Коэффициент тормозного излучения дейтериевой плазмы.

    T_eV: температура в эВ, число или массив.
    Результат: эрг·см³/с.
    Объемные потери: P_ff = n_e * n_Dplus * L_rad(T_eV).
    """
    T = np.asarray(T_eV, dtype=float)

    if np.any(~np.isfinite(T)) or np.any(T < 0):
        raise ValueError("T_eV должна быть конечной и >= 0.")

    result = 2.15e-25 * np.sqrt(T)

    return float(result) if result.ndim == 0 else result


def R_br(parameters):
    """Безразмерный коэффициент тормозного излучения.

    parameters — словарь, возвращённый load_cx_config.
    Используется Lambda_br при характерной температуре T_*.
    """
    keys = (
        "L_cm",
        "A_star_cm2",
        "n_star_cm3",
        "T_star_eV",
        "W0_erg_per_s",
    )
    L, A_star, n_star, T_star_eV, W0 = (
        float(parameters[key]) for key in keys
    )

    if (
        not np.isfinite([L, A_star, n_star, T_star_eV, W0]).all()
        or min(L, A_star, n_star, T_star_eV, W0) <= 0
    ):
        raise ValueError(
            "Все характерные масштабы должны быть конечными и > 0."
        )

    result = L * A_star * n_star**2 * L_rad(T_star_eV) / W0

    if not np.isfinite(result):
        raise ValueError("Получен неконечный R_br: проверьте масштабы.")

    return float(result)
