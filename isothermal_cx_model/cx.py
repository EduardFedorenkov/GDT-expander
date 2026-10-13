import numpy as np
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator
from scipy.special import exprel


EV_TO_ERG = 1.602176634e-12  # эрг/эВ
M_D = 3.3435837768e-24      # г, масса дейтрона
M_U = 1.66053906892e-24     # г, атомная единица массы

# IAEA APID, vol. 8, стр. 92: D+ + D, столбец Spin Exchange.
# Энергия в системе центра масс, эВ.
_E_CM = np.array([
    0.1000, 0.1995, 0.5012, 1.0000, 1.9950,
    5.0120, 10.0000, 19.9500, 50.1200, 100.0000,
])

# Сечения: атомные единицы -> см².
_SIGMA = 2.80028e-17 * np.array([
    221.3106, 205.6762, 180.2378, 175.2253, 161.7602,
    144.8380, 132.6478, 120.9559, 106.0618, 95.38493,
])

_LOG_SIGMA = PchipInterpolator(
    np.log(_E_CM),
    np.log(_SIGMA),
    extrapolate=False,
)


def sigma_cx_D(E_cm_eV):
    """
    Сечение атомарной перезарядки D+ + D(1s), см².

    E_cm_eV: энергия в системе центра масс, эВ;
             число или массив.

    0.1–100 эВ: интерполяция D-таблицы МАГАТЭ.
    Ниже 0.1 эВ: постоянное сечение.
    Выше 100 эВ: H-аппроксимация Swaczyna et al. (2025),
                 перенесённая на D при одинаковой скорости.
    """
    E = np.asarray(E_cm_eV, dtype=float)

    if np.any(~np.isfinite(E)) or np.any(E < 0):
        raise ValueError("E_cm_eV должна быть конечной и >= 0.")

    result = np.empty_like(E)

    low = E < _E_CM[0]
    mid = (E >= _E_CM[0]) & (E <= _E_CM[-1])
    high = E > _E_CM[-1]

    result[low] = _SIGMA[0]
    result[mid] = np.exp(_LOG_SIGMA(np.log(E[mid])))

    # Энергия на единицу массы, кэВ/u:
    # E_lab = 2 * E_cm для одинаковых масс и неподвижной мишени.
    X = 2.0 * E[high] / (1000.0 * (M_D / M_U))

    result[high] = (
        1e-16 * 4.8569 * np.log(21.906 / X + 31.487)
        / (
            1.0
            + 0.12018 * X
            + 4.1402e-6 * X**3.7524
            + 8.8476e-12 * X**6.1091
        )
    )

    return float(result) if result.ndim == 0 else result


def k_cx(T_i_eV, u_z_cms):
    """
    <sigma_cx * v_rel> для неподвижных холодных нейтралов
    и смещённого изотропного максвеллиана ионов, см³/с.

    T_i_eV: ионная температура, эВ, >= 0.
    u_z_cms: средняя осевая скорость ионов, см/с.
    """
    T = float(T_i_eV)
    u = abs(float(u_z_cms))

    if not np.isfinite(T) or T < 0:
        raise ValueError("T_i_eV должна быть конечной и >= 0.")
    if not np.isfinite(u):
        raise ValueError("u_z_cms должна быть конечной.")

    # Предел холодного направленного пучка.
    if T == 0:
        E_cm = M_D * u**2 / (4.0 * EV_TO_ERG)
        return u * sigma_cx_D(E_cm)

    v_th = np.sqrt(2.0 * EV_TO_ERG * T / M_D)
    d = u / v_th

    # x = v_rel / v_th.
    # За пределами ±12 тепловых скоростей вклад пренебрежимо мал.
    left = max(0.0, d - 12.0)
    right = d + 12.0

    # Отмечаем узлы сечения для адаптивного интегрирования.
    points = np.sqrt(2.0 * _E_CM / T)
    points = points[(points > left) & (points < right)]

    integral, _ = quad(
        lambda x: (
            x**3
            * sigma_cx_D(0.5 * T * x**2)
            * np.exp(-(x - d)**2)
            * exprel(-4.0 * d * x)
        ),
        left,
        right,
        points=points,
        epsabs=0.0,
        epsrel=1e-8,
        limit=200,
    )

    return float(4.0 * v_th / np.sqrt(np.pi) * integral)


def D_cx(theta, parameters, n0_cm3=None):
    """Вычислить безразмерный D(theta) с k_cx(T, u=0).

    parameters — словарь из load_cx_config.
    n0_cm3 — необязательная замена плотности из JSON, см^-3.
    """
    theta = float(theta)
    n0 = float(
        parameters["n0_cm3"] if n0_cm3 is None else n0_cm3
    )
    L = float(parameters["L_cm"])
    u_star = float(parameters["U_star_cm_per_s"])
    T_star = float(parameters["T_star_eV"])

    if (
        not np.isfinite([theta, n0, L, u_star, T_star]).all()
        or theta < 0
        or n0 < 0
        or min(L, u_star, T_star) <= 0
    ):
        raise ValueError(
            "Нужно theta,n0 >= 0; L,u_*,T_* > 0; всё конечное."
        )

    T_eV = theta * T_star
    return float(
        L * n0 / u_star * k_cx(T_eV, u_z_cms=0.0)
    )


def make_D(parameters, mode="constant", n0_cm3=None):
    """Вернуть число D или функцию D(theta).

    constant: k_cx из JSON, D не зависит от температуры.
    temperature: k_cx(T, u=0) из функции D_cx.
    """
    n0 = float(
        parameters["n0_cm3"] if n0_cm3 is None else n0_cm3
    )
    if not np.isfinite(n0) or n0 < 0:
        raise ValueError("n0 должна быть конечной и >= 0.")

    if mode == "constant":
        return float(
            parameters["L_cm"]
            * n0
            * parameters["k_cx_cm3_per_s"]
            / parameters["U_star_cm_per_s"]
        )

    if mode == "temperature":
        return lambda theta: D_cx(
            theta, parameters, n0_cm3=n0
        )

    raise ValueError(
        "mode должен быть 'constant' или 'temperature'."
    )
