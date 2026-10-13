from math import isfinite, log, sqrt
import warnings

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.integrate import cumulative_trapezoid


def solve_mu(theta, beta, D, n_points=400, *, rtol=1e-9, atol=1e-11):
    """Решить задачу импульса для одной пробной температуры.

    Предположения: A ~ z**2, T = const, только перезарядка,
    дозвуковая ветвь. D — число либо функция D(theta).

    Результат: список словарей с alpha, xi, mu, u_over_cs.
    Пустой список означает отсутствие допустимой границы.
    """
    theta, beta = float(theta), float(beta)
    D_value = float(D(theta) if callable(D) else D)

    if not all(isfinite(v) and v > 0 for v in (theta, beta, rtol, atol)):
        raise ValueError("theta, beta, rtol, atol должны быть положительными.")
    if not isfinite(D_value) or D_value < 0:
        raise ValueError("D должен быть конечным и неотрицательным.")
    if not isinstance(n_points, (int, np.integer)) or n_points < 2:
        raise ValueError("n_points должно быть целым числом >= 2.")

    b = sqrt(8 * beta * theta)
    if not isfinite(b) or not 0 < b < 1:
        raise ValueError("Требуется 0 < sqrt(8*beta*theta) < 1.")

    # Необходимые аналитические условия существования.
    if D_value <= 1 + sqrt((1 - b) * (1 + b)):
        return []
    if log(D_value) + b / D_value <= 1:
        return []

    def H(mu):
        # Защита пробных этапов интегратора при локализации событий.
        # Физическая область готового профиля проверяется отдельно.
        return mu + sqrt(max((mu - b) * (mu + b), 0.0))

    def rhs(g, state):
        # dmu/dg = H(mu)/g - D; g = z/L.
        return [H(state[0]) / g - D_value]

    def maximum(g, state):
        return rhs(g, state)[0]

    maximum.terminal = True
    maximum.direction = 1  # Знак события при интегрировании назад.

    # От плазмаприёмника до единственного максимума mu.
    wall_segment = solve_ivp(
        rhs, (1.0, b / D_value), [b],
        method="DOP853", events=maximum, dense_output=True,
        rtol=rtol, atol=atol, max_step=0.02,
    )
    if not wall_segment.success:
        raise RuntimeError(wall_segment.message)
    if not wall_segment.t_events[0].size:
        raise RuntimeError("Не удалось локализовать максимум mu.")

    g_max = float(wall_segment.t_events[0][0])
    mu_max = float(wall_segment.y_events[0][0, 0])
    boundary_tol = 20 * (atol + rtol)

    if mu_max < 1 - boundary_tol:
        return []

    inlet_segment = None

    if abs(mu_max - 1) <= boundary_tol:
        warnings.warn(
            "Максимум численно неотличим от mu=1. Возвращён кандидат "
            "на касание; для уточнения уменьшите rtol и atol.",
            RuntimeWarning, stacklevel=2,
        )
        roots = [(g_max, "tangent")]
    else:
        # Пересечение mu=1 на участке, прилегающем к приёмнику.
        alpha_right = brentq(
            lambda g: float(wall_segment.sol(g)[0]) - 1,
            g_max, 1.0,
            xtol=np.nextafter(0.0, 1.0),
            rtol=4 * np.finfo(float).eps,
        )

        # Выбираем нижнюю границу по аналитической оценке:
        # mu(g) < g * (b - D*log(g)).
        lower = g_max / 2
        while lower * (b - D_value * log(lower)) >= 1:
            lower /= 2

        def inlet(g, state):
            return state[0] - 1

        inlet.terminal = True
        inlet.direction = -1

        # Продолжаем назад от максимума до второго пересечения.
        inlet_segment = solve_ivp(
            rhs, (g_max, lower), [mu_max],
            method="DOP853", events=inlet, dense_output=True,
            rtol=rtol, atol=atol, max_step=0.02,
        )
        if not inlet_segment.success:
            raise RuntimeError(inlet_segment.message)
        if not inlet_segment.t_events[0].size:
            raise RuntimeError("Не удалось найти второе пересечение mu=1.")

        alpha_left = float(inlet_segment.t_events[0][0])
        roots = [
            (alpha_left, "with_maximum"),
            (alpha_right, "monotone"),
        ]

    solutions = []
    for alpha, branch in roots:
        xi = np.linspace(0.0, 1.0, n_points)
        g = alpha + (1 - alpha) * xi

        mu = np.empty_like(g)
        wall_mask = g >= g_max
        mu[wall_mask] = wall_segment.sol(g[wall_mask])[0]
        if (~wall_mask).any():
            mu[~wall_mask] = inlet_segment.sol(g[~wall_mask])[0]

        if not np.isfinite(mu).all() or np.any(mu < b - boundary_tol):
            raise RuntimeError("Профиль вышел из физической области.")

        radical = np.sqrt(np.maximum((mu - b) * (mu + b), 0.0))
        solutions.append({
            "theta": theta,
            "b": b,
            "D": D_value,
            "alpha": alpha,
            "xi": xi,
            "mu": mu,
            "u_over_cs": b / (mu + radical),
            "branch": branch,
            "inlet_residual": float(mu[0] - 1),
        })

    return solutions


def compute_omega(solution, beta, lambda_sh):
    """Восстановить omega и невязку входного условия энергии.

    solution — один словарь, возвращённый solve_mu.
    Возвращает (omega, R_E). Температура не подбирается.
    """
    theta = float(solution["theta"])
    alpha = float(solution["alpha"])
    D = float(solution["D"])
    b = float(solution["b"])
    beta, lambda_sh = float(beta), float(lambda_sh)

    if not all(
        np.isfinite(v) and v > 0
        for v in (theta, beta, lambda_sh)
    ):
        raise ValueError("theta, beta и lambda_sh должны быть положительными.")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("Требуется 0 < alpha < 1.")
    if not np.isfinite(D) or D < 0:
        raise ValueError("D должен быть конечным и неотрицательным.")
    if not np.isfinite(b) or not 0 < b < 1:
        raise ValueError("Требуется 0 < b < 1.")
    if not np.isclose(b**2, 8 * beta * theta, rtol=1e-9, atol=0):
        raise ValueError("beta не согласуется с b и theta в решении.")

    xi = np.asarray(solution["xi"], dtype=float)
    mu = np.asarray(solution["mu"], dtype=float)

    if xi.ndim != 1 or mu.shape != xi.shape or xi.size < 2:
        raise ValueError("xi и mu должны быть одномерными одинакового размера.")
    if not np.isfinite(xi).all() or not np.isfinite(mu).all():
        raise ValueError("Массивы должны содержать только конечные значения.")
    if np.any(np.diff(xi) <= 0):
        raise ValueError("xi должна строго возрастать.")
    if not np.allclose([xi[0], xi[-1]], [0, 1], rtol=0, atol=1e-12):
        raise ValueError("Сетка xi должна покрывать [0, 1].")

    roundoff = (
        32 * np.finfo(float).eps
        * max(1.0, float(np.max(np.abs(mu))))
    )
    if np.any(mu < b - roundoff):
        raise ValueError("Нарушена физическая область mu >= b.")
    if not np.isclose(mu[-1], b, rtol=1e-9, atol=roundoff):
        raise ValueError("Не выполнено условие mu(1) = b.")
    if not np.isclose(mu[0], 1, rtol=1e-6, atol=roundoff):
        raise ValueError("Не выполнено условие mu(0) = 1.")

    radical = np.sqrt(np.maximum((mu - b) * (mu + b), 0.0))
    loss = (1 - alpha) * D * (5 * mu + radical) / (8 * beta)

    # Интеграл от каждой точки xi до плазмаприёмника.
    # Обратная сетка даёт отрицательный интеграл — меняем знак.
    integral_to_wall = -cumulative_trapezoid(
        loss[::-1], xi[::-1], initial=0
    )[::-1]

    omega = lambda_sh * theta + integral_to_wall
    if not np.isfinite(omega).all():
        raise FloatingPointError("Получен неконечный поток энергии.")

    return omega, float(omega[0] - 1)


solutions = solve_mu(theta=0.0625, beta=0.5, D=4.0)

for solution in solutions:
    print(
        f"alpha={solution['alpha']:.8f}, "
        f"branch={solution['branch']}"
    )