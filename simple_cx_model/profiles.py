"""Spatial profile calculations for the simple model."""

import numpy as np


def cx_profiles(theta, beta, lambda_sh, alpha, L_cm, n_points=500):
    """Calculate coordinate, velocity, density, and energy profiles."""
    if not all(
        np.isfinite(value) and value > 0
        for value in (theta, beta, lambda_sh, L_cm)
    ):
        raise ValueError(
            "theta, beta, lambda_sh и L_cm должны быть "
            "положительными и конечными."
        )

    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("Требуется 0 < alpha < 1.")

    if not isinstance(n_points, (int, np.integer)) or n_points < 2:
        raise ValueError("n_points должно быть целым числом >= 2.")

    mu_t = np.sqrt(8 * beta * theta)

    if not 0 < mu_t < 1:
        raise ValueError("Требуется 0 < sqrt(8*beta*theta) < 1.")

    xi = np.linspace(0.0, 1.0, n_points)
    z_cm = L_cm * (alpha + (1 - alpha) * xi)

    mu = 1 - (1 - mu_t) * xi
    mu[-1] = mu_t

    delta = np.maximum((mu - mu_t) * (mu + mu_t), 0.0)
    square_root = np.sqrt(delta)
    velocity_ratio = mu_t / (mu + square_root)
    density_ratio = 2.0 / (mu_t * velocity_ratio)

    energy_integral = 2.5 * delta + 0.5 * (
        mu * square_root
        - mu_t**2 * np.log((mu + square_root) / mu_t)
    )
    energy_ratio = lambda_sh * theta + energy_integral / (8 * beta)

    if not np.isclose(energy_ratio[0], 1.0, rtol=1e-6, atol=1e-9):
        raise ValueError(
            "theta не удовлетворяет уравнению (3): "
            f"omega(a)={energy_ratio[0]:.9g}, должно быть 1."
        )

    return z_cm, velocity_ratio, density_ratio, energy_ratio
