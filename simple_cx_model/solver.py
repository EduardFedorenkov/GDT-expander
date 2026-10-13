"""Solvers for the dimensionless model parameters."""

from math import exp, isfinite, sqrt

import numpy as np
from scipy.optimize import brentq


def find_alpha(theta, beta, D):
    """Return the dimensionless boundary position ``alpha = a / L``."""
    if not all(isfinite(value) and value > 0 for value in (theta, beta, D)):
        raise ValueError(
            "theta, beta и D должны быть положительными и конечными."
        )

    mu_t = sqrt(8 * beta * theta)

    if not 0 < mu_t < 1:
        raise ValueError(
            "Для рассматриваемого решения требуется 0 < mu_t < 1."
        )

    alpha = 1 - (1 - mu_t) / D

    if not 0 < alpha < 1:
        raise ValueError(
            f"Получено alpha={alpha:.6g}: граница не лежит внутри (0, L)."
        )

    return alpha


def find_theta(lambda_sh, beta):
    """Return all valid temperature roots for equation (3)."""
    lambda_sh = float(lambda_sh)
    beta = float(beta)

    if (
        not np.isfinite([lambda_sh, beta]).all()
        or lambda_sh <= 0
        or beta <= 0
    ):
        raise ValueError(
            "lambda_sh и beta должны быть конечными и положительными."
        )

    def root(function, left, right):
        return brentq(
            function,
            left,
            right,
            xtol=np.nextafter(0.0, 1.0),
            rtol=1e-12,
            maxiter=1000,
        )

    def sech(value):
        exponent = np.exp(-value)
        return 2 * exponent / (1 + exponent * exponent)

    def convert(mu_values):
        temperatures = sorted(
            float(mu * mu / (8 * beta)) for mu in mu_values
        )

        if any(
            temperature <= 0 or not np.isfinite(temperature)
            for temperature in temperatures
        ):
            raise FloatingPointError(
                "Температура вышла за пределы точности float64."
            )

        return temperatures

    if beta == 3 / 8:
        if lambda_sh <= 3:
            return []

        def residual_y(value):
            return (
                lambda_sh - 2.5 - value / 2
                - 1 / (2 * (1 + np.tanh(value)))
            )

        value = root(residual_y, 0.0, 2 * lambda_sh - 5)
        return convert([sech(value)])

    def residual(mu):
        if mu == 0:
            return 3 - 8 * beta

        if mu == 1:
            return lambda_sh - 8 * beta

        q = np.sqrt((1 - mu) * (1 + mu))
        ell = np.log1p(q) - np.log(mu)

        return (
            mu * mu
            * (
                lambda_sh - 2.5
                - ell / 2
                - 1 / (2 * (1 + q))
            )
            - (8 * beta - 3)
        )

    if lambda_sh <= 2.5:
        if not lambda_sh / 8 < beta < 3 / 8:
            return []

        return convert([root(residual, 0.0, 1.0)])

    mu_c = sech(2 * lambda_sh - 5)
    beta_min = (5 + np.tanh(2 * lambda_sh - 5)) / 16
    beta_max = max(3, lambda_sh) / 8

    if beta < beta_min or beta >= beta_max:
        return []

    if beta == beta_min:
        return convert([mu_c])

    roots = []

    if beta < 3 / 8:
        roots.append(root(residual, 0.0, mu_c))

    if beta < lambda_sh / 8:
        roots.append(root(residual, mu_c, 1.0))

    return convert(roots)


def has_theta_solution(beta, lambda_sh):
    """Check whether equation (3) has a root with ``0 < 8*beta*theta < 1``."""
    if not all(
        isfinite(value) and value > 0
        for value in (beta, lambda_sh)
    ):
        raise ValueError(
            "beta и lambda_sh должны быть положительными и конечными."
        )

    if lambda_sh <= 2.5:
        return lambda_sh / 8 < beta < 3 / 8

    exponent = exp(10 - 4 * lambda_sh)
    beta_min = 3 / 8 - exponent / (8 * (1 + exponent))
    beta_max = max(3.0, lambda_sh) / 8

    return beta_min <= beta < beta_max
