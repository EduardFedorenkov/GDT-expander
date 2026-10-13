"""Plotting helpers kept separate from the numerical model."""

import matplotlib.pyplot as plt
import numpy as np


def plot_solution_region(
    lambda_range=(0.05, 10.0),
    beta_range=None,
    points=2000,
):
    """Plot the parameter region where a temperature solution exists."""
    lower_lambda, upper_lambda = lambda_range

    if not 0 < lower_lambda < upper_lambda or points < 20:
        raise ValueError(
            "Нужно 0 < lambda_min < lambda_max и points >= 20."
        )

    lambda_values = np.unique(
        np.r_[
            np.linspace(lower_lambda, upper_lambda, points),
            [
                value
                for value in (2.5, 3.0)
                if lower_lambda <= value <= upper_lambda
            ],
        ]
    )

    lower_beta = np.where(
        lambda_values <= 2.5,
        lambda_values / 8,
        (5 + np.tanh(2 * lambda_values - 5)) / 16,
    )
    upper_beta = np.maximum(3.0, lambda_values) / 8

    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.fill_between(
        lambda_values,
        lower_beta,
        upper_beta,
        color="tab:blue",
        alpha=0.25,
        label=r"Есть решение: $0<\mu_t<1$",
    )

    boundary_mask = (lambda_values <= 2.5) | (lambda_values >= 3)
    if boundary_mask.any():
        axis.plot(
            lambda_values,
            np.where(boundary_mask, lambda_values / 8, np.nan),
            "--",
            color="black",
            linewidth=1.7,
            label=r"$\beta=\lambda_{\rm sh}/8$ — граница не входит",
        )

    boundary_mask = lambda_values <= 3
    if boundary_mask.any():
        axis.plot(
            lambda_values[boundary_mask],
            np.full(np.count_nonzero(boundary_mask), 3 / 8),
            "--",
            color="tab:orange",
            linewidth=1.7,
            label=r"$\beta=3/8$ — граница не входит",
        )

    boundary_mask = lambda_values > 2.5
    if boundary_mask.any():
        axis.plot(
            lambda_values[boundary_mask],
            lower_beta[boundary_mask],
            color="tab:blue",
            linewidth=2,
            label=(
                r"$\beta=[5+\tanh(2\lambda_{\rm sh}-5)]/16$"
                " — входит"
            ),
        )

    if lower_lambda <= 2.5 <= upper_lambda:
        axis.plot(
            2.5,
            5 / 16,
            "o",
            markersize=6,
            markerfacecolor="white",
            markeredgecolor="black",
            zorder=5,
        )

    axis.set(
        xlabel=r"$\lambda_{\rm sh}$",
        ylabel=r"$\beta$",
        title="Область существования решения уравнения на температуру",
        xlim=(lower_lambda, upper_lambda),
    )
    axis.set_ylim(
        beta_range
        if beta_range is not None
        else (0, 1.08 * float(upper_beta.max()))
    )
    axis.grid(alpha=0.3)
    axis.legend(fontsize=9, loc="upper left")
    figure.tight_layout()

    return figure, axis


def plot_cx_profiles(z_cm, velocity_ratio, density_ratio, energy_ratio):
    """Plot the calculated spatial profiles and return figure and axes."""
    arrays = [
        np.asarray(values, dtype=float)
        for values in (z_cm, velocity_ratio, density_ratio, energy_ratio)
    ]

    if any(values.ndim != 1 or values.size < 2 for values in arrays):
        raise ValueError(
            "Ожидаются одномерные массивы длиной >= 2."
        )

    if any(values.shape != arrays[0].shape for values in arrays):
        raise ValueError("Все массивы должны иметь одинаковый размер.")

    if any(not np.isfinite(values).all() for values in arrays):
        raise ValueError(
            "Массивы должны содержать только конечные значения."
        )

    z_cm, velocity_ratio, density_ratio, energy_ratio = arrays
    figure, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    labels = (r"$u/c_s$", r"$n/n_*$", r"$\omega=W/W_0$")

    for axis, values, label in zip(
        axes,
        (velocity_ratio, density_ratio, energy_ratio),
        labels,
    ):
        axis.plot(z_cm, values, color="tab:blue", linewidth=2)
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)

    axes[0].axhline(1.0, color="gray", linestyle="--", linewidth=1)
    axes[-1].set_xlabel(r"$z$, cm")
    figure.tight_layout()

    return figure, axes
