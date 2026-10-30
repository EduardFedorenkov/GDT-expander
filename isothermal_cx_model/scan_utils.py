"""Shared scan validation, output and plotting helpers; no solver imports."""

from pathlib import Path
import csv

import numpy as np


def print_source_summary(parameters):
    """Show source inputs and integrated throat flows, NOT flux densities.

    Use the values already derived by load_cx_config, without recomputing
    or changing the boundary conditions. n_c is not the scanned neutral n0.
    """
    print(
        "\nSource and mirror-throat reference flows (CGS; temperature in eV):\n"
        f"  n_c = {parameters['n_c_cm3']:.6e} cm^-3, "
        f"T_c = {parameters['T_c_eV']:.6g} eV\n"
        f"  n_m = {parameters['n_m_cm3']:.6e} cm^-3, "
        f"A_m = {parameters['A_m_cm2']:.6g} cm^2, "
        f"chi_E = {parameters['chi_E']:.6g}\n"
        f"  J_0 = {parameters['J0_per_s']:.6e} particles/s\n"
        f"  M_0 = {parameters['M0_dyn']:.6e} dyn (g*cm/s^2)\n"
        f"  W_0 = {parameters['W0_erg_per_s']:.6e} erg/s",
        flush=True,
    )


def positive_grid(values, name):
    values = np.asarray(values, dtype=float)

    if (
        values.ndim != 1
        or values.size == 0
        or not np.isfinite(values).all()
        or np.any(values <= 0)
    ):
        raise ValueError(
            f"{name}: нужна непустая положительная конечная сетка."
        )

    return np.unique(values)


def save_csv(path, records, fields):
    """CSV с разделителем ';'; неопределённые минимумы — пустые ячейки."""
    with Path(path).open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, delimiter=";"
        )
        writer.writeheader()

        for record in records:
            row = {}

            for key in fields:
                value = record.get(key, "")

                if (
                    isinstance(value, (float, np.floating))
                    and not np.isfinite(value)
                ):
                    value = ""

                row[key] = value

            writer.writerow(row)


def plot_grid(grid):
    """Цвет показывает число найденных корней в проверенной точке."""
    import matplotlib.pyplot as plt

    counts = np.array(
        [row["n_roots"] for row in grid], dtype=int
    )
    maximum = max(1, int(counts.max()))

    figure, axis = plt.subplots(figsize=(8, 5))

    points = axis.scatter(
        [row["n0_cm3"] for row in grid],
        [row["lambda_sh"] for row in grid],
        c=counts,
        s=45,
        cmap="viridis",
        vmin=0,
        vmax=maximum,
    )

    figure.colorbar(
        points,
        ax=axis,
        ticks=range(maximum + 1),
        label="Number of roots found",
    )

    axis.set_xscale("log")
    axis.set_xlabel(r"$n_0$ [cm$^{-3}$]")
    axis.set_ylabel(r"$\lambda_{\rm sh}$")
    axis.grid(alpha=0.2)
    figure.tight_layout()

    return figure, axis
