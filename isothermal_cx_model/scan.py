from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np

from simple_cx_model.config import load_cx_config
from isothermal_cx_model.solver import solve_mu, compute_omega
from .cx import make_D


def scan_theta(theta_values, beta, D, lambda_sh, n_points=1600):
    """Вернуть таблицу theta, b, D, alpha, branch, R_E.

    D — число либо функция D(theta).
    Температуры без решений импульса не создают строк.
    Корни энергетического условия пока не ищем.
    """
    theta_values = np.asarray(theta_values, dtype=float)

    if (
        theta_values.ndim != 1
        or theta_values.size == 0
        or not np.isfinite(theta_values).all()
        or np.any(theta_values <= 0)
        or np.any(np.diff(theta_values) <= 0)
    ):
        raise ValueError("Температуры должны быть положительными и возрастать.")

    if not all(np.isfinite(v) and v > 0 for v in (beta, lambda_sh)):
        raise ValueError("beta и lambda_sh должны быть положительными.")

    if np.any(8 * beta * theta_values >= 1):
        raise ValueError("Для каждой температуры требуется 8*beta*theta < 1.")

    records = []
    for index, theta in enumerate(theta_values):
        solutions = solve_mu(theta, beta, D, n_points=n_points)

        for solution in solutions:
            _, residual = compute_omega(solution, beta, lambda_sh)
            records.append({
                "index": index,
                "theta": float(theta),
                "b": solution["b"],
                "D": solution["D"],
                "alpha": solution["alpha"],
                "branch": solution["branch"],
                "R_E": residual,
            })

    return records


def plot_scan(theta_values, records, lambda_sh):
    """Построить ветви, не соединяя участки отсутствия решений."""
    theta_values = np.asarray(theta_values, dtype=float)
    figure, axis = plt.subplots(figsize=(8, 4.8))

    branches = (
        ("monotone", "tab:blue", "Монотонная ветвь"),
        ("with_maximum", "tab:orange", "Ветвь с максимумом"),
        ("tangent", "tab:purple", "Кандидат на касание"),
    )

    for branch, color, label in branches:
        residual = np.full(theta_values.shape, np.nan)

        for row in records:
            if row["branch"] == branch:
                residual[row["index"]] = row["R_E"]

        if np.isfinite(residual).any():
            axis.plot(
                theta_values, residual,
                color=color, label=label,
                marker="o" if branch == "tangent" else None,
                linestyle="none" if branch == "tangent" else "-",
            )

    axis.axhline(0, color="black", linestyle="--", linewidth=1)

    energy_limit = 1 / lambda_sh
    if energy_limit < theta_values[-1]:
        axis.axvspan(
            0, energy_limit,
            color="tab:green", alpha=0.08,
            label=r"Необходимое условие энергии: $\theta<1/\lambda_{\rm sh}$",
        )
        axis.axvline(energy_limit, color="tab:green", linestyle=":")

    axis.set(
        xlabel=r"$\theta=T/T_*$",
        ylabel=r"$R_E=\omega(0)-1$",
        title=r"Энергетическая невязка: $A(z)\propto z^2$",
        xlim=(0, theta_values[-1]),
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=9)
    figure.tight_layout()

    return figure, axis


def main(config_path=None, cx_mode="constant"):
    if config_path is None:
        config_path = (
            Path(__file__).resolve().parents[1]
            / "simple_cx_model" / "cx_config.json"
        )

    parameters = load_cx_config(config_path)

    # Флаг theta_exists этого парсера относится к A=const.
    # Для нашей геометрии его НЕ используем.
    beta = parameters["beta"]
    D = make_D(parameters, mode=cx_mode)
    lambda_sh = parameters["lambda_sh"]

    theta_values = np.linspace(1e-6, 1 - 1e-6, 250) / (8 * beta)
    records = scan_theta(theta_values, beta, D, lambda_sh)

    print(
        f"beta={beta:.8g}, lambda_sh={lambda_sh:.8g}; "
        f"CX={cx_mode}"
    )

    for branch in ("monotone", "with_maximum", "tangent"):
        rows = [row for row in records if row["branch"] == branch]
        if rows:
            temperatures = [row["theta"] for row in rows]
            minimum = min(row["R_E"] for row in rows)
            print(
                f"{branch}: theta=[{min(temperatures):.6g}, "
                f"{max(temperatures):.6g}], min(R_E)={minimum:.6g}"
            )

    if not records:
        print("На заданной сетке решений импульса нет.")

    plot_scan(theta_values, records, lambda_sh)
    plt.show()

    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", nargs="?", type=Path)
    parser.add_argument(
        "--cx-mode",
        choices=("constant", "temperature"),
        default="constant",
    )
    args = parser.parse_args()

    main(
        config_path=args.config_path,
        cx_mode=args.cx_mode,
    )
