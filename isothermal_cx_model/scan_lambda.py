"""Скан по lambda_sh и n0 с сохранением всех найденных корней."""

from collections import Counter
from pathlib import Path
import argparse
import csv

import numpy as np

from simple_cx_model.config import load_cx_config
from .scan_neutral import scan_neutrals


ROOT_FIELDS = (
    "lambda_sh", "n0_cm3", "cx_mode", "branch",
    "T_eV", "theta", "a_cm", "alpha", "D", "b",
    "R_E", "R_E_check", "delta_alpha",
)

GRID_FIELDS = (
    "lambda_sh", "n0_cm3", "cx_mode", "n_roots",
    "min_R_E_monotone", "min_R_E_with_maximum",
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


def scan_lambda_neutral(
    lambdas,
    densities,
    parameters,
    cx_mode="constant",
    n_points=1600,
    b_points=100,
):
    """Для каждой lambda запускать существующий скан по n0.

    Возвращает:
      roots — отдельная строка для каждого найденного корня;
      grid — строка для каждой проверенной пары (lambda, n0).

    Исходный словарь parameters не изменяется.
    """
    lambdas = positive_grid(lambdas, "lambda_sh")
    densities = positive_grid(densities, "n0")

    if cx_mode not in ("constant", "temperature"):
        raise ValueError("Неизвестный cx_mode.")
    if n_points < 100 or b_points < 20:
        raise ValueError("Нужно n_points >= 100 и b_points >= 20.")

    roots, grid = [], []

    for index, lam in enumerate(lambdas, start=1):
        # Меняем только lambda, не изменяя исходный конфиг.
        current = dict(parameters, lambda_sh=float(lam))

        print(
            f"\n[{index}/{len(lambdas)}] "
            f"lambda_sh={lam:g}, CX={cx_mode}",
            flush=True,
        )

        found, diagnostic = scan_neutrals(
            densities,
            current,
            n_points=n_points,
            b_points=b_points,
            cx_mode=cx_mode,
        )

        roots.extend(
            dict(row, lambda_sh=float(lam), cx_mode=cx_mode)
            for row in found
        )

        counts = Counter(row["n0_cm3"] for row in found)

        for row in diagnostic:
            grid.append({
                "lambda_sh": float(lam),
                "n0_cm3": row["n0_cm3"],
                "cx_mode": cx_mode,
                "n_roots": counts[row["n0_cm3"]],
                "min_R_E_monotone": row["monotone"],
                "min_R_E_with_maximum": row["with_maximum"],
            })

    return roots, grid


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--config",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "simple_cx_model"
            / "cx_config.json"
        ),
    )

    parser.add_argument("--lambdas", type=float, nargs="+")
    parser.add_argument("--lambda-min", type=float, default=5.0)
    parser.add_argument("--lambda-max", type=float, default=9.0)
    parser.add_argument("--lambda-count", type=int, default=5)

    parser.add_argument("--densities", type=float, nargs="+")
    parser.add_argument("--n0-min", type=float, default=1e12)
    parser.add_argument("--n0-max", type=float, default=1e15)
    parser.add_argument("--n0-count", type=int, default=31)

    parser.add_argument("--points", type=int, default=1600)
    parser.add_argument("--b-points", type=int, default=100)

    parser.add_argument(
        "--cx-mode",
        choices=("constant", "temperature"),
        default="constant",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lambda_scan_results"),
    )
    parser.add_argument("--no-plot", action="store_true")

    args = parser.parse_args()

    if (
        args.lambda_count < 1
        or args.n0_count < 1
        or args.points < 100
        or args.b_points < 20
    ):
        parser.error("Неверные размеры сеток.")

    if not (
        np.isfinite([
            args.lambda_min, args.lambda_max,
            args.n0_min, args.n0_max,
        ]).all()
        and 0 < args.lambda_min <= args.lambda_max
        and 0 < args.n0_min <= args.n0_max
    ):
        parser.error(
            "Границы сеток должны быть положительными "
            "и упорядоченными."
        )

    lambdas = (
        args.lambdas
        if args.lambdas is not None
        else np.linspace(
            args.lambda_min, args.lambda_max, args.lambda_count
        )
    )

    densities = (
        args.densities
        if args.densities is not None
        else np.geomspace(
            args.n0_min, args.n0_max, args.n0_count
        )
    )

    parameters = load_cx_config(args.config)

    roots, grid = scan_lambda_neutral(
        lambdas,
        densities,
        parameters,
        cx_mode=args.cx_mode,
        n_points=args.points,
        b_points=args.b_points,
    )

    args.output.mkdir(parents=True, exist_ok=True)

    save_csv(
        args.output / f"roots_{args.cx_mode}.csv",
        roots,
        ROOT_FIELDS,
    )
    save_csv(
        args.output / f"grid_{args.cx_mode}.csv",
        grid,
        GRID_FIELDS,
    )

    print(
        f"\nПроверено пар: {len(grid)}, "
        f"найдено корней: {len(roots)}."
        f"\nРезультаты: {args.output.resolve()}"
    )

    if not args.no_plot:
        import matplotlib.pyplot as plt

        plot_grid(grid)
        plt.show()


if __name__ == "__main__":
    main()
