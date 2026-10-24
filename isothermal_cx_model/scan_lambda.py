"""Скан по lambda_sh и n0 с сохранением всех найденных корней."""

from collections import Counter
from pathlib import Path
import argparse
import csv
import json

import numpy as np

from simple_cx_model.config import load_cx_config
from .scan_neutral import scan_neutrals
from .inlet import add_inlet_arguments
from .excitation import add_excitation_arguments, validate_excitation_scale
from .geometry import (add_geometry_arguments, geometry_from_args,
                       quadratic_area_legacy)


ROOT_FIELDS = (
    "lambda_sh", "n0_cm3", "cx_mode", "branch",
    "T_eV", "theta", "a_cm", "alpha", "D", "b",
    "R_E", "R_E_check", "delta_alpha",
    "R_M", "R_M_check", "root_kind", "geometry",
    "inlet_momentum", "chi", "T_kin_eV", "K_a", "kinetic_path_check",
    "M_kin_dyn", "mu_a", "mu_target", "inlet_target_sonic_margin",
    "R_M_scaled", "R_M_tolerance",
    "excitation_scale", "E_exc",
)

GRID_FIELDS = (
    "lambda_sh", "n0_cm3", "cx_mode", "n_roots",
    "min_R_M", "max_R_M", "min_abs_R_M",
    "has_unresolved_points", "statuses",
    "boundary_refinements", "boundary_evaluations", "boundary_resolution_limited",
    "inlet_momentum", "chi", "kinetic_path_incomplete", "kinetic_domain_reasons",
    "excitation_scale",
)


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


def scan_lambda_neutral(
    lambdas,
    densities,
    parameters,
    cx_mode="constant",
    n_points=1600,
    b_points=100,
    geometry=None,
    *, inlet_mode="kinetic", excitation_scale=0.0,
):
    """Для каждой lambda запускать существующий скан по n0.

    Возвращает:
      roots — отдельная строка для каждого найденного корня;
      grid — строка для каждой проверенной пары (lambda, n0).

    Исходный словарь parameters не изменяется.
    """
    geometry = quadratic_area_legacy() if geometry is None else geometry
    lambdas = positive_grid(lambdas, "lambda_sh")
    densities = positive_grid(densities, "n0")
    excitation_scale = validate_excitation_scale(excitation_scale)

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
            f"lambda_sh={lam:g}, CX={cx_mode}, excitation eta={excitation_scale:g}",
            flush=True,
        )

        found, diagnostic = scan_neutrals(
            densities,
            current,
            n_points=n_points,
            b_points=b_points,
            cx_mode=cx_mode,
            geometry=geometry,
            inlet_mode=inlet_mode,
            excitation_scale=excitation_scale,
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
                "excitation_scale": excitation_scale,
                "n_roots": counts[row["n0_cm3"]],
                "min_R_M": row["min_R_M"],
                "max_R_M": row["max_R_M"],
                "min_abs_R_M": row["min_abs_R_M"],
                "has_unresolved_points": row["has_unresolved_points"],
                "statuses": json.dumps(row["statuses"], sort_keys=True),
                "boundary_refinements": row["boundary_refinements"],
                "boundary_evaluations": row["boundary_evaluations"],
                "boundary_resolution_limited": row["boundary_resolution_limited"],
                "inlet_momentum": row["inlet_momentum"], "chi": row["chi"],
                "kinetic_path_incomplete": row["kinetic_path_incomplete"],
                "kinetic_domain_reasons": json.dumps(
                    row["kinetic_domain_reasons"], sort_keys=True),
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
        default=Path("general_geometry_results"),
    )
    parser.add_argument("--no-plot", action="store_true")

    add_geometry_arguments(parser)
    add_inlet_arguments(parser)
    add_excitation_arguments(parser)
    args = parser.parse_args()
    try:
        geometry = geometry_from_args(args)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

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
    print_source_summary(parameters)

    roots, grid = scan_lambda_neutral(
        lambdas,
        densities,
        parameters,
        cx_mode=args.cx_mode,
        n_points=args.points,
        b_points=args.b_points,
        geometry=geometry,
        inlet_mode=args.inlet_momentum,
        excitation_scale=args.excitation_scale,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "geometry.json").write_text(
        json.dumps(geometry.to_dict(), indent=2), encoding="utf-8")
    (args.output / "parameters.json").write_text(json.dumps(dict(
        parameters, inlet_momentum=args.inlet_momentum, cx_mode=args.cx_mode,
        excitation_scale=args.excitation_scale,
    ), indent=2), encoding="utf-8")

    suffix = "_excitation" if args.excitation_scale > 0 else ""
    save_csv(
        args.output / f"roots_{args.cx_mode}{suffix}.csv",
        roots,
        ROOT_FIELDS,
    )
    save_csv(
        args.output / f"grid_{args.cx_mode}{suffix}.csv",
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
