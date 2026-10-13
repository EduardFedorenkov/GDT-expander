"""Построение T(n0) и a(n0) по roots_*.csv."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_roots_csv(path):
    """Прочитать CSV, записанный скриптом scan_lambda.py."""
    numeric = ("lambda_sh", "n0_cm3", "T_eV", "a_cm")
    rows = []

    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=";")

        missing = set(numeric) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"Отсутствуют колонки: {', '.join(sorted(missing))}"
            )

        for line, row in enumerate(reader, start=2):
            try:
                for name in numeric:
                    row[name] = float(row[name])
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Некорректное число в строке {line}"
                ) from exc

            values = [row[name] for name in numeric]
            if not np.isfinite(values).all() or min(values) <= 0:
                raise ValueError(
                    f"Неположительное или неконечное значение: строка {line}"
                )

            rows.append(row)

    return rows


def plot_roots(rows, lambdas=None):
    """Каждое решение отображается отдельно, без объединения корней."""
    if lambdas is not None:
        for lam in lambdas:
            if not any(
                np.isclose(r["lambda_sh"], lam, rtol=1e-9, atol=0)
                for r in rows
            ):
                raise ValueError(f"В CSV нет решений для lambda_sh={lam:g}")

        rows = [
            r for r in rows
            if any(
                np.isclose(r["lambda_sh"], lam, rtol=1e-9, atol=0)
                for lam in lambdas
            )
        ]

    if not rows:
        raise ValueError("Нет решений для построения графиков")

    groups = sorted({
        (r.get("cx_mode", ""), r["lambda_sh"]) for r in rows
    })
    branches = sorted({r.get("branch", "") for r in rows})
    mixed_modes = len({mode for mode, _ in groups}) > 1

    markers = {"monotone": "o", "with_maximum": "^", "tangent": "s"}
    colors = plt.get_cmap("tab10", len(groups))

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    for i, (mode, lam) in enumerate(groups):
        for branch in branches:
            selected = [
                r for r in rows
                if r.get("cx_mode", "") == mode
                and r["lambda_sh"] == lam
                and r.get("branch", "") == branch
            ]
            if not selected:
                continue

            label = rf"$\lambda_{{\rm sh}}={lam:.4g}$"
            if mixed_modes:
                label += f", {mode}"
            if len(branches) > 1:
                label += f", {branch}"

            n0 = [r["n0_cm3"] for r in selected]

            for ax, name in zip(axes, ("T_eV", "a_cm")):
                ax.scatter(
                    n0,
                    [r[name] for r in selected],
                    s=20,
                    color=colors(i),
                    marker=markers.get(branch, "x"),
                    label=label,
                )

    axes[0].set_ylabel(r"$T$ [eV]")
    axes[1].set_ylabel(r"$a$ [cm]")
    axes[1].set_xlabel(r"$n_0$ [cm$^{-3}$]")

    axes[0].legend(
        fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
    )

    for ax in axes:
        ax.set_xscale("log")
        ax.grid(alpha=0.25)

    fig.tight_layout()
    return fig, axes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--lambdas", type=float, nargs="+")
    parser.add_argument("--save", type=Path)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    try:
        rows = load_roots_csv(args.csv_path)
        fig, _ = plot_roots(rows, args.lambdas)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    if args.save:
        fig.savefig(args.save, dpi=200, bbox_inches="tight")

    if not args.no_show:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
