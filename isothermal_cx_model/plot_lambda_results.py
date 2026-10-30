"""Построение T(n0), a(n0) и j_t(n0), если j_target есть в roots_*.csv."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator


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

        # В старых CX-таблицах j_target отсутствует: не подменяем его единицей.
        if "j_target" in reader.fieldnames:
            numeric += ("j_target",)

        for line, row in enumerate(reader, start=2):
            try:
                for name in numeric:
                    row[name] = float(row[name])
                if 'ionization_scale' in row:
                    row['ionization_scale'] = float(row['ionization_scale'])
                    if not np.isfinite(row['ionization_scale']) or not 0 <= row['ionization_scale'] <= 1:
                        raise ValueError('Invalid ionization scale.')
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


def _temperature_tracks(rows, density_grid):
    """Visual tracks by temperature rank, NOT physical branch continuation.

    Local solution IDs are not persistent branch labels. Restart all tracks
    when the root count changes or a sampled density has no selected roots.
    """
    by_density = {}
    for row in rows:
        by_density.setdefault(row["n0_cm3"], []).append(row)

    tracks, active = [], []
    for density in density_grid:
        roots = sorted(by_density.get(density, []), key=lambda row: row["T_eV"])
        if len(roots) != len(active):
            active = [[] for _ in roots]
            tracks.extend(active)
        for track, row in zip(active, roots):
            track.append(row)
    return tracks


def plot_roots(rows, lambdas=None, *, connect=True):
    """Show every root; optionally interpolate temperature-ranked tracks.

    Lines are visual guides, not extra solutions or a stability statement.
    PCHIP in log10(n0) preserves local ranges and does not extrapolate.
    """
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

    show_target = any("j_target" in row for row in rows)
    if show_target:
        if not all("j_target" in row for row in rows):
            raise ValueError("j_target должен быть указан для каждого решения")
        target_values = np.asarray([row["j_target"] for row in rows], dtype=float)
        if not np.isfinite(target_values).all() or np.any(target_values <= 0):
            raise ValueError("j_target должен быть положительным и конечным")

    groups = sorted({
        (r.get("cx_mode", ""), r["lambda_sh"], float(r.get("ionization_scale", 0.))) for r in rows
    })
    branches = sorted({r.get("branch", "") for r in rows})
    mixed_modes = len({mode for mode, _, _ in groups}) > 1
    show_ionization = any('ionization_scale' in r for r in rows)

    markers = {"monotone": "o", "with_maximum": "^", "tangent": "s"}
    colors = plt.get_cmap("tab10", len(groups))

    columns = ["T_eV", "a_cm"]
    if show_target:
        columns.append("j_target")
    fig, axes = plt.subplots(
        len(columns), 1, figsize=(9, 10 if show_target else 7), sharex=True
    )
    density_grid = sorted({row["n0_cm3"] for row in rows})
    if show_target:
        axes[2].axhline(1., color="0.5", linestyle="--", linewidth=1, zorder=0)

    for i, (mode, lam, ionization_scale) in enumerate(groups):
        for branch in branches:
            selected = [
                r for r in rows
                if r.get("cx_mode", "") == mode
                and r["lambda_sh"] == lam
                and float(r.get('ionization_scale', 0.)) == ionization_scale
                and r.get("branch", "") == branch
            ]
            if not selected:
                continue

            label = rf"$\lambda_{{\rm sh}}={lam:.4g}$"
            if mixed_modes:
                label += f", {mode}"
            if show_ionization:
                label += f", ionization={ionization_scale:g}"
            if len(branches) > 1:
                label += f", {branch}"

            n0 = [r["n0_cm3"] for r in selected]

            if connect:
                for track in _temperature_tracks(selected, density_grid):
                    if len(track) < 2:
                        continue
                    log_n0 = np.log10([row["n0_cm3"] for row in track])
                    # Keep every original knot in the rendered curve.
                    smooth_x = np.unique(np.r_[
                        log_n0, np.linspace(log_n0[0], log_n0[-1], 200)
                    ])
                    for ax, name in zip(axes, columns):
                        smooth_y = PchipInterpolator(
                            log_n0, [row[name] for row in track]
                        )(smooth_x)
                        ax.plot(10.**smooth_x, smooth_y, color=colors(i),
                                linewidth=.85, alpha=.8, zorder=1)

            for ax, name in zip(axes, columns):
                ax.scatter(
                    n0,
                    [r[name] for r in selected],
                    s=12,
                    color=colors(i),
                    marker=markers.get(branch, "x"),
                    label=label,
                    zorder=2,
                )

    axes[0].set_ylabel(r"$T$ [eV]")
    axes[1].set_ylabel(r"$a$ [cm]")
    if show_target:
        axes[2].set_ylabel(r"$j_t = J(L)/J_0$")
    axes[-1].set_xlabel(r"$n_0$ [cm$^{-3}$]")

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
    parser.add_argument("--points-only", action="store_true",
                        help="Show only computed roots, without interpolating lines.")
    args = parser.parse_args()

    try:
        rows = load_roots_csv(args.csv_path)
        fig, _ = plot_roots(rows, args.lambdas, connect=not args.points_only)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    if args.save:
        fig.savefig(args.save, dpi=200, bbox_inches="tight")

    if not args.no_show:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
