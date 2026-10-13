"""Сканирование n0 для изотермической модели: A(z) ~ z**2, только CX."""

from pathlib import Path
import argparse
import contextlib
import io

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from simple_cx_model.config import load_cx_config
from .cx import make_D

# В solver.py сейчас есть пример, печатающий результат при импорте.
with contextlib.redirect_stdout(io.StringIO()):
    from isothermal_cx_model.solver import solve_mu, compute_omega

BRANCHES = ("monotone", "with_maximum")


def find_density_solutions(n0, parameters, n_points=1600, b_points=100,
                            cx_mode="constant",):
    """Найти полные решения при одной плотности нейтралов [см^-3].

    Возвращает (roots, minima).
    n_points — сетка интегрирования энергии.
    b_points — сетка поиска температуры через b = sqrt(8*beta*theta).
    """
    if not np.isfinite(n0) or n0 <= 0:
        raise ValueError("n0 должна быть положительной и конечной.")

    beta = parameters["beta"]
    lam = parameters["lambda_sh"]

    D = make_D(
        parameters, mode=cx_mode, n0_cm3=n0
    )

    # Потери положительны: theta < 1/lambda_sh.
    # Условие входного импульса требует b < 1.
    b_hi = min(1.0, np.sqrt(8 * beta / lam)) * (1 - 1e-8)
    b_lo = b_hi * 1e-7
    cache = {}

    def evaluate(b):
        b = float(b)

        if b not in cache:
            theta = b * b / (8 * beta)
            states = solve_mu(theta, beta, D, n_points=n_points)
            cache[b] = {}

            for state in states:
                _, residual = compute_omega(state, beta, lam)

                data = {
                    "R_E": residual,
                    "D": state["D"],
                    "alpha": state["alpha"],
                    "theta": state["theta"],
                    "b": b,
                }

                if state["branch"] == "tangent":
                    for branch in BRANCHES:
                        cache[b][branch] = data
                else:
                    cache[b][state["branch"]] = data

        return cache[b]

    def residual(b, branch):
        row = evaluate(b).get(branch)
        return np.nan if row is None else row["R_E"]

    # Смешанная сетка охватывает и очень холодные решения.
    grid = np.unique(
        np.r_[
            np.geomspace(b_lo, b_hi, b_points // 2),
            np.linspace(b_lo, b_hi, b_points),
        ]
    )

    for b in grid:
        evaluate(b)

    roots, minima = [], {}

    for branch in BRANCHES:
        values = np.array([residual(b, branch) for b in grid])
        nodes = list(grid)

        # Уточняем минимумы: между положительными значениями
        # на грубой сетке могут скрываться сразу два корня.
        for i in range(1, len(grid) - 1):
            triple = values[i - 1:i + 2]

            if (
                np.isfinite(triple).all()
                and triple[1] <= triple[0]
                and triple[1] <= triple[2]
            ):
                result = minimize_scalar(
                    lambda b: residual(b, branch),
                    bounds=(grid[i - 1], grid[i + 1]),
                    method="bounded",
                    options={"xatol": 1e-10},
                )

                if not result.success or not np.isfinite(result.fun):
                    raise RuntimeError("Не удалось уточнить минимум невязки.")

                nodes.append(result.x)

        nodes = np.unique(nodes)
        values = np.array([residual(b, branch) for b in nodes])

        finite = values[np.isfinite(values)]
        minima[branch] = float(finite.min()) if finite.size else np.nan

        root_b = []

        for left, right, f_left, f_right in zip(
            nodes[:-1], nodes[1:], values[:-1], values[1:]
        ):
            # Не соединяем участки, где нет решения импульса.
            if not np.isfinite([f_left, f_right]).all():
                continue

            if f_left == 0:
                root_b.append(left)

            if f_left * f_right < 0:
                root_b.append(
                    brentq(
                        lambda b: residual(b, branch),
                        left,
                        right,
                        xtol=1e-11,
                    )
                )

        if values[-1] == 0:
            root_b.append(nodes[-1])

        for b in sorted(set(root_b)):
            row = evaluate(b)[branch]

            roots.append({
                **row,
                "n0_cm3": float(n0),
                "D": float(row["D"]),
                "branch": branch,
                "T_eV": row["theta"] * parameters["T_star_eV"],
                "a_cm": row["alpha"] * parameters["L_cm"],
            })

    return roots, minima


def scan_neutrals(densities, parameters, n_points=1600, b_points=100,
                   cx_mode="constant",):
    """Просканировать плотности и проверить корни на удвоенной сетке."""
    roots, diagnostic = [], []
    beta = parameters["beta"]
    lam = parameters["lambda_sh"]

    for n0 in densities:
        found, minima = find_density_solutions(
            n0,
            parameters,
            n_points=n_points,
            b_points=b_points,
            cx_mode=cx_mode,
        )

        for row in found:
            refined = solve_mu(
                row["theta"],
                beta,
                row["D"],
                n_points=2 * n_points,
                rtol=2e-10,
                atol=2e-12,
            )

            state = next(
                (
                    s for s in refined
                    if s["branch"] == row["branch"]
                    or s["branch"] == "tangent"
                ),
                None,
            )

            if state is None:
                raise RuntimeError("При проверке исчезла ветвь решения.")

            _, row["R_E_check"] = compute_omega(state, beta, lam)
            row["delta_alpha"] = state["alpha"] - row["alpha"]

            if abs(row["R_E_check"]) > 1e-5:
                raise RuntimeError(
                    "Проверка энергии не пройдена: увеличьте --points."
                )

        roots.extend(found)
        diagnostic.append({"n0_cm3": float(n0), **minima})

        text = "; ".join(
            f"{r['branch']}: T={r['T_eV']:.6g} eV, "
            f"a={r['a_cm']:.6g} cm, Rcheck={r['R_E_check']:.2e}"
            for r in found
        )

        print(
            f"n0={n0:.6e}: " + (text or "корень не найден"),
            flush=True,
        )

    return roots, diagnostic


def plot_results(roots, diagnostic):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    for branch, color in zip(BRANCHES, ("tab:blue", "tab:orange")):
        selected = [r for r in roots if r["branch"] == branch]

        axes[0].scatter(
            [r["n0_cm3"] for r in selected],
            [r["T_eV"] for r in selected],
            s=14, color=color, label=branch,
        )

        axes[1].scatter(
            [r["n0_cm3"] for r in selected],
            [r["a_cm"] for r in selected],
            s=14, color=color,
        )

        axes[2].plot(
            [r["n0_cm3"] for r in diagnostic],
            [r[branch] for r in diagnostic],
            color=color,
        )

    axes[0].set_ylabel("T [eV]")
    axes[1].set_ylabel("a [cm]")
    axes[2].set_ylabel("min R_E")
    axes[2].set_xlabel(r"$n_0$ [cm$^{-3}$]")
    axes[2].set_yscale("symlog", linthresh=1e-3)
    axes[2].axhline(0, color="black", linewidth=0.8)

    for ax in axes:
        ax.set_xscale("log")
        ax.grid(alpha=0.25)

    axes[0].legend()
    fig.tight_layout()
    plt.show()


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
    parser.add_argument("--points", type=int, default=1600)
    parser.add_argument("--b-points", type=int, default=100)
    parser.add_argument("--densities", type=float, nargs="+")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--cx-mode",
        choices=("constant", "temperature"),
        default="constant",
        help="constant: k_cx из JSON; temperature: k_cx(T, u=0).",
    )

    args = parser.parse_args()

    if args.points < 100 or args.b_points < 20:
        parser.error("Требуется --points >= 100 и --b-points >= 20.")

    parameters = load_cx_config(args.config)

    # theta_exists этого парсера относится к A=const.
    # Для нашей геометрии этот флаг НЕ используем.
    if args.densities is None:
        densities = np.unique(
            np.r_[
                np.geomspace(1e12, 1e15, 61),
                # Сгущение около порога для текущего JSON.
                np.linspace(2.8e13, 3e13, 21),
            ]
        )
    else:
        densities = np.unique(args.densities)

    print(
        f"beta={parameters['beta']:.9g}, "
        f"lambda_sh={parameters['lambda_sh']:.9g}, "
        f"T_star={parameters['T_star_eV']:.9g} eV",
        flush=True,
    )

    roots, diagnostic = scan_neutrals(
        densities,
        parameters,
        n_points=args.points,
        b_points=args.b_points,
        cx_mode=args.cx_mode,
    )

    if not args.no_plot:
        plot_results(roots, diagnostic)


if __name__ == "__main__":
    main()
