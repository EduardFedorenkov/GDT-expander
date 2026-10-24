"""Графики W/W0, q_e, u/u_* и n/n_* для найденных решений."""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np

from simple_cx_model.config import load_cx_config
from .scan_neutral import find_density_solutions
from .solver_general import solve_at_theta
from .geometry import add_geometry_arguments, geometry_from_args
from .profiles import compute_profiles
from .inlet import make_inlet_momentum, add_inlet_arguments
from .excitation import add_excitation_arguments


def plot_profiles(profiles, title=None):
    """Наложить все переданные решения на четыре графика."""
    profiles = list(profiles)
    if not profiles:
        raise ValueError("Нет профилей для построения.")

    figure, axes = plt.subplots(
        5, 1, figsize=(8.5, 12), sharex=True
    )

    keys = (
        "omega",
        "Aq_e_over_J0T",
        "chi_q",
        "u_over_u_star",
        "n_over_n_star",
    )
    labels = (
        r"$W/W_0$",
        r"$Aq_e/(J_0T)$",
        r"$\chi_q=|q_e|/(nT v_{Te})$",
        r"$u/u_*$",
        r"$n/n_*$",
    )

    for profile in profiles:
        z = np.asarray(profile["z_cm"], dtype=float)
        values = [
            np.asarray(profile[key], dtype=float)
            for key in keys
        ]

        if (
            z.ndim != 1
            or z.size < 2
            or not np.isfinite(z).all()
            or np.any(np.diff(z) <= 0)
            or any(
                y.shape != z.shape or not np.isfinite(y).all()
                for y in values
            )
        ):
            raise ValueError("Некорректные массивы профилей.")

        label = (
            f"T={profile['T_eV']:.4g} eV, "
            f"a={profile['a_cm']:.3f} cm, "
            f"{profile['branch']}"
        )

        for axis, y in zip(axes, values):
            axis.plot(z, y, linewidth=2, label=label)

    for axis, label in zip(axes, labels):
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)

    # Не обрезаем отрицательный теплопроводный поток.
    axes[1].axhline(
        0, color="black", linestyle="--", linewidth=0.8
    )
    axes[1].ticklabel_format(
        axis="y", style="sci", scilimits=(0, 0)
    )

    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel(r"$z$ [cm]")

    axes[2].set_ylim(bottom=0)

    if title:
        axes[0].set_title(title)

    figure.tight_layout()
    return figure, axes


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
    parser.add_argument(
        "--density",
        type=float,
        help="Плотность нейтралов [см^-3]; иначе из JSON.",
    )
    parser.add_argument(
        "--cx-mode",
        choices=("constant", "temperature"),
        default="constant",
        help="constant: k_cx из JSON; temperature: k_cx(T, u=0).",
    )
    add_geometry_arguments(parser)
    add_inlet_arguments(parser)
    add_excitation_arguments(parser)
    parser.add_argument("--points", type=int, default=1600)
    parser.add_argument("--b-points", type=int, default=100)
    parser.add_argument(
        "--save", type=Path,
        help="Сохранить рисунок в файл.",
    )
    parser.add_argument("--no-show", action="store_true")

    args = parser.parse_args()

    if args.points < 100 or args.b_points < 20:
        parser.error("Нужно --points >= 100 и --b-points >= 20.")
    try:
        geometry = geometry_from_args(args)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    parameters = load_cx_config(args.config)
    density = (
        parameters["n0_cm3"]
        if args.density is None
        else args.density
    )

    roots, _ = find_density_solutions(
        density,
        parameters,
        n_points=args.points,
        b_points=args.b_points,
        cx_mode=args.cx_mode,
        geometry=geometry,
        inlet_mode=args.inlet_momentum,
        excitation_scale=args.excitation_scale,
    )

    if not roots:
        print(f"При n0={density:.6e} см^-3 корни не найдены.")
        return

    profiles = []
    inlet = make_inlet_momentum(parameters, geometry, mode=args.inlet_momentum)

    for root in roots:
        state = solve_at_theta(
            root["theta"], parameters["beta"], root["D"],
            parameters["lambda_sh"], geometry, n_points=args.points,
            rtol=1e-10, atol=1e-12,
            inlet_momentum=inlet,
            E_exc=root["E_exc"],
        )
        if state["status"] != "energy_matched":
            raise RuntimeError("Не удалось восстановить решение.")
        if abs(state["R_M"]) > root["R_M_tolerance"]:
            raise RuntimeError("Не выполнено входное условие импульса.")

        profile = compute_profiles(
            state, parameters, geometry=geometry
        )

        if abs(profile["R_E"]) > 1e-5:
            raise RuntimeError(
                "Не выполнено входное условие энергии."
            )

        profiles.append(profile)

        print(
            f"T={profile['T_eV']:.6g} eV, "
            f"a={profile['a_cm']:.6g} cm, "
            f"R_E={profile['R_E']:.2e}, R_M={profile['R_M']:.2e}, "
            f"mu_a={profile['mu_a']:.6g}, G_a={profile['mu_target']:.6g}, "
            f"q_e(L)={profile['q_e_erg_cm2_s'][-1]:.6g} "
            "erg/(cm^2*s)"
        )

    print(
        f"u_*={parameters['U_star_cm_per_s']:.6g} cm/s, "
        f"n_*={parameters['n_star_cm3']:.6g} cm^-3, "
            f"geometry={geometry.name}, excitation eta={args.excitation_scale:g}"
    )

    figure, _ = plot_profiles(
        profiles,
        title=(
            f"n0={density:.3e} cm^-3, "
            f"{geometry.name}, CX={args.cx_mode}, "
            f"excitation eta={args.excitation_scale:g}"
        ),
    )

    if args.save:
        figure.savefig(
            args.save, dpi=180, bbox_inches="tight"
        )

    if args.no_show:
        plt.close(figure)
    else:
        plt.show()


if __name__ == "__main__":
    main()
