"""Neutral-density scans for general magnetic geometry (isothermal CX only)."""

from pathlib import Path
import argparse
import json
import numpy as np

from simple_cx_model.config import load_cx_config
from .cx import make_D
from .geometry import (add_geometry_arguments, geometry_from_args,
                       quadratic_area_legacy)
from .scan_general import find_temperature_solutions


def find_density_solutions(n0, parameters, n_points=1600, b_points=100,
                           cx_mode="constant", geometry=None):
    if not np.isfinite(n0) or n0 <= 0:
        raise ValueError("n0 must be positive and finite.")
    geometry = quadratic_area_legacy() if geometry is None else geometry
    D = make_D(parameters, mode=cx_mode, n0_cm3=n0)
    states, diagnostic = find_temperature_solutions(
        parameters["beta"], D, parameters["lambda_sh"], geometry,
        n_points=n_points, b_points=b_points,
    )
    roots = []
    for state in states:
        roots.append({
            key: state[key] for key in (
                "theta", "b", "D", "alpha", "branch", "root_kind",
                "R_M", "R_E", "R_M_check", "R_E_check", "delta_alpha",
            )
        })
        roots[-1].update(
            n0_cm3=float(n0),
            T_eV=state["theta"]*parameters["T_star_eV"],
            a_cm=state["alpha"]*parameters["L_cm"],
            geometry=json.dumps(geometry.to_dict(), sort_keys=True),
        )
    return roots, diagnostic


def scan_neutrals(densities, parameters, n_points=1600, b_points=100,
                  cx_mode="constant", geometry=None):
    geometry = quadratic_area_legacy() if geometry is None else geometry
    densities = np.asarray(densities, dtype=float)
    if (densities.ndim != 1 or not densities.size
            or not np.isfinite(densities).all() or np.any(densities <= 0)):
        raise ValueError("densities must be a nonempty positive finite vector.")
    roots, diagnostics = [], []
    for n0 in densities:
        found, diagnostic = find_density_solutions(
            n0, parameters, n_points, b_points, cx_mode, geometry,
        )
        roots.extend(found)
        diagnostics.append({"n0_cm3": float(n0), **diagnostic})
        text = "; ".join(
            f"{r['branch']}: T={r['T_eV']:.6g} eV, "
            f"a={r['a_cm']:.6g} cm, R_M={r['R_M_check']:.2e}"
            for r in found
        )
        suffix = " [unresolved points: inspect diagnostics]" if diagnostic[
            "has_unresolved_points"] else ""
        print(f"n0={n0:.6e}: " + (text or "no roots found on this search grid")
              + suffix, flush=True)
    return roots, diagnostics


def plot_results(roots, diagnostic):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    branches = sorted({r["branch"] for r in roots})
    for branch in branches:
        selected = [r for r in roots if r["branch"] == branch]
        axes[0].scatter([r["n0_cm3"] for r in selected],
                        [r["T_eV"] for r in selected], s=14, label=branch)
        axes[1].scatter([r["n0_cm3"] for r in selected],
                        [r["a_cm"] for r in selected], s=14, label=branch)
    axes[2].plot([r["n0_cm3"] for r in diagnostic],
                 [r["min_abs_R_M"] for r in diagnostic])
    axes[0].set_ylabel("T [eV]")
    axes[1].set_ylabel("a [cm]")
    axes[2].set_ylabel("min sampled |R_M|")
    axes[2].set_xlabel(r"$n_0$ [cm$^{-3}$]")
    axes[2].set_yscale("symlog", linthresh=1e-6)
    for ax in axes:
        ax.set_xscale("log")
        ax.grid(alpha=0.25)
    if roots:
        axes[0].legend()
    fig.tight_layout()
    return fig, axes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=(
        Path(__file__).resolve().parents[1] / "simple_cx_model/cx_config.json"))
    parser.add_argument("--points", type=int, default=1600)
    parser.add_argument("--b-points", type=int, default=100)
    parser.add_argument("--densities", type=float, nargs="+")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--cx-mode", choices=("constant", "temperature"),
                        default="constant")
    add_geometry_arguments(parser)
    args = parser.parse_args()
    try:
        geometry = geometry_from_args(args)
        parameters = load_cx_config(args.config)
        densities = args.densities if args.densities is not None else np.unique(
            np.r_[np.geomspace(1e12, 1e15, 61), np.linspace(2.8e13, 3e13, 21)])
        if args.points < 100 or args.b_points < 20:
            parser.error("Need --points>=100 and --b-points>=20.")
        roots, diagnostic = scan_neutrals(
            densities, parameters, args.points, args.b_points,
            args.cx_mode, geometry,
        )
    except (ValueError, OSError, NotImplementedError) as exc:
        parser.error(str(exc))
    print("Geometry:", geometry.to_dict())
    if not args.no_plot:
        import matplotlib.pyplot as plt
        plot_results(roots, diagnostic)
        plt.show()


if __name__ == "__main__":
    main()
