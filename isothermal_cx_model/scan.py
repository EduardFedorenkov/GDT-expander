"""Scan the momentum residual versus trial temperature for general B(g)."""

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

from simple_cx_model.config import load_cx_config
from .cx import make_D
from .geometry import add_geometry_arguments, geometry_from_args
from .solver_general import solve_at_theta
from .inlet import make_inlet_momentum, add_inlet_arguments


def scan_theta(theta_values, beta, D, lambda_sh, n_points=1600, geometry=None,
               *, inlet_momentum=None):
    theta_values = np.asarray(theta_values, dtype=float)
    if (theta_values.ndim != 1 or not theta_values.size
            or not np.isfinite(theta_values).all()
            or np.any(theta_values <= 0) or np.any(np.diff(theta_values) <= 0)):
        raise ValueError("Temperatures must be positive and strictly increasing.")
    records = []
    for index, theta in enumerate(theta_values):
        state = solve_at_theta(theta, beta, D, lambda_sh, geometry, n_points,
                               inlet_momentum=inlet_momentum)
        records.append({
            "index": index,
            **{key: state[key] for key in (
                "theta", "b", "D", "alpha", "R_M", "R_E", "status")},
            "branch": state.get("branch", ""),
            "mu_target": state.get("mu_target", np.nan),
        })
    return records


def plot_scan(theta_values, records, lambda_sh, title="General magnetic geometry"):
    residual = np.full(len(theta_values), np.nan)
    for row in records:
        residual[row["index"]] = row["R_M"]
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.plot(theta_values, residual, label=r"$R_M=\mu(\alpha)-G(\alpha)$")
    axis.axhline(0, color="black", linestyle="--", linewidth=1)
    axis.set(xlabel=r"$\theta=T/T_*$", ylabel=r"$R_M$",
             title=title)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    return figure, axis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path", nargs="?", type=Path, default=(
        Path(__file__).resolve().parents[1] / "simple_cx_model/cx_config.json"))
    parser.add_argument("--cx-mode", choices=("constant", "temperature"),
                        default="constant")
    parser.add_argument("--count", type=int, default=250)
    parser.add_argument("--no-plot", action="store_true")
    add_geometry_arguments(parser)
    add_inlet_arguments(parser)
    args = parser.parse_args()
    try:
        geometry = geometry_from_args(args)
        parameters = load_cx_config(args.config_path)
        if args.count < 2:
            parser.error("--count must be >=2.")
        beta, lam = parameters["beta"], parameters["lambda_sh"]
        D = make_D(parameters, mode=args.cx_mode)
        inlet = make_inlet_momentum(parameters, geometry, mode=args.inlet_momentum)
        theta_hi = (min(1/(8*beta), 1/lam) if inlet is None else 1/lam)*(1-1e-8)
        theta_values = np.geomspace(theta_hi*1e-12, theta_hi, args.count)
        records = scan_theta(theta_values, beta, D, lam, geometry=geometry,
                            inlet_momentum=inlet)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    for row in records:
        print(f"theta={row['theta']:.8g}, status={row['status']}, "
              f"alpha={row['alpha']:.8g}, R_M={row['R_M']:.3e}")
    if not args.no_plot:
        plot_scan(theta_values, records, lam, title=geometry.name)
        plt.show()


if __name__ == "__main__":
    main()
