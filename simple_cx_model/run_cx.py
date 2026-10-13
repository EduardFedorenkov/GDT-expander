import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simple_cx_model import (
    load_cx_config,
    find_theta,
    find_alpha,
    cx_profiles,
    plot_cx_profiles,
)


def main(config_path):
    p = load_cx_config(config_path)

    beta = p["beta"]
    D = p["D"]
    lambda_sh = p["lambda_sh"]
    L_cm = p["L_cm"]

    print(f"beta={beta:.8g}, D={D:.8g}, lambda_sh={lambda_sh:.8g}")

    if not p["theta_exists"]:
        print("Для заданных beta и lambda_sh решения theta нет.")
        return

    roots = find_theta(lambda_sh, beta)

    if roots is None or np.size(roots) == 0:
        print("Функция find_theta не нашла корней.")
        return

    count = 0

    for theta in np.atleast_1d(roots).astype(float):
        try:
            alpha = find_alpha(theta, beta, D)
        except ValueError as error:
            print(f"theta={theta:.8g}: решение отклонено — {error}")
            continue

        profiles = cx_profiles(
            theta, beta, lambda_sh, alpha, L_cm
        )
        fig, axes = plot_cx_profiles(*profiles)

        a_cm = alpha * L_cm
        T_eV = theta * p["T_star_eV"]

        count += 1
        title = f"Решение {count}: T={T_eV:.5g} эВ, a={a_cm:.5g} см"
        fig.suptitle(title)
        fig.tight_layout()

        print(
            f"{title}\n"
            f"  theta={theta:.8g}, alpha={alpha:.8g}"
        )

    if count:
        plt.show()
    else:
        print("Корни theta найдены, но допустимой границы 0 < alpha < 1 нет.")


if __name__ == "__main__":
    config_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).with_name("cx_config.json")
    )
    main(config_path)
