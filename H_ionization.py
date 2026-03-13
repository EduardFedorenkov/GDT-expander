import numpy as np
import matplotlib.pyplot as plt


RY_EV = 1.358e1  # энергия Ридберга, эВ (ry в исходном коде)

def eionhr_np(T_e: np.ndarray, n: float) -> np.ndarray:
    """
    Аналог подпрограммы eionhr из Fortran, принимает массив T_e.

    Parameters
    ----------
    T_e : np.ndarray
        Массив температур электронов в эВ.
    n : float
        Главное квантовое число начального состояния (pcf(1)).

    Returns
    -------
    np.ndarray
        Массив коэффициентов скорости ионизации в см^3/с.
    """
    # Convert inputs to arrays for consistent processing
    T_e = np.asarray(T_e)
    
    # Initialize result array
    result = np.zeros_like(T_e, dtype=float)
    
    # Mask for positive T_e values
    mask = T_e > 0.0
    
    if np.any(mask):
        an = n
        enion = RY_EV / (an * an)
        
        beta = enion / T_e[mask]
        power = -1.5
        
        numerator = 9.56e-06 * (T_e[mask] ** power) * np.exp(-beta)
        denominator = (
            beta ** 2.33
            + 4.38 * (beta ** 1.72)
            + 1.32 * beta
        )
        result[mask] = numerator / denominator
    
    return result

def eionhr(T_e: float, n: float) -> float:
    """
    Аналог подпрограммы eionhr из Fortran.

    Parameters
    ----------
    T_e : float
        Температура электронов в эВ (pt в исходном коде).
    n : float
        Главное квантовое число начального состояния (pcf(1)).

    Returns
    -------
    float
        Коэффициент скорости ионизации в см^3/с (prate).
    """
    if T_e <= 0.0:
        return 0.0

    an = n
    enion = RY_EV / (an * an)
    beta = enion / T_e
    power = -1.5

    numerator = 9.56e-06 * (T_e ** power) * np.exp(-beta)
    denominator = (
        beta ** 2.33
        + 4.38 * (beta ** 1.72)
        + 1.32 * beta
    )
    return float(numerator / denominator)

# TO-DO: remove this function
def k_ion_approx(Te: float) -> float:
    """Аппроксимация коэффициента ионизации H, см³/с (как в gdt_expander_model до изменений)."""
    return 5.0e-14 * np.sqrt(Te) * np.exp(-13.6 / Te) / (1 + np.sqrt(Te / 10.0))


if __name__ == "__main__":
    # Диапазон температур в эВ
    T_min, T_max = 0.1, 1e3
    T_values = np.logspace(np.log10(T_min), np.log10(T_max), 300)

    # Главное квантовое число (можно менять при необходимости)
    n = 2.0

    rates_eionhr = [eionhr(T, n) for T in T_values]
    rates_k_ion = [k_ion_approx(T) for T in T_values]

    plt.figure(figsize=(8, 5))
    plt.loglog(T_values, rates_eionhr, label=f"eionhr (Fortran fit), n={n}")
    plt.loglog(T_values, rates_k_ion, label="k_ion_approx (old analytic)")
    plt.xlabel("T_e, eV (log scale)")
    plt.ylabel("Rate coefficient, cm³/s (log scale)")
    plt.title("Electron impact ionization rate coefficient (H)")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()