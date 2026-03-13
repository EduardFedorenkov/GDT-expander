"""
Python implementation of radiative recombination reaction rate coefficient calculation (jrrec3).

This module calculates the reaction rate coefficient in cm^3/s as a function of
electron temperature in eV for radiative recombination to a final state defined
in terms of its principal quantum number alone, based on the original
Fortran subroutine jrrec3 by J. J. Smith (IAEA Atomic and Molecular Data Unit).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expi


RY_EV = 13.58  # энергия Ридберга, эВ (ry в исходном коде)


def exint(x: np.ndarray) -> np.ndarray:
    """
    Вычисляет экспоненциальный интеграл E1(x) = integral_x^inf (exp(-t)/t) dt.

    В Fortran используется функция EXINT, которая обычно возвращает E1(x).
    scipy.special.expi(x) возвращает Ei(x), поэтому используем связь:
    E1(x) = -Ei(-x) для x > 0.

    Parameters
    ----------
    x : np.ndarray или float
        Аргумент экспоненциального интеграла.

    Returns
    -------
    np.ndarray или float
        Значение экспоненциального интеграла E1(x).
    """
    x = np.asarray(x)
    result = -expi(-x)
    if np.isscalar(x):
        return float(result)
    return result


def heionen(n: int, l: int, m: int, sumen: int, eth: float, kermsg: str):
    """
    Заглушка для функции heionen из Fortran.

    В оригинальном коде это внешняя подпрограмма, которая вычисляет
    энергию ионизации гелия для заданных квантовых чисел.

    Для целей данной реализации используем приближенную формулу
    для энергии связи электрона в гелии.

    Parameters
    ----------
    n : int
        Главное квантовое число.
    l : int
        Орбитальное квантовое число.
    m : int
        Магнитное квантовое число.
    sumen : int
        Флаг (в оригинале используется для выбора режима).
    eth : float
        Выходная энергия (изменяемый параметр).
    kermsg : str
        Сообщение об ошибке.

    Returns
    -------
    tuple
        (eth, kermsg) - вычисленная энергия и сообщение.
    """
    # Для He+ (Z=2, q=1) используем водородоподобную формулу с поправкой
    # Энергия связи для He+ (один электрон, Z=2): E = Z^2 * Ry / n^2 = 4 * Ry / n^2
    # Но для нейтрального гелия с одним электроном на высокой орбите
    # эффективный заряд ~1, поэтому используем эмпирическую формулу

    # Для простоты используем водородоподобное приближение с эффективным зарядом
    # Для высоких n: eth ≈ Ry / n^2 (как для водорода)
    an = float(n)

    # Эмпирическая формула для энергии ионизации возбужденного гелия
    # Для высоких n стремится к водородоподобному значению
    eth = RY_EV / (an * an)

    kermsg = ' '
    return eth, kermsg


def jrrec3_np(T_e: np.ndarray, pcf: list) -> np.ndarray:
    """
    Аналог подпрограммы jrrec3 из Fortran, принимает массив T_e.

    Параметры
    ----------
    T_e : np.ndarray
        Массив температур электронов в эВ.
    pcf : list или tuple
        Массив коэффициентов [z, q, n], где:
        - z: атомный номер мишени
        - q: зарядовое состояние иона мишени
        - n: главное квантовое число конечного состояния

    Возвращает
    -------
    np.ndarray
        Массив коэффициентов скорости рекомбинации в см^3/с.
    """
    T_e = np.asarray(T_e)

    # Initialize result array
    result = np.zeros_like(T_e, dtype=float)

    # Mask for positive T_e values
    mask = T_e > 0.0

    if np.any(mask):
        z = pcf[0]
        q = pcf[1]
        n = pcf[2]
        an = float(n)

        # Вычисление пороговой энергии eth в зависимости от z и q
        if z == 1:
            eth = RY_EV / (an * an)
        elif z == 2 and q == 1:
            # Для He+ используем heionen
            eth, _ = heionen(n, 0, 0, 1, 0.0, ' ')
        elif z == 2 and q == 2:
            # Для He++ (голый гелий)
            eth = 4.0 * RY_EV / (an * an)
        else:
            # Недопустимая комбинация - возвращаем нули
            print(f"Warning: invalid combination of coefficients in jrrec3: z={z}, q={q}")
            return result

        beta = eth / T_e[mask]

        # Formula: pfit = 5.201 * 1.0e-14 * (beta**1.5) * exp(beta) * exint(beta)
        result[mask] = 5.201 * 1.0e-14 * (beta ** 1.5) * np.exp(beta) * exint(beta)

    return result


def jrrec3(T_e: float, pcf: list) -> float:
    """
    Аналог подпрограммы jrrec3 из Fortran.

    Параметры
    ----------
    T_e : float
        Температура электронов в эВ (pt в исходном коде).
    pcf : list или tuple
        Массив коэффициентов [z, q, n], где:
        - z: атомный номер мишени
        - q: зарядовое состояние иона мишени
        - n: главное квантовое число конечного состояния

    Возвращает
    -------
    float
        Коэффициент скорости рекомбинации в см^3/с (pfit).
    """
    if T_e <= 0.0:
        return 0.0

    if len(pcf) != 3:
        raise ValueError("pcf must contain exactly 3 elements: [z, q, n]")

    z = pcf[0]
    q = pcf[1]
    n = pcf[2]
    an = float(n)

    # Вычисление пороговой энергии eth в зависимости от z и q
    if z == 1:
        eth = RY_EV / (an * an)
    elif z == 2 and q == 1:
        # Для He+ используем heionen
        eth, kermsg = heionen(n, 0, 0, 1, 0.0, ' ')
        if kermsg != ' ':
            print(f"Error in heionen: {kermsg}")
            return 0.0
    elif z == 2 and q == 2:
        # Для He++ (голый гелий)
        eth = 4.0 * RY_EV / (an * an)
    else:
        raise ValueError(f"Invalid combination of coefficients in jrrec3: z={z}, q={q}")

    beta = eth / T_e

    # Formula: pfit = 5.201 * 1.0e-14 * (beta**1.5) * exp(beta) * exint(beta)
    pfit = 5.201 * 1.0e-14 * (beta ** 1.5) * np.exp(beta) * exint(beta)

    return float(pfit)


if __name__ == "__main__":
    # Диапазон температур в эВ
    T_min, T_max = 0.1, 1e3
    T_values = np.logspace(np.log10(T_min), np.log10(T_max), 300)

    # Пример 1: Водород (Z=1)
    pcf_h = [1, 0, 2]  # z=1, q=0 (нейтральный), n=2
    rates_h = jrrec3_np(T_values, pcf_h)

    plt.figure(figsize=(10, 6))
    plt.loglog(T_values, rates_h, label=f"H (Z={pcf_h[0]}, n={pcf_h[2]})")
    plt.xlabel("T_e, eV (log scale)")
    plt.ylabel("Rate coefficient, cm³/s (log scale)")
    plt.title("Radiative recombination rate coefficient (jrrec3)")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
