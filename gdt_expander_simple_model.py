#!/usr/bin/env python3
# ============================================================================
# Two-Point Model для детачмента плазмы
# Единицы: Гауссова система (CGS)
# ============================================================================

import numpy as np
from scipy.optimize import root_scalar
from scipy.optimize import brentq, minimize_scalar
import matplotlib.pyplot as plt
from H_ionization import eionhr, eionhr_np
from recombination import jrrec3, jrrec3_np
import warnings

# ============================================================================
# ФИЗИЧЕСКИЕ КОНСТАНТЫ (CGS)
# ============================================================================

class Const:
    e      = 4.803e-10      # заряд электрона, statC
    m_i    = 1.673e-24      # масса иона (протон), г
    m_e    = 9.109e-28      # масса электрона, г
    k_B    = 1.381e-16      # постоянная Больцмана, эрг/К
    eV2erg = 1.602e-12      # 1 эВ в эргах
    
    # Типичные значения
    E_ion        = 30.0 * eV2erg  # стоимость ионизации, эрг (~30 эВ)
    E_rec        = 13.6 * eV2erg  # стоимость рекомбинации, эрг (~13.6 эВ)
    E_MAR        = 13.6 * eV2erg  # стоимость ионизации, эрг (~25 эВ)
    alpha        = 7.0            # коэффициент передачи энергии

# ============================================================================
# КОЭФФИЦИЕНТЫ СКОРОСТЕЙ РЕАКЦИЙ
# ============================================================================

def W_rec_radiation(Te):
    return 1.69e-25 * np.sqrt(Te) * (13.6 / Te)

def W_bremsstr_radiation(Te):
    return 1.69e-25 * np.sqrt(Te)

def k_rec(Te):
    Te_array = np.asarray(Te)
    result = np.where(
        Te_array > 10, 
        jrrec3_np(Te_array, [1, 0, 2]), 
        1e-14
    )
    if np.isscalar(Te):
        return result.item()
    return result

def k_ion(Te):
    """Коэффициент ионизации H, см³/с (обёртка над eionhr)."""
    # return eionhr_np(Te, 3)
    return 1e-7

def k_ion_H2(Te):
    """Ионизация H₂, см³/с, Te в эВ"""
    # return 1e-2 * eionhr_np(Te, 3)
    return 1.0e-9

def k_MAR(Te):
    """Коэффициент MAR, см³/с, Te в эВ"""
    return 5.0e-9

def sigma_CX(E_ion):
    """Сечение CX, см², E_ion в эВ"""
    return 5.0e-15

def k_CX(T_i_mirror):
    """Коэффициент CX, см³/с, T_i_mirror в эВ"""
    return 1e-7

# ============================================================================
# TWO-POINT MODEL
# ============================================================================

class TPM:
    def __init__(self, params):
        """
        Параметры:
        - q1: поток энергии на входе, эрг/(см^2·с)
        - P1: давление на входе, дин/см^2
        - L: длина силовой линии, см
        - n_n: плотность нейтралов, см⁻³
        """
        self.n1          = params['n1']
        self.T1          = params['T1']
        self.L           = params['L']
        self.n_n         = params['n_n']

        T = self.T1 * Const.eV2erg
        self.q1             = 0.5 * self.n1 / 10 * T * np.sqrt(T / Const.m_e)
        self.P1             = self.n1 * T
        self.n1_fast_ions   = self.n1 * np.exp(-self.n_n * sigma_CX(self.T1) * self.L)
    
    def solve(self, is_plot: bool):
        """Решает систему уравнений TPM"""
        
        T_min, T_max = 1e-1, 5e1
        T_min_root, T_max_root = 1e-4, 1e3

        roots = []
        tolerance = 1e-5  # для определения знака и проверки корня
        num_intervals = 400  # количество интервалов для поиска смены знака
        T_test = np.linspace(T_min_root, T_max_root, num_intervals)

        f_test = self._energy_balance(T_test)

        # Поиск интервалов, где функция меняет знак
        sign_changes = []
        for i in range(len(f_test) - 1):
            if f_test[i] * f_test[i + 1] < -tolerance**2:  # смена знака
                sign_changes.append((T_test[i], T_test[i+1]))

        for a, b in sign_changes[:2]:
            try:
                root = brentq(self._energy_balance, a, b)
                roots.append(root)
            except ValueError:
                continue
            if len(roots) == 2:
                break
        
        if len(roots) == 2:
            T_min_plot, T_max_plot = roots[0] * 0.65, roots[1] * 1.35
        else:
            T_min_plot, T_max_plot = 1e-4, 5e3

        valid_roots = [r for r in roots if T_min <= r <= T_max]

        if len(valid_roots) == 0:
            print(f"n_n : {self.n_n:.2e}")
            print("Warning: No roots found within the specified range.")
        elif len(valid_roots) == 1:
            selected_T2 = valid_roots[0]
        elif len(valid_roots) >= 2:
            warnings.warn("Multiple roots found within the range. Selecting the smallest one.", UserWarning)
            selected_T2 = min(valid_roots)


        valid_roots_plot = [r for r in roots if T_min_plot <= r <= T_max_plot]
        
        min_value = None
        if len(roots) == 2:
            T_bracket_min = max(roots[0], T_min_plot)
            T_bracket_max = min(roots[1], T_max_plot)

            res = minimize_scalar(lambda T: self._energy_balance(T), bounds=(T_bracket_min, T_bracket_max), method='bounded')
            T2_min = res.x
            min_value = res.fun
        
        if is_plot:
            # 2. Строим график функции баланса энергии на этом интервале
            T_values = np.linspace(T_min_plot, T_max_plot, 100000)
            f_values = np.array([self._energy_balance(T) for T in T_values])

            plt.figure(figsize=(8, 5))
            plt.plot(T_values, f_values, label="energy_balance(T2)")
            if len(valid_roots_plot) > 0:
                for i, root in enumerate(valid_roots_plot):
                    root_value = self._energy_balance(root)
                    plt.scatter([root], [root_value], color='red', s=100, zorder=5, 
                            label=f"Root {i+1}: T={root:.3f}" if i == 0 else f"Root {i+1}: T={root:.3f}", 
                            edgecolors='darkred', linewidth=2)
                    # Add annotation for each root
                    plt.annotate(f'{root:.3f}', xy=(root, root_value), xytext=(root, root_value + max(abs(min(f_values)), max(f_values))*0.05),
                                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                                fontsize=9, ha='center', color='red', weight='bold')
            
            if min_value is not None:
                plt.scatter([T2_min], [min_value], color='blue', s=100, zorder=5, 
                        label=f"Min: T={T2_min:.3f}", 
                        edgecolors='darkblue', linewidth=2)
                # Add annotation for the minimum
                plt.annotate(f'({T2_min:.3f}, {min_value:.3f})', xy=(T2_min, min_value), 
                            xytext=(T2_min, min_value + max(abs(min(f_values)), max(f_values))*0.1),
                            arrowprops=dict(color='blue', alpha=0.7),
                            fontsize=9, ha='center', color='blue', weight='bold')

            plt.xlabel("T, eV")
            plt.ylabel("energy_balance(T)")
            plt.title("Баланс энергии как функция T: q2 + Q_ion_H + Q_ion_H2 + Q_MAR + Q_cx - q1 = 0")
            plt.grid(True, which="both", ls="--", lw=0.5)
            plt.legend()
            plt.tight_layout()
            plt.show()
        else:
            if min_value is not None:
                print(f"T_min : {T2_min}; q_crit : {min_value}")
            if len(roots) == 2:
                print(f"T_left : {roots[0]}")
                print(f"T_right : {roots[1]}")
        
        T2 = selected_T2
        n2 = self._compute_n2(T2)
        J2 = self._current_end(T2)
        self._critical_q1()
        
        return {
            'T2': T2,                                                       # температура на стенке, эВ
            'n2': n2,                                                       # плотность на стенке, см⁻³
            'J2': J2,                                                       # [мА/см^2]
            'S_ion_H': k_ion(T2) * n2 * self.n_n,                           # ионизация H, см⁻³·с⁻¹
            'S_ion_H2': k_ion_H2(T2) * n2 * self.n_n,                       # ионизация H2, см⁻³·с⁻¹
            'S_MAR': k_MAR(T2) * n2 * self.n_n,                             # MAR H, см⁻³·с⁻¹
            'S_cx': k_CX(self.T1) * self.n1_fast_ions * self.n_n,           # перезарядка H, см⁻³·с⁻¹
            'S_rec': k_rec(T2) * n2 * n2                                    # рекомбинация H, см⁻³·с⁻¹
        }
    
    def _energy_balance(self, T2):
        """Уравнение баланса энергии: q1 = q2 + Q_ion_H + Q_ion_H2 + Q_rec + Q_MAR + Q_cx"""        
        n2 = self._compute_n2(T2)
        c_s = self._c_s(T2)
        
        # q2
        q_wall = Const.alpha * n2 * T2 * Const.eV2erg * c_s
        
        # Q_ion_H
        S_ion_H = k_ion(T2) * n2 * self.n_n
        Q_ion_H = Const.E_ion * S_ion_H * self.L

        # Q_ion_H2
        S_ion_H2 = k_ion_H2(T2) * n2 * self.n_n
        Q_ion_H2 = Const.E_ion * S_ion_H2 * self.L

        # Q_MAR
        S_MAR = k_MAR(T2) * n2 * self.n_n
        Q_MAR = (Const.E_MAR + 3 * T2 * Const.eV2erg) * S_MAR * self.L

        # Q_rec
        S_rec = k_rec(T2) * n2 * n2
        Q_rec = (Const.E_rec + 3 * T2 * Const.eV2erg) * S_rec * self.L
        Q_rec_rad = W_rec_radiation(T2) * n2 * n2 * self.L

        # Q_bremss
        Q_bremss = W_bremsstr_radiation(T2) * n2 * n2 * self.L

        # Q_cx
        S_cx = k_CX(self.T1) * self.n1_fast_ions * self.n_n
        Q_cx = self.T1 * Const.eV2erg * S_cx * self.L
        
        # Баланс
        return (q_wall + Q_ion_H + Q_ion_H2 + Q_MAR + Q_rec + Q_rec_rad + Q_bremss + Q_cx) /self.q1 - 1

    def _current_end(self, T2):
        J0 = self._c_s(self.T1) * (self.n1 / 40) * Const.e * 3.3356e-7     # Initial ion current [мА/см^2]

        n2 = self._compute_n2(T2)
        J_ion = (k_ion(T2) + k_ion_H2(T2)) * n2 * self.n_n * self.L * Const.e * 3.3356e-7   # Current from ionization [мА/см^2]
        J_rec = (k_rec(T2) * n2 + k_MAR(T2) * self.n_n) * n2 * self.L * Const.e * 3.3356e-7 # Current from recombination [мА/см^2]

        print(f"J0 : {J0:.2e}")
        print(f"J_ion : {J_ion:.2e}")
        print(f"J_rec : {J_rec:.2e}")
        return (J0 + J_ion - J_rec) / 100

    def _critical_q1(self):
        T_min, T_max = 0.01, 50
        T_test = np.linspace(T_min, T_max, 100000)
        f_test = self._energy_balance(T_test)
        min_idx = np.argmin(f_test)
        T_crit = T_test[min_idx]
        q_crit = (f_test[min_idx] + 1) * self.q1
        print(f"T_crit : {T_crit:.2e}")
        print(f"f_crit : {f_test[min_idx]:.2e}")
        print(f"q_crit : {q_crit:.2e}")


    def _compute_f_mom(self):
        """
        Коэффициент уменьшения давления:
        - 2 -- нет трения
        - (> 2) -- есть трения
        """
        return 2.0
    
    def _compute_n2(self, T2):
        """Плотность на стенке из баланса давления"""
        # p1 = n1 * T1, p2 = n2 * T2
        # n2 = P1/(f_mom*T2)
        return self.P1 / (self._compute_f_mom() * T2 * Const.eV2erg)
    
    def _c_s(self, T_e):
        """Ионно-звуковая скорость, см/с"""
        return np.sqrt(T_e * Const.eV2erg / Const.m_i)

# ============================================================================
# ПРИМЕР ЗАПУСКА
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TWO-POINT MODEL ДЛЯ ДЕТАЧМЕНТА")
    print("=" * 60)
    
    params = {
        'n1':           1e12,  # см^(-3)
        'T1':           100,     # эВ
        'L':            60,     # длина, см (1.8 м)
        'n_n':          6.925e11,    # плотность нейтралов, см⁻³
    }
    
    # Решаем
    model = TPM(params)
    result = model.solve(True)
    
    # Вывод результатов
    print(f"\nInitial params:")
    print(f"  n1         = {params['n1']:.2e} cm^-3")
    print(f"  T1         = {params['T1']:.2e} eV")
    print(f"  L          = {params['L']} cm")
    print(f"  n_n        = {params['n_n']:.2e} cm^-3")

    
    q_wall = Const.alpha * result['n2'] * result['T2'] * Const.eV2erg * np.sqrt(result['T2'] * Const.eV2erg / Const.m_i)
    q1     = 0.5 * params['n1'] / (100) * params['T1'] * Const.eV2erg * np.sqrt(params['T1'] * Const.eV2erg / Const.m_e)

    print(f"\nResults:")
    print(f"  T2        = {result['T2']} eV")
    print(f"  n2        = {result['n2']:.2e} cm^-3")
    print(f"  J2        = {result['J2']:.2e} mA/cm^2")
    print(f"  q2        = {q_wall:.2e} erg / cm^2")
    print(f"  q1        = {q1:.2e} erg / cm^2")
    print(f"  S_ion_H   = {result['S_ion_H']:.2e} cm^-3 * s^-1")
    print(f"  S_ion_H2  = {result['S_ion_H2']:.2e} cm^-3 * s^-1")
    print(f"  S_MAR     = {result['S_MAR']:.2e} cm^-3 * s^-1")
    print(f"  S_cx      = {result['S_cx']:.2e} cm^-3 * s^-1")
    print(f"  S_rec     = {result['S_rec']:.2e} cm^-3 * s^-1")
    
    # Проверка детачмента
    detached = result['S_MAR'] + result['S_rec'] > result['S_ion_H'] + result['S_ion_H2']
    print(f"\n  Режим: {'DETACHED (MAR доминирует)' if detached else 'ATTACHED'}")
    print("=" * 60)
