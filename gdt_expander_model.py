#!/usr/bin/env python3
# ============================================================================
# Two-Point Model для детачмента плазмы
# Единицы: Гауссова система (CGS)
# ============================================================================

import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

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
    E_MAR        = 15.0 * eV2erg  # стоимость ионизации, эрг (~25 эВ)
    E_fast_ions  = 5.0  * eV2erg  # стоимость ионизации, эрг (~5 эВ)
    alpha        = 7.0            # коэффициент передачи энергии
    f_mom        = 2.0            # коэффициент потерь импульса (без трения)

# ============================================================================
# КОЭФФИЦИЕНТЫ СКОРОСТЕЙ РЕАКЦИЙ
# ============================================================================

def k_ion(Te):
    """Коэффициент ионизации H, см³/с"""
    Te = max(Te, 0.5)
    return 5.0e-14 * np.sqrt(Te) * np.exp(-13.6/Te) / (1 + np.sqrt(Te/10))

def k_ion_H2(Te):
    """Ионизация H₂, см³/с, Te в эВ"""
    Te = max(Te, 0.5)
    return 3.0e-14 * np.sqrt(Te) * np.exp(-15.0/Te) / (1 + np.sqrt(Te/15))

def k_EIR(Te):
    """EIR рекомбинация, см³/с, Te в эВ"""
    Te = max(Te, 0.5)
    return 3.0e-19 * Te**(-4.5)

def k_MAR(Te):
    """Коэффициент MAR, см³/с, Te в эВ"""
    Te = max(Te, 0.5)
    k_base = 1.0e-13 * np.exp(-((Te - 2.5)/2.5)**2)
    
    if Te > 7.0:
        k_base *= np.exp(-(Te - 7.0)/3.0)
    return k_base

def sigma_CX(E_ion):
    """Сечение CX, см², E_ion в эВ"""
    # Более точная аппроксимация для 100 эВ - 10 кэВ
    if E_ion < 100:
        return 6.0e-15  # Немного больше при низких энергиях
    elif E_ion < 10000:
        return 5.0e-15 * (1 + 25000/E_ion)**(-0.3)
    else:
        return 4.0e-15  # Спад при высоких энергиях

def k_CX(T_i_mirror):
    """Коэффициент CX, см³/с, T_i_mirror в эВ"""
    v_ion = np.sqrt(2 * T_i_mirror * Const.eV2erg / Const.m_i)  # см/с
    return sigma_CX(T_i_mirror) * v_ion

# ============================================================================
# TWO-POINT MODEL
# ============================================================================

class TPM:
    def __init__(self, params):
        """
        Параметры:
        - q1: поток энергии на входе, эрг/(см^2·с)
        - P1: давление на входе, дин/см^2
        - f_exp: коэффициент расширения магнитной трубки (B2/B1)
        - L: длина силовой линии, см
        - n_n: плотность нейтралов, см⁻³
        """
        self.n1          = params['n1']
        self.T1          = params['T1']
        self.f_exp       = params['f_exp']
        self.L           = params['L']
        self.A1          = params['A1']
        self.n_n         = params['n_n']
        self.dis_ratio   = params['dis_ratio']
        self.vibr_ratio  = params['vibr_ratio']
        
        self.V = self.L * self.A1 / 3 * (1 + self.f_exp + np.sqrt(self.f_exp))

        self.n_H     = self.n_n * self.dis_ratio
        self.n_H2    = self.n_n * (1 - self.dis_ratio) * (1 - self.vibr_ratio)
        self.n_H2_nu = self.n_n * (1 - self.dis_ratio) * self.vibr_ratio

        T = self.T1 * Const.eV2erg
        self.q1             = 0.01 * self.n1 * T * np.sqrt(T / Const.m_e)
        self.P1             = self.n1 * T
        self.n1_fast_ions   = self.n1 * np.exp(-self.n_H2 * sigma_CX(self.T1) * self.L)
    
    def solve(self):
        """Решает систему уравнений TPM"""
        
        # 1. Определяем физически разумный интервал для T2
        #    Detachment обычно происходит в диапазоне 0.5 - 50 эВ
        T_min, T_max = 1e-4, 1e1
        
        # 2. Строим график функции баланса энергии на этом интервале (двойной логарифмический масштаб)
        T_values = np.linspace(T_min, T_max, 100000)
        f_values = [self._energy_balance(T) for T in T_values]

        plt.figure(figsize=(8, 5))
        # Используем |f(T2)|, чтобы можно было перейти в логарифмический масштаб по оси Y
        plt.loglog(T_values, np.abs(f_values), label="|energy_balance(T2)|")
        plt.xlabel("T2, eV (log scale)")
        plt.ylabel("|energy_balance(T2)| (log scale)")
        plt.title("Баланс энергии как функция T2 (log-log)")
        plt.grid(True, which="both", ls="--", lw=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # В качестве приблизительного решения для T2 берём точку,
        # где |energy_balance| минимально на построенной сетке.
        idx_min = int(np.argmin(np.abs(f_values)))
        T2 = float(T_values[idx_min])

        n2 = self._compute_n2(T2)
        j2 = self._c_s(T2) * n2 * Const.e * 3.3356e-6
        
        return {
            'T2': T2,                                                       # температура на стенке, эВ
            'n2': n2,                                                       # плотность на стенке, см⁻³
            'j2': j2,                                                       # плотность тока, А / м^2
            'S_ion_H': k_ion(T2) * n2 * self.n_H,                           # ионизация H, см⁻³·с⁻¹
            'S_ion_H2': k_ion_H2(T2) * n2 * (self.n_H2_nu + self.n_H2),     # ионизация H, см⁻³·с⁻¹
            'S_rec': k_EIR(T2) * n2 * self.n_H,                             # ионизация H, см⁻³·с⁻¹
            'S_MAR': k_MAR(T2) * n2 * self.n_H2_nu,                         # ионизация H, см⁻³·с⁻¹
            'S_cx': k_CX(self.T1) * self.n1_fast_ions * self.n_H,           # ионизация H, см⁻³·с⁻¹
        }
    
    def _energy_balance(self, T2):
        """Уравнение баланса энергии: q1 = q2*f_exp + Q_ion_H + Q_ion_H2 + Q_rec + Q_MAR + Q_cx"""        
        n2 = self._compute_n2(T2)
        c_s = self._c_s(T2)
        
        # q2
        q_wall = Const.alpha * n2 * T2 * Const.eV2erg * c_s
        
        # Q_ion_H
        S_ion_H = k_ion(T2) * n2 * self.n_H
        Q_ion_H = Const.E_ion * S_ion_H * self.V

        # Q_ion_H2
        S_ion_H2 = k_ion_H2(T2) * n2 * (self.n_H2_nu + self.n_H2)
        Q_ion_H2 = Const.E_ion * S_ion_H2 * self.V

        # Q_rec
        S_rec = k_EIR(T2) * n2 * self.n_H
        Q_rec = Const.E_rec * S_rec * self.V

        # Q_MAR
        S_MAR = k_MAR(T2) * n2 * self.n_H2_nu
        Q_MAR = Const.E_MAR * S_MAR * self.V

        # Q_cx
        S_cx = k_CX(self.T1) * self.n1_fast_ions * self.n_H
        Q_cx = self.T1 * Const.eV2erg * S_cx * self.V
        
        # Баланс
        return (q_wall * self.f_exp + Q_ion_H + Q_ion_H2 + Q_rec + Q_MAR + Q_cx) - self.q1

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
        'n1':           1.7e11,  # см^(-3)
        'T1':           100,     # эВ
        'f_exp':        100,     # расширение магнитной трубки
        'L':            180,     # длина, см (1.8 м)
        'A1':           16,      # площадь поперечного сечения, см^2
        'n_n':          1e11,    # плотность нейтралов, см⁻³
        'dis_ratio':    0.3,     # доля H
        'vibr_ratio':   0.7,    # доля H2_vibr
    }

    n_H     = params['n_n'] * params['dis_ratio']
    n_H2    = params['n_n'] * (1 - params['dis_ratio']) * (1 - params['vibr_ratio'])
    n_H2_nu = params['n_n'] * (1 - params['dis_ratio']) * params['vibr_ratio']
    
    # Решаем
    model = TPM(params)
    result = model.solve()
    
    # Вывод результатов
    print(f"\nInitial params:")
    print(f"  n1         = {params['n1']:.2e} cm^-3")
    print(f"  T1         = {params['T1']:.2e} eV")
    print(f"  f_exp      = {params['f_exp']}")
    print(f"  L          = {params['L']} cm")
    print(f"  A1         = {params['A1']} cm^2")
    print(f"  n_H        = {n_H:.2e} cm^-3")
    print(f"  n_H2       = {n_H2:.2e} cm^-3")
    print(f"  n_H2_vibr  = {n_H2_nu:.2e} cm^-3")

    
    q_wall = Const.alpha * result['n2'] * result['T2'] * Const.eV2erg * np.sqrt(result['T2'] * Const.eV2erg / Const.m_i)
    q1     = 0.5 * params['n1'] * params['T1'] * Const.eV2erg * np.sqrt(params['T1'] * Const.eV2erg / Const.m_e)

    print(f"\nResults:")
    print(f"  T2        = {result['T2']} eV")
    print(f"  n2        = {result['n2']:.2e} cm^-3")
    print(f"  j2        = {result['j2']:.2e} A / m^2")
    print(f"  q2        = {q_wall:.2e} erg / cm^2")
    print(f"  q1        = {q1:.2e} erg / cm^2")
    print(f"  S_ion_H   = {result['S_ion_H']:.2e} cm^-3 * s^-1")
    print(f"  S_ion_H2  = {result['S_ion_H2']:.2e} cm^-3 * s^-1")
    print(f"  S_MAR     = {result['S_MAR']:.2e} cm^-3 * s^-1")
    print(f"  S_rec     = {result['S_rec']:.2e} cm^-3 * s^-1")
    print(f"  S_cx      = {result['S_cx']:.2e} cm^-3 * s^-1")
    
    # Проверка детачмента
    detached = (result['S_MAR'] + result['S_rec']) > (result['S_ion_H'] + result['S_ion_H2'])
    print(f"\n  Режим: {'DETACHED (MAR доминирует)' if detached else 'ATTACHED'}")
    print("=" * 60)