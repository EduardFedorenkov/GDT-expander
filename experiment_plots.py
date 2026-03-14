import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Настройка стиля графиков
rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 12
rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['legend.fontsize'] = 12

# ============================================================
# ВХОДНЫЕ ДАННЫЕ (заполняются вручную)
# ============================================================

# Независимая переменная (например, время, номер эксперимента, напряжение и т.д.)
x_data = np.array([4, 16.6, 28.2, 39.8, 51.5])

# 1. Экспериментальная плотность тока на плазмаприемнике [mA/cm^2]
# 6 - 6.5 ms
j_exp6 = np.array([13.3, 8.5, 8.33, 7.62, 11.0])
j_exp6_gas = np.array([12.4, 13.3, 9.97, 7.19, 5.8])

# 7.5 - 8 ms
j_exp8 = np.array([5.99, 5.9, 7.94, 9.69, 11.6])
j_exp8_gas = np.array([7.85, 9.98, 8.47, 8.68, 7.26])

# 2. Экспериментальная плотность потока энергии [Вт/см^2]
# 6 - 6.5 ms
q_exp6 = np.array([7.3, 14.81, 8.56, 4.33, 2.69])
q_exp6_gas = np.array([24.54, 16.03, 9.14, 5.09, 2.86])

# 7.5 - 8 ms
q_exp8 = np.array([7.61, 11.4, 8.73, 6.87, 4.71])
q_exp8_gas = np.array([16.29, 13.54, 9.71, 6.82, 4.41])

# 3. Параметры для теоретического расчета плотности тока
r_data = np.array([0, 3.2, 6.35, 9.55, 12.75, 15.6])

# 6 ms
n_e6 = np.array([1.6e13, 1.49e13, 9.71e12, 5.98e12, 6.07e12, 4.84e12])    # концентрация электронов [см^-3]
T_e6 = np.array([126.1, 127.4, 116.1, 89.8, 48, 55.9])                    # температура электронов [эВ]
n_e6_gas = np.array([1.87e13, 1.69e13, 1.17e13, 6.33e12, 5.3e12, 4.9e12]) # концентрация электронов [см^-3]
T_e6_gas = np.array([130, 108.6, 109.1, 88.4, 41.6, 42.5])                # температура электронов [эВ]

# 8 ms
n_e8 = np.array([1.23e13, 9.89e12, 8.62e12, 7.1e12, 7.23e12, 6.87e12])    # концентрация электронов [см^-3]
T_e8 = np.array([170.1, 175.9, 115, 90.3, 51.8, 55])                      # температура электронов [эВ]
n_e8_gas = np.array([1.15e13, 1.2e13, 8.1e12, 6.77e12, 6.91e12, 4.43e12]) # концентрация электронов [см^-3]
T_e8_gas = np.array([141.8, 151.5, 126.2, 78, 44.1, 43.3])                # температура электронов [эВ]

# 4. Магнитное поле

B0 = 1
Bm = 35
Bw = 35 / 135

r2x_data = r_data * np.sqrt(B0/Bw)
print(f"r2x_data : {r2x_data}")

# 5. Параметры для теоретического расчета энергии (те же n_e и T_e)

# ============================================================
# ФИЗИЧЕСКИЕ КОНСТАНТЫ
# ============================================================

e_charge = 1.602e-19       # заряд электрона [Кл]
m_i      = 1.673e-27       # масса электрона [кг]
k_B      = 1.381e-23       # постоянная Больцмана [Дж/К]
eV_to_J  = 1.602e-19       # перевод эВ в Джоули

# ============================================================
# ТЕОРЕТИЧЕСКИЕ РАСЧЕТЫ
# ============================================================

def calc_current_density_theory(n_e, T_e):
    """
    Расчет теоретической плотности тока
    
    j_th = e * n_e * sqrt(T_e / (2 * pi * m_e))
    
    Возвращает значение в [mA/cm^2]
    """
    T_e = T_e * eV_to_J                            # перевод эВ в J
    v_th = 100 * np.sqrt(T_e / (2 * np.pi * m_i))  # тепловая скорость [cm/s]
    j_th_mA_cm2 = e_charge * n_e * v_th * 1e3      # [mА/cm^2]
    return j_th_mA_cm2

def calc_energy_flux_theory(n_e, T_e):
    """
    Расчет теоретической плотности потока энергии
    
    q_th = n_e * T_e * sqrt(T_e / (2 * pi * m_i))
    
    Возвращает значение в [Вт/см^2]
    """
    T_e = T_e * eV_to_J
    v_th = 100 * np.sqrt(T_e / (2 * np.pi * m_i))   # тепловая скорость [cm/s]
    q_th_W_cm2 = n_e * T_e * v_th                   # [Вт/cm^2]
    return q_th_W_cm2

# Вычисление теоретических значений
j_theory6 = calc_current_density_theory(n_e6, T_e6)
q_theory6 = calc_energy_flux_theory(n_e6, T_e6)

j_theory8 = calc_current_density_theory(n_e8, T_e8)
q_theory8 = calc_energy_flux_theory(n_e8, T_e8)

j_gas_theory6 = calc_current_density_theory(n_e6_gas, T_e6_gas)
q_gas_theory6 = calc_energy_flux_theory(n_e6_gas, T_e6_gas)

j_gas_theory8 = calc_current_density_theory(n_e8_gas, T_e8_gas)
q_gas_theory8 = calc_energy_flux_theory(n_e8_gas, T_e8_gas)

# ============================================================
# ПОСТРОЕНИЕ ГРАФИКОВ
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Плотность тока и плотность потока тепла на плазмаприемнике', fontsize=16)

# # --- График 1: Плотность тока (эксперимент t = 6 ms) ---
# axes[0, 0].semilogy(x_data, j_exp6, 'ro-', linewidth=2, markersize=8, label='Эксперимент без напуска газа (t = 6ms)')
# axes[0, 0].semilogy(x_data, j_exp6_gas, 'go-', linewidth=2, markersize=8, label='Эксперимент с напуском газа (t = 6ms)')
# axes[0, 0].semilogy(r2x_data, j_theory6, 'bo-', linewidth=2, markersize=8, label='Оценка без напуска газа (t = 6ms)')
# axes[0, 0].semilogy(r2x_data, j_gas_theory6, 'yo-', linewidth=2, markersize=8, label='Оценка с напуском газа (t = 6ms)')
# axes[0, 0].set_xlabel('радиус [cm]')
# axes[0, 0].set_ylabel('Плотность тока [mA/cm²]')
# axes[0, 0].set_title('1. Плотность тока на плазмаприемнике (t = 6 ms)')
# axes[0, 0].grid(True, alpha=0.3)
# axes[0, 0].legend()
# 
# # --- График 2: Плотность тока (Эксперимент t = 8 ms) ---
# axes[0, 1].semilogy(x_data, j_exp8, 'ro-', linewidth=2, markersize=8, label='Эксперимент без напуска газа')
# axes[0, 1].semilogy(x_data, j_exp8_gas, 'go-', linewidth=2, markersize=8, label='Эксперимент с напуском газа')
# axes[0, 1].semilogy(r2x_data, j_theory8, 'bo-', linewidth=2, markersize=8, label='Оценка без напуска газа')
# axes[0, 1].semilogy(r2x_data, j_gas_theory8, 'yo-', linewidth=2, markersize=8, label='Оценка с напуском газа')
# axes[0, 1].set_xlabel('радиус [cm]')
# axes[0, 1].set_ylabel('Плотность тока [mA/cm²]')
# axes[0, 1].set_title('2. Плотность тока на плазмаприемнике (t = 8 ms)')
# axes[0, 1].grid(True, alpha=0.3)
# axes[0, 1].legend()
# 
# # --- График 3: Плотность потока энергии (эксперимент) ---
# axes[1, 0].semilogy(x_data, q_exp6, 'ro-', linewidth=2, markersize=8, label='Эксперимент без напуска газа')
# axes[1, 0].semilogy(x_data, q_exp6_gas, 'go-', linewidth=2, markersize=8, label='Эксперимент с напуском газа')
# axes[1, 0].semilogy(r2x_data, q_theory6, 'bo-', linewidth=2, markersize=8, label='Оценка без напуска газа')
# axes[1, 0].semilogy(r2x_data, q_gas_theory6, 'yo-', linewidth=2, markersize=8, label='Оценка с напуском газа')
# axes[1, 0].set_xlabel('радиус [cm]')
# axes[1, 0].set_ylabel('Плотность потока энергии [W/cm²]')
# axes[1, 0].set_title('3. Плотность потока энергии не плазмаприемнике (t = 6 ms)')
# axes[1, 0].grid(True, alpha=0.3)
# axes[1, 0].legend()
# 
# # --- График 4: Плотность потока энергии (теория) ---
# axes[1, 1].semilogy(x_data, q_exp8, 'ro-', linewidth=2, markersize=8, label='Эксперимент без напуска газа')
# axes[1, 1].semilogy(x_data, q_exp8_gas, 'go-', linewidth=2, markersize=8, label='Эксперимент с напуском газа')
# axes[1, 1].semilogy(r2x_data, q_theory8, 'bo-', linewidth=2, markersize=8, label='Оценка без напуска газа')
# axes[1, 1].semilogy(r2x_data, q_gas_theory8, 'yo-', linewidth=2, markersize=8, label='Оценка с напуском газа')
# axes[1, 1].set_xlabel('радиус [cm]')
# axes[1, 1].set_ylabel('Плотность потока энергии [W/cm²]')
# axes[1, 1].set_title('4. Плотность потока энергии не плазмаприемнике (t = 8 ms)')
# axes[1, 1].grid(True, alpha=0.3)
# axes[1, 1].legend()


# --- График 1: Плотность тока (эксперимент t = 6 ms) ---
axes[0, 0].plot(x_data, j_exp6, 'ro-', linewidth=2, markersize=8, label='без напуска газа')
axes[0, 0].plot(x_data, j_exp6_gas, 'go-', linewidth=2, markersize=8, label='с напуском газа')
axes[0, 0].set_xlabel('радиус [cm]')
axes[0, 0].set_ylabel('Плотность тока [mA/cm²]')
axes[0, 0].set_title('1. Эксперимент: Плотность тока на плазмаприемнике')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

# --- График 2: Плотность тока (Эксперимент t = 8 ms) ---
axes[0, 1].plot(r2x_data, j_theory6, 'ro-', linewidth=2, markersize=8, label='Оценка без напуска газа')
axes[0, 1].plot(r2x_data, j_gas_theory6, 'go-', linewidth=2, markersize=8, label='Оценка с напуском газа')
axes[0, 1].set_xlabel('радиус [cm]')
axes[0, 1].set_ylabel('Плотность тока [mA/cm²]')
axes[0, 1].set_title('2. Оценка: Плотность тока на плазмаприемнике')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

# --- График 3: Плотность потока энергии (эксперимент) ---
axes[1, 0].plot(x_data, q_exp6, 'ro-', linewidth=2, markersize=8, label='без напуска газа')
axes[1, 0].plot(x_data, q_exp6_gas, 'go-', linewidth=2, markersize=8, label='с напуском газа')
axes[1, 0].set_xlabel('радиус [cm]')
axes[1, 0].set_ylabel('Плотность потока энергии [W/cm²]')
axes[1, 0].set_title('3. Эксперимент: Плотность потока энергии не плазмаприемнике')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

# --- График 4: Плотность потока энергии (теория) ---
axes[1, 1].plot(r2x_data, q_theory6, 'ro-', linewidth=2, markersize=8, label='Оценка без напуска газа')
axes[1, 1].plot(r2x_data, q_gas_theory6, 'go-', linewidth=2, markersize=8, label='Оценка с напуском газа')
axes[1, 1].set_xlabel('радиус [cm]')
axes[1, 1].set_ylabel('Плотность потока энергии [W/cm²]')
axes[1, 1].set_title('4. Оценка: Плотность потока энергии не плазмаприемнике')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('plasma_experiment_results.png', dpi=300, bbox_inches='tight')
plt.show()
