# Трёхпараметрическая карта невязок

`residual_map.py` — первый этап автоматического поиска. Он не меняет
уравнения, граничные условия или существующий `solve_stationary()`.
При импорте модуля ничего не рассчитывается.

## Что вычисляется

Для каждого заданного `(theta, j_target, alpha)` вызывается существующий
`transport.solve_trial()`. На входе в гидродинамический слой вычисляются:

```text
F_J = log(j(a))
F_M = log(mu(a) / mu_target(alpha))
F_E = omega(a) - 1
rho = max(abs(F_J), abs(F_M), abs(F_E))
```

Это те же невязки, что использует локальный стационарный решатель.
Дополнительно сохраняются исходные `R_J`, `R_M`, `R_E`, входные значения
потоков, коэффициент `D` и производная звукового зазора на стенке.

Все трёхмерные массивы имеют порядок индексов **[alpha, j_target, theta]**.
Оси должны быть возрастающими и конечными; `theta > 0`, `j_target > 0`,
`geometry.g_min < alpha < 1`. Ось может содержать одно значение для среза.
`theta = T_eV / T_star_eV`, `alpha = a / L`. Это не координата `xi`.

`valid=True` означает только, что пробное интегрирование дало определённые
невязки. Это НЕ найденное стационарное решение. Невыполненные условия
на входе допустимы для пробного расчёта. Не вводятся ограничения `j_t<=1`,
монотонности энергии или положительности `Q_eff`.

Неудачные интегрирования хранятся со своим статусом и `NaN` вместо невязок.
Выход за диапазон таблиц ADAS не доказывает отсутствия физического решения.
Ошибки файлов и функций коэффициентов не перехватываются как «нет корней».

## Минимальный пример подключения

Код ниже предназначен для отдельного скрипта в корне проекта. Его запуск
уже выполнит физические пробные интегрирования; при подготовке модуля этот
пример **не запускался**. Диапазоны сетки иллюстративные, не доказанно
достаточные для поиска всех решений.

```python
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Только сохранение изображений.
import matplotlib.pyplot as plt

from isothermal_model_adas.config import load_config
from isothermal_model_adas.geometry import quadratic_area_legacy
from isothermal_model_adas.inlet import make_inlet_momentum
from isothermal_model_adas.recombination_adas import make_adas_coefficients
from isothermal_model_adas.cx_adas import make_D
from isothermal_model_adas.ionization_adas import make_ionization_coefficients
from isothermal_model_adas.exitation_adas import make_E_exc
from isothermal_model_adas.residual_map import (
    scan_residuals, choose_starts, plot_residuals, save_residuals,
)

p = load_config(Path('isothermal_model_adas/config.json'))
geometry = quadratic_area_legacy(K=500.)
inlet_mode = 'kinetic' if p['source_mode'] == 'central' else 'constant'
inlet = make_inlet_momentum(p, geometry, inlet_mode)

# Такие же коэффициенты и замыкания, как в run_stationary.py.
recombination = make_adas_coefficients(p, scale=1.)
D = make_D(p)
ionization = make_ionization_coefficients(p, scale=1.)
excitation = make_E_exc(p, scale=1.)

T_eV = np.geomspace(.2, 100., 8)
j_targets = np.geomspace(1e-3, 3., 8)  # Ионизация допускает j_target > 1.
alphas = np.linspace(.5, .99, 5)
print('Пробных интегрирований:', len(T_eV)*len(j_targets)*len(alphas))

scan = scan_residuals(
    T_eV/p['T_star_eV'], j_targets, alphas, p['beta'], p['lambda_sh'],
    recombination, geometry, D=D, ionization=ionization,
    excitation=excitation, inlet_momentum=inlet,
)
starts = choose_starts(scan, max_starts=12, separation=2)

# Не перезаписывать предыдущие результаты.
output = Path('residual_map_adas_run_001')
output.mkdir(exist_ok=False)
save_residuals(output/'residuals.npz', scan, metadata={
    'parameters': p, 'inlet_momentum': inlet_mode,
    'recombination_scale': 1., 'ionization_scale': 1., 'excitation_scale': 1.,
})

fig, axes = plot_residuals(scan, p['T_star_eV'], starts=starts)
fig.savefig(output/'projection.png', dpi=160)
plt.close(fig)

fig, axes = plot_residuals(scan, p['T_star_eV'], alpha_index=2, starts=starts)
fig.savefig(output/'slice.png', dpi=160)
plt.close(fig)

print(scan['summary'])
for start in starts:
    print(start['theta'], start['j_target'], start['alpha'])
```

Для воспроизводимости настоящего исследования следует также сохранять
контрольные суммы используемых ADAS-файлов и версию кода. `metadata` — место
для этих сведений; модуль сам не может восстановить происхождение произвольной
переданной функции коэффициентов. Сохранение и пакетная организация экспериментов
будут задачей будущего `scan_stationary.py`.

## Как смотреть карту

`plot_residuals(..., alpha_index=k)` показывает срез при заданной `alpha[k]`.
Без `alpha_index` выбирается минимальное **полное** `rho` по дискретной оси
`alpha` для каждой пары `(theta,j_target)`. Все три отображаемые невязки
относятся к одному выбранному `alpha`, а не к трём независимым минимумам.
При равенстве норм выбирается первый, то есть меньший `alpha`.

Шесть панелей: `log10(rho)`, `F_J`, `F_M`, `F_E`, выбранная `alpha` и число
допустимых пробных `alpha` во всём кубе для данной пары `(theta,j_target)`.
Серый цвет обозначает неопределённые пробы. Последняя панель показывает
число допустимых проб, включая ноль, а не маскирует их серым.

По умолчанию вертикальная ось — логарифмическая `j_target`; также доступны
`j_axis='linear'` и `j_axis='loss'` (знаковая `1-j_target`, symlog).
Не проводится интерполяция невязок или нулевых контуров через смену выбранной
`alpha` и недопустимые узлы. График — отображение выборки, не непрерывная
граница существования решений. Все значения `rho < 1e-12` показаны одним цветом.

На проекции начальные приближения разных `alpha` могут наложиться друг
на друга. На срезе показываются только маркеры с соответствующей `alpha`.
Для маркеров найденных решений аргумент `solutions` принимает только записи
с `stationary_verified=True`; сама карта решений не верифицирует.

## Выбор начальных приближений

`candidate_cells(scan)` возвращает индексы нижних углов кубиков, у которых
все восемь вершин допустимы и диапазон каждой невязки содержит ноль.
Это только эвристика: она не доказывает наличия общего нуля трёх функций.

`choose_starts()` отдаёт приоритет лучшим вершинам таких кубиков, затем
локальным минимумам `rho`, затем остальным узлам с малыми невязками.
Результат — список словарей с `theta`, `j_target`, `alpha`, индексами узла,
`source` и `grid_residual_norm`. Выбираются только реально рассчитанные
допустимые узлы, не интерполированные центры кубиков.

`separation=2` разносит старты по индексам трёхмерной сетки. Это не физическое
расстояние и не правило удаления дубликатов корней. Для грубой сетки можно
поставить `separation=1`. Кратные корни без смены знака, узкие области
допустимости и корни между редкими узлами могут быть пропущены. Нужны
проверки при сгущении сетки и разные начальные приближения.

Следующий этап: передать выбранные тройки в `solve_stationary()`, независимо
проверить сходимость и бюджеты, а затем удалить дубликаты найденных корней.
Автоматический запуск этого этапа пока намеренно не включён в карту.

## Сохранение и тесты

`save_residuals()` сохраняет полный куб в NPZ без объектных массивов,
поэтому файл читается с `np.load(path, allow_pickle=False)`. Поля
`settings_json`, `summary_json`, `metadata_json` содержат JSON-строки.
Файл не перезаписывается, родительская папка должна существовать.

Проверка нового модуля из корня проекта:

```powershell
python -B -m unittest isothermal_model_adas.tests.test_residual_map -v
```

Тесты используют искусственные невязки с известными нулями, проверяют
маски недопустимых проб, выбор стартов, проекцию, сохранение и отрисовку.
Для проверки стыковки с `transport` интегратор заменён тестовой функцией:
физические уравнения не интегрируются. Тесты не доказывают существования
решений конкретной плазменной задачи.
