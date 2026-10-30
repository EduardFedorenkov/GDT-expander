# Изотермическая 1D-модель расширителя

Пакет пока сохраняет имя isothermal_cx_model. Имена модулей отражают их роль,
а не набор реакций. Физика при переименовании не менялась.

- transport.py: пробное интегрирование j, mu, omega и проверка балансов.
- stationary.py: локальный/мультистартовый поиск проверенных решений.
- residual_map.py: карта невязок и выбор стартов.
- parameter_scan.py: пакетный скан физических параметров, сохранение и resume.
- profiles.py: размерные профили и графики.
- diagnostics.py, scan_utils.py: диагностика и общие вспомогательные функции.
- cx.py, excitation.py, recombination.py, ionization.py: коэффициенты процессов.

CLI по умолчанию включает CX(T, u=0), возбуждение, Case-A RR и ионизацию.
Низкоуровневый API сохраняет нулевые значения необязательных процессов;
для него коэффициенты необходимо передать явно. Молекул пока нет.

## Проверка из корня проекта

```powershell
python -B -m unittest discover -s tests -v
python -B -m isothermal_cx_model.trial_example --help
python -B -m isothermal_cx_model.run_stationary --help
python -B -m isothermal_cx_model.scan_stationary --help
python -B -m isothermal_cx_model.scan_lambda --help
```

Два известных решения и график профилей (n0=1e13 cm^-3, lambda=9):

```powershell
python -B -m isothermal_cx_model.run_stationary --density 1e13 --lambda-sh 9 --ionization-scale 1 --guess 5.716 7.7403 --guess 48.954 2.3913 --no-show --output check_profiles
```

Карта без уточнения корней:

```powershell
python -B -m isothermal_cx_model.scan_stationary --density 1e13 --lambda-sh 9 --ionization-scale 1 --map-only --no-show --output check_map
```

Для уточнения корней уберите --map-only и выберите новую папку.
Короткий пакетный тест в окне горячего корня:

```powershell
python -B -m isothermal_cx_model.scan_lambda --lambdas 7 7.1 --densities 9e14 1e15 --ionization-scale 1 --temperature-min 80 --temperature-max 110 --temperature-count 7 --j-min 1.35 --j-max 1.9 --j-count 7 --j-grid linear --max-starts 2 --no-show --output check_lambda
```

Файлы profiles.png, JSON и NPZ создаёт run_stationary. Параметры источника
читаются из simple_cx_model/cx_config.json. Папки вывода должны быть новыми
или пустыми; для завершённых частей одного и того же скана существует --resume.

Исторический суффикс recombination в именах выходных CSV/PNG сохранён:
это формат результатов, а не ограничение физики. Изменение исходников
меняет отпечаток кода и может запретить resume старого расчёта.
Карта и локальные уточнения не доказывают полноту поиска или устойчивость корней.

## Тесты

Тесты коэффициентов, геометрии, кинетического входа, аналитических пределов,
балансов, карт, пакетного скана и CLI используют текущий решатель.
Отключённые процессы в тестах — полезные предельные случаи, не старые реализации.
Удалены проверки специализированного одномерного поиска и старых обёрток.
