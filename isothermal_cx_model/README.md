# Изотермическая 1D-модель расширителя

Пакет пока сохраняет имя isothermal_cx_model. Имена модулей отражают их роль,
а не набор реакций. Физика при переименовании не менялась.

- transport.py: пробное интегрирование j, mu, omega и проверка балансов.
- stationary.py: локальный/мультистартовый поиск проверенных решений.
- residual_map.py: карта невязок и выбор стартов.
- parameter_scan.py: пакетный скан физических параметров, сохранение и resume.
- profiles.py: размерные профили и графики.
- diagnostics.py, scan_utils.py: диагностика и общие вспомогательные функции.
- sources.py: локальные источники по каналам и процессам; геометрия отдельно.
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

## Локальный численный референс

В tests/local_reference/atomic_lambda сохранены настройки и итоговые таблицы
прежнего скана (36 пар lambda/n0, 64 найденных корня). Для проверки рефакторинга:

```powershell
python -B -m isothermal_cx_model.scan_lambda --lambdas 7 8 9 10 --densities 1e12 5e12 1e13 5e13 1e14 5e14 1e15 5e15 1e16 --no-show --output atomic_lambda_test_1
python -B tests/compare_local_reference.py atomic_lambda_test_1 --json-output reference_comparison.json
```

Сравнение не запускается при обычном unittest discovery: полный скан затратен.
Discovery проверяет структуру референса и саму процедуру сравнения.
Старые пространственные профили не сохранились, поэтому этот референс
проверяет корни, коэффициенты, невязки и балансы по итоговым таблицам, не профили.
Подробности: tests/local_reference/atomic_lambda/README.md.

## Источники и добавление процессов

atomic_rhs в transport.py восстанавливает локальные величины и геометрический
член. atomic_sources в sources.py возвращает SourceTerms: именованные вклады
в поток частиц, импульса и потери энергии на единицу g=z/L. Это уже
безразмерные интегрированные по трубке вклады, не размерные объёмные скорости.
sources.totals возвращает три суммы; sources.by_process() — разбивку процессов.
sources.rhs(geometric_momentum) сохраняет прежний порядок вычисления RHS.

balance_checks намеренно не использует этот сборщик: проверка балансов
остаётся независимым расчётом через плотность и скорость. При добавлении
процесса необходимо отдельно дополнить проверку и пересмотреть звуковой старт,
критерии допустимости, передачу коэффициентов и формат отчётов, если это нужно.
Текущий патч не вводит молекулярную физику, новые неизвестные или новые ГУ.
