# Описание: sensitivity_analysis.py — Анализ чувствительности

## Обзор

Модуль анализа чувствительности для 20-переменной SDE системы регенерации тканей.
Предоставляет три вычислительных метода и визуализацию за единым интерфейсом `SensitivityAnalyzer`:

1. **Sobol indices** — глобальная variance-based чувствительность (SALib)
2. **Morris screening** — скрининг 40+ параметров методом элементарных эффектов (SALib)
3. **Local sensitivity** — локальные частные производные конечными разностями (NumPy)
4. **Tornado diagrams** — визуализация ранжированной чувствительности (Matplotlib)

Архитектура:

```
ParameterSet (105+ параметров)
       │
       ▼
SensitivityAnalyzer ──► ExtendedSDEModel (forward model)
  │   │   │
  │   │   └── run_sobol()  → SALib (Saltelli + Sobol analyze)
  │   │
  │   └────── run_morris() → SALib (Morris trajectories + analyze)
  │
  └────────── run_local()  → NumPy (central finite differences)
                    │
                    ▼
    SobolResult / MorrisResult / LocalSensitivityResult
                    │
                    ▼
         TornadoPlotter ──► matplotlib → Figure / PNG
```

**Зависимости модуля:**
- `src.core.parameters.ParameterSet` — 105+ параметров модели
- `src.core.extended_sde.ExtendedSDEModel, ExtendedSDEState, ExtendedSDETrajectory` — SDE симуляция
- Внешние (lazy import): `SALib` (sample/analyze), `matplotlib`

---

## SensitivityMethod

**Назначение:** Enum выбора метода анализа чувствительности.

| Значение | Строковое | Библиотека | Вычислительная сложность |
|----------|-----------|------------|--------------------------|
| `SOBOL` | `"sobol"` | SALib | N*(2D+2) запусков модели |
| `MORRIS` | `"morris"` | SALib | n_trajectories*(D+1) запусков |
| `LOCAL` | `"local"` | NumPy | 2D+1 запусков |

Где D — число параметров, N — число базовых сэмплов.

**Тестовые сценарии:**

| Сценарий | Вход | Ожидание |
|----------|------|----------|
| Создание из строки | `SensitivityMethod("sobol")` | `SensitivityMethod.SOBOL` |
| Все значения | — | 3 члена: SOBOL, MORRIS, LOCAL |
| Невалидная строка | `SensitivityMethod("invalid")` | `ValueError` |

---

## ParameterBounds

**Назначение:** Границы варьирования одного параметра модели при анализе чувствительности.
Аналог `PriorSpec` из `parameter_estimation.py`, но упрощённый.

**Поля:**

| Поле | Тип | По умолчанию | Описание |
|------|-----|-------------|----------|
| `name` | `str` | (обязательно) | Имя параметра (поле `ParameterSet`) |
| `lower` | `float` | (обязательно) | Нижняя граница диапазона |
| `upper` | `float` | (обязательно) | Верхняя граница диапазона |
| `nominal` | `float \| None` | `None` | Номинальное (литературное) значение. `None` → берётся из `ParameterSet` |

**Инварианты:**
- `lower < upper`
- `lower >= 0` (все параметры модели строго положительные)
- Если `nominal` задано: `lower <= nominal <= upper`
- `name` должно совпадать с полем `ParameterSet`

**Тестовые сценарии:**

| Сценарий | Вход | Ожидание |
|----------|------|----------|
| Создание корректных bounds | `ParameterBounds("r_F", 0.01, 0.06, 0.03)` | Все поля заполнены |
| Без номинала | `ParameterBounds("r_F", 0.01, 0.06)` | `nominal=None` |
| lower >= upper | `ParameterBounds("r_F", 0.06, 0.01)` | При валидации → `ValueError` |
| nominal вне диапазона | `ParameterBounds("r_F", 0.01, 0.06, 0.1)` | При валидации → `ValueError` |
| Несуществующее имя | `ParameterBounds("nonexistent", 0.0, 1.0)` | При валидации → `ValueError` |

**Граничные случаи:**
- `lower == 0.0` — допустимо (параметр может начинаться с 0)
- `upper` очень большое — допустимо, но может давать нестабильные SDE

---

## SensitivityConfig

**Назначение:** Единая конфигурация для всех методов анализа чувствительности.
Аналог `EstimationConfig` из `parameter_estimation.py`.

**Поля:**

| Поле | Тип | По умолчанию | Описание |
|------|-----|-------------|----------|
| `method` | `SensitivityMethod` | `SOBOL` | Метод анализа |
| `parameter_bounds` | `list[ParameterBounds]` | `[]` | Границы параметров (пустой → автогенерация) |
| `output_variables` | `list[str]` | `["F"]` | Наблюдаемые переменные SDE |
| `output_time_index` | `int` | `-1` | Индекс временного шага для метрики |
| `output_aggregation` | `str` | `"final"` | Агрегация: `"final"`, `"mean"`, `"max"`, `"auc"` |
| `t_span` | `tuple[float, float]` | `(0.0, 720.0)` | Временной диапазон симуляции (часы) |
| `dt` | `float` | `0.01` | Шаг SDE солвера |
| `rng_seed` | `int \| None` | `None` | Seed для воспроизводимости |

### SensitivityConfig.validate

**Назначение:** Валидация всех полей конфигурации.

**Алгоритм:**
1. `parameter_bounds` не пуст → иначе `ValueError`
2. `dt > 0` → иначе `ValueError`
3. `t_span[1] > t_span[0]` → иначе `ValueError`
4. `output_variables` не пуст → иначе `ValueError`
5. `output_aggregation` ∈ {"final", "mean", "max", "auc"} → иначе `ValueError`
6. Для каждого bound: `lower < upper`, `nominal ∈ [lower, upper]`
7. Все `bound.name` существуют в `ParameterSet`

**Возвращает:** `True` если валидна, иначе `ValueError`.

**Тестовые сценарии:**

| Сценарий | Вход | Ожидание |
|----------|------|----------|
| Валидная конфигурация | Корректные bounds, dt=0.01 | `True` |
| Пустые bounds | `parameter_bounds=[]` | `ValueError` |
| dt <= 0 | `dt=-0.01` | `ValueError` |
| Инвертированный t_span | `t_span=(720, 0)` | `ValueError` |
| Невалидная агрегация | `output_aggregation="median"` | `ValueError` |
| Пустые output_variables | `output_variables=[]` | `ValueError` |
| lower >= upper в bounds | bounds с `lower=1, upper=0` | `ValueError` |
| nominal вне [lower, upper] | bounds с `nominal=5, lower=0, upper=1` | `ValueError` |
| Несуществующий параметр | bounds с `name="fake"` | `ValueError` |

---

## SobolResult

**Назначение:** Контейнер результатов глобального анализа Sobol.
Содержит first-order (S1), total-effect (ST), second-order (S2) индексы и CI.

**Поля:**

| Поле | Тип | По умолчанию | Описание |
|------|-----|-------------|----------|
| `parameter_names` | `list[str]` | `[]` | Имена анализируемых параметров |
| `S1` | `np.ndarray` | `array([])` | First-order indices, shape `(n_params,)`. Доля дисперсии, объяснённая параметром напрямую |
| `ST` | `np.ndarray` | `array([])` | Total-effect indices, shape `(n_params,)`. Включает взаимодействия |
| `S2` | `np.ndarray \| None` | `None` | Second-order indices, shape `(n_params, n_params)` |
| `S1_conf` | `np.ndarray` | `array([])` | 95% CI ширина для S1 |
| `ST_conf` | `np.ndarray` | `array([])` | 95% CI ширина для ST |
| `output_variable` | `str` | `""` | Имя выходной переменной |
| `n_samples` | `int` | `0` | Число базовых сэмплов N |
| `n_model_runs` | `int` | `0` | Фактическое число запусков модели |
| `elapsed_seconds` | `float` | `0.0` | Время выполнения (секунды) |

**Инварианты:**
- `0 <= S1[i] <= 1` для всех i (теоретически; численно может немного выходить)
- `S1[i] <= ST[i]` для всех i
- `sum(S1) <= 1`
- `len(S1) == len(ST) == len(parameter_names)`

### SobolResult.get_ranking

**Назначение:** Ранжирование параметров по убыванию total-effect индекса ST.

**Входные параметры:** нет

**Выходные данные:** `list[tuple[str, float, float]]` — список `(name, S1, ST)`, отсортированный по ST убыванию.

**Алгоритм:**
1. `indices = np.argsort(-self.ST)`
2. Вернуть `[(parameter_names[i], S1[i], ST[i]) for i in indices]`

**Тестовые сценарии:**

| Сценарий | Вход | Ожидание |
|----------|------|----------|
| 3 параметра | S1=[0.1, 0.3, 0.2], ST=[0.15, 0.5, 0.35] | Первый элемент: ("param2", 0.3, 0.5) |
| Пустой результат | parameter_names=[] | Пустой список |
| Одинаковые ST | ST=[0.3, 0.3, 0.3] | Все 3 элемента (порядок стабильный) |

---

## MorrisResult

**Назначение:** Контейнер результатов скрининга методом Morris.
Содержит mu (среднее), mu_star (абсолютное среднее — главная метрика), sigma (СКО).

**Поля:**

| Поле | Тип | По умолчанию | Описание |
|------|-----|-------------|----------|
| `parameter_names` | `list[str]` | `[]` | Имена анализируемых параметров |
| `mu` | `np.ndarray` | `array([])` | Среднее элементарных эффектов, shape `(n_params,)` |
| `mu_star` | `np.ndarray` | `array([])` | Среднее абсолютных эффектов (главная метрика скрининга) |
| `sigma` | `np.ndarray` | `array([])` | СКО эффектов (нелинейность / взаимодействия) |
| `mu_star_conf` | `np.ndarray` | `array([])` | 95% CI для mu_star |
| `output_variable` | `str` | `""` | Имя выходной переменной |
| `n_trajectories` | `int` | `0` | Число Morris траекторий |
| `n_levels` | `int` | `4` | Число уровней сетки |
| `n_model_runs` | `int` | `0` | Фактическое число запусков модели |
| `elapsed_seconds` | `float` | `0.0` | Время выполнения (секунды) |

**Инварианты:**
- `mu_star[i] >= 0` для всех i
- `sigma[i] >= 0` для всех i
- `mu_star[i] >= |mu[i]|` (среднее абсолютных ≥ абсолютное среднего)
- `len(mu) == len(mu_star) == len(sigma) == len(parameter_names)`

**Интерпретация (σ vs μ*):**
- Высокое μ*, низкое σ → параметр влиятелен, эффект линейный
- Высокое μ*, высокое σ → параметр влиятелен, эффект нелинейный или с взаимодействиями
- Низкое μ* → параметр невлиятелен (можно фиксировать)

### MorrisResult.get_influential

**Назначение:** Отбор влиятельных параметров по порогу mu_star.
Это ключевой метод для скрининга из 40+ параметров — отбирает 10–15 ключевых.

**Входные параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `threshold_ratio` | `float` | `0.1` | Доля от max(mu_star). Параметры с mu_star >= threshold возвращаются |

**Выходные данные:** `list[str]` — имена влиятельных параметров.

**Алгоритм:**
1. `threshold = threshold_ratio * max(mu_star)`
2. Вернуть `[name for name, ms in zip(names, mu_star) if ms >= threshold]`

**Тестовые сценарии:**

| Сценарий | Вход | Ожидание |
|----------|------|----------|
| 5 параметров, порог 0.1 | mu_star=[100, 80, 5, 3, 1], threshold=0.1 | ["p1", "p2"] (≥10) |
| Все равны | mu_star=[1, 1, 1], threshold=0.5 | Все 3 параметра |
| Пустой результат | mu_star=[] | Пустой список |
| threshold_ratio=0 | threshold_ratio=0.0 | `ValueError` |
| threshold_ratio=1.0 | threshold_ratio=1.0 | Только максимальный параметр |

**Граничные случаи:**
- `threshold_ratio <= 0` → `ValueError`
- `threshold_ratio > 1` → `ValueError`
- Все `mu_star` равны нулю → threshold=0, условие `ms >= 0` истинно для всех → **все параметры возвращаются** (не пустой список)

---

## LocalSensitivityResult

**Назначение:** Контейнер результатов локального анализа чувствительности.
Содержит частные производные и безразмерные индексы эластичности.

**Поля:**

| Поле | Тип | По умолчанию | Описание |
|------|-----|-------------|----------|
| `parameter_names` | `list[str]` | `[]` | Имена анализируемых параметров |
| `partial_derivatives` | `np.ndarray` | `array([])` | dY/dp_i, shape `(n_params,)` |
| `elasticity_indices` | `np.ndarray` | `array([])` | (p_i/Y)*(dY/dp_i), безразмерные, shape `(n_params,)` |
| `nominal_output` | `float` | `0.0` | Y при номинальных параметрах |
| `nominal_params` | `dict[str, float]` | `{}` | Номинальные значения параметров |
| `delta` | `float` | `0.01` | Относительное возмущение |
| `output_variable` | `str` | `""` | Имя выходной переменной |
| `elapsed_seconds` | `float` | `0.0` | Время выполнения |

**Инварианты:**
- `len(partial_derivatives) == len(elasticity_indices) == len(parameter_names)`
- `delta > 0`
- Если `nominal_output == 0`, elasticity не определена (inf)

### LocalSensitivityResult.get_ranking

**Назначение:** Ранжирование параметров по убыванию |elasticity|.

**Выходные данные:** `list[tuple[str, float, float]]` — список `(name, partial_derivative, elasticity)`, отсортированный по |elasticity| убыванию.

**Алгоритм:**
1. `indices = np.argsort(-np.abs(self.elasticity_indices))`
2. Вернуть `[(names[i], derivatives[i], elasticity[i]) for i in indices]`

**Тестовые сценарии:**

| Сценарий | Вход | Ожидание |
|----------|------|----------|
| 3 параметра | elasticity=[0.5, -2.0, 1.0] | Первый: ("p2", deriv2, -2.0) |
| Пустой результат | parameter_names=[] | Пустой список |
| Нулевые эластичности | elasticity=[0, 0, 0] | Все 3, порядок стабильный |

---

## TornadoData

**Назначение:** Унифицированный контейнер данных для tornado diagram.
Создаётся из результатов любого метода через `TornadoPlotter.from_*` classmethods.

**Поля:**

| Поле | Тип | По умолчанию | Описание |
|------|-----|-------------|----------|
| `parameter_names` | `list[str]` | `[]` | Имена параметров (отсортированы по важности) |
| `values` | `np.ndarray` | `array([])` | Значения метрики чувствительности, shape `(n_params,)` |
| `lower_values` | `np.ndarray \| None` | `None` | Нижние CI (для error bars) |
| `upper_values` | `np.ndarray \| None` | `None` | Верхние CI |
| `metric_name` | `str` | `""` | Название метрики: `"S1"`, `"ST"`, `"mu_star"`, `"elasticity"` |
| `title` | `str` | `""` | Заголовок диаграммы |
| `source_method` | `SensitivityMethod \| None` | `None` | Какой метод породил данные |
| `top_n` | `int \| None` | `None` | Показано top N параметров |

**Инварианты:**
- `len(parameter_names) == len(values)`
- Если `lower_values` не None → `len(lower_values) == len(values)`
- Если `upper_values` не None → `len(upper_values) == len(values)`

**Тестовые сценарии:**

| Сценарий | Вход | Ожидание |
|----------|------|----------|
| Создание из Sobol | `TornadoPlotter.from_sobol(result)` | metric_name="ST", source=SOBOL |
| Создание из Morris | `TornadoPlotter.from_morris(result)` | metric_name="mu_star", source=MORRIS |
| Создание из Local | `TornadoPlotter.from_local(result)` | metric_name="elasticity", source=LOCAL |
| top_n=5 при 20 параметрах | top_n=5 | len(parameter_names)==5 |
| top_n=None | top_n=None | Все параметры |

---

## SensitivityAnalyzer

**Назначение:** Главный оркестратор анализа чувствительности.
Управляет тремя вычислительными методами за единым интерфейсом.
Аналог `BaseEstimator` из `parameter_estimation.py`, но без наследования.

### SensitivityAnalyzer.__init__

**Входные параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `model` | `ExtendedSDEModel` | Экземпляр 20-переменной SDE модели |
| `params` | `ParameterSet` | Номинальный набор параметров (литературные defaults) |
| `config` | `SensitivityConfig` | Конфигурация анализа чувствительности |

**Алгоритм:**
1. Сохранить `model`, `params`, `config`
2. Если `config.parameter_bounds` пуст → вызвать `_auto_bounds()` для автогенерации (±50% от номинала)
3. Вызвать `config.validate()` → `ValueError` при невалидной конфигурации

**Граничные случаи:**
- Пустой `parameter_bounds` → автогенерация из `ParameterSet`
- Невалидная конфигурация → `ValueError` при создании

---

### run_sobol

**Назначение:** Глобальный анализ чувствительности методом Sobol.

**Входные параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `output_variables` | `list[str] \| None` | `None` | Переменные для анализа (None → из config) |
| `n_samples` | `int` | `1024` | Число базовых сэмплов N |

**Выходные данные:** `SobolResult`

**Алгоритм:**
1. Lazy import `SALib.sample.saltelli`, `SALib.analyze.sobol`
2. `problem = self._build_salib_problem()` — SALib problem definition
3. `param_values = saltelli.sample(problem, n_samples)` — генерирует N*(2D+2) строк
4. `Y = self._evaluate_model(param_values)` — запустить модель для всех сэмплов
5. `si = sobol.analyze(problem, Y)` — вычислить S1, ST, S2
6. Обработать NaN: заменить на 0.0
7. Собрать `SobolResult` с CI из `si.to_df()`

**Вычислительная сложность:** N*(2D+2) запусков. При D=40 и N=1024 → ~83,968 запусков.

**Граничные случаи:**
- Все выходы одинаковы (Y.std() == 0) → S1=ST=0, warning
- NaN в выходах → заменить на 0.0
- SALib не установлен → `ImportError` с понятным сообщением
- `n_samples < 16` → `ValueError` (недостаточная выборка)

**Тестовые сценарии:**

| Сценарий | Вход | Ожидание |
|----------|------|----------|
| Stub вызов | Любые аргументы | `NotImplementedError` (до реализации) |
| SALib не установлен | — | `ImportError` |
| n_samples < 16 | n_samples=8 | `ValueError` |
| Корректный запуск | n_samples=64, 3 параметра | SobolResult с S1.shape==(3,) |

---

### run_morris

**Назначение:** Скрининг параметров методом Morris (Elementary Effects).
Эффективен для 40+ параметров: n_trajectories*(D+1) запусков.

**Входные параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `output_variables` | `list[str] \| None` | `None` | Переменные для анализа |
| `n_trajectories` | `int` | `10` | Число Morris траекторий (рек. 10–20) |
| `n_levels` | `int` | `4` | Число уровней сетки (рек. 4–8) |

**Выходные данные:** `MorrisResult`

**Алгоритм:**
1. Lazy import `SALib.sample.morris`, `SALib.analyze.morris`
2. `problem = self._build_salib_problem()`
3. `param_values = morris_sample.sample(problem, N=n_trajectories, num_levels=n_levels)` — генерирует n_trajectories*(D+1) строк
4. `Y = self._evaluate_model(param_values)`
5. `si = morris_analyze.analyze(problem, param_values, Y, num_levels=n_levels)`
6. Собрать `MorrisResult` с mu, mu_star, sigma, CI

**Вычислительная сложность:** n_trajectories*(D+1). При D=40, n_trajectories=10 → 410 запусков.

**Граничные случаи:**
- `n_trajectories < 2` → `ValueError`
- `n_levels < 2` → `ValueError`
- SALib не установлен → `ImportError`

**Тестовые сценарии:**

| Сценарий | Вход | Ожидание |
|----------|------|----------|
| Stub вызов | Любые аргументы | `NotImplementedError` |
| n_trajectories < 2 | n_trajectories=1 | `ValueError` |
| n_levels < 2 | n_levels=1 | `ValueError` |
| Корректный запуск | n_trajectories=10, 5 параметров | MorrisResult, n_model_runs=60 |

---

### run_local

**Назначение:** Локальный анализ чувствительности конечными разностями.

**Входные параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `output_variables` | `list[str] \| None` | `None` | Переменные для анализа |
| `delta` | `float` | `0.01` | Относительное возмущение (1%) |

**Выходные данные:** `LocalSensitivityResult`

**Алгоритм:**
1. Получить номинальные значения из `self.params` и `config.parameter_bounds`
2. Вычислить `Y_nom = self._evaluate_model_single(nominal_params)`
3. Для каждого параметра p_i с номинальным значением p0_i:
   - `dp = p0_i * delta`
   - `params_plus = {…, p_i: p0_i + dp, …}`
   - `params_minus = {…, p_i: p0_i - dp, …}`
   - `Y_plus = self._evaluate_model_single(params_plus)`
   - `Y_minus = self._evaluate_model_single(params_minus)`
   - `dY/dp_i = (Y_plus - Y_minus) / (2 * dp)` — центральные конечные разности
   - `elasticity_i = (p0_i / Y_nom) * (dY/dp_i)` — безразмерный
4. Собрать `LocalSensitivityResult`

**Вычислительная сложность:** 2D+1 запусков. При D=40 → 81 запуск. Самый быстрый метод.

**Граничные случаи:**
- `delta <= 0` → `ValueError`
- `delta >= 1.0` → `ValueError` (слишком большое возмущение; docstring определяет диапазон 0 < delta < 1)
- `Y_nom == 0` → elasticity = inf, warning
- Параметр `p0_i == 0` → dp=0, single-sided difference или skip
- `p0_i - dp < 0` → single-sided difference (forward only)

**Тестовые сценарии:**

| Сценарий | Вход | Ожидание |
|----------|------|----------|
| Stub вызов | Любые аргументы | `NotImplementedError` |
| delta <= 0 | delta=-0.01 | `ValueError` |
| delta >= 1 | delta=1.5 | Warning (допустимо, но предупреждение) |
| Корректный запуск | delta=0.01, 5 параметров | LocalSensitivityResult, len=5 |

---

### _evaluate_model

**Назначение:** Запуск модели для массива параметрических сэмплов (batch evaluation).

**Входные параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `param_values` | `np.ndarray` | (обязательно) | Матрица сэмплов, shape `(n_runs, n_params)` |
| `output_variable` | `str \| None` | `None` | Имя выходной переменной (None → первая из config) |
| `progress_callback` | `Callable[[int, int], None] \| None` | `None` | Функция `(current, total)` для прогресса |

**Выходные данные:** `np.ndarray` shape `(n_runs,)` — скалярные агрегированные выходы.

**Алгоритм:**
1. `output_var = output_variable or config.output_variables[0]`
2. `param_names = [b.name for b in config.parameter_bounds]`
3. Для каждой строки `i` в `param_values`:
   - Создать `param_dict = {name: value for name, value in zip(param_names, param_values[i])}`
   - `Y[i] = self._evaluate_model_single(param_dict, output_var)`
   - Вызвать `progress_callback(i+1, n_runs)` если задан
4. Вернуть `Y`

**Связи:** Повторяет логику `_execute_sensitivity` из `analysis_service.py` (строки 98–162), но обобщённую.

---

### _evaluate_model_single

**Назначение:** Запуск модели для одного набора параметров.

**Входные параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `param_dict` | `dict[str, float]` | (обязательно) | {имя_параметра: значение} |
| `output_variable` | `str \| None` | `None` | Имя выходной переменной |

**Выходные данные:** `float` — скалярный агрегированный выход.

**Алгоритм:**
1. Копировать `self.params`, подставить значения из `param_dict`
2. Создать `ExtendedSDEModel(params=modified_params)`
3. `traj = model.simulate(initial_state, t_span=config.t_span)`
4. Извлечь переменную `output_variable` из траектории
5. Агрегировать по `config.output_aggregation`:
   - `"final"` → последнее значение
   - `"mean"` → среднее по времени
   - `"max"` → максимум по времени
   - `"auc"` → интеграл (trapz) по времени
6. Вернуть скалярное значение

---

### _build_salib_problem

**Назначение:** Построение SALib problem definition из `config.parameter_bounds`.

**Выходные данные:** `dict` с ключами `"num_vars"`, `"names"`, `"bounds"`.

**Алгоритм:**
```python
{
    "num_vars": len(bounds),
    "names": [b.name for b in bounds],
    "bounds": [[b.lower, b.upper] for b in bounds],
}
```

**Связи:** Повторяет структуру из `analysis_service.py` строки 117–121.

---

### _auto_bounds

**Назначение:** Автоматическая генерация bounds из `ParameterSet` (±50% от номинала).

**Выходные данные:** `list[ParameterBounds]`

**Алгоритм:**
1. `param_dict = self.params.to_dict()`
2. Для каждого `(name, value)` где `value > 0` и `isinstance(value, (int, float))`:
   - `lower = value * 0.5`
   - `upper = value * 2.0`
   - `nominal = value`
3. Вернуть список `ParameterBounds`

**Связи:** Использует `ParameterSet.to_dict()` и расширяет логику `_get_parameter_bounds()` из `analysis_service.py`.

---

## TornadoPlotter

**Назначение:** Визуализация ранжированной чувствительности (tornado diagram).
Предоставляет classmethods для конвертации результатов и метод `plot()`.

### TornadoPlotter.from_sobol

**Входные параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `result` | `SobolResult` | (обязательно) | Результат Sobol анализа |
| `metric` | `str` | `"ST"` | Метрика: `"S1"` или `"ST"` |
| `top_n` | `int \| None` | `15` | Top N параметров (None → все) |

**Выходные данные:** `TornadoData`

**Алгоритм:**
1. Выбрать массив значений по `metric` (S1 или ST) и соответствующие CI
2. `indices = np.argsort(-values)`
3. Если `top_n` задан → `indices = indices[:top_n]`
4. Собрать `TornadoData` с отсортированными значениями

**Граничные случаи:**
- `metric` не "S1" и не "ST" → `ValueError`
- `top_n > len(parameters)` → показать все
- Пустой `SobolResult` → пустой `TornadoData`

### TornadoPlotter.from_morris

**Входные параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `result` | `MorrisResult` | (обязательно) | Результат Morris скрининга |
| `top_n` | `int \| None` | `15` | Top N параметров |

**Выходные данные:** `TornadoData` отсортированный по mu_star убыванию.

### TornadoPlotter.from_local

**Входные параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `result` | `LocalSensitivityResult` | (обязательно) | Результат локальной чувствительности |
| `top_n` | `int \| None` | `15` | Top N параметров |

**Выходные данные:** `TornadoData` отсортированный по |elasticity| убыванию.

### TornadoPlotter.plot

**Назначение:** Построение tornado diagram (горизонтальная столбчатая диаграмма).

**Входные параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `data` | `TornadoData` | (обязательно) | Данные для визуализации |
| `output_path` | `str \| None` | `None` | Путь для PNG (None → не сохранять) |

**Выходные данные:** `matplotlib.figure.Figure`

**Алгоритм:**
1. Lazy import `matplotlib.pyplot as plt`
2. `fig, ax = plt.subplots(figsize=(10, max(6, len(data.parameter_names)*0.4)))`
3. `ax.barh(parameter_names, values)` — горизонтальные столбцы
4. Если `lower_values`/`upper_values` → добавить error bars через `xerr`
5. `ax.set_xlabel(metric_name)`, `ax.set_title(title)`
6. `ax.invert_yaxis()` — самые важные сверху
7. Если `output_path` → `fig.savefig(output_path, dpi=150, bbox_inches="tight")`
8. Вернуть `fig`

**Граничные случаи:**
- Пустой `TornadoData` → пустой Figure с warning
- `output_path` с несуществующим каталогом → `FileNotFoundError`
- matplotlib не установлен → `ImportError`

**Тестовые сценарии:**

| Сценарий | Вход | Ожидание |
|----------|------|----------|
| Stub вызов | Любые аргументы | `NotImplementedError` |
| matplotlib не установлен | — | `ImportError` |
| Пустые данные | TornadoData() | Пустой Figure |
| 15 параметров | TornadoData с 15 элементами | Figure с 15 столбцами |
| Сохранение в файл | output_path="test.png" | Файл создан |

---

## run_sensitivity_analysis

**Назначение:** Convenience-функция для быстрого запуска анализа чувствительности.
Создаёт все объекты и запускает выбранный метод. Аналог `estimate_parameters()`.

**Входные параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `method` | `str \| SensitivityMethod` | `"sobol"` | Метод анализа |
| `params` | `ParameterSet \| None` | `None` | Параметры (None → defaults) |
| `parameter_names` | `list[str] \| None` | `None` | Имена параметров (None → все) |
| `output_variables` | `list[str] \| None` | `None` | Выходные переменные (None → ["F"]) |
| `n_samples` | `int` | `1024` | Число сэмплов |
| `config` | `SensitivityConfig \| None` | `None` | Готовая конфигурация |

**Выходные данные:** `SobolResult | MorrisResult | LocalSensitivityResult`

**Алгоритм:**
1. `params = params or ParameterSet()`
2. `model = ExtendedSDEModel(params=params)`
3. Если `config` не задан → создать `SensitivityConfig` из аргументов
4. `analyzer = SensitivityAnalyzer(model, params, config)`
5. Вызвать `run_sobol`, `run_morris` или `run_local` в зависимости от `method`

**Граничные случаи:**
- Неизвестный `method` → `ValueError`
- `method` как строка → конвертировать в `SensitivityMethod`

---

## Интеграция с analysis_service.py

Текущий `AnalysisService._execute_sensitivity()` (строки 98–208) содержит встроенную Sobol логику.
После реализации модуля, `analysis_service.py` должен делегировать в `SensitivityAnalyzer`:

```python
from src.core.sensitivity_analysis import SensitivityAnalyzer, SensitivityConfig, ParameterBounds

analyzer = SensitivityAnalyzer(model, params, config)
if request.method == "sobol":
    result = analyzer.run_sobol(n_samples=request.n_samples)
elif request.method == "morris":
    result = analyzer.run_morris(n_trajectories=request.n_trajectories)
```

`_get_parameter_bounds()` из `analysis_service.py` (строки 293–311) используется как фабрика для `ParameterBounds` dataclass-ов.
