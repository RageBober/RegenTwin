# Описание: parameter_estimation.py — Параметрическая идентификация

## Обзор

Модуль параметрической идентификации для 20-переменной SDE системы регенерации тканей.
Предоставляет три метода оценки параметров за единым интерфейсом `fit() → EstimationResult`:

1. **BayesianEstimator** — полный Байесовский вывод через PyMC 5 (NUTS sampler)
2. **MCMCEstimator** — ансамблевый MCMC через emcee (gradient-free)
3. **MLEstimator** — оценка максимального правдоподобия через scipy.optimize

Архитектура:

```
TimeSeriesData (наблюдения)
       │
       ▼
BaseEstimator ──► ForwardModelWrapper ──► ExtendedSDEModel
  │   │   │              │
  │   │   │              └──► ParameterSet
  │   │   │
  │   │   └── BayesianEstimator → PyMC 5
  │   └────── MCMCEstimator ────→ emcee
  └────────── MLEstimator ──────→ scipy.optimize
                    │
                    ▼
           ConvergenceAnalyzer → ArviZ
                    │
                    ▼
            EstimationResult
```

**Зависимости модуля:**
- `src.core.parameters.ParameterSet` — 105+ параметров модели
- `src.core.extended_sde.ExtendedSDEModel, ExtendedSDEState, ExtendedSDETrajectory` — SDE симуляция
- `src.data.dataset_loader.TimeSeriesData` — наблюдательные данные
- Внешние (lazy import): `pymc`, `emcee`, `arviz`, `scipy.optimize`

---

## Классы данных (Dataclasses)

### PriorSpec

**Назначение:** Спецификация априорного распределения для одного параметра модели.
Каждый из 105+ параметров `ParameterSet` может иметь свой `PriorSpec`.
Параметры с `fixed=True` фиксируются на значении `mean` и не оцениваются.

**Поля:**

| Поле | Тип | По умолчанию | Описание |
|------|-----|-------------|----------|
| `name` | `str` | (обязательно) | Имя параметра (должно совпадать с полем `ParameterSet`) |
| `distribution` | `str` | `"lognormal"` | Тип распределения: `"normal"`, `"lognormal"`, `"uniform"`, `"halfnormal"`, `"gamma"` |
| `mean` | `float` | `0.0` | Центр распределения (μ для normal/lognormal, mid для uniform) |
| `std` | `float` | `1.0` | Разброс (σ для normal/lognormal) |
| `lower` | `float` | `0.0` | Нижняя граница (для uniform, или усечения) |
| `upper` | `float` | `inf` | Верхняя граница (для uniform, или усечения) |
| `fixed` | `bool` | `False` | Если `True` — параметр фиксируется на `mean`, не оценивается |
| `source` | `str` | `""` | Литературная ссылка (напр. `"Vodovotz 2006"`) |

**Mapping распределений:**

| `distribution` | PyMC | log-prior (emcee) | scipy bounds |
|---------------|------|-------------------|--------------|
| `"normal"` | `pm.Normal(name, mu=mean, sigma=std)` | `−(θ−μ)²/(2σ²)` | `(mean−4*std, mean+4*std)` |
| `"lognormal"` | `pm.LogNormal(name, mu=log(mean), sigma=std)` | `−(log θ−μ)²/(2σ²) − log θ` | `(0, mean*10)` |
| `"uniform"` | `pm.Uniform(name, lower=lower, upper=upper)` | `0 если lower≤θ≤upper, иначе −∞` | `(lower, upper)` |
| `"halfnormal"` | `pm.HalfNormal(name, sigma=std)` | `−θ²/(2σ²)` для `θ≥0` | `(0, 5*std)` |
| `"gamma"` | `pm.Gamma(name, alpha=mean²/std², beta=mean/std²)` | стандартная log-gamma | `(0, mean+5*std)` |

**Граничные случаи и ошибки:**
- `name` не совпадает с полем `ParameterSet` → `ValueError` при валидации
- `std <= 0` → `ValueError`
- `lower >= upper` для uniform → `ValueError`
- `distribution` неизвестен → `ValueError`

**Тестовые сценарии:**

| Сценарий | Вход | Ожидание |
|----------|------|----------|
| Создание с defaults | `PriorSpec(name="r_F")` | `distribution="lognormal"`, `fixed=False` |
| Фиксированный параметр | `PriorSpec(name="dt", fixed=True, mean=0.01)` | `fixed=True` |
| Невалидное имя | `name="nonexistent"` | При валидации → `ValueError` |
| Отрицательный std | `std=-1.0` | При валидации → `ValueError` |

---

### EstimationConfig

**Назначение:** Единая конфигурация для всех методов параметрической идентификации.
Содержит настройки forward model, MCMC, MLE, и критерии сходимости.

**Поля:**

| Поле | Тип | По умолчанию | Описание |
|------|-----|-------------|----------|
| `priors` | `list[PriorSpec]` | `[]` | Спецификации приоров для оцениваемых параметров |
| `observed_variables` | `list[str]` | `[]` | Наблюдаемые переменные SDE (напр. `["F", "C_TNF"]`) |
| `t_span` | `tuple[float, float]` | `(0.0, 720.0)` | Временной диапазон симуляции (часы) |
| `dt` | `float` | `0.01` | Шаг SDE солвера |
| `n_sde_realizations` | `int` | `1` | Число реализаций SDE для усреднения (сглаживание likelihood) |
| `solver` | `str` | `"euler_maruyama"` | Имя SDE солвера |
| `noise_model` | `str` | `"gaussian"` | Модель шума likelihood: `"gaussian"` или `"lognormal"` |
| `sigma_obs` | `float \| None` | `None` | Фиксированный шум наблюдений (`None` = оценивать) |
| `n_samples` | `int` | `2000` | Число posterior samples |
| `n_tune` | `int` | `1000` | Burn-in / tuning steps |
| `n_chains` | `int` | `4` | Число независимых MCMC цепей |
| `n_walkers` | `int` | `32` | Число walkers для emcee (игнорируется PyMC) |
| `target_accept` | `float` | `0.8` | Target acceptance rate для PyMC NUTS |
| `mle_method` | `str` | `"L-BFGS-B"` | Метод scipy.optimize |
| `mle_maxiter` | `int` | `1000` | Макс. итерации MLE |
| `rhat_threshold` | `float` | `1.05` | Порог R-hat для сходимости |
| `ess_min` | `int` | `100` | Минимальный ESS |
| `rng_seed` | `int \| None` | `None` | Seed для воспроизводимости |

**Метод validate():**

**Назначение:** Валидация конфигурации.

**Сигнатура:**
```python
def validate(self) -> bool
```

**Возвращает:** `True` если валидна.

**Алгоритм:**
1. Проверить `n_samples > 0`, `n_tune >= 0`, `n_chains >= 1`
2. Проверить `n_walkers >= 2 * len(free_params)` (требование emcee)
3. Проверить `dt > 0`, `t_span[1] > t_span[0]`
4. Проверить `noise_model` ∈ `{"gaussian", "lognormal"}`
5. Проверить `solver` ∈ допустимых солверов
6. Проверить `0 < target_accept < 1`
7. Проверить `rhat_threshold > 1.0`
8. Валидировать каждый `PriorSpec` в `priors`

**Ошибки:** `ValueError` с описанием невалидного поля.

**Граничные случаи:**
- Пустой `priors` → допустимо (будет использоваться `PriorBuilder.from_parameter_set`)
- Пустой `observed_variables` → `ValueError` (нет наблюдений)
- `n_sde_realizations = 0` → `ValueError`

**Тестовые сценарии:**

| Сценарий | Вход | Ожидание |
|----------|------|----------|
| Defaults | `EstimationConfig()` | Все поля заполнены |
| Валидная конфигурация | Все поля корректны | `validate() → True` |
| Негативный n_samples | `n_samples=-1` | `ValueError` |
| target_accept вне [0,1] | `target_accept=1.5` | `ValueError` |

---

### ConvergenceDiagnostics

**Назначение:** Контейнер метрик сходимости MCMC цепей, вычисленных через ArviZ.
Используется `ConvergenceAnalyzer` и встраивается в `EstimationResult`.

**Поля:**

| Поле | Тип | По умолчанию | Описание |
|------|-----|-------------|----------|
| `rhat` | `dict[str, float]` | `{}` | R-hat (Gelman-Rubin) для каждого параметра. `< 1.05` = сошлось |
| `ess_bulk` | `dict[str, float]` | `{}` | Bulk ESS для каждого параметра |
| `ess_tail` | `dict[str, float]` | `{}` | Tail ESS для каждого параметра |
| `converged` | `bool` | `False` | `True` если ВСЕ R-hat < threshold И ВСЕ ESS > min |
| `summary_table` | `Any` | `None` | `pd.DataFrame` из `az.summary()` |
| `warnings` | `list[str]` | `[]` | Предупреждения о сходимости |

**Тестовые сценарии:**

| Сценарий | Вход | Ожидание |
|----------|------|----------|
| Пустые метрики | `ConvergenceDiagnostics()` | `converged=False`, пустые dict |
| Сошлось | Все R-hat < 1.05, ESS > 100 | `converged=True` |
| Не сошлось | Один R-hat = 1.2 | `converged=False`, warning в списке |

---

### EstimationResult

**Назначение:** Унифицированный контейнер результатов параметрической идентификации.
Возвращается всеми тремя estimator-ами через `fit()`.

**Поля:**

| Поле | Тип | По умолчанию | Описание |
|------|-----|-------------|----------|
| `method` | `str` | `""` | `"bayesian_pymc"`, `"mcmc_emcee"`, `"mle_scipy"` |
| `point_estimates` | `dict[str, float]` | `{}` | Точечные оценки (MAP / MLE / posterior mean) |
| `ci_lower` | `dict[str, float]` | `{}` | Нижняя граница 95% CI (2.5% квантиль) |
| `ci_upper` | `dict[str, float]` | `{}` | Верхняя граница 95% CI (97.5% квантиль) |
| `posterior_samples` | `dict[str, np.ndarray] \| None` | `None` | Posterior samples `{name: array(n_samples,)}`. `None` для MLE |
| `inference_data` | `Any` | `None` | `az.InferenceData`. `None` для MLE |
| `diagnostics` | `ConvergenceDiagnostics \| None` | `None` | Метрики сходимости. `None` для MLE |
| `fitted_params` | `ParameterSet \| None` | `None` | `ParameterSet` с подставленными оценками |
| `log_likelihood` | `float \| None` | `None` | Log-likelihood в оптимуме |
| `aic` | `float \| None` | `None` | Akaike Information Criterion: `2k − 2·log_lik` |
| `bic` | `float \| None` | `None` | Bayesian IC: `k·log(n) − 2·log_lik` |
| `n_observations` | `int` | `0` | Число точек наблюдений |
| `n_estimated_params` | `int` | `0` | Число оценённых параметров |
| `elapsed_seconds` | `float` | `0.0` | Время выполнения |
| `config` | `EstimationConfig \| None` | `None` | Использованная конфигурация |

**Тестовые сценарии:**

| Сценарий | Вход | Ожидание |
|----------|------|----------|
| Пустой результат | `EstimationResult()` | Все поля по умолчанию |
| MLE результат | `method="mle_scipy"` | `posterior_samples=None`, `diagnostics=None` |
| Bayesian результат | Полный | Все поля заполнены, `diagnostics.converged` проверяем |
| AIC/BIC | `log_lik=-100, k=5, n=50` | `aic=210`, `bic≈208.05` |

---

## Классы

### ForwardModelWrapper

**Назначение:** Обёртка SDE модели для параметрической идентификации. Принимает вектор
оцениваемых параметров `theta`, встраивает их в `ParameterSet`, запускает SDE симуляцию,
возвращает предсказания в точках наблюдений.

**Зависимости:**
- `ParameterSet` — базовые параметры + подстановка theta
- `ExtendedSDEModel` — SDE солвер
- `ExtendedSDEState` — начальное состояние
- `ExtendedSDETrajectory` — результат симуляции

#### `__init__`

**Сигнатура:**
```python
def __init__(
    self,
    base_params: ParameterSet,
    initial_state: ExtendedSDEState,
    estimated_param_names: list[str],
    observed_variables: list[str],
    observation_times: np.ndarray,
    config: EstimationConfig | None = None,
) -> None
```

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `base_params` | `ParameterSet` | Базовый набор параметров (литературные defaults) |
| `initial_state` | `ExtendedSDEState` | Начальное состояние SDE (20 переменных + t) |
| `estimated_param_names` | `list[str]` | Имена параметров, которые будут оцениваться (порядок = порядок в theta) |
| `observed_variables` | `list[str]` | Имена наблюдаемых переменных SDE (напр. `["F", "C_TNF"]`) |
| `observation_times` | `np.ndarray` | Временные точки наблюдений (часы), shape `(n_obs,)` |
| `config` | `EstimationConfig \| None` | Конфигурация (для `n_sde_realizations`, `solver`, `dt`) |

**Алгоритм:**
1. Сохранить все параметры как атрибуты
2. Проверить что `estimated_param_names` ⊆ полям `ParameterSet`
3. Проверить что `observed_variables` ⊆ полям `ExtendedSDEState` (без `t`)
4. Проверить что `observation_times` отсортирован и ∈ `[0, t_max]`

**Ошибки:**
- `ValueError` если имя параметра не найдено в `ParameterSet`
- `ValueError` если имя переменной не найдено в `ExtendedSDEState`
- `ValueError` если `observation_times` пуст

#### `predict`

**Назначение:** Запуск SDE с параметрами theta, возврат предсказаний в точках наблюдений.

**Сигнатура:**
```python
def predict(self, theta: np.ndarray) -> dict[str, np.ndarray]
```

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `theta` | `np.ndarray` | Значения оцениваемых параметров, shape `(n_params,)` |

**Возвращает:** `dict[str, np.ndarray]` — `{variable_name: predictions}`, каждый shape `(n_obs_times,)`.

**Алгоритм:**
1. `_build_parameter_set(theta)` → `ParameterSet`
2. Если `n_sde_realizations > 1`: запустить `n_sde_realizations` симуляций, усреднить
3. Иначе: одна `_run_simulation(params)` → trajectory
4. `_extract_at_times(trajectory, observation_times)` → предсказания

**Граничные случаи:**
- `theta` содержит отрицательные значения для строго положительных параметров → SDE может diverge, возвращать NaN. Caller (estimator) должен проверять log-prior.
- `len(theta) != len(estimated_param_names)` → `ValueError`
- SDE divergence (NaN в trajectory) → вернуть dict с NaN массивами

**Тестовые сценарии:**

| Сценарий | Вход | Ожидание |
|----------|------|----------|
| Литературные параметры | theta = defaults | Предсказания ≈ baseline trajectory |
| Размерность theta | len(theta) != n_params | `ValueError` |
| Выход формат | Любой валидный theta | dict с ключами из `observed_variables` |
| NaN в theta | `[nan, 0.01, ...]` | dict с NaN значениями |

#### `_build_parameter_set`

**Назначение:** Создание `ParameterSet`: копия `base_params` с подставленными значениями из theta.

**Сигнатура:**
```python
def _build_parameter_set(self, theta: np.ndarray) -> ParameterSet
```

**Алгоритм:**
1. `base_params.to_dict()` → dict
2. Для `i, name` в `enumerate(estimated_param_names)`: `dict[name] = theta[i]`
3. `ParameterSet.from_dict(dict)` → новый `ParameterSet`

**Тестовые сценарии:**

| Сценарий | Вход | Ожидание |
|----------|------|----------|
| Подстановка одного | theta=[0.05], names=["r_F"] | `result.r_F == 0.05`, остальные = defaults |
| Все defaults | theta = default values | Результат == `ParameterSet()` |

#### `_run_simulation`

**Назначение:** Запуск SDE симуляции с данными параметрами.

**Сигнатура:**
```python
def _run_simulation(self, params: ParameterSet) -> ExtendedSDETrajectory
```

**Алгоритм:**
1. Создать `ExtendedSDEModel(params=params)`
2. Вызвать `model.simulate(initial_state, t_max=observation_times[-1])`
3. Вернуть trajectory

**Тестовые сценарии:**

| Сценарий | Вход | Ожидание |
|----------|------|----------|
| Default params | `ParameterSet()` | Trajectory с `len(states) > 0` |
| Невалидные params | Отрицательная rate | SDE может diverge |

#### `_extract_at_times`

**Назначение:** Извлечение значений наблюдаемых переменных из trajectory в точках наблюдений.
Используется линейная интерполяция если временные точки trajectory не совпадают с observation_times.

**Сигнатура:**
```python
def _extract_at_times(
    self,
    trajectory: ExtendedSDETrajectory,
    times: np.ndarray,
) -> dict[str, np.ndarray]
```

**Алгоритм:**
1. Для каждой `var` в `observed_variables`:
   - `trajectory.get_variable(var)` → массив значений
   - `np.interp(times, trajectory.times, values)` → интерполированные значения
2. Вернуть dict

**Тестовые сценарии:**

| Сценарий | Вход | Ожидание |
|----------|------|----------|
| Точное совпадение times | obs_times == traj.times[::100] | Значения без интерполяции |
| Интерполяция | obs_times между traj.times | Линейно-интерполированные значения |

---

### PriorBuilder

**Назначение:** Генератор априорных распределений из списка `PriorSpec`.
Конвертирует в формат нужный каждому backend-у: PyMC distributions, emcee log-prior, scipy bounds.

**Зависимости:**
- `PriorSpec` — входные спецификации
- `ParameterSet` — для `from_parameter_set()` factory

#### `__init__`

**Сигнатура:**
```python
def __init__(self, priors: list[PriorSpec]) -> None
```

**Алгоритм:**
1. Сохранить `priors`
2. Разделить на `free_priors` (fixed=False) и `fixed_priors` (fixed=True)
3. Сохранить `free_param_names = [p.name for p in free_priors]`

#### `from_parameter_set` (classmethod)

**Назначение:** Автоматическое создание приоров из литературных значений `ParameterSet`.
Для каждого параметра в `estimated_names` создаёт `PriorSpec` с:
- `mean` = литературное значение
- `std` = `mean * default_cv` (coefficient of variation)
- `distribution` = `"lognormal"` (все параметры строго положительные)

**Сигнатура:**
```python
@classmethod
def from_parameter_set(
    cls,
    params: ParameterSet,
    estimated_names: list[str],
    default_cv: float = 0.3,
) -> PriorBuilder
```

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `params` | `ParameterSet` | Источник литературных значений |
| `estimated_names` | `list[str]` | Имена параметров для оценки |
| `default_cv` | `float` | Коэффициент вариации (std/mean) |

**Возвращает:** `PriorBuilder` с настроенными приорами.

**Алгоритм:**
1. `params.to_dict()` → dict литературных значений
2. Для каждого `name` в `estimated_names`:
   - `value = dict[name]`
   - Создать `PriorSpec(name=name, distribution="lognormal", mean=value, std=value*default_cv, source="literature")`
3. Вернуть `cls(priors)`

**Граничные случаи:**
- `estimated_names` содержит имя не из `ParameterSet` → `ValueError`
- `default_cv <= 0` → `ValueError`
- Пустой `estimated_names` → допустимо (нет свободных параметров)

**Тестовые сценарии:**

| Сценарий | Вход | Ожидание |
|----------|------|----------|
| Один параметр | `estimated_names=["r_F"]` | `PriorSpec(name="r_F", mean=0.03, std=0.009)` |
| CV=0.5 | `default_cv=0.5` | std = 0.5 * mean |
| Невалидное имя | `["nonexistent"]` | `ValueError` |

#### `get_free_param_names`

**Назначение:** Возвращает имена оцениваемых (не зафиксированных) параметров в порядке theta.

**Сигнатура:**
```python
def get_free_param_names(self) -> list[str]
```

**Возвращает:** `list[str]` — имена параметров с `fixed=False`.

**Тестовые сценарии:**

| Сценарий | Вход | Ожидание |
|----------|------|----------|
| Все свободные | 3 PriorSpec, все fixed=False | Список из 3 имён |
| Смешанные | 2 free + 1 fixed | Список из 2 имён |
| Все fixed | Все fixed=True | Пустой список |

#### `build_pymc_priors`

**Назначение:** Создание PyMC prior distributions внутри `pm.Model` context.

**Сигнатура:**
```python
def build_pymc_priors(self, model: Any) -> dict[str, Any]
```

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `model` | `pm.Model` | PyMC model context (должен быть активен) |

**Возвращает:** `dict[str, pm.Distribution]` — `{param_name: prior_rv}`.

**Алгоритм:**
1. Для каждого `spec` в `free_priors`:
   - В зависимости от `spec.distribution` создать соответствующий `pm.Distribution`
   - См. таблицу mapping в секции PriorSpec
2. Вернуть dict

**Граничные случаи:**
- Вызов вне `pm.Model` context → `TypeError` (PyMC requirement)
- Неизвестный `distribution` → `ValueError`

**Тестовые сценарии:**

| Сценарий | Вход | Ожидание |
|----------|------|----------|
| Normal prior | `distribution="normal"` | `pm.Normal` объект |
| LogNormal prior | `distribution="lognormal"` | `pm.LogNormal` объект |
| Пустые priors | Нет free params | Пустой dict |

#### `build_log_prior_fn`

**Назначение:** Создание функции log-prior для emcee.

**Сигнатура:**
```python
def build_log_prior_fn(self) -> Callable[[np.ndarray], float]
```

**Возвращает:** Функция `theta → log_prior(theta)`. Возвращает `-inf` если theta вне support.

**Алгоритм:**
1. Для каждого `spec` в `free_priors` определить log-pdf
2. Вернуть замыкание: `sum(log_pdf_i(theta[i]) for i in range(n))`

**Тестовые сценарии:**

| Сценарий | Вход | Ожидание |
|----------|------|----------|
| В support | theta в разумных пределах | Конечное float значение |
| Вне support | Отрицательный theta для lognormal | `-inf` |
| Uniform bounds | theta за пределами [lower, upper] | `-inf` |

#### `build_scipy_bounds`

**Назначение:** Создание bounds для `scipy.optimize`.

**Сигнатура:**
```python
def build_scipy_bounds(self) -> list[tuple[float, float]]
```

**Возвращает:** `[(lower, upper), ...]` для каждого свободного параметра.

**Алгоритм:**
- Для каждого `spec` в `free_priors`: определить bounds из таблицы mapping

**Тестовые сценарии:**

| Сценарий | Вход | Ожидание |
|----------|------|----------|
| LogNormal | `distribution="lognormal"`, mean=0.03 | `(0, 0.3)` |
| Uniform | lower=0.01, upper=0.1 | `(0.01, 0.1)` |
| Длина | 5 free params | 5 tuples |

#### `get_initial_guess`

**Назначение:** Начальное приближение — mean каждого свободного приора.

**Сигнатура:**
```python
def get_initial_guess(self) -> np.ndarray
```

**Возвращает:** `np.ndarray` shape `(n_free_params,)` — литературные значения.

**Тестовые сценарии:**

| Сценарий | Вход | Ожидание |
|----------|------|----------|
| Один параметр | `PriorSpec(name="r_F", mean=0.03)` | `array([0.03])` |
| Несколько | 3 priors | Shape `(3,)`, значения = means |

---

### BaseEstimator

**Назначение:** Базовый класс для всех методов параметрической идентификации.
Определяет единый интерфейс `fit() → EstimationResult` и общие вспомогательные методы.

**Зависимости:**
- `ForwardModelWrapper` — forward model
- `TimeSeriesData` — наблюдения
- `EstimationConfig` — конфигурация
- `PriorBuilder` — приоры

#### `__init__`

**Сигнатура:**
```python
def __init__(
    self,
    forward_model: ForwardModelWrapper,
    observed_data: TimeSeriesData,
    config: EstimationConfig,
    prior_builder: PriorBuilder,
) -> None
```

**Алгоритм:**
1. Сохранить все аргументы как атрибуты
2. Извлечь `observed_values` из `observed_data.values` для переменных из `config.observed_variables`

#### `fit`

**Назначение:** Запуск оценки параметров. Абстрактный — переопределяется в подклассах.

**Сигнатура:**
```python
def fit(self) -> EstimationResult
```

#### `_compute_log_likelihood`

**Назначение:** Вычисление log-likelihood для вектора параметров theta.

**Сигнатура:**
```python
def _compute_log_likelihood(self, theta: np.ndarray) -> float
```

**Алгоритм:**
1. `forward_model.predict(theta)` → предсказания
2. Для каждой наблюдаемой переменной:
   - residuals = observed - predicted
   - Если `noise_model == "gaussian"`: `log_lik += -0.5 * sum(residuals² / sigma²)`
   - Если `noise_model == "lognormal"`: `log_lik += -0.5 * sum((log(obs) - log(pred))² / sigma²)`
3. Если предсказания содержат NaN → вернуть `-inf`

**Граничные случаи:**
- NaN в предсказаниях → `-inf`
- `sigma_obs = None` → использовать оценку из данных (std residuals)
- Отрицательные предсказания при lognormal → `-inf`

**Тестовые сценарии:**

| Сценарий | Вход | Ожидание |
|----------|------|----------|
| Идеальное совпадение | predicted == observed | log_lik ≈ 0 (максимум) |
| NaN предсказания | theta вызывает divergence | `-inf` |
| Большие отклонения | predicted >> observed | Сильно отрицательный log_lik |

#### `_build_fitted_params`

**Назначение:** Создание `ParameterSet` с подставленными оценёнными параметрами.

**Сигнатура:**
```python
def _build_fitted_params(self, theta: np.ndarray) -> ParameterSet
```

**Алгоритм:** Делегирует `forward_model._build_parameter_set(theta)`.

#### `_compute_information_criteria`

**Назначение:** Вычисление AIC и BIC из log-likelihood.

**Сигнатура:**
```python
def _compute_information_criteria(
    self, log_lik: float, n_params: int, n_obs: int,
) -> tuple[float, float]
```

**Возвращает:** `(aic, bic)`.

**Алгоритм:**
- `AIC = 2*k - 2*log_lik`
- `BIC = k*log(n) - 2*log_lik`
- где `k = n_params`, `n = n_obs`

**Граничные случаи:**
- `n_obs <= 0` → `ValueError`
- `log_lik = -inf` → `aic = inf`, `bic = inf`

**Тестовые сценарии:**

| Сценарий | Вход | Ожидание |
|----------|------|----------|
| Стандарт | log_lik=-100, k=5, n=50 | AIC=210, BIC≈219.56 |
| n_obs=0 | n=0 | `ValueError` |

---

### BayesianEstimator

**Назначение:** Байесовская оценка параметров через PyMC 5 с NUTS sampler.
Строит полную вероятностную модель с информативными приорами из литературы
и получает posterior распределения для 40+ параметров.

**Наследует:** `BaseEstimator`

**Зависимости:** `pymc` (lazy import)

#### `fit`

**Назначение:** Полный Байесовский вывод: построение модели → NUTS sampling → извлечение результатов.

**Сигнатура:**
```python
def fit(self) -> EstimationResult
```

**Алгоритм:**
1. `import pymc as pm`
2. `model = self._build_pymc_model()`
3. `idata = self._sample(model)`
4. `result = self._extract_results(idata)`
5. Добавить diagnostics через `ConvergenceAnalyzer`
6. Вернуть `EstimationResult`

**Граничные случаи:**
- Divergences в NUTS → warning в `diagnostics.warnings`
- PyMC не установлен → `ImportError` с понятным сообщением

#### `_build_pymc_model`

**Назначение:** Создание `pm.Model` с прiors и likelihood.

**Сигнатура:**
```python
def _build_pymc_model(self) -> Any
```

**Возвращает:** `pm.Model`

**Алгоритм:**
1. `pm.Model()` context
2. `prior_builder.build_pymc_priors(model)` → prior RVs
3. Для likelihood: использовать `pm.Potential` с custom log-likelihood функцией
   (т.к. likelihood вычисляется через SDE forward model, не стандартный PyMC likelihood)
4. Вернуть model

**Ключевая логика:** Поскольку likelihood требует запуска SDE симуляции, используется
`pm.Potential("likelihood", log_lik_value)` вместо стандартных observed distributions.
PyMC будет вызывать forward model через Theano/PyTensor Op.

#### `_sample`

**Назначение:** Запуск NUTS sampler.

**Сигнатура:**
```python
def _sample(self, model: Any) -> Any
```

**Возвращает:** `az.InferenceData`

**Алгоритм:**
1. `pm.sample(draws=n_samples, tune=n_tune, chains=n_chains, target_accept=target_accept, random_seed=rng_seed, return_inferencedata=True)`
2. Вернуть InferenceData

#### `_extract_results`

**Назначение:** Извлечение результатов из InferenceData в `EstimationResult`.

**Сигнатура:**
```python
def _extract_results(self, idata: Any) -> EstimationResult
```

**Алгоритм:**
1. Для каждого параметра:
   - `point_estimates[name]` = posterior mean
   - `ci_lower[name]` = 2.5% quantile
   - `ci_upper[name]` = 97.5% quantile
   - `posterior_samples[name]` = все samples (flattened across chains)
2. `diagnostics` = `ConvergenceAnalyzer(config).analyze(idata)`
3. `fitted_params` = `_build_fitted_params(point_estimates_as_theta)`
4. `log_likelihood`, `aic`, `bic`

**Тестовые сценарии:**

| Сценарий | Вход | Ожидание |
|----------|------|----------|
| Успешный fit | Синтетические данные | `diagnostics.converged = True` для простой модели |
| Все параметры | 40+ params | 40+ entries в `point_estimates` и `ci_*` |
| CI содержит true | Данные от known params | true value ∈ [ci_lower, ci_upper] (95%) |

---

### MCMCEstimator

**Назначение:** MCMC оценка параметров через emcee (Ensemble Sampler).
Альтернатива PyMC для случаев когда нужен gradient-free sampler.
emcee хорошо работает с "black-box" likelihood (SDE симуляция).

**Наследует:** `BaseEstimator`

**Зависимости:** `emcee` (lazy import)

#### `fit`

**Назначение:** Полный MCMC sampling через emcee.

**Сигнатура:**
```python
def fit(self) -> EstimationResult
```

**Алгоритм:**
1. `import emcee`
2. `walkers = self._initialize_walkers()`
3. `sampler = self._run_sampler()`
4. `idata = self._sampler_to_inference_data(sampler)`
5. Извлечь results (аналогично BayesianEstimator._extract_results)
6. Добавить diagnostics
7. Вернуть `EstimationResult`

#### `_log_probability`

**Назначение:** Log-posterior = log-prior + log-likelihood.

**Сигнатура:**
```python
def _log_probability(self, theta: np.ndarray) -> float
```

**Алгоритм:**
1. `log_prior = self._log_prior_fn(theta)` (из PriorBuilder)
2. Если `log_prior == -inf` → вернуть `-inf`
3. `log_lik = self._compute_log_likelihood(theta)`
4. Вернуть `log_prior + log_lik`

**Граничные случаи:**
- theta вне support приора → `-inf` (быстрый reject, без запуска SDE)
- SDE divergence → `_compute_log_likelihood` вернёт `-inf`

**Тестовые сценарии:**

| Сценарий | Вход | Ожидание |
|----------|------|----------|
| В support | Разумный theta | Конечное float |
| Вне support | Отрицательный rate | `-inf` |

#### `_initialize_walkers`

**Назначение:** Начальные позиции walkers вокруг initial guess.

**Сигнатура:**
```python
def _initialize_walkers(self) -> np.ndarray
```

**Возвращает:** `np.ndarray` shape `(n_walkers, n_params)`

**Алгоритм:**
1. `initial = prior_builder.get_initial_guess()` → shape `(n_params,)`
2. Для каждого walker: `initial * (1 + 0.01 * np.random.randn(n_params))`
   (1% perturbation от initial guess)
3. Вернуть массив

**Граничные случаи:**
- `n_walkers < 2 * n_params` → `ValueError` (emcee requirement)

**Тестовые сценарии:**

| Сценарий | Вход | Ожидание |
|----------|------|----------|
| Shape | n_walkers=32, n_params=5 | Shape `(32, 5)` |
| Центр | | Среднее по walkers ≈ initial_guess |

#### `_run_sampler`

**Назначение:** Запуск emcee.EnsembleSampler.

**Сигнатура:**
```python
def _run_sampler(self) -> Any
```

**Возвращает:** `emcee.EnsembleSampler` (с результатами)

**Алгоритм:**
1. `sampler = emcee.EnsembleSampler(n_walkers, n_params, self._log_probability)`
2. `sampler.run_mcmc(initial_walkers, n_samples + n_tune, progress=True)`
3. Вернуть sampler

#### `_sampler_to_inference_data`

**Назначение:** Конвертация emcee результатов в `az.InferenceData` для единой диагностики.

**Сигнатура:**
```python
def _sampler_to_inference_data(self, sampler: Any) -> Any
```

**Возвращает:** `az.InferenceData`

**Алгоритм:**
1. `import arviz as az`
2. `chain = sampler.get_chain(discard=n_tune, flat=False)` → shape `(n_samples, n_walkers, n_params)`
3. Reshape для ArviZ: walkers → chains
4. `az.from_dict(posterior={name: chain[:, :, i] for i, name in enumerate(param_names)})`

**Тестовые сценарии:**

| Сценарий | Вход | Ожидание |
|----------|------|----------|
| Конвертация | Заполненный sampler | Валидный InferenceData |
| Имена params | | Правильные var_names в idata |

---

### MLEstimator

**Назначение:** Оценка максимального правдоподобия через `scipy.optimize`.
Быстрая точечная оценка для:
- Проверки начальных приближений перед MCMC
- Быстрый pipeline без Байесовского анализа
- Инициализация MCMC цепей

**Наследует:** `BaseEstimator`

**Зависимости:** `scipy.optimize` (lazy import)

#### `fit`

**Назначение:** MLE: minimize(-log_likelihood) with scipy.optimize.

**Сигнатура:**
```python
def fit(self) -> EstimationResult
```

**Алгоритм:**
1. `from scipy.optimize import minimize`
2. `x0 = prior_builder.get_initial_guess()`
3. `bounds = prior_builder.build_scipy_bounds()`
4. `result = minimize(self._objective, x0, method=mle_method, bounds=bounds, options={"maxiter": mle_maxiter})`
5. `theta_opt = result.x`
6. `ci_lower, ci_upper = self._estimate_ci_from_hessian(theta_opt)`
7. Собрать `EstimationResult` (без posterior_samples, diagnostics)

**Граничные случаи:**
- Оптимизация не сходится (`result.success == False`) → warning в result
- Hessian singular → CI = `(NaN, NaN)`

**Тестовые сценарии:**

| Сценарий | Вход | Ожидание |
|----------|------|----------|
| Простая модель | 2-3 params, синтетические данные | `result.success == True` |
| Результат формат | | `method="mle_scipy"`, `posterior_samples=None` |

#### `_objective`

**Назначение:** Целевая функция: `-log_likelihood(theta)` (минимизация).

**Сигнатура:**
```python
def _objective(self, theta: np.ndarray) -> float
```

**Возвращает:** `-log_likelihood(theta)`. Возвращает `inf` если log_lik = `-inf`.

#### `_estimate_ci_from_hessian`

**Назначение:** Оценка 95% CI из обратного Гессиана (Wald intervals).

**Сигнатура:**
```python
def _estimate_ci_from_hessian(
    self, theta_opt: np.ndarray,
) -> tuple[dict[str, float], dict[str, float]]
```

**Возвращает:** `(ci_lower_dict, ci_upper_dict)`

**Алгоритм:**
1. Численный Гессиан: `H = approx_hessian(self._objective, theta_opt)`
   (конечные разности или `scipy.optimize.approx_fprime`)
2. Ковариационная матрица: `cov = inv(H)`
3. Стандартные ошибки: `se = sqrt(diag(cov))`
4. 95% CI: `theta_opt ± 1.96 * se`

**Граничные случаи:**
- Гессиан сингулярный → `se = NaN`, CI = `(NaN, NaN)` + warning
- Отрицательная дисперсия на диагонали → warning

**Тестовые сценарии:**

| Сценарий | Вход | Ожидание |
|----------|------|----------|
| Квадратичная поверхность | theta_opt в минимуме | CI определены, конечные |
| Сингулярный Гессиан | Плоская поверхность | CI содержат NaN |
| CI ширина | | ci_upper > ci_lower для каждого параметра |

---

### ConvergenceAnalyzer

**Назначение:** Анализ сходимости MCMC цепей через ArviZ.
Вычисляет R-hat, ESS, генерирует summary. Используется обоими MCMC estimator-ами.

**Зависимости:** `arviz` (lazy import)

#### `__init__`

**Сигнатура:**
```python
def __init__(self, config: EstimationConfig) -> None
```

**Алгоритм:** Сохранить `config` (для `rhat_threshold`, `ess_min`).

#### `analyze`

**Назначение:** Полная диагностика сходимости.

**Сигнатура:**
```python
def analyze(self, inference_data: Any) -> ConvergenceDiagnostics
```

**Возвращает:** `ConvergenceDiagnostics`

**Алгоритм:**
1. `rhat = self.compute_rhat(inference_data)`
2. `ess_bulk, ess_tail = self.compute_ess(inference_data)`
3. `summary_table = self.summary(inference_data)`
4. `converged = self.check_convergence(ConvergenceDiagnostics(rhat, ess_bulk, ess_tail))`
5. Собрать warnings для несошедшихся параметров
6. Вернуть `ConvergenceDiagnostics`

**Граничные случаи:**
- Одна цепь → R-hat не определён → warning
- Очень мало samples → ESS может быть < 1

#### `compute_rhat`

**Назначение:** R-hat (Gelman-Rubin) для каждого параметра.

**Сигнатура:**
```python
def compute_rhat(self, inference_data: Any) -> dict[str, float]
```

**Алгоритм:**
1. `import arviz as az`
2. `rhat_data = az.rhat(inference_data)`
3. Конвертировать в dict

**Тестовые сценарии:**

| Сценарий | Вход | Ожидание |
|----------|------|----------|
| Сошедшиеся цепи | Хорошо перемешанные | Все R-hat ∈ [0.99, 1.05] |
| Не сошедшиеся | Разные цепи в разных модах | Некоторые R-hat > 1.05 |

#### `compute_ess`

**Назначение:** ESS bulk и tail для каждого параметра.

**Сигнатура:**
```python
def compute_ess(
    self, inference_data: Any,
) -> tuple[dict[str, float], dict[str, float]]
```

**Возвращает:** `(ess_bulk_dict, ess_tail_dict)`

**Алгоритм:**
1. `az.ess(inference_data, method="bulk")` → bulk
2. `az.ess(inference_data, method="tail")` → tail
3. Конвертировать в dicts

**Тестовые сценарии:**

| Сценарий | Вход | Ожидание |
|----------|------|----------|
| Достаточно samples | 2000 draws, 4 chains | ESS > 100 для каждого |
| Мало samples | 10 draws | ESS < 100 |

#### `summary`

**Назначение:** Таблица `az.summary()` с основными статистиками.

**Сигнатура:**
```python
def summary(self, inference_data: Any) -> Any
```

**Возвращает:** `pd.DataFrame` из `az.summary()` (mean, sd, hdi_3%, hdi_97%, ess, rhat).

#### `check_convergence`

**Назначение:** Проверка: все R-hat < threshold И все ESS > min.

**Сигнатура:**
```python
def check_convergence(self, diagnostics: ConvergenceDiagnostics) -> bool
```

**Алгоритм:**
1. Для каждого параметра: `rhat[name] < rhat_threshold`
2. Для каждого параметра: `ess_bulk[name] > ess_min` AND `ess_tail[name] > ess_min`
3. Вернуть `True` только если ВСЕ условия выполнены

**Тестовые сценарии:**

| Сценарий | Вход | Ожидание |
|----------|------|----------|
| Всё ОК | Все rhat<1.05, ess>100 | `True` |
| Один плохой rhat | rhat["r_F"]=1.2 | `False` |
| Один низкий ESS | ess_bulk["r_F"]=50 | `False` |
| Пустые метрики | Пустые dicts | `False` |

---

## Функции

### estimate_parameters

**Назначение:** Удобная функция для запуска параметрической идентификации.
Создаёт все необходимые объекты (ForwardModelWrapper, PriorBuilder, Estimator)
и вызывает `fit()`. Для быстрого использования без ручного конструирования.

**Сигнатура:**
```python
def estimate_parameters(
    observed_data: TimeSeriesData,
    method: str = "bayesian",
    initial_state: ExtendedSDEState | None = None,
    base_params: ParameterSet | None = None,
    estimated_param_names: list[str] | None = None,
    config: EstimationConfig | None = None,
) -> EstimationResult
```

**Параметры:**

| Параметр | Тип | Описание | По умолчанию |
|----------|-----|----------|-------------|
| `observed_data` | `TimeSeriesData` | Наблюдательные данные | (обязательно) |
| `method` | `str` | `"bayesian"`, `"mcmc"`, `"mle"` | `"bayesian"` |
| `initial_state` | `ExtendedSDEState \| None` | Начальное состояние SDE | `None` → defaults |
| `base_params` | `ParameterSet \| None` | Базовые параметры | `None` → `ParameterSet()` |
| `estimated_param_names` | `list[str] \| None` | Какие параметры оценивать | `None` → все кроме numerical/sigma |
| `config` | `EstimationConfig \| None` | Конфигурация | `None` → defaults |

**Возвращает:** `EstimationResult`

**Алгоритм:**
1. Заполнить defaults:
   - `base_params` → `ParameterSet.get_literature_defaults()`
   - `initial_state` → `ExtendedSDEState()` с defaults
   - `estimated_param_names` → все поля `ParameterSet` кроме dt, t_max, epsilon, sigma_*
   - `config` → `EstimationConfig()` с `observed_variables` из `observed_data.values.keys()`
2. `prior_builder = PriorBuilder.from_parameter_set(base_params, estimated_param_names)`
3. `forward_model = ForwardModelWrapper(base_params, initial_state, estimated_param_names, ...)`
4. Выбрать estimator:
   - `"bayesian"` → `BayesianEstimator`
   - `"mcmc"` → `MCMCEstimator`
   - `"mle"` → `MLEstimator`
5. `estimator.fit()` → result

**Граничные случаи:**
- `method` неизвестен → `ValueError`
- `observed_data.values` пуст → `ValueError`
- `observed_data` содержит переменные не из SDE → `ValueError`

**Тестовые сценарии:**

| Сценарий | Вход | Ожидание |
|----------|------|----------|
| Default method | `method="bayesian"` | Вызывается BayesianEstimator |
| MLE | `method="mle"` | Вызывается MLEstimator |
| Неизвестный метод | `method="unknown"` | `ValueError` |
| Минимальный вызов | Только observed_data | Результат с defaults |

---

## Примеры использования

```python
from src.core.parameter_estimation import (
    estimate_parameters,
    EstimationConfig,
    PriorSpec,
    PriorBuilder,
    BayesianEstimator,
    MCMCEstimator,
    MLEstimator,
    ForwardModelWrapper,
    ConvergenceAnalyzer,
)
from src.core.parameters import ParameterSet
from src.core.extended_sde import ExtendedSDEState
from src.data.dataset_loader import TimeSeriesData

# === Quick start: convenience function ===
result = estimate_parameters(
    observed_data=my_time_series,
    method="mle",
    estimated_param_names=["r_F", "delta_Ne", "gamma_TNF"],
)
print(result.point_estimates)
print(result.ci_lower, result.ci_upper)

# === Full control: manual setup ===
params = ParameterSet.get_literature_defaults()
config = EstimationConfig(
    observed_variables=["F", "C_TNF"],
    n_samples=5000,
    n_chains=4,
)
prior_builder = PriorBuilder.from_parameter_set(
    params, ["r_F", "delta_F", "gamma_TNF"], default_cv=0.3,
)
forward_model = ForwardModelWrapper(
    base_params=params,
    initial_state=ExtendedSDEState(F=1000.0, C_TNF=5.0),
    estimated_param_names=prior_builder.get_free_param_names(),
    observed_variables=config.observed_variables,
    observation_times=my_time_series.time_points,
    config=config,
)
estimator = BayesianEstimator(forward_model, my_time_series, config, prior_builder)
result = estimator.fit()

# === Check convergence ===
if result.diagnostics and result.diagnostics.converged:
    print("Converged!")
    print(result.diagnostics.summary_table)
else:
    print("Warnings:", result.diagnostics.warnings)
```
