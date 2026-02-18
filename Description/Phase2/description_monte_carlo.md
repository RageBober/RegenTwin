# Описание: monte_carlo.py

## Обзор

Monte Carlo симулятор для стохастических моделей регенерации тканей. Запускает множество независимых траекторий и агрегирует результаты для получения ансамблевой статистики, доверительных интервалов и распределений.

---

## Теоретическое обоснование

### Принцип Monte Carlo

Для стохастических систем одна траектория недостаточна для характеристики поведения. Необходимо:

1. Запустить N независимых траекторий с разными seeds
2. Агрегировать результаты для получения статистик

### Оценка среднего и дисперсии

```
E[X] ≈ (1/N) Σᵢ xᵢ
Var[X] ≈ (1/(N-1)) Σᵢ (xᵢ - E[X])²
```

### Доверительные интервалы

Для уровня доверия 1-α:
```
CI = [μ - z_{α/2} · σ/√N, μ + z_{α/2} · σ/√N]
```

Или через квантили:
```
CI_95% = [Q_{0.025}, Q_{0.975}]
```

### Количество траекторий

| Точность | N траекторий |
|----------|--------------|
| Грубая оценка | 50-100 |
| Стандартная | 100-500 |
| Высокая | 500-1000 |
| Очень высокая | 1000+ |

---

## Классы

### MonteCarloConfig

**Назначение:** Конфигурация Monte Carlo симулятора.

**Атрибуты:**

| Атрибут | Тип | По умолчанию | Описание |
|---------|-----|--------------|----------|
| n_trajectories | int | 100 | Количество траекторий |
| model_type | str | "sde" | Тип модели ("sde", "abm", "integrated") |
| sde_config | SDEConfig \| None | None | Конфигурация SDE |
| abm_config | ABMConfig \| None | None | Конфигурация ABM |
| integration_config | IntegrationConfig \| None | None | Конфигурация интеграции |
| n_jobs | int | 1 | Параллельные процессы |
| use_multiprocessing | bool | False | Использовать multiprocessing |
| base_seed | int \| None | None | Базовый seed |
| quantiles | list[float] | [0.05, 0.25, 0.5, 0.75, 0.95] | Квантили для расчёта |
| progress_callback | Callable \| None | None | Callback прогресса |

---

### TrajectoryResult

**Назначение:** Результат одной траектории.

**Атрибуты:**

| Атрибут | Тип | Описание |
|---------|-----|----------|
| trajectory_id | int | ID траектории |
| random_seed | int \| None | Использованный seed |
| sde_trajectory | SDETrajectory \| None | SDE траектория |
| abm_trajectory | ABMTrajectory \| None | ABM траектория |
| integrated_trajectory | IntegratedTrajectory \| None | Интегрированная |
| final_N | float | Финальная плотность |
| final_C | float | Финальные цитокины |
| max_N | float | Максимальная плотность |
| growth_rate | float | Скорость роста |
| success | bool | Успешность |
| error_message | str \| None | Сообщение об ошибке |
| computation_time | float | Время вычисления (сек) |

---

### MonteCarloResults

**Назначение:** Агрегированные результаты всех траекторий.

**Атрибуты:**

| Атрибут | Тип | Описание |
|---------|-----|----------|
| trajectories | list[TrajectoryResult] | Все траектории |
| config | MonteCarloConfig | Конфигурация |
| times | np.ndarray | Временная ось |
| mean_N | np.ndarray | Средняя N(t) |
| std_N | np.ndarray | Стд. откл. N(t) |
| mean_C | np.ndarray | Средняя C(t) |
| std_C | np.ndarray | Стд. откл. C(t) |
| quantiles_N | dict[float, np.ndarray] | Квантили N |
| quantiles_C | dict[float, np.ndarray] | Квантили C |
| n_successful | int | Успешных траекторий |
| n_failed | int | Неудачных траекторий |
| total_computation_time | float | Общее время |

---

### MonteCarloSimulator

**Назначение:** Основной класс симулятора.

---

## Методы

### MonteCarloSimulator.run

**Сигнатура:**
```python
def run(self, initial_params: ModelParameters) -> MonteCarloResults
```

**Алгоритм:**
```python
import time

results = []
start_time = time.time()

# Запуск траекторий
for i in range(self._config.n_trajectories):
    # Progress callback
    if self._config.progress_callback:
        self._config.progress_callback(i, self._config.n_trajectories)

    # Запуск одной траектории
    result = self._run_single_trajectory(
        trajectory_id=i,
        initial_params=initial_params,
        random_seed=self._seeds[i],
    )
    results.append(result)

# Агрегация
mc_results = self._aggregate_trajectories(results)
mc_results.total_computation_time = time.time() - start_time

return mc_results
```

---

### MonteCarloSimulator._run_single_trajectory

**Сигнатура:**
```python
def _run_single_trajectory(
    self,
    trajectory_id: int,
    initial_params: ModelParameters,
    random_seed: int | None,
) -> TrajectoryResult
```

**Алгоритм:**
```python
import time

start = time.time()
result = TrajectoryResult(trajectory_id=trajectory_id, random_seed=random_seed)

try:
    if self._config.model_type == "sde":
        trajectory = self._run_sde_trajectory(initial_params, random_seed)
        result.sde_trajectory = trajectory
        result.final_N = trajectory.N_values[-1]
        result.final_C = trajectory.C_values[-1]
        result.max_N = np.max(trajectory.N_values)

    elif self._config.model_type == "abm":
        trajectory = self._run_abm_trajectory(initial_params, random_seed)
        result.abm_trajectory = trajectory
        # Извлечь статистики из ABM

    elif self._config.model_type == "integrated":
        trajectory = self._run_integrated_trajectory(initial_params, random_seed)
        result.integrated_trajectory = trajectory
        # Извлечь статистики

    # Рассчитать growth_rate
    result.growth_rate = self._calculate_growth_rate(result)
    result.success = True

except Exception as e:
    result.success = False
    result.error_message = str(e)

result.computation_time = time.time() - start
return result
```

---

### MonteCarloSimulator._aggregate_trajectories

**Сигнатура:**
```python
def _aggregate_trajectories(
    self,
    results: list[TrajectoryResult],
) -> MonteCarloResults
```

**Алгоритм:**
```python
# Фильтрация успешных траекторий
successful = [r for r in results if r.success]
n_successful = len(successful)
n_failed = len(results) - n_successful

if n_successful == 0:
    raise ValueError("Все траектории завершились с ошибкой")

# Извлечь массивы траекторий
times, N_array = self._extract_trajectories_array(successful, "N")
_, C_array = self._extract_trajectories_array(successful, "C")

# Статистики
mean_N = np.mean(N_array, axis=0)
std_N = np.std(N_array, axis=0)
mean_C = np.mean(C_array, axis=0)
std_C = np.std(C_array, axis=0)

# Квантили
quantiles_N = self._calculate_quantiles(N_array, self._config.quantiles)
quantiles_C = self._calculate_quantiles(C_array, self._config.quantiles)

return MonteCarloResults(
    trajectories=results,
    config=self._config,
    times=times,
    mean_N=mean_N,
    std_N=std_N,
    mean_C=mean_C,
    std_C=std_C,
    quantiles_N=quantiles_N,
    quantiles_C=quantiles_C,
    n_successful=n_successful,
    n_failed=n_failed,
)
```

---

### MonteCarloResults.get_confidence_interval

**Сигнатура:**
```python
def get_confidence_interval(
    self,
    variable: str = "N",
    confidence_level: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]
```

**Алгоритм:**
```python
alpha = 1 - confidence_level
lower_q = alpha / 2
upper_q = 1 - alpha / 2

if variable == "N":
    quantiles = self.quantiles_N
elif variable == "C":
    quantiles = self.quantiles_C
else:
    raise ValueError(f"Unknown variable: {variable}")

# Найти ближайшие квантили
available_q = sorted(quantiles.keys())

lower_key = min(available_q, key=lambda x: abs(x - lower_q))
upper_key = min(available_q, key=lambda x: abs(x - upper_q))

return (quantiles[lower_key], quantiles[upper_key])
```

---

## Примеры использования

```python
from src.data.parameter_extraction import ModelParameters
from src.core.sde_model import SDEConfig, TherapyProtocol
from src.core.monte_carlo import (
    MonteCarloSimulator,
    MonteCarloConfig,
    run_monte_carlo,
    compare_therapies,
)

# Начальные параметры
params = ModelParameters(
    n0=5000.0,
    c0=10.0,
    stem_cell_fraction=0.05,
    macrophage_fraction=0.03,
    apoptotic_fraction=0.02,
    inflammation_level=0.3,
)

# Конфигурация Monte Carlo
mc_config = MonteCarloConfig(
    n_trajectories=100,
    model_type="sde",
    sde_config=SDEConfig(t_max=30.0),
    base_seed=42,
    quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
)

# Способ 1: convenience функция
results = run_monte_carlo(
    initial_params=params,
    config=mc_config,
)

# Анализ результатов
print(f"Успешных: {results.n_successful}/{results.n_successful + results.n_failed}")
print(f"Финальная N: {results.mean_N[-1]:.0f} ± {results.std_N[-1]:.0f}")

# Доверительный интервал
lower, upper = results.get_confidence_interval("N", confidence_level=0.95)
print(f"95% CI в конце: [{lower[-1]:.0f}, {upper[-1]:.0f}]")

# Распределение финальных значений
final_dist = results.get_final_distribution("N")
print(f"Медиана финальной N: {np.median(final_dist):.0f}")

# Способ 2: сравнение терапий
therapies = {
    "control": TherapyProtocol(),
    "prp": TherapyProtocol(prp_enabled=True, prp_intensity=1.0),
    "pemf": TherapyProtocol(pemf_enabled=True, pemf_intensity=1.0),
    "combined": TherapyProtocol(
        prp_enabled=True, pemf_enabled=True,
        synergy_factor=1.3,
    ),
}

comparison = compare_therapies(
    initial_params=params,
    therapies=therapies,
    config=mc_config,
)

for name, res in comparison.items():
    print(f"{name}: final N = {res.mean_N[-1]:.0f} ± {res.std_N[-1]:.0f}")
```

---

## Параллелизация

```python
# Последовательный запуск (по умолчанию)
config = MonteCarloConfig(n_trajectories=100, n_jobs=1)

# Параллельный запуск (требует multiprocessing)
config = MonteCarloConfig(
    n_trajectories=100,
    n_jobs=4,
    use_multiprocessing=True,
)
```

**Примечание:** Multiprocessing требует дополнительной реализации с учётом сериализации объектов.

---

## Seed Management

Для воспроизводимости используется управление seeds:

```python
# С base_seed - воспроизводимые результаты
config = MonteCarloConfig(base_seed=42)

# Seeds для траекторий генерируются детерминистически:
# seed_i = rng.integers(0, 2^31) где rng = np.random.default_rng(base_seed)
```

---

## Зависимости

- numpy
- dataclasses (stdlib)
- typing (stdlib)
- src.core.sde_model
- src.core.abm_model
- src.core.integration
- src.data.parameter_extraction

---

## Параллелизация (Phase 2 расширение)

### MonteCarloSimulator._run_parallel

**Назначение:** Параллельный запуск траекторий через ProcessPoolExecutor.

**Сигнатура:**

```python
def _run_parallel(self, initial_params: ModelParameters) -> list[TrajectoryResult]
```

**Поведение:**
1. Разделить n_trajectories на n_jobs частей
2. Для каждой части -- создать seed (base_seed + offset)
3. Запустить через ProcessPoolExecutor
4. Собрать результаты в единый список
5. При ошибке в процессе -- пометить как failed

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| n_jobs=1 | Эквивалентно sequential |
| n_jobs=2, n_trajectories=10 | 10 результатов |
| base_seed=42, дважды | Идентичные результаты |
| Одна траектория fails | n_successful = n-1 |
| n_jobs=1, n_trajectories=1 | 1 результат |

**Инварианты:**
- len(results) == n_trajectories
- Воспроизводимость при фиксированном seed
- Результаты не зависят от n_jobs (только скорость)

---

### MonteCarloSimulator._progress_callback_wrapper

**Назначение:** Thread-safe обёртка для progress_callback при параллельном выполнении.

**Сигнатура:**

```python
def _progress_callback_wrapper(self, completed: int, total: int) -> None
```

**Поведение:**
1. Захватить threading.Lock
2. Обновить суммарный прогресс
3. Вызвать пользовательский callback с (total_completed, n_trajectories)

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Один вызов (5, 10) | callback(5, total) |
| Два последовательных вызова | Корректная агрегация |
| callback=None | Без ошибок |
| Конкурентные вызовы | Thread-safe (через Lock) |

**Инварианты:**
- Thread-safe (используется Lock)
- total_completed монотонно растёт

---

### MonteCarloSimulator._validate_parallel_config

**Назначение:** Проверка конфигурации для параллельного запуска.

**Сигнатура:**

```python
def _validate_parallel_config(self, config: MonteCarloConfig) -> bool
```

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| n_jobs=1 | True |
| n_jobs > cpu_count | ValueError |
| n_jobs=0 | ValueError |
| multiprocessing недоступен | RuntimeError |

**Ошибки:**
- `ValueError`: n_jobs > os.cpu_count() или n_jobs < 1
- `RuntimeError`: multiprocessing модуль недоступен

**Инварианты:**
- True → safe to run _run_parallel
- 1 ≤ n_jobs ≤ cpu_count
