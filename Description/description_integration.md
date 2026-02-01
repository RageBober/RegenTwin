# Описание: integration.py

## Обзор

Модуль интеграции SDE и ABM моделей для мультимасштабного моделирования регенерации тканей. Связывает непрерывную макроскопическую динамику (SDE) с дискретными микроскопическими событиями (ABM) через метод operator splitting.

---

## Теоретическое обоснование

### Мультимасштабное моделирование

| Уровень | Модель | Переменные | Масштаб времени |
|---------|--------|------------|-----------------|
| Макро | SDE | N(t), C(t) | Дни |
| Микро | ABM | Агенты (x, y, state) | Часы |

### Operator Splitting

Метод расщепления оператора позволяет решать сложную систему по частям:

```
∂u/∂t = L₁(u) + L₂(u)
```

Разбивается на:
1. Решить ∂u/∂t = L₁(u) на [t, t+Δt]
2. Решить ∂u/∂t = L₂(u) на [t, t+Δt]
3. Синхронизировать состояния

### Связь SDE ↔ ABM

**SDE → ABM:**
- Концентрация цитокинов C(t) из SDE обновляет cytokine_field в ABM
- Влияет на поведение агентов (хемотаксис, пролиферация)

**ABM → SDE:**
- Количество агентов корректирует плотность N(t) в SDE
- Учитывает реальную пространственную динамику

### Коррекция рассогласования

```
N_corrected = N_sde + α · (N_abm - N_sde)
```

Где α = coupling_strength · correction_rate.

---

## Классы

### IntegrationConfig

**Назначение:** Конфигурация интегрированной модели.

**Атрибуты:**

| Атрибут | Тип | По умолчанию | Описание |
|---------|-----|--------------|----------|
| sde_config | SDEConfig | SDEConfig() | Конфигурация SDE |
| abm_config | ABMConfig | ABMConfig() | Конфигурация ABM |
| sync_interval | float | 1.0 | Интервал синхронизации (часы) |
| coupling_strength | float | 0.5 | Сила связи ABM→SDE [0, 1] |
| mode | str | "bidirectional" | Режим интеграции |
| correction_rate | float | 0.1 | Скорость коррекции [0, 1] |
| max_discrepancy | float | 0.5 | Макс. допустимое рассогласование |

**Режимы интеграции:**

| Режим | Описание |
|-------|----------|
| "sde_only" | Только SDE, ABM для визуализации |
| "abm_only" | Только ABM, SDE для глобальных метрик |
| "bidirectional" | Полная двусторонняя связь |

---

### IntegratedState

**Назначение:** Состояние системы в точке синхронизации.

**Атрибуты:**

| Атрибут | Тип | Описание |
|---------|-----|----------|
| t | float | Время (часы) |
| sde_N | float | Плотность из SDE |
| sde_C | float | Цитокины из SDE |
| abm_agent_counts | dict[str, int] | Агенты по типам |
| abm_total | int | Всего агентов |
| discrepancy | float | Рассогласование |
| correction_applied | float | Применённая коррекция |

---

### IntegratedTrajectory

**Назначение:** Результат интегрированной симуляции.

**Атрибуты:**

| Атрибут | Тип | Описание |
|---------|-----|----------|
| times | np.ndarray | Точки синхронизации |
| states | list[IntegratedState] | Состояния |
| sde_trajectory | SDETrajectory | Полная SDE траектория |
| abm_trajectory | ABMTrajectory | Полная ABM траектория |
| config | IntegrationConfig | Конфигурация |

---

### IntegratedModel

**Назначение:** Основной класс интегрированной модели.

---

## Методы

### IntegratedModel.simulate

**Сигнатура:**
```python
def simulate(self, initial_params: ModelParameters) -> IntegratedTrajectory
```

**Алгоритм:**
```python
# 1. Инициализация
self._abm_model.initialize_from_parameters(initial_params)
N = initial_params.n0
C = initial_params.c0
t = 0.0

sync_states = []
sde_N_history = [N]
sde_C_history = [C]
abm_snapshots = []

# 2. Определить точки синхронизации
t_max_hours = self._config.sde_config.t_max * 24  # дни → часы
sync_times = np.arange(0, t_max_hours, self._config.sync_interval)

# 3. Цикл по интервалам синхронизации
for i in range(len(sync_times) - 1):
    t_start = sync_times[i]
    t_end = sync_times[i + 1]

    # 3a. Запуск SDE сегмента
    N, C, N_seg, C_seg = self._run_sde_segment(t_start, t_end, N, C)
    sde_N_history.extend(N_seg)
    sde_C_history.extend(C_seg)

    # 3b. Запуск ABM сегмента
    abm_snapshot = self._run_abm_segment(t_start, t_end)
    abm_snapshots.append(abm_snapshot)

    # 3c. Синхронизация
    N_corrected, C_corrected, discrepancy = self._synchronize(
        N, C, abm_snapshot
    )

    # 3d. Сохранение состояния
    correction = N_corrected - N
    state = self._create_integrated_state(
        t_end, N_corrected, C_corrected, abm_snapshot,
        discrepancy, correction
    )
    sync_states.append(state)

    # Обновить для следующей итерации
    N = N_corrected
    C = C_corrected

# 4. Создать траектории
return IntegratedTrajectory(
    times=sync_times,
    states=sync_states,
    sde_trajectory=SDETrajectory(...),
    abm_trajectory=ABMTrajectory(snapshots=abm_snapshots, ...),
    config=self._config,
)
```

---

### IntegratedModel._synchronize

**Сигнатура:**
```python
def _synchronize(
    self,
    sde_N: float,
    sde_C: float,
    abm_snapshot: ABMSnapshot,
) -> tuple[float, float, float]
```

**Алгоритм:**
```python
# 1. Получить количество агентов ABM
abm_total = abm_snapshot.get_total_agents()

# 2. Рассчитать рассогласование
discrepancy = self._calculate_discrepancy(sde_N, abm_total)

# 3. Применить коррекцию (если режим bidirectional или abm_only)
if self._config.mode in ["bidirectional", "abm_only"]:
    N_corrected = self._apply_correction(sde_N, abm_total, discrepancy)
else:
    N_corrected = sde_N

# 4. Обновить ABM окружение (если режим bidirectional или sde_only)
if self._config.mode in ["bidirectional", "sde_only"]:
    self._update_abm_environment(sde_C)

return (N_corrected, sde_C, discrepancy)
```

---

### IntegratedModel._calculate_discrepancy

**Сигнатура:**
```python
def _calculate_discrepancy(self, sde_N: float, abm_count: int) -> float
```

**Формула:**
```python
# Относительное рассогласование
if sde_N > 0:
    discrepancy = abs(sde_N - abm_count) / sde_N
else:
    discrepancy = float(abm_count) if abm_count > 0 else 0.0

return discrepancy
```

---

### IntegratedModel._apply_correction

**Сигнатура:**
```python
def _apply_correction(
    self,
    sde_N: float,
    abm_count: int,
    discrepancy: float,
) -> float
```

**Формула:**
```python
# Коррекция пропорциональна coupling_strength и correction_rate
correction_factor = (
    self._config.coupling_strength
    * self._config.correction_rate
)

# Ограничить коррекцию при большом рассогласовании
if discrepancy > self._config.max_discrepancy:
    correction_factor *= self._config.max_discrepancy / discrepancy

# Применить коррекцию
N_corrected = sde_N + correction_factor * (abm_count - sde_N)

# Граничное условие
return max(0.0, N_corrected)
```

---

## Примеры использования

```python
from src.data.parameter_extraction import ModelParameters
from src.core.sde_model import SDEConfig, TherapyProtocol
from src.core.abm_model import ABMConfig
from src.core.integration import (
    IntegratedModel,
    IntegrationConfig,
    simulate_integrated,
    create_default_integration_config,
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

# Способ 1: Быстрая конфигурация
config = create_default_integration_config(
    t_max_days=30.0,
    sync_interval_hours=1.0,
    mode="bidirectional",
)

trajectory = simulate_integrated(
    initial_params=params,
    integration_config=config,
    random_seed=42,
)

# Способ 2: Детальная конфигурация
sde_config = SDEConfig(r=0.3, K=1e6, t_max=30.0)
abm_config = ABMConfig(
    space_size=(100.0, 100.0),
    t_max=720.0,
    initial_stem_cells=50,
)

config = IntegrationConfig(
    sde_config=sde_config,
    abm_config=abm_config,
    sync_interval=2.0,
    coupling_strength=0.7,
    mode="bidirectional",
)

therapy = TherapyProtocol(
    prp_enabled=True,
    prp_start_time=1.0,
)

model = IntegratedModel(config=config, therapy=therapy, random_seed=42)
trajectory = model.simulate(params)

# Анализ результатов
stats = trajectory.get_statistics()
print(f"Финальная плотность SDE: {stats['final_sde_N']:.0f}")
print(f"Финальное количество ABM: {stats['final_abm_total']:.0f}")
print(f"Среднее рассогласование: {stats['mean_discrepancy']:.3f}")
print(f"Макс. рассогласование: {stats['max_discrepancy']:.3f}")

# Временной ряд рассогласования
times, discrepancies = trajectory.get_discrepancy_timeseries()
```

---

## Согласование временных масштабов

| Параметр | SDE | ABM | Связь |
|----------|-----|-----|-------|
| Единицы времени | Дни | Часы | 1 день = 24 часа |
| dt | 0.01 дня | 0.1 часа | — |
| t_max | 30 дней | 720 часов | 30 × 24 = 720 |
| sync_interval | — | 1-4 часа | — |

---

## Зависимости

- numpy
- dataclasses (stdlib)
- src.core.sde_model
- src.core.abm_model
- src.data.parameter_extraction
