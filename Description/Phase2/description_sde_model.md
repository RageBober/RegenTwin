# Описание: sde_model.py

## Обзор

Модуль стохастических дифференциальных уравнений (SDE) для моделирования регенерации тканей на макроуровне. Описывает динамику плотности клеток и концентрации цитокинов с учётом терапевтических вмешательств PRP и PEMF.

---

## Теоретическое обоснование

### Основное уравнение Ланжевена

```
dNₜ = [rNₜ(1 - Nₜ/K) + αf(PRP) + βg(PEMF) - δNₜ]dt + σₙNₜdWₜ
dCₜ = [ηNₜ - γCₜ + S_PRP(t)]dt + σ_cCₜdWₜ
```

Где:
- **Nₜ** — плотность клеток (клеток/см²)
- **Cₜ** — концентрация цитокинов (нг/мл)
- **r** — скорость пролиферации (0.1-0.5 day⁻¹)
- **K** — carrying capacity (10⁶-10⁷ клеток/см²)
- **δ** — скорость естественной гибели
- **α, β** — коэффициенты терапевтических эффектов
- **σₙ, σ_c** — волатильность (стохастический шум)
- **dWₜ** — Винеровский процесс

### Функции воздействия терапий

**PRP эффект (экспоненциальное затухание):**
```
f(PRP) = C₀ · e^(-λt) · I_PRP
```
Где C₀ — начальная концентрация, λ — скорость затухания, I_PRP — интенсивность.

**PEMF эффект (сигмоидальный отклик):**
```
g(PEMF) = I_PEMF / (1 + e^(-k(f - f₀)))
```
Где f — частота PEMF, f₀ — оптимальная частота (50-75 Гц), k — крутизна.

### Численный метод Эйлера-Маруямы

```
N_{n+1} = Nₙ + μ_N(Nₙ, Cₙ, tₙ)·Δt + σ_N(Nₙ)·√Δt·ξₙ
C_{n+1} = Cₙ + μ_C(Nₙ, Cₙ, tₙ)·Δt + σ_C(Cₙ)·√Δt·ηₙ
```

Где ξₙ, ηₙ ~ N(0, 1) — независимые стандартные нормальные случайные величины.

---

## Классы

### SDEConfig

**Назначение:** Dataclass с параметрами SDE модели.

**Атрибуты:**

| Атрибут | Тип | По умолчанию | Описание | Диапазон |
|---------|-----|--------------|----------|----------|
| r | float | 0.3 | Скорость пролиферации | 0.1-0.5 day⁻¹ |
| K | float | 1e6 | Carrying capacity | 10⁶-10⁷ cells/cm² |
| delta | float | 0.05 | Скорость гибели | 0.01-0.1 day⁻¹ |
| sigma_n | float | 0.05 | Волатильность N | 0.01-0.1 |
| sigma_c | float | 0.02 | Волатильность C | 0.01-0.05 |
| gamma | float | 0.5 | Деградация цитокинов | 0.1-1.0 day⁻¹ |
| eta | float | 0.001 | Секреция цитокинов | 0.0001-0.01 |
| alpha_prp | float | 0.5 | Коэффициент PRP | 0.1-1.0 |
| beta_pemf | float | 0.1 | Коэффициент PEMF | 0.01-0.5 |
| lambda_prp | float | 0.3 | Затухание PRP | 0.1-0.5 day⁻¹ |
| f0_pemf | float | 50.0 | Оптимальная частота PEMF | 50-75 Гц |
| k_pemf | float | 0.1 | Крутизна сигмоиды PEMF | 0.05-0.2 |
| dt | float | 0.01 | Шаг времени | 0.001-0.1 дни |
| t_max | float | 30.0 | Время симуляции | 7-60 дней |

---

### TherapyProtocol

**Назначение:** Протокол терапевтического вмешательства.

**Атрибуты:**

| Атрибут | Тип | По умолчанию | Описание |
|---------|-----|--------------|----------|
| prp_enabled | bool | False | Включить PRP |
| prp_start_time | float | 0.0 | Начало PRP (дни) |
| prp_duration | float | 7.0 | Длительность PRP (дни) |
| prp_intensity | float | 1.0 | Интенсивность PRP (0-2) |
| prp_initial_concentration | float | 10.0 | Начальная концентрация (нг/мл) |
| pemf_enabled | bool | False | Включить PEMF |
| pemf_start_time | float | 0.0 | Начало PEMF (дни) |
| pemf_duration | float | 14.0 | Длительность PEMF (дни) |
| pemf_frequency | float | 50.0 | Частота PEMF (Гц) |
| pemf_intensity | float | 1.0 | Интенсивность PEMF (0-2) |
| synergy_factor | float | 1.2 | Синергия при комбинации |

---

### SDEState

**Назначение:** Состояние системы в момент времени.

**Атрибуты:**

| Атрибут | Тип | Описание |
|---------|-----|----------|
| t | float | Время (дни) |
| N | float | Плотность клеток (клеток/см²) |
| C | float | Концентрация цитокинов (нг/мл) |
| prp_active | bool | PRP терапия активна |
| pemf_active | bool | PEMF терапия активна |

---

### SDETrajectory

**Назначение:** Результат SDE симуляции.

**Атрибуты:**

| Атрибут | Тип | Описание |
|---------|-----|----------|
| times | np.ndarray | Временные точки [n_steps] |
| N_values | np.ndarray | Плотность клеток [n_steps] |
| C_values | np.ndarray | Цитокины [n_steps] |
| therapy_markers | dict | Boolean маски терапий |
| config | SDEConfig | Использованная конфигурация |
| initial_state | SDEState | Начальное состояние |

---

### SDEModel

**Назначение:** Основной класс SDE модели.

---

## Методы

### SDEModel.__init__

**Сигнатура:**
```python
def __init__(
    self,
    config: SDEConfig | None = None,
    therapy: TherapyProtocol | None = None,
    random_seed: int | None = None,
) -> None
```

**Алгоритм:**
1. Инициализировать конфигурацию (или создать по умолчанию)
2. Валидировать параметры
3. Сохранить протокол терапии
4. Инициализировать генератор случайных чисел

---

### SDEModel.simulate

**Сигнатура:**
```python
def simulate(self, initial_params: ModelParameters) -> SDETrajectory
```

**Алгоритм:**
```python
# 1. Инициализация
n_steps = int(self._config.t_max / self._config.dt)
times = np.linspace(0, self._config.t_max, n_steps + 1)
N = np.zeros(n_steps + 1)
C = np.zeros(n_steps + 1)

# Начальные условия из ModelParameters
N[0] = initial_params.n0
C[0] = initial_params.c0

# 2. Основной цикл Эйлера-Маруямы
for i in range(n_steps):
    t = times[i]

    # Drift и diffusion
    drift_N, drift_C = self._calculate_drift(t, N[i], C[i])
    diff_N, diff_C = self._calculate_diffusion(t, N[i], C[i])

    # Случайные приращения
    dW_N = self._rng.standard_normal() * np.sqrt(self._config.dt)
    dW_C = self._rng.standard_normal() * np.sqrt(self._config.dt)

    # Эйлер-Маруяма шаг
    N[i+1] = N[i] + drift_N * self._config.dt + diff_N * dW_N
    C[i+1] = C[i] + drift_C * self._config.dt + diff_C * dW_C

    # Граничные условия
    N[i+1], C[i+1] = self._apply_boundary_conditions(N[i+1], C[i+1])

# 3. Создать траекторию
return SDETrajectory(
    times=times,
    N_values=N,
    C_values=C,
    therapy_markers={
        "prp": self._get_therapy_mask(times, "prp"),
        "pemf": self._get_therapy_mask(times, "pemf"),
    },
    config=self._config,
    initial_state=SDEState(t=0, N=N[0], C=C[0]),
)
```

---

### SDEModel._calculate_drift

**Сигнатура:**
```python
def _calculate_drift(self, t: float, N: float, C: float) -> tuple[float, float]
```

**Формула:**
```python
# Drift для N
drift_N = (
    self._logistic_growth(N)
    + self._prp_effect(t, N, C)
    + self._pemf_effect(t, N)
    - self._config.delta * N
)

# Drift для C
drift_C = (
    self._config.eta * N
    - self._config.gamma * C
    + self._therapy_prp_secretion(t)
)

return (drift_N, drift_C)
```

---

### SDEModel._logistic_growth

**Сигнатура:**
```python
def _logistic_growth(self, N: float) -> float
```

**Формула:**
```python
return self._config.r * N * (1 - N / self._config.K)
```

---

### SDEModel._prp_effect

**Сигнатура:**
```python
def _prp_effect(self, t: float, N: float, C: float) -> float
```

**Алгоритм:**
```python
if not self._is_therapy_active(t, "prp"):
    return 0.0

# Время с начала терапии
t_therapy = t - self._therapy.prp_start_time

# Экспоненциальное затухание
effect = (
    self._config.alpha_prp
    * self._therapy.prp_initial_concentration
    * np.exp(-self._config.lambda_prp * t_therapy)
    * self._therapy.prp_intensity
)

# Синергия с PEMF
if self._is_therapy_active(t, "pemf"):
    effect *= self._therapy.synergy_factor

return effect
```

---

### SDEModel._pemf_effect

**Сигнатура:**
```python
def _pemf_effect(self, t: float, N: float) -> float
```

**Алгоритм:**
```python
if not self._is_therapy_active(t, "pemf"):
    return 0.0

# Сигмоидальный отклик на частоту
freq_diff = self._therapy.pemf_frequency - self._config.f0_pemf
sigmoid = 1 / (1 + np.exp(-self._config.k_pemf * freq_diff))

effect = (
    self._config.beta_pemf
    * sigmoid
    * self._therapy.pemf_intensity
    * N  # Пропорционально количеству клеток
)

return effect
```

---

## Примеры использования

```python
from src.data.parameter_extraction import ModelParameters
from src.core.sde_model import (
    SDEModel,
    SDEConfig,
    TherapyProtocol,
    simulate_sde,
)

# Начальные параметры (из flow cytometry)
params = ModelParameters(
    n0=5000.0,
    c0=10.0,
    stem_cell_fraction=0.05,
    macrophage_fraction=0.03,
    apoptotic_fraction=0.02,
    inflammation_level=0.3,
)

# Конфигурация модели
config = SDEConfig(
    r=0.3,
    K=1e6,
    dt=0.01,
    t_max=30.0,
)

# Протокол PRP терапии
therapy = TherapyProtocol(
    prp_enabled=True,
    prp_start_time=1.0,
    prp_duration=7.0,
    prp_intensity=1.5,
)

# Способ 1: через класс
model = SDEModel(config=config, therapy=therapy, random_seed=42)
trajectory = model.simulate(params)

print(f"Финальная плотность: {trajectory.N_values[-1]:.0f} клеток/см²")
print(f"Финальная концентрация: {trajectory.C_values[-1]:.2f} нг/мл")

# Способ 2: convenience функция
trajectory = simulate_sde(
    initial_params=params,
    config=config,
    therapy=therapy,
    random_seed=42,
)

# Получить статистику
stats = trajectory.get_statistics()
print(f"Максимальная плотность: {stats['max_N']:.0f}")
print(f"Скорость роста: {stats['growth_rate']:.3f}")
```

---

## Зависимости

- numpy
- dataclasses (stdlib)
- src.data.parameter_extraction (ModelParameters)

---

## Параметры из литературы

| Параметр | Значение | Источник |
|----------|----------|----------|
| r | 0.1-0.5 day⁻¹ | Cell proliferation studies |
| K | 10⁶-10⁷ cells/cm² | Tissue density limits |
| σ | 0.01-0.1 | Stochastic modeling |
| λ_PRP | 0.1-0.5 day⁻¹ | PRP half-life ~2-7 days |
| f₀_PEMF | 50-75 Hz | Optimal PEMF frequency |
