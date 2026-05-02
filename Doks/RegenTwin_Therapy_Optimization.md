# Оптимизация терапий (Therapy Optimization) — Контекст и план реализации

## Что это и зачем

Оптимизация терапий — поиск **наилучшей комбинации параметров PRP и PEMF** для конкретного пациента (набора начальных условий). Это переход от вопроса *«что произойдёт при таких параметрах?»* (текущая симуляция) к вопросу *«какие параметры дадут лучший результат?»*.

**Клиническая ценность:** врач загружает данные пациента (.fcs) → система рекомендует оптимальный протокол терапии (дозу PRP, частоту PEMF, тайминг, комбинацию).

**Научная ценность:** автоматический поиск оптимумов в 10+ мерном пространстве параметров, что невозможно вручную.

---

## Что уже есть в проекте (инфраструктура)

Всё необходимое для реализации оптимизатора уже существует:

### Модель (forward model)
- `src/core/extended_sde.py` — 20-переменная SDE, принимает `ParameterSet` + `ExtendedSDEState` → `ExtendedSDETrajectory`
- `src/core/therapy_models.py` — механистические модели PRP (двухфазная кинетика 4 факторов) и PEMF (3 пути)
- `src/core/monte_carlo.py` — ансамблевые симуляции (100-1000 траекторий) для учёта стохастики

### Конфигурации терапий (пространство поиска)
- `PRPConfig` (12 параметров): `dose` (3-5x), `pdgf_c0`, `vegf_c0`, `tgfb_c0`, `egf_c0`, `tau_burst_*`, `tau_sustained_*`, `alpha_PRP_S`
- `PEMFConfig` (10 параметров): `B_amplitude` (мТ), `frequency` (Гц), `f_opt_anti_inflam`, `epsilon_max_anti_inflam`, `f_center_prolif`, `epsilon_prolif_max`, `epsilon_migration_max`
- `SynergyConfig`: `beta_synergy`

### Анализ чувствительности
- `src/core/sensitivity_analysis.py` — Sobol/Morris/Local показывают *какие* параметры влияют больше всего. Результат Sobol используется для сужения пространства поиска перед оптимизацией.

### Существующие зависимости
- `scipy.optimize` — уже в проекте (используется в `parameter_estimation.py` для MLE)
- `SALib` — уже в проекте
- `numpy`, `scipy` — уже в проекте

---

## Пространство оптимизируемых параметров

### Ключевые параметры терапий (оптимизируемые)

Из sensitivity analysis (Sobol) выделяются наиболее влиятельные параметры. Типичный набор для оптимизации:

| Параметр | Диапазон | Единицы | Источник |
|----------|----------|---------|----------|
| `prp_dose` | 1.0 — 8.0 | fold | PRPConfig.dose |
| `prp_application_time` | 0 — 48 | ч | Тайминг инъекции |
| `pemf_frequency` | 10 — 100 | Гц | PEMFConfig.frequency |
| `pemf_B_amplitude` | 0.1 — 5.0 | мТ | PEMFConfig.B_amplitude |
| `pemf_start_time` | 0 — 72 | ч | Когда включать PEMF |
| `pemf_session_duration` | 1 — 8 | ч/день | Длительность сеансов |
| `prp_enabled` | True/False | — | Включён ли PRP |
| `pemf_enabled` | True/False | — | Включён ли PEMF |

Пространство: 6 непрерывных + 2 бинарных = 8 измерений. При grid search 10 точек на параметр = 10^6 вычислений × ~0.5с каждое = непрактично без surrogate.

---

## Целевые функции (objectives)

Оптимизатор минимизирует/максимизирует скалярную целевую функцию, вычисляемую по результатам симуляции. Несколько вариантов:

### Вариант 1: Скорость заживления (простейший)
```
J₁ = -rho_collagen(T_final)
```
Максимизация финальной плотности коллагена — косвенный индикатор качества заживления.

### Вариант 2: Время до порога заживления
```
J₂ = min{t : rho_collagen(t) > threshold AND D(t) < D_threshold}
```
Минимизация времени, за которое коллаген достигает порога И повреждение (DAMPs) снижается. Более клинически релевантно.

### Вариант 3: Комбинированный score (рекомендуемый)
```
J₃ = w₁ · healing_time + w₂ · (1 - final_collagen/K) + w₃ · max_inflammation + w₄ · scar_risk
```
Где:
- `healing_time` = время до D(t) < 0.1 · D(0) — нормализованное
- `final_collagen` = rho_collagen(T) — качество ткани
- `max_inflammation` = max(C_TNF(t)) / C_TNF_threshold — пик воспаления
- `scar_risk` = final Mf(T) / (F(T) + Mf(T)) — доля миофибробластов (маркер рубца)
- `w₁..w₄` — веса, настраиваемые пользователем (по умолчанию равные)

### Вариант 4: M1/M2 переключение
```
J₄ = -∫₀ᵀ M2(t)/(M1(t) + M2(t) + ε) dt
```
Максимизация интеграла M2-доминирования — ключевой индикатор перехода от воспаления к регенерации.

---

## Уровни реализации

### Уровень 1: Grid Search (1-2 дня)

**Подход:** Перебор фиксированной сетки по 2-3 ключевым параметрам.

**Применение:** Быстрая визуализация landscape — heatmap «доза PRP × частота PEMF → healing score».

**Реализация:**
```python
# Псевдокод
results = {}
for dose in np.linspace(1.0, 8.0, 8):
    for freq in np.linspace(10, 100, 10):
        prp_cfg = PRPConfig(dose=dose)
        pemf_cfg = PEMFConfig(frequency=freq)
        traj = run_extended_sde(params, state, prp_cfg, pemf_cfg)
        results[(dose, freq)] = compute_healing_score(traj)
# Heatmap visualisation
```

**Достоинства:** Просто, визуализируемо, не требует convergence.
**Недостатки:** Экспоненциальный рост при >3 параметрах. Не находит точный оптимум.

**Файлы:**
- `src/core/therapy_optimizer.py` — класс `GridSearchOptimizer`
- `src/visualization/optimization_plots.py` — heatmap landscape
- `tests/unit/core/test_therapy_optimizer.py`

---

### Уровень 2: Scipy.optimize (3-5 дней) — РЕКОМЕНДУЕМЫЙ

**Подход:** `scipy.optimize.minimize` (L-BFGS-B или Nelder-Mead) + `scipy.optimize.differential_evolution` для глобального поиска.

**Почему два метода:**
- `differential_evolution` — глобальный, находит область оптимума в многомерном пространстве
- `L-BFGS-B` — локальный, уточняет оптимум внутри найденной области

**Реализация (архитектура):**

```python
@dataclass
class OptimizationConfig:
    """Конфигурация оптимизации терапий."""
    # Какие параметры оптимизировать
    optimize_params: list[TherapyParamBound]
    # Целевая функция
    objective: str = "combined"  # "healing_time", "collagen", "combined", "m2_dominance"
    # Веса для combined
    weights: dict[str, float] = field(default_factory=lambda: {
        "healing_time": 0.3, "collagen": 0.3,
        "inflammation": 0.2, "scar_risk": 0.2,
    })
    # Forward model
    t_max: float = 720.0  # ч (30 дней)
    dt: float = 0.1
    n_trajectories: int = 5  # MC ансамбль для робастности
    # Оптимизатор
    method: str = "differential_evolution"  # или "L-BFGS-B", "Nelder-Mead"
    maxiter: int = 100
    seed: int | None = 42

@dataclass
class TherapyParamBound:
    """Граница одного оптимизируемого параметра."""
    name: str           # "prp_dose", "pemf_frequency", ...
    lower: float
    upper: float
    initial: float | None = None  # Для локальных методов

@dataclass
class OptimizationResult:
    """Результат оптимизации."""
    optimal_params: dict[str, float]    # Найденный оптимум
    optimal_score: float                # Значение целевой функции
    scores_by_component: dict[str, float]  # Разложение по компонентам
    n_evaluations: int                  # Число вызовов forward model
    convergence_history: list[float]    # История J по итерациям
    elapsed_seconds: float
    # Сравнение с baseline (без терапий)
    baseline_score: float
    improvement_percent: float

class TherapyOptimizer:
    """Оптимизатор терапевтических протоколов."""

    def __init__(self, config: OptimizationConfig,
                 params: ParameterSet,
                 initial_state: ExtendedSDEState):
        ...

    def _objective(self, x: np.ndarray) -> float:
        """Целевая функция: параметры → скаляр (чем меньше, тем лучше)."""
        # 1. Распаковать x в PRPConfig/PEMFConfig
        # 2. Запустить MC ансамбль (n_trajectories симуляций)
        # 3. Усреднить healing score по траекториям
        # 4. Вернуть -score (для минимизации)
        ...

    def optimize(self) -> OptimizationResult:
        """Запуск оптимизации."""
        ...

    def compute_healing_score(self, trajectory) -> dict[str, float]:
        """Вычисление компонентов целевой функции по траектории."""
        ...
```

**Файлы:**
- `src/core/therapy_optimizer.py` — `TherapyOptimizer`, `GridSearchOptimizer`, `OptimizationConfig`, `OptimizationResult`
- `src/api/routes/optimize.py` — API endpoint `POST /api/v1/optimize`
- `src/api/services/optimization_service.py` — background task для длительной оптимизации
- `src/visualization/optimization_plots.py` — landscape heatmap, convergence plot, comparison bar chart
- `ui/src/components/Optimization/OptimizationView.tsx` — UI компонент
- `tests/unit/core/test_therapy_optimizer.py`
- `Description/Phase4/description_therapy_optimizer.md`

**Оценка времени вычисления:**
- 1 вызов forward model (extended SDE, 720ч, dt=0.1): ~0.3-1.0 сек
- MC ансамбль (5 траекторий): ~1.5-5.0 сек
- `differential_evolution` (100 итераций × 15 pop_size): ~100-750 вызовов → 2.5-62 мин
- **Итого:** 3-60 минут на одну оптимизацию (приемлемо для background task)

---

### Уровень 3: Optimal Control — Понтрягин (2-4 недели, PhD)

**Подход:** Forward-backward sweep для time-varying control — оптимальный протокол PEMF(t), PRP(t) как функции времени, а не константы.

**Формулировка:**
```
min J = ∫₀ᵀ L(x(t), u(t)) dt + Φ(x(T))

subject to:
    dx/dt = f(x, u) + σ(x) dW   (20-var SDE)
    x(0) = x₀                    (из flow cytometry)
    u(t) ∈ U                     (допустимые терапии)
```

Где:
- `x(t)` — вектор состояния (20 переменных SDE)
- `u(t) = (PRP_dose(t), PEMF_freq(t), PEMF_amplitude(t))` — управление
- `L(x, u)` — текущие издержки (воспаление + стоимость терапии)
- `Φ(x(T))` — терминальные издержки (качество ткани)
- `U` — ограничения (макс. доза PRP, макс. число сеансов PEMF)

**Результат:** Временной профиль терапии — «на 1 день: PRP 4x, PEMF 50Гц; на 3 день: PRP 2x, PEMF 75Гц; ...». Это клинически наиболее ценный формат.

**Почему PhD-уровень:**
- Стохастическое optimal control (SDE, не ODE) — открытая исследовательская задача
- Для 20 переменных — high-dimensional, требует специализированных методов
- Уже есть в `RegenTwin_PhD_Enhancement_Plan.md` как отдельная фаза

---

## Связь с существующими модулями

```
                   Sensitivity Analysis (Sobol)
                          │
                          │ выделяет ключевые параметры
                          ▼
    Initial State ──► TherapyOptimizer ──► OptimizationResult
    (из flow data)        │                      │
                          │ вызывает             │
                          ▼                      ▼
                   ExtendedSDEModel         Visualization
                   + PRPModel               (landscape,
                   + PEMFModel               convergence,
                   + MonteCarloSim           comparison)
```

## Визуализация результатов оптимизации

1. **Landscape heatmap** — 2D карта «param₁ × param₂ → score» с отмеченным оптимумом
2. **Convergence plot** — J(iteration) для оптимизатора
3. **Comparison bar chart** — «без терапии» vs «PRP only» vs «PEMF only» vs «оптимум»
4. **Trajectory overlay** — кривые M1/M2/collagen для baseline vs optimal
5. **Sensitivity tornado** — какие параметры больше всего влияют на score вблизи оптимума

---

## Рекомендуемый порядок реализации

1. **Целевые функции** — `compute_healing_score()` по готовой траектории (1 день)
2. **Grid Search** — 2D heatmap по PRP dose × PEMF frequency (1 день)
3. **Scipy optimizer** — `differential_evolution` + `L-BFGS-B` (2-3 дня)
4. **API endpoint** — `POST /api/v1/optimize` с background task (1 день)
5. **UI компонент** — страница оптимизации (2-3 дня frontend)
6. *(Опционально, PhD)* Optimal control — forward-backward sweep (2-4 недели)

**Итого уровень 2: ~5-7 рабочих дней** для полного цикла backend + API + визуализация.
