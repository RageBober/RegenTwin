# sde_numerics.py — Численные методы для SDE

## Назначение

Модуль реализует продвинутые численные солверы для 20-переменной SDE системы регенерации тканей:
- Euler-Maruyama (базовый, strong order 0.5)
- Milstein (strong order 1.0, Itô-Taylor expansion)
- IMEX splitting (implicit-explicit для стиффных систем)
- Адаптивный контроль шага (PI-контроллер)
- Stochastic Runge-Kutta SRI2W1 (strong order 1.0, мультимерный шум)

Архитектура: Strategy pattern — каждый солвер реализует протокол `SDESolver`.

Математическое обоснование: Kloeden & Platen (1992), Rößler (2010).
Модель: Doks/RegenTwin_Mathematical_Framework.md §2

---

## SolverType

**Назначение:** Enum для выбора типа численного солвера.

**Значения:**

| Значение | Описание | Strong order |
|----------|----------|-------------|
| EM | Euler-Maruyama | 0.5 |
| MILSTEIN | Milstein | 1.0 |
| IMEX | Implicit-Explicit splitting | зависит от компонент |
| SRK | Stochastic Runge-Kutta SRI2W1 | 1.0 |
| ADAPTIVE | Адаптивный шаг (обёртка) | зависит от base |

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| SolverType.EM.value | "euler_maruyama" |
| SolverType("milstein") | SolverType.MILSTEIN |
| Итерация по SolverType | 5 элементов |

---

## SolverConfig

**Назначение:** Dataclass с параметрами интегрирования для всех солверов.

**Поля:**

| Поле | Тип | Default | Описание |
|------|-----|---------|----------|
| solver_type | SolverType | EM | Тип солвера |
| dt | float | 0.01 | Базовый шаг времени (ч) |
| dt_min | float | 1e-6 | Минимальный шаг |
| dt_max | float | 1.0 | Максимальный шаг |
| tolerance | float | 1e-3 | Допуск ошибки (adaptive) |
| max_steps | int | 100000 | Максимум шагов |
| safety_factor | float | 0.9 | Запас PI-контроллера |
| fd_epsilon | float | 1e-6 | Epsilon для конечных разностей |

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| SolverConfig() | Все поля = default |
| SolverConfig(dt=0.001) | dt == 0.001, остальные = default |
| dt_min > dt_max | Невалидная конфигурация |

**Инварианты:**
- 0 < dt_min ≤ dt ≤ dt_max
- tolerance > 0
- max_steps > 0
- 0 < safety_factor ≤ 1

---

## StepResult

**Назначение:** Результат одного шага интегрирования.

**Поля:**

| Поле | Тип | Default | Описание |
|------|-----|---------|----------|
| new_state | np.ndarray | — | Новое состояние shape (20,) |
| dt_used | float | 0.0 | Фактический шаг |
| error_estimate | float | 0.0 | Оценка локальной ошибки |
| n_function_evals | int | 0 | Число вызовов drift/diffusion |
| rejected | bool | False | Шаг отклонён (adaptive) |

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| StepResult(new_state=np.zeros(20)) | new_state.shape == (20,) |
| StepResult(rejected=True) | rejected == True |

**Инварианты:**
- new_state.shape == (20,)
- dt_used ≥ 0
- error_estimate ≥ 0
- n_function_evals ≥ 0

---

## SDESolver

**Назначение:** Protocol (интерфейс) для всех SDE солверов. Позволяет ExtendedSDEModel работать с любым солвером через Strategy pattern.

### SDESolver.step

**Сигнатура:**
```python
def step(
    self,
    state: np.ndarray,
    drift: np.ndarray,
    diffusion: np.ndarray,
    dt: float,
    dW: np.ndarray,
) -> StepResult
```

**Поведение:**
1. Принять текущее состояние X_n и вычисленные drift μ, diffusion σ
2. Применить схему интегрирования для вычисления X_{n+1}
3. Вернуть StepResult с новым состоянием

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Нулевой drift и diffusion | X_{n+1} == X_n |
| Только drift (σ=0) | Детерминированный Euler |
| isinstance(solver, SDESolver) | True для всех конкретных солверов |

---

### SDESolver.simulate

**Сигнатура:**
```python
def simulate(
    self,
    model: ExtendedSDEModel,
    initial_state: ExtendedSDEState,
    params: ParameterSet,
) -> ExtendedSDETrajectory
```

**Поведение:**
1. Извлечь dt, t_max из params
2. Цикл: compute drift/diffusion → step → apply boundary conditions
3. Собрать траекторию (times, states)
4. Вернуть ExtendedSDETrajectory

---

## EulerMaruyamaSolver

**Назначение:** Базовый солвер Эйлера-Маруямы (strong order 0.5).

**Формула:** `X_{n+1} = X_n + μ(X_n)·Δt + σ(X_n)·ΔW_n`

### EulerMaruyamaSolver.step

**Сигнатура:**
```python
def step(self, state, drift, diffusion, dt, dW) -> StepResult
```

**Поведение:**
1. x_new = state + drift * dt + diffusion * dW
2. Вернуть StepResult(new_state=x_new, dt_used=dt, n_function_evals=1)

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| drift=[1,0,...], σ=0, dt=0.1 | X[0] += 0.1 |
| drift=0, σ=[1,0,...], dW=[0.5,...] | X[0] += 0.5 |
| drift=0, σ=0 | X_{n+1} == X_n |

**Edge cases:**
- Очень большой dt (>1) — может дать нефизичные значения
- Отрицательный drift — может дать X < 0

**Инварианты:**
- dt_used == dt (фиксированный шаг)
- n_function_evals == 1
- rejected == False

---

## MilsteinSolver

**Назначение:** Солвер Милштейна со strong order 1.0.

**Формула:** `X_{n+1} = X_n + μ·Δt + σ·ΔW + 0.5·σ·σ'·(ΔW² - Δt)`

Поправка Милштейна `0.5·σ(X_n)·σ'(X_n)·(ΔW² - Δt)` возникает из Itô-Taylor expansion второго порядка. σ' вычисляется численно через конечные разности.

### MilsteinSolver.step

**Сигнатура:**
```python
def step(self, state, drift, diffusion, dt, dW,
         diffusion_derivative=None) -> StepResult
```

**Поведение:**
1. Если diffusion_derivative is None — вычислить через _compute_diffusion_derivative
2. milstein_correction = 0.5 * diffusion * diffusion_derivative * (dW**2 - dt)
3. x_new = state + drift * dt + diffusion * dW + milstein_correction
4. Вернуть StepResult(new_state=x_new, n_function_evals=2 если вычислялась σ')

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| σ = const (σ'=0) | Совпадает с EM |
| σ = X (σ'=1) | Milstein correction = 0.5·X·(dW²-dt) |
| dW=0 | correction = -0.5·σ·σ'·dt |
| Convergence test: dt→0 | Strong error ~ O(dt^1.0) |

**Edge cases:**
- σ'(X) ≈ 0 → Milstein ≈ EM
- Очень маленький fd_epsilon → числ. нестабильность σ'
- σ = 0 → degenerate SDE, Milstein = детерминист. Euler

**Инварианты:**
- При σ' = 0 результат идентичен EulerMaruyama
- Strong order сходимости = 1.0 (теоретический)
- n_function_evals ∈ {1, 2} (с/без вычисления σ')

### _compute_diffusion_derivative

**Назначение:** Численная аппроксимация σ'(X) через forward difference.

**Сигнатура:**
```python
def _compute_diffusion_derivative(
    self, model, state, eps=None
) -> np.ndarray
```

**Поведение:**
1. Для каждой компоненты i:
   - state_plus = state.copy(); state_plus[i] += eps
   - σ_plus = model._compute_diffusion(state_plus)
   - σ_base = model._compute_diffusion(state)
   - σ'[i] = (σ_plus[i] - σ_base[i]) / eps
2. Вернуть ndarray shape (20,) с покомпонентными производными

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| σ(X) = const | σ' ≈ 0 (до машинной точности) |
| σ(X) = X | σ' ≈ 1.0 |
| σ(X) = X² | σ' ≈ 2X |

**Edge cases:**
- eps слишком маленький (< 1e-10) → потеря точности
- eps слишком большой → грубая аппроксимация
- state[i] = 0 с eps → может дать нефизичное состояние

**Инварианты:**
- Возвращает ndarray shape (20,)
- Точность: O(eps) для forward difference

---

## IMEXSplitter

**Назначение:** Implicit-Explicit splitting для стиффных SDE.

Цитокины (индексы 8–14) — стиффная часть (быстрая деградация γ ≈ 0.1–0.5 h⁻¹).
Клетки (0–7) + ECM (15–17) + auxiliary (18–19) — нестиффная часть.

**Формулы:**
- Fast (implicit): `X_fast^{n+1} = X_fast^n + μ_fast(X^{n+1})·Δt`
- Slow (explicit): `X_slow^{n+1} = X_slow^n + μ_slow(X^n)·Δt + σ_slow(X^n)·ΔW`

### IMEXSplitter.step

**Сигнатура:**
```python
def step(self, state, drift, diffusion, dt, dW) -> StepResult
```

**Поведение:**
1. Разделить state на fast (цитокины) и slow (клетки + ECM)
2. Explicit шаг для slow: X_slow += μ_slow·dt + σ_slow·dW
3. Implicit шаг для fast: решить X_fast = X_fast + μ_fast(X_new)·dt через fixed-point
4. Объединить в полное состояние
5. Вернуть StepResult

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Нестиффная система (γ << 1) | Результат ≈ EM |
| Только цитокины ненулевые | Только implicit шаг |
| Только клетки ненулевые | Только explicit шаг |
| Стиффная система (γ = 10) | IMEX стабилен, EM дивергирует |

**Edge cases:**
- Fixed-point не сходится (max_iter) → логирование + последнее приближение
- Все переменные в fast → чисто implicit
- Все переменные в slow → чисто EM

**Инварианты:**
- len(fast_indices) + len(slow_indices) == 20
- fast_indices ∩ slow_indices == ∅
- Порядок переменных в merge == StateIndex

### _implicit_step

**Сигнатура:**
```python
def _implicit_step(
    self, state_fast, drift_fast, dt, max_iter=10, tol=1e-8
) -> np.ndarray
```

**Поведение:**
1. x = state_fast (начальное приближение)
2. Для k = 1..max_iter:
   - x_new = state_fast + drift_fast(x) * dt
   - Если ||x_new - x|| < tol: break
   - x = x_new
3. Вернуть x_new

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| drift = -γ·X (линейная деградация) | Сходимость за 2-3 итерации |
| dt → 0 | x_new ≈ state_fast |
| γ очень большое | Всё равно стабилен |

### _split_state / _merge_state

**Сигнатура:**
```python
def _split_state(self, state) -> tuple[np.ndarray, np.ndarray]
def _merge_state(self, state_fast, state_slow) -> np.ndarray
```

**Поведение:**
- split: state[fast_indices], state[slow_indices]
- merge: np.zeros(20); result[fast_indices] = state_fast; result[slow_indices] = state_slow

**Инварианты:**
- merge(split(state)) == state (roundtrip)

---

## AdaptiveTimestepper

**Назначение:** Обёртка для адаптивного контроля шага на основе оценки локальной ошибки.

**PI-контроллер (Gustafsson):**
```
dt_new = dt · safety · (tol/error)^(k_I/p) · (error_prev/error)^(k_P/p)
k_I = 0.3, k_P = 0.4, safety = 0.9
```

### AdaptiveTimestepper.step

**Сигнатура:**
```python
def step(self, state, drift, diffusion, dt, dW) -> StepResult
```

**Поведение:**
1. full_step = base_solver.step(state, drift, diffusion, dt, dW)
2. dW1 = dW * sqrt(0.5); dW2 = dW * sqrt(0.5)  (разделение Wiener)
3. half1 = base_solver.step(state, ..., dt/2, dW1)
4. half2 = base_solver.step(half1.new_state, ..., dt/2, dW2)
5. error = _estimate_error(full_step.new_state, half2.new_state)
6. Если error < tol: принять half2, dt_new = _pi_controller(...)
7. Если error ≥ tol: rejected=True, dt_used=0

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Медленная динамика | dt увеличивается |
| Быстрая динамика | dt уменьшается |
| error >> tol | Шаг отклонён (rejected=True) |
| error << tol | dt → dt_max |

**Edge cases:**
- dt_new < dt_min → шаг принимается с dt_min + warning
- dt_new > dt_max → ограничение сверху
- error = 0 → dt = dt_max

**Инварианты:**
- dt_min ≤ dt_used ≤ dt_max
- Если rejected: повторить с меньшим dt

### _estimate_error

**Сигнатура:**
```python
def _estimate_error(self, state_full, state_half, order=1) -> float
```

**Поведение:**
1. diff = state_full - state_half
2. error = np.linalg.norm(diff) / (2^order - 1)

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| state_full == state_half | error == 0 |
| order=1 | делитель = 1 |
| order=2 | делитель = 3 |

### _pi_controller

**Сигнатура:**
```python
def _pi_controller(self, error, tolerance, dt_current, order=1) -> float
```

**Поведение:**
1. k_I = 0.3 / order, k_P = 0.4 / order
2. factor = safety * (tol/error)^k_I * (error_prev/error)^k_P
3. dt_new = dt_current * clip(factor, 0.2, 5.0)
4. Ограничить в [dt_min, dt_max]
5. Обновить error_prev = error

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| error == tol | factor ≈ safety |
| error << tol | factor > 1 (увеличение dt) |
| error >> tol | factor < 1 (уменьшение dt) |

---

## StochasticRungeKutta

**Назначение:** Метод SRI2W1 (Rößler 2010) — strong order 1.0 без вычисления σ'.

Butcher tableau SRI2W1 использует 2 стадии для детерминированной части
и 2 стадии для стохастической, обеспечивая strong order 1.0
для диагонального мультимерного шума.

### StochasticRungeKutta.step

**Сигнатура:**
```python
def step(self, state, drift, diffusion, dt, dW) -> StepResult
```

**Поведение:**
1. Вычислить детерминированные стадии k1, k2
2. Вычислить стохастические стадии с ΔW и вспомогательным ΔZ
3. X_{n+1} = X_n + weighted sum of stages
4. Вернуть StepResult(n_function_evals=4)

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| σ=0 | Совпадает с детерминист. RK2 |
| Convergence test | Strong error ~ O(dt^1.0) |
| Сравнение с Milstein | Близкие результаты для diagonal noise |

**Edge cases:**
- dt очень маленький → 4 function evals дороже EM
- σ constant → SRK ≈ EM + overhead

**Инварианты:**
- n_function_evals == 4
- Strong order 1.0 для диагонального шума

---

## create_solver

**Назначение:** Фабричная функция для создания солвера по конфигурации.

**Сигнатура:**
```python
def create_solver(config: SolverConfig) -> SDESolver
```

**Поведение:**
1. Если config.solver_type == EM → EulerMaruyamaSolver(config)
2. MILSTEIN → MilsteinSolver(config)
3. IMEX → IMEXSplitter(config)
4. SRK → StochasticRungeKutta(config)
5. ADAPTIVE → AdaptiveTimestepper(base_solver=EulerMaruyamaSolver(), config=config)
6. Иначе → ValueError

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| SolverConfig(solver_type=SolverType.EM) | isinstance(result, EulerMaruyamaSolver) |
| SolverConfig(solver_type=SolverType.MILSTEIN) | isinstance(result, MilsteinSolver) |
| Невалидный тип | ValueError |

**Инварианты:**
- Возвращает SDESolver-совместимый объект
- isinstance(create_solver(config), SDESolver) == True
