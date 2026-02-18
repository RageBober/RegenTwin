# numerical_utils.py — Утилиты для численной робастности

## Назначение

Модуль обеспечивает робастность стохастических вычислений:
- Отсечение нефизичных значений (отрицательные концентрации)
- Детекцию дивергенции (NaN, Inf, overflow) с fallback-стратегией
- Адаптивный шаг времени
- Контекстный менеджер для безопасных вычислений
- Структурированное логирование через Loguru

Используется в SDE/ABM интеграторах для предотвращения численной нестабильности.

Подробное описание: Description/Phase2/description_numerical_utils.md

---

## DivergenceInfo

**Назначение:** Dataclass с диагностикой дивергенции в численном решении.

**Поля:**

| Поле | Тип | Default | Описание |
|------|-----|---------|----------|
| has_nan | bool | False | Есть NaN |
| has_inf | bool | False | Есть Inf |
| nan_variables | list[str] | [] | Имена NaN-переменных |
| inf_variables | list[str] | [] | Имена Inf-переменных |
| max_value | float | 0.0 | Максимальное |value| |
| message | str | "" | Диагностическое сообщение |

**Свойства:**

| Свойство | Тип | Описание |
|----------|-----|----------|
| is_diverged | bool | True если has_nan or has_inf |

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| DivergenceInfo() | is_diverged == False |
| DivergenceInfo(has_nan=True) | is_diverged == True |
| DivergenceInfo(has_inf=True) | is_diverged == True |
| DivergenceInfo(has_nan=True, has_inf=True) | is_diverged == True |
| DivergenceInfo(nan_variables=["N"]) | has_nan should be True (если задано) |

**Инварианты:**
- is_diverged == (has_nan or has_inf)
- len(nan_variables) > 0 → has_nan == True (при корректном использовании)

---

## clip_negative_concentrations

**Назначение:** Отсечение отрицательных значений для физичности. Концентрации не могут быть < 0.

**Сигнатура:**

```python
def clip_negative_concentrations(
    state: dict[str, float],
    variables: list[str] | None = None,
    min_value: float = 0.0,
) -> dict[str, float]
```

**Поведение:**
1. Если variables=None — обработать все ключи state
2. Для каждой переменной: state[var] = max(state[var], min_value)
3. Вернуть новый dict (не мутировать оригинал)

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| {"N": 100, "C": -5} | {"N": 100, "C": 0.0} |
| Все положительные | Без изменений |
| Все отрицательные | Все = min_value |
| variables=["C"], {"N": -1, "C": -5} | {"N": -1, "C": 0.0} |
| Пустой dict {} | Пустой dict {} |
| min_value=1.0, {"N": 0.5} | {"N": 1.0} |

**Edge cases:**
- state с NaN → NaN остаётся (clip не обрабатывает NaN)
- min_value отрицательный → допустимо (пользователь задаёт порог)

**Инварианты:**
- Возвращает НОВЫЙ dict (не мутирует вход)
- result[var] ≥ min_value для всех обработанных variables
- len(result) == len(state)

---

## detect_divergence

**Назначение:** Детекция NaN, Inf и аномально больших значений. Логирует предупреждения через Loguru при обнаружении проблем.

**Сигнатура:**

```python
def detect_divergence(
    state: dict[str, float],
    max_allowed: float = 1e15,
) -> DivergenceInfo
```

**Поведение:**
1. Для каждой переменной:
   - np.isnan(value) → has_nan=True, добавить в nan_variables
   - np.isinf(value) → has_inf=True, добавить в inf_variables
   - |value| > max_allowed → is_diverged через max_value
2. max_value = max(|value| для всех переменных)
3. Сформировать диагностическое message
4. При обнаружении дивергенции — логировать через `logger.warning()`:
   - NaN: `"Divergence detected: NaN in variables {nan_variables}"`
   - Inf: `"Divergence detected: Inf in variables {inf_variables}"`
   - Overflow: `"Divergence detected: max_value={max_value:.2e} exceeds {max_allowed:.2e}"`

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| {"N": 100, "C": 5.0} | is_diverged=False |
| {"N": float("nan")} | has_nan=True, nan_variables=["N"] |
| {"N": float("inf")} | has_inf=True, inf_variables=["N"] |
| {"N": float("-inf")} | has_inf=True |
| {"N": 1e20} при max_allowed=1e15 | is_diverged=True, max_value=1e20 |
| Пустой dict {} | is_diverged=False |
| {"A": nan, "B": inf, "C": 1.0} | has_nan=True, has_inf=True |

**Edge cases:**
- max_allowed=0 → любое ненулевое значение считается дивергенцией
- Отрицательные значения → проверка по abs(value)

**Инварианты:**
- is_diverged == (has_nan or has_inf) (через property)
- Все имена в nan_variables действительно содержат NaN
- max_value ≥ 0

---

## handle_divergence

**Назначение:** Стратегия реагирования на обнаруженную дивергенцию. Выполняет остановку шага + fallback (откат к предыдущему состоянию с уменьшенным dt).

**Сигнатура:**

```python
def handle_divergence(
    divergence_info: DivergenceInfo,
    state_current: dict[str, float],
    state_previous: dict[str, float],
    dt_current: float,
    dt_min: float = 1e-6,
    max_retries: int = 3,
) -> tuple[dict[str, float], float, bool]
```

**Возвращает:** (safe_state, new_dt, should_stop)
- safe_state — откат к state_previous (или клиппированное state_current если дивергенция мягкая)
- new_dt — уменьшенный шаг (dt_current / 2)
- should_stop — True если dt < dt_min или превышены max_retries

**Алгоритм:**
```python
# 1. Логировать через Loguru
logger.warning(
    "Divergence fallback: {message}, dt={dt:.2e} → {new_dt:.2e}",
    message=divergence_info.message,
    dt=dt_current,
    new_dt=dt_current / 2,
)

# 2. Определить стратегию
if divergence_info.has_nan or divergence_info.has_inf:
    # Жёсткая дивергенция → откат к предыдущему состоянию
    safe_state = dict(state_previous)
    new_dt = dt_current / 2
else:
    # Мягкая дивергенция (overflow) → клиппинг + уменьшение dt
    safe_state = clip_negative_concentrations(state_current)
    new_dt = dt_current / 2

# 3. Проверить порог остановки
should_stop = new_dt < dt_min

# 4. Ограничить dt
new_dt = max(new_dt, dt_min)

return (safe_state, new_dt, should_stop)
```

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| NaN дивергенция | safe_state == state_previous, dt = dt/2 |
| Inf дивергенция | safe_state == state_previous, dt = dt/2 |
| Overflow (max_value > max_allowed) | safe_state клиппирован, dt = dt/2 |
| dt_current / 2 < dt_min | should_stop = True, dt = dt_min |
| Нет дивергенции (is_diverged=False) | state_current без изменений, dt без изменений |

**Edge cases:**
- state_previous пустой → ValueError
- dt_current == dt_min → dt остаётся dt_min, should_stop=True при следующей дивергенции
- max_retries=0 → should_stop сразу True

**Инварианты:**
- safe_state не содержит NaN/Inf
- new_dt ≥ dt_min
- should_stop == True → дальнейшая симуляция невозможна без изменения параметров

---

## adaptive_timestep

**Назначение:** Адаптивный шаг времени на основе скорости изменения состояния.

**Сигнатура:**

```python
def adaptive_timestep(
    state_current: dict[str, float],
    state_previous: dict[str, float],
    dt_current: float,
    tolerance: float = 0.1,
    dt_min: float = 1e-6,
    dt_max: float = 1.0,
) -> float
```

**Алгоритм:**
1. max_relative_change = max(|x_new - x_old| / max(|x_old|, eps))
2. Если change > tolerance → dt_new = dt_current × tolerance / change
3. Если change < tolerance/4 → dt_new = dt_current × 2.0
4. Иначе → dt_new = dt_current
5. Ограничить: dt_min ≤ dt_new ≤ dt_max

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Малое изменение (change < tol/4) | dt увеличивается (× 2) |
| Большое изменение (change > tol) | dt уменьшается |
| state_current == state_previous | dt = dt_max (нулевое изменение) |
| tolerance=0.01 | Более агрессивное уменьшение |
| Результат < dt_min | Возвращает dt_min |
| Результат > dt_max | Возвращает dt_max |

**Edge cases:**
- Пустые dict → dt_max (нет переменных для проверки)
- state_previous содержит 0 → eps предотвращает деление на 0
- Одинаковые ключи → корректное сравнение

**Инварианты:**
- dt_min ≤ result ≤ dt_max
- Монотонность: больше изменение → меньше dt

---

## NumericalGuard

**Назначение:** Контекстный менеджер для безопасных численных вычислений. Перехватывает numpy warnings (overflow, underflow, invalid). Логирует предупреждения через Loguru.

**Сигнатуры:**

```python
class NumericalGuard:
    def __init__(self, clip_on_overflow: bool = True, log_warnings: bool = True) -> None
    def __enter__(self) -> "NumericalGuard"
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool
    @property
    def had_warnings(self) -> bool
    @property
    def warnings(self) -> list[str]
```

**Поведение:**
1. `__enter__`: сохранить np.seterr settings, установить фильтры warnings
2. Внутри блока: перехватывать RuntimeWarning (overflow, underflow, invalid)
3. `__exit__`: восстановить settings, собрать warnings в список
4. `had_warnings`: True если были предупреждения
5. `warnings`: список текстовых описаний
6. При `log_warnings=True` — каждое предупреждение логируется через `logger.warning()`:
   - `"NumericalGuard: {warning_type} in computation: {message}"`

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| `with NumericalGuard(): 1+1` | had_warnings=False, warnings=[] |
| `with NumericalGuard(): np.log(-1)` | had_warnings=True |
| `with NumericalGuard(): np.float64(1e308)*2` | had_warnings=True (overflow) |
| Без контекста: guard.had_warnings | False |
| guard.warnings после ошибки | list[str], len > 0 |
| `log_warnings=True` + overflow | logger.warning вызван 1 раз |
| `log_warnings=False` + overflow | logger.warning не вызван |

**Edge cases:**
- Вложенные NumericalGuard → каждый восстанавливает свои settings
- Exception внутри блока → settings всё равно восстанавливаются
- clip_on_overflow=False → не отсекает, только логирует

**Инварианты:**
- numpy settings восстанавливаются даже при exception
- had_warnings == (len(warnings) > 0)
- warnings содержит только str

---

## Loguru-логирование в core-модулях

**Назначение:** Структурированное логирование численных предупреждений и диагностики через Loguru. Все core-модули (sde_model, abm_model, integration, monte_carlo, numerical_utils) используют единый паттерн логирования.

**Инициализация (в каждом модуле):**

```python
from loguru import logger
```

**Уровни логирования:**

| Уровень | Когда используется | Пример |
|---------|-------------------|--------|
| `logger.debug()` | Шаг интегрирования, промежуточные значения | `"SDE step t={t:.4f}, N={N:.2f}, C={C:.4f}"` |
| `logger.info()` | Начало/конец симуляции, milestone | `"Simulation started: t_max={t_max}, dt={dt}"` |
| `logger.warning()` | Отрицательные значения, клиппинг, мягкая дивергенция | `"Clipped {n} negative values: {variables}"` |
| `logger.error()` | Жёсткая дивергенция (NaN/Inf), остановка | `"Divergence: NaN in {variables}, stopping"` |

**Точки логирования в numerical_utils:**

| Функция | Событие | Уровень | Сообщение |
|---------|---------|---------|-----------|
| `clip_negative_concentrations` | Клиппинг произошёл | warning | `"Clipped {n} negative values in {variables}: min was {min_val:.2e}"` |
| `clip_negative_concentrations` | Нет клиппинга | debug | не логирует (нормальный режим) |
| `detect_divergence` | NaN обнаружен | warning | `"Divergence: NaN in {nan_variables}"` |
| `detect_divergence` | Inf обнаружен | warning | `"Divergence: Inf in {inf_variables}"` |
| `detect_divergence` | Overflow | warning | `"Divergence: max_value={max_value:.2e} exceeds {max_allowed:.2e}"` |
| `detect_divergence` | Всё в норме | — | не логирует |
| `handle_divergence` | Fallback | warning | `"Fallback: {message}, dt {dt:.2e} → {new_dt:.2e}"` |
| `handle_divergence` | Остановка | error | `"Stopping: dt={dt:.2e} < dt_min={dt_min:.2e}"` |
| `adaptive_timestep` | dt уменьшен | debug | `"Adaptive dt: {dt_old:.2e} → {dt_new:.2e} (change={change:.4f})"` |
| `adaptive_timestep` | dt увеличен | debug | `"Adaptive dt: {dt_old:.2e} → {dt_new:.2e} (stable)"` |
| `NumericalGuard.__exit__` | Предупреждения | warning | `"NumericalGuard: {warning_type}: {message}"` |

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| clip с отрицательными | logger.warning вызван с именами переменных |
| clip без отрицательных | logger.warning не вызван |
| detect_divergence с NaN | logger.warning вызван |
| detect_divergence без проблем | logger не вызван |
| handle_divergence → остановка | logger.error вызван |
| NumericalGuard + overflow + log_warnings=True | logger.warning вызван |
| NumericalGuard + overflow + log_warnings=False | logger.warning не вызван |

**Инварианты:**
- Все сообщения содержат имена затронутых переменных (для диагностики)
- warning — восстанавливаемые ситуации, error — остановка
- debug не используется по умолчанию (уровень INFO)
- Loguru подключается через `from loguru import logger` — без дополнительной конфигурации

---

## Зависимости

- numpy
- loguru
- dataclasses (stdlib)
