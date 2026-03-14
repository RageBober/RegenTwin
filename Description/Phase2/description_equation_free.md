# equation_free.py — Equation-Free Framework: мультимасштабная интеграция SDE↔ABM

## Назначение

Equation-Free (EF) Framework для связи расширенной 20-переменной SDE (`extended_sde.py`)
и агентной модели (`abm_model.py`) через принцип "lift → micro-simulate → restrict".

Используется в Фазе 2.9 проекта RegenTwin для:
- Исследования микроскопических эффектов (пространственная неоднородность, хемотаксис)
  при сохранении макроскопических переменных (20-вектор цитокинов/клеток/ECM)
- Subcycling интеграции: разные `dt` для быстрых (цитокины) и медленных (ECM) процессов
- Применения терапий PRP/PEMF одновременно на обоих уровнях

Подробное описание: Description/Phase2/description_equation_free.md

---

## EquationFreeConfig

**Назначение:** Единый dataclass конфигурации EF-интегратора.

**Сигнатура:**
```python
@dataclass
class EquationFreeConfig:
    dt_macro: float = 1.0
    dt_micro: float = 0.1
    n_micro_steps: int = 10
    volume: float = 1e6
    n_agents_scale: float = 1e-3
```

**Поля:**

| Поле | Тип | Default | Единицы | Описание |
|------|-----|---------|---------|----------|
| `dt_macro` | float | 1.0 | ч | Шаг макроскопического (SDE) времени |
| `dt_micro` | float | 0.1 | ч | Шаг микроскопического (ABM) времени |
| `n_micro_steps` | int | 10 | — | Количество ABM шагов на один EF-шаг |
| `volume` | float | 1e6 | мкм³ | Объём расчётной области (для нормировки) |
| `n_agents_scale` | float | 1e-3 | агентов/клетку | Масштаб конвертации концентрации→число агентов |

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| `EquationFreeConfig()` | все дефолтные значения |
| `dt_macro < dt_micro` | не вызывает ошибку при создании (проверка в Integrator) |
| `n_micro_steps = 0` | `ValueError` при создании (в `__post_init__`) |
| `dt_macro <= 0` | `ValueError` при создании (в `__post_init__`) |
| `dt_micro <= 0` | `ValueError` при создании (в `__post_init__`) |
| `volume <= 0` | `ValueError` при создании (в `__post_init__`) |
| `n_agents_scale <= 0` | `ValueError` при создании (в `__post_init__`) |

**Инварианты (валидация в `__post_init__`):**
- `dt_macro > 0`, `dt_micro > 0`, `volume > 0`, `n_agents_scale > 0`
- `n_micro_steps >= 1`
- При нарушении — `ValueError` с описанием поля

---

## Lifter

**Назначение:** Macro→micro lifting — распределение агентов ABM согласно
макроскопическому состоянию `ExtendedSDEState`. Реализует первый шаг EF-цикла.

### `__init__(config, abm_config)`

**Сигнатура:**
```python
def __init__(self, config: EquationFreeConfig, abm_config: ABMConfig) -> None
```

**Поведение:**
1. Сохраняет `config` и `abm_config` как атрибуты
2. Инициализирует `rng` (numpy random Generator) для воспроизводимости
3. Строит маппинг: тип агента → класс агента (StemCell, Macrophage, ...)

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| `Lifter(cfg, abm_cfg)` | создаётся без ошибок |
| `lifter.config` | равен переданному `cfg` |
| `lifter.abm_config` | равен переданному `abm_cfg` |

---

### `lift(macro_state, n_agents_hint, volume)`

**Сигнатура:**
```python
def lift(
    self,
    macro_state: ExtendedSDEState,
    n_agents_hint: int,
    volume: float
) -> list[Agent]
```

**Поведение:**
1. Для каждого клеточного типа в `macro_state` (P, Ne, M1, M2, F, Mf, E, S):
   - Вычисляет `n_i = round(concentration_i * volume * n_agents_scale)`
   - Вызывает `distribute_population(...)` для создания агентов
2. Вызывает `assign_cytokine_fields(agents, cytokine_levels)` для назначения окружения
3. Возвращает полный список всех созданных агентов

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Нулевое состояние | возвращает `[]` (0 агентов) |
| `P=100`, остальные=0 | возвращает только Platelet-агентов |
| Сумма всех агентов | пропорциональна суммарной концентрации |
| Все концентрации > 0 | агенты всех 8 типов присутствуют |

**Инварианты:**
- `len(result) >= 0`
- Тип каждого агента соответствует исходной концентрации
- Пространственное распределение: равномерное случайное в `[0, space_size]`

---

### `distribute_population(population, agent_class, n_agents, space_size, rng)`

**Сигнатура:**
```python
def distribute_population(
    self,
    population: float,
    agent_class: type,
    n_agents: int,
    space_size: tuple[float, float],
    rng: np.random.Generator
) -> list[Agent]
```

**Поведение:**
1. Если `n_agents == 0`: возвращает `[]`
2. Генерирует `n_agents` случайных позиций `(x, y)` в `[0, space_size[0]] × [0, space_size[1]]`
3. Создаёт агентов: `agent_class(agent_id=..., x=x_i, y=y_i, rng=rng)`
4. Возвращает список агентов длиной `n_agents`

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| `n_agents=0` | `[]` |
| `n_agents=10` | список длиной 10 |
| Все агенты имеют тип `agent_class` | `isinstance(a, agent_class)` для всех |
| Позиции в пределах space_size | `0 <= x <= space_size[0]` |

**Инварианты:**
- `len(result) == n_agents`
- Все `agent_id` уникальны

---

### `assign_cytokine_fields(agents, cytokine_levels)`

**Сигнатура:**
```python
def assign_cytokine_fields(
    self,
    agents: list[Agent],
    cytokine_levels: dict[str, float]
) -> None
```

**Поведение:**
1. Для каждого агента устанавливает атрибут `cytokine_environment` = `cytokine_levels`
2. Если агент не поддерживает атрибут — пропускает (no-op)
3. Мутирует агентов in-place, возвращает `None`

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Пустой список агентов | no-op, нет ошибок |
| Агент с атрибутом | `agent.cytokine_environment == cytokine_levels` |
| `cytokine_levels={}` | агентам назначается пустой словарь |
| Агент без поддержки атрибута | пропускается без ошибки (try/except AttributeError) |

---

## Restrictor

**Назначение:** Micro→macro restriction — агрегация микроскопического состояния ABM
в 20-мерный вектор `ExtendedSDEState`. Реализует последний шаг EF-цикла.
Формула: `X_macro = Σ(agent_states) / volume`

### `__init__(config)`

**Сигнатура:**
```python
def __init__(self, config: EquationFreeConfig) -> None
```

**Поведение:**
1. Сохраняет `config`
2. Инициализирует маппинг: `AGENT_TYPE` → `StateIndex` переменная

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| `Restrictor(cfg)` | создаётся без ошибок |
| `restrictor.config` | равен переданному `cfg` |

---

### `restrict(agents, volume, t)`

**Сигнатура:**
```python
def restrict(
    self,
    agents: list[Agent],
    volume: float,
    t: float
) -> ExtendedSDEState
```

**Поведение:**
1. Вызывает `count_population(agents, type)` для каждого из 8 клеточных типов
2. Вычисляет концентрации: `conc = count / volume`
3. Вызывает `aggregate_cytokines(agents)` для цитокиновых уровней
4. Собирает ECM поля: агрегирует из ABM полей или оставляет 0.0
5. Создаёт и возвращает `ExtendedSDEState(**populations, **cytokines, t=t)`

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Пустой список агентов | все концентрации = 0.0, `state.t == t` |
| 100 Fibroblast-агентов, volume=1e6 | `state.F ≈ 100/1e6` |
| Тип возврата | `isinstance(result, ExtendedSDEState)` |
| `state.t == t` | всегда |

**Инварианты:**
- Все поля результата `>= 0.0`
- `result.t == t`
- Сумма клеточных концентраций соответствует числу агентов / volume

---

### `count_population(agents, agent_type)`

**Сигнатура:**
```python
def count_population(
    self,
    agents: list[Agent],
    agent_type: str
) -> float
```

**Поведение:**
1. Считает агентов, у которых `agent.AGENT_TYPE == agent_type` И `agent.alive == True`
2. Возвращает `float` (для дальнейшего деления на volume)

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Пустой список | `0.0` |
| 5 живых + 3 мёртвых того же типа | `5.0` |
| Тип не представлен | `0.0` |
| Смешанные типы | только совпадающий тип считается |

---

### `aggregate_cytokines(agents)`

**Сигнатура:**
```python
def aggregate_cytokines(
    self,
    agents: list[Agent]
) -> dict[str, float]
```

**Поведение:**
1. Собирает `cytokine_environment` из каждого агента (если атрибут существует)
2. Вычисляет среднее по всем агентам для каждого цитокина
3. Если у агентов нет цитокинового окружения — возвращает нули для всех 7 цитокинов

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Пустой список | `{cytokine: 0.0 for cytokine in CYTOKINE_NAMES}` |
| Агенты без `cytokine_environment` | нули для всех |
| Агенты с однородным полем | среднее == значение поля |

---

## EquationFreeIntegrator

**Назначение:** Главный EF-интегратор, координирующий цикл lift→ABM→restrict.
Связывает `ExtendedSDEModel` (макро) с `ABMModel` (микро) через `Lifter` и `Restrictor`.

### `__init__(sde_model, abm_model, lifter, restrictor, config)`

**Сигнатура:**
```python
def __init__(
    self,
    sde_model: ExtendedSDEModel,
    abm_model: ABMModel,
    lifter: Lifter,
    restrictor: Restrictor,
    config: EquationFreeConfig
) -> None
```

**Поведение:**
1. Сохраняет все компоненты как атрибуты
2. Валидирует конфигурацию: `config.n_micro_steps >= 1`
3. Инициализирует историю `trajectory: list[ExtendedSDEState] = []`

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Корректная конфигурация | создаётся без ошибок |
| `n_micro_steps=0` | `ValueError` |

---

### `step(macro_state, t, dt)`

**Сигнатура:**
```python
def step(
    self,
    macro_state: ExtendedSDEState,
    t: float,
    dt: float
) -> ExtendedSDEState
```

**Поведение (EF-цикл):**
1. `agents = self._lift_step(macro_state, t)` — lifting
2. `agents = self._micro_step(agents, dt)` — `n_micro_steps` шагов ABM
3. `new_macro = self._restrict_step(agents, t + dt)` — restricting
4. Добавляет `new_macro` в `self.trajectory`
5. Возвращает `new_macro`

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Нулевое состояние | возвращает `ExtendedSDEState` с `t == t + dt` |
| Нормальное состояние | тип возврата `ExtendedSDEState` |
| После n шагов | `len(self.trajectory) == n` |

**Инварианты:**
- `result.t == t + dt`
- Все концентрации `>= 0.0`

---

### `_lift_step(macro_state, t)`

**Сигнатура:**
```python
def _lift_step(
    self,
    macro_state: ExtendedSDEState,
    t: float
) -> list[Agent]
```

**Поведение:**
1. Вычисляет `n_agents_hint` из общей клеточной концентрации и `config.n_agents_scale`
2. Делегирует вызов `self.lifter.lift(macro_state, n_agents_hint, config.volume)`
3. Возвращает список агентов

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Нулевое состояние | `[]` |
| Ненулевые клетки | список агентов `len > 0` |

---

### `_micro_step(agents, dt_micro)`

**Сигнатура:**
```python
def _micro_step(
    self,
    agents: list[Agent],
    dt_micro: float
) -> list[Agent]
```

**Поведение:**
1. Выполняет `config.n_micro_steps` итераций ABM:
   - Для каждого шага: `agent.update(dt_micro, environment)` для живых агентов
   - Удаляет умерших (`alive == False`) после каждого шага
   - Проверяет `agent.dividing == True`: если да, вызывает `agent.divide(new_id)` и добавляет дочернего агента в список
2. Возвращает финальный список живых агентов

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Пустой список | возвращает `[]` |
| Мёртвые агенты | фильтруются из результата |
| `dt_micro > 0` | нет ошибок для любых агентов |
| Делящийся агент (`dividing=True`) | дочерний агент добавлен в результат |

**Инварианты:**
- Все агенты в результате: `agent.alive == True`

---

### `_restrict_step(agents, t)`

**Сигнатура:**
```python
def _restrict_step(
    self,
    agents: list[Agent],
    t: float
) -> ExtendedSDEState
```

**Поведение:**
1. Делегирует `self.restrictor.restrict(agents, config.volume, t)`
2. Возвращает результат

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Пустой список | нулевой `ExtendedSDEState` с `t==t` |
| Нормальные агенты | `ExtendedSDEState` с корректными концентрациями |

---

### `run(t_span, dt_macro, dt_micro)`

**Сигнатура:**
```python
def run(
    self,
    t_span: tuple[float, float],
    dt_macro: float,
    dt_micro: float
) -> list[ExtendedSDEState]
```

**Поведение:**
1. Инициализирует `macro_state = self.sde_model.initial_state` (атрибут `ExtendedSDEState`)
2. Цикл по `t` от `t_span[0]` до `t_span[1]` с шагом `dt_macro`:
   - `macro_state = self.step(macro_state, t, dt_macro)`
3. Возвращает `self.trajectory` (список `ExtendedSDEState`)

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| `t_span=(0, 10)`, `dt_macro=1.0` | `len(result) == 10` |
| Каждый элемент | `isinstance(s, ExtendedSDEState)` |
| Монотонный рост `t` | `result[i+1].t > result[i].t` |

**Инварианты:**
- `len(result) == round((t_span[1] - t_span[0]) / dt_macro)`
- Временная монотонность: `result[i].t < result[i+1].t`

---

### `apply_therapy(agents, macro_state, therapy)`

**Сигнатура:**
```python
def apply_therapy(
    self,
    agents: list[Agent],
    macro_state: ExtendedSDEState,
    therapy: TherapyProtocol
) -> tuple[list[Agent], ExtendedSDEState]
```

**Поведение:**
1. Применяет `therapy` к `macro_state` через `sde_model.apply_therapy_effect(...)`
2. Применяет `therapy` к каждому агенту через `agent.apply_therapy(therapy)` (если поддерживается)
3. Возвращает `(modified_agents, modified_macro_state)`

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Нулевые агенты | `macro_state` модифицирован, агенты без изменений |
| PRP терапия | увеличивает P, PDGF, VEGF в `macro_state` |
| PEMF терапия | модифицирует M2/F в `macro_state` |
| Делегирование к `sde_model` | `sde_model.apply_therapy_effect()` вызван |
| Делегирование к агентам | `agent.apply_therapy(therapy)` вызван для каждого агента |

**Инварианты:**
- Возврат: `tuple[list[Agent], ExtendedSDEState]`
- Исходные объекты не мутируются (возвращаются копии)
