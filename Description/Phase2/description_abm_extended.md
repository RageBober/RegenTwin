# Описание: abm_spatial.py — Расширенные механики ABM (Phase 2.8)

## Обзор

Расширенные пространственные и клеточные механики для Agent-Based модели регенерации тканей. Модуль дополняет базовую ABM (`abm_model.py`) новыми типами агентов, движками для хемотаксиса, контактного ингибирования, эффероцитоза и механотрансдукции, а также мультицитокиновыми полями и subcycling-менеджером.

Файл: `src/core/abm_spatial.py`

---

## Теоретическое обоснование

### Хемотаксис (направленное движение)

Biased random walk — суперпозиция диффузии и дрейфа по градиенту:

```
dx = √(2D·dt)·ξ + χ · (∂C/∂x) / |∇C| · dt
dy = √(2D·dt)·η + χ · (∂C/∂y) / |∇C| · dt
```

Где:
- D — коэффициент диффузии агента
- χ — чувствительность к хемоаттрактанту (CHEMOTAXIS_SENSITIVITY)
- C — концентрация хемоаттрактанта
- ξ, η ~ N(0, 1)

Каждый тип агента имеет свой аттрактант:

| Тип агента | Хемоаттрактант | χ (sensitivity) |
|------------|---------------|-----------------|
| Neutrophil | IL-8 | 0.8 |
| Macrophage | MCP-1 | 0.5 |
| Endothelial | VEGF | 0.6 |
| Fibroblast | PDGF | 0.3 |
| Platelet | TGF-β | 0.2 |

### Контактное ингибирование

```
modifier = max(0, 1 - n_neighbors / threshold)
P(division) = p_div · modifier
```

При `n_neighbors ≥ threshold` деление полностью блокируется.

### Эффероцитоз

Макрофаги фагоцитируют апоптотические нейтрофилы → сдвиг поляризации в M2 и выброс IL-10:

```
IL-10_released = n_phagocytosed · il10_release_rate · dt
polarization_shift = -0.1 · n_phagocytosed  (сдвиг к M2)
```

### Механотрансдукция

Механический стресс от ECM и соседних клеток → активация фибробластов в миофибробласты:

```
stress = Σ (F_adhesion + F_contraction) · ecm_density
P(activation) = p_act · sigmoid(stress - stress_threshold)
```

### Subcycling

Цитокиновые поля имеют более быструю динамику (dt_field < dt_agent):

```
n_substeps = ceil(dt_agent / dt_field)
```

---

## PlateletAgent

Тромбоцит — анукленная клетка, формирующая первичный тромб и выделяющая факторы роста при дегрануляции α-гранул.

### Константы класса

| Константа | Тип | Значение | Описание |
|-----------|-----|----------|----------|
| AGENT_TYPE | str | "platelet" | Идентификатор типа |
| LIFESPAN | float | 72.0 | Время жизни (часы, 3 дня) |
| MAX_DIVISIONS | int | 0 | Не пролиферирует |
| DIVISION_PROBABILITY | float | 0.0 | Не делится |
| DEATH_PROBABILITY | float | 0.014 | t₁/₂ ≈ 48ч |
| DIVISION_ENERGY_THRESHOLD | float | 1.0 | Недостижимый порог |
| DEGRANULATION_RATE | float | 0.05 | Скорость дегрануляции (1/час) |
| PDGF_RELEASE_RATE | float | 0.02 | Выброс PDGF (нг/мл/час) |
| TGFB_RELEASE_RATE | float | 0.015 | Выброс TGF-β |
| VEGF_RELEASE_RATE | float | 0.01 | Выброс VEGF |

### Атрибуты экземпляра

| Атрибут | Тип | По умолчанию | Описание |
|---------|-----|--------------|----------|
| degranulated | bool | False | Флаг дегрануляции |
| factors_released | dict[str, float] | {"PDGF": 0, "TGFb": 0, "VEGF": 0} | Кумулятивный выброс |

### Методы

#### `__init__(self, agent_id, x, y, age=0.0, rng=None)`
Конструктор. Вызывает `super().__init__()`, инициализирует `degranulated=False` и `factors_released`.

#### `update(self, dt, environment) -> None` ⬜ STUB
Обновление состояния: потребление энергии, старение, дегрануляция при активации.
- Если `environment.get("thrombin", 0) > 0.1`: запускает дегрануляцию
- Расход энергии: `energy -= 0.005 * dt`

#### `divide(self, new_id) -> None`
Всегда возвращает `None` — тромбоциты не делятся.

#### `degranulate(self, dt: float) -> dict[str, float]` ⬜ STUB
Дегрануляция α-гранул → выброс PDGF, TGF-β, VEGF.

**Возвращает:** `{"PDGF": rate*dt, "TGFb": rate*dt, "VEGF": rate*dt}`

#### `release_factors(self, dt: float) -> dict[str, float]` ⬜ STUB
Постепенный выброс факторов роста после дегрануляции.

#### `secrete_cytokines(self, dt: float) -> dict[str, float]` ⬜ STUB
Аналог `degranulate`, совместимый с интерфейсом ABMModel.

### Тестовые сценарии

| Сценарий | Ожидаемый результат |
|----------|-------------------|
| Создание агента | `degranulated=False`, `factors_released` = нули |
| `can_divide()` | Всегда `False` |
| `divide(new_id)` | Всегда `None` |
| `degranulate(dt)` | `NotImplementedError` |
| `release_factors(dt)` | `NotImplementedError` |
| `secrete_cytokines(dt)` | `NotImplementedError` |
| `update(dt, env)` | `NotImplementedError` |
| `get_state()` | `AgentState` с type="platelet" |
| `should_die()` при age > 72.0 | `True` (старение) |

---

## ChemotaxisEngine

Движок градиентного хемотаксиса — вычисляет смещение агента на основе градиента хемоаттрактанта.

### Константы класса

| Константа | Тип | Значение |
|-----------|-----|----------|
| AGENT_ATTRACTANT_MAP | dict[str, str] | {"neutro": "IL8", "macro": "MCP1", "endo": "VEGF", "fibro": "PDGF", "platelet": "TGFb"} |

### Методы

#### `__init__(self, config: ABMConfig) -> None`
Сохраняет конфигурацию (space_size, grid_resolution, chemotaxis_strength).

#### `compute_displacement(self, agent, cytokine_fields, dt) -> tuple[float, float]` ⬜ STUB

**Параметры:**
- `agent: Agent` — агент, для которого вычисляется смещение
- `cytokine_fields: dict[str, np.ndarray]` — поля цитокинов (имя → 2D массив)
- `dt: float` — шаг времени

**Возвращает:** `(dx, dy)` — смещение по осям

**Алгоритм:**
1. Определить аттрактант по `agent.AGENT_TYPE` через `AGENT_ATTRACTANT_MAP`
2. Если тип не в карте → `(0.0, 0.0)`
3. Вычислить градиент поля `_compute_gradient(field, x, y, resolution)`
4. Нормализовать: `(gx, gy) / |∇C|`
5. Масштабировать: `χ · (gx, gy) · dt`

#### `_compute_gradient(self, field, x, y, grid_resolution) -> tuple[float, float]` ⬜ STUB

Центральные разности на 2D сетке:
```
∂C/∂x = (C[i+1, j] - C[i-1, j]) / (2·dx)
∂C/∂y = (C[i, j+1] - C[i, j-1]) / (2·dy)
```

### Тестовые сценарии

| Сценарий | Ожидаемый результат |
|----------|-------------------|
| Создание с ABMConfig | Сохранена конфигурация |
| AGENT_ATTRACTANT_MAP keys | 5 типов: neutro, macro, endo, fibro, platelet |
| `compute_displacement(...)` | `NotImplementedError` |
| `_compute_gradient(...)` | `NotImplementedError` |

---

## ContactInhibitionEngine

Подавление деления клеток при высокой локальной плотности.

### Методы

#### `__init__(self, threshold: int, radius: float) -> None`
Сохраняет `threshold` (макс. соседей) и `radius` (радиус поиска).

#### `compute_modifier(self, neighbor_count: int) -> float` ⬜ STUB
Множитель вероятности деления: `max(0, 1 - n / threshold)`.

**Возвращает:** float ∈ [0.0, 1.0]

#### `should_block_division(self, agent, neighbor_count) -> bool` ⬜ STUB
Блокировать ли деление: `neighbor_count >= threshold`.

### Тестовые сценарии

| Сценарий | Ожидаемый результат |
|----------|-------------------|
| Создание | `threshold` и `radius` сохранены |
| `compute_modifier(n)` | `NotImplementedError` |
| `should_block_division(agent, n)` | `NotImplementedError` |

---

## EfferocytosisEngine

Эффероцитоз — фагоцитоз апоптотических нейтрофилов макрофагами с выбросом IL-10.

### Методы

#### `__init__(self, il10_release_rate: float = 0.05) -> None`
Сохраняет скорость выброса IL-10.

#### `process(self, macrophage, apoptotic_neutrophils) -> dict[str, float]` ⬜ STUB

**Параметры:**
- `macrophage: Macrophage` — макрофаг-фагоцит
- `apoptotic_neutrophils: list[NeutrophilAgent]` — список апоптотических нейтрофилов

**Возвращает:** `{"IL10": amount, "phagocytosed": count}`

**Алгоритм:**
1. Фагоцитоз до `PHAGOCYTOSIS_CAPACITY` нейтрофилов
2. Помечает нейтрофилы как мёртвые
3. Сдвигает поляризацию макрофага к M2
4. Возвращает количество выброшенного IL-10

### Тестовые сценарии

| Сценарий | Ожидаемый результат |
|----------|-------------------|
| Создание с rate=0.05 | `il10_release_rate == 0.05` |
| Создание с default | `il10_release_rate == 0.05` |
| `process(macro, neutros)` | `NotImplementedError` |

---

## MechanotransductionEngine

Механотрансдукция — преобразование механических сигналов в биохимические. Механический стресс активирует фибробласты в миофибробласты.

### Методы

#### `__init__(self, stress_threshold: float = 0.5, activation_probability: float = 0.01) -> None`
Сохраняет пороги.

#### `compute_stress(self, agent, neighbors, ecm_density) -> float` ⬜ STUB

**Параметры:**
- `agent: Agent` — агент
- `neighbors: list[Agent]` — соседние агенты
- `ecm_density: float` — плотность ECM в позиции агента

**Возвращает:** Скалярное значение стресса ≥ 0.

#### `should_activate(self, fibroblast, stress) -> bool` ⬜ STUB

**Параметры:**
- `fibroblast: Fibroblast` — фибробласт-кандидат
- `stress: float` — вычисленный стресс

**Возвращает:** `True` если активация в миофибробласт должна произойти.

### Тестовые сценарии

| Сценарий | Ожидаемый результат |
|----------|-------------------|
| Создание с defaults | `stress_threshold=0.5, activation_probability=0.01` |
| Создание с custom | Сохранены пользовательские значения |
| `compute_stress(...)` | `NotImplementedError` |
| `should_activate(...)` | `NotImplementedError` |

---

## MultiCytokineField

Раздельные 2D поля для каждого цитокина. Заменяет единое `cytokine_field` из базовой ABM.

### Константы класса

| Константа | Тип | Значение |
|-----------|-----|----------|
| CYTOKINE_NAMES | list[str] | ["TNF", "IL10", "PDGF", "VEGF", "TGFb", "MCP1", "IL8"] |

### Атрибуты экземпляра

| Атрибут | Тип | Описание |
|---------|-----|----------|
| fields | dict[str, np.ndarray] | Словарь: имя цитокина → 2D массив концентраций |

### Методы

#### `__init__(self, grid_shape, cytokine_names=None) -> None`
Создаёт `self.fields` — словарь `{name: np.zeros(grid_shape)}` для каждого цитокина.

- `grid_shape: tuple[int, int]` — размер сетки
- `cytokine_names: list[str] | None` — список цитокинов (по умолчанию `CYTOKINE_NAMES`)

#### `update(self, dt, agents, config) -> None` ⬜ STUB
Обновление всех полей: диффузия, распад, секреция агентами.

#### `get_gradient(self, cytokine_name, x, y, grid_resolution) -> tuple[float, float]` ⬜ STUB
Градиент конкретного цитокина в точке (x, y).

#### `get_concentration(self, cytokine_name, x, y, grid_resolution) -> float` ⬜ STUB
Концентрация цитокина в точке (x, y).

### Тестовые сценарии

| Сценарий | Ожидаемый результат |
|----------|-------------------|
| Создание с default names | 7 полей, все zeros |
| Создание с custom names | Только указанные поля |
| `fields["TNF"].shape` | == grid_shape |
| `update(...)` | `NotImplementedError` |
| `get_gradient(...)` | `NotImplementedError` |
| `get_concentration(...)` | `NotImplementedError` |

---

## KDTreeNeighborSearch

Адаптер KD-Tree с единым интерфейсом поиска соседей. Обёртка над `KDTreeSpatialIndex` из `abm_model.py` с поддержкой `exclude` и консистентным API.

### Методы

#### `__init__(self, space_size, periodic=True) -> None`
Создаёт внутренний `KDTreeSpatialIndex`.

- `space_size: tuple[float, float]` — размер пространства
- `periodic: bool` — периодические граничные условия

#### `rebuild(self, agents) -> None` ⬜ STUB
Перестроение дерева по текущим позициям агентов.

#### `query_radius(self, position, radius, exclude=None) -> list[Agent]` ⬜ STUB
Все агенты в радиусе `radius` от `position`, исключая `exclude`.

#### `query_nearest(self, position, k=1, exclude=None) -> list[Agent]` ⬜ STUB
`k` ближайших соседей от `position`.

### Тестовые сценарии

| Сценарий | Ожидаемый результат |
|----------|-------------------|
| Создание | Внутренний `_index` создан |
| `rebuild(agents)` | `NotImplementedError` |
| `query_radius(pos, r)` | `NotImplementedError` |
| `query_nearest(pos, k)` | `NotImplementedError` |

---

## SubcyclingManager

Менеджер subcycling — разные шаги по времени для агентов и цитокиновых полей.

### Поля (dataclass)

| Поле | Тип | Описание |
|------|-----|----------|
| agent_dt | float | Шаг времени для агентов |
| field_dt | float | Шаг времени для полей (обычно < agent_dt) |

### Методы

#### `n_field_substeps` (property) ⬜ STUB
Количество подшагов полей за один шаг агента: `ceil(agent_dt / field_dt)`.

#### `should_update_field(self, agent_step_count: int) -> bool` ⬜ STUB
Нужно ли обновлять поля на данном подшаге.

#### `get_field_dt(self) -> float` ⬜ STUB
Фактический шаг для полей (может отличаться от `field_dt` для точного деления).

### Тестовые сценарии

| Сценарий | Ожидаемый результат |
|----------|-------------------|
| Создание | `agent_dt`, `field_dt` сохранены |
| `n_field_substeps` | `NotImplementedError` |
| `should_update_field(n)` | `NotImplementedError` |
| `get_field_dt()` | `NotImplementedError` |

---

## Расширения существующих агентов (в abm_model.py)

### StemCell.prp_mobilization

```python
def prp_mobilization(self, prp_level: float) -> float
```

PRP-зависимая мобилизация стволовой клетки. Повышает энергию и вероятность деления при высоком уровне PRP.

**Параметры:** `prp_level: float` — концентрация PRP (0–1)

**Возвращает:** `float` — модификатор активности (множитель > 1.0 при PRP > 0)

**Формула:**
```
modifier = 1.0 + k_prp · prp_level / (K_m + prp_level)
```
Где `k_prp = 2.0`, `K_m = 0.3` (Michaelis-Menten).

---

### Macrophage.efferocytose

```python
def efferocytose(self, apoptotic_neutrophils: list[NeutrophilAgent]) -> dict[str, float]
```

Фагоцитоз апоптотических нейтрофилов. Сдвигает поляризацию к M2, выбрасывает IL-10.

**Параметры:** `apoptotic_neutrophils: list[NeutrophilAgent]` — список апоптотических нейтрофилов в радиусе фагоцитоза

**Возвращает:** `{"IL10": amount, "phagocytosed": count}`

> Примечание: в будущем `polarization_state` станет `float` ∈ [0, 1] вместо дискретного M0/M1/M2. Это изменение произойдёт при полной реализации метода.

---

### Fibroblast.tgfb_activation

```python
def tgfb_activation(self, tgfb_level: float) -> MyofibroblastAgent | None
```

TGF-β-зависимая активация фибробласта в миофибробласт.

**Параметры:** `tgfb_level: float` — концентрация TGF-β в окрестности

**Возвращает:** `MyofibroblastAgent | None` — новый миофибробласт (замена) или None

**Порог активации:** `tgfb_level > 0.5` и `random() < 0.02 * dt`

---

## Инварианты

1. PlateletAgent никогда не делится (`MAX_DIVISIONS = 0`)
2. Все stub-методы выбрасывают `NotImplementedError` до реализации
3. `MultiCytokineField.fields` всегда содержит все заявленные цитокины
4. `KDTreeNeighborSearch` оборачивает `KDTreeSpatialIndex` из `abm_model.py`
5. `SubcyclingManager.n_field_substeps ≥ 1` всегда
6. Контактное ингибирование: `compute_modifier` ∈ [0, 1]
7. Эффероцитоз помечает нейтрофилы как мёртвые (`alive = False`)

## Граничные случаи

- Нулевой градиент → хемотаксисное смещение = (0, 0)
- Нет соседей → `compute_modifier = 1.0` (нет ингибирования)
- Пустой список нейтрофилов → эффероцитоз ничего не делает
- `ecm_density = 0` → стресс минимален
- `field_dt > agent_dt` → `n_substeps = 1` (нет subcycling)
