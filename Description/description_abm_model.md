# Описание: abm_model.py

## Обзор

Agent-Based модель (ABM) для моделирования регенерации тканей на микроуровне. Симулирует индивидуальные клетки как автономных агентов с собственными правилами поведения, включая движение, деление, гибель и взаимодействия.

---

## Теоретическое обоснование

### Типы агентов

| Тип | Маркеры | Функции | Lifespan |
|-----|---------|---------|----------|
| StemCell | CD34+ | Пролиферация, дифференциация, секреция PDGF/VEGF | 240ч (10 дней) |
| Macrophage | CD14+/CD68+ | Фагоцитоз, воспаление, поляризация M1/M2 | 168ч (7 дней) |
| Fibroblast | — | Производство ECM, ремоделирование | 360ч (15 дней) |

### Правила движения

**Random walk (диффузия):**
```
dx = √(2D·dt) · ξ
dy = √(2D·dt) · η
```
Где D — коэффициент диффузии, ξ, η ~ N(0, 1).

**Хемотаксис (направленное движение):**
```
v_chemotaxis = χ · ∇C / |∇C|
```
Где χ — чувствительность, C — концентрация цитокинов.

### Правила деления

```
P(division) = p_div · dt · (E > E_threshold) · (n_div < n_max)
```
Где:
- p_div — базовая вероятность деления
- E — уровень энергии агента
- E_threshold — порог энергии для деления
- n_div — количество прошедших делений
- n_max — максимальное количество делений

### Правила гибели

```
P(death) = p_death · dt + P(age > lifespan) + P(energy ≤ 0)
```

---

## Классы

### ABMConfig

**Назначение:** Конфигурация ABM модели.

**Атрибуты:**

| Атрибут | Тип | По умолчанию | Описание |
|---------|-----|--------------|----------|
| space_size | tuple[float, float] | (100, 100) | Размер пространства (мкм) |
| boundary_type | str | "periodic" | Тип границ |
| dt | float | 0.1 | Шаг времени (часы) |
| t_max | float | 720.0 | Макс. время (часы = 30 дней) |
| initial_stem_cells | int | 50 | Начальное кол-во стволовых |
| initial_macrophages | int | 30 | Начальное кол-во макрофагов |
| initial_fibroblasts | int | 20 | Начальное кол-во фибробластов |
| max_agents | int | 10000 | Макс. кол-во агентов |
| diffusion_coefficient | float | 1.0 | Коэфф. диффузии (мкм²/час) |
| chemotaxis_strength | float | 0.1 | Сила хемотаксиса |
| interaction_radius | float | 5.0 | Радиус взаимодействия (мкм) |
| grid_resolution | float | 10.0 | Разрешение сетки цитокинов |

---

### AgentState

**Назначение:** Состояние агента в момент времени.

**Атрибуты:**

| Атрибут | Тип | Описание |
|---------|-----|----------|
| agent_id | int | Уникальный ID |
| agent_type | str | "stem", "macro", "fibro" |
| x, y | float | Координаты (мкм) |
| age | float | Возраст (часы) |
| division_count | int | Количество делений |
| energy | float | Энергия (0-1) |
| alive | bool | Жив ли агент |
| dividing | bool | В процессе деления |

---

### ABMSnapshot

**Назначение:** Снимок состояния ABM в момент времени.

**Атрибуты:**

| Атрибут | Тип | Описание |
|---------|-----|----------|
| t | float | Время (часы) |
| agents | list[AgentState] | Список агентов |
| cytokine_field | np.ndarray | 2D поле цитокинов |
| ecm_field | np.ndarray | 2D поле ECM |

---

### Agent (базовый класс)

**Назначение:** Абстрактный базовый класс для всех типов агентов.

**Константы класса:**

| Константа | Значение | Описание |
|-----------|----------|----------|
| AGENT_TYPE | "base" | Тип агента |
| LIFESPAN | 240.0 | Продолжительность жизни (часы) |
| DIVISION_ENERGY_THRESHOLD | 0.7 | Порог энергии для деления |
| MAX_DIVISIONS | 10 | Макс. количество делений |
| DIVISION_PROBABILITY | 0.01 | Вероятность деления (1/час) |
| DEATH_PROBABILITY | 0.001 | Вероятность гибели (1/час) |

---

### StemCell

**Назначение:** CD34+ стволовая клетка.

**Дополнительные константы:**

| Константа | Значение | Описание |
|-----------|----------|----------|
| DIFFERENTIATION_PROBABILITY | 0.005 | Вероятность дифференциации |
| CYTOKINE_SECRETION_RATE | 0.1 | Скорость секреции (нг/мл/час) |

**Специфические методы:**
- `should_differentiate()` — проверка дифференциации
- `differentiate(new_id)` — превращение в фибробласт
- `secrete_cytokines(dt)` — секреция факторов роста

---

### Macrophage

**Назначение:** CD14+/CD68+ макрофаг.

**Дополнительные константы:**

| Константа | Значение | Описание |
|-----------|----------|----------|
| PHAGOCYTOSIS_RADIUS | 3.0 | Радиус фагоцитоза (мкм) |
| PHAGOCYTOSIS_CAPACITY | 5 | Макс. debris за шаг |
| CHEMOTAXIS_SENSITIVITY | 0.5 | Чувствительность к градиенту |

**Дополнительные атрибуты:**
- `polarization_state` — состояние поляризации (M0/M1/M2)
- `phagocytosed_count` — количество поглощённых частиц

**Специфические методы:**
- `phagocytose(debris_count)` — фагоцитоз debris
- `polarize(inflammation_level)` — поляризация M1/M2
- `secrete_cytokines(dt)` — секреция цитокинов

---

### Fibroblast

**Назначение:** Фибробласт.

**Дополнительные константы:**

| Константа | Значение | Описание |
|-----------|----------|----------|
| ECM_PRODUCTION_RATE | 0.5 | Скорость производства ECM |
| CONTRACTION_STRENGTH | 0.1 | Сила контракции |

**Дополнительные атрибуты:**
- `ecm_produced` — количество произведённого ECM
- `activated` — активирован ли (миофибробласт)

**Специфические методы:**
- `produce_ecm(dt)` — производство ECM
- `activate()` — активация в миофибробласт

---

## Методы ABMModel

### simulate

**Сигнатура:**
```python
def simulate(
    self,
    initial_params: ModelParameters,
    snapshot_interval: float = 24.0,
) -> ABMTrajectory
```

**Алгоритм:**
```python
# 1. Инициализация
self.initialize_from_parameters(initial_params)
snapshots = []
t = 0.0

# 2. Основной цикл
while t < self._config.t_max:
    # Шаг симуляции
    self.step(self._config.dt)
    t += self._config.dt

    # Сохранение снимка
    if t % snapshot_interval < self._config.dt:
        snapshots.append(self._get_snapshot())

# 3. Создать траекторию
return ABMTrajectory(snapshots=snapshots, config=self._config)
```

---

### step

**Сигнатура:**
```python
def step(self, dt: float) -> None
```

**Алгоритм:**
```python
# 1. Обновить всех агентов
self._update_agents(dt)

# 2. Обработать деления
self._handle_divisions()

# 3. Обработать дифференциации
self._handle_differentiations()

# 4. Удалить мёртвых
self._remove_dead_agents()

# 5. Обновить поля
self._update_cytokine_field(dt)
self._update_ecm_field(dt)

# 6. Обновить время
self._current_time += dt
```

---

### _update_agents

**Сигнатура:**
```python
def _update_agents(self, dt: float) -> None
```

**Алгоритм:**
```python
for agent in self._agents:
    if not agent.alive:
        continue

    # Получить окружение
    env = self._get_environment(agent.x, agent.y)

    # Обновить агента
    agent.update(dt, env)

    # Движение
    dx, dy = agent._random_walk_displacement(
        self._config.diffusion_coefficient, dt
    )
    # Добавить хемотаксис для макрофагов
    if isinstance(agent, Macrophage):
        grad = self._get_cytokine_gradient(agent.x, agent.y)
        dx += self._config.chemotaxis_strength * grad[0] * dt
        dy += self._config.chemotaxis_strength * grad[1] * dt

    agent.move(dx, dy, self._config.space_size, self._config.boundary_type)
```

---

## Примеры использования

```python
from src.data.parameter_extraction import ModelParameters
from src.core.abm_model import (
    ABMModel,
    ABMConfig,
    simulate_abm,
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

# Конфигурация
config = ABMConfig(
    space_size=(100.0, 100.0),
    dt=0.1,
    t_max=720.0,  # 30 дней
    initial_stem_cells=50,
    initial_macrophages=30,
    initial_fibroblasts=20,
)

# Способ 1: через класс
model = ABMModel(config=config, random_seed=42)
trajectory = model.simulate(params, snapshot_interval=24.0)

# Анализ результатов
dynamics = trajectory.get_population_dynamics()
print(f"Стволовые клетки: {dynamics['stem'][-1]}")
print(f"Макрофаги: {dynamics['macro'][-1]}")
print(f"Фибробласты: {dynamics['fibro'][-1]}")

# Способ 2: convenience функция
trajectory = simulate_abm(
    initial_params=params,
    config=config,
    random_seed=42,
    snapshot_interval=24.0,
)

# Статистика
stats = trajectory.get_statistics()
print(f"Финальное количество клеток: {stats['final_total']}")
```

---

## Взаимодействия агентов

### Контактное ингибирование

```python
# Если соседей > порога, деление блокируется
neighbors = self._count_neighbors(agent, self._config.contact_inhibition_radius)
if neighbors > CONTACT_INHIBITION_THRESHOLD:
    agent.dividing = False
```

### Хемотаксис макрофагов

```python
# Движение к области высокой концентрации цитокинов
gradient = self._get_cytokine_gradient(x, y)
dx_chemotaxis = CHEMOTAXIS_SENSITIVITY * gradient[0]
dy_chemotaxis = CHEMOTAXIS_SENSITIVITY * gradient[1]
```

### Поляризация макрофагов

```python
# M1 (провоспалительный) при высоком воспалении
# M2 (противовоспалительный) при низком
if inflammation_level > 0.5:
    macrophage.polarization_state = "M1"
else:
    macrophage.polarization_state = "M2"
```

---

## Зависимости

- numpy
- abc (stdlib)
- dataclasses (stdlib)
- src.data.parameter_extraction (ModelParameters)

---

## Параметры из литературы

| Параметр | Значение | Источник |
|----------|----------|----------|
| Stem cell lifespan | 7-14 дней | Cell culture studies |
| Macrophage lifespan | 5-10 дней | In vivo studies |
| Fibroblast lifespan | 10-20 дней | Tissue studies |
| Diffusion coefficient | 0.5-2 мкм²/час | Cell migration studies |
| Division time | 12-24 часа | Cell cycle duration |
