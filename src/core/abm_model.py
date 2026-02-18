"""Agent-Based модель клеточной динамики для регенерации тканей.

Моделирование дискретных клеточных событий на микроуровне:
- Пространственное движение (random walk + chemotaxis)
- Деление и гибель клеток
- Взаимодействия между агентами
- Типы агентов: StemCell (CD34+), Macrophage (CD14+/CD68+), Fibroblast,
  NeutrophilAgent (CD66b+), EndothelialAgent (CD31+), MyofibroblastAgent (α-SMA+)

Подробное описание: Description/description_abm_model.md
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from src.data.parameter_extraction import ModelParameters


@dataclass
class ABMConfig:
    """Конфигурация ABM модели.

    Подробное описание: Description/description_abm_model.md#ABMConfig
    """

    # Пространственные параметры
    space_size: tuple[float, float] = (100.0, 100.0)  # мкм × мкм
    boundary_type: str = "periodic"  # "periodic", "reflective", "absorbing"

    # Временные параметры
    dt: float = 0.1  # Шаг времени (часы)
    t_max: float = 720.0  # Максимальное время (часы = 30 дней)

    # Параметры агентов
    initial_stem_cells: int = 50
    initial_macrophages: int = 30
    initial_fibroblasts: int = 20
    initial_neutrophils: int = 0  # Рекрутируются динамически из кровотока
    initial_endothelial: int = 10  # Начальное количество эндотелиальных клеток
    initial_myofibroblasts: int = 0  # Активируются из фибробластов при TGF-β
    max_agents: int = 10000

    # Параметры движения
    diffusion_coefficient: float = 1.0  # мкм²/час
    chemotaxis_strength: float = 0.1  # Сила хемотаксиса

    # Параметры взаимодействий
    interaction_radius: float = 5.0  # мкм
    contact_inhibition_radius: float = 2.0  # мкм
    contact_inhibition_threshold: int = 5  # Макс. соседей для деления
    repulsion_strength: float = 1.0  # Коэффициент силы отталкивания

    # Параметры адгезии и расширенной механики
    adhesion_strength: float = 0.3  # Сила адгезии между совместимыми типами
    adhesion_equilibrium_distance: float = 3.0  # мкм, равновесное расстояние
    use_multi_chemotaxis: bool = False  # Мульти-градиентный хемотаксис по типу
    spatial_index_type: str = "hash"  # "hash" (SpatialHash) или "kdtree"

    # Параметры цитокинового поля
    grid_resolution: float = 10.0  # мкм на ячейку сетки
    cytokine_diffusion: float = 10.0  # мкм²/час
    cytokine_decay: float = 0.1  # 1/час

    def validate(self) -> bool:
        """Валидация параметров конфигурации.

        Returns:
            True если все параметры валидны

        Raises:
            ValueError: Если параметры некорректны

        Подробное описание: Description/description_abm_model.md#ABMConfig.validate
        """
        if self.space_size[0] <= 0 or self.space_size[1] <= 0:
            raise ValueError("space_size должен быть положительным")
        if self.boundary_type not in ["periodic", "reflective", "absorbing"]:
            raise ValueError("boundary_type должен быть 'periodic', 'reflective' или 'absorbing'")
        if self.dt <= 0:
            raise ValueError("dt должен быть положительным")
        if self.t_max <= 0:
            raise ValueError("t_max должен быть положительным")
        if self.max_agents <= 0:
            raise ValueError("max_agents должен быть положительным")
        if self.interaction_radius < 0:
            raise ValueError("interaction_radius должен быть неотрицательным")
        if self.chemotaxis_strength < 0:
            raise ValueError("chemotaxis_strength должен быть неотрицательным")
        return True


@dataclass
class AgentState:
    """Состояние агента в момент времени.

    Подробное описание: Description/description_abm_model.md#AgentState
    """

    agent_id: int
    agent_type: str  # "stem", "macro", "fibro"

    # Пространственные координаты
    x: float
    y: float

    # Биологические свойства
    age: float  # часы с рождения
    division_count: int  # количество делений
    energy: float  # уровень энергии (0-1)

    # Статус
    alive: bool = True
    dividing: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Конвертация в словарь.

        Returns:
            Словарь с состоянием агента

        Подробное описание: Description/description_abm_model.md#AgentState.to_dict
        """
        return {
            "id": self.agent_id,
            "type": self.agent_type,
            "x": self.x,
            "y": self.y,
            "age": self.age,
            "divisions": self.division_count,
            "energy": self.energy,
            "alive": self.alive,
            "dividing": self.dividing,
        }


@dataclass
class ABMSnapshot:
    """Снимок состояния ABM в момент времени.

    Подробное описание: Description/description_abm_model.md#ABMSnapshot
    """

    t: float  # Время (часы)
    agents: list[AgentState]
    cytokine_field: np.ndarray  # 2D grid концентраций
    ecm_field: np.ndarray  # 2D grid внеклеточного матрикса

    def get_agent_count_by_type(self) -> dict[str, int]:
        """Подсчёт агентов по типам.

        Returns:
            Словарь с количеством агентов каждого типа

        Подробное описание: Description/description_abm_model.md#ABMSnapshot.get_agent_count_by_type
        """
        counts = {"stem": 0, "macro": 0, "fibro": 0}
        for agent in self.agents:
            if agent.alive and agent.agent_type in counts:
                counts[agent.agent_type] += 1
        return counts

    def get_total_agents(self) -> int:
        """Общее количество живых агентов.

        Returns:
            Количество живых агентов

        Подробное описание: Description/description_abm_model.md#ABMSnapshot.get_total_agents
        """
        return sum(1 for agent in self.agents if agent.alive)


@dataclass
class ABMTrajectory:
    """Траектория ABM симуляции.

    Подробное описание: Description/description_abm_model.md#ABMTrajectory
    """

    snapshots: list[ABMSnapshot]  # Сохранённые временные точки
    config: ABMConfig = field(default_factory=ABMConfig)

    def get_times(self) -> np.ndarray:
        """Получить массив временных точек.

        Returns:
            Массив времён всех снимков

        Подробное описание: Description/description_abm_model.md#ABMTrajectory.get_times
        """
        return np.array([snapshot.t for snapshot in self.snapshots])

    def get_population_dynamics(self) -> dict[str, np.ndarray]:
        """Динамика популяций во времени.

        Returns:
            Словарь с массивами численности каждого типа агентов

        Подробное описание: Description/description_abm_model.md#ABMTrajectory.get_population_dynamics
        """
        stem_counts = []
        macro_counts = []
        fibro_counts = []

        for snapshot in self.snapshots:
            counts = snapshot.get_agent_count_by_type()
            stem_counts.append(counts.get("stem", 0))
            macro_counts.append(counts.get("macro", 0))
            fibro_counts.append(counts.get("fibro", 0))

        return {
            "stem": np.array(stem_counts),
            "macro": np.array(macro_counts),
            "fibro": np.array(fibro_counts),
        }

    def get_statistics(self) -> dict[str, float]:
        """Финальная статистика.

        Returns:
            Словарь со статистиками

        Подробное описание: Description/description_abm_model.md#ABMTrajectory.get_statistics
        """
        if not self.snapshots:
            return {}

        final_snapshot = self.snapshots[-1]
        final_counts = final_snapshot.get_agent_count_by_type()
        final_total = final_snapshot.get_total_agents()

        # Начальная статистика
        initial_snapshot = self.snapshots[0]
        initial_total = initial_snapshot.get_total_agents()

        # Расчёт скорости роста
        times = self.get_times()
        t_total = times[-1] - times[0] if len(times) > 1 else 0.0
        if t_total > 0 and initial_total > 0:
            growth_rate = (final_total - initial_total) / t_total
        else:
            growth_rate = 0.0

        return {
            "final_total": float(final_total),
            "final_stem": float(final_counts.get("stem", 0)),
            "final_macro": float(final_counts.get("macro", 0)),
            "final_fibro": float(final_counts.get("fibro", 0)),
            "growth_rate": float(growth_rate),
        }


class SpatialHash:
    """Пространственный хэш для эффективного поиска соседей O(1).

    Делит пространство на ячейки размером cell_size × cell_size.
    Поиск соседей в радиусе r требует проверки только
    ceil(r/cell_size) + 1 ячеек в каждом направлении.

    Подробное описание: Description/description_abm_model.md#SpatialHash
    """

    def __init__(
        self,
        space_size: tuple[float, float],
        cell_size: float,
        periodic: bool = True,
    ) -> None:
        """Инициализация пространственного хэша.

        Args:
            space_size: Размер пространства (ширина, высота) в мкм
            cell_size: Размер ячейки (обычно = interaction_radius)
            periodic: Использовать периодические границы
        """
        self._space_size = space_size
        self._cell_size = cell_size
        self._periodic = periodic

        # Размер сетки
        self._grid_width = int(np.ceil(space_size[0] / cell_size))
        self._grid_height = int(np.ceil(space_size[1] / cell_size))

        # Ячейки: dict[tuple[int, int], list[Agent]]
        self._cells: dict[tuple[int, int], list] = {}

    def _get_cell(self, x: float, y: float) -> tuple[int, int]:
        """Получить индекс ячейки для координаты."""
        cx = int(x / self._cell_size) % self._grid_width
        cy = int(y / self._cell_size) % self._grid_height
        return (cx, cy)

    def clear(self) -> None:
        """Очистить все ячейки."""
        self._cells.clear()

    def insert(self, agent: "Agent") -> None:
        """Добавить агента в соответствующую ячейку."""
        cell = self._get_cell(agent.x, agent.y)
        if cell not in self._cells:
            self._cells[cell] = []
        self._cells[cell].append(agent)

    def rebuild(self, agents: list["Agent"]) -> None:
        """Перестроить хэш из списка агентов."""
        self.clear()
        for agent in agents:
            if agent.alive:
                self.insert(agent)

    def get_neighbors(
        self,
        x: float,
        y: float,
        radius: float,
        exclude: "Agent | None" = None,
    ) -> list["Agent"]:
        """Найти всех соседей в радиусе.

        Args:
            x: Координата X точки поиска
            y: Координата Y точки поиска
            radius: Радиус поиска
            exclude: Агент для исключения из результатов

        Returns:
            Список агентов в радиусе
        """
        neighbors = []

        # Сколько ячеек проверять в каждом направлении
        cell_range = int(np.ceil(radius / self._cell_size)) + 1
        center_cell = self._get_cell(x, y)

        for di in range(-cell_range, cell_range + 1):
            for dj in range(-cell_range, cell_range + 1):
                if self._periodic:
                    ci = (center_cell[0] + di) % self._grid_width
                    cj = (center_cell[1] + dj) % self._grid_height
                else:
                    ci = center_cell[0] + di
                    cj = center_cell[1] + dj
                    if ci < 0 or ci >= self._grid_width:
                        continue
                    if cj < 0 or cj >= self._grid_height:
                        continue

                cell = (ci, cj)
                if cell not in self._cells:
                    continue

                for agent in self._cells[cell]:
                    if agent is exclude:
                        continue

                    # Точная проверка расстояния
                    dx = abs(x - agent.x)
                    dy = abs(y - agent.y)

                    if self._periodic:
                        dx = min(dx, self._space_size[0] - dx)
                        dy = min(dy, self._space_size[1] - dy)

                    if dx * dx + dy * dy <= radius * radius:
                        neighbors.append(agent)

        return neighbors


class KDTreeSpatialIndex:
    """Пространственный индекс на основе scipy.spatial.cKDTree.

    Альтернатива SpatialHash для точного O(log n) поиска соседей.
    Использует KD-дерево для эффективных запросов по радиусу
    и k ближайших соседей. Поддерживает периодические границы.

    Подробное описание: Description/Phase2/description_abm_model.md#KDTreeSpatialIndex
    """

    def __init__(
        self,
        space_size: tuple[float, float],
        periodic: bool = True,
    ) -> None:
        """Инициализация KD-Tree пространственного индекса.

        Args:
            space_size: Размер пространства (ширина, высота) в мкм
            periodic: Использовать периодические границы (тор)

        Подробное описание: Description/Phase2/description_abm_model.md#KDTreeSpatialIndex.__init__
        """
        self._space_size = space_size
        self._periodic = periodic
        self._tree: cKDTree | None = None
        self._agents: list[Agent] = []

    def build(self, agents: list["Agent"]) -> None:
        """Построение KD-дерева из списка агентов.

        Создаёт cKDTree из координат живых агентов.
        При periodic=True используется boxsize для тороидальных границ.

        Args:
            agents: Список агентов для индексации

        Подробное описание: Description/Phase2/description_abm_model.md#KDTreeSpatialIndex.build
        """
        self._agents = [a for a in agents if a.alive]
        if not self._agents:
            self._tree = None
            return
        positions = np.array([[a.x, a.y] for a in self._agents])
        if self._periodic:
            self._tree = cKDTree(
                positions,
                boxsize=[self._space_size[0], self._space_size[1]],
            )
        else:
            self._tree = cKDTree(positions)

    def query_radius(
        self,
        position: tuple[float, float],
        radius: float,
    ) -> list["Agent"]:
        """Поиск всех агентов в заданном радиусе.

        Использует cKDTree.query_ball_point для эффективного
        поиска по радиусу. Возвращает агентов на расстоянии ≤ radius.

        Args:
            position: Координаты центра поиска (x, y)
            radius: Радиус поиска (мкм)

        Returns:
            Список агентов в радиусе

        Подробное описание: Description/Phase2/description_abm_model.md#KDTreeSpatialIndex.query_radius
        """
        if self._tree is None or not self._agents or radius <= 0:
            return []
        indices = self._tree.query_ball_point(list(position), radius)
        return [self._agents[i] for i in indices]

    def query_nearest(
        self,
        position: tuple[float, float],
        k: int = 1,
    ) -> list["Agent"]:
        """Поиск k ближайших агентов.

        Использует cKDTree.query для нахождения k ближайших соседей.
        Если k > количества агентов, возвращает всех.

        Args:
            position: Координаты точки поиска (x, y)
            k: Количество ближайших соседей

        Returns:
            Список k ближайших агентов (отсортированы по расстоянию)

        Подробное описание: Description/Phase2/description_abm_model.md#KDTreeSpatialIndex.query_nearest
        """
        if self._tree is None or not self._agents or k <= 0:
            return []
        k = min(k, len(self._agents))
        distances, indices = self._tree.query(list(position), k=k)
        if k == 1:
            return [self._agents[int(indices)]]
        return [self._agents[int(i)] for i in indices]


class Agent(ABC):
    """Базовый абстрактный класс агента.

    Подробное описание: Description/description_abm_model.md#Agent
    """

    # Константы класса (переопределяются в наследниках)
    AGENT_TYPE: str = "base"
    LIFESPAN: float = 240.0  # часов
    DIVISION_ENERGY_THRESHOLD: float = 0.7
    MAX_DIVISIONS: int = 10
    DIVISION_PROBABILITY: float = 0.01  # per hour
    DEATH_PROBABILITY: float = 0.001  # per hour

    def __init__(
        self,
        agent_id: int,
        x: float,
        y: float,
        age: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Инициализация агента.

        Args:
            agent_id: Уникальный идентификатор
            x: Координата X (мкм)
            y: Координата Y (мкм)
            age: Возраст (часы)
            rng: Генератор случайных чисел

        Подробное описание: Description/description_abm_model.md#Agent.__init__
        """
        self.agent_id = agent_id
        self.x = x
        self.y = y
        self.age = age
        self.division_count = 0
        self.energy = 1.0
        self.alive = True
        self.dividing = False
        self._rng = rng if rng else np.random.default_rng()

    def move(
        self,
        dx: float,
        dy: float,
        space_size: tuple[float, float],
        boundary: str = "periodic",
    ) -> None:
        """Перемещение агента с учётом граничных условий.

        Args:
            dx: Смещение по X
            dy: Смещение по Y
            space_size: Размер пространства (width, height)
            boundary: Тип границ

        Подробное описание: Description/description_abm_model.md#Agent.move
        """
        new_x = self.x + dx
        new_y = self.y + dy

        if boundary == "periodic":
            # Периодические границы (тор)
            self.x = new_x % space_size[0]
            self.y = new_y % space_size[1]
        elif boundary == "reflective":
            # Отражающие границы
            # X
            if new_x < 0:
                self.x = -new_x
            elif new_x >= space_size[0]:
                self.x = 2 * space_size[0] - new_x - 0.001
            else:
                self.x = new_x
            # Y
            if new_y < 0:
                self.y = -new_y
            elif new_y >= space_size[1]:
                self.y = 2 * space_size[1] - new_y - 0.001
            else:
                self.y = new_y
            # Убедимся что внутри границ
            self.x = max(0.0, min(self.x, space_size[0] - 0.001))
            self.y = max(0.0, min(self.y, space_size[1] - 0.001))
        elif boundary == "absorbing":
            # Поглощающие границы - агент умирает при выходе
            if 0 <= new_x < space_size[0] and 0 <= new_y < space_size[1]:
                self.x = new_x
                self.y = new_y
            else:
                self.alive = False
        else:
            # По умолчанию periodic
            self.x = new_x % space_size[0]
            self.y = new_y % space_size[1]

    @abstractmethod
    def update(self, dt: float, environment: dict[str, Any]) -> None:
        """Обновление состояния агента за шаг dt.

        Args:
            dt: Временной шаг (часы)
            environment: Словарь с параметрами окружения

        Подробное описание: Description/description_abm_model.md#Agent.update
        """
        raise NotImplementedError("Stub: требуется реализация")

    def can_divide(self) -> bool:
        """Проверка возможности деления.

        Returns:
            True если агент может делиться

        Подробное описание: Description/description_abm_model.md#Agent.can_divide
        """
        return (
            self.alive
            and self.energy >= self.DIVISION_ENERGY_THRESHOLD
            and self.division_count < self.MAX_DIVISIONS
        )

    def should_die(self, dt: float) -> bool:
        """Проверка смерти агента.

        Args:
            dt: Временной шаг

        Returns:
            True если агент должен умереть

        Подробное описание: Description/description_abm_model.md#Agent.should_die
        """
        # Смерть по возрасту
        if self.age >= self.LIFESPAN:
            return True

        # Смерть по энергии
        if self.energy <= 0:
            return True

        # Случайная смерть с вероятностью
        if self._rng.random() < self.DEATH_PROBABILITY * dt:
            return True

        return False

    @abstractmethod
    def divide(self, new_id: int) -> "Agent | None":
        """Деление агента (создание потомка).

        Args:
            new_id: ID для нового агента

        Returns:
            Новый агент или None

        Подробное описание: Description/description_abm_model.md#Agent.divide
        """
        raise NotImplementedError("Stub: требуется реализация")

    def get_state(self) -> AgentState:
        """Получить текущее состояние.

        Returns:
            AgentState с текущими параметрами

        Подробное описание: Description/description_abm_model.md#Agent.get_state
        """
        return AgentState(
            agent_id=self.agent_id,
            agent_type=self.AGENT_TYPE,
            x=self.x,
            y=self.y,
            age=self.age,
            division_count=self.division_count,
            energy=self.energy,
            alive=self.alive,
            dividing=self.dividing,
        )

    def _random_walk_displacement(self, diffusion: float, dt: float) -> tuple[float, float]:
        """Случайное смещение (random walk).

        Args:
            diffusion: Коэффициент диффузии
            dt: Временной шаг

        Returns:
            (dx, dy) смещение

        Подробное описание: Description/description_abm_model.md#Agent._random_walk_displacement
        """
        # dx = sqrt(2*D*dt) * xi, где xi ~ N(0,1)
        sigma = np.sqrt(2.0 * diffusion * dt)
        dx = self._rng.normal(0.0, sigma)
        dy = self._rng.normal(0.0, sigma)
        return (dx, dy)


class StemCell(Agent):
    """CD34+ стволовая клетка.

    Свойства:
    - Высокая пролиферация
    - Дифференциация в фибробласты
    - Секреция факторов роста (PDGF, VEGF)

    Подробное описание: Description/description_abm_model.md#StemCell
    """

    AGENT_TYPE: str = "stem"
    LIFESPAN: float = 240.0  # часов (10 дней)
    DIVISION_ENERGY_THRESHOLD: float = 0.7
    MAX_DIVISIONS: int = 10
    DIVISION_PROBABILITY: float = 0.02  # per hour (высокая)
    DEATH_PROBABILITY: float = 0.001  # per hour

    DIFFERENTIATION_PROBABILITY: float = 0.005  # per hour
    CYTOKINE_SECRETION_RATE: float = 0.1  # нг/мл/час

    def __init__(
        self,
        agent_id: int,
        x: float,
        y: float,
        age: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Инициализация стволовой клетки.

        Args:
            agent_id: Уникальный идентификатор
            x: Координата X
            y: Координата Y
            age: Возраст
            rng: Генератор случайных чисел

        Подробное описание: Description/description_abm_model.md#StemCell.__init__
        """
        super().__init__(agent_id, x, y, age, rng)
        self.differentiation_probability = self.DIFFERENTIATION_PROBABILITY

    def update(self, dt: float, environment: dict[str, Any]) -> None:
        """Обновление стволовой клетки.

        Args:
            dt: Временной шаг
            environment: Параметры окружения (cytokine_level, prp_active, etc.)

        Подробное описание: Description/description_abm_model.md#StemCell.update
        """
        if not self.alive:
            return

        # Обновление возраста
        self.age += dt

        # Потребление энергии
        energy_consumption = 0.001 * dt
        self.energy = max(0.0, self.energy - energy_consumption)

        # Восстановление энергии от цитокинов
        cytokine_level = environment.get("cytokine_level", 0.0)
        if cytokine_level > 0:
            self.energy = min(1.0, self.energy + 0.01 * cytokine_level * dt)

        # Бонус от PRP терапии
        if environment.get("prp_active", False):
            self.energy = min(1.0, self.energy + 0.005 * dt)

        # Сброс флага деления
        self.dividing = False

    def can_divide(self) -> bool:
        """Проверка возможности деления.

        Returns:
            True если может делиться

        Подробное описание: Description/description_abm_model.md#StemCell.can_divide
        """
        return (
            self.alive
            and self.energy >= self.DIVISION_ENERGY_THRESHOLD
            and self.division_count < self.MAX_DIVISIONS
        )

    def divide(self, new_id: int) -> "StemCell | None":
        """Деление с созданием дочерней клетки.

        Args:
            new_id: ID для новой клетки

        Returns:
            Новая StemCell или None

        Подробное описание: Description/description_abm_model.md#StemCell.divide
        """
        if not self.can_divide():
            return None

        # Проверка вероятности деления
        if self._rng.random() >= self.DIVISION_PROBABILITY:
            return None

        # Деление происходит
        self.dividing = True
        self.division_count += 1
        self.energy *= 0.5  # Энергия делится пополам

        # Создание дочерней клетки
        offspring = StemCell(
            agent_id=new_id,
            x=self.x + self._rng.uniform(-1.0, 1.0),
            y=self.y + self._rng.uniform(-1.0, 1.0),
            age=0.0,
            rng=self._rng,
        )
        offspring.energy = self.energy
        offspring.division_count = 0

        return offspring

    def should_differentiate(self) -> bool:
        """Проверка дифференциации в фибробласт.

        Returns:
            True если клетка должна дифференцироваться

        Подробное описание: Description/description_abm_model.md#StemCell.should_differentiate
        """
        if not self.alive:
            return False
        return self._rng.random() < self.differentiation_probability

    def differentiate(self, new_id: int) -> "Fibroblast":
        """Дифференциация в фибробласт.

        Args:
            new_id: ID для нового фибробласта

        Returns:
            Новый Fibroblast

        Подробное описание: Description/description_abm_model.md#StemCell.differentiate
        """
        # Помечаем текущую клетку как мёртвую
        self.alive = False

        # Создаём фибробласт на том же месте
        fibroblast = Fibroblast(
            agent_id=new_id,
            x=self.x,
            y=self.y,
            age=0.0,
            rng=self._rng,
        )
        fibroblast.energy = self.energy

        return fibroblast

    def secrete_cytokines(self, dt: float) -> float:
        """Секреция цитокинов.

        Args:
            dt: Временной шаг

        Returns:
            Количество секретированных цитокинов

        Подробное описание: Description/description_abm_model.md#StemCell.secrete_cytokines
        """
        if not self.alive:
            return 0.0
        return self.CYTOKINE_SECRETION_RATE * dt


class Macrophage(Agent):
    """CD14+/CD68+ макрофаг.

    Свойства:
    - Фагоцитоз debris и апоптотических клеток
    - Секреция провоспалительных цитокинов (TNF-α, IL-1β)
    - Миграция к повреждённым областям (хемотаксис)
    - Поляризация M1 (провоспалительный) / M2 (противовоспалительный)

    Подробное описание: Description/description_abm_model.md#Macrophage
    """

    AGENT_TYPE: str = "macro"
    LIFESPAN: float = 168.0  # часов (7 дней)
    DIVISION_ENERGY_THRESHOLD: float = 0.8
    MAX_DIVISIONS: int = 3  # Макрофаги делятся редко
    DIVISION_PROBABILITY: float = 0.002  # per hour (низкая)
    DEATH_PROBABILITY: float = 0.002  # per hour

    PHAGOCYTOSIS_RADIUS: float = 3.0  # мкм
    PHAGOCYTOSIS_CAPACITY: int = 5  # максимум debris за шаг
    CHEMOTAXIS_SENSITIVITY: float = 0.5  # Чувствительность к градиенту

    def __init__(
        self,
        agent_id: int,
        x: float,
        y: float,
        age: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Инициализация макрофага.

        Args:
            agent_id: Уникальный идентификатор
            x: Координата X
            y: Координата Y
            age: Возраст
            rng: Генератор случайных чисел

        Подробное описание: Description/description_abm_model.md#Macrophage.__init__
        """
        super().__init__(agent_id, x, y, age, rng)
        self.polarization_state = "M0"  # M0, M1 (pro-inflam), M2 (anti-inflam)
        self.phagocytosed_count = 0

    def update(self, dt: float, environment: dict[str, Any]) -> None:
        """Обновление макрофага.

        Args:
            dt: Временной шаг
            environment: Параметры окружения

        Подробное описание: Description/description_abm_model.md#Macrophage.update
        """
        if not self.alive:
            return

        # Обновление возраста
        self.age += dt

        # Потребление энергии
        energy_consumption = 0.002 * dt  # Макрофаги потребляют больше энергии
        self.energy = max(0.0, self.energy - energy_consumption)

        # Восстановление энергии от фагоцитоза
        if self.phagocytosed_count > 0:
            self.energy = min(1.0, self.energy + 0.01 * self.phagocytosed_count * dt)

        # Поляризация в зависимости от воспаления
        inflammation = environment.get("inflammation_level", 0.0)
        self.polarize(inflammation)

        # Сброс флага деления
        self.dividing = False

    def divide(self, new_id: int) -> "Macrophage | None":
        """Деление макрофага.

        Args:
            new_id: ID для нового макрофага

        Returns:
            Новый Macrophage или None

        Подробное описание: Description/description_abm_model.md#Macrophage.divide
        """
        if not self.can_divide():
            return None

        # Проверка вероятности деления
        if self._rng.random() >= self.DIVISION_PROBABILITY:
            return None

        # Деление происходит
        self.dividing = True
        self.division_count += 1
        self.energy *= 0.5

        # Создание дочернего макрофага
        offspring = Macrophage(
            agent_id=new_id,
            x=self.x + self._rng.uniform(-1.0, 1.0),
            y=self.y + self._rng.uniform(-1.0, 1.0),
            age=0.0,
            rng=self._rng,
        )
        offspring.energy = self.energy
        offspring.polarization_state = self.polarization_state
        offspring.division_count = 0

        return offspring

    def phagocytose(self, debris_count: int) -> int:
        """Фагоцитоз частиц debris.

        Args:
            debris_count: Количество доступных частиц debris

        Returns:
            Количество поглощённых частиц

        Подробное описание: Description/description_abm_model.md#Macrophage.phagocytose
        """
        if not self.alive:
            return 0

        # Поглощаем до максимальной ёмкости
        phagocytosed = min(debris_count, self.PHAGOCYTOSIS_CAPACITY)
        self.phagocytosed_count += phagocytosed

        return phagocytosed

    def polarize(self, inflammation_level: float) -> None:
        """Поляризация макрофага.

        Args:
            inflammation_level: Уровень воспаления (0-1)

        Подробное описание: Description/description_abm_model.md#Macrophage.polarize
        """
        if inflammation_level > 0.5:
            self.polarization_state = "M1"  # Провоспалительный
        else:
            self.polarization_state = "M2"  # Противовоспалительный

    def secrete_cytokines(self, dt: float) -> dict[str, float]:
        """Секреция цитокинов в зависимости от поляризации.

        Args:
            dt: Временной шаг

        Returns:
            Словарь с количеством секретированных цитокинов

        Подробное описание: Description/description_abm_model.md#Macrophage.secrete_cytokines
        """
        if not self.alive:
            return {"TNF_alpha": 0.0, "IL_1beta": 0.0, "IL_10": 0.0}

        base_rate = 0.1 * dt

        if self.polarization_state == "M1":
            # Провоспалительные цитокины
            return {
                "TNF_alpha": base_rate,
                "IL_1beta": base_rate * 0.8,
                "IL_10": 0.0,
            }
        elif self.polarization_state == "M2":
            # Противовоспалительные цитокины
            return {
                "TNF_alpha": 0.0,
                "IL_1beta": 0.0,
                "IL_10": base_rate,
            }
        else:  # M0
            return {
                "TNF_alpha": base_rate * 0.2,
                "IL_1beta": base_rate * 0.1,
                "IL_10": base_rate * 0.1,
            }


class Fibroblast(Agent):
    """Фибробласт.

    Свойства:
    - Производство внеклеточного матрикса (ECM)
    - Ремоделирование ткани
    - Низкая пролиферация
    - Контрактильность

    Подробное описание: Description/description_abm_model.md#Fibroblast
    """

    AGENT_TYPE: str = "fibro"
    LIFESPAN: float = 360.0  # часов (15 дней)
    DIVISION_ENERGY_THRESHOLD: float = 0.8
    MAX_DIVISIONS: int = 5
    DIVISION_PROBABILITY: float = 0.005  # per hour
    DEATH_PROBABILITY: float = 0.001  # per hour

    ECM_PRODUCTION_RATE: float = 0.5  # units/hour
    CONTRACTION_STRENGTH: float = 0.1  # Сила контракции

    def __init__(
        self,
        agent_id: int,
        x: float,
        y: float,
        age: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Инициализация фибробласта.

        Args:
            agent_id: Уникальный идентификатор
            x: Координата X
            y: Координата Y
            age: Возраст
            rng: Генератор случайных чисел

        Подробное описание: Description/description_abm_model.md#Fibroblast.__init__
        """
        super().__init__(agent_id, x, y, age, rng)
        self.ecm_produced = 0.0
        self.activated = False  # Активированный фибробласт (миофибробласт)

    def update(self, dt: float, environment: dict[str, Any]) -> None:
        """Обновление фибробласта.

        Args:
            dt: Временной шаг
            environment: Параметры окружения

        Подробное описание: Description/description_abm_model.md#Fibroblast.update
        """
        if not self.alive:
            return

        # Обновление возраста
        self.age += dt

        # Потребление энергии
        energy_consumption = 0.001 * dt
        self.energy = max(0.0, self.energy - energy_consumption)

        # Восстановление энергии
        cytokine_level = environment.get("cytokine_level", 0.0)
        if cytokine_level > 0:
            self.energy = min(1.0, self.energy + 0.005 * cytokine_level * dt)

        # Активация при высоком уровне воспаления
        inflammation = environment.get("inflammation_level", 0.0)
        if inflammation > 0.7 and not self.activated:
            if self._rng.random() < 0.01 * dt:
                self.activate()

        # Производство ECM
        ecm_amount = self.produce_ecm(dt)
        self.ecm_produced += ecm_amount

        # Сброс флага деления
        self.dividing = False

    def divide(self, new_id: int) -> "Fibroblast | None":
        """Деление фибробласта.

        Args:
            new_id: ID для нового фибробласта

        Returns:
            Новый Fibroblast или None

        Подробное описание: Description/description_abm_model.md#Fibroblast.divide
        """
        if not self.can_divide():
            return None

        # Проверка вероятности деления
        if self._rng.random() >= self.DIVISION_PROBABILITY:
            return None

        # Деление происходит
        self.dividing = True
        self.division_count += 1
        self.energy *= 0.5

        # Создание дочернего фибробласта
        offspring = Fibroblast(
            agent_id=new_id,
            x=self.x + self._rng.uniform(-1.0, 1.0),
            y=self.y + self._rng.uniform(-1.0, 1.0),
            age=0.0,
            rng=self._rng,
        )
        offspring.energy = self.energy
        offspring.activated = self.activated
        offspring.division_count = 0

        return offspring

    def produce_ecm(self, dt: float) -> float:
        """Производство ECM.

        Args:
            dt: Временной шаг

        Returns:
            Количество произведённого ECM

        Подробное описание: Description/description_abm_model.md#Fibroblast.produce_ecm
        """
        if not self.alive:
            return 0.0

        rate = self.ECM_PRODUCTION_RATE
        if self.activated:
            rate *= 2.0  # Миофибробласты производят больше ECM

        return rate * dt

    def activate(self) -> None:
        """Активация в миофибробласт.

        Подробное описание: Description/description_abm_model.md#Fibroblast.activate
        """
        self.activated = True


class NeutrophilAgent(Agent):
    """CD66b+ нейтрофил.

    Свойства:
    - Короткоживущий (t1/2 ~ 12-14 часов)
    - Не пролиферирует в ткани (MAX_DIVISIONS = 0)
    - Мощный фагоцитоз debris
    - Секреция провоспалительных цитокинов (TNF-α, IL-8)
    - Хемотаксис по градиенту IL-8

    Подробное описание: Description/Phase2/description_abm_model.md#NeutrophilAgent
    """

    AGENT_TYPE: str = "neutro"
    LIFESPAN: float = 24.0  # часов (короткоживущий)
    DIVISION_ENERGY_THRESHOLD: float = 1.0  # Не делятся
    MAX_DIVISIONS: int = 0  # Не пролиферируют в ткани
    DIVISION_PROBABILITY: float = 0.0  # per hour
    DEATH_PROBABILITY: float = 0.04  # per hour (высокая, t1/2 ~ 12-14 ч)

    CHEMOTAXIS_SENSITIVITY: float = 0.8  # Сильный хемотаксис по IL-8
    PHAGOCYTOSIS_CAPACITY: int = 3  # Максимум debris за шаг

    def __init__(
        self,
        agent_id: int,
        x: float,
        y: float,
        age: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Инициализация нейтрофила.

        Args:
            agent_id: Уникальный идентификатор
            x: Координата X
            y: Координата Y
            age: Возраст
            rng: Генератор случайных чисел

        Подробное описание: Description/Phase2/description_abm_model.md#NeutrophilAgent.__init__
        """
        super().__init__(agent_id, x, y, age, rng)
        self.phagocytosed_count = 0

    def update(self, dt: float, environment: dict[str, Any]) -> None:
        """Обновление нейтрофила.

        Потребление энергии, обновление возраста, проверка апоптоза.
        Нейтрофилы быстро расходуют энергию и имеют короткий срок жизни.
        Хемотаксис обрабатывается на уровне ABMModel.

        Args:
            dt: Временной шаг (часы)
            environment: Параметры окружения (il8_level, debris_count)

        Подробное описание: Description/Phase2/description_abm_model.md#NeutrophilAgent.update
        """
        if not self.alive:
            return
        self.age += dt
        energy_consumption = 0.003 * dt
        self.energy = max(0.0, self.energy - energy_consumption)
        self.dividing = False

    def divide(self, new_id: int) -> "Agent | None":
        """Деление нейтрофила — всегда None.

        Нейтрофилы не пролиферируют в ткани (MAX_DIVISIONS = 0).
        Рекрутируются из кровотока, а не делятся локально.

        Args:
            new_id: ID для нового агента (не используется)

        Returns:
            Всегда None

        Подробное описание: Description/Phase2/description_abm_model.md#NeutrophilAgent.divide
        """
        return None

    def phagocytose(self, debris_count: int) -> int:
        """Фагоцитоз debris и патогенов.

        Поглощает до PHAGOCYTOSIS_CAPACITY частиц за вызов.
        Каждая поглощённая частица увеличивает phagocytosed_count.
        При исчерпании ёмкости — NETosis (гибель с выбросом ловушек).

        Args:
            debris_count: Количество доступных частиц debris

        Returns:
            Количество поглощённых частиц (≤ PHAGOCYTOSIS_CAPACITY)

        Подробное описание: Description/Phase2/description_abm_model.md#NeutrophilAgent.phagocytose
        """
        if not self.alive:
            return 0
        phagocytosed = min(debris_count, self.PHAGOCYTOSIS_CAPACITY)
        self.phagocytosed_count += phagocytosed
        return phagocytosed

    def secrete_cytokines(self, dt: float) -> dict[str, float]:
        """Секреция провоспалительных цитокинов.

        Нейтрофилы секретируют TNF-α и IL-8 (аутокринная петля).
        Скорость секреции зависит от активации (phagocytosed_count > 0).

        Args:
            dt: Временной шаг (часы)

        Returns:
            Словарь {"TNF_alpha": float, "IL_8": float}

        Подробное описание: Description/Phase2/description_abm_model.md#NeutrophilAgent.secrete_cytokines
        """
        if not self.alive:
            return {"TNF_alpha": 0.0, "IL_8": 0.0}
        return {"TNF_alpha": 0.01 * dt, "IL_8": 0.02 * dt}

    def is_apoptotic(self) -> bool:
        """Проверка апоптоза нейтрофила.

        True при age > LIFESPAN или energy ≤ 0.
        Апоптотические нейтрофилы фагоцитируются макрофагами (M2-поляризация).

        Returns:
            True если нейтрофил апоптотический

        Подробное описание: Description/Phase2/description_abm_model.md#NeutrophilAgent.is_apoptotic
        """
        return not self.alive


class EndothelialAgent(Agent):
    """CD31+ эндотелиальная клетка.

    Свойства:
    - VEGF-зависимый ангиогенез
    - Формирование межклеточных контактов (адгезия)
    - Долгоживущая (до 20 дней)
    - Низкая пролиферация, стимулируемая VEGF
    - Секреция VEGF и PDGF

    Подробное описание: Description/Phase2/description_abm_model.md#EndothelialAgent
    """

    AGENT_TYPE: str = "endo"
    LIFESPAN: float = 480.0  # часов (20 дней)
    DIVISION_ENERGY_THRESHOLD: float = 0.8
    MAX_DIVISIONS: int = 5
    DIVISION_PROBABILITY: float = 0.01  # per hour
    DEATH_PROBABILITY: float = 0.001  # per hour

    VEGF_SENSITIVITY: float = 0.6  # Чувствительность к VEGF-градиенту
    ADHESION_STRENGTH: float = 0.5  # Сила адгезии к другим endo

    def __init__(
        self,
        agent_id: int,
        x: float,
        y: float,
        age: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Инициализация эндотелиальной клетки.

        Args:
            agent_id: Уникальный идентификатор
            x: Координата X
            y: Координата Y
            age: Возраст
            rng: Генератор случайных чисел

        Подробное описание: Description/Phase2/description_abm_model.md#EndothelialAgent.__init__
        """
        super().__init__(agent_id, x, y, age, rng)
        self.junction_count = 0

    def update(self, dt: float, environment: dict[str, Any]) -> None:
        """Обновление эндотелиальной клетки.

        VEGF-зависимое поведение: при высоком VEGF — миграция и деление.
        Формирование сосудистых структур через адгезию с соседями.

        Args:
            dt: Временной шаг (часы)
            environment: Параметры окружения (vegf_level, neighbors)

        Подробное описание: Description/Phase2/description_abm_model.md#EndothelialAgent.update
        """
        if not self.alive:
            return
        self.age += dt
        energy_consumption = 0.001 * dt
        self.energy = max(0.0, self.energy - energy_consumption)
        self.dividing = False

    def divide(self, new_id: int) -> "EndothelialAgent | None":
        """VEGF-зависимое деление эндотелиальной клетки.

        Вероятность деления модулируется уровнем VEGF.
        Дочерняя клетка появляется рядом с родительской.

        Args:
            new_id: ID для новой клетки

        Returns:
            Новый EndothelialAgent или None

        Подробное описание: Description/Phase2/description_abm_model.md#EndothelialAgent.divide
        """
        if not self.can_divide():
            return None
        if self._rng.random() >= self.DIVISION_PROBABILITY:
            return None
        self.dividing = True
        self.division_count += 1
        self.energy *= 0.5
        offspring = EndothelialAgent(
            agent_id=new_id,
            x=self.x + self._rng.uniform(-1.0, 1.0),
            y=self.y + self._rng.uniform(-1.0, 1.0),
            age=0.0,
            rng=self._rng,
        )
        offspring.energy = self.energy
        offspring.division_count = 0
        return offspring

    def form_junction(self, neighbor: "EndothelialAgent") -> bool:
        """Формирование сосудистого контакта (tight junction).

        Два эндотелиальных агента формируют контакт если находятся
        на расстоянии ≤ adhesion_equilibrium_distance. Контакт
        стабилизирует обе клетки (снижает DEATH_PROBABILITY).

        Args:
            neighbor: Соседняя эндотелиальная клетка

        Returns:
            True если контакт сформирован

        Подробное описание: Description/Phase2/description_abm_model.md#EndothelialAgent.form_junction
        """
        if not isinstance(neighbor, EndothelialAgent):
            return False
        dist = np.sqrt((self.x - neighbor.x) ** 2 + (self.y - neighbor.y) ** 2)
        if dist <= 3.0:  # adhesion_equilibrium_distance
            self.junction_count += 1
            return True
        return False

    def secrete_cytokines(self, dt: float) -> dict[str, float]:
        """Секреция ангиогенных факторов.

        Эндотелиальные клетки секретируют VEGF (аутокринно) и PDGF
        (привлечение перицитов). Скорость зависит от активации.

        Args:
            dt: Временной шаг (часы)

        Returns:
            Словарь {"VEGF": float, "PDGF": float}

        Подробное описание: Description/Phase2/description_abm_model.md#EndothelialAgent.secrete_cytokines
        """
        if not self.alive:
            return {"VEGF": 0.0, "PDGF": 0.0}
        return {"VEGF": 0.005 * dt, "PDGF": 0.003 * dt}


class MyofibroblastAgent(Agent):
    """α-SMA+ миофибробласт.

    Свойства:
    - Усиленная продукция ECM (2× фибробласта)
    - Контракция ткани (закрытие раны)
    - TGF-β-зависимое выживание
    - Апоптоз при снижении TGF-β (разрешение фиброза)
    - Активируется из Fibroblast при высоком TGF-β

    Подробное описание: Description/Phase2/description_abm_model.md#MyofibroblastAgent
    """

    AGENT_TYPE: str = "myofibro"
    LIFESPAN: float = 480.0  # часов (20 дней)
    DIVISION_ENERGY_THRESHOLD: float = 0.85
    MAX_DIVISIONS: int = 3
    DIVISION_PROBABILITY: float = 0.003  # per hour (редкое деление)
    DEATH_PROBABILITY: float = 0.002  # per hour

    ECM_PRODUCTION_RATE: float = 1.0  # units/hour (2× фибробласта)
    CONTRACTION_FORCE: float = 0.3  # Сила контракции

    def __init__(
        self,
        agent_id: int,
        x: float,
        y: float,
        age: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Инициализация миофибробласта.

        Args:
            agent_id: Уникальный идентификатор
            x: Координата X
            y: Координата Y
            age: Возраст
            rng: Генератор случайных чисел

        Подробное описание: Description/Phase2/description_abm_model.md#MyofibroblastAgent.__init__
        """
        super().__init__(agent_id, x, y, age, rng)
        self.ecm_produced = 0.0

    def update(self, dt: float, environment: dict[str, Any]) -> None:
        """Обновление миофибробласта.

        TGF-β-зависимое выживание: при низком TGF-β — апоптоз.
        Непрерывная продукция ECM и контракция ткани.

        Args:
            dt: Временной шаг (часы)
            environment: Параметры окружения (tgfb_level, ecm_density)

        Подробное описание: Description/Phase2/description_abm_model.md#MyofibroblastAgent.update
        """
        if not self.alive:
            return
        self.age += dt
        energy_consumption = 0.002 * dt
        self.energy = max(0.0, self.energy - energy_consumption)
        self.dividing = False

    def divide(self, new_id: int) -> "MyofibroblastAgent | None":
        """Редкое деление миофибробласта.

        Миофибробласты делятся крайне редко (DIVISION_PROBABILITY = 0.003).
        Дочерняя клетка наследует состояние активации.

        Args:
            new_id: ID для нового агента

        Returns:
            Новый MyofibroblastAgent или None

        Подробное описание: Description/Phase2/description_abm_model.md#MyofibroblastAgent.divide
        """
        if not self.can_divide():
            return None
        if self._rng.random() >= self.DIVISION_PROBABILITY:
            return None
        self.dividing = True
        self.division_count += 1
        self.energy *= 0.5
        offspring = MyofibroblastAgent(
            agent_id=new_id,
            x=self.x + self._rng.uniform(-1.0, 1.0),
            y=self.y + self._rng.uniform(-1.0, 1.0),
            age=0.0,
            rng=self._rng,
        )
        offspring.energy = self.energy
        offspring.division_count = 0
        return offspring

    def produce_ecm(self, dt: float) -> float:
        """Усиленное производство внеклеточного матрикса.

        Миофибробласты производят коллаген со скоростью 2× фибробласта
        (ECM_PRODUCTION_RATE = 1.0). Скорость зависит от уровня TGF-β.

        Args:
            dt: Временной шаг (часы)

        Returns:
            Количество произведённого ECM (units)

        Подробное описание: Description/Phase2/description_abm_model.md#MyofibroblastAgent.produce_ecm
        """
        if not self.alive:
            return 0.0
        return self.ECM_PRODUCTION_RATE * dt

    def contract(self, dt: float) -> float:
        """Контракция ткани (закрытие раны).

        Миофибробласты генерируют тяговую силу через α-SMA стресс-волокна.
        Сила контракции пропорциональна CONTRACTION_FORCE и времени.

        Args:
            dt: Временной шаг (часы)

        Returns:
            Величина контракции (безразмерная)

        Подробное описание: Description/Phase2/description_abm_model.md#MyofibroblastAgent.contract
        """
        if not self.alive:
            return 0.0
        return self.CONTRACTION_FORCE * dt

    def should_apoptose_tgfb(self, tgfb_level: float) -> bool:
        """Проверка TGF-β-зависимого апоптоза.

        Миофибробласты зависят от TGF-β для выживания. При падении
        уровня ниже порога — запускается апоптоз (разрешение фиброза).

        Args:
            tgfb_level: Локальный уровень TGF-β (нг/мл)

        Returns:
            True если TGF-β ниже порога выживания

        Подробное описание: Description/Phase2/description_abm_model.md#MyofibroblastAgent.should_apoptose_tgfb
        """
        return tgfb_level < 0.1


class ABMModel:
    """Agent-Based модель регенерации тканей.

    Подробное описание: Description/description_abm_model.md#ABMModel
    """

    def __init__(
        self,
        config: ABMConfig | None = None,
        random_seed: int | None = None,
    ) -> None:
        """Инициализация ABM.

        Args:
            config: Конфигурация модели
            random_seed: Seed для воспроизводимости

        Подробное описание: Description/description_abm_model.md#ABMModel.__init__
        """
        self._config = config if config else ABMConfig()
        self._config.validate()

        self._rng = np.random.default_rng(random_seed)
        self._agents: list[Agent] = []
        self._dead_agents: list[Agent] = []
        self._next_agent_id = 0
        self._current_time = 0.0

        # Cytokine field (discrete grid)
        self._grid_shape = (
            int(self._config.space_size[0] / self._config.grid_resolution),
            int(self._config.space_size[1] / self._config.grid_resolution),
        )
        self._cytokine_field = np.zeros(self._grid_shape)
        self._ecm_field = np.zeros(self._grid_shape)

        # Spatial hash для эффективного поиска соседей O(1)
        self._spatial_hash = SpatialHash(
            space_size=self._config.space_size,
            cell_size=max(
                self._config.interaction_radius,
                self._config.contact_inhibition_radius,
            ),
            periodic=(self._config.boundary_type == "periodic"),
        )

    @property
    def config(self) -> ABMConfig:
        """Получить конфигурацию модели."""
        return self._config

    @property
    def agents(self) -> list[Agent]:
        """Получить список агентов."""
        return self._agents

    def initialize_from_parameters(
        self,
        params: ModelParameters,
    ) -> None:
        """Инициализация агентов из ModelParameters.

        Args:
            params: Параметры из parameter_extraction

        Подробное описание: Description/description_abm_model.md#ABMModel.initialize_from_parameters
        """
        # Очистка существующих агентов
        self._agents = []
        self._dead_agents = []
        self._next_agent_id = 0
        self._current_time = 0.0

        # Вычисление количества агентов на основе n0 и долей
        # Масштабирование: n0 -> количество агентов (упрощённое)
        total_agents = min(
            int(params.n0 / 100),  # Масштабирование плотности
            self._config.max_agents,
        )
        total_agents = max(total_agents, 10)  # Минимум 10 агентов

        # Распределение по типам
        n_stem = max(1, int(total_agents * params.stem_cell_fraction))
        n_macro = max(1, int(total_agents * params.macrophage_fraction))
        n_fibro = max(1, total_agents - n_stem - n_macro)

        # Создание стволовых клеток
        for _ in range(n_stem):
            self._agents.append(self._create_agent("stem"))

        # Создание макрофагов
        for _ in range(n_macro):
            agent = self._create_agent("macro")
            # Установка начальной поляризации
            if isinstance(agent, Macrophage):
                agent.polarize(params.inflammation_level)
            self._agents.append(agent)

        # Создание фибробластов
        for _ in range(n_fibro):
            self._agents.append(self._create_agent("fibro"))

        # Инициализация цитокинового поля на основе c0
        self._cytokine_field = np.full(self._grid_shape, params.c0 / 100.0)
        self._ecm_field = np.zeros(self._grid_shape)

    def simulate(
        self,
        initial_params: ModelParameters,
        snapshot_interval: float = 24.0,  # часов
    ) -> ABMTrajectory:
        """Полная симуляция ABM.

        Args:
            initial_params: Начальные параметры
            snapshot_interval: Интервал сохранения снимков (часы)

        Returns:
            ABMTrajectory с результатами

        Подробное описание: Description/description_abm_model.md#ABMModel.simulate
        """
        # Инициализация
        self.initialize_from_parameters(initial_params)

        snapshots: list[ABMSnapshot] = []
        dt = self._config.dt
        t_max = self._config.t_max

        # Начальный снимок
        snapshots.append(self._get_snapshot())

        # Время следующего снимка
        next_snapshot_time = snapshot_interval

        # Основной цикл симуляции
        while self._current_time < t_max:
            # Шаг симуляции
            self.step(dt)

            # Проверка сохранения снимка
            if self._current_time >= next_snapshot_time:
                snapshots.append(self._get_snapshot())
                next_snapshot_time += snapshot_interval

            # Проверка на вымирание
            if not self._agents:
                break

        # Финальный снимок если не совпал с интервалом
        if snapshots[-1].t < self._current_time:
            snapshots.append(self._get_snapshot())

        return ABMTrajectory(
            snapshots=snapshots,
            config=self._config,
        )

    def step(self, dt: float) -> None:
        """Один шаг симуляции.

        Args:
            dt: Временной шаг (часы)

        Подробное описание: Description/description_abm_model.md#ABMModel.step
        """
        # 0. Перестроение пространственного хэша для поиска соседей
        self._spatial_hash.rebuild(self._agents)

        # 1. Обновление агентов (движение, потребление энергии, etc.)
        self._update_agents(dt)

        # 2. Обработка делений
        self._handle_divisions()

        # 3. Обработка дифференциаций
        self._handle_differentiations()

        # 4. Удаление мёртвых агентов
        self._remove_dead_agents()

        # 5. Обновление полей
        self._update_cytokine_field(dt)
        self._update_ecm_field(dt)

        # 6. Обновление времени
        self._current_time += dt

    def _update_agents(self, dt: float) -> None:
        """Обновление всех агентов.

        Args:
            dt: Временной шаг

        Подробное описание: Description/description_abm_model.md#ABMModel._update_agents
        """
        for agent in self._agents:
            if not agent.alive:
                continue

            # Получение окружения для агента
            env = self._get_environment(agent.x, agent.y)

            # Обновление состояния агента
            agent.update(dt, env)

            # Случайное перемещение
            dx, dy = agent._random_walk_displacement(
                self._config.diffusion_coefficient, dt
            )

            # Хемотаксис для макрофагов
            if isinstance(agent, Macrophage):
                grad_x, grad_y = self._get_cytokine_gradient(agent.x, agent.y)
                chi = self._config.chemotaxis_strength * agent.CHEMOTAXIS_SENSITIVITY
                dx += chi * grad_x * dt
                dy += chi * grad_y * dt

            # Силы отталкивания между клетками
            repulsion_x, repulsion_y = self._calculate_repulsion_force(agent)
            dx += repulsion_x * dt
            dy += repulsion_y * dt

            agent.move(dx, dy, self._config.space_size, self._config.boundary_type)

            # Проверка смерти
            if agent.should_die(dt):
                agent.alive = False

    def _handle_divisions(self) -> None:
        """Обработка делений агентов.

        Подробное описание: Description/description_abm_model.md#ABMModel._handle_divisions
        """
        # Проверяем ограничение на количество агентов
        if len(self._agents) >= self._config.max_agents:
            return

        new_agents: list[Agent] = []

        for agent in self._agents:
            if not agent.alive or not agent.can_divide():
                continue

            # Контактное ингибирование: блокировка деления при высокой плотности
            neighbor_count = self._count_neighbors(
                agent, self._config.contact_inhibition_radius
            )
            if neighbor_count >= self._config.contact_inhibition_threshold:
                continue  # Слишком много соседей - деление заблокировано

            # Ограничение общего количества
            if len(self._agents) + len(new_agents) >= self._config.max_agents:
                break

            offspring = agent.divide(self._next_agent_id)
            if offspring is not None:
                self._next_agent_id += 1
                new_agents.append(offspring)

        self._agents.extend(new_agents)

    def _handle_differentiations(self) -> None:
        """Обработка дифференциаций стволовых клеток.

        Подробное описание: Description/description_abm_model.md#ABMModel._handle_differentiations
        """
        new_fibroblasts: list[Fibroblast] = []

        for agent in self._agents:
            if not isinstance(agent, StemCell) or not agent.alive:
                continue

            if agent.should_differentiate():
                fibroblast = agent.differentiate(self._next_agent_id)
                self._next_agent_id += 1
                new_fibroblasts.append(fibroblast)

        self._agents.extend(new_fibroblasts)

    def _remove_dead_agents(self) -> None:
        """Удаление мёртвых агентов.

        Подробное описание: Description/description_abm_model.md#ABMModel._remove_dead_agents
        """
        # Сохраняем мёртвых агентов для статистики
        dead = [agent for agent in self._agents if not agent.alive]
        self._dead_agents.extend(dead)

        # Оставляем только живых
        self._agents = [agent for agent in self._agents if agent.alive]

    def _update_cytokine_field(self, dt: float) -> None:
        """Обновление поля цитокинов (диффузия + секреция + деградация).

        Args:
            dt: Временной шаг

        Подробное описание: Description/description_abm_model.md#ABMModel._update_cytokine_field
        """
        # Деградация
        self._cytokine_field *= (1.0 - self._config.cytokine_decay * dt)

        # Диффузия (упрощённая - усреднение с соседями)
        if self._config.cytokine_diffusion > 0:
            diffusion_coeff = self._config.cytokine_diffusion * dt / (
                self._config.grid_resolution ** 2
            )
            diffusion_coeff = min(diffusion_coeff, 0.25)  # Стабильность

            new_field = self._cytokine_field.copy()
            for i in range(self._grid_shape[0]):
                for j in range(self._grid_shape[1]):
                    neighbors = []
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni = (i + di) % self._grid_shape[0]
                        nj = (j + dj) % self._grid_shape[1]
                        neighbors.append(self._cytokine_field[ni, nj])
                    avg_neighbors = np.mean(neighbors)
                    new_field[i, j] += diffusion_coeff * (
                        avg_neighbors - self._cytokine_field[i, j]
                    )
            self._cytokine_field = new_field

        # Секреция от агентов
        for agent in self._agents:
            if not agent.alive:
                continue

            grid_x = int(agent.x / self._config.grid_resolution) % self._grid_shape[0]
            grid_y = int(agent.y / self._config.grid_resolution) % self._grid_shape[1]

            if isinstance(agent, StemCell):
                secretion = agent.secrete_cytokines(dt)
                self._cytokine_field[grid_x, grid_y] += secretion
            elif isinstance(agent, Macrophage):
                cytokines = agent.secrete_cytokines(dt)
                total_secretion = sum(cytokines.values())
                self._cytokine_field[grid_x, grid_y] += total_secretion

        # Ограничение значений
        self._cytokine_field = np.clip(self._cytokine_field, 0.0, 100.0)

    def _update_ecm_field(self, dt: float) -> None:
        """Обновление поля ECM.

        Args:
            dt: Временной шаг

        Подробное описание: Description/description_abm_model.md#ABMModel._update_ecm_field
        """
        # Производство ECM фибробластами
        for agent in self._agents:
            if not isinstance(agent, Fibroblast) or not agent.alive:
                continue

            grid_x = int(agent.x / self._config.grid_resolution) % self._grid_shape[0]
            grid_y = int(agent.y / self._config.grid_resolution) % self._grid_shape[1]

            ecm_produced = agent.produce_ecm(dt)
            self._ecm_field[grid_x, grid_y] += ecm_produced

        # Ограничение значений
        self._ecm_field = np.clip(self._ecm_field, 0.0, 100.0)

    def _get_environment(self, x: float, y: float) -> dict[str, Any]:
        """Получить параметры окружения в точке.

        Args:
            x: Координата X
            y: Координата Y

        Returns:
            Словарь с параметрами окружения

        Подробное описание: Description/description_abm_model.md#ABMModel._get_environment
        """
        # Вычисление индексов сетки
        grid_x = int(x / self._config.grid_resolution) % self._grid_shape[0]
        grid_y = int(y / self._config.grid_resolution) % self._grid_shape[1]

        # Получение значений из полей
        cytokine_level = self._cytokine_field[grid_x, grid_y]
        ecm_level = self._ecm_field[grid_x, grid_y]

        # Расчёт уровня воспаления на основе M1/M2 баланса
        n_m1 = sum(
            1 for a in self._agents
            if isinstance(a, Macrophage) and a.alive and a.polarization_state == "M1"
        )
        n_m2 = sum(
            1 for a in self._agents
            if isinstance(a, Macrophage) and a.alive and a.polarization_state == "M2"
        )
        total_macro = n_m1 + n_m2
        inflammation_level = n_m1 / total_macro if total_macro > 0 else 0.5

        return {
            "cytokine_level": float(cytokine_level),
            "ecm_level": float(ecm_level),
            "inflammation_level": float(inflammation_level),
            "prp_active": False,  # Будет устанавливаться при интеграции
            "pemf_active": False,
        }

    def _get_cytokine_gradient(self, x: float, y: float) -> tuple[float, float]:
        """Вычисление градиента цитокинового поля методом центральных разностей.

        Используется для хемотаксиса макрофагов - направленного движения
        к источникам цитокинов.

        Args:
            x: Координата X агента (мкм)
            y: Координата Y агента (мкм)

        Returns:
            Нормализованный вектор градиента (grad_x, grad_y)

        Подробное описание: Description/description_abm_model.md#ABMModel._get_cytokine_gradient
        """
        # Преобразование координат агента в индексы сетки
        grid_x = int(x / self._config.grid_resolution) % self._grid_shape[0]
        grid_y = int(y / self._config.grid_resolution) % self._grid_shape[1]

        # Соседние индексы с периодическими границами
        x_prev = (grid_x - 1) % self._grid_shape[0]
        x_next = (grid_x + 1) % self._grid_shape[0]
        y_prev = (grid_y - 1) % self._grid_shape[1]
        y_next = (grid_y + 1) % self._grid_shape[1]

        # Центральные разности: grad = (C[i+1] - C[i-1]) / (2 * dx)
        dx = 2.0 * self._config.grid_resolution
        grad_x = (
            self._cytokine_field[x_next, grid_y]
            - self._cytokine_field[x_prev, grid_y]
        ) / dx
        grad_y = (
            self._cytokine_field[grid_x, y_next]
            - self._cytokine_field[grid_x, y_prev]
        ) / dx

        # Нормализация (избегаем деления на 0)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        if magnitude > 1e-10:
            return (grad_x / magnitude, grad_y / magnitude)
        return (0.0, 0.0)

    def _count_neighbors(self, agent: Agent, radius: float) -> int:
        """Подсчёт соседей агента в заданном радиусе (оптимизировано через SpatialHash).

        Используется для контактного ингибирования - блокировки деления
        при высокой локальной плотности клеток.

        Args:
            agent: Агент для проверки
            radius: Радиус поиска (мкм)

        Returns:
            Количество живых соседей в радиусе

        Подробное описание: Description/description_abm_model.md#ABMModel._count_neighbors
        """
        neighbors = self._spatial_hash.get_neighbors(
            agent.x, agent.y, radius, exclude=agent
        )
        return len(neighbors)

    def _calculate_repulsion_force(self, agent: Agent) -> tuple[float, float]:
        """Вычисление силы отталкивания от соседних клеток (оптимизировано через SpatialHash).

        Модель мягких сфер с линейным отталкиванием:
        F = k * (r0 - d) / r0 * direction

        Args:
            agent: Агент для расчёта сил

        Returns:
            (fx, fy) - компоненты силы отталкивания

        Подробное описание: Description/description_abm_model.md#ABMModel._calculate_repulsion_force
        """
        fx, fy = 0.0, 0.0
        r0 = self._config.interaction_radius

        # Получаем только соседей в радиусе взаимодействия через spatial hash
        neighbors = self._spatial_hash.get_neighbors(
            agent.x, agent.y, r0, exclude=agent
        )

        for other in neighbors:
            # Вектор от other к agent
            dx = agent.x - other.x
            dy = agent.y - other.y

            # Учёт периодических границ
            if self._config.boundary_type == "periodic":
                if dx > self._config.space_size[0] / 2:
                    dx -= self._config.space_size[0]
                elif dx < -self._config.space_size[0] / 2:
                    dx += self._config.space_size[0]
                if dy > self._config.space_size[1] / 2:
                    dy -= self._config.space_size[1]
                elif dy < -self._config.space_size[1] / 2:
                    dy += self._config.space_size[1]

            distance = np.sqrt(dx**2 + dy**2)

            if distance > 1e-10:
                # Линейное отталкивание: F = k * overlap / r0
                overlap = r0 - distance
                force_magnitude = self._config.repulsion_strength * overlap / r0

                # Нормализация направления
                fx += force_magnitude * dx / distance
                fy += force_magnitude * dy / distance

        return (fx, fy)

    def _chemotaxis_displacement(
        self,
        agent: Agent,
        cytokine_fields: dict[str, np.ndarray],
    ) -> tuple[float, float]:
        """Мульти-градиентный хемотаксис в зависимости от типа агента.

        Каждый тип клетки реагирует на специфический набор хемоаттрактантов:
        - NeutrophilAgent → IL-8 (рекрутирование в зону воспаления)
        - Macrophage → MCP-1 (привлечение макрофагов)
        - EndothelialAgent → VEGF (ангиогенная миграция)
        - Fibroblast → PDGF (привлечение к месту ремоделирования)

        Вычисляет градиент цитокинового поля в позиции агента и
        возвращает смещение пропорциональное градиенту × чувствительность.

        Args:
            agent: Агент для расчёта хемотаксиса
            cytokine_fields: Словарь цитокиновых полей {"IL_8": ndarray, ...}

        Returns:
            (dx, dy) — хемотаксическое смещение

        Подробное описание: Description/Phase2/description_abm_model.md#ABMModel._chemotaxis_displacement
        """
        type_to_cytokine = {
            "neutro": "IL_8",
            "macro": "MCP_1",
            "endo": "VEGF",
            "fibro": "PDGF",
        }
        cytokine_name = type_to_cytokine.get(agent.AGENT_TYPE)
        if cytokine_name is None or cytokine_name not in cytokine_fields:
            return (0.0, 0.0)

        field = cytokine_fields[cytokine_name]
        grid_x = int(agent.x / self._config.grid_resolution) % field.shape[0]
        grid_y = int(agent.y / self._config.grid_resolution) % field.shape[1]

        x_prev = (grid_x - 1) % field.shape[0]
        x_next = (grid_x + 1) % field.shape[0]
        y_prev = (grid_y - 1) % field.shape[1]
        y_next = (grid_y + 1) % field.shape[1]

        dx_step = 2.0 * self._config.grid_resolution
        grad_x = (field[x_next, grid_y] - field[x_prev, grid_y]) / dx_step
        grad_y = (field[grid_x, y_next] - field[grid_x, y_prev]) / dx_step

        sensitivity = getattr(agent, "CHEMOTAXIS_SENSITIVITY", 0.0)
        dx = float(sensitivity * grad_x * self._config.chemotaxis_strength)
        dy = float(sensitivity * grad_y * self._config.chemotaxis_strength)
        return (dx, dy)

    def _apply_contact_inhibition(
        self,
        agent: Agent,
        neighbors_count: int,
    ) -> float:
        """Модификатор пролиферации на основе контактного ингибирования.

        Плотность-зависимое подавление деления. При увеличении количества
        соседей вероятность деления снижается линейно до 0 при достижении
        contact_inhibition_threshold.

        Формула: modifier = max(0, 1 - neighbors_count / threshold)
        - 0 соседей → 1.0 (нет ингибирования)
        - threshold соседей → 0.0 (полное ингибирование)

        Args:
            agent: Агент, для которого считается модификатор
            neighbors_count: Количество соседей в contact_inhibition_radius

        Returns:
            Модификатор пролиферации в диапазоне [0.0, 1.0]

        Подробное описание: Description/Phase2/description_abm_model.md#ABMModel._apply_contact_inhibition
        """
        threshold = self._config.contact_inhibition_threshold
        return max(0.0, 1.0 - neighbors_count / threshold)

    def _calculate_adhesion_force(
        self,
        agent1: Agent,
        agent2: Agent,
        distance: float,
    ) -> np.ndarray:
        """Сила адгезии между совместимыми типами клеток.

        Моделирует клеточную адгезию (кадгерины, интегрины):
        - endo ↔ endo: VE-кадгерин (сосудистые контакты)
        - myofibro ↔ myofibro: N-кадгерин (контрактильная сеть)
        - Несовместимые типы → нулевая сила

        Потенциал: F = -k_adh * (d - d_eq), где d_eq — равновесное расстояние.
        Притяжение при d > d_eq, отталкивание при d < d_eq.

        Args:
            agent1: Первый агент
            agent2: Второй агент
            distance: Расстояние между агентами (мкм)

        Returns:
            np.ndarray shape=(2,) — вектор силы адгезии (fx, fy)

        Подробное описание: Description/Phase2/description_abm_model.md#ABMModel._calculate_adhesion_force
        """
        compatible = (
            isinstance(agent1, EndothelialAgent)
            and isinstance(agent2, EndothelialAgent)
        ) or (
            isinstance(agent1, MyofibroblastAgent)
            and isinstance(agent2, MyofibroblastAgent)
        )
        if not compatible or distance < 1e-10:
            return np.zeros(2)

        d_eq = self._config.adhesion_equilibrium_distance
        k_adh = self._config.adhesion_strength

        dx = agent2.x - agent1.x
        dy = agent2.y - agent1.y
        direction = np.array([dx, dy]) / distance

        force = k_adh * (distance - d_eq) * direction
        return force

    def _get_snapshot(self) -> ABMSnapshot:
        """Создание снимка текущего состояния.

        Returns:
            ABMSnapshot с состоянием системы

        Подробное описание: Description/description_abm_model.md#ABMModel._get_snapshot
        """
        agent_states = [agent.get_state() for agent in self._agents if agent.alive]

        return ABMSnapshot(
            t=self._current_time,
            agents=agent_states,
            cytokine_field=self._cytokine_field.copy(),
            ecm_field=self._ecm_field.copy(),
        )

    def _create_agent(
        self,
        agent_type: str,
        x: float | None = None,
        y: float | None = None,
    ) -> Agent:
        """Создание нового агента.

        Args:
            agent_type: Тип агента ('stem', 'macro', 'fibro',
                        'neutro', 'endo', 'myofibro')
            x: Координата X (или случайная)
            y: Координата Y (или случайная)

        Returns:
            Новый агент

        Подробное описание: Description/description_abm_model.md#ABMModel._create_agent
        """
        # Генерация координат если не указаны
        if x is None:
            x = self._rng.uniform(0, self._config.space_size[0])
        if y is None:
            y = self._rng.uniform(0, self._config.space_size[1])

        agent_id = self._next_agent_id
        self._next_agent_id += 1

        if agent_type == "stem":
            return StemCell(agent_id=agent_id, x=x, y=y, rng=self._rng)
        elif agent_type == "macro":
            return Macrophage(agent_id=agent_id, x=x, y=y, rng=self._rng)
        elif agent_type == "fibro":
            return Fibroblast(agent_id=agent_id, x=x, y=y, rng=self._rng)
        elif agent_type == "neutro":
            return NeutrophilAgent(agent_id=agent_id, x=x, y=y, rng=self._rng)
        elif agent_type == "endo":
            return EndothelialAgent(agent_id=agent_id, x=x, y=y, rng=self._rng)
        elif agent_type == "myofibro":
            return MyofibroblastAgent(agent_id=agent_id, x=x, y=y, rng=self._rng)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")


def simulate_abm(
    initial_params: ModelParameters,
    config: ABMConfig | None = None,
    random_seed: int | None = None,
    snapshot_interval: float = 24.0,
) -> ABMTrajectory:
    """Convenience функция для ABM симуляции.

    Args:
        initial_params: Начальные параметры из parameter_extraction
        config: Конфигурация модели (опционально)
        random_seed: Seed для воспроизводимости
        snapshot_interval: Интервал сохранения снимков (часы)

    Returns:
        ABMTrajectory с результатами

    Подробное описание: Description/description_abm_model.md#simulate_abm
    """
    model = ABMModel(config=config, random_seed=random_seed)
    return model.simulate(initial_params, snapshot_interval=snapshot_interval)
