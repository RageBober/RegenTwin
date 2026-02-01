"""Agent-Based модель клеточной динамики для регенерации тканей.

Моделирование дискретных клеточных событий на микроуровне:
- Пространственное движение (random walk + chemotaxis)
- Деление и гибель клеток
- Взаимодействия между агентами
- Типы агентов: StemCell (CD34+), Macrophage (CD14+/CD68+), Fibroblast

Подробное описание: Description/description_abm_model.md
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

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
    max_agents: int = 10000

    # Параметры движения
    diffusion_coefficient: float = 1.0  # мкм²/час
    chemotaxis_strength: float = 0.1  # Сила хемотаксиса

    # Параметры взаимодействий
    interaction_radius: float = 5.0  # мкм
    contact_inhibition_radius: float = 2.0  # мкм

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
            agent_type: Тип агента ('stem', 'macro', 'fibro')
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
