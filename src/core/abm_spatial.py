"""Расширенные пространственные и клеточные механики для ABM модели.

Включает:
- PlateletAgent: тромбоцит, дегрануляция, выброс факторов роста
- ChemotaxisEngine: градиентное направленное движение
- ContactInhibitionEngine: подавление деления при высокой плотности
- EfferocytosisEngine: фагоцитоз апоптотических нейтрофилов макрофагами
- MechanotransductionEngine: механический стресс → активация миофибробластов
- MultiCytokineField: раздельные поля для TNF, IL-10, PDGF, VEGF, TGF-β, MCP-1, IL-8
- KDTreeNeighborSearch: адаптер cKDTree с единым интерфейсом
- SubcyclingManager: разные dt для полей и агентов

Подробное описание: Description/Phase2/description_abm_extended.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.core.abm_model import (
    ABMConfig,
    Agent,
    Fibroblast,
    KDTreeSpatialIndex,
    Macrophage,
    NeutrophilAgent,
)

# =============================================================================
# PlateletAgent
# =============================================================================


class PlateletAgent(Agent):
    """Тромбоцит — дегрануляция и выброс факторов роста.

    Свойства:
    - Анукленная клетка (не делится)
    - Короткоживущий (t₁/₂ ≈ 48ч)
    - Дегрануляция α-гранул → PDGF, TGF-β, VEGF
    - Формирование первичного тромба

    Подробное описание: Description/Phase2/description_abm_extended.md#PlateletAgent
    """

    AGENT_TYPE: str = "platelet"
    LIFESPAN: float = 72.0  # часы (3 дня)
    MAX_DIVISIONS: int = 0
    DIVISION_PROBABILITY: float = 0.0
    DEATH_PROBABILITY: float = 0.014  # t₁/₂ ≈ 48ч
    DIVISION_ENERGY_THRESHOLD: float = 1.0  # Недостижимый порог

    DEGRANULATION_RATE: float = 0.05  # 1/час
    PDGF_RELEASE_RATE: float = 0.02  # нг/мл/час
    TGFB_RELEASE_RATE: float = 0.015
    VEGF_RELEASE_RATE: float = 0.01

    def __init__(
        self,
        agent_id: int,
        x: float,
        y: float,
        age: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(agent_id=agent_id, x=x, y=y, age=age, rng=rng)
        self.degranulated: bool = False
        self.factors_released: dict[str, float] = {
            "PDGF": 0.0,
            "TGFb": 0.0,
            "VEGF": 0.0,
        }

    def update(self, dt: float, environment: dict[str, Any]) -> None:
        """Обновление состояния тромбоцита.

        Подробное описание: Description/Phase2/description_abm_extended.md#PlateletAgent
        """
        if not self.alive:
            return
        self.age += dt
        self.energy = max(0.0, self.energy - 0.005 * dt)
        thrombin = environment.get("thrombin", 0.0)
        if thrombin > 0.1 and not self.degranulated:
            self.degranulate(dt)

    def divide(self, new_id: int) -> Agent | None:  # noqa: ARG002
        """Тромбоциты не делятся.

        Подробное описание: Description/Phase2/description_abm_extended.md#PlateletAgent
        """
        return None

    def degranulate(self, dt: float) -> dict[str, float]:
        """Дегрануляция α-гранул → выброс PDGF, TGF-β, VEGF.

        Подробное описание: Description/Phase2/description_abm_extended.md#PlateletAgent
        """
        self.degranulated = True
        result = {
            "PDGF": self.PDGF_RELEASE_RATE * dt,
            "TGFb": self.TGFB_RELEASE_RATE * dt,
            "VEGF": self.VEGF_RELEASE_RATE * dt,
        }
        self.factors_released["PDGF"] += result["PDGF"]
        self.factors_released["TGFb"] += result["TGFb"]
        self.factors_released["VEGF"] += result["VEGF"]
        return result

    def release_factors(self, dt: float) -> dict[str, float]:
        """Постепенный выброс факторов роста после дегрануляции.

        Подробное описание: Description/Phase2/description_abm_extended.md#PlateletAgent
        """
        if not self.degranulated:
            return {"PDGF": 0.0, "TGFb": 0.0, "VEGF": 0.0}
        result = {
            "PDGF": self.PDGF_RELEASE_RATE * dt,
            "TGFb": self.TGFB_RELEASE_RATE * dt,
            "VEGF": self.VEGF_RELEASE_RATE * dt,
        }
        self.factors_released["PDGF"] += result["PDGF"]
        self.factors_released["TGFb"] += result["TGFb"]
        self.factors_released["VEGF"] += result["VEGF"]
        return result

    def secrete_cytokines(self, dt: float) -> dict[str, float]:
        """Секреция цитокинов (интерфейс совместимый с ABMModel).

        Подробное описание: Description/Phase2/description_abm_extended.md#PlateletAgent
        """
        if not self.degranulated:
            return self.degranulate(dt)
        return self.release_factors(dt)


# =============================================================================
# ChemotaxisEngine
# =============================================================================


class ChemotaxisEngine:
    """Движок градиентного хемотаксиса (biased random walk).

    Подробное описание: Description/Phase2/description_abm_extended.md#ChemotaxisEngine
    """

    AGENT_ATTRACTANT_MAP: dict[str, str] = {
        "neutro": "IL8",
        "macro": "MCP1",
        "endo": "VEGF",
        "fibro": "PDGF",
        "platelet": "TGFb",
    }

    def __init__(self, config: ABMConfig) -> None:
        self._config = config

    def compute_displacement(
        self,
        agent: Agent,
        cytokine_fields: dict[str, np.ndarray],
        dt: float,
    ) -> tuple[float, float]:
        """Вычисление хемотаксисного смещения агента.

        Подробное описание: Description/Phase2/description_abm_extended.md#ChemotaxisEngine
        """
        attractant = self.AGENT_ATTRACTANT_MAP.get(agent.AGENT_TYPE)
        if attractant is None or attractant not in cytokine_fields:
            return (0.0, 0.0)
        field = cytokine_fields[attractant]
        gx, gy = self._compute_gradient(
            field,
            agent.x,
            agent.y,
            self._config.grid_resolution,
        )
        mag = (gx**2 + gy**2) ** 0.5
        if mag < 1e-12:
            return (0.0, 0.0)
        dx = (gx / mag) * self._config.chemotaxis_strength * dt
        dy = (gy / mag) * self._config.chemotaxis_strength * dt
        return (float(dx), float(dy))

    def _compute_gradient(
        self,
        field: np.ndarray,
        x: float,
        y: float,
        grid_resolution: float,
    ) -> tuple[float, float]:
        """Центральные разности для градиента 2D поля.

        Подробное описание: Description/Phase2/description_abm_extended.md#ChemotaxisEngine
        """
        i = int(x / grid_resolution)
        j = int(y / grid_resolution)
        i = max(1, min(i, field.shape[0] - 2))
        j = max(1, min(j, field.shape[1] - 2))
        gx = (field[i + 1, j] - field[i - 1, j]) / (2.0 * grid_resolution)
        gy = (field[i, j + 1] - field[i, j - 1]) / (2.0 * grid_resolution)
        return (float(gx), float(gy))


# =============================================================================
# ContactInhibitionEngine
# =============================================================================


class ContactInhibitionEngine:
    """Подавление деления при высокой локальной плотности клеток.

    Подробное описание: Description/Phase2/description_abm_extended.md#ContactInhibitionEngine
    """

    def __init__(self, threshold: int, radius: float) -> None:
        self.threshold = threshold
        self.radius = radius

    def compute_modifier(self, neighbor_count: int) -> float:
        """Множитель вероятности деления: max(0, 1 - n/threshold).

        Подробное описание: Description/Phase2/description_abm_extended.md#ContactInhibitionEngine
        """
        return max(0.0, 1.0 - neighbor_count / self.threshold)

    def should_block_division(
        self,
        agent: Agent,  # noqa: ARG002
        neighbor_count: int,
    ) -> bool:
        """Блокировать ли деление: neighbor_count >= threshold.

        Подробное описание: Description/Phase2/description_abm_extended.md#ContactInhibitionEngine
        """
        return neighbor_count >= self.threshold


# =============================================================================
# EfferocytosisEngine
# =============================================================================


class EfferocytosisEngine:
    """Эффероцитоз — фагоцитоз апоптотических нейтрофилов макрофагами.

    Подробное описание: Description/Phase2/description_abm_extended.md#EfferocytosisEngine
    """

    def __init__(self, il10_release_rate: float = 0.05) -> None:
        self.il10_release_rate = il10_release_rate

    def process(
        self,
        macrophage: Macrophage,
        apoptotic_neutrophils: list[NeutrophilAgent],
    ) -> dict[str, float]:
        """Обработка эффероцитоза: фагоцитоз → IL-10 + сдвиг поляризации.

        Подробное описание: Description/Phase2/description_abm_extended.md#EfferocytosisEngine
        """
        if not apoptotic_neutrophils:
            return {"IL10": 0.0, "phagocytosed": 0}
        count = min(len(apoptotic_neutrophils), macrophage.PHAGOCYTOSIS_CAPACITY)
        for n in apoptotic_neutrophils[:count]:
            n.alive = False
        # Dual-mode polarization shift toward M2
        if isinstance(macrophage.polarization_state, str):
            macrophage.polarization_state = "M2"
        else:
            macrophage.polarization_state = max(0.0, macrophage.polarization_state - 0.1 * count)
        return {"IL10": count * self.il10_release_rate, "phagocytosed": count}


# =============================================================================
# MechanotransductionEngine
# =============================================================================


class MechanotransductionEngine:
    """Механотрансдукция — мех. стресс → активация миофибробластов.

    Подробное описание: Description/Phase2/description_abm_extended.md#MechanotransductionEngine
    """

    def __init__(
        self,
        stress_threshold: float = 0.5,
        activation_probability: float = 0.01,
    ) -> None:
        self.stress_threshold = stress_threshold
        self.activation_probability = activation_probability

    def compute_stress(
        self,
        agent: Agent,  # noqa: ARG002
        neighbors: list[Agent],
        ecm_density: float,
    ) -> float:
        """Вычисление механического стресса на агента.

        Подробное описание: Description/Phase2/description_abm_extended.md#MechanotransductionEngine
        """
        stress = len(neighbors) * 0.1 * ecm_density + ecm_density * 0.5
        return max(0.0, float(stress))

    def should_activate(
        self,
        fibroblast: Fibroblast,
        stress: float,
    ) -> bool:
        """Должен ли фибробласт активироваться в миофибробласт.

        Подробное описание: Description/Phase2/description_abm_extended.md#MechanotransductionEngine
        """
        if stress <= self.stress_threshold:
            return False
        return bool(fibroblast._rng.random() < self.activation_probability)


# =============================================================================
# MultiCytokineField
# =============================================================================


class MultiCytokineField:
    """Раздельные 2D поля для каждого цитокина.

    Подробное описание: Description/Phase2/description_abm_extended.md#MultiCytokineField
    """

    CYTOKINE_NAMES: list[str] = [
        "TNF",
        "IL10",
        "PDGF",
        "VEGF",
        "TGFb",
        "MCP1",
        "IL8",
    ]

    def __init__(
        self,
        grid_shape: tuple[int, int],
        cytokine_names: list[str] | None = None,
    ) -> None:
        names = cytokine_names if cytokine_names is not None else self.CYTOKINE_NAMES
        self.fields: dict[str, np.ndarray] = {name: np.zeros(grid_shape) for name in names}

    def update(
        self,
        dt: float,
        agents: list[Agent],  # noqa: ARG002
        config: ABMConfig,
    ) -> None:
        """Обновление всех полей: диффузия, распад, секреция.

        Подробное описание: Description/Phase2/description_abm_extended.md#MultiCytokineField
        """
        res = config.grid_resolution
        for _name, field in self.fields.items():
            # Decay
            field *= 1.0 - config.cytokine_decay * dt
            # Diffusion (5-point Laplacian)
            if config.cytokine_diffusion > 0 and field.shape[0] > 2 and field.shape[1] > 2:
                laplacian = np.zeros_like(field)
                laplacian[1:-1, 1:-1] = (
                    field[2:, 1:-1]
                    + field[:-2, 1:-1]
                    + field[1:-1, 2:]
                    + field[1:-1, :-2]
                    - 4.0 * field[1:-1, 1:-1]
                ) / (res * res)
                field += config.cytokine_diffusion * dt * laplacian
            # Non-negative clamping
            np.maximum(field, 0.0, out=field)

    def get_gradient(
        self,
        cytokine_name: str,
        x: float,
        y: float,
        grid_resolution: float,
    ) -> tuple[float, float]:
        """Градиент цитокина в точке (x, y).

        Подробное описание: Description/Phase2/description_abm_extended.md#MultiCytokineField
        """
        fld = self.fields[cytokine_name]
        i = int(x / grid_resolution)
        j = int(y / grid_resolution)
        i = max(1, min(i, fld.shape[0] - 2))
        j = max(1, min(j, fld.shape[1] - 2))
        gx = (fld[i + 1, j] - fld[i - 1, j]) / (2.0 * grid_resolution)
        gy = (fld[i, j + 1] - fld[i, j - 1]) / (2.0 * grid_resolution)
        return (float(gx), float(gy))

    def get_concentration(
        self,
        cytokine_name: str,
        x: float,
        y: float,
        grid_resolution: float,
    ) -> float:
        """Концентрация цитокина в точке (x, y).

        Подробное описание: Description/Phase2/description_abm_extended.md#MultiCytokineField
        """
        fld = self.fields[cytokine_name]
        i = int(x / grid_resolution)
        j = int(y / grid_resolution)
        i = max(0, min(i, fld.shape[0] - 1))
        j = max(0, min(j, fld.shape[1] - 1))
        return float(fld[i, j])


# =============================================================================
# KDTreeNeighborSearch
# =============================================================================


class KDTreeNeighborSearch:
    """Адаптер KD-Tree с единым интерфейсом поиска соседей.

    Подробное описание: Description/Phase2/description_abm_extended.md#KDTreeNeighborSearch
    """

    def __init__(
        self,
        space_size: tuple[float, float],
        periodic: bool = True,
    ) -> None:
        self._space_size = space_size
        self._periodic = periodic
        self._index = KDTreeSpatialIndex(space_size, periodic=periodic)

    def rebuild(self, agents: list[Agent]) -> None:
        """Перестроение KD-дерева по текущим позициям агентов.

        Подробное описание: Description/Phase2/description_abm_extended.md#KDTreeNeighborSearch
        """
        self._index.build(agents)

    def query_radius(
        self,
        position: tuple[float, float],
        radius: float,
        exclude: Agent | None = None,
    ) -> list[Agent]:
        """Все агенты в радиусе, исключая exclude.

        Подробное описание: Description/Phase2/description_abm_extended.md#KDTreeNeighborSearch
        """
        results = self._index.query_radius(position, radius)
        if exclude is not None:
            results = [a for a in results if a is not exclude]
        return results

    def query_nearest(
        self,
        position: tuple[float, float],
        k: int = 1,
        exclude: Agent | None = None,
    ) -> list[Agent]:
        """K ближайших соседей.

        Подробное описание: Description/Phase2/description_abm_extended.md#KDTreeNeighborSearch
        """
        if self._index._tree is None or not self._index._agents:
            return []
        k_query = k + 1 if exclude is not None else k
        results = self._index.query_nearest(position, k_query)
        if exclude is not None:
            results = [a for a in results if a is not exclude]
        return results[:k]


# =============================================================================
# SubcyclingManager
# =============================================================================


@dataclass
class SubcyclingManager:
    """Менеджер subcycling — разные dt для агентов и цитокиновых полей.

    Подробное описание: Description/Phase2/description_abm_extended.md#SubcyclingManager
    """

    agent_dt: float
    field_dt: float

    @property
    def n_field_substeps(self) -> int:
        """Количество подшагов полей за один шаг агента.

        Подробное описание: Description/Phase2/description_abm_extended.md#SubcyclingManager
        """
        return max(1, math.ceil(self.agent_dt / self.field_dt))

    def should_update_field(self, agent_step_count: int) -> bool:
        """Нужно ли обновлять поля на данном подшаге.

        Подробное описание: Description/Phase2/description_abm_extended.md#SubcyclingManager
        """
        return agent_step_count % self.n_field_substeps == 0

    def get_field_dt(self) -> float:
        """Фактический шаг для полей.

        Подробное описание: Description/Phase2/description_abm_extended.md#SubcyclingManager
        """
        n = self.n_field_substeps
        if abs(self.agent_dt - n * self.field_dt) < 1e-12:
            return self.field_dt
        return self.agent_dt / n
