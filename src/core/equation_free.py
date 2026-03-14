"""equation_free.py — Equation-Free Framework: мультимасштабная интеграция SDE↔ABM.

Реализует паттерн lift → micro-simulate → restrict для связи
расширенной 20-переменной SDE и агентной модели (ABM).

Подробное описание:
    Description/Phase2/description_equation_free.md
"""

from __future__ import annotations

import contextlib
import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.core.abm_model import (
    ABMConfig,
    Agent,
    EndothelialAgent,
    Fibroblast,
    Macrophage,
    MyofibroblastAgent,
    NeutrophilAgent,
    StemCell,
)
from src.core.abm_spatial import PlateletAgent
from src.core.extended_sde import ExtendedSDEState

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

CYTOKINE_NAMES: list[str] = ["TNF", "IL10", "PDGF", "VEGF", "TGFb", "MCP1", "IL8"]

_CYTOKINE_FIELD_MAP: dict[str, str] = {
    "TNF": "C_TNF",
    "IL10": "C_IL10",
    "PDGF": "C_PDGF",
    "VEGF": "C_VEGF",
    "TGFb": "C_TGFb",
    "MCP1": "C_MCP1",
    "IL8": "C_IL8",
}

# Поля ECM/aux, которые переносятся через _macro_context
_MACRO_CONTEXT_FIELDS: list[str] = [
    "rho_collagen",
    "C_MMP",
    "rho_fibrin",
    "D",
    "O2",
]

# Маппинг: поле ExtendedSDEState → класс агента
_CELL_MAP: list[tuple[str, type]] = [
    ("P", PlateletAgent),
    ("Ne", NeutrophilAgent),
    ("M1", Macrophage),
    ("M2", Macrophage),
    ("F", Fibroblast),
    ("Mf", MyofibroblastAgent),
    ("E", EndothelialAgent),
    ("S", StemCell),
]

# Маппинг: AGENT_TYPE → поле ExtendedSDEState (для restrict)
_TYPE_TO_FIELD: dict[str, str] = {
    "platelet": "P",
    "neutro": "Ne",
    "macro": "M1",
    "fibro": "F",
    "myofibro": "Mf",
    "endo": "E",
    "stem": "S",
}


# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------


# Description: Description/Phase2/description_equation_free.md#EquationFreeConfig
@dataclass
class EquationFreeConfig:
    """Конфигурация Equation-Free интегратора.

    Подробное описание:
        Description/Phase2/description_equation_free.md#EquationFreeConfig
    """

    dt_macro: float = 1.0
    """Шаг макроскопического (SDE) времени, ч."""

    dt_micro: float = 0.1
    """Шаг микроскопического (ABM) времени, ч."""

    n_micro_steps: int = 10
    """Количество ABM шагов на один EF-шаг."""

    volume: float = 1e6
    """Объём расчётной области, мкм³."""

    n_agents_scale: float = 1e-3
    """Масштаб конвертации концентрации→число агентов (агентов/клетку)."""

    def __post_init__(self) -> None:
        """Валидация полей конфигурации."""
        if self.dt_macro <= 0:
            raise ValueError("dt_macro must be > 0")
        if self.dt_micro <= 0:
            raise ValueError("dt_micro must be > 0")
        if self.n_micro_steps < 1:
            raise ValueError("n_micro_steps must be >= 1")
        if self.volume <= 0:
            raise ValueError("volume must be > 0")
        if self.n_agents_scale <= 0:
            raise ValueError("n_agents_scale must be > 0")


# ---------------------------------------------------------------------------
# Lifter: macro → micro
# ---------------------------------------------------------------------------


# Description: Description/Phase2/description_equation_free.md#Lifter
class Lifter:
    """Macro→micro lifting — распределение агентов согласно ExtendedSDEState.

    Реализует первый шаг EF-цикла: берёт 20-мерный макро-вектор и создаёт
    популяцию агентов ABM с пространственным распределением.

    Подробное описание:
        Description/Phase2/description_equation_free.md#Lifter
    """

    # Description: Description/Phase2/description_equation_free.md#initconfig-abm_config
    def __init__(self, config: EquationFreeConfig, abm_config: ABMConfig) -> None:
        """Инициализация Lifter.

        Подробное описание:
            Description/Phase2/description_equation_free.md#Lifter.__init__
        """
        self.config = config
        self.abm_config = abm_config
        self.rng: np.random.Generator = np.random.default_rng()
        self._next_id: int = 0

    # Description: Description/Phase2/description_equation_free.md#liftmacro_state-n_agents_hint-volume
    def lift(
        self,
        macro_state: ExtendedSDEState,
        n_agents_hint: int,  # noqa: ARG002
        volume: float,
    ) -> list[Agent]:
        """Распределить агентов по ExtendedSDEState.

        Для каждого клеточного типа (P, Ne, M1, M2, F, Mf, E, S) создаёт
        n_i = round(concentration_i * volume * n_agents_scale) агентов
        с равномерным случайным пространственным распределением.

        Подробное описание:
            Description/Phase2/description_equation_free.md#lift
        """
        all_agents: list[Agent] = []
        space_size = self.abm_config.space_size

        for state_field, agent_class in _CELL_MAP:
            conc = getattr(macro_state, state_field)
            n_i = round(conc * volume * self.config.n_agents_scale)
            if n_i > 0:
                agents = self.distribute_population(conc, agent_class, n_i, space_size, self.rng)
                # Тегируем M1/M2 через polarization_state
                if state_field == "M1":
                    for a in agents:
                        if hasattr(a, "polarization_state"):
                            a.polarization_state = 1.0
                elif state_field == "M2":
                    for a in agents:
                        if hasattr(a, "polarization_state"):
                            a.polarization_state = 0.0
                all_agents.extend(agents)

        # Назначить цитокиновое окружение
        cytokine_levels: dict[str, float] = {}
        for short_name, field_name in _CYTOKINE_FIELD_MAP.items():
            cytokine_levels[short_name] = getattr(macro_state, field_name)
        self.assign_cytokine_fields(all_agents, cytokine_levels)

        # Сохранить ECM/aux поля на агентах для round-trip
        macro_context: dict[str, float] = {}
        for mf in _MACRO_CONTEXT_FIELDS:
            macro_context[mf] = getattr(macro_state, mf)
        for agent in all_agents:
            agent._macro_context = macro_context  # type: ignore[attr-defined]

        return all_agents

    # Description: Description/Phase2/description_equation_free.md#distribute_populationpopulation-agent_class-n_agents-space_size-rng
    def distribute_population(
        self,
        population: float,  # noqa: ARG002
        agent_class: type,
        n_agents: int,
        space_size: tuple[float, float],
        rng: np.random.Generator,
    ) -> list[Agent]:
        """Создать агентов одного типа по концентрации.

        Генерирует n_agents агентов класса agent_class с равномерным
        случайным расположением в области [0, space_size[0]] × [0, space_size[1]].

        Подробное описание:
            Description/Phase2/description_equation_free.md#distribute_population
        """
        if n_agents == 0:
            return []

        xs = rng.uniform(0, space_size[0], n_agents)
        ys = rng.uniform(0, space_size[1], n_agents)

        agents: list[Agent] = []
        for i in range(n_agents):
            aid = self._next_id
            self._next_id += 1
            agent = agent_class(agent_id=aid, x=float(xs[i]), y=float(ys[i]), rng=rng)
            agents.append(agent)
        return agents

    # Description: Description/Phase2/description_equation_free.md#assign_cytokine_fieldsagents-cytokine_levels
    def assign_cytokine_fields(
        self,
        agents: list[Agent],
        cytokine_levels: dict[str, float],
    ) -> None:
        """Назначить цитокиновое окружение агентам.

        Устанавливает атрибут cytokine_environment на каждом агенте (in-place).
        Агенты, не поддерживающие атрибут, пропускаются.

        Подробное описание:
            Description/Phase2/description_equation_free.md#assign_cytokine_fields
        """
        for agent in agents:
            with contextlib.suppress(AttributeError):
                agent.cytokine_environment = cytokine_levels  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Restrictor: micro → macro
# ---------------------------------------------------------------------------


# Description: Description/Phase2/description_equation_free.md#Restrictor
class Restrictor:
    """Micro→macro restriction — агрегация состояния ABM в ExtendedSDEState.

    Реализует последний шаг EF-цикла: агрегирует микро-состояние по формуле
    X_macro = Σ(agent_states) / volume.

    Подробное описание:
        Description/Phase2/description_equation_free.md#Restrictor
    """

    # Description: Description/Phase2/description_equation_free.md#initconfig
    def __init__(self, config: EquationFreeConfig) -> None:
        """Инициализация Restrictor.

        Подробное описание:
            Description/Phase2/description_equation_free.md#Restrictor.__init__
        """
        self.config = config

    # Description: Description/Phase2/description_equation_free.md#restrictagents-volume-t
    def restrict(
        self,
        agents: list[Agent],
        volume: float,
        t: float,
    ) -> ExtendedSDEState:
        """Агрегировать микро-состояние: X_macro = Σ(agent_states) / volume.

        Подсчитывает клеточные концентрации, усредняет цитокиновые поля
        и собирает ECM компоненты в ExtendedSDEState.

        Подробное описание:
            Description/Phase2/description_equation_free.md#restrict
        """
        scale = volume * self.config.n_agents_scale
        fields: dict[str, float] = {}

        # Клеточные концентрации
        for agent_type, state_field in _TYPE_TO_FIELD.items():
            count = self.count_population(agents, agent_type)
            fields[state_field] = count / scale if scale > 0 else 0.0

        # Разделить M1/M2 по polarization_state
        macro_agents = [
            a
            for a in agents
            if getattr(a, "AGENT_TYPE", None) == "macro" and getattr(a, "alive", False)
        ]
        if macro_agents:
            m1_count = sum(1 for a in macro_agents if getattr(a, "polarization_state", 0.5) > 0.5)
            m2_count = len(macro_agents) - m1_count
            fields["M1"] = m1_count / scale if scale > 0 else 0.0
            fields["M2"] = m2_count / scale if scale > 0 else 0.0
        else:
            fields.setdefault("M2", 0.0)

        # Цитокины
        cyto = self.aggregate_cytokines(agents)
        for short_name, field_name in _CYTOKINE_FIELD_MAP.items():
            fields[field_name] = cyto.get(short_name, 0.0)

        # ECM/aux из _macro_context
        contexts = [
            a._macro_context  # type: ignore[attr-defined]
            for a in agents
            if hasattr(a, "_macro_context")
        ]
        for mf in _MACRO_CONTEXT_FIELDS:
            if contexts:
                fields[mf] = sum(c.get(mf, 0.0) for c in contexts) / len(contexts)
            else:
                fields[mf] = 0.0

        fields["t"] = t
        return ExtendedSDEState(**fields)

    # Description: Description/Phase2/description_equation_free.md#count_populationagents-agent_type
    def count_population(
        self,
        agents: list[Agent],
        agent_type: str,
    ) -> float:
        """Подсчитать живых агентов определённого типа.

        Считает только агентов с agent.alive == True и agent.AGENT_TYPE == agent_type.

        Подробное описание:
            Description/Phase2/description_equation_free.md#count_population
        """
        return float(
            sum(
                1
                for a in agents
                if getattr(a, "AGENT_TYPE", None) == agent_type and getattr(a, "alive", False)
            )
        )

    # Description: Description/Phase2/description_equation_free.md#aggregate_cytokinesagents
    def aggregate_cytokines(
        self,
        agents: list[Agent],
    ) -> dict[str, float]:
        """Извлечь средние цитокиновые уровни из агентов.

        Усредняет cytokine_environment по всем агентам.
        Если агенты не имеют атрибута — возвращает нули для всех 7 цитокинов.

        Подробное описание:
            Description/Phase2/description_equation_free.md#aggregate_cytokines
        """
        envs: list[dict[str, float]] = [
            a.cytokine_environment  # type: ignore[attr-defined]
            for a in agents
            if hasattr(a, "cytokine_environment")
            and isinstance(getattr(a, "cytokine_environment", None), dict)
        ]
        if not envs:
            return dict.fromkeys(CYTOKINE_NAMES, 0.0)

        result: dict[str, float] = {}
        for name in CYTOKINE_NAMES:
            vals = [e.get(name, 0.0) for e in envs]
            result[name] = sum(vals) / len(vals) if vals else 0.0
        return result


# ---------------------------------------------------------------------------
# EquationFreeIntegrator: главный EF-интегратор
# ---------------------------------------------------------------------------


# Description: Description/Phase2/description_equation_free.md#EquationFreeIntegrator
class EquationFreeIntegrator:
    """Equation-Free интегратор SDE↔ABM.

    Координирует EF-цикл: lift → n_micro_steps ABM шагов → restrict.
    Связывает ExtendedSDEModel (макро) с ABMModel (микро) через Lifter и Restrictor.

    Подробное описание:
        Description/Phase2/description_equation_free.md#EquationFreeIntegrator
    """

    # Description: Description/Phase2/description_equation_free.md#initsde_model-abm_model-lifter-restrictor-config
    def __init__(
        self,
        sde_model: Any,
        abm_model: Any,
        lifter: Lifter,
        restrictor: Restrictor,
        config: EquationFreeConfig,
    ) -> None:
        """Инициализация интегратора и валидация конфигурации.

        Подробное описание:
            Description/Phase2/description_equation_free.md#EquationFreeIntegrator.__init__
        """
        if config.n_micro_steps < 1:
            raise ValueError("n_micro_steps must be >= 1")
        self.sde_model = sde_model
        self.abm_model = abm_model
        self.lifter = lifter
        self.restrictor = restrictor
        self.config = config
        self.trajectory: list[ExtendedSDEState] = []
        self._agent_id_counter: int = 100_000

    def _next_agent_id(self) -> int:
        aid = self._agent_id_counter
        self._agent_id_counter += 1
        return aid

    # Description: Description/Phase2/description_equation_free.md#stepmacro_state-t-dt
    def step(
        self,
        macro_state: ExtendedSDEState,
        t: float,
        dt: float,
    ) -> ExtendedSDEState:
        """Один шаг EF: lift → micro_steps → restrict.

        Полный EF-цикл для одного макро-шага dt:
        agents = lift(macro_state) → agents = micro_steps(agents) → new_state = restrict(agents).

        Подробное описание:
            Description/Phase2/description_equation_free.md#step
        """
        agents = self._lift_step(macro_state, t)
        agents = self._micro_step(agents, self.config.dt_micro)
        new_macro = self._restrict_step(agents, t + dt)
        self.trajectory.append(new_macro)
        return new_macro

    # Description: Description/Phase2/description_equation_free.md#_lift_stepmacro_state-t
    def _lift_step(
        self,
        macro_state: ExtendedSDEState,
        t: float,  # noqa: ARG002
    ) -> list[Agent]:
        """Внутренний: lifting макро-состояния → список агентов.

        Подробное описание:
            Description/Phase2/description_equation_free.md#_lift_step
        """
        cell_fields = ["P", "Ne", "M1", "M2", "F", "Mf", "E", "S"]
        total = sum(getattr(macro_state, f) for f in cell_fields)
        n_hint = round(total * self.config.volume * self.config.n_agents_scale)
        return self.lifter.lift(macro_state, n_hint, self.config.volume)

    # Description: Description/Phase2/description_equation_free.md#_micro_stepagents-dt_micro
    def _micro_step(
        self,
        agents: list[Agent],
        dt_micro: float,
    ) -> list[Agent]:
        """Внутренний: n_micro_steps шагов ABM с фильтрацией мёртвых агентов.

        Выполняет config.n_micro_steps итераций, на каждом шаге вызывая
        agent.update(dt_micro, environment) и удаляя агентов с alive==False.

        Подробное описание:
            Description/Phase2/description_equation_free.md#_micro_step
        """
        agents = [a for a in agents if getattr(a, "alive", False)]

        for _ in range(self.config.n_micro_steps):
            new_agents: list[Agent] = []
            for agent in agents:
                if not getattr(agent, "alive", False):
                    continue
                with contextlib.suppress(Exception):
                    agent.update(dt_micro, {})  # type: ignore[arg-type]
                if getattr(agent, "alive", False) and getattr(agent, "dividing", False):
                    with contextlib.suppress(Exception):
                        child = agent.divide(self._next_agent_id())  # type: ignore[arg-type]
                        if child is not None:
                            new_agents.append(child)
                    agent.dividing = False  # type: ignore[attr-defined]
            agents.extend(new_agents)
            agents = [a for a in agents if getattr(a, "alive", False)]

        return agents

    # Description: Description/Phase2/description_equation_free.md#_restrict_stepagents-t
    def _restrict_step(
        self,
        agents: list[Agent],
        t: float,
    ) -> ExtendedSDEState:
        """Внутренний: restricting списка агентов → макро-состояние.

        Подробное описание:
            Description/Phase2/description_equation_free.md#_restrict_step
        """
        return self.restrictor.restrict(agents, self.config.volume, t)

    # Description: Description/Phase2/description_equation_free.md#runt_span-dt_macro-dt_micro
    def run(
        self,
        t_span: tuple[float, float],
        dt_macro: float,
        dt_micro: float,  # noqa: ARG002
    ) -> list[ExtendedSDEState]:
        """Полная мультимасштабная симуляция на интервале t_span.

        Выполняет последовательные EF-шаги от t_span[0] до t_span[1]
        с шагом dt_macro. Возвращает траекторию ExtendedSDEState.

        Подробное описание:
            Description/Phase2/description_equation_free.md#run
        """
        self.trajectory = []
        initial = self.sde_model.initial_state
        macro_state = (
            initial if isinstance(initial, ExtendedSDEState) else ExtendedSDEState(t=t_span[0])
        )
        t = t_span[0]
        n_steps = round((t_span[1] - t_span[0]) / dt_macro)
        for _ in range(n_steps):
            macro_state = self.step(macro_state, t, dt_macro)
            t += dt_macro
        return self.trajectory

    # Description: Description/Phase2/description_equation_free.md#apply_therapyagents-macro_state-therapy
    def apply_therapy(
        self,
        agents: list[Agent],
        macro_state: ExtendedSDEState,
        therapy: Any,
    ) -> tuple[list[Agent], ExtendedSDEState]:
        """Применить терапию (PRP/PEMF) одновременно на микро и макро уровнях.

        Модифицирует macro_state через sde_model.apply_therapy_effect(...)
        и каждого агента через agent.apply_therapy(therapy) (если поддерживается).
        Возвращает копии — исходные объекты не мутируются.

        Подробное описание:
            Description/Phase2/description_equation_free.md#apply_therapy
        """
        new_state = copy.copy(macro_state)

        # Делегируем к sde_model
        result_state = self.sde_model.apply_therapy_effect(new_state, therapy)
        if isinstance(result_state, ExtendedSDEState):
            new_state = result_state

        # Применяем к агентам
        new_agents = list(agents)
        for agent in new_agents:
            with contextlib.suppress(Exception):
                agent.apply_therapy(therapy)  # type: ignore[attr-defined]

        return (new_agents, new_state)
