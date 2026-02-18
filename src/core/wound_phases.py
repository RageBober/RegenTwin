"""Детекция фаз заживления раны.

4 фазы регенерации тканей (Gurtner et al., Nature 2008):
- Гемостаз (0-6 ч): тромбоциты, фибрин, PDGF/TGF-β/VEGF
- Воспаление (6 ч - 4-6 дней): нейтрофилы, M1, TNF-α, IL-8
- Пролиферация (4-21 дней): фибробласты, M2, VEGF, коллаген
- Ремоделирование (21 д - 1 год): MMP/TIMP баланс, апоптоз Mf

Биологическое обоснование:
    Gurtner et al., Nature 2008; Eming et al., STM 2014

Подробное описание: Description/Phase2/description_wound_phases.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from src.core.extended_sde import ExtendedSDEState, ExtendedSDETrajectory
from src.core.parameters import ParameterSet


class WoundPhase(Enum):
    """Фазы заживления раны.

    4 последовательных перекрывающихся фазы:
    - HEMOSTASIS: 0-6 ч (тромбоциты, фибрин)
    - INFLAMMATION: 6 ч - 4-6 дней (нейтрофилы, M1)
    - PROLIFERATION: 4-21 дней (фибробласты, M2, ангиогенез)
    - REMODELING: 21 д - 1 год (коллаген, MMP, апоптоз Mf)

    Подробное описание:
        Description/Phase2/description_wound_phases.md#WoundPhase
    """

    HEMOSTASIS = "hemostasis"
    INFLAMMATION = "inflammation"
    PROLIFERATION = "proliferation"
    REMODELING = "remodeling"


@dataclass
class PhaseIndicators:
    """Индикаторы текущей фазы заживления.

    Содержит определённую фазу, уверенность детекции,
    доминирующие клетки и цитокины, прогресс внутри фазы.

    Подробное описание:
        Description/Phase2/description_wound_phases.md#PhaseIndicators
    """

    phase: WoundPhase = WoundPhase.HEMOSTASIS  # Текущая фаза
    confidence: float = 0.0       # Уверенность (0-1)
    dominant_cells: list[str] = field(
        default_factory=list,
    )  # Доминирующие клетки
    dominant_cytokines: list[str] = field(
        default_factory=list,
    )  # Доминирующие цитокины
    phase_progress: float = 0.0   # Прогресс внутри фазы (0-1)


class WoundPhaseDetector:
    """Детектор фаз заживления на основе состояния SDE системы.

    Анализирует соотношения клеточных популяций и цитокинов
    для определения текущей фазы заживления раны.

    Алгоритм: вычисляет confidence для каждой из 4 фаз,
    выбирает фазу с максимальной уверенностью.

    Подробное описание:
        Description/Phase2/description_wound_phases.md#WoundPhaseDetector
    """

    def __init__(self, params: ParameterSet | None = None) -> None:
        """Инициализация детектора фаз.

        Args:
            params: Параметры модели (для порогов детекции).
                    None → литературные defaults

        Подробное описание:
            Description/Phase2/description_wound_phases.md#__init__
        """
        self.params = params if params is not None else ParameterSet()

    def detect_phase(
        self, state: ExtendedSDEState,
    ) -> PhaseIndicators:
        """Определение текущей фазы заживления по состоянию.

        Вычисляет confidence для каждой из 4 фаз (гемостаз,
        воспаление, пролиферация, ремоделирование),
        выбирает фазу с максимальным confidence.

        Args:
            state: Текущее состояние 20-переменной системы

        Returns:
            PhaseIndicators с определённой фазой и уверенностью

        Подробное описание:
            Description/Phase2/description_wound_phases.md#detect_phase
        """
        confidences = {
            WoundPhase.HEMOSTASIS: self._is_hemostasis(state),
            WoundPhase.INFLAMMATION: self._is_inflammation(state),
            WoundPhase.PROLIFERATION: self._is_proliferation(state),
            WoundPhase.REMODELING: self._is_remodeling(state),
        }
        best_phase = max(confidences, key=confidences.get)
        best_confidence = confidences[best_phase]

        # Доминирующие клетки (топ-3 по значению)
        cells = {
            "P": state.P, "Ne": state.Ne, "M1": state.M1, "M2": state.M2,
            "F": state.F, "Mf": state.Mf, "E": state.E, "S": state.S,
        }
        dominant_cells = [
            name for name, val
            in sorted(cells.items(), key=lambda x: x[1], reverse=True)[:3]
            if val > 0
        ]

        # Доминирующие цитокины (топ-3 по значению)
        cytos = {
            "TNF": state.C_TNF, "IL10": state.C_IL10,
            "PDGF": state.C_PDGF, "VEGF": state.C_VEGF,
            "TGFb": state.C_TGFb, "MCP1": state.C_MCP1,
            "IL8": state.C_IL8,
        }
        dominant_cytokines = [
            name for name, val
            in sorted(cytos.items(), key=lambda x: x[1], reverse=True)[:3]
            if val > 0
        ]

        return PhaseIndicators(
            phase=best_phase,
            confidence=best_confidence,
            dominant_cells=dominant_cells,
            dominant_cytokines=dominant_cytokines,
            phase_progress=max(0.0, min(1.0, best_confidence)),
        )

    def detect_phase_trajectory(
        self, trajectory: ExtendedSDETrajectory,
    ) -> list[PhaseIndicators]:
        """Определение фаз для всей траектории.

        Применяет detect_phase к каждому состоянию в траектории.

        Args:
            trajectory: Траектория симуляции

        Returns:
            Список PhaseIndicators для каждого временного шага

        Подробное описание:
            Description/Phase2/description_wound_phases.md#detect_trajectory
        """
        return [self.detect_phase(s) for s in trajectory.states]

    def _is_hemostasis(self, state: ExtendedSDEState) -> float:
        """Оценка уверенности для фазы гемостаза (0-6 ч).

        Критерии:
        - Высокая концентрация тромбоцитов P
        - Высокий уровень фибрина rho_fibrin
        - Сильный damage signal D
        - Ранние DAMPs → продукция PDGF, TGF-β

        Args:
            state: Текущее состояние

        Returns:
            Confidence score ∈ [0, 1]

        Подробное описание:
            Description/Phase2/description_wound_phases.md#_is_hemostasis
        """
        p_score = min(state.P / self.params.P_max, 1.0) if self.params.P_max > 0 else 0.0
        fibrin_score = min(state.rho_fibrin, 1.0)
        d_score = min(state.D / self.params.D0, 1.0) if self.params.D0 > 0 else 0.0
        immune_low = 1.0 / (1.0 + state.Ne / 100.0 + state.M1 / 100.0)
        confidence = (
            0.3 * p_score + 0.3 * fibrin_score
            + 0.2 * d_score + 0.2 * immune_low
        )
        return max(0.0, min(1.0, confidence))

    def _is_inflammation(self, state: ExtendedSDEState) -> float:
        """Оценка уверенности для фазы воспаления (6ч - 4-6 дней).

        Критерии:
        - Высокая концентрация нейтрофилов Ne
        - M1 > M2 (провоспалительная доминантность)
        - Высокий TNF-α и IL-8
        - Снижающийся damage signal

        Args:
            state: Текущее состояние

        Returns:
            Confidence score ∈ [0, 1]

        Подробное описание:
            Description/Phase2/description_wound_phases.md#_is_inflammation
        """
        ne_score = min(state.Ne / 500.0, 1.0)
        m_total = state.M1 + state.M2 + 1e-10
        m1_ratio = state.M1 / m_total
        tnf_score = min(state.C_TNF / 5.0, 1.0)
        il8_score = min(state.C_IL8 / 3.0, 1.0)
        activity = min((state.M1 + state.Ne) / 300.0, 1.0)
        confidence = (
            0.3 * ne_score + 0.25 * m1_ratio + 0.2 * tnf_score
            + 0.15 * il8_score + 0.1 * activity
        )
        return max(0.0, min(1.0, confidence))

    def _is_proliferation(self, state: ExtendedSDEState) -> float:
        """Оценка уверенности для фазы пролиферации (4-21 дней).

        Критерии:
        - Рост фибробластов F
        - M2 > M1 (репаративная доминантность)
        - Растущий коллаген rho_collagen
        - Высокий VEGF и PDGF
        - Активный ангиогенез (рост E)

        Args:
            state: Текущее состояние

        Returns:
            Confidence score ∈ [0, 1]

        Подробное описание:
            Description/Phase2/description_wound_phases.md#_is_proliferation
        """
        f_score = min(state.F / 1000.0, 1.0)
        m_total = state.M1 + state.M2 + 1e-10
        m2_ratio = state.M2 / m_total
        collagen_score = min(
            state.rho_collagen / self.params.rho_c_max, 1.0,
        )
        vegf_score = min(state.C_VEGF / 2.0, 1.0)
        pdgf_score = min(state.C_PDGF / 3.0, 1.0)
        gf_score = 0.5 * vegf_score + 0.5 * pdgf_score
        e_score = min(state.E / 200.0, 1.0)
        confidence = (
            0.25 * f_score + 0.2 * m2_ratio + 0.2 * collagen_score
            + 0.2 * gf_score + 0.15 * e_score
        )
        return max(0.0, min(1.0, confidence))

    def _is_remodeling(self, state: ExtendedSDEState) -> float:
        """Оценка уверенности для фазы ремоделирования (21д - 1 год).

        Критерии:
        - Стабильный/высокий коллаген
        - Активный MMP/TIMP баланс
        - Снижение клеточности (Ne → 0, M1 → 0)
        - Апоптоз миофибробластов (Mf снижается)
        - Низкий фибрин

        Args:
            state: Текущее состояние

        Returns:
            Confidence score ∈ [0, 1]

        Подробное описание:
            Description/Phase2/description_wound_phases.md#_is_remodeling
        """
        collagen_score = min(
            state.rho_collagen / self.params.rho_c_max, 1.0,
        )
        mmp_score = min(state.C_MMP / 0.5, 1.0)
        low_ne = 1.0 / (1.0 + state.Ne / 50.0)
        low_m1 = 1.0 / (1.0 + state.M1 / 50.0)
        low_cellularity = 0.5 * low_ne + 0.5 * low_m1
        low_fibrin = 1.0 / (1.0 + state.rho_fibrin / 0.1)
        low_mf = 1.0 / (1.0 + state.Mf / 50.0)
        # Ремоделирование требует значимого коллагена как обязательного маркера
        maturity_gate = min(collagen_score / 0.3, 1.0)
        raw = (
            0.3 * collagen_score + 0.2 * mmp_score
            + 0.2 * low_cellularity + 0.15 * low_fibrin
            + 0.15 * low_mf
        )
        confidence = raw * maturity_gate
        return max(0.0, min(1.0, confidence))

    def get_phase_boundaries(
        self, trajectory: ExtendedSDETrajectory,
    ) -> dict[WoundPhase, tuple[float, float]]:
        """Определение временных границ каждой фазы в траектории.

        Сканирует траекторию, находит интервалы доминирования
        каждой фазы (момент, когда фаза становится/перестаёт быть
        максимально уверенной).

        Args:
            trajectory: Траектория симуляции

        Returns:
            {WoundPhase: (t_start, t_end)} для каждой обнаруженной
            фазы. Фазы могут перекрываться.

        Подробное описание:
            Description/Phase2/description_wound_phases.md#get_phase_boundaries
        """
        if not trajectory.states:
            return {}

        phases = self.detect_phase_trajectory(trajectory)
        times = trajectory.times
        boundaries: dict[WoundPhase, tuple[float, float]] = {}

        for i, pi in enumerate(phases):
            t = float(times[i])
            if pi.phase not in boundaries:
                boundaries[pi.phase] = (t, t)
            else:
                boundaries[pi.phase] = (boundaries[pi.phase][0], t)

        return boundaries
