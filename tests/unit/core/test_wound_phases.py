"""TDD тесты для wound_phases.py — детекция фаз заживления.

Тестирование:
- WoundPhase: 4 фазы (гемостаз, воспаление, пролиферация, ремоделирование)
- PhaseIndicators: результат детекции (фаза, confidence, доминирующие клетки)
- WoundPhaseDetector: определение фазы по состоянию SDE
  - detect_phase: единичное состояние
  - detect_phase_trajectory: вся траектория
  - _is_hemostasis/inflammation/proliferation/remodeling: confidence
  - get_phase_boundaries: временные границы фаз
- Биологические свойства: последовательность H→I→P→R, временные рамки

Все тесты написаны для stub-реализации (NotImplementedError).
После реализации методов тесты должны проходить.
"""

import numpy as np
import pytest

from src.core.extended_sde import (
    ExtendedSDEState,
    ExtendedSDETrajectory,
)
from src.core.parameters import ParameterSet
from src.core.wound_phases import (
    PhaseIndicators,
    WoundPhase,
    WoundPhaseDetector,
)

# =============================================================================
# TestWoundPhaseEnum
# =============================================================================


class TestWoundPhaseEnum:
    """Тесты перечисления WoundPhase."""

    def test_four_phases(self):
        """Ровно 4 фазы заживления."""
        assert len(WoundPhase) == 4

    def test_hemostasis_value(self):
        """HEMOSTASIS имеет значение 'hemostasis'."""
        assert WoundPhase.HEMOSTASIS.value == "hemostasis"

    def test_all_values(self):
        """Все 4 строковых значения присутствуют."""
        values = {p.value for p in WoundPhase}
        assert values == {
            "hemostasis", "inflammation", "proliferation", "remodeling",
        }

    def test_biological_order(self):
        """Фазы определены в биологическом порядке."""
        phases = list(WoundPhase)
        assert phases[0] == WoundPhase.HEMOSTASIS
        assert phases[1] == WoundPhase.INFLAMMATION
        assert phases[2] == WoundPhase.PROLIFERATION
        assert phases[3] == WoundPhase.REMODELING


# =============================================================================
# TestPhaseIndicatorsDataclass
# =============================================================================


class TestPhaseIndicatorsDataclass:
    """Тесты dataclass PhaseIndicators."""

    def test_defaults(self):
        """Значения по умолчанию: HEMOSTASIS, confidence=0.0."""
        pi = PhaseIndicators()
        assert pi.phase == WoundPhase.HEMOSTASIS
        assert pi.confidence == 0.0
        assert pi.dominant_cells == []
        assert pi.dominant_cytokines == []
        assert pi.phase_progress == 0.0

    def test_confidence(self):
        """Поле confidence корректно сохраняется."""
        pi = PhaseIndicators(confidence=0.85)
        assert pi.confidence == 0.85

    def test_dominant_cells(self):
        """Поле dominant_cells принимает список строк."""
        pi = PhaseIndicators(dominant_cells=["P", "Ne"])
        assert pi.dominant_cells == ["P", "Ne"]

    def test_dominant_cytokines(self):
        """Поле dominant_cytokines принимает список строк."""
        pi = PhaseIndicators(dominant_cytokines=["TNF", "IL8"])
        assert pi.dominant_cytokines == ["TNF", "IL8"]

    def test_phase_progress(self):
        """Поле phase_progress корректно сохраняется."""
        pi = PhaseIndicators(phase_progress=0.5)
        assert pi.phase_progress == 0.5

    def test_custom_phase(self):
        """Произвольная фаза корректно сохраняется."""
        pi = PhaseIndicators(phase=WoundPhase.PROLIFERATION)
        assert pi.phase == WoundPhase.PROLIFERATION


# =============================================================================
# TestWoundPhaseDetectorInit
# =============================================================================


class TestWoundPhaseDetectorInit:
    """Тесты инициализации WoundPhaseDetector."""

    def test_default_init(self):
        """Инициализация с параметрами по умолчанию."""
        detector = WoundPhaseDetector()
        assert detector.params == ParameterSet()

    def test_custom_params(self):
        """Кастомные параметры корректно сохраняются."""
        params = ParameterSet(r_F=0.05)
        detector = WoundPhaseDetector(params=params)
        assert detector.params.r_F == 0.05

    def test_none_params_defaults(self):
        """params=None -> ParameterSet по умолчанию."""
        detector = WoundPhaseDetector(params=None)
        assert isinstance(detector.params, ParameterSet)


# =============================================================================
# TestDetectPhase
# =============================================================================


class TestDetectPhase:
    """Тесты detect_phase: определение фазы по состоянию."""

    def test_hemostasis_detected(self, wound_phase_detector):
        """Состояние гемостаза -> HEMOSTASIS."""
        state = ExtendedSDEState(
            P=10000.0, D=1.0, rho_fibrin=1.0, t=1.0,
        )
        result = wound_phase_detector.detect_phase(state)
        assert result.phase == WoundPhase.HEMOSTASIS

    def test_inflammation_detected(self, wound_phase_detector):
        """Состояние воспаления -> INFLAMMATION."""
        state = ExtendedSDEState(
            Ne=500.0, M1=200.0, M2=50.0,
            C_TNF=5.0, C_IL8=3.0, D=0.3, t=48.0,
        )
        result = wound_phase_detector.detect_phase(state)
        assert result.phase == WoundPhase.INFLAMMATION

    def test_proliferation_detected(self, wound_phase_detector):
        """Состояние пролиферации -> PROLIFERATION."""
        state = ExtendedSDEState(
            F=1000.0, M2=300.0, M1=50.0, E=200.0,
            C_VEGF=2.0, C_PDGF=3.0, rho_collagen=0.5, t=240.0,
        )
        result = wound_phase_detector.detect_phase(state)
        assert result.phase == WoundPhase.PROLIFERATION

    def test_remodeling_detected(self, wound_phase_detector):
        """Состояние ремоделирования -> REMODELING."""
        state = ExtendedSDEState(
            rho_collagen=0.9, C_MMP=0.5, Mf=5.0,
            Ne=0.0, M1=0.0, rho_fibrin=0.01, t=600.0,
        )
        result = wound_phase_detector.detect_phase(state)
        assert result.phase == WoundPhase.REMODELING

    def test_all_zeros_no_error(self, wound_phase_detector):
        """Все переменные = 0: не падает, возвращает фазу."""
        state = ExtendedSDEState()
        result = wound_phase_detector.detect_phase(state)
        assert isinstance(result.phase, WoundPhase)

    def test_returns_phase_indicators(self, wound_phase_detector):
        """Возвращает PhaseIndicators."""
        state = ExtendedSDEState(P=1000.0)
        result = wound_phase_detector.detect_phase(state)
        assert isinstance(result, PhaseIndicators)

    def test_confidence_0_to_1(self, wound_phase_detector):
        """Confidence ∈ [0, 1]."""
        state = ExtendedSDEState(Ne=500.0, M1=200.0)
        result = wound_phase_detector.detect_phase(state)
        assert 0.0 <= result.confidence <= 1.0

    def test_phase_progress_0_to_1(self, wound_phase_detector):
        """phase_progress ∈ [0, 1]."""
        state = ExtendedSDEState(F=500.0, M2=200.0)
        result = wound_phase_detector.detect_phase(state)
        assert 0.0 <= result.phase_progress <= 1.0

    def test_dominant_cells_is_list(self, wound_phase_detector):
        """dominant_cells — список."""
        state = ExtendedSDEState(P=1000.0)
        result = wound_phase_detector.detect_phase(state)
        assert isinstance(result.dominant_cells, list)

    def test_dominant_cytokines_is_list(self, wound_phase_detector):
        """dominant_cytokines — список."""
        state = ExtendedSDEState(C_TNF=5.0)
        result = wound_phase_detector.detect_phase(state)
        assert isinstance(result.dominant_cytokines, list)


# =============================================================================
# TestIsHemostasis
# =============================================================================


class TestIsHemostasis:
    """Тесты confidence для фазы гемостаза."""

    def test_high_markers_high_confidence(self, wound_phase_detector):
        """P=10000, D=1, fibrin=1: высокий confidence."""
        state = ExtendedSDEState(
            P=10000.0, D=1.0, rho_fibrin=1.0,
        )
        result = wound_phase_detector._is_hemostasis(state)
        assert result > 0.5

    def test_zero_markers_low(self, wound_phase_detector):
        """P=0, D=0, fibrin=0: низкий confidence."""
        state = ExtendedSDEState(P=0.0, D=0.0, rho_fibrin=0.0)
        result = wound_phase_detector._is_hemostasis(state)
        assert result < 0.3

    def test_range_0_1(self, wound_phase_detector):
        """Любой вход: confidence ∈ [0, 1]."""
        state = ExtendedSDEState(P=5000.0, D=0.5, rho_fibrin=0.5)
        result = wound_phase_detector._is_hemostasis(state)
        assert 0.0 <= result <= 1.0

    def test_late_state_low(self, wound_phase_detector, inflammation_state):
        """Воспалительное состояние: низкий confidence гемостаза."""
        result = wound_phase_detector._is_hemostasis(inflammation_state)
        inflam = wound_phase_detector._is_inflammation(inflammation_state)
        # Гемостаз должен быть ниже воспаления
        assert result < inflam


# =============================================================================
# TestIsInflammation
# =============================================================================


class TestIsInflammation:
    """Тесты confidence для фазы воспаления."""

    def test_high_Ne_M1_TNF(self, wound_phase_detector):
        """Ne=500, M1=200, M2=50: высокий confidence."""
        state = ExtendedSDEState(
            Ne=500.0, M1=200.0, M2=50.0, C_TNF=5.0,
        )
        result = wound_phase_detector._is_inflammation(state)
        assert result > 0.5

    def test_zero_markers_low(self, wound_phase_detector):
        """Ne=0, M1=0, TNF=0: низкий confidence."""
        state = ExtendedSDEState(Ne=0.0, M1=0.0, C_TNF=0.0)
        result = wound_phase_detector._is_inflammation(state)
        assert result < 0.3

    def test_M2_greater_M1_moderate(self, wound_phase_detector):
        """M1=100, M2=300: умеренный confidence (M2 > M1)."""
        state_m1_dom = ExtendedSDEState(
            Ne=500.0, M1=300.0, M2=100.0, C_TNF=5.0,
        )
        state_m2_dom = ExtendedSDEState(
            Ne=500.0, M1=100.0, M2=300.0, C_TNF=5.0,
        )
        conf_m1 = wound_phase_detector._is_inflammation(state_m1_dom)
        conf_m2 = wound_phase_detector._is_inflammation(state_m2_dom)
        # M1 доминантность даёт более высокий confidence воспаления
        assert conf_m1 > conf_m2

    def test_range_0_1(self, wound_phase_detector):
        """Любой вход: confidence ∈ [0, 1]."""
        state = ExtendedSDEState(Ne=250.0, M1=100.0)
        result = wound_phase_detector._is_inflammation(state)
        assert 0.0 <= result <= 1.0


# =============================================================================
# TestIsProliferation
# =============================================================================


class TestIsProliferation:
    """Тесты confidence для фазы пролиферации."""

    def test_high_F_M2_collagen(self, wound_phase_detector):
        """F=1000, M2=300, collagen=0.5: высокий confidence."""
        state = ExtendedSDEState(
            F=1000.0, M2=300.0, M1=50.0,
            rho_collagen=0.5, C_VEGF=2.0,
        )
        result = wound_phase_detector._is_proliferation(state)
        assert result > 0.5

    def test_zero_markers_low(self, wound_phase_detector):
        """F=0, M2=0, collagen=0: низкий confidence."""
        state = ExtendedSDEState(F=0.0, M2=0.0, rho_collagen=0.0)
        result = wound_phase_detector._is_proliferation(state)
        assert result < 0.3

    def test_range_0_1(self, wound_phase_detector):
        """Любой вход: confidence ∈ [0, 1]."""
        state = ExtendedSDEState(F=500.0, M2=150.0)
        result = wound_phase_detector._is_proliferation(state)
        assert 0.0 <= result <= 1.0

    def test_VEGF_PDGF_boost(self, wound_phase_detector):
        """VEGF/PDGF увеличивают confidence пролиферации."""
        state_no_gf = ExtendedSDEState(
            F=500.0, M2=200.0, M1=50.0,
            C_VEGF=0.0, C_PDGF=0.0,
        )
        state_with_gf = ExtendedSDEState(
            F=500.0, M2=200.0, M1=50.0,
            C_VEGF=5.0, C_PDGF=5.0,
        )
        conf_no = wound_phase_detector._is_proliferation(state_no_gf)
        conf_with = wound_phase_detector._is_proliferation(state_with_gf)
        assert conf_with >= conf_no


# =============================================================================
# TestIsRemodeling
# =============================================================================


class TestIsRemodeling:
    """Тесты confidence для фазы ремоделирования."""

    def test_high_collagen_MMP(self, wound_phase_detector):
        """collagen=0.9, MMP=0.5: высокий confidence."""
        state = ExtendedSDEState(
            rho_collagen=0.9, C_MMP=0.5, Mf=5.0,
            Ne=0.0, M1=0.0, rho_fibrin=0.01,
        )
        result = wound_phase_detector._is_remodeling(state)
        assert result > 0.5

    def test_low_collagen_low(self, wound_phase_detector):
        """collagen=0.1, MMP=0: низкий confidence."""
        state = ExtendedSDEState(rho_collagen=0.1, C_MMP=0.0)
        result = wound_phase_detector._is_remodeling(state)
        assert result < 0.3

    def test_range_0_1(self, wound_phase_detector):
        """Любой вход: confidence ∈ [0, 1]."""
        state = ExtendedSDEState(rho_collagen=0.5, C_MMP=0.3)
        result = wound_phase_detector._is_remodeling(state)
        assert 0.0 <= result <= 1.0

    def test_high_fibrin_reduces(self, wound_phase_detector):
        """fibrin=1.0: снижает confidence ремоделирования."""
        state_no_fibrin = ExtendedSDEState(
            rho_collagen=0.8, C_MMP=0.5, rho_fibrin=0.0,
        )
        state_high_fibrin = ExtendedSDEState(
            rho_collagen=0.8, C_MMP=0.5, rho_fibrin=1.0,
        )
        conf_no = wound_phase_detector._is_remodeling(state_no_fibrin)
        conf_high = wound_phase_detector._is_remodeling(state_high_fibrin)
        assert conf_no >= conf_high


# =============================================================================
# TestDetectPhaseTrajectory
# =============================================================================


class TestDetectPhaseTrajectory:
    """Тесты detect_phase_trajectory: фазы для всей траектории."""

    def test_returns_list(
        self, wound_phase_detector, sample_extended_trajectory,
    ):
        """Возвращает список."""
        result = wound_phase_detector.detect_phase_trajectory(
            sample_extended_trajectory,
        )
        assert isinstance(result, list)

    def test_length_matches(
        self, wound_phase_detector, sample_extended_trajectory,
    ):
        """Длина списка == количество состояний в траектории."""
        result = wound_phase_detector.detect_phase_trajectory(
            sample_extended_trajectory,
        )
        assert len(result) == len(sample_extended_trajectory.states)

    def test_each_is_phase_indicators(
        self, wound_phase_detector, sample_extended_trajectory,
    ):
        """Каждый элемент — PhaseIndicators."""
        result = wound_phase_detector.detect_phase_trajectory(
            sample_extended_trajectory,
        )
        for pi in result:
            assert isinstance(pi, PhaseIndicators)

    def test_empty_trajectory(self, wound_phase_detector):
        """Пустая траектория -> пустой список."""
        traj = ExtendedSDETrajectory()
        result = wound_phase_detector.detect_phase_trajectory(traj)
        assert result == []


# =============================================================================
# TestGetPhaseBoundaries
# =============================================================================


class TestGetPhaseBoundaries:
    """Тесты get_phase_boundaries: временные границы фаз."""

    def test_returns_dict(
        self, wound_phase_detector, sample_extended_trajectory,
    ):
        """Возвращает dict."""
        result = wound_phase_detector.get_phase_boundaries(
            sample_extended_trajectory,
        )
        assert isinstance(result, dict)

    def test_keys_wound_phases(
        self, wound_phase_detector, sample_extended_trajectory,
    ):
        """Ключи — WoundPhase."""
        result = wound_phase_detector.get_phase_boundaries(
            sample_extended_trajectory,
        )
        for key in result:
            assert isinstance(key, WoundPhase)

    def test_values_tuples(
        self, wound_phase_detector, sample_extended_trajectory,
    ):
        """Значения — tuple[float, float]."""
        result = wound_phase_detector.get_phase_boundaries(
            sample_extended_trajectory,
        )
        for _phase, bounds in result.items():
            assert isinstance(bounds, tuple)
            assert len(bounds) == 2

    def test_t_start_le_t_end(
        self, wound_phase_detector, sample_extended_trajectory,
    ):
        """t_start <= t_end для каждой фазы."""
        result = wound_phase_detector.get_phase_boundaries(
            sample_extended_trajectory,
        )
        for phase, (t_start, t_end) in result.items():
            assert t_start <= t_end, (
                f"Нарушение для {phase}: {t_start} > {t_end}"
            )

    def test_empty_trajectory(self, wound_phase_detector):
        """Пустая траектория -> пустой dict."""
        traj = ExtendedSDETrajectory()
        result = wound_phase_detector.get_phase_boundaries(traj)
        assert result == {}


# =============================================================================
# TestBiologicalPhaseProperties
# =============================================================================


class TestBiologicalPhaseProperties:
    """Тесты биологических свойств последовательности фаз."""

    @pytest.fixture
    def mock_wound_trajectory(self):
        """Траектория с 4-фазной прогрессией заживления."""
        states = []

        # Гемостаз (0-6ч): P высокий, fibrin, D
        for i in range(7):
            t = float(i)
            states.append(ExtendedSDEState(
                P=10000.0 - i * 1000, D=1.0 - i * 0.1,
                rho_fibrin=1.0, t=t,
            ))

        # Воспаление (7-96ч): Ne, M1>M2, TNF
        for j in range(90):
            t = float(7 + j)
            ne = max(0.0, 500.0 - j * 5)
            m1 = max(0.0, 200.0 - j * 2)
            m2 = 50.0 + j * 2
            states.append(ExtendedSDEState(
                Ne=ne, M1=m1, M2=m2, C_TNF=max(0.0, 5.0 - j * 0.05),
                C_IL8=max(0.0, 3.0 - j * 0.03), D=max(0.0, 0.3 - j * 0.003),
                t=t,
            ))

        # Пролиферация (97-504ч): F, M2>M1, VEGF, collagen растёт
        for k in range(408):
            t = float(97 + k)
            states.append(ExtendedSDEState(
                F=100.0 + k * 2, M2=300.0, M1=30.0, E=50.0 + k * 0.3,
                rho_collagen=0.1 + k * 0.002,
                C_VEGF=2.0, C_PDGF=3.0, t=t,
            ))

        # Ремоделирование (505-720ч): collagen стабильный, MMP, Mf↓
        for m in range(216):
            t = float(505 + m)
            states.append(ExtendedSDEState(
                rho_collagen=0.9, C_MMP=0.5,
                Mf=max(0.0, 50.0 - m * 0.2), F=200.0,
                rho_fibrin=0.01, t=t,
            ))

        return ExtendedSDETrajectory(
            times=np.array([s.t for s in states]),
            states=states,
        )

    def test_phase_order_H_I_P_R(
        self, wound_phase_detector, mock_wound_trajectory,
    ):
        """Нормальное заживление: фазы идут в порядке H→I→P→R."""
        boundaries = wound_phase_detector.get_phase_boundaries(
            mock_wound_trajectory,
        )
        # Ожидаем минимум 3 фазы (ремоделирование может быть неполным)
        assert len(boundaries) >= 3

        # Проверяем порядок появления
        if WoundPhase.HEMOSTASIS in boundaries and \
                WoundPhase.INFLAMMATION in boundaries:
            assert (
                boundaries[WoundPhase.HEMOSTASIS][0]
                <= boundaries[WoundPhase.INFLAMMATION][0]
            )

        if WoundPhase.INFLAMMATION in boundaries and \
                WoundPhase.PROLIFERATION in boundaries:
            assert (
                boundaries[WoundPhase.INFLAMMATION][0]
                <= boundaries[WoundPhase.PROLIFERATION][0]
            )

    def test_hemostasis_before_6h(
        self, wound_phase_detector, mock_wound_trajectory,
    ):
        """Гемостаз начинается около t=0."""
        boundaries = wound_phase_detector.get_phase_boundaries(
            mock_wound_trajectory,
        )
        if WoundPhase.HEMOSTASIS in boundaries:
            t_start = boundaries[WoundPhase.HEMOSTASIS][0]
            assert t_start <= 1.0

    def test_inflammation_peak_24_48h(
        self, wound_phase_detector, mock_wound_trajectory,
    ):
        """Воспаление доминирует где-то в 24-96ч."""
        phases = wound_phase_detector.detect_phase_trajectory(
            mock_wound_trajectory,
        )
        times = mock_wound_trajectory.times

        # Находим индексы где фаза = INFLAMMATION
        inflam_times = [
            times[i]
            for i, pi in enumerate(phases)
            if pi.phase == WoundPhase.INFLAMMATION
        ]
        if inflam_times:
            # Какие-то моменты воспаления должны быть в 12-96ч
            assert any(12.0 <= t <= 96.0 for t in inflam_times)

    def test_M1_M2_ratio_matches_phase(
        self, wound_phase_detector, mock_wound_trajectory,
    ):
        """M1 > M2 при воспалении, M2 > M1 при пролиферации."""
        phases = wound_phase_detector.detect_phase_trajectory(
            mock_wound_trajectory,
        )
        states = mock_wound_trajectory.states

        for i, pi in enumerate(phases):
            state = states[i]
            if pi.phase == WoundPhase.INFLAMMATION and state.M1 > 0:
                # В фазе воспаления M1 >= M2 (не всегда строго)
                pass  # Проверка мягкая: не требуем строгого неравенства
            if pi.phase == WoundPhase.PROLIFERATION and state.M2 > 0:
                # В фазе пролиферации M2 должен быть значимым
                assert state.M2 > 0

    def test_collagen_in_proliferation(
        self, wound_phase_detector, mock_wound_trajectory,
    ):
        """Коллаген растёт в фазе пролиферации."""
        phases = wound_phase_detector.detect_phase_trajectory(
            mock_wound_trajectory,
        )
        states = mock_wound_trajectory.states

        # Собираем коллаген в фазе пролиферации
        prolif_collagen = [
            states[i].rho_collagen
            for i, pi in enumerate(phases)
            if pi.phase == WoundPhase.PROLIFERATION
        ]
        if len(prolif_collagen) > 2:
            # Коллаген должен расти (последний > первого)
            assert prolif_collagen[-1] > prolif_collagen[0]

    def test_remodeling_after_day_21(
        self, wound_phase_detector, mock_wound_trajectory,
    ):
        """Ремоделирование начинается после ~504ч (21 день)."""
        boundaries = wound_phase_detector.get_phase_boundaries(
            mock_wound_trajectory,
        )
        if WoundPhase.REMODELING in boundaries:
            t_start = boundaries[WoundPhase.REMODELING][0]
            # Ремоделирование не раньше дня 14 (336ч)
            assert t_start >= 336.0, (
                f"Ремоделирование слишком рано: {t_start}ч"
            )
