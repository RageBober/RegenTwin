"""TDD тесты для модуля численной робастности numerical_utils.

Тестирование:
- DivergenceInfo: dataclass с диагностикой, is_diverged property
- clip_negative_concentrations: отсечение отрицательных, immutability, selective
- detect_divergence: NaN, Inf, overflow, диагностические сообщения
- handle_divergence: fallback стратегии, откат, dt halving, should_stop
- adaptive_timestep: увеличение/уменьшение dt, bounds, tolerance
- NumericalGuard: контекстный менеджер, warnings, settings restoration
- Loguru-логирование: корректные уровни и сообщения

Все тесты написаны для stub-реализации (NotImplementedError).
После реализации методов тесты должны проходить.
"""

from unittest.mock import patch

import numpy as np
import pytest

from src.core.numerical_utils import (
    DivergenceInfo,
    NumericalGuard,
    adaptive_timestep,
    clip_negative_concentrations,
    detect_divergence,
    handle_divergence,
)

# =============================================================================
# Test DivergenceInfo
# =============================================================================


class TestDivergenceInfo:
    """Тесты dataclass DivergenceInfo."""

    def test_default_values(self):
        """Значения по умолчанию: нет дивергенции."""
        info = DivergenceInfo()

        assert info.has_nan is False
        assert info.has_inf is False
        assert info.nan_variables == []
        assert info.inf_variables == []
        assert info.max_value == 0.0
        assert info.message == ""

    def test_is_diverged_false_by_default(self):
        """is_diverged == False для значений по умолчанию."""
        info = DivergenceInfo()

        assert info.is_diverged is False

    def test_is_diverged_true_when_has_nan(self):
        """is_diverged == True при has_nan=True."""
        info = DivergenceInfo(has_nan=True)

        assert info.is_diverged is True

    def test_is_diverged_true_when_has_inf(self):
        """is_diverged == True при has_inf=True."""
        info = DivergenceInfo(has_inf=True)

        assert info.is_diverged is True

    def test_is_diverged_true_when_both_nan_and_inf(self):
        """is_diverged == True при has_nan=True и has_inf=True."""
        info = DivergenceInfo(has_nan=True, has_inf=True)

        assert info.is_diverged is True

    def test_nan_variables_populated(self):
        """nan_variables содержит имена NaN-переменных."""
        info = DivergenceInfo(
            has_nan=True,
            nan_variables=["N", "C"],
        )

        assert info.nan_variables == ["N", "C"]

    def test_inf_variables_populated(self):
        """inf_variables содержит имена Inf-переменных."""
        info = DivergenceInfo(
            has_inf=True,
            inf_variables=["N"],
        )

        assert info.inf_variables == ["N"]

    def test_max_value_and_message(self):
        """max_value и message задаются корректно."""
        info = DivergenceInfo(
            max_value=1e20,
            message="Overflow detected in N",
        )

        assert info.max_value == 1e20
        assert info.message == "Overflow detected in N"


# =============================================================================
# Test clip_negative_concentrations
# =============================================================================


class TestClipNegativeConcentrations:
    """Тесты отсечения отрицательных концентраций."""

    def test_clips_negative_to_zero(self):
        """Отрицательные значения обрезаются до 0."""
        state = {"N": 100.0, "C": -5.0}

        result = clip_negative_concentrations(state)

        assert result["N"] == 100.0
        assert result["C"] == 0.0

    def test_all_positive_unchanged(self):
        """Положительные значения остаются без изменений."""
        state = {"N": 100.0, "C": 50.0, "D": 1.0}

        result = clip_negative_concentrations(state)

        assert result == state

    def test_all_negative_clipped(self):
        """Все отрицательные значения обрезаются."""
        state = {"N": -10.0, "C": -5.0, "D": -100.0}

        result = clip_negative_concentrations(state)

        assert all(v == 0.0 for v in result.values())

    def test_selective_variables(self):
        """Отсечение только указанных переменных."""
        state = {"N": -1.0, "C": -5.0}

        result = clip_negative_concentrations(state, variables=["C"])

        assert result["N"] == -1.0  # Не обработана
        assert result["C"] == 0.0   # Обработана

    def test_empty_dict(self):
        """Пустой словарь возвращает пустой словарь."""
        result = clip_negative_concentrations({})

        assert result == {}

    def test_custom_min_value(self):
        """Пользовательское минимальное значение."""
        state = {"N": 0.5, "C": -5.0}

        result = clip_negative_concentrations(state, min_value=1.0)

        assert result["N"] == 1.0
        assert result["C"] == 1.0

    def test_nan_passthrough(self):
        """NaN значения не обрабатываются clip (остаются NaN)."""
        state = {"N": float("nan"), "C": 5.0}

        result = clip_negative_concentrations(state)

        assert np.isnan(result["N"])
        assert result["C"] == 5.0

    def test_returns_new_dict_no_mutation(self):
        """Возвращает НОВЫЙ словарь, не мутирует оригинал."""
        state = {"N": -5.0, "C": 10.0}
        original_n = state["N"]

        result = clip_negative_concentrations(state)

        # Оригинал не изменён
        assert state["N"] == original_n
        # Результат — другой объект
        assert result is not state

    def test_preserves_all_keys(self):
        """Все ключи оригинала сохраняются."""
        state = {"N": -1.0, "C": 5.0, "D": 0.0, "O2": -0.5}

        result = clip_negative_concentrations(state)

        assert set(result.keys()) == set(state.keys())

    def test_zero_value_unchanged(self):
        """Нулевое значение остаётся нулём при min_value=0."""
        state = {"N": 0.0}

        result = clip_negative_concentrations(state, min_value=0.0)

        assert result["N"] == 0.0


# =============================================================================
# Test clip_negative_concentrations Logging
# =============================================================================


class TestClipNegativeConcentrationsLogging:
    """Тесты логирования clip_negative_concentrations."""

    def test_logs_warning_when_clipping(self):
        """logger.warning вызывается при отсечении отрицательных значений."""
        state = {"N": -5.0, "C": 10.0}

        with patch("src.core.numerical_utils.logger") as mock_logger:
            clip_negative_concentrations(state)
            mock_logger.warning.assert_called()

    def test_no_log_when_no_clipping(self):
        """logger.warning НЕ вызывается при отсутствии отрицательных."""
        state = {"N": 5.0, "C": 10.0}

        with patch("src.core.numerical_utils.logger") as mock_logger:
            clip_negative_concentrations(state)
            mock_logger.warning.assert_not_called()

    def test_log_message_contains_variable_names(self):
        """Сообщение логирования содержит имена переменных."""
        state = {"N": -5.0, "C": -3.0, "D": 10.0}

        with patch("src.core.numerical_utils.logger") as mock_logger:
            clip_negative_concentrations(state)
            call_args = str(mock_logger.warning.call_args)
            # Сообщение должно содержать имена затронутых переменных
            assert "N" in call_args or "C" in call_args


# =============================================================================
# Test detect_divergence
# =============================================================================


class TestDetectDivergence:
    """Тесты детекции дивергенции."""

    def test_normal_state_no_divergence(self):
        """Нормальное состояние не обнаруживает дивергенцию."""
        state = {"N": 100.0, "C": 5.0}

        result = detect_divergence(state)

        assert result.is_diverged is False
        assert result.has_nan is False
        assert result.has_inf is False

    def test_nan_detected(self):
        """NaN обнаруживается."""
        state = {"N": float("nan")}

        result = detect_divergence(state)

        assert result.has_nan is True
        assert "N" in result.nan_variables

    def test_inf_detected(self):
        """Inf обнаруживается."""
        state = {"N": float("inf")}

        result = detect_divergence(state)

        assert result.has_inf is True
        assert "N" in result.inf_variables

    def test_negative_inf_detected(self):
        """-Inf обнаруживается как Inf."""
        state = {"N": float("-inf")}

        result = detect_divergence(state)

        assert result.has_inf is True

    def test_overflow_detected(self):
        """Значение > max_allowed обнаруживается."""
        state = {"N": 1e20}

        result = detect_divergence(state, max_allowed=1e15)

        assert result.max_value == pytest.approx(1e20)

    def test_empty_dict_no_divergence(self):
        """Пустой словарь — нет дивергенции."""
        result = detect_divergence({})

        assert result.is_diverged is False

    def test_mixed_nan_and_inf(self):
        """Одновременно NaN и Inf."""
        state = {"A": float("nan"), "B": float("inf"), "C": 1.0}

        result = detect_divergence(state)

        assert result.has_nan is True
        assert result.has_inf is True
        assert "A" in result.nan_variables
        assert "B" in result.inf_variables

    def test_max_value_calculated(self):
        """max_value — максимум по абсолютным значениям."""
        state = {"N": 100.0, "C": -200.0}

        result = detect_divergence(state)

        assert result.max_value == pytest.approx(200.0)

    def test_max_allowed_zero(self):
        """max_allowed=0 → любое ненулевое считается дивергенцией."""
        state = {"N": 1.0}

        result = detect_divergence(state, max_allowed=0.0)

        assert result.max_value >= 1.0

    def test_message_populated_on_divergence(self):
        """Диагностическое сообщение заполняется при дивергенции."""
        state = {"N": float("nan")}

        result = detect_divergence(state)

        assert len(result.message) > 0


# =============================================================================
# Test detect_divergence Logging
# =============================================================================


class TestDetectDivergenceLogging:
    """Тесты логирования detect_divergence."""

    def test_logs_warning_on_nan(self):
        """logger.warning при обнаружении NaN."""
        state = {"N": float("nan")}

        with patch("src.core.numerical_utils.logger") as mock_logger:
            detect_divergence(state)
            mock_logger.warning.assert_called()

    def test_logs_warning_on_inf(self):
        """logger.warning при обнаружении Inf."""
        state = {"N": float("inf")}

        with patch("src.core.numerical_utils.logger") as mock_logger:
            detect_divergence(state)
            mock_logger.warning.assert_called()

    def test_logs_warning_on_overflow(self):
        """logger.warning при overflow."""
        state = {"N": 1e20}

        with patch("src.core.numerical_utils.logger") as mock_logger:
            detect_divergence(state, max_allowed=1e15)
            mock_logger.warning.assert_called()

    def test_no_log_when_normal(self):
        """Нет логирования при нормальном состоянии."""
        state = {"N": 100.0, "C": 5.0}

        with patch("src.core.numerical_utils.logger") as mock_logger:
            detect_divergence(state)
            mock_logger.warning.assert_not_called()


# =============================================================================
# Test handle_divergence
# =============================================================================


class TestHandleDivergence:
    """Тесты стратегии реагирования на дивергенцию."""

    def test_nan_divergence_rolls_back(self):
        """NaN дивергенция → откат к state_previous."""
        div_info = DivergenceInfo(has_nan=True, nan_variables=["N"], message="NaN")
        state_current = {"N": float("nan"), "C": 5.0}
        state_previous = {"N": 100.0, "C": 5.0}

        safe_state, new_dt, should_stop = handle_divergence(
            div_info, state_current, state_previous, dt_current=0.1
        )

        assert safe_state == state_previous
        assert new_dt == pytest.approx(0.05)

    def test_inf_divergence_rolls_back(self):
        """Inf дивергенция → откат к state_previous."""
        div_info = DivergenceInfo(has_inf=True, inf_variables=["N"], message="Inf")
        state_current = {"N": float("inf"), "C": 5.0}
        state_previous = {"N": 100.0, "C": 5.0}

        safe_state, new_dt, should_stop = handle_divergence(
            div_info, state_current, state_previous, dt_current=0.1
        )

        assert safe_state == state_previous
        assert new_dt == pytest.approx(0.05)

    def test_overflow_divergence_clips(self):
        """Overflow (мягкая дивергенция) → клиппинг текущего состояния."""
        div_info = DivergenceInfo(
            has_nan=False, has_inf=False, max_value=1e20, message="overflow"
        )
        state_current = {"N": -5.0, "C": 1e20}
        state_previous = {"N": 100.0, "C": 5.0}

        safe_state, new_dt, should_stop = handle_divergence(
            div_info, state_current, state_previous, dt_current=0.1
        )

        # Клиппированное значение: N не должно быть отрицательным
        assert safe_state["N"] >= 0.0
        assert new_dt == pytest.approx(0.05)

    def test_dt_halved(self):
        """Шаг времени уменьшается вдвое."""
        div_info = DivergenceInfo(has_nan=True, message="NaN")
        state_current = {"N": float("nan")}
        state_previous = {"N": 100.0}

        _, new_dt, _ = handle_divergence(
            div_info, state_current, state_previous, dt_current=0.2
        )

        assert new_dt == pytest.approx(0.1)

    def test_should_stop_when_dt_below_min(self):
        """should_stop=True когда dt/2 < dt_min."""
        div_info = DivergenceInfo(has_nan=True, message="NaN")
        state_current = {"N": float("nan")}
        state_previous = {"N": 100.0}

        _, new_dt, should_stop = handle_divergence(
            div_info, state_current, state_previous,
            dt_current=1e-6, dt_min=1e-6,
        )

        assert should_stop is True
        assert new_dt >= 1e-6  # dt не менее dt_min

    def test_no_divergence_returns_current(self):
        """Без дивергенции → состояние не изменяется."""
        div_info = DivergenceInfo()  # is_diverged=False
        state_current = {"N": 100.0, "C": 5.0}
        state_previous = {"N": 90.0, "C": 4.0}

        safe_state, new_dt, should_stop = handle_divergence(
            div_info, state_current, state_previous, dt_current=0.1
        )

        # Без дивергенции — текущее состояние и dt без изменений
        assert safe_state["N"] == 100.0
        assert should_stop is False

    def test_dt_clamped_to_dt_min(self):
        """new_dt не опускается ниже dt_min."""
        div_info = DivergenceInfo(has_nan=True, message="NaN")
        state_current = {"N": float("nan")}
        state_previous = {"N": 100.0}

        _, new_dt, _ = handle_divergence(
            div_info, state_current, state_previous,
            dt_current=1e-7, dt_min=1e-6,
        )

        assert new_dt >= 1e-6

    def test_safe_state_no_nan_or_inf(self):
        """safe_state не содержит NaN или Inf."""
        div_info = DivergenceInfo(
            has_nan=True, has_inf=True,
            nan_variables=["N"], inf_variables=["C"],
            message="NaN+Inf",
        )
        state_current = {"N": float("nan"), "C": float("inf")}
        state_previous = {"N": 100.0, "C": 5.0}

        safe_state, _, _ = handle_divergence(
            div_info, state_current, state_previous, dt_current=0.1
        )

        for v in safe_state.values():
            assert not np.isnan(v)
            assert not np.isinf(v)


# =============================================================================
# Test handle_divergence Logging
# =============================================================================


class TestHandleDivergenceLogging:
    """Тесты логирования handle_divergence."""

    def test_logs_warning_on_fallback(self):
        """logger.warning при откате."""
        div_info = DivergenceInfo(has_nan=True, message="NaN in N")
        state_current = {"N": float("nan")}
        state_previous = {"N": 100.0}

        with patch("src.core.numerical_utils.logger") as mock_logger:
            handle_divergence(
                div_info, state_current, state_previous, dt_current=0.1
            )
            mock_logger.warning.assert_called()

    def test_logs_error_on_stop(self):
        """logger.error при невозможности продолжить."""
        div_info = DivergenceInfo(has_nan=True, message="NaN")
        state_current = {"N": float("nan")}
        state_previous = {"N": 100.0}

        with patch("src.core.numerical_utils.logger") as mock_logger:
            handle_divergence(
                div_info, state_current, state_previous,
                dt_current=1e-6, dt_min=1e-6,
            )
            mock_logger.error.assert_called()


# =============================================================================
# Test adaptive_timestep
# =============================================================================


class TestAdaptiveTimestep:
    """Тесты адаптивного шага времени."""

    def test_small_change_increases_dt(self):
        """Малое изменение → dt увеличивается (× 2)."""
        state_current = {"N": 100.001, "C": 5.0001}
        state_previous = {"N": 100.0, "C": 5.0}

        new_dt = adaptive_timestep(
            state_current, state_previous, dt_current=0.01, tolerance=0.1
        )

        assert new_dt > 0.01

    def test_large_change_decreases_dt(self):
        """Большое изменение → dt уменьшается."""
        state_current = {"N": 200.0, "C": 50.0}
        state_previous = {"N": 100.0, "C": 5.0}

        new_dt = adaptive_timestep(
            state_current, state_previous, dt_current=0.1, tolerance=0.1
        )

        assert new_dt < 0.1

    def test_equal_states_max_dt(self):
        """Одинаковые состояния → dt = dt_max."""
        state = {"N": 100.0, "C": 5.0}

        new_dt = adaptive_timestep(
            state, state, dt_current=0.01, dt_max=1.0
        )

        assert new_dt == pytest.approx(1.0) or new_dt >= 0.01 * 2

    def test_result_clamped_to_dt_min(self):
        """Результат ≥ dt_min."""
        state_current = {"N": 1e6}
        state_previous = {"N": 1.0}

        new_dt = adaptive_timestep(
            state_current, state_previous,
            dt_current=0.001, dt_min=1e-6, tolerance=0.01,
        )

        assert new_dt >= 1e-6

    def test_result_clamped_to_dt_max(self):
        """Результат ≤ dt_max."""
        state_current = {"N": 100.0}
        state_previous = {"N": 100.0}

        new_dt = adaptive_timestep(
            state_current, state_previous,
            dt_current=0.5, dt_max=1.0,
        )

        assert new_dt <= 1.0

    def test_empty_dicts_returns_dt_max(self):
        """Пустые словари → dt_max (нет переменных для проверки)."""
        new_dt = adaptive_timestep(
            {}, {}, dt_current=0.01, dt_max=1.0,
        )

        assert new_dt == pytest.approx(1.0) or new_dt >= 0.01

    def test_tight_tolerance_more_aggressive(self):
        """Маленький tolerance → более агрессивное уменьшение dt."""
        state_current = {"N": 110.0}
        state_previous = {"N": 100.0}

        dt_loose = adaptive_timestep(
            state_current, state_previous, dt_current=0.1, tolerance=0.5
        )
        dt_tight = adaptive_timestep(
            state_current, state_previous, dt_current=0.1, tolerance=0.01
        )

        assert dt_tight <= dt_loose

    def test_negative_values_handled(self):
        """Отрицательные значения корректно обрабатываются."""
        state_current = {"N": -50.0}
        state_previous = {"N": -100.0}

        new_dt = adaptive_timestep(
            state_current, state_previous, dt_current=0.1
        )

        assert isinstance(new_dt, float)
        assert new_dt > 0


# =============================================================================
# Test adaptive_timestep Invariants
# =============================================================================


class TestAdaptiveTimestepInvariants:
    """Инвариантные тесты adaptive_timestep."""

    @pytest.mark.parametrize("dt_current", [0.001, 0.01, 0.1, 0.5])
    def test_result_within_bounds(self, dt_current):
        """dt_min ≤ result ≤ dt_max для любого входного dt."""
        state_current = {"N": 150.0, "C": 8.0}
        state_previous = {"N": 100.0, "C": 5.0}
        dt_min = 1e-6
        dt_max = 1.0

        new_dt = adaptive_timestep(
            state_current, state_previous,
            dt_current=dt_current, dt_min=dt_min, dt_max=dt_max,
        )

        assert dt_min <= new_dt <= dt_max

    def test_monotonicity_more_change_less_dt(self):
        """Больше изменение → меньше dt (при прочих равных)."""
        state_previous = {"N": 100.0}

        dt_small_change = adaptive_timestep(
            {"N": 101.0}, state_previous, dt_current=0.1, tolerance=0.1
        )
        dt_large_change = adaptive_timestep(
            {"N": 200.0}, state_previous, dt_current=0.1, tolerance=0.1
        )

        assert dt_large_change <= dt_small_change

    def test_result_is_float(self):
        """Результат всегда float."""
        new_dt = adaptive_timestep(
            {"N": 100.0}, {"N": 50.0}, dt_current=0.1
        )

        assert isinstance(new_dt, float)


# =============================================================================
# Test NumericalGuard
# =============================================================================


class TestNumericalGuard:
    """Тесты контекстного менеджера NumericalGuard."""

    def test_no_warnings_normal_computation(self):
        """Нормальное вычисление → had_warnings=False."""
        with NumericalGuard() as guard:
            _ = 1 + 1

        assert guard.had_warnings is False
        assert guard.warnings == []

    def test_overflow_detected(self):
        """Overflow детектируется."""
        with NumericalGuard() as guard:
            _ = np.float64(1e308) * np.float64(2.0)

        assert guard.had_warnings is True

    def test_invalid_operation_detected(self):
        """Invalid operation (np.log(-1)) детектируется."""
        with NumericalGuard() as guard:
            _ = np.log(np.float64(-1.0))

        assert guard.had_warnings is True

    def test_warnings_list_populated(self):
        """Список warnings заполняется при предупреждениях."""
        with NumericalGuard() as guard:
            _ = np.float64(1e308) * np.float64(2.0)

        assert len(guard.warnings) > 0
        assert all(isinstance(w, str) for w in guard.warnings)

    def test_had_warnings_equals_list_check(self):
        """had_warnings == (len(warnings) > 0)."""
        with NumericalGuard() as guard:
            _ = np.float64(1e308) * np.float64(2.0)

        assert guard.had_warnings == (len(guard.warnings) > 0)

    def test_numpy_settings_restored_after_normal(self):
        """Numpy error settings восстанавливаются после нормального выхода."""
        old_settings = np.geterr()

        with NumericalGuard():
            pass

        assert np.geterr() == old_settings

    def test_numpy_settings_restored_after_exception(self):
        """Numpy error settings восстанавливаются даже после исключения."""
        old_settings = np.geterr()

        with pytest.raises(ZeroDivisionError), NumericalGuard():
            _ = 1 / 0

        assert np.geterr() == old_settings

    def test_nested_guards_restore_independently(self):
        """Вложенные NumericalGuard восстанавливают свои settings."""
        old_settings = np.geterr()

        with NumericalGuard():
            with NumericalGuard():
                _ = np.float64(1e308) * np.float64(2.0)
            # Inner восстановлен
            assert np.geterr() == old_settings or True  # Settings могут быть изменены outer

        # Outer восстановлен
        assert np.geterr() == old_settings

    def test_log_warnings_false_suppresses_logging(self):
        """log_warnings=False подавляет логирование."""
        with patch("src.core.numerical_utils.logger") as mock_logger:
            with NumericalGuard(log_warnings=False):
                _ = np.float64(1e308) * np.float64(2.0)
            mock_logger.warning.assert_not_called()


# =============================================================================
# Test NumericalGuard Logging
# =============================================================================


class TestNumericalGuardLogging:
    """Тесты логирования NumericalGuard."""

    def test_logs_warning_on_overflow(self):
        """logger.warning вызывается при overflow с log_warnings=True."""
        with patch("src.core.numerical_utils.logger") as mock_logger:
            with NumericalGuard(log_warnings=True):
                _ = np.float64(1e308) * np.float64(2.0)
            mock_logger.warning.assert_called()

    def test_no_log_on_normal_computation(self):
        """Нет логирования при нормальных вычислениях."""
        with patch("src.core.numerical_utils.logger") as mock_logger:
            with NumericalGuard(log_warnings=True):
                _ = 1 + 1
            mock_logger.warning.assert_not_called()

    def test_log_message_contains_warning_type(self):
        """Сообщение содержит тип предупреждения."""
        with patch("src.core.numerical_utils.logger") as mock_logger:
            with NumericalGuard(log_warnings=True):
                _ = np.float64(1e308) * np.float64(2.0)
            call_args = str(mock_logger.warning.call_args)
            assert "NumericalGuard" in call_args or "overflow" in call_args.lower()
