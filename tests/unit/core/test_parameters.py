"""TDD тесты для parameters.py — ParameterSet.

Тестирование:
- ParameterSet: значения по умолчанию из §8 Mathematical Framework
- validate(): физическая осмысленность параметров
- to_dict() / from_dict(): сериализация и десериализация
- get_literature_defaults(): фабричный метод

Все тесты написаны для stub-реализации (NotImplementedError).
После реализации методов тесты должны проходить.
"""

import dataclasses

import pytest

from src.core.parameters import ParameterSet

# =============================================================================
# TestParameterSetDefaults
# =============================================================================


class TestParameterSetDefaults:
    """Тесты значений по умолчанию из §8 Mathematical Framework."""

    def test_proliferation_defaults(self):
        """Скорости пролиферации соответствуют литературным значениям."""
        ps = ParameterSet()
        assert ps.r_F == 0.03
        assert ps.r_E == 0.02
        assert ps.r_S == 0.01

    def test_death_rate_defaults(self):
        """Скорости апоптоза соответствуют литературным значениям."""
        ps = ParameterSet()
        assert ps.delta_P == 0.1
        assert ps.delta_Ne == 0.05
        assert ps.delta_M == 0.01
        assert ps.delta_F == 0.003
        assert ps.delta_Mf == 0.01
        assert ps.delta_E == 0.005
        assert ps.delta_S == 0.005

    def test_switching_defaults(self):
        """Параметры переключения M1/M2 и активации F→Mf."""
        ps = ParameterSet()
        assert ps.k_switch == 0.02
        assert ps.k_reverse == 0.005
        assert ps.k_act == 0.01

    def test_carrying_capacity_defaults(self):
        """Carrying capacity для клеточных популяций."""
        ps = ParameterSet()
        assert ps.K_F == 5e5
        assert ps.K_E == 1e5
        assert ps.K_S == 1e4

    def test_platelet_defaults(self):
        """Параметры тромбоцитов."""
        ps = ParameterSet()
        assert ps.P_max == 1e4
        assert ps.tau_P == 2.0
        assert ps.k_deg == 0.05

    def test_cytokine_degradation_defaults(self):
        """Скорости деградации цитокинов."""
        ps = ParameterSet()
        assert ps.gamma_TNF == 0.5
        assert ps.gamma_IL10 == 0.3
        assert ps.gamma_PDGF == 0.2
        assert ps.gamma_VEGF == 0.3
        assert ps.gamma_TGF == 0.15
        assert ps.gamma_MCP1 == 0.4
        assert ps.gamma_IL8 == 0.5
        assert ps.gamma_MMP == 0.1

    def test_sigma_defaults(self):
        """Параметры шума (sigma) для всех переменных."""
        ps = ParameterSet()
        assert ps.sigma_P == 0.05
        assert ps.sigma_Ne == 0.05
        assert ps.sigma_M == 0.03
        assert ps.sigma_F == 0.02
        assert ps.sigma_Mf == 0.02
        assert ps.sigma_E == 0.02
        assert ps.sigma_S == 0.02
        assert ps.sigma_TNF == 0.05
        assert ps.sigma_IL10 == 0.03
        assert ps.sigma_PDGF == 0.03
        assert ps.sigma_VEGF == 0.03
        assert ps.sigma_TGF == 0.02
        assert ps.sigma_MCP1 == 0.05
        assert ps.sigma_IL8 == 0.05

    def test_numerical_defaults(self):
        """Численные параметры: шаг, время, epsilon."""
        ps = ParameterSet()
        assert ps.dt == 0.01
        assert ps.t_max == 720.0
        assert ps.epsilon == 1e-10

    def test_total_field_count(self):
        """ParameterSet содержит >= 90 полей."""
        fields = dataclasses.fields(ParameterSet)
        assert len(fields) >= 90


# =============================================================================
# TestParameterSetValidate
# =============================================================================


class TestParameterSetValidate:
    """Тесты валидации физической осмысленности параметров."""

    def test_validate_defaults_returns_true(self):
        """Параметры по умолчанию валидны."""
        ps = ParameterSet()
        assert ps.validate() is True

    def test_validate_negative_proliferation_rate(self):
        """Отрицательная скорость пролиферации -> ValueError."""
        ps = ParameterSet(r_F=-0.01)
        with pytest.raises(ValueError, match="r_F"):
            ps.validate()

    def test_validate_zero_carrying_capacity(self):
        """Нулевая carrying capacity -> ValueError."""
        ps = ParameterSet(K_F=0)
        with pytest.raises(ValueError, match="K_F"):
            ps.validate()

    def test_validate_negative_sigma(self):
        """Отрицательный sigma -> ValueError."""
        ps = ParameterSet(sigma_P=-1)
        with pytest.raises(ValueError, match="sigma_P"):
            ps.validate()

    def test_validate_zero_dt(self):
        """Нулевой шаг времени -> ValueError."""
        ps = ParameterSet(dt=0)
        with pytest.raises(ValueError, match="dt"):
            ps.validate()

    def test_validate_zero_degradation_rate(self):
        """Нулевая скорость деградации -> ValueError."""
        ps = ParameterSet(gamma_TNF=0)
        with pytest.raises(ValueError, match="gamma_TNF"):
            ps.validate()

    def test_validate_zero_hill_coefficient(self):
        """Нулевой коэффициент Хилла -> ValueError."""
        ps = ParameterSet(n_hill=0)
        with pytest.raises(ValueError, match="n_hill"):
            ps.validate()

    def test_validate_negative_half_saturation(self):
        """Отрицательная константа полунасыщения -> ValueError."""
        ps = ParameterSet(K_IL8=-1.0)
        with pytest.raises(ValueError, match="K_IL8"):
            ps.validate()

    def test_validate_very_large_values_ok(self):
        """Очень большие значения допустимы (нет верхней границы)."""
        ps = ParameterSet(r_F=1e15)
        assert ps.validate() is True

    def test_validate_error_contains_param_name(self):
        """Сообщение ValueError содержит имя нарушенного параметра."""
        ps = ParameterSet(delta_P=-0.5)
        with pytest.raises(ValueError) as exc_info:
            ps.validate()
        assert "delta_P" in str(exc_info.value)

    def test_validate_negative_death_rate(self):
        """Отрицательная скорость смерти -> ValueError."""
        ps = ParameterSet(delta_P=-0.01)
        with pytest.raises(ValueError, match="delta_P"):
            ps.validate()

    def test_validate_negative_epsilon(self):
        """Отрицательный epsilon -> ValueError."""
        ps = ParameterSet(epsilon=-1e-10)
        with pytest.raises(ValueError, match="epsilon"):
            ps.validate()


# =============================================================================
# TestParameterSetToDict
# =============================================================================


class TestParameterSetToDict:
    """Тесты сериализации ParameterSet в словарь."""

    def test_to_dict_returns_dict(self):
        """to_dict() возвращает dict."""
        ps = ParameterSet()
        result = ps.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_key_count(self):
        """Словарь содержит >= 90 ключей."""
        ps = ParameterSet()
        result = ps.to_dict()
        assert len(result) >= 90

    def test_to_dict_r_F_value(self):
        """Значение r_F в словаре соответствует полю."""
        ps = ParameterSet()
        result = ps.to_dict()
        assert result["r_F"] == 0.03

    def test_to_dict_n_hill_is_int(self):
        """n_hill в словаре имеет тип int."""
        ps = ParameterSet()
        result = ps.to_dict()
        assert isinstance(result["n_hill"], int)
        assert result["n_hill"] == 2

    def test_to_dict_all_values_numeric(self):
        """Все значения в словаре — float или int."""
        ps = ParameterSet()
        result = ps.to_dict()
        for key, value in result.items():
            assert isinstance(value, (float, int)), (
                f"Значение {key}={value} не float/int"
            )

    def test_to_dict_matches_fields(self):
        """Каждое значение в словаре совпадает с полем dataclass."""
        ps = ParameterSet()
        result = ps.to_dict()
        for f in dataclasses.fields(ps):
            assert result[f.name] == getattr(ps, f.name), (
                f"Несовпадение для {f.name}"
            )


# =============================================================================
# TestParameterSetFromDict
# =============================================================================


class TestParameterSetFromDict:
    """Тесты десериализации ParameterSet из словаря."""

    def test_from_dict_empty_equals_defaults(self):
        """Пустой словарь -> ParameterSet с defaults."""
        result = ParameterSet.from_dict({})
        expected = ParameterSet()
        assert result == expected

    def test_from_dict_overrides_one_field(self):
        """Словарь с одним ключом переопределяет одно поле."""
        result = ParameterSet.from_dict({"r_F": 0.05})
        assert result.r_F == 0.05

    def test_from_dict_ignores_unknown_keys(self):
        """Неизвестные ключи игнорируются."""
        result = ParameterSet.from_dict({"unknown_key": 42})
        expected = ParameterSet()
        assert result == expected

    def test_from_dict_round_trip(self):
        """Round-trip: from_dict(ps.to_dict()) == ps."""
        ps = ParameterSet(r_F=0.05, delta_P=0.2, sigma_P=0.1)
        restored = ParameterSet.from_dict(ps.to_dict())
        assert restored == ps

    def test_from_dict_invalid_type_raises(self):
        """Некорректный тип значения -> TypeError."""
        with pytest.raises(TypeError):
            ParameterSet.from_dict({"r_F": "not_a_number"})

    def test_from_dict_preserves_unspecified(self):
        """Неуказанные поля сохраняют значения по умолчанию."""
        result = ParameterSet.from_dict({"r_F": 0.05})
        assert result.delta_P == 0.1
        assert result.K_F == 5e5


# =============================================================================
# TestGetLiteratureDefaults
# =============================================================================


class TestGetLiteratureDefaults:
    """Тесты фабричного метода get_literature_defaults."""

    def test_returns_parameter_set(self):
        """Возвращает экземпляр ParameterSet."""
        result = ParameterSet.get_literature_defaults()
        assert isinstance(result, ParameterSet)

    def test_equals_default_constructor(self):
        """Идентичен ParameterSet() (все defaults литературные)."""
        result = ParameterSet.get_literature_defaults()
        assert result == ParameterSet()

    def test_specific_literature_value(self):
        """Конкретное литературное значение r_F = 0.03."""
        result = ParameterSet.get_literature_defaults()
        assert result.r_F == 0.03
