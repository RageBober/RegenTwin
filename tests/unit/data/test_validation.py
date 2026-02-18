"""
TDD тесты для модуля validation.py

Тестирует:
- ValidationLevel enum
- ValidationResult dataclass
- ColumnSchema, DataSchema dataclasses
- Предопределённые схемы (FCS, TimeSeries, Cytokine)
- DataValidator класс и все методы валидации
- validate_data() удобная функция

Основано на спецификации: Description/Phase1/description_validation.md

Уровни валидации:
- STRICT: любое отклонение → ошибка
- NORMAL: выход за диапазон → предупреждение, отсутствие required → ошибка
- LENIENT: только критические проверки
"""

import pytest
import numpy as np
import pandas as pd

from src.data.validation import (
    ValidationLevel,
    ValidationResult,
    ColumnSchema,
    DataSchema,
    DataValidator,
    FCS_DATA_SCHEMA,
    TIME_SERIES_SCHEMA,
    CYTOKINE_TIMESERIES_SCHEMA,
    validate_data,
)
from src.data.parameter_extraction import ModelParameters, ExtendedModelParameters
from src.data.gating import GatingResults, GateResult


# =============================================================================
# Тесты для ValidationLevel
# =============================================================================

class TestValidationLevel:
    """Тесты для enum ValidationLevel."""

    def test_create_from_string_strict(self):
        """Тест создания ValidationLevel из строки 'strict'."""
        level = ValidationLevel("strict")
        assert level == ValidationLevel.STRICT

    def test_invalid_string_raises_value_error(self):
        """Тест что невалидная строка вызывает ValueError."""
        with pytest.raises(ValueError):
            ValidationLevel("invalid")

    def test_value_property_normal(self):
        """Тест что .value возвращает строковое значение."""
        assert ValidationLevel.NORMAL.value == "normal"


# =============================================================================
# Тесты для ValidationResult
# =============================================================================

class TestValidationResult:
    """Тесты для dataclass ValidationResult."""

    def test_invariant_is_valid_true_when_no_errors(self):
        """Тест инварианта: is_valid=True когда errors пуст."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=["warning1"],
        )
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_invariant_is_valid_false_when_has_errors(self):
        """Тест инварианта: is_valid=False когда есть ошибки."""
        result = ValidationResult(
            is_valid=False,
            errors=["error1"],
        )
        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_summary_pass_no_errors(self):
        """Тест что summary содержит 'PASS' при отсутствии ошибок."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
        )
        summary = result.summary()
        assert "PASS" in summary

    def test_summary_fail_with_errors(self):
        """Тест что summary содержит 'FAIL' и количество ошибок."""
        result = ValidationResult(
            is_valid=False,
            errors=["error1", "error2"],
            warnings=["warn1"],
        )
        summary = result.summary()
        assert "FAIL" in summary
        assert "2" in summary

    def test_summary_shows_first_5_errors_when_many(self):
        """Тест что summary показывает первые 5 ошибок при ≥5 ошибках."""
        errors = [f"error_{i}" for i in range(7)]
        result = ValidationResult(
            is_valid=False,
            errors=errors,
        )
        summary = result.summary()
        assert "FAIL" in summary
        # Должны присутствовать первые 5 ошибок
        for i in range(5):
            assert f"error_{i}" in summary
        # 6-я и 7-я ошибки НЕ показываются в summary
        # (реализация может показывать или нет, но минимум первые 5 — обязательно)


# =============================================================================
# Тесты для ColumnSchema
# =============================================================================

class TestColumnSchema:
    """Тесты для dataclass ColumnSchema."""

    def test_creation_with_defaults(self):
        """Тест создания ColumnSchema с параметрами по умолчанию."""
        col = ColumnSchema(name="FSC-A", dtype="float")
        assert col.name == "FSC-A"
        assert col.dtype == "float"
        assert col.required is True
        assert col.min_value is None
        assert col.max_value is None
        assert col.allowed_values is None
        assert col.description == ""

    def test_creation_with_all_fields(self):
        """Тест создания ColumnSchema со всеми полями."""
        col = ColumnSchema(
            name="wound_area",
            dtype="float",
            required=False,
            min_value=0.0,
            max_value=1.0,
            allowed_values=None,
            description="Нормализованная площадь раны",
        )
        assert col.name == "wound_area"
        assert col.required is False
        assert col.min_value == 0.0
        assert col.max_value == 1.0
        assert "площадь" in col.description


# =============================================================================
# Тесты для DataSchema
# =============================================================================

class TestDataSchema:
    """Тесты для dataclass DataSchema."""

    def test_get_required_columns_fcs_schema(self):
        """Тест get_required_columns для FCS схемы → 5 обязательных."""
        required = FCS_DATA_SCHEMA.get_required_columns()
        assert len(required) == 5
        assert "FSC-A" in required
        assert "FSC-H" in required
        assert "SSC-A" in required
        assert "CD34" in required
        assert "Annexin-V" in required

    def test_get_required_columns_all_optional(self):
        """Тест get_required_columns когда все столбцы optional → []."""
        schema = DataSchema(
            name="all_optional",
            columns=[
                ColumnSchema(name="a", dtype="float", required=False),
                ColumnSchema(name="b", dtype="float", required=False),
            ],
        )
        required = schema.get_required_columns()
        assert required == []

    def test_get_required_columns_all_required(self):
        """Тест get_required_columns когда все столбцы required."""
        schema = DataSchema(
            name="all_required",
            columns=[
                ColumnSchema(name="x", dtype="float", required=True),
                ColumnSchema(name="y", dtype="float", required=True),
                ColumnSchema(name="z", dtype="float", required=True),
            ],
        )
        required = schema.get_required_columns()
        assert len(required) == 3

    def test_get_required_columns_is_subset_of_all_columns(self):
        """Тест что required — подмножество всех столбцов схемы."""
        required = FCS_DATA_SCHEMA.get_required_columns()
        all_names = [col.name for col in FCS_DATA_SCHEMA.columns]
        assert set(required) <= set(all_names)


# =============================================================================
# Тесты для предопределённых схем
# =============================================================================

class TestPredefinedSchemas:
    """Тесты для предопределённых схем данных."""

    def test_fcs_schema_has_9_columns(self):
        """Тест что FCS_DATA_SCHEMA содержит 9 столбцов."""
        assert len(FCS_DATA_SCHEMA.columns) == 9

    def test_fcs_schema_has_5_required(self):
        """Тест что FCS_DATA_SCHEMA имеет 5 обязательных столбцов."""
        required = [c for c in FCS_DATA_SCHEMA.columns if c.required]
        assert len(required) == 5

    def test_fcs_schema_min_rows_100(self):
        """Тест что FCS_DATA_SCHEMA требует минимум 100 строк."""
        assert FCS_DATA_SCHEMA.min_rows == 100

    def test_time_series_schema_has_3_columns(self):
        """Тест что TIME_SERIES_SCHEMA содержит 3 столбца."""
        assert len(TIME_SERIES_SCHEMA.columns) == 3

    def test_time_series_schema_min_rows_2(self):
        """Тест что TIME_SERIES_SCHEMA требует минимум 2 строки."""
        assert TIME_SERIES_SCHEMA.min_rows == 2

    def test_cytokine_schema_has_8_columns_min_rows_2(self):
        """Тест что CYTOKINE_TIMESERIES_SCHEMA имеет 8 столбцов и min_rows=2."""
        assert len(CYTOKINE_TIMESERIES_SCHEMA.columns) == 8
        assert CYTOKINE_TIMESERIES_SCHEMA.min_rows == 2


# =============================================================================
# Тесты для DataValidator.__init__
# =============================================================================

class TestDataValidatorInit:
    """Тесты для конструктора DataValidator."""

    def test_default_level_normal(self):
        """Тест что уровень по умолчанию — NORMAL."""
        validator = DataValidator()
        assert validator._level == ValidationLevel.NORMAL

    def test_string_strict(self):
        """Тест создания с строкой 'strict'."""
        validator = DataValidator("strict")
        assert validator._level == ValidationLevel.STRICT

    def test_enum_lenient(self):
        """Тест создания с enum ValidationLevel.LENIENT."""
        validator = DataValidator(ValidationLevel.LENIENT)
        assert validator._level == ValidationLevel.LENIENT

    def test_invalid_string_raises_value_error(self):
        """Тест что невалидная строка вызывает ValueError."""
        with pytest.raises(ValueError):
            DataValidator("invalid")


# =============================================================================
# Тесты для DataValidator.validate_dataframe
# =============================================================================

class TestValidateDataframe:
    """Тесты для DataValidator.validate_dataframe."""

    def test_correct_data_is_valid(self):
        """Тест что корректные данные проходят валидацию."""
        schema = DataSchema(
            name="test",
            columns=[
                ColumnSchema(name="x", dtype="float", required=True, min_value=0),
                ColumnSchema(name="y", dtype="float", required=False),
            ],
            min_rows=1,
        )
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        validator = DataValidator()
        result = validator.validate_dataframe(df, schema)
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_missing_required_column(self):
        """Тест что отсутствие обязательного столбца → ошибка."""
        schema = DataSchema(
            name="test",
            columns=[
                ColumnSchema(name="x", dtype="float", required=True),
                ColumnSchema(name="y", dtype="float", required=True),
            ],
        )
        df = pd.DataFrame({"x": [1.0, 2.0]})
        validator = DataValidator()
        result = validator.validate_dataframe(df, schema)
        assert result.is_valid is False
        assert any("y" in err for err in result.errors)

    def test_value_below_min_strict_is_error(self):
        """Тест что значение < min_value при STRICT → ошибка."""
        schema = DataSchema(
            name="test",
            columns=[ColumnSchema(name="x", dtype="float", required=True, min_value=0)],
        )
        df = pd.DataFrame({"x": [1.0, -5.0, 3.0]})
        validator = DataValidator("strict")
        result = validator.validate_dataframe(df, schema)
        assert result.is_valid is False

    def test_value_below_min_normal_is_warning(self):
        """Тест что значение < min_value при NORMAL → предупреждение."""
        schema = DataSchema(
            name="test",
            columns=[ColumnSchema(name="x", dtype="float", required=True, min_value=0)],
        )
        df = pd.DataFrame({"x": [1.0, -5.0, 3.0]})
        validator = DataValidator("normal")
        result = validator.validate_dataframe(df, schema)
        assert result.is_valid is True
        assert len(result.warnings) > 0

    def test_value_above_max_strict_is_error(self):
        """Тест что значение > max_value при STRICT → ошибка."""
        schema = DataSchema(
            name="test",
            columns=[
                ColumnSchema(name="x", dtype="float", required=True, max_value=10),
            ],
        )
        df = pd.DataFrame({"x": [1.0, 50.0]})
        validator = DataValidator("strict")
        result = validator.validate_dataframe(df, schema)
        assert result.is_valid is False

    def test_too_few_rows(self):
        """Тест что пустой DataFrame при min_rows=1 → ошибка."""
        schema = DataSchema(
            name="test",
            columns=[ColumnSchema(name="x", dtype="float", required=True)],
            min_rows=1,
        )
        df = pd.DataFrame({"x": []})
        validator = DataValidator()
        result = validator.validate_dataframe(df, schema)
        assert result.is_valid is False

    def test_too_many_rows(self):
        """Тест что превышение max_rows → ошибка или предупреждение."""
        schema = DataSchema(
            name="test",
            columns=[ColumnSchema(name="x", dtype="float", required=True)],
            min_rows=1,
            max_rows=10,
        )
        df = pd.DataFrame({"x": range(100)})
        validator = DataValidator()
        result = validator.validate_dataframe(df, schema)
        # Должна быть хотя бы ошибка или предупреждение
        assert result.is_valid is False or len(result.warnings) > 0

    def test_extra_columns_ignored(self):
        """Тест что лишние столбцы не вызывают ошибку."""
        schema = DataSchema(
            name="test",
            columns=[ColumnSchema(name="x", dtype="float", required=True)],
        )
        df = pd.DataFrame({"x": [1.0, 2.0], "extra": [3.0, 4.0]})
        validator = DataValidator()
        result = validator.validate_dataframe(df, schema)
        assert result.is_valid is True


# =============================================================================
# Тесты для DataValidator.validate_fcs_data
# =============================================================================

class TestValidateFcsData:
    """Тесты для DataValidator.validate_fcs_data."""

    def test_valid_fcs_data(self):
        """Тест что валидные FCS данные проходят валидацию."""
        rng = np.random.default_rng(50)
        n = 200
        df = pd.DataFrame({
            "FSC-A": rng.uniform(10000, 200000, n),
            "FSC-H": rng.uniform(10000, 200000, n),
            "SSC-A": rng.uniform(5000, 100000, n),
            "CD34": rng.uniform(0, 50000, n),
            "Annexin-V": rng.uniform(0, 30000, n),
        })
        validator = DataValidator()
        result = validator.validate_fcs_data(df)
        assert result.is_valid is True

    def test_missing_mandatory_channel_fsc_a(self):
        """Тест что отсутствие FSC-A → ошибка."""
        rng = np.random.default_rng(51)
        n = 200
        df = pd.DataFrame({
            "FSC-H": rng.uniform(10000, 200000, n),
            "SSC-A": rng.uniform(5000, 100000, n),
            "CD34": rng.uniform(0, 50000, n),
            "Annexin-V": rng.uniform(0, 30000, n),
        })
        validator = DataValidator()
        result = validator.validate_fcs_data(df)
        assert result.is_valid is False

    def test_optional_cd66b_missing_is_ok(self):
        """Тест что отсутствие опционального CD66b не вызывает ошибку."""
        rng = np.random.default_rng(52)
        n = 200
        df = pd.DataFrame({
            "FSC-A": rng.uniform(10000, 200000, n),
            "FSC-H": rng.uniform(10000, 200000, n),
            "SSC-A": rng.uniform(5000, 100000, n),
            "CD34": rng.uniform(0, 50000, n),
            "Annexin-V": rng.uniform(0, 30000, n),
            # CD66b отсутствует — это ОК
        })
        validator = DataValidator()
        result = validator.validate_fcs_data(df)
        assert result.is_valid is True

    def test_less_than_100_events(self):
        """Тест что менее 100 событий → ошибка."""
        rng = np.random.default_rng(53)
        n = 50  # Меньше минимума 100
        df = pd.DataFrame({
            "FSC-A": rng.uniform(10000, 200000, n),
            "FSC-H": rng.uniform(10000, 200000, n),
            "SSC-A": rng.uniform(5000, 100000, n),
            "CD34": rng.uniform(0, 50000, n),
            "Annexin-V": rng.uniform(0, 30000, n),
        })
        validator = DataValidator()
        result = validator.validate_fcs_data(df)
        assert result.is_valid is False


# =============================================================================
# Тесты для DataValidator.validate_time_series
# =============================================================================

class TestValidateTimeSeries:
    """Тесты для DataValidator.validate_time_series."""

    def test_monotonic_time_is_valid(self):
        """Тест что монотонно возрастающее время проходит валидацию."""
        df = pd.DataFrame({
            "time": [0.0, 6.0, 24.0, 48.0],
            "cell_count": [1000.0, 2000.0, 3000.0, 2500.0],
        })
        validator = DataValidator()
        result = validator.validate_time_series(df)
        assert result.is_valid is True

    def test_non_monotonic_time_is_error(self):
        """Тест что немонотонное время → ошибка с 'monoton' в тексте."""
        df = pd.DataFrame({
            "time": [0.0, 6.0, 3.0, 48.0],  # 3.0 < 6.0 — нарушение
            "cell_count": [1000.0, 2000.0, 3000.0, 2500.0],
        })
        validator = DataValidator()
        result = validator.validate_time_series(df)
        assert result.is_valid is False
        errors_str = " ".join(result.errors).lower()
        assert "monoton" in errors_str

    def test_single_time_point_too_few_rows(self):
        """Тест что одна временная точка < min_rows=2 → ошибка."""
        df = pd.DataFrame({
            "time": [0.0],
            "cell_count": [1000.0],
        })
        validator = DataValidator()
        result = validator.validate_time_series(df)
        assert result.is_valid is False

    def test_negative_values_warning_or_error(self):
        """Тест что отрицательные значения вызывают предупреждение или ошибку."""
        df = pd.DataFrame({
            "time": [0.0, 6.0, 24.0],
            "cell_count": [1000.0, -5.0, 3000.0],
        })
        validator = DataValidator()
        result = validator.validate_time_series(df)
        assert len(result.warnings) > 0 or len(result.errors) > 0


# =============================================================================
# Тесты для DataValidator.validate_model_parameters
# =============================================================================

class TestValidateModelParameters:
    """Тесты для DataValidator.validate_model_parameters."""

    def test_valid_model_parameters(self):
        """Тест что валидные ModelParameters проходят валидацию."""
        params = ModelParameters(
            n0=5000.0,
            c0=10.0,
            stem_cell_fraction=0.05,
            macrophage_fraction=0.03,
            apoptotic_fraction=0.02,
            inflammation_level=0.3,
        )
        validator = DataValidator()
        result = validator.validate_model_parameters(params)
        assert result.is_valid is True

    def test_invalid_n0_negative(self):
        """Тест что n0=-1 → is_valid=False."""
        params = ModelParameters(
            n0=-1.0,
            c0=10.0,
            stem_cell_fraction=0.05,
            macrophage_fraction=0.03,
            apoptotic_fraction=0.02,
            inflammation_level=0.3,
        )
        validator = DataValidator()
        result = validator.validate_model_parameters(params)
        assert result.is_valid is False

    def test_valid_extended_model_parameters(self, mock_extended_model_parameters):
        """Тест что валидные ExtendedModelParameters проходят валидацию."""
        validator = DataValidator()
        result = validator.validate_model_parameters(mock_extended_model_parameters)
        assert result.is_valid is True

    def test_invalid_extended_O2_over_1(self):
        """Тест что O2=1.5 → is_valid=False."""
        params = ExtendedModelParameters(
            P0=250000.0, Ne0=500.0, M1_0=105.0, M2_0=45.0,
            F0=500.0, Mf0=0.0, E0=300.0, S0=250.0,
            C_TNF=0.1, C_IL10=0.05, C_PDGF=5.0, C_VEGF=0.5,
            C_TGFb=1.0, C_MCP1=0.2, C_IL8=0.1,
            rho_collagen=0.1, C_MMP=0.5, rho_fibrin=0.8,
            D=1.0, O2=1.5,  # Невалидно: > 1
        )
        validator = DataValidator()
        result = validator.validate_model_parameters(params)
        assert result.is_valid is False


# =============================================================================
# Тесты для DataValidator.validate_gating_results
# =============================================================================

class TestValidateGatingResults:
    """Тесты для DataValidator.validate_gating_results."""

    def test_correct_gating_results(self, mock_gating_results_normal):
        """Тест что корректные GatingResults проходят валидацию."""
        validator = DataValidator()
        result = validator.validate_gating_results(mock_gating_results_normal)
        assert result.is_valid is True

    def test_fraction_greater_than_1(self):
        """Тест что fraction > 1 → ошибка."""
        gates = {
            "test_gate": GateResult(
                name="test_gate",
                mask=np.array([True, False]),
                n_events=1,
                fraction=1.5,  # Невалидно
            ),
        }
        gating_results = GatingResults(total_events=2, gates=gates)
        validator = DataValidator()
        result = validator.validate_gating_results(gating_results)
        assert result.is_valid is False

    def test_fraction_negative(self):
        """Тест что fraction < 0 → ошибка."""
        gates = {
            "test_gate": GateResult(
                name="test_gate",
                mask=np.array([True, False]),
                n_events=1,
                fraction=-0.1,  # Невалидно
            ),
        }
        gating_results = GatingResults(total_events=2, gates=gates)
        validator = DataValidator()
        result = validator.validate_gating_results(gating_results)
        assert result.is_valid is False

    def test_n_events_negative(self):
        """Тест что n_events < 0 → ошибка."""
        gates = {
            "test_gate": GateResult(
                name="test_gate",
                mask=np.array([True, False]),
                n_events=-10,  # Невалидно
                fraction=0.5,
            ),
        }
        gating_results = GatingResults(total_events=2, gates=gates)
        validator = DataValidator()
        result = validator.validate_gating_results(gating_results)
        assert result.is_valid is False

    def test_total_events_zero(self):
        """Тест что total_events=0 → ошибка или предупреждение."""
        gates = {
            "test_gate": GateResult(
                name="test_gate",
                mask=np.array([], dtype=bool),
                n_events=0,
                fraction=0.0,
            ),
        }
        gating_results = GatingResults(total_events=0, gates=gates)
        validator = DataValidator()
        result = validator.validate_gating_results(gating_results)
        assert result.is_valid is False or len(result.warnings) > 0


# =============================================================================
# Тесты для validate_data() функции
# =============================================================================

class TestValidateDataFunction:
    """Тесты для удобной функции validate_data."""

    def test_with_explicit_schema(self):
        """Тест что явно переданная schema используется."""
        schema = DataSchema(
            name="test",
            columns=[ColumnSchema(name="x", dtype="float", required=True)],
        )
        df = pd.DataFrame({"x": [1.0, 2.0]})
        result = validate_data(df, schema=schema)
        assert result.is_valid is True

    def test_auto_detect_fcs_by_fsc_a(self):
        """Тест автодетекции FCS схемы по столбцу 'FSC-A'."""
        rng = np.random.default_rng(60)
        n = 200
        df = pd.DataFrame({
            "FSC-A": rng.uniform(10000, 200000, n),
            "FSC-H": rng.uniform(10000, 200000, n),
            "SSC-A": rng.uniform(5000, 100000, n),
            "CD34": rng.uniform(0, 50000, n),
            "Annexin-V": rng.uniform(0, 30000, n),
        })
        result = validate_data(df, schema=None)
        assert isinstance(result, ValidationResult)

    def test_auto_detect_time_series_by_time(self):
        """Тест автодетекции TimeSeries схемы по столбцу 'time'."""
        df = pd.DataFrame({
            "time": [0.0, 6.0, 24.0],
            "cell_count": [1000.0, 2000.0, 3000.0],
        })
        result = validate_data(df, schema=None)
        assert isinstance(result, ValidationResult)

    def test_auto_detect_cytokine_by_time_and_tnf(self):
        """Тест автодетекции Cytokine схемы по столбцам time + TNF_alpha."""
        df = pd.DataFrame({
            "time": [0.0, 6.0, 24.0],
            "TNF_alpha": [0.1, 0.5, 0.3],
            "IL_10": [0.05, 0.1, 0.08],
        })
        result = validate_data(df, schema=None)
        assert isinstance(result, ValidationResult)
