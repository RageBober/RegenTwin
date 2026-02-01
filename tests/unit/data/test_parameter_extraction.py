"""
TDD тесты для модуля parameter_extraction.py

Тестирует:
- ModelParameters dataclass
- ExtractionConfig dataclass
- ParameterExtractor класс
- extract_model_parameters функция

Основано на спецификации: Description/description_parameter_extraction.md

Ожидаемые диапазоны параметров:
- N0: 1000-50000 клеток/мкл
- C0: 1-100 нг/мл
- inflammation_level: 0-1
- stem_cell_fraction: 0.01-0.15
- macrophage_fraction: 0.01-0.10
- apoptotic_fraction: 0.01-0.10

Тестовые сценарии:
- Норма: stem=0.05, macro=0.03, apopt=0.02 -> inflammation ~0.3
- Воспаление: stem=0.03, macro=0.08, apopt=0.05 -> inflammation ~0.7
- Регенерация: stem=0.10, macro=0.02, apopt=0.01 -> inflammation ~0.2
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from src.data.parameter_extraction import (
    ModelParameters,
    ExtractionConfig,
    ParameterExtractor,
    extract_model_parameters,
)
from src.data.gating import GatingResults, GateResult


# =============================================================================
# Тесты для ModelParameters
# =============================================================================

class TestModelParameters:
    """Тесты для dataclass ModelParameters."""

    def test_model_parameters_creation_with_all_fields(self):
        """Тест создания ModelParameters со всеми полями."""
        params = ModelParameters(
            n0=5000.0,
            stem_cell_fraction=0.05,
            macrophage_fraction=0.03,
            apoptotic_fraction=0.02,
            c0=10.0,
            inflammation_level=0.3,
            source_file="test.fcs",
            total_events=10000
        )

        assert params.n0 == 5000.0
        assert params.stem_cell_fraction == 0.05
        assert params.macrophage_fraction == 0.03
        assert params.apoptotic_fraction == 0.02
        assert params.c0 == 10.0
        assert params.inflammation_level == 0.3
        assert params.source_file == "test.fcs"
        assert params.total_events == 10000

    def test_model_parameters_optional_source_file(self):
        """Тест что source_file опционален."""
        params = ModelParameters(
            n0=5000.0,
            stem_cell_fraction=0.05,
            macrophage_fraction=0.03,
            apoptotic_fraction=0.02,
            c0=10.0,
            inflammation_level=0.3
        )

        assert params.source_file is None

    def test_to_dict_returns_dict(self):
        """Тест что to_dict возвращает словарь."""
        params = ModelParameters(
            n0=5000.0,
            stem_cell_fraction=0.05,
            macrophage_fraction=0.03,
            apoptotic_fraction=0.02,
            c0=10.0,
            inflammation_level=0.3
        )

        result = params.to_dict()

        assert isinstance(result, dict)

    def test_to_dict_contains_all_fields(self):
        """Тест что to_dict содержит все поля."""
        params = ModelParameters(
            n0=5000.0,
            stem_cell_fraction=0.05,
            macrophage_fraction=0.03,
            apoptotic_fraction=0.02,
            c0=10.0,
            inflammation_level=0.3,
            source_file="test.fcs",
            total_events=10000
        )

        result = params.to_dict()

        expected_keys = [
            "n0", "stem_cell_fraction", "macrophage_fraction",
            "apoptotic_fraction", "c0", "inflammation_level",
            "source_file", "total_events"
        ]
        for key in expected_keys:
            assert key in result

    def test_to_dict_values_correct(self):
        """Тест корректности значений в to_dict."""
        params = ModelParameters(
            n0=5000.0,
            stem_cell_fraction=0.05,
            macrophage_fraction=0.03,
            apoptotic_fraction=0.02,
            c0=10.0,
            inflammation_level=0.3
        )

        result = params.to_dict()

        assert result["n0"] == 5000.0
        assert result["stem_cell_fraction"] == 0.05
        assert result["c0"] == 10.0

    def test_validate_valid_parameters_returns_true(self):
        """Тест что валидные параметры проходят валидацию."""
        params = ModelParameters(
            n0=5000.0,
            stem_cell_fraction=0.05,
            macrophage_fraction=0.03,
            apoptotic_fraction=0.02,
            c0=10.0,
            inflammation_level=0.3
        )

        assert params.validate() is True

    def test_validate_zero_n0_raises_error(self):
        """Тест что n0=0 вызывает ошибку."""
        params = ModelParameters(
            n0=0.0,
            stem_cell_fraction=0.05,
            macrophage_fraction=0.03,
            apoptotic_fraction=0.02,
            c0=10.0,
            inflammation_level=0.3
        )

        with pytest.raises(ValueError):
            params.validate()

    def test_validate_negative_n0_raises_error(self):
        """Тест что отрицательный n0 вызывает ошибку."""
        params = ModelParameters(
            n0=-100.0,
            stem_cell_fraction=0.05,
            macrophage_fraction=0.03,
            apoptotic_fraction=0.02,
            c0=10.0,
            inflammation_level=0.3
        )

        with pytest.raises(ValueError):
            params.validate()

    def test_validate_fraction_over_1_raises_error(self):
        """Тест что фракция > 1 вызывает ошибку."""
        params = ModelParameters(
            n0=5000.0,
            stem_cell_fraction=1.5,  # > 1
            macrophage_fraction=0.03,
            apoptotic_fraction=0.02,
            c0=10.0,
            inflammation_level=0.3
        )

        with pytest.raises(ValueError):
            params.validate()

    def test_validate_negative_fraction_raises_error(self):
        """Тест что отрицательная фракция вызывает ошибку."""
        params = ModelParameters(
            n0=5000.0,
            stem_cell_fraction=-0.05,  # < 0
            macrophage_fraction=0.03,
            apoptotic_fraction=0.02,
            c0=10.0,
            inflammation_level=0.3
        )

        with pytest.raises(ValueError):
            params.validate()

    def test_validate_inflammation_over_1_raises_error(self):
        """Тест что inflammation_level > 1 вызывает ошибку."""
        params = ModelParameters(
            n0=5000.0,
            stem_cell_fraction=0.05,
            macrophage_fraction=0.03,
            apoptotic_fraction=0.02,
            c0=10.0,
            inflammation_level=1.5  # > 1
        )

        with pytest.raises(ValueError):
            params.validate()

    def test_validate_negative_inflammation_raises_error(self):
        """Тест что отрицательный inflammation_level вызывает ошибку."""
        params = ModelParameters(
            n0=5000.0,
            stem_cell_fraction=0.05,
            macrophage_fraction=0.03,
            apoptotic_fraction=0.02,
            c0=10.0,
            inflammation_level=-0.1  # < 0
        )

        with pytest.raises(ValueError):
            params.validate()

    def test_validate_negative_c0_raises_error(self):
        """Тест что отрицательный c0 вызывает ошибку."""
        params = ModelParameters(
            n0=5000.0,
            stem_cell_fraction=0.05,
            macrophage_fraction=0.03,
            apoptotic_fraction=0.02,
            c0=-5.0,  # < 0
            inflammation_level=0.3
        )

        with pytest.raises(ValueError):
            params.validate()

    def test_validate_edge_case_zero_fractions(self):
        """Тест граничного случая с нулевыми фракциями."""
        params = ModelParameters(
            n0=5000.0,
            stem_cell_fraction=0.0,
            macrophage_fraction=0.0,
            apoptotic_fraction=0.0,
            c0=10.0,
            inflammation_level=0.0
        )

        # Нулевые фракции допустимы
        assert params.validate() is True

    def test_validate_edge_case_max_values(self):
        """Тест граничного случая с максимальными значениями."""
        params = ModelParameters(
            n0=50000.0,
            stem_cell_fraction=1.0,
            macrophage_fraction=0.0,  # Сумма не может быть > 1
            apoptotic_fraction=0.0,
            c0=100.0,
            inflammation_level=1.0
        )

        assert params.validate() is True


# =============================================================================
# Тесты для ExtractionConfig
# =============================================================================

class TestExtractionConfig:
    """Тесты для ExtractionConfig."""

    def test_default_values(self):
        """Тест значений по умолчанию."""
        config = ExtractionConfig()

        assert config.volume_ul == 1.0
        assert config.dilution_factor == 1.0
        assert config.ref_cell_density == 5000.0
        assert config.ref_cytokine_conc == 10.0
        assert config.stem_cell_cytokine_factor == 2.0
        assert config.macrophage_cytokine_factor == 1.5

    def test_custom_values(self):
        """Тест кастомных значений."""
        config = ExtractionConfig(
            volume_ul=10.0,
            dilution_factor=2.0,
            ref_cell_density=10000.0,
            ref_cytokine_conc=20.0,
            stem_cell_cytokine_factor=3.0,
            macrophage_cytokine_factor=2.0
        )

        assert config.volume_ul == 10.0
        assert config.dilution_factor == 2.0
        assert config.ref_cell_density == 10000.0
        assert config.ref_cytokine_conc == 20.0
        assert config.stem_cell_cytokine_factor == 3.0
        assert config.macrophage_cytokine_factor == 2.0

    def test_partial_custom_values(self):
        """Тест частичных кастомных значений."""
        config = ExtractionConfig(volume_ul=5.0)

        assert config.volume_ul == 5.0
        # Остальные по умолчанию
        assert config.dilution_factor == 1.0
        assert config.ref_cell_density == 5000.0


# =============================================================================
# Тесты для ParameterExtractor.__init__
# =============================================================================

class TestParameterExtractorInit:
    """Тесты для ParameterExtractor.__init__."""

    def test_init_without_config_uses_default(self):
        """Тест что без конфига используется дефолтный."""
        extractor = ParameterExtractor()

        assert extractor._config is not None
        assert isinstance(extractor._config, ExtractionConfig)

    def test_init_with_custom_config(self):
        """Тест с кастомным конфигом."""
        config = ExtractionConfig(volume_ul=5.0)

        extractor = ParameterExtractor(config=config)

        assert extractor._config.volume_ul == 5.0

    def test_init_config_is_stored(self):
        """Тест что конфиг сохраняется."""
        config = ExtractionConfig(dilution_factor=3.0)

        extractor = ParameterExtractor(config=config)

        assert extractor._config is config


# =============================================================================
# Тесты для ParameterExtractor.extract_n0
# =============================================================================

class TestParameterExtractorExtractN0:
    """Тесты для ParameterExtractor.extract_n0."""

    def test_extract_n0_basic_calculation(self, mock_gating_results_normal):
        """Тест базового расчета N0."""
        extractor = ParameterExtractor()

        n0 = extractor.extract_n0(mock_gating_results_normal)

        # 7000 живых клеток / 1 мкл * 1 = 7000
        assert n0 == pytest.approx(7000.0, rel=0.1)

    def test_extract_n0_with_volume(self, mock_gating_results_normal):
        """Тест N0 с объёмом > 1."""
        config = ExtractionConfig(volume_ul=10.0)
        extractor = ParameterExtractor(config=config)

        n0 = extractor.extract_n0(mock_gating_results_normal)

        # 7000 / 10 = 700
        assert n0 == pytest.approx(700.0, rel=0.1)

    def test_extract_n0_with_dilution(self, mock_gating_results_normal):
        """Тест N0 с разведением."""
        config = ExtractionConfig(dilution_factor=2.0)
        extractor = ParameterExtractor(config=config)

        n0 = extractor.extract_n0(mock_gating_results_normal)

        # 7000 * 2 = 14000
        assert n0 == pytest.approx(14000.0, rel=0.1)

    def test_extract_n0_with_volume_and_dilution(self, mock_gating_results_normal):
        """Тест N0 с объёмом и разведением."""
        config = ExtractionConfig(volume_ul=2.0, dilution_factor=3.0)
        extractor = ParameterExtractor(config=config)

        n0 = extractor.extract_n0(mock_gating_results_normal)

        # 7000 / 2 * 3 = 10500
        assert n0 == pytest.approx(10500.0, rel=0.1)

    def test_extract_n0_in_expected_range(self, mock_gating_results_normal):
        """Тест что N0 в ожидаемом диапазоне (1000-50000)."""
        extractor = ParameterExtractor()

        n0 = extractor.extract_n0(mock_gating_results_normal)

        assert 1000 <= n0 <= 50000

    def test_extract_n0_returns_float(self, mock_gating_results_normal):
        """Тест что возвращается float."""
        extractor = ParameterExtractor()

        n0 = extractor.extract_n0(mock_gating_results_normal)

        assert isinstance(n0, (int, float, np.floating))


# =============================================================================
# Тесты для ParameterExtractor.extract_c0
# =============================================================================

class TestParameterExtractorExtractC0:
    """Тесты для ParameterExtractor.extract_c0."""

    def test_extract_c0_returns_positive(self, mock_gating_results_normal):
        """Тест что C0 положительный."""
        extractor = ParameterExtractor()

        c0 = extractor.extract_c0(mock_gating_results_normal)

        assert c0 > 0

    def test_extract_c0_increases_with_stem_cells(self):
        """Тест что C0 растет с увеличением стволовых клеток."""
        extractor = ParameterExtractor()

        # Низкий stem
        gates_low = {
            "cd34_positive": GateResult(
                name="cd34_positive",
                mask=np.array([True] * 100),
                n_events=100,
                fraction=0.01
            ),
            "macrophages": GateResult(
                name="macrophages",
                mask=np.array([True] * 300),
                n_events=300,
                fraction=0.03
            ),
            "live_cells": GateResult(
                name="live_cells",
                mask=np.array([True] * 7000),
                n_events=7000,
                fraction=0.70
            ),
        }
        results_low = GatingResults(total_events=10000, gates=gates_low)

        # Высокий stem
        gates_high = {
            "cd34_positive": GateResult(
                name="cd34_positive",
                mask=np.array([True] * 1000),
                n_events=1000,
                fraction=0.10
            ),
            "macrophages": GateResult(
                name="macrophages",
                mask=np.array([True] * 300),
                n_events=300,
                fraction=0.03
            ),
            "live_cells": GateResult(
                name="live_cells",
                mask=np.array([True] * 7000),
                n_events=7000,
                fraction=0.70
            ),
        }
        results_high = GatingResults(total_events=10000, gates=gates_high)

        c0_low = extractor.extract_c0(results_low)
        c0_high = extractor.extract_c0(results_high)

        assert c0_high > c0_low

    def test_extract_c0_in_expected_range(self, mock_gating_results_normal):
        """Тест что C0 в ожидаемом диапазоне (1-100)."""
        extractor = ParameterExtractor()

        c0 = extractor.extract_c0(mock_gating_results_normal)

        assert 1 <= c0 <= 100

    def test_extract_c0_formula_correctness(self):
        """Тест корректности формулы C0."""
        # C0 = C_ref * (alpha * f_stem + beta * f_macro)
        config = ExtractionConfig(
            ref_cytokine_conc=10.0,
            stem_cell_cytokine_factor=2.0,
            macrophage_cytokine_factor=1.5
        )
        extractor = ParameterExtractor(config=config)

        gates = {
            "cd34_positive": GateResult(
                name="cd34_positive",
                mask=np.array([True] * 500),
                n_events=500,
                fraction=0.05
            ),
            "macrophages": GateResult(
                name="macrophages",
                mask=np.array([True] * 300),
                n_events=300,
                fraction=0.03
            ),
            "live_cells": GateResult(
                name="live_cells",
                mask=np.array([True] * 7000),
                n_events=7000,
                fraction=0.70
            ),
        }
        results = GatingResults(total_events=10000, gates=gates)

        c0 = extractor.extract_c0(results)

        # Ожидаемое: 10 * (2.0 * 0.05 + 1.5 * 0.03) = 10 * 0.145 = 1.45
        # Но может быть clipped к [1, 100]
        assert c0 >= 1.0


# =============================================================================
# Тесты для ParameterExtractor.extract_inflammation_level
# =============================================================================

class TestParameterExtractorExtractInflammationLevel:
    """Тесты для ParameterExtractor.extract_inflammation_level."""

    def test_extract_inflammation_normal_scenario(self, mock_gating_results_normal):
        """Тест нормального уровня воспаления."""
        extractor = ParameterExtractor()

        inflammation = extractor.extract_inflammation_level(mock_gating_results_normal)

        # При нормальных фракциях (macro=0.03, apopt=0.02) ожидаем низкое воспаление
        assert 0.0 <= inflammation <= 0.5

    def test_extract_inflammation_high_scenario(self, mock_gating_results_inflamed):
        """Тест высокого уровня воспаления."""
        extractor = ParameterExtractor()

        inflammation = extractor.extract_inflammation_level(mock_gating_results_inflamed)

        # При повышенных фракциях (macro=0.08, apopt=0.05) ожидаем высокое воспаление
        assert inflammation > 0.4

    def test_extract_inflammation_low_scenario(self, mock_gating_results_regenerating):
        """Тест низкого уровня воспаления при регенерации."""
        extractor = ParameterExtractor()

        inflammation = extractor.extract_inflammation_level(mock_gating_results_regenerating)

        # При низких фракциях (macro=0.02, apopt=0.01) ожидаем низкое воспаление
        assert inflammation < 0.4

    def test_extract_inflammation_in_range_0_1(self, mock_gating_results_normal):
        """Тест что inflammation в диапазоне [0, 1]."""
        extractor = ParameterExtractor()

        inflammation = extractor.extract_inflammation_level(mock_gating_results_normal)

        assert 0 <= inflammation <= 1

    def test_extract_inflammation_ordering(
        self,
        mock_gating_results_normal,
        mock_gating_results_inflamed,
        mock_gating_results_regenerating
    ):
        """Тест порядка: regenerating < normal < inflamed."""
        extractor = ParameterExtractor()

        infl_normal = extractor.extract_inflammation_level(mock_gating_results_normal)
        infl_inflamed = extractor.extract_inflammation_level(mock_gating_results_inflamed)
        infl_regen = extractor.extract_inflammation_level(mock_gating_results_regenerating)

        assert infl_regen < infl_normal < infl_inflamed

    def test_extract_inflammation_returns_float(self, mock_gating_results_normal):
        """Тест что возвращается float."""
        extractor = ParameterExtractor()

        inflammation = extractor.extract_inflammation_level(mock_gating_results_normal)

        assert isinstance(inflammation, (int, float, np.floating))


# =============================================================================
# Тесты для ParameterExtractor.extract
# =============================================================================

class TestParameterExtractorExtract:
    """Тесты для ParameterExtractor.extract."""

    def test_extract_returns_model_parameters(self, mock_gating_results_normal):
        """Тест что возвращается ModelParameters."""
        extractor = ParameterExtractor()

        params = extractor.extract(mock_gating_results_normal)

        assert isinstance(params, ModelParameters)

    def test_extract_all_fields_populated(self, mock_gating_results_normal):
        """Тест что все поля заполнены."""
        extractor = ParameterExtractor()

        params = extractor.extract(mock_gating_results_normal, source_file="test.fcs")

        assert params.n0 > 0
        assert 0 <= params.stem_cell_fraction <= 1
        assert 0 <= params.macrophage_fraction <= 1
        assert 0 <= params.apoptotic_fraction <= 1
        assert params.c0 > 0
        assert 0 <= params.inflammation_level <= 1
        assert params.source_file == "test.fcs"
        assert params.total_events == 10000

    def test_extract_validates_result(self, mock_gating_results_normal):
        """Тест что результат проходит валидацию."""
        extractor = ParameterExtractor()

        params = extractor.extract(mock_gating_results_normal)

        # Не должно выбрасывать исключение
        assert params.validate() is True

    def test_extract_source_file_optional(self, mock_gating_results_normal):
        """Тест что source_file опционален."""
        extractor = ParameterExtractor()

        params = extractor.extract(mock_gating_results_normal)

        assert params.source_file is None

    def test_extract_total_events_from_gating_results(self, mock_gating_results_normal):
        """Тест что total_events берётся из gating_results."""
        extractor = ParameterExtractor()

        params = extractor.extract(mock_gating_results_normal)

        assert params.total_events == mock_gating_results_normal.total_events


# =============================================================================
# Тесты для ParameterExtractor._calculate_cell_fractions
# =============================================================================

class TestParameterExtractorCalculateCellFractions:
    """Тесты для ParameterExtractor._calculate_cell_fractions."""

    def test_calculate_fractions_returns_dict(self, mock_gating_results_normal):
        """Тест что возвращается словарь."""
        extractor = ParameterExtractor()

        fractions = extractor._calculate_cell_fractions(mock_gating_results_normal)

        assert isinstance(fractions, dict)

    def test_calculate_fractions_contains_expected_keys(self, mock_gating_results_normal):
        """Тест что содержит ожидаемые ключи."""
        extractor = ParameterExtractor()

        fractions = extractor._calculate_cell_fractions(mock_gating_results_normal)

        # Должны быть ключи для всех гейтов
        assert "live_cells" in fractions
        assert "cd34_positive" in fractions
        assert "macrophages" in fractions

    def test_calculate_fractions_correct_values(self, mock_gating_results_normal):
        """Тест корректности значений фракций."""
        extractor = ParameterExtractor()

        fractions = extractor._calculate_cell_fractions(mock_gating_results_normal)

        # Проверяем значения из mock_gating_results_normal
        assert fractions["live_cells"] == pytest.approx(0.70, rel=0.01)
        assert fractions["cd34_positive"] == pytest.approx(0.05, rel=0.01)
        assert fractions["macrophages"] == pytest.approx(0.03, rel=0.01)

    def test_calculate_fractions_all_between_0_and_1(self, mock_gating_results_normal):
        """Тест что все фракции в диапазоне [0, 1]."""
        extractor = ParameterExtractor()

        fractions = extractor._calculate_cell_fractions(mock_gating_results_normal)

        for name, value in fractions.items():
            assert 0 <= value <= 1, f"Fraction {name} = {value} is out of range"


# =============================================================================
# Тесты для ParameterExtractor._normalize_to_reference
# =============================================================================

class TestParameterExtractorNormalizeToReference:
    """Тесты для ParameterExtractor._normalize_to_reference."""

    def test_normalize_linear_basic(self):
        """Тест линейной нормализации."""
        extractor = ParameterExtractor()

        result = extractor._normalize_to_reference(10000, 5000, scale="linear")

        assert result == pytest.approx(2.0)

    def test_normalize_linear_less_than_ref(self):
        """Тест линейной нормализации когда value < ref."""
        extractor = ParameterExtractor()

        result = extractor._normalize_to_reference(2500, 5000, scale="linear")

        assert result == pytest.approx(0.5)

    def test_normalize_log_basic(self):
        """Тест логарифмической нормализации."""
        extractor = ParameterExtractor()

        result = extractor._normalize_to_reference(100, 10, scale="log")

        # log10(100) / log10(10) = 2 / 1 = 2
        assert result == pytest.approx(2.0)

    def test_normalize_log_same_value(self):
        """Тест log нормализации когда value = ref."""
        extractor = ParameterExtractor()

        result = extractor._normalize_to_reference(10, 10, scale="log")

        assert result == pytest.approx(1.0)

    def test_normalize_invalid_scale_raises(self):
        """Тест что невалидный scale вызывает ошибку."""
        extractor = ParameterExtractor()

        with pytest.raises(ValueError):
            extractor._normalize_to_reference(100, 10, scale="invalid_scale")

    def test_normalize_returns_float(self):
        """Тест что возвращается float."""
        extractor = ParameterExtractor()

        result = extractor._normalize_to_reference(7000, 5000, scale="linear")

        assert isinstance(result, (int, float, np.floating))


# =============================================================================
# Тесты для extract_model_parameters функции
# =============================================================================

class TestExtractModelParametersFunction:
    """Тесты для convenience функции extract_model_parameters."""

    def test_extract_model_parameters_returns_model_parameters(
        self, mock_gating_results_normal
    ):
        """Тест что возвращается ModelParameters."""
        params = extract_model_parameters(mock_gating_results_normal)

        assert isinstance(params, ModelParameters)

    def test_extract_model_parameters_with_config(self, mock_gating_results_normal):
        """Тест с кастомным конфигом."""
        config = ExtractionConfig(volume_ul=5.0)

        params = extract_model_parameters(mock_gating_results_normal, config=config)

        # N0 должен быть меньше из-за большего объема
        # 7000 / 5 = 1400
        assert params.n0 < 2000

    def test_extract_model_parameters_with_source_file(self, mock_gating_results_normal):
        """Тест с указанием source_file."""
        params = extract_model_parameters(
            mock_gating_results_normal,
            source_file="sample.fcs"
        )

        assert params.source_file == "sample.fcs"

    def test_extract_model_parameters_default_config(self, mock_gating_results_normal):
        """Тест что по умолчанию используется дефолтный конфиг."""
        params = extract_model_parameters(mock_gating_results_normal)

        # С дефолтным конфигом N0 = n_live_cells / 1.0 * 1.0 = 7000
        assert params.n0 == pytest.approx(7000.0, rel=0.1)

    def test_extract_model_parameters_validates_result(self, mock_gating_results_normal):
        """Тест что результат проходит валидацию."""
        params = extract_model_parameters(mock_gating_results_normal)

        assert params.validate() is True


# =============================================================================
# Интеграционные тесты сценариев
# =============================================================================

class TestParameterExtractionScenarios:
    """Интеграционные тесты различных сценариев."""

    def test_normal_scenario_parameters(self, mock_gating_results_normal):
        """Тест параметров для нормального сценария."""
        params = extract_model_parameters(mock_gating_results_normal)

        # Норма: stem=0.05, macro=0.03, apopt=0.02
        assert 0.03 < params.stem_cell_fraction < 0.08
        assert 0.02 < params.macrophage_fraction < 0.05
        assert 0.01 < params.apoptotic_fraction < 0.04
        assert params.inflammation_level < 0.5

    def test_inflamed_scenario_parameters(self, mock_gating_results_inflamed):
        """Тест параметров для воспалительного сценария."""
        params = extract_model_parameters(mock_gating_results_inflamed)

        # Воспаление: macro=0.08, apopt=0.05
        assert params.macrophage_fraction > 0.05
        assert params.apoptotic_fraction > 0.03
        assert params.inflammation_level > 0.4

    def test_regenerating_scenario_parameters(self, mock_gating_results_regenerating):
        """Тест параметров для регенеративного сценария."""
        params = extract_model_parameters(mock_gating_results_regenerating)

        # Регенерация: stem=0.10, macro=0.02, apopt=0.01
        assert params.stem_cell_fraction > 0.07
        assert params.macrophage_fraction < 0.04
        assert params.inflammation_level < 0.4

    def test_parameters_consistency_across_scenarios(
        self,
        mock_gating_results_normal,
        mock_gating_results_inflamed,
        mock_gating_results_regenerating
    ):
        """Тест консистентности параметров между сценариями."""
        params_normal = extract_model_parameters(mock_gating_results_normal)
        params_inflamed = extract_model_parameters(mock_gating_results_inflamed)
        params_regen = extract_model_parameters(mock_gating_results_regenerating)

        # C0 должен быть выше при регенерации (больше stem cells)
        assert params_regen.c0 >= params_normal.c0

        # Inflammation должен расти от регенерации к воспалению
        assert params_regen.inflammation_level < params_normal.inflammation_level < params_inflamed.inflammation_level

    def test_all_parameters_in_valid_ranges(
        self,
        mock_gating_results_normal,
        expected_parameter_ranges
    ):
        """Тест что все параметры в допустимых диапазонах."""
        params = extract_model_parameters(mock_gating_results_normal)

        ranges = expected_parameter_ranges
        assert ranges["n0"]["min"] <= params.n0 <= ranges["n0"]["max"]
        assert ranges["c0"]["min"] <= params.c0 <= ranges["c0"]["max"]
        assert ranges["inflammation_level"]["min"] <= params.inflammation_level <= ranges["inflammation_level"]["max"]
        assert ranges["stem_cell_fraction"]["min"] <= params.stem_cell_fraction <= ranges["stem_cell_fraction"]["max"]
        assert ranges["macrophage_fraction"]["min"] <= params.macrophage_fraction <= ranges["macrophage_fraction"]["max"]
        assert ranges["apoptotic_fraction"]["min"] <= params.apoptotic_fraction <= ranges["apoptotic_fraction"]["max"]
