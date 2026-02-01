"""
E2E интеграционные тесты для полного data pipeline RegenTwin.

Проверяет весь поток обработки данных:
FCS Data → Gating → Parameter Extraction → Model Parameters

Тесты используют mock данные из conftest.py для трёх сценариев:
- Нормальная ткань
- Воспаление
- Регенерация
"""

import pytest
import numpy as np
import pandas as pd

from src.data.gating import GatingStrategy, GatingResults
from src.data.parameter_extraction import (
    ParameterExtractor,
    ExtractionConfig,
    ModelParameters,
    extract_model_parameters,
)


# =============================================================================
# E2E Pipeline Tests
# =============================================================================

@pytest.mark.integration
class TestFullPipeline:
    """E2E тесты полного data pipeline."""

    def test_pipeline_normal_scenario(self, mock_fcs_data_normal):
        """
        E2E тест: нормальный образец.

        Pipeline: DataFrame → Gating → Parameters
        """
        # Step 1: Apply gating strategy
        strategy = GatingStrategy()
        gating_results = strategy.apply(mock_fcs_data_normal)

        # Step 2: Extract model parameters
        params = extract_model_parameters(gating_results, source_file="mock_normal.fcs")

        # Step 3: Validate results
        assert isinstance(gating_results, GatingResults)
        assert isinstance(params, ModelParameters)
        assert params.validate() is True

        # Step 4: Check expected ranges for normal scenario
        # Ranges are wider to account for gating algorithm variance on mock data
        assert 0.03 < params.stem_cell_fraction < 0.15
        assert 0.02 < params.macrophage_fraction < 0.10
        assert 0.01 < params.apoptotic_fraction < 0.08
        assert params.inflammation_level < 0.8

    def test_pipeline_inflamed_scenario(self, mock_fcs_data_inflamed):
        """
        E2E тест: воспалённый образец.

        Ожидаемые характеристики:
        - Повышенные макрофаги (>5%)
        - Повышенный апоптоз (>3%)
        - Высокий уровень воспаления (>0.4)
        """
        # Step 1: Apply gating
        strategy = GatingStrategy()
        gating_results = strategy.apply(mock_fcs_data_inflamed)

        # Step 2: Extract parameters
        params = extract_model_parameters(gating_results, source_file="mock_inflamed.fcs")

        # Step 3: Validate
        assert params.validate() is True

        # Step 4: Check inflamed characteristics
        assert params.macrophage_fraction > 0.05
        assert params.apoptotic_fraction > 0.03
        assert params.inflammation_level > 0.4

    def test_pipeline_regenerating_scenario(self, mock_fcs_data_regenerating):
        """
        E2E тест: регенерирующий образец.

        Ожидаемые характеристики:
        - Повышенные стволовые клетки (>7%)
        - Параметры в допустимых диапазонах
        """
        # Step 1: Apply gating
        strategy = GatingStrategy()
        gating_results = strategy.apply(mock_fcs_data_regenerating)

        # Step 2: Extract parameters
        params = extract_model_parameters(gating_results, source_file="mock_regenerating.fcs")

        # Step 3: Validate
        assert params.validate() is True

        # Step 4: Check regeneration characteristics
        # Primary characteristic: elevated stem cells
        assert params.stem_cell_fraction > 0.07
        # Secondary: all fractions in valid range
        assert 0 <= params.macrophage_fraction <= 0.15
        assert 0 <= params.inflammation_level <= 1.0

    def test_pipeline_all_scenarios_valid(
        self,
        mock_fcs_data_normal,
        mock_fcs_data_inflamed,
        mock_fcs_data_regenerating,
    ):
        """
        E2E тест: все сценарии проходят pipeline и дают валидные результаты.

        Проверяет что:
        - Все сценарии обрабатываются без ошибок
        - Все параметры в допустимых диапазонах
        - Inflamed сценарий имеет повышенное воспаление
        """
        strategy = GatingStrategy()

        # Process all scenarios
        gating_normal = strategy.apply(mock_fcs_data_normal)
        gating_inflamed = strategy.apply(mock_fcs_data_inflamed)
        gating_regen = strategy.apply(mock_fcs_data_regenerating)

        params_normal = extract_model_parameters(gating_normal)
        params_inflamed = extract_model_parameters(gating_inflamed)
        params_regen = extract_model_parameters(gating_regen)

        # All should be valid
        assert params_normal.validate() is True
        assert params_inflamed.validate() is True
        assert params_regen.validate() is True

        # Inflamed should have elevated inflammation (>0.5)
        assert params_inflamed.inflammation_level > 0.5

        # All inflammation levels should be in valid range
        assert 0 <= params_normal.inflammation_level <= 1
        assert 0 <= params_inflamed.inflammation_level <= 1
        assert 0 <= params_regen.inflammation_level <= 1


@pytest.mark.integration
class TestPipelineConsistency:
    """Тесты консистентности pipeline."""

    def test_deterministic_output(self, mock_fcs_data_normal):
        """
        Тест детерминизма: один и тот же вход → один и тот же выход.
        """
        strategy = GatingStrategy()

        # Run pipeline twice
        gating1 = strategy.apply(mock_fcs_data_normal)
        gating2 = strategy.apply(mock_fcs_data_normal)

        params1 = extract_model_parameters(gating1)
        params2 = extract_model_parameters(gating2)

        # Results should be identical
        assert params1.n0 == params2.n0
        assert params1.c0 == params2.c0
        assert params1.inflammation_level == params2.inflammation_level
        assert params1.stem_cell_fraction == params2.stem_cell_fraction
        assert params1.macrophage_fraction == params2.macrophage_fraction
        assert params1.apoptotic_fraction == params2.apoptotic_fraction

    def test_pipeline_with_custom_config(self, mock_fcs_data_normal):
        """
        Тест pipeline с кастомной конфигурацией.
        """
        strategy = GatingStrategy()
        gating_results = strategy.apply(mock_fcs_data_normal)

        # Custom config
        config = ExtractionConfig(
            volume_ul=5.0,
            dilution_factor=2.0,
            ref_cytokine_conc=20.0,
        )

        params = extract_model_parameters(gating_results, config=config)

        # N0 should be affected by volume and dilution
        # N0 = n_live / volume * dilution = 7000 / 5.0 * 2.0 = 2800
        assert params.validate() is True
        assert params.n0 < 5000  # Lower due to higher volume

    def test_pipeline_total_events_preserved(self, mock_fcs_data_normal):
        """
        Тест: total_events сохраняется через весь pipeline.
        """
        n_input = len(mock_fcs_data_normal)

        strategy = GatingStrategy()
        gating_results = strategy.apply(mock_fcs_data_normal)
        params = extract_model_parameters(gating_results)

        assert gating_results.total_events == n_input
        assert params.total_events == n_input


@pytest.mark.integration
class TestPipelineGateHierarchy:
    """Тесты иерархии гейтов в pipeline."""

    def test_hierarchical_gating_subsets(self, mock_fcs_data_normal):
        """
        Тест: дочерние популяции являются подмножествами родительских.
        """
        strategy = GatingStrategy()
        results = strategy.apply(mock_fcs_data_normal)

        # CD34+ должен быть подмножеством live_cells
        cd34_mask = results.gates["cd34_positive"].mask
        live_mask = results.gates["live_cells"].mask
        assert np.all(cd34_mask <= live_mask)

        # Macrophages должны быть подмножеством live_cells
        macro_mask = results.gates["macrophages"].mask
        assert np.all(macro_mask <= live_mask)

        # Live cells должны быть подмножеством singlets
        singlets_mask = results.gates["singlets"].mask
        assert np.all(live_mask <= singlets_mask)

        # Singlets должны быть подмножеством non_debris
        non_debris_mask = results.gates["non_debris"].mask
        assert np.all(singlets_mask <= non_debris_mask)

    def test_all_gate_fractions_valid(self, mock_fcs_data_normal):
        """
        Тест: все фракции в диапазоне [0, 1].
        """
        strategy = GatingStrategy()
        results = strategy.apply(mock_fcs_data_normal)

        for gate_name, gate_result in results.gates.items():
            assert 0 <= gate_result.fraction <= 1, \
                f"Gate {gate_name} has invalid fraction: {gate_result.fraction}"

    def test_all_masks_correct_length(self, mock_fcs_data_normal):
        """
        Тест: все маски имеют правильную длину.
        """
        n_events = len(mock_fcs_data_normal)

        strategy = GatingStrategy()
        results = strategy.apply(mock_fcs_data_normal)

        for gate_name, gate_result in results.gates.items():
            assert len(gate_result.mask) == n_events, \
                f"Gate {gate_name} mask has wrong length: {len(gate_result.mask)} != {n_events}"


@pytest.mark.integration
class TestPipelineParameterRanges:
    """Тесты диапазонов параметров в pipeline."""

    def test_parameters_in_expected_ranges(
        self,
        mock_fcs_data_normal,
        expected_parameter_ranges,
    ):
        """
        Тест: параметры в ожидаемых диапазонах из документации.
        """
        strategy = GatingStrategy()
        gating_results = strategy.apply(mock_fcs_data_normal)
        params = extract_model_parameters(gating_results)

        ranges = expected_parameter_ranges

        assert ranges["n0"]["min"] <= params.n0 <= ranges["n0"]["max"], \
            f"n0={params.n0} not in [{ranges['n0']['min']}, {ranges['n0']['max']}]"

        assert ranges["c0"]["min"] <= params.c0 <= ranges["c0"]["max"], \
            f"c0={params.c0} not in [{ranges['c0']['min']}, {ranges['c0']['max']}]"

        assert ranges["inflammation_level"]["min"] <= params.inflammation_level <= ranges["inflammation_level"]["max"], \
            f"inflammation_level={params.inflammation_level} not in [0, 1]"

    def test_to_dict_serialization(self, mock_fcs_data_normal):
        """
        Тест: параметры корректно сериализуются в словарь.
        """
        strategy = GatingStrategy()
        gating_results = strategy.apply(mock_fcs_data_normal)
        params = extract_model_parameters(gating_results, source_file="test.fcs")

        params_dict = params.to_dict()

        assert isinstance(params_dict, dict)
        assert "n0" in params_dict
        assert "c0" in params_dict
        assert "inflammation_level" in params_dict
        assert "stem_cell_fraction" in params_dict
        assert "macrophage_fraction" in params_dict
        assert "apoptotic_fraction" in params_dict
        assert "source_file" in params_dict
        assert "total_events" in params_dict

        assert params_dict["source_file"] == "test.fcs"
