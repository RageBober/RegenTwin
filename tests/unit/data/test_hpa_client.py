"""Тесты для модуля hpa_client.py.

Проверяет:
- HPASkinExpression dataclass
- Baseline концентрации для 7 цитокинов
- ValidationDataset совместимость
- Метаданные
"""

import pytest

from src.data.dataset_loader import DatasetSource, ValidationDataset
from src.data.hpa_client import (
    HPASkinExpression,
    get_baseline_concentrations,
    get_hpa_metadata,
    get_hpa_validation_dataset,
    get_skin_baseline,
)

# =============================================================================
# Skin baseline data
# =============================================================================


class TestSkinBaseline:
    @pytest.fixture
    def baseline(self) -> dict[str, HPASkinExpression]:
        return get_skin_baseline()

    def test_returns_dict(self, baseline: dict[str, HPASkinExpression]):
        assert isinstance(baseline, dict)

    def test_has_7_cytokines(self, baseline: dict[str, HPASkinExpression]):
        expected_genes = {"TNF", "IL10", "PDGFA", "VEGFA", "TGFB1", "CCL2", "CXCL8"}
        assert set(baseline.keys()) == expected_genes

    def test_expression_dataclass(self, baseline: dict[str, HPASkinExpression]):
        for gene, expr in baseline.items():
            assert isinstance(expr, HPASkinExpression)
            assert expr.gene_symbol == gene
            assert len(expr.ensembl_id) > 0
            assert len(expr.model_variable) > 0

    def test_rna_ntpm_positive(self, baseline: dict[str, HPASkinExpression]):
        for expr in baseline.values():
            assert expr.rna_ntpm > 0, f"{expr.gene_symbol} has non-positive nTPM"

    def test_estimated_ng_ml_positive(self, baseline: dict[str, HPASkinExpression]):
        for expr in baseline.values():
            assert expr.estimated_ng_ml > 0, f"{expr.gene_symbol} has non-positive ng/mL"

    def test_protein_levels_valid(self, baseline: dict[str, HPASkinExpression]):
        valid_levels = {"Not detected", "Low", "Medium", "High"}
        for expr in baseline.values():
            assert expr.protein_level in valid_levels, (
                f"{expr.gene_symbol}: invalid protein level '{expr.protein_level}'"
            )

    def test_model_variable_mapping(self, baseline: dict[str, HPASkinExpression]):
        """Каждый ген должен маппиться на переменную модели."""
        expected_vars = {"C_TNF", "C_IL10", "C_PDGF", "C_VEGF", "C_TGFb", "C_MCP1", "C_IL8"}
        actual_vars = {expr.model_variable for expr in baseline.values()}
        assert actual_vars == expected_vars


# =============================================================================
# Baseline concentrations
# =============================================================================


class TestBaselineConcentrations:
    def test_returns_dict(self):
        conc = get_baseline_concentrations()
        assert isinstance(conc, dict)

    def test_uses_model_variable_names(self):
        conc = get_baseline_concentrations()
        for key in conc:
            assert key.startswith("C_"), f"Key '{key}' should be a model variable name"

    def test_values_positive(self):
        conc = get_baseline_concentrations()
        for var, val in conc.items():
            assert val > 0, f"{var} should have positive concentration"

    def test_tgfb_highest_baseline(self):
        """TGF-β1 должен иметь наивысший baseline в коже."""
        conc = get_baseline_concentrations()
        assert conc["C_TGFb"] >= max(v for k, v in conc.items() if k != "C_TGFb")


# =============================================================================
# ValidationDataset integration
# =============================================================================


class TestHPAValidationDataset:
    @pytest.fixture
    def dataset(self) -> ValidationDataset:
        return get_hpa_validation_dataset()

    def test_returns_validation_dataset(self, dataset: ValidationDataset):
        assert isinstance(dataset, ValidationDataset)

    def test_has_cytokine_levels(self, dataset: ValidationDataset):
        assert dataset.cytokine_levels is not None

    def test_single_time_point(self, dataset: ValidationDataset):
        """HPA — baseline (t=0)."""
        ts = dataset.cytokine_levels
        assert ts is not None
        assert len(ts.time_points) == 1
        assert ts.time_points[0] == 0.0

    def test_units_ng_ml(self, dataset: ValidationDataset):
        ts = dataset.cytokine_levels
        assert ts is not None
        for var, unit in ts.units.items():
            assert unit == "ng/mL", f"{var} unit should be ng/mL"


# =============================================================================
# Metadata
# =============================================================================


class TestHPAMetadata:
    def test_metadata_fields(self):
        meta = get_hpa_metadata()
        assert meta.dataset_id == "HPA-skin-baseline"
        assert meta.source == DatasetSource.LOCAL
        assert meta.species == "human"
        assert meta.tissue_type == "skin"
        assert meta.url is not None
        assert meta.citation is not None
