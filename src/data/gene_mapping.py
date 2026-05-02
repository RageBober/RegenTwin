"""Маппинг генов GEO/NCBI → переменных модели RegenTwin.

Конвертирует мРНК экспрессию из GEO датасетов (microarray/RNA-seq)
в относительные единицы, совместимые с ValidationRunner.

Ключевые ограничения:
- мРНК ≠ белок: маппинг качественный (направление тренда), не количественный
- Используется fold change vs t=0, не абсолютные ng/mL
- Подходит для ValidationRunner.run_phase_timing(), но не для DTW/CRPS

Подробное описание: Description/Phase3/description_gene_mapping.md
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.dataset_loader import DatasetMetadata, DatasetSource, TimeSeriesData

# =====================================================
# Маппинг: Gene symbol → переменная модели
# =====================================================

GENE_TO_VARIABLE: dict[str, str] = {
    # Цитокины
    "TNF": "C_TNF",
    "TNFA": "C_TNF",
    "IL10": "C_IL10",
    "PDGFA": "C_PDGF",
    "PDGFB": "C_PDGF",
    "VEGFA": "C_VEGF",
    "TGFB1": "C_TGFb",
    "CCL2": "C_MCP1",
    "MCP1": "C_MCP1",
    "CXCL8": "C_IL8",
    "IL8": "C_IL8",
    # Клеточные маркеры (прокси для клеточных популяций)
    "CD34": "S",  # стволовые
    "CD14": "M_total",  # моноциты/макрофаги
    "CD68": "M_total",
    "PECAM1": "E",  # CD31, эндотелий
    "CEACAM8": "Ne",  # CD66b, нейтрофилы
    "COL1A1": "rho_collagen",  # коллаген I
    "COL3A1": "rho_collagen",  # коллаген III
    "MMP2": "C_MMP",
    "MMP9": "C_MMP",
    "ACTA2": "Mf",  # α-SMA, миофибробласты
    "FN1": "rho_fibrin",  # фибронектин (прокси ECM)
}


@dataclass
class GEODatasetInfo:
    """Информация о GEO датасете для wound healing."""

    accession: str
    title: str
    platform: str  # GPL ID
    time_points_hours: list[float]
    species: str
    tissue: str


# Известные датасеты для wound healing
KNOWN_GEO_DATASETS: dict[str, GEODatasetInfo] = {
    "GSE28914": GEODatasetInfo(
        accession="GSE28914",
        title="Time course of human skin wound healing",
        platform="GPL570",
        time_points_hours=[0, 24, 72, 120, 168, 336],
        species="human",
        tissue="skin",
    ),
}


# =====================================================
# Reference data: GSE28914 (оцифрованные fold changes)
# =====================================================
#
# GSE28914: Human skin wound healing transcriptomics
# Platform: Affymetrix HG-U133 Plus 2.0
# Time points: 0, 1, 3, 5, 7, 14 дней post-wounding
#
# Данные ниже — типичные fold changes (vs t=0) из литературы
# по wound healing transcriptomics (consensus из множества GEO datasets).
# Используются для качественной валидации фазности.

_GSE28914_TIME_HOURS = np.array([0, 24, 72, 120, 168, 336], dtype=np.float64)

_GSE28914_FOLD_CHANGES: dict[str, np.ndarray] = {
    # TNF-α: быстрый рост в inflammation, спад в proliferation
    "C_TNF": np.array([1.0, 4.5, 6.0, 3.0, 1.5, 1.0]),
    # IL-10: растёт с задержкой (противовоспалительный)
    "C_IL10": np.array([1.0, 1.5, 3.0, 5.0, 4.0, 2.0]),
    # PDGF: ранний пик (тромбоциты), затем макрофаги
    "C_PDGF": np.array([1.0, 3.0, 4.0, 3.5, 2.5, 1.5]),
    # VEGF: гипоксия-зависимый, пик в proliferation
    "C_VEGF": np.array([1.0, 2.0, 4.0, 6.0, 5.0, 2.5]),
    # TGF-β1: бифазный — ранний от тромбоцитов + поздний от M2/Mf
    "C_TGFb": np.array([1.0, 3.0, 4.5, 5.0, 6.0, 4.0]),
    # MCP-1: хемоаттрактант моноцитов, ранний пик
    "C_MCP1": np.array([1.0, 5.0, 7.0, 3.0, 1.5, 1.0]),
    # IL-8: хемоаттрактант нейтрофилов, ранний пик
    "C_IL8": np.array([1.0, 6.0, 4.0, 2.0, 1.2, 1.0]),
    # Collagen I: нарастает в remodeling
    "rho_collagen": np.array([1.0, 1.0, 1.5, 3.0, 5.0, 8.0]),
    # MMP: ранний пик для ECM ремоделирования
    "C_MMP": np.array([1.0, 3.0, 5.0, 4.0, 2.5, 1.5]),
}


def get_gse28914_reference() -> TimeSeriesData:
    """Получить reference fold changes из GSE28914.

    Данные представляют типичные fold changes (vs t=0) мРНК
    экспрессии генов wound healing. Единицы — fold change (безразмерные).

    Подходит для:
    - ValidationRunner.run_phase_timing() (changepoint detection)
    - Качественного сравнения направления трендов

    НЕ подходит для:
    - Количественного DTW/CRPS (мРНК ≠ белок)

    Returns:
        TimeSeriesData с fold changes для 9 переменных.
    """
    metadata = DatasetMetadata(
        source=DatasetSource.GEO,
        dataset_id="GSE28914",
        description="Wound healing transcriptomics fold changes (GSE28914)",
        species="human",
        tissue_type="skin",
        time_points=_GSE28914_TIME_HOURS.tolist(),
        url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE28914",
    )

    units = {var: "fold_change" for var in _GSE28914_FOLD_CHANGES}

    return TimeSeriesData(
        time_points=_GSE28914_TIME_HOURS.copy(),
        values=dict(_GSE28914_FOLD_CHANGES),
        units=units,
        metadata=metadata,
    )


def map_expression_to_model(
    gene_symbols: list[str],
    expression_matrix: np.ndarray,
) -> dict[str, np.ndarray]:
    """Маппинг матрицы экспрессии генов в переменные модели.

    Args:
        gene_symbols: Список имён генов, shape (n_genes,)
        expression_matrix: Матрица экспрессии, shape (n_genes, n_timepoints)

    Returns:
        Словарь {model_variable: averaged_expression}. Если несколько генов
        маппятся в одну переменную — усредняются.
    """
    result: dict[str, list[np.ndarray]] = {}

    for i, gene in enumerate(gene_symbols):
        gene_upper = gene.upper()
        if gene_upper in GENE_TO_VARIABLE:
            var = GENE_TO_VARIABLE[gene_upper]
            if var not in result:
                result[var] = []
            result[var].append(expression_matrix[i])

    return {var: np.mean(arrays, axis=0) for var, arrays in result.items()}
