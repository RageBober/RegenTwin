"""Data Pipeline для обработки flow cytometry данных.

Содержит модули:
- fcs_parser — парсинг .fcs файлов
- gating — автоматический gating
- parameter_extraction — извлечение параметров для модели
- validation — валидация данных по схемам
- dataset_loader — загрузка публичных датасетов

Подробное описание: Description/description_data.md
"""

from src.data.dataset_loader import (
    AVAILABLE_DATASETS,
    DatasetLoader,
    DatasetMetadata,
    DatasetSource,
    TimeSeriesData,
    ValidationDataset,
    load_dataset,
)
from src.data.fcs_parser import FCSLoader, FCSMetadata, load_fcs
from src.data.gating import GateResult, GatingResults, GatingStrategy
from src.data.gene_mapping import (
    GENE_TO_VARIABLE,
    get_gse28914_reference,
    map_expression_to_model,
)
from src.data.hpa_client import (
    HPASkinExpression,
    get_baseline_concentrations,
    get_hpa_validation_dataset,
    get_skin_baseline,
)
from src.data.literature_data import (
    LiteratureCitation,
    ReferenceSource,
    get_flegg2010_reference,
    get_variable_mapping,
    get_xue2009_phase_breakpoints,
    get_xue2009_reference,
)
from src.data.parameter_extraction import (
    ExtendedModelParameters,
    ExtractionConfig,
    ModelParameters,
    ParameterExtractor,
    extract_extended_parameters,
    extract_model_parameters,
)
from src.data.validation import (
    ColumnSchema,
    DataSchema,
    DataValidator,
    ValidationLevel,
    ValidationResult,
    validate_data,
)

__all__ = [
    # fcs_parser
    "FCSLoader",
    "FCSMetadata",
    "load_fcs",
    # gating
    "GatingStrategy",
    "GatingResults",
    "GateResult",
    # parameter_extraction
    "ParameterExtractor",
    "ModelParameters",
    "ExtendedModelParameters",
    "ExtractionConfig",
    "extract_model_parameters",
    "extract_extended_parameters",
    # validation
    "ValidationLevel",
    "ValidationResult",
    "ColumnSchema",
    "DataSchema",
    "DataValidator",
    "validate_data",
    # dataset_loader
    "DatasetSource",
    "DatasetMetadata",
    "TimeSeriesData",
    "ValidationDataset",
    "DatasetLoader",
    "AVAILABLE_DATASETS",
    "load_dataset",
    # literature_data
    "ReferenceSource",
    "LiteratureCitation",
    "get_xue2009_reference",
    "get_flegg2010_reference",
    "get_xue2009_phase_breakpoints",
    "get_variable_mapping",
    # gene_mapping
    "GENE_TO_VARIABLE",
    "get_gse28914_reference",
    "map_expression_to_model",
    # hpa_client
    "HPASkinExpression",
    "get_skin_baseline",
    "get_baseline_concentrations",
    "get_hpa_validation_dataset",
]
