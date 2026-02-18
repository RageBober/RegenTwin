"""Data Pipeline для обработки flow cytometry данных.

Содержит модули:
- fcs_parser — парсинг .fcs файлов
- gating — автоматический gating
- parameter_extraction — извлечение параметров для модели
- image_loader — загрузка и анализ изображений scatter plots
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
from src.data.image_loader import (
    ImageAnalysisResult,
    ImageAnalyzer,
    ImageConfig,
    ImageLoader,
    ImageMetadata,
    ScatterPlotData,
    ScatterPlotExtractor,
    analyze_image,
    extract_scatter_plot,
    load_image,
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
    # image_loader
    "ImageConfig",
    "ImageMetadata",
    "ImageLoader",
    "ScatterPlotData",
    "ScatterPlotExtractor",
    "ImageAnalysisResult",
    "ImageAnalyzer",
    "load_image",
    "extract_scatter_plot",
    "analyze_image",
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
]
