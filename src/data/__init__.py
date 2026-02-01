"""Data Pipeline для обработки flow cytometry данных.

Содержит модули:
- fcs_parser — парсинг .fcs файлов
- gating — автоматический gating
- parameter_extraction — извлечение параметров для модели

Подробное описание: Description/description_data.md
"""

from src.data.fcs_parser import FCSLoader, FCSMetadata, load_fcs
from src.data.gating import GateResult, GatingResults, GatingStrategy
from src.data.parameter_extraction import (
    ExtractionConfig,
    ModelParameters,
    ParameterExtractor,
    extract_model_parameters,
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
    "ExtractionConfig",
    "extract_model_parameters",
]
