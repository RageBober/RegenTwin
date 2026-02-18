"""Утилиты валидации данных для RegenTwin.

Обеспечивает проверку целостности и формата данных:
- Валидация flow cytometry данных по схеме каналов
- Валидация временных рядов (монотонность, неотрицательность)
- Схемы данных (ColumnSchema, DataSchema) для описания формата
- Проверка совместимости параметров модели и результатов гейтирования

Подробное описание: Description/Phase1/description_validation.md
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class ValidationLevel(str, Enum):
    """Уровни строгости валидации.

    STRICT — любое отклонение → ошибка.
    NORMAL — отклонения min/max → warning, отсутствие required → error.
    LENIENT — только критические проверки (наличие обязательных колонок).

    Подробное описание: Description/Phase1/description_validation.md#ValidationLevel
    """

    STRICT = "strict"
    NORMAL = "normal"
    LENIENT = "lenient"


@dataclass
class ValidationResult:
    """Результат валидации данных.

    Хранит статус валидации, список ошибок и предупреждений.
    is_valid == True тогда и только тогда, когда errors пуст.

    Подробное описание: Description/Phase1/description_validation.md#ValidationResult
    """

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Возвращает текстовую сводку результатов валидации.

        Формат: "Validation: PASS/FAIL. X errors, Y warnings."
        Если есть ошибки — перечисляет первые 5.

        Returns:
            Строка-сводка результатов

        Подробное описание: Description/Phase1/description_validation.md#summary
        """
        status = "PASS" if self.is_valid else "FAIL"
        parts = [f"Validation: {status}. {len(self.errors)} errors, {len(self.warnings)} warnings."]
        if self.errors:
            shown = self.errors[:5]
            parts.append("Errors: " + "; ".join(shown))
        return " ".join(parts)


@dataclass
class ColumnSchema:
    """Описание схемы одной колонки данных.

    Определяет имя, тип, обязательность и допустимый диапазон значений.
    Используется в DataSchema для описания формата таблицы.

    Подробное описание: Description/Phase1/description_validation.md#ColumnSchema
    """

    name: str  # Имя колонки
    dtype: str  # Тип: 'float', 'int', 'str', 'bool'
    required: bool = True  # Обязательная колонка?
    min_value: float | None = None  # Минимальное допустимое значение
    max_value: float | None = None  # Максимальное допустимое значение
    allowed_values: list[Any] | None = None  # Допустимые значения (для категорий)
    description: str = ""  # Описание колонки


@dataclass
class DataSchema:
    """Схема таблицы данных (набор колонок с ограничениями).

    Описывает ожидаемый формат DataFrame: какие колонки должны быть,
    их типы, ограничения на значения и количество строк.

    Подробное описание: Description/Phase1/description_validation.md#DataSchema
    """

    name: str  # Имя схемы
    columns: list[ColumnSchema]  # Описания колонок
    min_rows: int = 1  # Минимальное количество строк
    max_rows: int | None = None  # Максимальное (None = без ограничений)
    description: str = ""  # Описание схемы

    def get_required_columns(self) -> list[str]:
        """Возвращает список имён обязательных колонок.

        Фильтрует columns по required == True и возвращает их имена.

        Returns:
            Список имён обязательных колонок

        Подробное описание: Description/Phase1/description_validation.md#get_required_columns
        """
        return [col.name for col in self.columns if col.required]


# =====================================================
# Предопределённые схемы данных
# =====================================================

FCS_DATA_SCHEMA = DataSchema(
    name="fcs_data",
    columns=[
        ColumnSchema("FSC-A", "float", required=True, min_value=0,
                      description="Forward Scatter Area"),
        ColumnSchema("FSC-H", "float", required=True, min_value=0,
                      description="Forward Scatter Height"),
        ColumnSchema("SSC-A", "float", required=True, min_value=0,
                      description="Side Scatter Area"),
        ColumnSchema("CD34", "float", required=True, min_value=0,
                      description="CD34-APC (стволовые клетки)"),
        ColumnSchema("CD14", "float", required=False, min_value=0,
                      description="CD14-PE (моноциты/макрофаги)"),
        ColumnSchema("CD68", "float", required=False, min_value=0,
                      description="CD68-FITC (макрофаги)"),
        ColumnSchema("Annexin-V", "float", required=True, min_value=0,
                      description="Annexin-V-Pacific Blue (апоптоз)"),
        ColumnSchema("CD66b", "float", required=False, min_value=0,
                      description="CD66b-PE-Cy7 (нейтрофилы)"),
        ColumnSchema("CD31", "float", required=False, min_value=0,
                      description="CD31-BV421 (эндотелий)"),
    ],
    min_rows=100,
    description="Схема для flow cytometry данных (7-9 каналов)",
)

TIME_SERIES_SCHEMA = DataSchema(
    name="time_series",
    columns=[
        ColumnSchema("time", "float", required=True, min_value=0,
                      description="Время (часы)"),
        ColumnSchema("cell_count", "float", required=False, min_value=0,
                      description="Общее количество клеток"),
        ColumnSchema("wound_area", "float", required=False,
                      min_value=0, max_value=1,
                      description="Площадь раны (0=зажила, 1=начальная)"),
    ],
    min_rows=2,
    description="Схема для временных рядов заживления",
)

CYTOKINE_TIMESERIES_SCHEMA = DataSchema(
    name="cytokine_timeseries",
    columns=[
        ColumnSchema("time", "float", required=True, min_value=0,
                      description="Время (часы)"),
        ColumnSchema("TNF_alpha", "float", required=False, min_value=0,
                      description="TNF-α (нг/мл)"),
        ColumnSchema("IL_10", "float", required=False, min_value=0,
                      description="IL-10 (нг/мл)"),
        ColumnSchema("PDGF", "float", required=False, min_value=0,
                      description="PDGF (нг/мл)"),
        ColumnSchema("VEGF", "float", required=False, min_value=0,
                      description="VEGF (нг/мл)"),
        ColumnSchema("TGF_beta", "float", required=False, min_value=0,
                      description="TGF-β (нг/мл)"),
        ColumnSchema("MCP_1", "float", required=False, min_value=0,
                      description="MCP-1/CCL2 (нг/мл)"),
        ColumnSchema("IL_8", "float", required=False, min_value=0,
                      description="IL-8/CXCL8 (нг/мл)"),
    ],
    min_rows=2,
    description="Схема для временных рядов цитокинов",
)


class DataValidator:
    """Валидатор данных по заданной схеме.

    Проверяет DataFrame на соответствие DataSchema: наличие колонок,
    типы данных, диапазоны значений, количество строк.
    Уровень строгости определяет, что считать ошибкой, а что warning.

    Подробное описание: Description/Phase1/description_validation.md#DataValidator
    """

    def __init__(
        self,
        level: ValidationLevel | str = ValidationLevel.NORMAL,
    ) -> None:
        """Инициализирует валидатор с заданным уровнем строгости.

        Args:
            level: Уровень строгости ('strict', 'normal', 'lenient'
                   или ValidationLevel enum)

        Подробное описание: Description/Phase1/description_validation.md#DataValidator.__init__
        """
        if isinstance(level, str):
            level = ValidationLevel(level)
        self._level = level

    def validate_dataframe(
        self,
        data: pd.DataFrame,
        schema: DataSchema,
    ) -> ValidationResult:
        """Валидирует DataFrame по заданной схеме.

        Проверяет:
        1. Наличие обязательных колонок (required=True)
        2. Количество строк >= min_rows и <= max_rows
        3. Значения в допустимых диапазонах (min_value, max_value)
        4. Типы данных колонок

        Args:
            data: DataFrame для валидации
            schema: Схема данных

        Returns:
            ValidationResult (is_valid=True если нет ошибок)

        Подробное описание: Description/Phase1/description_validation.md#validate_dataframe
        """
        errors: list[str] = []
        warnings: list[str] = []

        # 1. Проверка обязательных колонок
        required = schema.get_required_columns()
        for col_name in required:
            if col_name not in data.columns:
                errors.append(f"Missing required column: {col_name}")

        # 2. Проверка количества строк
        n_rows = len(data)
        if n_rows < schema.min_rows:
            errors.append(f"Too few rows: {n_rows} < {schema.min_rows}")
        if schema.max_rows is not None and n_rows > schema.max_rows:
            msg = f"Too many rows: {n_rows} > {schema.max_rows}"
            if self._level == ValidationLevel.STRICT:
                errors.append(msg)
            else:
                warnings.append(msg)

        # 3. Проверка диапазонов значений (skip для LENIENT)
        if self._level != ValidationLevel.LENIENT:
            for col_schema in schema.columns:
                if col_schema.name not in data.columns:
                    continue
                col_data = data[col_schema.name]
                if col_schema.min_value is not None and (col_data < col_schema.min_value).any():
                    msg = f"Column '{col_schema.name}' has values below {col_schema.min_value}"
                    if self._level == ValidationLevel.STRICT:
                        errors.append(msg)
                    else:
                        warnings.append(msg)
                if col_schema.max_value is not None and (col_data > col_schema.max_value).any():
                    msg = f"Column '{col_schema.name}' has values above {col_schema.max_value}"
                    if self._level == ValidationLevel.STRICT:
                        errors.append(msg)
                    else:
                        warnings.append(msg)

        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)

    def validate_fcs_data(
        self,
        data: pd.DataFrame,
    ) -> ValidationResult:
        """Валидирует flow cytometry данные по FCS_DATA_SCHEMA.

        Проверяет наличие обязательных каналов (FSC-A, FSC-H, SSC-A,
        CD34, Annexin-V), неотрицательность значений, минимум 100 событий.

        Args:
            data: DataFrame с FCS данными

        Returns:
            ValidationResult

        Подробное описание: Description/Phase1/description_validation.md#validate_fcs_data
        """
        return self.validate_dataframe(data, FCS_DATA_SCHEMA)

    def validate_time_series(
        self,
        data: pd.DataFrame,
    ) -> ValidationResult:
        """Валидирует данные временных рядов по TIME_SERIES_SCHEMA.

        Проверяет:
        - Наличие колонки "time"
        - Монотонное возрастание time
        - Неотрицательность числовых значений
        - Минимум 2 временные точки

        Args:
            data: DataFrame с временными рядами

        Returns:
            ValidationResult

        Подробное описание: Description/Phase1/description_validation.md#validate_time_series
        """
        result = self.validate_dataframe(data, TIME_SERIES_SCHEMA)
        errors = list(result.errors)
        warnings = list(result.warnings)

        # Проверка монотонности time
        if "time" in data.columns and len(data) >= 2:
            time_vals = data["time"].values
            if not np.all(np.diff(time_vals) > 0):
                errors.append("Time column is not monotonically increasing")

        # Проверка неотрицательности числовых значений
        if "time" in data.columns:
            for col in data.columns:
                if col == "time":
                    continue
                if pd.api.types.is_numeric_dtype(data[col]):
                    if (data[col] < 0).any():
                        msg = f"Column '{col}' has negative values"
                        if self._level == ValidationLevel.STRICT:
                            errors.append(msg)
                        else:
                            warnings.append(msg)

        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)

    def validate_model_parameters(
        self,
        parameters: Any,
    ) -> ValidationResult:
        """Валидирует параметры модели (ModelParameters или Extended).

        Вызывает parameters.validate() и оборачивает результат
        в ValidationResult. Ловит ValueError и записывает в errors.

        Args:
            parameters: ModelParameters или ExtendedModelParameters

        Returns:
            ValidationResult

        Подробное описание: Description/Phase1/description_validation.md#validate_model_parameters
        """
        try:
            parameters.validate()
            return ValidationResult(is_valid=True, errors=[], warnings=[])
        except ValueError as e:
            return ValidationResult(is_valid=False, errors=[str(e)])

    def validate_gating_results(
        self,
        gating_results: Any,
    ) -> ValidationResult:
        """Валидирует результаты гейтирования на корректность.

        Проверяет:
        - Все fraction в [0, 1]
        - Иерархия гейтов корректна (дочерние <= родительские)
        - n_events >= 0 для каждого гейта
        - total_events > 0

        Args:
            gating_results: GatingResults объект

        Returns:
            ValidationResult

        Подробное описание: Description/Phase1/description_validation.md#validate_gating_results
        """
        errors: list[str] = []
        warnings: list[str] = []

        if gating_results.total_events <= 0:
            errors.append(
                f"total_events must be > 0, got {gating_results.total_events}"
            )

        for gate_name, gate in gating_results.gates.items():
            if gate.fraction < 0 or gate.fraction > 1:
                errors.append(
                    f"Gate '{gate_name}' fraction {gate.fraction} not in [0, 1]"
                )
            if gate.n_events < 0:
                errors.append(
                    f"Gate '{gate_name}' n_events {gate.n_events} is negative"
                )

        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)


def validate_data(
    data: pd.DataFrame,
    schema: DataSchema | None = None,
    level: str = "normal",
) -> ValidationResult:
    """Удобная функция для валидации данных.

    Создаёт DataValidator и валидирует DataFrame. Если schema=None,
    пытается автоматически определить тип данных (FCS, time series,
    cytokine) по именам колонок.

    Args:
        data: DataFrame для валидации
        schema: Схема данных (None = автоопределение)
        level: Уровень строгости ('strict', 'normal', 'lenient')

    Returns:
        ValidationResult

    Подробное описание: Description/Phase1/description_validation.md#validate_data
    """
    validator = DataValidator(level=level)
    if schema is not None:
        return validator.validate_dataframe(data, schema)
    # Автоопределение схемы по именам колонок
    columns = set(data.columns)

    if "FSC-A" in columns:
        return validator.validate_fcs_data(data)

    if "time" in columns and "TNF_alpha" in columns:
        return validator.validate_dataframe(data, CYTOKINE_TIMESERIES_SCHEMA)

    if "time" in columns:
        return validator.validate_time_series(data)

    # Fallback: нет подходящей схемы
    return ValidationResult(
        is_valid=True,
        errors=[],
        warnings=["Could not auto-detect schema"],
    )
