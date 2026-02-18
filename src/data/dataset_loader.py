"""Загрузчик публичных датасетов для валидации моделей RegenTwin.

Поддерживает загрузку данных из:
- FlowRepository (FR-FCM-*): flow cytometry данные ран
- GEO (NCBI): транскриптомные временные ряды
- Локальные файлы: data/validation/ директория

Предоставляет кэширование загруженных данных, интерполяцию
временных рядов и извлечение начальных условий для модели.

Подробное описание: Description/Phase1/description_dataset_loader.md
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class DatasetSource(str, Enum):
    """Источник данных для валидации.

    FLOW_REPOSITORY — FlowRepository.org (FR-FCM-* идентификаторы).
    GEO — Gene Expression Omnibus (GSE* идентификаторы).
    LOCAL — локальные файлы из data/validation/.

    Подробное описание: Description/Phase1/description_dataset_loader.md#DatasetSource
    """

    FLOW_REPOSITORY = "flow_repository"
    GEO = "geo"
    LOCAL = "local"


@dataclass
class DatasetMetadata:
    """Метаданные датасета: источник, описание, расположение файлов.

    Содержит всю информацию о датасете, необходимую для его загрузки
    и идентификации. Не содержит самих данных.

    Подробное описание: Description/Phase1/description_dataset_loader.md#DatasetMetadata
    """

    source: DatasetSource  # Откуда данные
    dataset_id: str  # Уникальный идентификатор
    description: str  # Описание датасета
    file_paths: list[str] = field(default_factory=list)  # Пути к файлам
    species: str = "human"  # Вид организма
    tissue_type: str = "skin"  # Тип ткани
    n_samples: int = 0  # Количество образцов
    time_points: list[float] | None = None  # Временные точки (часы)
    url: str | None = None  # URL источника
    citation: str | None = None  # Ссылка на публикацию


@dataclass
class TimeSeriesData:
    """Данные временного ряда для валидации модели.

    Хранит временные точки и набор переменных (клеточные популяции,
    цитокины, площадь раны). Поддерживает конвертацию в DataFrame
    и интерполяцию на новую временную сетку.

    Подробное описание: Description/Phase1/description_dataset_loader.md#TimeSeriesData
    """

    time_points: np.ndarray  # Временные точки (часы), shape=(n_points,)
    values: dict[str, np.ndarray]  # {variable_name: array}, каждый shape=(n_points,)
    units: dict[str, str]  # {variable_name: unit_string}
    metadata: DatasetMetadata | None = None  # Источник данных

    def to_dataframe(self) -> pd.DataFrame:
        """Конвертирует временной ряд в pandas DataFrame.

        Создаёт DataFrame с колонкой "time" и колонками для каждой
        переменной из values. Число строк = len(time_points).

        Returns:
            DataFrame с колонкой "time" и колонками переменных

        Подробное описание: Description/Phase1/description_dataset_loader.md#TimeSeriesData.to_dataframe
        """
        data = {"time": self.time_points}
        for name, values in self.values.items():
            data[name] = values
        return pd.DataFrame(data)

    def get_variable(self, name: str) -> np.ndarray:
        """Возвращает данные одной переменной по имени.

        Args:
            name: Имя переменной (ключ в values)

        Returns:
            NumPy массив значений, shape=(n_points,)

        Raises:
            KeyError: Если переменная не найдена в values

        Подробное описание: Description/Phase1/description_dataset_loader.md#get_variable
        """
        if name not in self.values:
            raise KeyError(
                f"Variable '{name}' not found. "
                f"Available: {list(self.values.keys())}"
            )
        return self.values[name]

    def interpolate(
        self,
        new_time_points: np.ndarray,
    ) -> "TimeSeriesData":
        """Интерполирует все переменные на новую временную сетку.

        Использует линейную интерполяцию (scipy.interpolate.interp1d).
        Возвращает новый TimeSeriesData с интерполированными значениями.
        Длина результата = len(new_time_points).

        Args:
            new_time_points: Новые временные точки (часы)

        Returns:
            Новый TimeSeriesData с интерполированными данными

        Подробное описание: Description/Phase1/description_dataset_loader.md#interpolate
        """
        new_values = {}
        for name, vals in self.values.items():
            new_values[name] = np.interp(
                new_time_points, self.time_points, vals
            )
        return TimeSeriesData(
            time_points=new_time_points,
            values=new_values,
            units=self.units.copy(),
            metadata=self.metadata,
        )


@dataclass
class ValidationDataset:
    """Полный датасет для валидации модели регенерации.

    Может содержать flow cytometry данные (для начальных условий),
    временные ряды клеточных популяций и цитокинов (для валидации
    динамики), данные по заживлению раны.

    Подробное описание: Description/Phase1/description_dataset_loader.md#ValidationDataset
    """

    metadata: DatasetMetadata  # Метаданные
    cell_counts: TimeSeriesData | None = None  # Временной ряд клеток
    cytokine_levels: TimeSeriesData | None = None  # Временной ряд цитокинов
    fcs_data: pd.DataFrame | None = None  # Raw FCS данные
    wound_closure: TimeSeriesData | None = None  # Динамика заживления
    raw_data: dict[str, Any] = field(default_factory=dict)  # Прочие данные

    def get_initial_conditions(self) -> dict[str, float]:
        """Извлекает начальные условия (t=0) из данных.

        Берёт значения при t=0 из cell_counts и cytokine_levels.
        Ключи соответствуют именам переменных модели (P0, Ne0, C_TNF...).

        Returns:
            Словарь {variable_name: initial_value}

        Подробное описание: Description/Phase1/description_dataset_loader.md#get_initial_conditions
        """
        result: dict[str, float] = {}
        for source in [self.cell_counts, self.cytokine_levels]:
            if source is not None and len(source.time_points) > 0:
                for name, values in source.values.items():
                    result[name] = float(values[0])
        return result

    def get_validation_targets(self) -> dict[str, "TimeSeriesData"]:
        """Возвращает целевые данные для сравнения с симуляцией.

        Ключи: "cell_counts", "cytokine_levels", "wound_closure"
        (только те, что доступны в датасете).

        Returns:
            Словарь {target_name: TimeSeriesData}

        Подробное описание: Description/Phase1/description_dataset_loader.md#get_validation_targets
        """
        targets: dict[str, TimeSeriesData] = {}
        if self.cell_counts is not None:
            targets["cell_counts"] = self.cell_counts
        if self.cytokine_levels is not None:
            targets["cytokine_levels"] = self.cytokine_levels
        if self.wound_closure is not None:
            targets["wound_closure"] = self.wound_closure
        return targets


# =====================================================
# Реестр доступных датасетов
# =====================================================

AVAILABLE_DATASETS: dict[str, DatasetMetadata] = {
    "FR-FCM-wound-healing": DatasetMetadata(
        source=DatasetSource.FLOW_REPOSITORY,
        dataset_id="FR-FCM-wound-healing",
        description="Flow cytometry данные заживления кожной раны",
        species="human",
        tissue_type="skin",
        url="https://flowrepository.org/",
    ),
    "GSE28914": DatasetMetadata(
        source=DatasetSource.GEO,
        dataset_id="GSE28914",
        description="Транскриптомные данные заживления раны, временной ряд",
        species="human",
        tissue_type="skin",
        time_points=[0, 1, 3, 5, 7, 14],
        url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE28914",
    ),
    "local-mock": DatasetMetadata(
        source=DatasetSource.LOCAL,
        dataset_id="local-mock",
        description="Локальные мок-данные для тестирования пайплайна",
        file_paths=["data/mock/"],
    ),
}


class DatasetLoader:
    """Загрузчик датасетов с кэшированием.

    Загружает датасеты по ID из реестра AVAILABLE_DATASETS.
    Поддерживает локальные файлы и скачивание из публичных репозиториев.
    Кэширует загруженные датасеты в памяти для повторного использования.

    Подробное описание: Description/Phase1/description_dataset_loader.md#DatasetLoader
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
    ) -> None:
        """Инициализирует загрузчик с директорией кэша.

        Args:
            cache_dir: Директория для кэширования загруженных файлов.
                       По умолчанию data/validation/.

        Подробное описание: Description/Phase1/description_dataset_loader.md#DatasetLoader.__init__
        """
        self._cache_dir = Path(cache_dir) if cache_dir else Path("data/validation")
        self._loaded_datasets: dict[str, ValidationDataset] = {}

    def list_available(self) -> list[DatasetMetadata]:
        """Возвращает список всех зарегистрированных датасетов.

        Читает AVAILABLE_DATASETS и возвращает список DatasetMetadata.
        Всегда содержит минимум 3 элемента (текущий реестр).

        Returns:
            Список DatasetMetadata для всех доступных датасетов

        Подробное описание: Description/Phase1/description_dataset_loader.md#list_available
        """
        return list(AVAILABLE_DATASETS.values())

    def load(
        self,
        dataset_id: str,
    ) -> ValidationDataset:
        """Загружает датасет по идентификатору.

        Порядок: (1) проверяет кэш в памяти, (2) ищет локальные файлы,
        (3) пытается скачать. Повторный вызов с тем же id возвращает
        закэшированный объект.

        Args:
            dataset_id: Идентификатор из AVAILABLE_DATASETS

        Returns:
            ValidationDataset с данными

        Raises:
            KeyError: Если dataset_id не найден в реестре
            FileNotFoundError: Если данные недоступны локально

        Подробное описание: Description/Phase1/description_dataset_loader.md#load
        """
        if dataset_id not in AVAILABLE_DATASETS:
            raise KeyError(f"Dataset '{dataset_id}' not found in registry")

        # Проверка кэша
        if dataset_id in self._loaded_datasets:
            return self._loaded_datasets[dataset_id]

        metadata = AVAILABLE_DATASETS[dataset_id]

        # Попытка загрузки из локальных файлов
        try:
            dataset = self._load_local(metadata)
            self._loaded_datasets[dataset_id] = dataset
            return dataset
        except Exception:
            pass

        raise FileNotFoundError(
            f"Dataset '{dataset_id}' not available locally"
        )

    def download(
        self,
        dataset_id: str,
        target_dir: str | Path | None = None,
    ) -> Path:
        """Скачивает датасет из удалённого репозитория.

        Args:
            dataset_id: Идентификатор датасета
            target_dir: Целевая директория (по умолчанию cache_dir)

        Returns:
            Path к директории с загруженными файлами

        Raises:
            KeyError: Если dataset_id не найден
            ConnectionError: Ошибка подключения к репозиторию

        Подробное описание: Description/Phase1/description_dataset_loader.md#download
        """
        if dataset_id not in AVAILABLE_DATASETS:
            raise KeyError(f"Dataset '{dataset_id}' not found in registry")

        raise ConnectionError(
            f"Download not implemented for dataset '{dataset_id}'"
        )

    def validate_dataset(
        self,
        dataset: ValidationDataset,
    ) -> bool:
        """Проверяет целостность загруженного датасета.

        Проверяет наличие обязательных полей, корректность временных
        точек (монотонность), неотрицательность значений.

        Args:
            dataset: Датасет для проверки

        Returns:
            True если датасет валиден

        Raises:
            ValueError: С описанием нарушения

        Подробное описание: Description/Phase1/description_dataset_loader.md#validate_dataset
        """
        if not dataset.metadata.dataset_id:
            raise ValueError("dataset_id must not be empty")

        for source_name, source in [
            ("cell_counts", dataset.cell_counts),
            ("cytokine_levels", dataset.cytokine_levels),
            ("wound_closure", dataset.wound_closure),
        ]:
            if source is not None:
                # Монотонность времени
                if len(source.time_points) >= 2:
                    if not np.all(np.diff(source.time_points) > 0):
                        raise ValueError(
                            f"{source_name}: time_points are not "
                            "monotonically increasing"
                        )
                # Неотрицательность значений
                for var_name, values in source.values.items():
                    if np.any(values < 0):
                        raise ValueError(
                            f"{source_name}.{var_name}: "
                            "contains negative values"
                        )

        return True

    def _load_local(
        self,
        metadata: DatasetMetadata,
    ) -> ValidationDataset:
        """Загружает датасет из локальных файлов.

        Ищет .fcs файлы в fcs/ поддиректории и .csv/.json файлы
        в time_series/ поддиректории cache_dir.

        Args:
            metadata: Метаданные с путями к файлам

        Returns:
            ValidationDataset

        Подробное описание: Description/Phase1/description_dataset_loader.md#_load_local
        """
        # Определяем базовую директорию
        if metadata.file_paths:
            base_dir = Path(metadata.file_paths[0])
        else:
            base_dir = self._cache_dir / metadata.dataset_id

        if not base_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {base_dir}"
            )

        # Загрузка FCS файлов
        fcs_dir = base_dir / "fcs"
        fcs_data = self._load_fcs_files(fcs_dir) if fcs_dir.exists() else None

        # Загрузка временных рядов
        ts_dir = base_dir / "time_series"
        cell_counts = None

        if ts_dir.exists():
            for file_path in ts_dir.iterdir():
                ts = self._load_time_series(file_path)
                if ts is not None and cell_counts is None:
                    cell_counts = ts

        return ValidationDataset(
            metadata=metadata,
            fcs_data=fcs_data,
            cell_counts=cell_counts,
        )

    def _load_fcs_files(
        self,
        directory: Path,
    ) -> pd.DataFrame | None:
        """Загружает и объединяет FCS файлы из директории.

        Использует FCSLoader для каждого .fcs файла и конкатенирует
        результаты в один DataFrame.

        Args:
            directory: Директория с .fcs файлами

        Returns:
            DataFrame или None если файлы не найдены

        Подробное описание: Description/Phase1/description_dataset_loader.md#_load_fcs_files
        """
        if not directory.exists():
            return None

        fcs_files = list(directory.glob("*.fcs"))
        if not fcs_files:
            return None

        frames = []
        for fcs_file in fcs_files:
            try:
                from src.data.fcs_parser import FCSLoader
                loader = FCSLoader()
                loader.load(str(fcs_file))
                df = loader.to_dataframe()
                frames.append(df)
            except Exception:
                continue

        if not frames:
            return None

        return pd.concat(frames, ignore_index=True)

    def _load_time_series(
        self,
        file_path: Path,
    ) -> TimeSeriesData | None:
        """Загружает временной ряд из CSV или JSON файла.

        CSV: первая колонка "time", остальные — переменные.
        JSON: {"time_points": [...], "values": {"var": [...]}, "units": {...}}.

        Args:
            file_path: Путь к файлу (.csv или .json)

        Returns:
            TimeSeriesData или None если формат не поддерживается

        Подробное описание: Description/Phase1/description_dataset_loader.md#_load_time_series
        """
        if not file_path.exists():
            return None

        suffix = file_path.suffix.lower()

        if suffix == ".csv":
            try:
                df = pd.read_csv(file_path)
                if "time" not in df.columns:
                    return None
                time_points = df["time"].values.astype(float)
                values = {}
                units = {}
                for col in df.columns:
                    if col != "time":
                        values[col] = df[col].values.astype(float)
                        units[col] = "unknown"
                return TimeSeriesData(
                    time_points=time_points,
                    values=values,
                    units=units,
                )
            except Exception:
                return None

        elif suffix == ".json":
            import json
            try:
                with open(file_path) as f:
                    data = json.load(f)
                time_points = np.array(data["time_points"], dtype=float)
                values = {
                    k: np.array(v, dtype=float)
                    for k, v in data["values"].items()
                }
                units = data.get(
                    "units", {k: "unknown" for k in values}
                )
                return TimeSeriesData(
                    time_points=time_points,
                    values=values,
                    units=units,
                )
            except Exception:
                return None

        return None  # Неподдерживаемый формат


def load_dataset(
    dataset_id: str,
    cache_dir: str | Path | None = None,
) -> ValidationDataset:
    """Удобная функция для загрузки датасета.

    Создаёт DatasetLoader и вызывает load().

    Args:
        dataset_id: Идентификатор из AVAILABLE_DATASETS
        cache_dir: Директория кэша (опционально)

    Returns:
        ValidationDataset

    Raises:
        KeyError: Если dataset_id не найден
        FileNotFoundError: Если данные недоступны

    Подробное описание: Description/Phase1/description_dataset_loader.md#load_dataset
    """
    loader = DatasetLoader(cache_dir=cache_dir)
    return loader.load(dataset_id)
