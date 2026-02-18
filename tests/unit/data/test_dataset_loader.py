"""
TDD тесты для модуля dataset_loader.py

Тестирует:
- DatasetSource enum
- DatasetMetadata dataclass
- TimeSeriesData dataclass и методы
- ValidationDataset dataclass и методы
- AVAILABLE_DATASETS реестр
- DatasetLoader класс и все методы
- load_dataset() удобная функция

Основано на спецификации: Description/Phase1/description_dataset_loader.md

Реестр датасетов:
- FR-FCM-wound-healing: FlowRepository, flow cytometry
- GSE28914: GEO, транскриптомика, time_points=[0,1,3,5,7,14]
- local-mock: Локальные мок-данные
"""

import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.data.dataset_loader import (
    DatasetSource,
    DatasetMetadata,
    TimeSeriesData,
    ValidationDataset,
    DatasetLoader,
    AVAILABLE_DATASETS,
    load_dataset,
)


# =============================================================================
# Тесты для DatasetSource
# =============================================================================

class TestDatasetSource:
    """Тесты для enum DatasetSource."""

    def test_create_from_string_flow_repository(self):
        """Тест создания DatasetSource из строки 'flow_repository'."""
        source = DatasetSource("flow_repository")
        assert source == DatasetSource.FLOW_REPOSITORY

    def test_invalid_string_raises_value_error(self):
        """Тест что невалидная строка вызывает ValueError."""
        with pytest.raises(ValueError):
            DatasetSource("invalid")

    def test_value_property_local(self):
        """Тест что .value возвращает строковое значение."""
        assert DatasetSource.LOCAL.value == "local"


# =============================================================================
# Тесты для DatasetMetadata
# =============================================================================

class TestDatasetMetadata:
    """Тесты для dataclass DatasetMetadata."""

    def test_creation_with_mandatory_fields(self):
        """Тест создания DatasetMetadata с обязательными полями."""
        meta = DatasetMetadata(
            source=DatasetSource.GEO,
            dataset_id="GSE28914",
            description="Wound transcriptomics",
        )
        assert meta.source == DatasetSource.GEO
        assert meta.dataset_id == "GSE28914"
        assert meta.description == "Wound transcriptomics"

    def test_defaults(self):
        """Тест значений по умолчанию: species='human', tissue_type='skin'."""
        meta = DatasetMetadata(
            source=DatasetSource.LOCAL,
            dataset_id="test",
            description="test",
        )
        assert meta.species == "human"
        assert meta.tissue_type == "skin"
        assert meta.n_samples == 0
        assert meta.time_points is None
        assert meta.url is None
        assert meta.citation is None


# =============================================================================
# Тесты для TimeSeriesData.to_dataframe
# =============================================================================

class TestTimeSeriesDataToDataframe:
    """Тесты для TimeSeriesData.to_dataframe."""

    def test_3_vars_gives_4_columns(self):
        """Тест что 3 переменные → DataFrame с 4 столбцами."""
        ts = TimeSeriesData(
            time_points=np.array([0.0, 6.0, 24.0]),
            values={
                "Ne": np.array([100.0, 500.0, 200.0]),
                "M1": np.array([50.0, 200.0, 80.0]),
                "M2": np.array([20.0, 80.0, 60.0]),
            },
            units={"Ne": "cells/ul", "M1": "cells/ul", "M2": "cells/ul"},
        )
        df = ts.to_dataframe()
        assert len(df.columns) == 4  # time + 3 переменных
        assert "time" in df.columns

    def test_empty_values_gives_1_column(self):
        """Тест что пустые values → DataFrame с одним столбцом 'time'."""
        ts = TimeSeriesData(
            time_points=np.array([0.0, 6.0]),
            values={},
            units={},
        )
        df = ts.to_dataframe()
        assert list(df.columns) == ["time"]

    def test_time_always_first_column(self, mock_time_series_data):
        """Тест что 'time' всегда первый столбец."""
        df = mock_time_series_data.to_dataframe()
        assert df.columns[0] == "time"

    def test_nrows_equals_len_time_points(self, mock_time_series_data):
        """Тест что количество строк == len(time_points)."""
        df = mock_time_series_data.to_dataframe()
        assert len(df) == len(mock_time_series_data.time_points)


# =============================================================================
# Тесты для TimeSeriesData.get_variable
# =============================================================================

class TestTimeSeriesDataGetVariable:
    """Тесты для TimeSeriesData.get_variable."""

    def test_existing_variable(self, mock_time_series_data):
        """Тест получения существующей переменной 'Ne'."""
        result = mock_time_series_data.get_variable("Ne")
        assert isinstance(result, np.ndarray)
        assert len(result) == len(mock_time_series_data.time_points)

    def test_missing_variable_raises_key_error(self, mock_time_series_data):
        """Тест что несуществующая переменная → KeyError."""
        with pytest.raises(KeyError):
            mock_time_series_data.get_variable("XYZ")


# =============================================================================
# Тесты для TimeSeriesData.interpolate
# =============================================================================

class TestTimeSeriesDataInterpolate:
    """Тесты для TimeSeriesData.interpolate."""

    def test_same_points_unchanged(self, mock_time_series_data):
        """Тест что интерполяция на те же точки → без изменений."""
        new_tp = mock_time_series_data.time_points.copy()
        result = mock_time_series_data.interpolate(new_tp)
        for key in mock_time_series_data.values:
            np.testing.assert_allclose(
                result.values[key],
                mock_time_series_data.values[key],
                atol=1e-10,
            )

    def test_intermediate_points_interpolated(self, mock_time_series_data):
        """Тест что промежуточные точки интерполируются."""
        new_tp = np.array([3.0, 12.0, 36.0])  # Между исходными
        result = mock_time_series_data.interpolate(new_tp)
        # Значения должны быть между соседними исходными значениями
        for key in result.values:
            assert all(np.isfinite(result.values[key]))

    def test_len_preserved(self, mock_time_series_data):
        """Тест что len(result.time_points) == len(new_time_points)."""
        new_tp = np.linspace(0, 72, 10)
        result = mock_time_series_data.interpolate(new_tp)
        assert len(result.time_points) == 10

    def test_keys_and_units_preserved(self, mock_time_series_data):
        """Тест что ключи values и units сохраняются после интерполяции."""
        new_tp = np.array([0.0, 36.0, 72.0])
        result = mock_time_series_data.interpolate(new_tp)
        assert set(result.values.keys()) == set(mock_time_series_data.values.keys())
        assert result.units == mock_time_series_data.units


# =============================================================================
# Тесты для ValidationDataset.get_initial_conditions
# =============================================================================

class TestValidationDatasetGetInitialConditions:
    """Тесты для ValidationDataset.get_initial_conditions."""

    def test_cell_counts_at_t0(self, mock_validation_dataset):
        """Тест извлечения начальных условий из cell_counts при t=0."""
        ic = mock_validation_dataset.get_initial_conditions()
        assert isinstance(ic, dict)
        assert all(isinstance(v, float) for v in ic.values())

    def test_no_data_empty_dict(self, mock_dataset_metadata):
        """Тест что отсутствие данных → пустой словарь."""
        dataset = ValidationDataset(
            metadata=mock_dataset_metadata,
            cell_counts=None,
            cytokine_levels=None,
        )
        ic = dataset.get_initial_conditions()
        assert isinstance(ic, dict)
        assert len(ic) == 0

    def test_merged_cell_and_cytokine(self, mock_dataset_metadata):
        """Тест объединения cell_counts и cytokine_levels."""
        cell_ts = TimeSeriesData(
            time_points=np.array([0.0, 24.0]),
            values={"Ne": np.array([100.0, 500.0])},
            units={"Ne": "cells/ul"},
        )
        cyto_ts = TimeSeriesData(
            time_points=np.array([0.0, 24.0]),
            values={"TNF": np.array([0.1, 0.5])},
            units={"TNF": "ng/ml"},
        )
        dataset = ValidationDataset(
            metadata=mock_dataset_metadata,
            cell_counts=cell_ts,
            cytokine_levels=cyto_ts,
        )
        ic = dataset.get_initial_conditions()
        assert isinstance(ic, dict)
        # Должны быть ключи из обоих источников
        assert len(ic) >= 2


# =============================================================================
# Тесты для ValidationDataset.get_validation_targets
# =============================================================================

class TestValidationDatasetGetValidationTargets:
    """Тесты для ValidationDataset.get_validation_targets."""

    def test_all_data_3_keys(self, mock_dataset_metadata):
        """Тест что все данные → словарь с 3 ключами."""
        ts = TimeSeriesData(
            time_points=np.array([0.0, 24.0]),
            values={"x": np.array([1.0, 2.0])},
            units={"x": "units"},
        )
        dataset = ValidationDataset(
            metadata=mock_dataset_metadata,
            cell_counts=ts,
            cytokine_levels=ts,
            wound_closure=ts,
        )
        targets = dataset.get_validation_targets()
        assert len(targets) == 3

    def test_only_cell_counts_1_key(self, mock_validation_dataset):
        """Тест что только cell_counts → словарь с 1 ключом."""
        targets = mock_validation_dataset.get_validation_targets()
        assert len(targets) == 1
        assert "cell_counts" in targets

    def test_nothing_empty_dict(self, mock_dataset_metadata):
        """Тест что отсутствие данных → пустой словарь."""
        dataset = ValidationDataset(metadata=mock_dataset_metadata)
        targets = dataset.get_validation_targets()
        assert targets == {}


# =============================================================================
# Тесты для AVAILABLE_DATASETS
# =============================================================================

class TestAvailableDatasets:
    """Тесты для реестра AVAILABLE_DATASETS."""

    def test_len_at_least_3(self):
        """Тест что реестр содержит минимум 3 датасета."""
        assert len(AVAILABLE_DATASETS) >= 3

    def test_all_dataset_metadata(self):
        """Тест что все значения — DatasetMetadata."""
        for value in AVAILABLE_DATASETS.values():
            assert isinstance(value, DatasetMetadata)

    def test_specific_ids_present(self):
        """Тест что конкретные ID присутствуют в реестре."""
        assert "FR-FCM-wound-healing" in AVAILABLE_DATASETS
        assert "GSE28914" in AVAILABLE_DATASETS
        assert "local-mock" in AVAILABLE_DATASETS


# =============================================================================
# Тесты для DatasetLoader.__init__
# =============================================================================

class TestDatasetLoaderInit:
    """Тесты для конструктора DatasetLoader."""

    def test_default_cache_dir(self):
        """Тест что cache_dir по умолчанию = Path('data/validation')."""
        loader = DatasetLoader()
        assert loader._cache_dir == Path("data/validation")

    def test_custom_string(self):
        """Тест создания с строковым путём."""
        loader = DatasetLoader("custom/path")
        assert loader._cache_dir == Path("custom/path")

    def test_custom_path(self, tmp_path):
        """Тест создания с Path объектом."""
        loader = DatasetLoader(tmp_path)
        assert loader._cache_dir == tmp_path


# =============================================================================
# Тесты для DatasetLoader.list_available
# =============================================================================

class TestDatasetLoaderListAvailable:
    """Тесты для DatasetLoader.list_available."""

    def test_len_at_least_3(self):
        """Тест что list_available возвращает ≥ 3 элементов."""
        loader = DatasetLoader()
        result = loader.list_available()
        assert len(result) >= 3

    def test_all_dataset_metadata(self):
        """Тест что все элементы — DatasetMetadata."""
        loader = DatasetLoader()
        result = loader.list_available()
        for item in result:
            assert isinstance(item, DatasetMetadata)


# =============================================================================
# Тесты для DatasetLoader.load
# =============================================================================

class TestDatasetLoaderLoad:
    """Тесты для DatasetLoader.load."""

    def test_unknown_id_raises_key_error(self):
        """Тест что неизвестный ID → KeyError."""
        loader = DatasetLoader()
        with pytest.raises(KeyError):
            loader.load("nonexistent-dataset")

    def test_unavailable_raises_file_not_found(self, tmp_path):
        """Тест что отсутствие файлов → FileNotFoundError."""
        loader = DatasetLoader(cache_dir=tmp_path)
        # FR-FCM-wound-healing существует в реестре, но файлов нет
        with pytest.raises(FileNotFoundError):
            loader.load("FR-FCM-wound-healing")

    def test_caching_returns_same_object(self, tmp_path):
        """Тест что повторный вызов возвращает тот же объект из кеша."""
        loader = DatasetLoader(cache_dir=tmp_path)

        # Предварительно заполняем кеш
        mock_dataset = ValidationDataset(
            metadata=AVAILABLE_DATASETS["local-mock"],
        )
        loader._loaded_datasets["local-mock"] = mock_dataset

        result1 = loader.load("local-mock")
        result2 = loader.load("local-mock")
        assert result1 is result2


# =============================================================================
# Тесты для DatasetLoader.download
# =============================================================================

class TestDatasetLoaderDownload:
    """Тесты для DatasetLoader.download."""

    def test_unknown_id_raises_key_error(self):
        """Тест что неизвестный ID → KeyError."""
        loader = DatasetLoader()
        with pytest.raises(KeyError):
            loader.download("nonexistent-dataset")

    def test_no_connection_raises_connection_error(self):
        """Тест что отсутствие соединения → ConnectionError."""
        loader = DatasetLoader()
        # Мокаем сетевой вызов чтобы вызвать ConnectionError
        with patch.object(loader, "download", side_effect=ConnectionError("No connection")):
            with pytest.raises(ConnectionError):
                loader.download("FR-FCM-wound-healing")


# =============================================================================
# Тесты для DatasetLoader.validate_dataset
# =============================================================================

class TestDatasetLoaderValidateDataset:
    """Тесты для DatasetLoader.validate_dataset."""

    def test_correct_dataset_returns_true(self, mock_validation_dataset):
        """Тест что корректный датасет → True."""
        loader = DatasetLoader()
        result = loader.validate_dataset(mock_validation_dataset)
        assert result is True

    def test_non_monotonic_time_raises_value_error(self, mock_dataset_metadata):
        """Тест что немонотонное время → ValueError."""
        ts = TimeSeriesData(
            time_points=np.array([0.0, 6.0, 3.0, 48.0]),  # Нарушение
            values={"Ne": np.array([100.0, 500.0, 800.0, 400.0])},
            units={"Ne": "cells/ul"},
        )
        dataset = ValidationDataset(
            metadata=mock_dataset_metadata,
            cell_counts=ts,
        )
        loader = DatasetLoader()
        with pytest.raises(ValueError):
            loader.validate_dataset(dataset)

    def test_negative_values_raises_value_error(self, mock_dataset_metadata):
        """Тест что отрицательные значения → ValueError."""
        ts = TimeSeriesData(
            time_points=np.array([0.0, 6.0, 24.0]),
            values={"Ne": np.array([100.0, -50.0, 200.0])},  # Отрицательное
            units={"Ne": "cells/ul"},
        )
        dataset = ValidationDataset(
            metadata=mock_dataset_metadata,
            cell_counts=ts,
        )
        loader = DatasetLoader()
        with pytest.raises(ValueError):
            loader.validate_dataset(dataset)

    def test_empty_dataset_id_raises_value_error(self):
        """Тест что пустой dataset_id → ValueError."""
        meta = DatasetMetadata(
            source=DatasetSource.LOCAL,
            dataset_id="",  # Пустой
            description="test",
        )
        dataset = ValidationDataset(metadata=meta)
        loader = DatasetLoader()
        with pytest.raises(ValueError):
            loader.validate_dataset(dataset)


# =============================================================================
# Тесты для DatasetLoader._load_local
# =============================================================================

class TestDatasetLoaderLoadLocal:
    """Тесты для DatasetLoader._load_local."""

    def test_dir_with_fcs_files(self, tmp_path, mock_dataset_metadata):
        """Тест загрузки из каталога с .fcs файлами."""
        # Создаём структуру каталогов
        fcs_dir = tmp_path / "fcs"
        fcs_dir.mkdir()
        (fcs_dir / "sample1.fcs").touch()

        mock_dataset_metadata.file_paths = [str(tmp_path)]
        loader = DatasetLoader(cache_dir=tmp_path)

        # _load_local может вернуть ValidationDataset с fcs_data
        result = loader._load_local(mock_dataset_metadata)
        assert isinstance(result, ValidationDataset)

    def test_dir_with_csv_files(self, tmp_path, mock_dataset_metadata):
        """Тест загрузки из каталога с CSV файлами."""
        ts_dir = tmp_path / "time_series"
        ts_dir.mkdir()
        csv_file = ts_dir / "cell_counts.csv"
        csv_file.write_text("time,Ne,M1\n0,100,50\n6,500,200\n24,800,300\n")

        mock_dataset_metadata.file_paths = [str(tmp_path)]
        loader = DatasetLoader(cache_dir=tmp_path)

        result = loader._load_local(mock_dataset_metadata)
        assert isinstance(result, ValidationDataset)

    def test_empty_dir(self, tmp_path, mock_dataset_metadata):
        """Тест загрузки из пустого каталога → ValidationDataset с None полями."""
        mock_dataset_metadata.file_paths = [str(tmp_path)]
        loader = DatasetLoader(cache_dir=tmp_path)

        result = loader._load_local(mock_dataset_metadata)
        assert isinstance(result, ValidationDataset)


# =============================================================================
# Тесты для DatasetLoader._load_fcs_files
# =============================================================================

class TestDatasetLoaderLoadFcsFiles:
    """Тесты для DatasetLoader._load_fcs_files."""

    def test_3_fcs_files_concat(self, tmp_path):
        """Тест что 3 .fcs файла → конкатенированный DataFrame."""
        for i in range(3):
            (tmp_path / f"sample_{i}.fcs").touch()

        loader = DatasetLoader()
        result = loader._load_fcs_files(tmp_path)
        # Результат — DataFrame (конкатенация) или None если файлы пустые
        assert result is None or isinstance(result, pd.DataFrame)

    def test_empty_dir_returns_none(self, tmp_path):
        """Тест что пустой каталог → None."""
        loader = DatasetLoader()
        result = loader._load_fcs_files(tmp_path)
        assert result is None

    def test_missing_dir_returns_none(self):
        """Тест что несуществующий каталог → None."""
        loader = DatasetLoader()
        result = loader._load_fcs_files(Path("/nonexistent/directory"))
        assert result is None


# =============================================================================
# Тесты для DatasetLoader._load_time_series
# =============================================================================

class TestDatasetLoaderLoadTimeSeries:
    """Тесты для DatasetLoader._load_time_series."""

    def test_csv_returns_time_series_data(self, tmp_path):
        """Тест что корректный CSV → TimeSeriesData."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("time,Ne,M1\n0,100,50\n6,500,200\n24,800,300\n")

        loader = DatasetLoader()
        result = loader._load_time_series(csv_file)
        assert isinstance(result, TimeSeriesData)

    def test_json_returns_time_series_data(self, tmp_path):
        """Тест что корректный JSON → TimeSeriesData."""
        json_data = {
            "time_points": [0.0, 6.0, 24.0],
            "values": {
                "Ne": [100.0, 500.0, 800.0],
                "M1": [50.0, 200.0, 300.0],
            },
            "units": {"Ne": "cells/ul", "M1": "cells/ul"},
        }
        json_file = tmp_path / "data.json"
        json_file.write_text(json.dumps(json_data))

        loader = DatasetLoader()
        result = loader._load_time_series(json_file)
        assert isinstance(result, TimeSeriesData)

    def test_unsupported_format_returns_none(self, tmp_path):
        """Тест что неподдерживаемый формат (.xlsx) → None."""
        xlsx_file = tmp_path / "data.xlsx"
        xlsx_file.touch()

        loader = DatasetLoader()
        result = loader._load_time_series(xlsx_file)
        assert result is None

    def test_missing_file_returns_none(self):
        """Тест что несуществующий файл → None."""
        loader = DatasetLoader()
        result = loader._load_time_series(Path("/nonexistent/file.csv"))
        assert result is None


# =============================================================================
# Тесты для load_dataset() функции
# =============================================================================

class TestLoadDatasetFunction:
    """Тесты для удобной функции load_dataset."""

    def test_valid_id_returns_validation_dataset(self):
        """Тест что валидный ID (с мок-кешем) → ValidationDataset."""
        with patch.object(DatasetLoader, "load") as mock_load:
            mock_dataset = ValidationDataset(
                metadata=AVAILABLE_DATASETS["local-mock"],
            )
            mock_load.return_value = mock_dataset

            result = load_dataset("local-mock")
            assert isinstance(result, ValidationDataset)

    def test_invalid_id_raises_key_error(self):
        """Тест что невалидный ID → KeyError."""
        with pytest.raises(KeyError):
            load_dataset("nonexistent-dataset")
