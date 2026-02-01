"""
TDD тесты для модуля fcs_parser.py

Тестирует:
- FCSMetadata dataclass
- FCSLoader класс
- load_fcs convenience функция

Основано на спецификации: Description/description_fcs_parser.md
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.data.fcs_parser import FCSMetadata, FCSLoader, load_fcs


# =============================================================================
# Тесты для FCSMetadata
# =============================================================================

class TestFCSMetadata:
    """Тесты для dataclass FCSMetadata."""

    def test_metadata_creation_with_all_fields(self):
        """Тест создания метаданных со всеми полями."""
        metadata = FCSMetadata(
            filename="test.fcs",
            n_events=10000,
            n_channels=7,
            channels=["FSC-A", "SSC-A", "CD34-APC"],
            cytometer="BD FACSAria",
            date="2026-01-22",
            fcs_version="3.1"
        )

        assert metadata.filename == "test.fcs"
        assert metadata.n_events == 10000
        assert metadata.n_channels == 7
        assert len(metadata.channels) == 3
        assert metadata.cytometer == "BD FACSAria"
        assert metadata.date == "2026-01-22"
        assert metadata.fcs_version == "3.1"

    def test_metadata_creation_with_optional_fields_none(self):
        """Тест создания метаданных с None для опциональных полей."""
        metadata = FCSMetadata(
            filename="test.fcs",
            n_events=5000,
            n_channels=5,
            channels=["FSC-A"],
            cytometer=None,
            date=None,
            fcs_version=None
        )

        assert metadata.filename == "test.fcs"
        assert metadata.n_events == 5000
        assert metadata.cytometer is None
        assert metadata.date is None
        assert metadata.fcs_version is None

    def test_metadata_channels_is_list(self):
        """Тест что channels - это список строк."""
        channels = ["FSC-A", "FSC-H", "SSC-A"]
        metadata = FCSMetadata(
            filename="test.fcs",
            n_events=1000,
            n_channels=3,
            channels=channels
        )

        assert isinstance(metadata.channels, list)
        assert all(isinstance(ch, str) for ch in metadata.channels)


# =============================================================================
# Тесты для FCSLoader.__init__
# =============================================================================

class TestFCSLoaderInit:
    """Тесты для FCSLoader.__init__."""

    def test_init_without_subsample(self):
        """Тест инициализации без субсэмплирования."""
        loader = FCSLoader()

        assert loader._subsample is None
        assert loader._sample is None
        assert loader._file_path is None

    def test_init_with_subsample(self):
        """Тест инициализации с указанным subsample."""
        loader = FCSLoader(subsample=5000)

        assert loader._subsample == 5000

    def test_init_with_zero_subsample(self):
        """Тест инициализации с subsample=0."""
        loader = FCSLoader(subsample=0)

        assert loader._subsample == 0

    def test_init_with_negative_subsample_should_raise(self):
        """Тест что отрицательный subsample вызывает ошибку."""
        with pytest.raises(ValueError):
            FCSLoader(subsample=-100)


# =============================================================================
# Тесты для FCSLoader.load
# =============================================================================

class TestFCSLoaderLoad:
    """Тесты для FCSLoader.load."""

    @patch('src.data.fcs_parser.Sample')
    def test_load_valid_file_returns_self(
        self, mock_sample_class, mock_fcs_file, mock_flowkit_sample
    ):
        """Тест что load возвращает self для цепочки вызовов."""
        mock_sample_class.return_value = mock_flowkit_sample

        loader = FCSLoader()
        result = loader.load(mock_fcs_file)

        assert result is loader

    @patch('src.data.fcs_parser.Sample')
    def test_load_sets_sample_attribute(
        self, mock_sample_class, mock_fcs_file, mock_flowkit_sample
    ):
        """Тест что load устанавливает _sample атрибут."""
        mock_sample_class.return_value = mock_flowkit_sample

        loader = FCSLoader()
        loader.load(mock_fcs_file)

        assert loader._sample is not None
        assert loader._sample == mock_flowkit_sample

    @patch('src.data.fcs_parser.Sample')
    def test_load_sets_file_path(
        self, mock_sample_class, mock_fcs_file, mock_flowkit_sample
    ):
        """Тест что load устанавливает _file_path."""
        mock_sample_class.return_value = mock_flowkit_sample

        loader = FCSLoader()
        loader.load(mock_fcs_file)

        assert loader._file_path == Path(mock_fcs_file)

    @patch('src.data.fcs_parser.Sample')
    def test_load_with_string_path(
        self, mock_sample_class, tmp_path, mock_flowkit_sample
    ):
        """Тест загрузки с путём как строка."""
        mock_sample_class.return_value = mock_flowkit_sample
        fcs_path = tmp_path / "test.fcs"
        fcs_path.touch()

        loader = FCSLoader()
        loader.load(str(fcs_path))

        assert loader._file_path == fcs_path

    def test_load_nonexistent_file_raises_file_not_found(self):
        """Тест что несуществующий файл вызывает FileNotFoundError."""
        loader = FCSLoader()

        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/to/file.fcs")

    @patch('src.data.fcs_parser.Sample')
    def test_load_invalid_fcs_format_raises_value_error(
        self, mock_sample_class, mock_fcs_file
    ):
        """Тест что невалидный формат вызывает ValueError."""
        mock_sample_class.side_effect = ValueError("Invalid FCS format")

        loader = FCSLoader()

        with pytest.raises(ValueError, match="Invalid FCS format"):
            loader.load(mock_fcs_file)

    @patch('src.data.fcs_parser.Sample')
    def test_load_passes_subsample_to_flowkit(
        self, mock_sample_class, mock_fcs_file, mock_flowkit_sample
    ):
        """Тест что subsample передаётся в FlowKit Sample."""
        mock_sample_class.return_value = mock_flowkit_sample

        loader = FCSLoader(subsample=5000)
        loader.load(mock_fcs_file)

        # Проверяем что Sample был вызван с правильными параметрами
        call_kwargs = mock_sample_class.call_args[1]
        assert call_kwargs.get('subsample') == 5000

    @patch('src.data.fcs_parser.Sample')
    def test_load_passes_ignore_offset_flags(
        self, mock_sample_class, mock_fcs_file, mock_flowkit_sample
    ):
        """Тест что флаги ignore_offset передаются."""
        mock_sample_class.return_value = mock_flowkit_sample

        loader = FCSLoader()
        loader.load(mock_fcs_file)

        call_kwargs = mock_sample_class.call_args[1]
        assert call_kwargs.get('ignore_offset_error') is True
        assert call_kwargs.get('ignore_offset_discrepancy') is True


# =============================================================================
# Тесты для FCSLoader.get_channels
# =============================================================================

class TestFCSLoaderGetChannels:
    """Тесты для FCSLoader.get_channels."""

    @patch('src.data.fcs_parser.Sample')
    def test_get_channels_returns_list(
        self, mock_sample_class, mock_fcs_file, mock_flowkit_sample
    ):
        """Тест что get_channels возвращает список."""
        mock_sample_class.return_value = mock_flowkit_sample

        loader = FCSLoader().load(mock_fcs_file)
        channels = loader.get_channels()

        assert isinstance(channels, list)

    @patch('src.data.fcs_parser.Sample')
    def test_get_channels_returns_strings(
        self, mock_sample_class, mock_fcs_file, mock_flowkit_sample
    ):
        """Тест что все элементы списка - строки."""
        mock_sample_class.return_value = mock_flowkit_sample

        loader = FCSLoader().load(mock_fcs_file)
        channels = loader.get_channels()

        assert all(isinstance(ch, str) for ch in channels)

    @patch('src.data.fcs_parser.Sample')
    def test_get_channels_expected_values(
        self, mock_sample_class, mock_fcs_file, mock_flowkit_sample, sample_channels
    ):
        """Тест что возвращаются ожидаемые каналы."""
        mock_flowkit_sample.pnn_labels = sample_channels
        mock_sample_class.return_value = mock_flowkit_sample

        loader = FCSLoader().load(mock_fcs_file)
        channels = loader.get_channels()

        assert channels == sample_channels

    def test_get_channels_without_load_raises_error(self):
        """Тест что вызов без load() вызывает ошибку."""
        loader = FCSLoader()

        with pytest.raises((RuntimeError, ValueError, AttributeError)):
            loader.get_channels()


# =============================================================================
# Тесты для FCSLoader.get_events
# =============================================================================

class TestFCSLoaderGetEvents:
    """Тесты для FCSLoader.get_events."""

    @patch('src.data.fcs_parser.Sample')
    def test_get_events_returns_ndarray(
        self, mock_sample_class, mock_fcs_file, mock_flowkit_sample
    ):
        """Тест что get_events возвращает numpy array."""
        mock_sample_class.return_value = mock_flowkit_sample

        loader = FCSLoader().load(mock_fcs_file)
        events = loader.get_events()

        assert isinstance(events, np.ndarray)

    @patch('src.data.fcs_parser.Sample')
    def test_get_events_shape_matches_events_channels(
        self, mock_sample_class, mock_fcs_file, mock_flowkit_sample
    ):
        """Тест формы массива [n_events, n_channels]."""
        mock_data = np.random.rand(10000, 7)
        mock_flowkit_sample.get_events.return_value = mock_data
        mock_sample_class.return_value = mock_flowkit_sample

        loader = FCSLoader().load(mock_fcs_file)
        events = loader.get_events()

        assert events.shape == (10000, 7)

    @patch('src.data.fcs_parser.Sample')
    def test_get_events_raw_source(
        self, mock_sample_class, mock_fcs_file, mock_flowkit_sample
    ):
        """Тест получения raw данных."""
        mock_sample_class.return_value = mock_flowkit_sample

        loader = FCSLoader().load(mock_fcs_file)
        loader.get_events(source="raw")

        mock_flowkit_sample.get_events.assert_called_with(source="raw")

    @patch('src.data.fcs_parser.Sample')
    def test_get_events_comp_source(
        self, mock_sample_class, mock_fcs_file, mock_flowkit_sample
    ):
        """Тест получения компенсированных данных."""
        mock_sample_class.return_value = mock_flowkit_sample

        loader = FCSLoader().load(mock_fcs_file)
        loader.get_events(source="comp")

        mock_flowkit_sample.get_events.assert_called_with(source="comp")

    @patch('src.data.fcs_parser.Sample')
    def test_get_events_xform_source(
        self, mock_sample_class, mock_fcs_file, mock_flowkit_sample
    ):
        """Тест получения трансформированных данных."""
        mock_sample_class.return_value = mock_flowkit_sample

        loader = FCSLoader().load(mock_fcs_file)
        loader.get_events(source="xform")

        mock_flowkit_sample.get_events.assert_called_with(source="xform")

    @patch('src.data.fcs_parser.Sample')
    def test_get_events_with_channel_filter(
        self, mock_sample_class, mock_fcs_file, mock_flowkit_sample, sample_channels
    ):
        """Тест фильтрации по каналам."""
        full_data = np.random.rand(10000, 7)
        mock_flowkit_sample.get_events.return_value = full_data
        mock_flowkit_sample.pnn_labels = sample_channels
        mock_sample_class.return_value = mock_flowkit_sample

        loader = FCSLoader().load(mock_fcs_file)
        events = loader.get_events(channels=["FSC-A", "SSC-A"])

        assert events.shape[1] == 2

    @patch('src.data.fcs_parser.Sample')
    def test_get_events_default_returns_all_channels(
        self, mock_sample_class, mock_fcs_file, mock_flowkit_sample, sample_channels
    ):
        """Тест что без фильтра возвращаются все каналы."""
        mock_flowkit_sample.pnn_labels = sample_channels
        mock_sample_class.return_value = mock_flowkit_sample

        loader = FCSLoader().load(mock_fcs_file)
        events = loader.get_events()

        assert events.shape[1] == len(sample_channels)

    def test_get_events_without_load_raises_error(self):
        """Тест что вызов без load() вызывает ошибку."""
        loader = FCSLoader()

        with pytest.raises((RuntimeError, ValueError, AttributeError)):
            loader.get_events()


# =============================================================================
# Тесты для FCSLoader.get_metadata
# =============================================================================

class TestFCSLoaderGetMetadata:
    """Тесты для FCSLoader.get_metadata."""

    @patch('src.data.fcs_parser.Sample')
    def test_get_metadata_returns_fcsmetadata(
        self, mock_sample_class, mock_fcs_file, mock_flowkit_sample
    ):
        """Тест что возвращается FCSMetadata."""
        mock_sample_class.return_value = mock_flowkit_sample

        loader = FCSLoader().load(mock_fcs_file)
        metadata = loader.get_metadata()

        assert isinstance(metadata, FCSMetadata)

    @patch('src.data.fcs_parser.Sample')
    def test_get_metadata_filename_from_path(
        self, mock_sample_class, mock_fcs_file, mock_flowkit_sample
    ):
        """Тест что filename извлекается из пути."""
        mock_sample_class.return_value = mock_flowkit_sample

        loader = FCSLoader().load(mock_fcs_file)
        metadata = loader.get_metadata()

        assert metadata.filename == mock_fcs_file.name

    @patch('src.data.fcs_parser.Sample')
    def test_get_metadata_n_events_from_sample(
        self, mock_sample_class, mock_fcs_file, mock_flowkit_sample
    ):
        """Тест корректности n_events."""
        mock_flowkit_sample.event_count = 10000
        mock_sample_class.return_value = mock_flowkit_sample

        loader = FCSLoader().load(mock_fcs_file)
        metadata = loader.get_metadata()

        assert metadata.n_events == 10000

    @patch('src.data.fcs_parser.Sample')
    def test_get_metadata_n_channels_correct(
        self, mock_sample_class, mock_fcs_file, mock_flowkit_sample, sample_channels
    ):
        """Тест корректности n_channels."""
        mock_flowkit_sample.pnn_labels = sample_channels
        mock_sample_class.return_value = mock_flowkit_sample

        loader = FCSLoader().load(mock_fcs_file)
        metadata = loader.get_metadata()

        assert metadata.n_channels == len(sample_channels)

    @patch('src.data.fcs_parser.Sample')
    def test_get_metadata_cytometer_from_cyt(
        self, mock_sample_class, mock_fcs_file, mock_flowkit_sample
    ):
        """Тест извлечения cytometer из $CYT."""
        mock_flowkit_sample.get_metadata.return_value = {
            '$CYT': 'BD FACSAria',
            '$DATE': '22-JAN-2026',
            'FCSversion': '3.1'
        }
        mock_sample_class.return_value = mock_flowkit_sample

        loader = FCSLoader().load(mock_fcs_file)
        metadata = loader.get_metadata()

        assert metadata.cytometer == 'BD FACSAria'

    @patch('src.data.fcs_parser.Sample')
    def test_get_metadata_fcs_version(
        self, mock_sample_class, mock_fcs_file, mock_flowkit_sample
    ):
        """Тест извлечения версии FCS."""
        mock_flowkit_sample.get_metadata.return_value = {'FCSversion': '3.1'}
        mock_sample_class.return_value = mock_flowkit_sample

        loader = FCSLoader().load(mock_fcs_file)
        metadata = loader.get_metadata()

        assert metadata.fcs_version == '3.1'

    @patch('src.data.fcs_parser.Sample')
    def test_get_metadata_missing_optional_fields_are_none(
        self, mock_sample_class, mock_fcs_file, mock_flowkit_sample
    ):
        """Тест что отсутствующие опциональные поля = None."""
        mock_flowkit_sample.get_metadata.return_value = {}  # Пустые метаданные
        mock_sample_class.return_value = mock_flowkit_sample

        loader = FCSLoader().load(mock_fcs_file)
        metadata = loader.get_metadata()

        assert metadata.cytometer is None
        assert metadata.date is None
        assert metadata.fcs_version is None

    def test_get_metadata_without_load_raises_error(self):
        """Тест что вызов без load() вызывает ошибку."""
        loader = FCSLoader()

        with pytest.raises((RuntimeError, ValueError, AttributeError)):
            loader.get_metadata()


# =============================================================================
# Тесты для FCSLoader.to_dataframe
# =============================================================================

class TestFCSLoaderToDataframe:
    """Тесты для FCSLoader.to_dataframe."""

    @patch('src.data.fcs_parser.Sample')
    def test_to_dataframe_returns_dataframe(
        self, mock_sample_class, mock_fcs_file, mock_flowkit_sample
    ):
        """Тест что возвращается pandas DataFrame."""
        mock_sample_class.return_value = mock_flowkit_sample

        loader = FCSLoader().load(mock_fcs_file)
        df = loader.to_dataframe()

        assert isinstance(df, pd.DataFrame)

    @patch('src.data.fcs_parser.Sample')
    def test_to_dataframe_columns_match_channels(
        self, mock_sample_class, mock_fcs_file, mock_flowkit_sample, sample_channels
    ):
        """Тест что колонки соответствуют каналам."""
        mock_df = pd.DataFrame(np.random.rand(100, 7), columns=sample_channels)
        mock_flowkit_sample.as_dataframe.return_value = mock_df
        mock_sample_class.return_value = mock_flowkit_sample

        loader = FCSLoader().load(mock_fcs_file)
        df = loader.to_dataframe()

        assert list(df.columns) == sample_channels

    @patch('src.data.fcs_parser.Sample')
    def test_to_dataframe_with_channel_filter(
        self, mock_sample_class, mock_fcs_file, mock_flowkit_sample, sample_channels
    ):
        """Тест фильтрации колонок."""
        mock_df = pd.DataFrame(np.random.rand(100, 7), columns=sample_channels)
        mock_flowkit_sample.as_dataframe.return_value = mock_df
        mock_sample_class.return_value = mock_flowkit_sample

        loader = FCSLoader().load(mock_fcs_file)
        df = loader.to_dataframe(channels=["FSC-A", "SSC-A"])

        assert list(df.columns) == ["FSC-A", "SSC-A"]

    @patch('src.data.fcs_parser.Sample')
    def test_to_dataframe_row_count_matches_events(
        self, mock_sample_class, mock_fcs_file, mock_flowkit_sample, sample_channels
    ):
        """Тест что количество строк соответствует событиям."""
        mock_df = pd.DataFrame(np.random.rand(10000, 7), columns=sample_channels)
        mock_flowkit_sample.as_dataframe.return_value = mock_df
        mock_sample_class.return_value = mock_flowkit_sample

        loader = FCSLoader().load(mock_fcs_file)
        df = loader.to_dataframe()

        assert len(df) == 10000

    def test_to_dataframe_without_load_raises_error(self):
        """Тест что вызов без load() вызывает ошибку."""
        loader = FCSLoader()

        with pytest.raises((RuntimeError, ValueError, AttributeError)):
            loader.to_dataframe()


# =============================================================================
# Тесты для FCSLoader.get_channel_data
# =============================================================================

class TestFCSLoaderGetChannelData:
    """Тесты для FCSLoader.get_channel_data."""

    @patch('src.data.fcs_parser.Sample')
    def test_get_channel_data_returns_1d_array(
        self, mock_sample_class, mock_fcs_file, mock_flowkit_sample
    ):
        """Тест что возвращается 1D массив."""
        mock_flowkit_sample.get_channel_events.return_value = np.random.rand(10000)
        mock_sample_class.return_value = mock_flowkit_sample

        loader = FCSLoader().load(mock_fcs_file)
        data = loader.get_channel_data("FSC-A")

        assert isinstance(data, np.ndarray)
        assert data.ndim == 1

    @patch('src.data.fcs_parser.Sample')
    def test_get_channel_data_correct_length(
        self, mock_sample_class, mock_fcs_file, mock_flowkit_sample
    ):
        """Тест корректной длины массива."""
        mock_flowkit_sample.get_channel_events.return_value = np.random.rand(10000)
        mock_flowkit_sample.event_count = 10000
        mock_sample_class.return_value = mock_flowkit_sample

        loader = FCSLoader().load(mock_fcs_file)
        data = loader.get_channel_data("FSC-A")

        assert len(data) == 10000

    @patch('src.data.fcs_parser.Sample')
    def test_get_channel_data_nonexistent_channel_raises(
        self, mock_sample_class, mock_fcs_file, mock_flowkit_sample
    ):
        """Тест что несуществующий канал вызывает ошибку."""
        mock_flowkit_sample.get_channel_events.side_effect = KeyError("Channel not found")
        mock_sample_class.return_value = mock_flowkit_sample

        loader = FCSLoader().load(mock_fcs_file)

        with pytest.raises((KeyError, ValueError)):
            loader.get_channel_data("NONEXISTENT_CHANNEL")

    @patch('src.data.fcs_parser.Sample')
    def test_get_channel_data_source_parameter(
        self, mock_sample_class, mock_fcs_file, mock_flowkit_sample
    ):
        """Тест что source параметр передаётся."""
        mock_flowkit_sample.get_channel_events.return_value = np.random.rand(100)
        mock_sample_class.return_value = mock_flowkit_sample

        loader = FCSLoader().load(mock_fcs_file)
        loader.get_channel_data("FSC-A", source="comp")

        mock_flowkit_sample.get_channel_events.assert_called_with("FSC-A", source="comp")

    def test_get_channel_data_without_load_raises_error(self):
        """Тест что вызов без load() вызывает ошибку."""
        loader = FCSLoader()

        with pytest.raises((RuntimeError, ValueError, AttributeError)):
            loader.get_channel_data("FSC-A")


# =============================================================================
# Тесты для FCSLoader.validate_required_channels
# =============================================================================

class TestFCSLoaderValidateRequiredChannels:
    """Тесты для FCSLoader.validate_required_channels."""

    @patch('src.data.fcs_parser.Sample')
    def test_validate_all_channels_present_returns_true(
        self, mock_sample_class, mock_fcs_file, mock_flowkit_sample, sample_channels
    ):
        """Тест что True возвращается когда все каналы присутствуют."""
        mock_flowkit_sample.pnn_labels = sample_channels
        mock_sample_class.return_value = mock_flowkit_sample

        loader = FCSLoader().load(mock_fcs_file)
        result = loader.validate_required_channels(["FSC-A", "SSC-A"])

        assert result is True

    @patch('src.data.fcs_parser.Sample')
    def test_validate_missing_channel_raises_value_error(
        self, mock_sample_class, mock_fcs_file, mock_flowkit_sample
    ):
        """Тест что отсутствующий канал вызывает ValueError."""
        mock_flowkit_sample.pnn_labels = ["FSC-A", "SSC-A"]
        mock_sample_class.return_value = mock_flowkit_sample

        loader = FCSLoader().load(mock_fcs_file)

        with pytest.raises(ValueError):
            loader.validate_required_channels(["FSC-A", "CD34-APC"])

    @patch('src.data.fcs_parser.Sample')
    def test_validate_partial_match_by_substring(
        self, mock_sample_class, mock_fcs_file, mock_flowkit_sample
    ):
        """Тест поиска по подстроке (CD34 -> CD34-APC)."""
        mock_flowkit_sample.pnn_labels = ["FSC-A", "SSC-A", "CD34-APC"]
        mock_sample_class.return_value = mock_flowkit_sample

        loader = FCSLoader().load(mock_fcs_file)
        result = loader.validate_required_channels(["CD34"])

        assert result is True

    @patch('src.data.fcs_parser.Sample')
    def test_validate_empty_required_list_returns_true(
        self, mock_sample_class, mock_fcs_file, mock_flowkit_sample
    ):
        """Тест с пустым списком обязательных каналов."""
        mock_sample_class.return_value = mock_flowkit_sample

        loader = FCSLoader().load(mock_fcs_file)
        result = loader.validate_required_channels([])

        assert result is True

    @patch('src.data.fcs_parser.Sample')
    def test_validate_error_message_lists_missing_channels(
        self, mock_sample_class, mock_fcs_file, mock_flowkit_sample
    ):
        """Тест что сообщение об ошибке содержит список недостающих каналов."""
        mock_flowkit_sample.pnn_labels = ["FSC-A"]
        mock_sample_class.return_value = mock_flowkit_sample

        loader = FCSLoader().load(mock_fcs_file)

        with pytest.raises(ValueError) as exc_info:
            loader.validate_required_channels(["CD34", "CD14", "Annexin-V"])

        error_message = str(exc_info.value)
        assert "CD34" in error_message
        assert "CD14" in error_message
        assert "Annexin-V" in error_message

    @patch('src.data.fcs_parser.Sample')
    def test_validate_case_sensitivity(
        self, mock_sample_class, mock_fcs_file, mock_flowkit_sample
    ):
        """Тест чувствительности к регистру."""
        mock_flowkit_sample.pnn_labels = ["FSC-A", "SSC-A"]
        mock_sample_class.return_value = mock_flowkit_sample

        loader = FCSLoader().load(mock_fcs_file)

        # fsc-a (нижний регистр) должен матчить FSC-A или нет?
        # Документация не уточняет, но substring match может быть case-insensitive
        # Тест проверит текущее поведение
        try:
            result = loader.validate_required_channels(["fsc-a"])
            # Если прошло без ошибки - case-insensitive
            assert result is True
        except ValueError:
            # Если ошибка - case-sensitive
            pass  # Это тоже приемлемое поведение


# =============================================================================
# Тесты для load_fcs функции
# =============================================================================

class TestLoadFcsFunction:
    """Тесты для convenience функции load_fcs."""

    @patch('src.data.fcs_parser.Sample')
    def test_load_fcs_returns_fcsloader(
        self, mock_sample_class, mock_fcs_file, mock_flowkit_sample
    ):
        """Тест что возвращается FCSLoader."""
        mock_sample_class.return_value = mock_flowkit_sample

        loader = load_fcs(mock_fcs_file)

        assert isinstance(loader, FCSLoader)

    @patch('src.data.fcs_parser.Sample')
    def test_load_fcs_with_subsample(
        self, mock_sample_class, mock_fcs_file, mock_flowkit_sample
    ):
        """Тест передачи subsample параметра."""
        mock_sample_class.return_value = mock_flowkit_sample

        loader = load_fcs(mock_fcs_file, subsample=5000)

        assert loader._subsample == 5000

    @patch('src.data.fcs_parser.Sample')
    def test_load_fcs_file_is_loaded(
        self, mock_sample_class, mock_fcs_file, mock_flowkit_sample
    ):
        """Тест что файл загружен."""
        mock_sample_class.return_value = mock_flowkit_sample

        loader = load_fcs(mock_fcs_file)

        assert loader._sample is not None

    @patch('src.data.fcs_parser.Sample')
    def test_load_fcs_can_chain_get_channels(
        self, mock_sample_class, mock_fcs_file, mock_flowkit_sample, sample_channels
    ):
        """Тест цепочки вызовов load_fcs().get_channels()."""
        mock_flowkit_sample.pnn_labels = sample_channels
        mock_sample_class.return_value = mock_flowkit_sample

        channels = load_fcs(mock_fcs_file).get_channels()

        assert channels == sample_channels
