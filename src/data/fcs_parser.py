"""Загрузчик .fcs файлов для RegenTwin.

Использует библиотеку FlowKit для парсинга FCS 2.0, 3.0, 3.1 файлов.

Подробное описание: Description/description_fcs_parser.md
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from flowkit import Sample


@dataclass
class FCSMetadata:
    """Метаданные FCS файла.

    Подробное описание: Description/description_fcs_parser.md#FCSMetadata
    """

    filename: str
    n_events: int
    n_channels: int
    channels: list[str]
    cytometer: str | None = None
    date: str | None = None
    fcs_version: str | None = None


class FCSLoader:
    """Загрузчик flow cytometry данных из .fcs файлов.

    Использует FlowKit Sample class для парсинга.
    Поддерживает FCS версии 2.0, 3.0, 3.1.

    Подробное описание: Description/description_fcs_parser.md#FCSLoader
    """

    def __init__(self, subsample: int | None = None) -> None:
        """Инициализация загрузчика.

        Args:
            subsample: Максимальное количество событий для загрузки (None = все)

        Подробное описание: Description/description_fcs_parser.md#__init__
        """
        if subsample is not None and subsample < 0:
            raise ValueError("subsample must be non-negative")
        self._subsample = subsample
        self._sample = None
        self._file_path = None

    def load(self, file_path: str | Path) -> "FCSLoader":
        """Загрузка .fcs файла.

        Args:
            file_path: Путь к .fcs файлу

        Returns:
            Self для цепочки вызовов

        Raises:
            FileNotFoundError: Файл не найден
            ValueError: Некорректный формат файла

        Подробное описание: Description/description_fcs_parser.md#load
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        kwargs: dict = {
            "fcs_path_or_data": file_path,
            "ignore_offset_error": True,
            "ignore_offset_discrepancy": True,
        }
        if self._subsample is not None:
            kwargs["subsample"] = self._subsample
        self._sample = Sample(**kwargs)
        self._file_path = file_path
        return self

    def get_channels(self) -> list[str]:
        """Получение списка каналов.

        Returns:
            Список названий каналов (PnN идентификаторы)

        Подробное описание: Description/description_fcs_parser.md#get_channels
        """
        if self._sample is None:
            raise RuntimeError("No file loaded. Call load() first.")
        return list(self._sample.pnn_labels)

    def get_events(
        self,
        source: str = "raw",
        channels: list[str] | None = None,
    ) -> np.ndarray:
        """Получение матрицы событий.

        Args:
            source: Тип данных ('raw', 'comp', 'xform')
            channels: Список каналов (None = все)

        Returns:
            NumPy массив [n_events, n_channels]

        Подробное описание: Description/description_fcs_parser.md#get_events
        """
        if self._sample is None:
            raise RuntimeError("No file loaded. Call load() first.")

        events = self._sample.get_events(source=source)

        if channels is not None:
            all_channels = list(self._sample.pnn_labels)
            indices = [all_channels.index(ch) for ch in channels]
            events = events[:, indices]

        return events

    def get_metadata(self) -> FCSMetadata:
        """Получение метаданных файла.

        Returns:
            FCSMetadata dataclass

        Подробное описание: Description/description_fcs_parser.md#get_metadata
        """
        if self._sample is None:
            raise RuntimeError("No file loaded. Call load() first.")

        meta = self._sample.get_metadata()

        return FCSMetadata(
            filename=self._file_path.name,
            n_events=self._sample.event_count,
            n_channels=len(self._sample.pnn_labels),
            channels=list(self._sample.pnn_labels),
            cytometer=meta.get("cyt", meta.get("$CYT")),
            date=meta.get("date", meta.get("$DATE")),
            fcs_version=meta.get("fcsversion", meta.get("FCSversion")),
        )

    def to_dataframe(
        self,
        source: str = "raw",
        channels: list[str] | None = None,
    ) -> pd.DataFrame:
        """Конвертация в pandas DataFrame.

        Args:
            source: Тип данных ('raw', 'comp', 'xform')
            channels: Список каналов (None = все)

        Returns:
            DataFrame с колонками-каналами

        Подробное описание: Description/description_fcs_parser.md#to_dataframe
        """
        if self._sample is None:
            raise RuntimeError("No file loaded. Call load() first.")

        df = self._sample.as_dataframe(source=source)

        if channels is not None:
            df = df[channels]

        return df

    def get_channel_data(self, channel: str, source: str = "raw") -> np.ndarray:
        """Получение данных одного канала.

        Args:
            channel: Название канала
            source: Тип данных ('raw', 'comp', 'xform')

        Returns:
            1D NumPy массив значений канала

        Подробное описание: Description/description_fcs_parser.md#get_channel_data
        """
        if self._sample is None:
            raise RuntimeError("No file loaded. Call load() first.")
        return self._sample.get_channel_events(channel, source=source)

    def validate_required_channels(self, required: list[str]) -> bool:
        """Проверка наличия необходимых каналов.

        Args:
            required: Список обязательных каналов

        Returns:
            True если все каналы присутствуют

        Raises:
            ValueError: Если каналы отсутствуют (с указанием каких)

        Подробное описание: Description/description_fcs_parser.md#validate_required_channels
        """
        if not required:
            return True

        available = self.get_channels()
        missing = []

        for req in required:
            # Exact match or substring match (case-insensitive for substring)
            found = any(
                req == ch or req.lower() in ch.lower() for ch in available
            )
            if not found:
                missing.append(req)

        if missing:
            raise ValueError(f"Missing required channels: {', '.join(missing)}")

        return True


def load_fcs(file_path: str | Path, subsample: int | None = None) -> FCSLoader:
    """Удобная функция для загрузки .fcs файла.

    Args:
        file_path: Путь к файлу
        subsample: Максимум событий

    Returns:
        Инициализированный FCSLoader

    Подробное описание: Description/description_fcs_parser.md#load_fcs
    """
    return FCSLoader(subsample=subsample).load(file_path)
