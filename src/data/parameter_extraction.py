"""Извлечение параметров для математических моделей SDE/ABM.

Конвертирует результаты гейтирования в параметры симуляции:
- N0: начальная плотность клеток
- C0: концентрация цитокинов/факторов роста
- Уровень воспаления (M1/M2 баланс)

Подробное описание: Description/description_parameter_extraction.md
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.data.gating import GatingResults


@dataclass
class ModelParameters:
    """Параметры для SDE/ABM моделей.

    Подробное описание: Description/description_parameter_extraction.md#ModelParameters
    """

    # Клеточные параметры
    n0: float  # Начальная плотность клеток (клеток/мкл)
    stem_cell_fraction: float  # Доля CD34+ стволовых клеток
    macrophage_fraction: float  # Доля макрофагов
    apoptotic_fraction: float  # Доля апоптотических клеток

    # Параметры цитокинов
    c0: float  # Начальная концентрация факторов роста (нг/мл)
    inflammation_level: float  # Уровень воспаления (0-1)

    # Метаданные
    source_file: str | None = None
    total_events: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Конвертация в словарь.

        Подробное описание: Description/description_parameter_extraction.md#to_dict
        """
        return {
            "n0": self.n0,
            "stem_cell_fraction": self.stem_cell_fraction,
            "macrophage_fraction": self.macrophage_fraction,
            "apoptotic_fraction": self.apoptotic_fraction,
            "c0": self.c0,
            "inflammation_level": self.inflammation_level,
            "source_file": self.source_file,
            "total_events": self.total_events,
        }

    def validate(self) -> bool:
        """Валидация параметров на физическую осмысленность.

        Подробное описание: Description/description_parameter_extraction.md#validate
        """
        if self.n0 <= 0:
            raise ValueError("n0 must be positive")

        for name, value in [
            ("stem_cell_fraction", self.stem_cell_fraction),
            ("macrophage_fraction", self.macrophage_fraction),
            ("apoptotic_fraction", self.apoptotic_fraction),
        ]:
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")

        if not 0 <= self.inflammation_level <= 1:
            raise ValueError("inflammation_level must be between 0 and 1")

        if self.c0 < 0:
            raise ValueError("c0 must be non-negative")

        return True


@dataclass
class ExtractionConfig:
    """Конфигурация извлечения параметров.

    Подробное описание: Description/description_parameter_extraction.md#ExtractionConfig
    """

    # Масштабирование
    volume_ul: float = 1.0  # Объём образца в мкл
    dilution_factor: float = 1.0  # Фактор разведения

    # Референсные значения
    ref_cell_density: float = 5000.0  # Референсная плотность (клеток/мкл)
    ref_cytokine_conc: float = 10.0  # Референсная концентрация (нг/мл)

    # Коэффициенты для расчёта C0
    stem_cell_cytokine_factor: float = 2.0  # Вклад стволовых клеток
    macrophage_cytokine_factor: float = 1.5  # Вклад макрофагов


class ParameterExtractor:
    """Экстрактор параметров из flow cytometry данных.

    Преобразует результаты гейтирования в параметры
    для математических моделей регенерации.

    Подробное описание: Description/description_parameter_extraction.md#ParameterExtractor
    """

    def __init__(self, config: ExtractionConfig | None = None) -> None:
        """Инициализация экстрактора.

        Args:
            config: Конфигурация извлечения (по умолчанию стандартная)

        Подробное описание: Description/description_parameter_extraction.md#__init__
        """
        self._config = config if config else ExtractionConfig()

    def extract(
        self,
        gating_results: GatingResults,
        source_file: str | None = None,
    ) -> ModelParameters:
        """Полное извлечение параметров из результатов гейтирования.

        Args:
            gating_results: Результаты GatingStrategy.apply()
            source_file: Имя исходного файла

        Returns:
            ModelParameters для симуляции

        Подробное описание: Description/description_parameter_extraction.md#extract
        """
        fractions = self._calculate_cell_fractions(gating_results)

        params = ModelParameters(
            n0=self.extract_n0(gating_results),
            stem_cell_fraction=fractions["cd34_positive"],
            macrophage_fraction=fractions["macrophages"],
            apoptotic_fraction=fractions["apoptotic"],
            c0=self.extract_c0(gating_results),
            inflammation_level=self.extract_inflammation_level(gating_results),
            source_file=source_file,
            total_events=gating_results.total_events,
        )

        params.validate()
        return params

    def extract_n0(
        self,
        gating_results: GatingResults,
    ) -> float:
        """Извлечение N0 — начальной плотности клеток.

        Args:
            gating_results: Результаты гейтирования

        Returns:
            N0 в клетках/мкл

        Подробное описание: Description/description_parameter_extraction.md#extract_n0
        """
        n_live = gating_results.gates["live_cells"].n_events
        n0 = n_live / self._config.volume_ul * self._config.dilution_factor
        return float(n0)

    def extract_c0(
        self,
        gating_results: GatingResults,
    ) -> float:
        """Извлечение C0 — начальной концентрации факторов роста.

        Оценивается на основе состава клеточных популяций.

        Args:
            gating_results: Результаты гейтирования

        Returns:
            C0 в нг/мл

        Подробное описание: Description/description_parameter_extraction.md#extract_c0
        """
        f_stem = gating_results.gates["cd34_positive"].fraction
        f_macro = gating_results.gates["macrophages"].fraction

        c0 = self._config.ref_cytokine_conc * (
            self._config.stem_cell_cytokine_factor * f_stem
            + self._config.macrophage_cytokine_factor * f_macro
        )

        return float(np.clip(c0, 1.0, 100.0))

    def extract_inflammation_level(
        self,
        gating_results: GatingResults,
        data: pd.DataFrame | None = None,
    ) -> float:
        """Извлечение уровня воспаления.

        Основан на:
        - Соотношении макрофагов к живым клеткам
        - Уровне апоптоза
        - (опционально) Интенсивности маркеров

        Args:
            gating_results: Результаты гейтирования
            data: Исходные данные для расчёта интенсивностей

        Returns:
            inflammation_level (0-1, где 1 = сильное воспаление)

        Подробное описание: Description/description_parameter_extraction.md#extract_inflammation_level
        """
        f_macro = gating_results.gates["macrophages"].fraction
        f_apopt = gating_results.gates["apoptotic"].fraction

        # Reference values for "normal" tissue
        REF_MACRO = 0.03
        REF_APOPT = 0.02

        # Calculate inflammation score based on deviation from normal
        # Weighted combination: macrophages (60%) + apoptosis (40%)
        inflammation = (
            0.6 * min(f_macro / REF_MACRO, 3.0)
            + 0.4 * min(f_apopt / REF_APOPT, 3.0)
        ) / 3.0

        return float(np.clip(inflammation, 0.0, 1.0))

    def _calculate_cell_fractions(
        self,
        gating_results: GatingResults,
    ) -> dict[str, float]:
        """Расчёт долей клеточных популяций.

        Returns:
            Словарь с долями каждой популяции

        Подробное описание: Description/description_parameter_extraction.md#_calculate_cell_fractions
        """
        fractions = {}
        for name, gate in gating_results.gates.items():
            fractions[name] = gate.fraction
        return fractions

    def _normalize_to_reference(
        self,
        value: float,
        ref_value: float,
        scale: str = "linear",
    ) -> float:
        """Нормализация значения относительно референса.

        Args:
            value: Исходное значение
            ref_value: Референсное значение
            scale: Тип шкалы ('linear', 'log')

        Returns:
            Нормализованное значение

        Подробное описание: Description/description_parameter_extraction.md#_normalize_to_reference
        """
        if scale == "linear":
            return value / ref_value
        elif scale == "log":
            return np.log10(value) / np.log10(ref_value)
        else:
            raise ValueError(f"Unknown scale: {scale}")


def extract_model_parameters(
    gating_results: GatingResults,
    config: ExtractionConfig | None = None,
    source_file: str | None = None,
) -> ModelParameters:
    """Удобная функция для извлечения параметров.

    Args:
        gating_results: Результаты гейтирования
        config: Конфигурация (опционально)
        source_file: Имя исходного файла

    Returns:
        ModelParameters

    Подробное описание: Description/description_parameter_extraction.md#extract_model_parameters
    """
    extractor = ParameterExtractor(config=config)
    return extractor.extract(gating_results, source_file=source_file)
