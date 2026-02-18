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
class ExtendedModelParameters:
    """Расширенные параметры для полной модели регенерации (20 переменных).

    Содержит начальные условия для всех переменных SDE системы:
    8 клеточных популяций, 7 цитокинов, 3 компонента ECM,
    2 вспомогательные переменные. Используется для запуска полной
    симуляции в отличие от базового ModelParameters (6 параметров).

    Подробное описание: Description/Phase1/description_parameter_extraction.md#ExtendedModelParameters
    """

    # === Клеточные популяции (начальные условия, клеток/мкл) ===
    P0: float  # Тромбоциты (platelets)
    Ne0: float  # Нейтрофилы (CD66b+)
    M1_0: float  # M1 макрофаги (провоспалительные)
    M2_0: float  # M2 макрофаги (репаративные)
    F0: float  # Фибробласты
    Mf0: float  # Миофибробласты (α-SMA+)
    E0: float  # Эндотелиальные клетки (CD31+)
    S0: float  # Стволовые/прогениторные клетки (CD34+)

    # === Цитокины (начальные концентрации, нг/мл) ===
    C_TNF: float  # TNF-α (провоспалительный)
    C_IL10: float  # IL-10 (противовоспалительный)
    C_PDGF: float  # PDGF (фактор роста тромбоцитов)
    C_VEGF: float  # VEGF (фактор роста эндотелия сосудов)
    C_TGFb: float  # TGF-β (трансформирующий фактор роста)
    C_MCP1: float  # MCP-1/CCL2 (хемоаттрактант моноцитов)
    C_IL8: float  # IL-8/CXCL8 (хемоаттрактант нейтрофилов)

    # === Внеклеточный матрикс (ECM) ===
    rho_collagen: float  # Плотность коллагена (безразмерная, 0-1)
    C_MMP: float  # Матриксные металлопротеиназы (нг/мл)
    rho_fibrin: float  # Плотность фибрина (безразмерная, 0-1)

    # === Вспомогательные переменные ===
    D: float  # Сигнал повреждения (DAMPs, безразмерный, >= 0)
    O2: float  # Уровень кислорода (нормализованный, 0-1)

    # === Метаданные ===
    source_file: str | None = None
    total_events: int = 0
    inflammation_level: float = 0.0  # Совместимость с ModelParameters

    def to_dict(self) -> dict[str, Any]:
        """Конвертирует все 20 переменных + метаданные в словарь.

        Возвращает dict с ключами для каждой переменной модели
        и метаданными. Используется для передачи в SDE/ABM симулятор.
        Содержит минимум 20 ключей переменных + ключи метаданных.

        Returns:
            Словарь со всеми параметрами

        Подробное описание: Description/Phase1/description_parameter_extraction.md#ExtendedModelParameters.to_dict
        """
        return {
            "P0": self.P0, "Ne0": self.Ne0,
            "M1_0": self.M1_0, "M2_0": self.M2_0,
            "F0": self.F0, "Mf0": self.Mf0,
            "E0": self.E0, "S0": self.S0,
            "C_TNF": self.C_TNF, "C_IL10": self.C_IL10,
            "C_PDGF": self.C_PDGF, "C_VEGF": self.C_VEGF,
            "C_TGFb": self.C_TGFb, "C_MCP1": self.C_MCP1,
            "C_IL8": self.C_IL8,
            "rho_collagen": self.rho_collagen,
            "C_MMP": self.C_MMP,
            "rho_fibrin": self.rho_fibrin,
            "D": self.D, "O2": self.O2,
            "source_file": self.source_file,
            "total_events": self.total_events,
            "inflammation_level": self.inflammation_level,
        }

    def validate(self) -> bool:
        """Проверяет все 20 переменных на физическую осмысленность.

        Правила валидации:
        - Все клеточные популяции >= 0
        - Все цитокины >= 0
        - rho_collagen в [0, 1]
        - rho_fibrin в [0, 1]
        - O2 в [0, 1]
        - D >= 0
        - C_MMP >= 0

        Returns:
            True если все проверки пройдены

        Raises:
            ValueError: С описанием нарушенного ограничения

        Подробное описание: Description/Phase1/description_parameter_extraction.md#ExtendedModelParameters.validate
        """
        # Клеточные популяции >= 0
        for name, value in [
            ("P0", self.P0), ("Ne0", self.Ne0),
            ("M1_0", self.M1_0), ("M2_0", self.M2_0),
            ("F0", self.F0), ("Mf0", self.Mf0),
            ("E0", self.E0), ("S0", self.S0),
        ]:
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")

        # Цитокины >= 0
        for name, value in [
            ("C_TNF", self.C_TNF), ("C_IL10", self.C_IL10),
            ("C_PDGF", self.C_PDGF), ("C_VEGF", self.C_VEGF),
            ("C_TGFb", self.C_TGFb), ("C_MCP1", self.C_MCP1),
            ("C_IL8", self.C_IL8), ("C_MMP", self.C_MMP),
        ]:
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")

        # Нормализованные переменные [0, 1]
        for name, value in [
            ("rho_collagen", self.rho_collagen),
            ("rho_fibrin", self.rho_fibrin),
            ("O2", self.O2),
        ]:
            if not (0 <= value <= 1):
                raise ValueError(f"{name} must be in [0, 1], got {value}")

        # D >= 0
        if self.D < 0:
            raise ValueError(f"D must be >= 0, got {self.D}")

        return True

    @classmethod
    def from_basic_parameters(
        cls,
        basic: "ModelParameters",
        total_cells: float | None = None,
    ) -> "ExtendedModelParameters":
        """Конвертирует базовые ModelParameters (6 полей) в расширенные (20).

        Использует эвристики для оценки недостающих параметров:
        - P0: из ref_platelet_density
        - Ne0: 0 (нет данных CD66b в базовом гейтинге)
        - M1_0/M2_0: из macrophage_fraction × n0, соотношение 70/30
        - F0: оценка из (1 - сумма известных фракций) × n0
        - Mf0: 0 (начальное состояние)
        - E0: 0 (нет данных CD31 в базовом гейтинге)
        - S0: stem_cell_fraction × n0
        - Цитокины: из inflammation_level и c0
        - ECM: начальные значения по умолчанию

        Args:
            basic: Базовые ModelParameters
            total_cells: Общее количество клеток (если None, используется n0)

        Returns:
            ExtendedModelParameters с оценёнными значениями

        Подробное описание: Description/Phase1/description_parameter_extraction.md#from_basic_parameters
        """
        config = ExtractionConfig()
        n0 = total_cells if total_cells is not None else basic.n0

        # Клеточные популяции
        P0 = config.ref_platelet_density  # 250000
        Ne0 = 0.0  # Нет данных CD66b в базовом гейтинге
        M_total = basic.macrophage_fraction * n0
        M1_0 = 0.7 * M_total
        M2_0 = 0.3 * M_total
        S0 = basic.stem_cell_fraction * n0
        E0 = 0.0  # Нет данных CD31 в базовом гейтинге
        Mf0 = 0.0  # Начальное состояние
        remaining = max(
            1.0 - basic.stem_cell_fraction
            - basic.macrophage_fraction
            - basic.apoptotic_fraction,
            0.0,
        )
        F0 = n0 * remaining * 0.1

        # Цитокины, масштабированные по inflammation_level
        infl = basic.inflammation_level
        C_TNF = config.ref_TNF * (1.0 + infl * 2.0)
        C_IL10 = config.ref_IL10 * (1.0 + (1.0 - infl))
        C_PDGF = config.ref_PDGF * (1.0 + infl * 0.5)
        C_VEGF = config.ref_VEGF * (1.0 + infl * 0.3)
        C_TGFb = config.ref_TGFb * (1.0 + infl)
        C_MCP1 = config.ref_MCP1 * (1.0 + infl * 1.5)
        C_IL8 = config.ref_IL8 * (1.0 + infl * 1.5)

        # ECM: начальные значения (ранняя фаза заживления)
        rho_collagen = 0.1
        C_MMP = 0.5
        rho_fibrin = 0.8

        # Вспомогательные
        D = 1.0  # Начальный сигнал повреждения
        O2 = 0.95  # Почти нормальный кислород

        return cls(
            P0=P0, Ne0=Ne0, M1_0=M1_0, M2_0=M2_0,
            F0=F0, Mf0=Mf0, E0=E0, S0=S0,
            C_TNF=C_TNF, C_IL10=C_IL10, C_PDGF=C_PDGF, C_VEGF=C_VEGF,
            C_TGFb=C_TGFb, C_MCP1=C_MCP1, C_IL8=C_IL8,
            rho_collagen=rho_collagen, C_MMP=C_MMP, rho_fibrin=rho_fibrin,
            D=D, O2=O2,
            source_file=basic.source_file,
            total_events=basic.total_events,
            inflammation_level=basic.inflammation_level,
        )

    def to_basic_parameters(self) -> "ModelParameters":
        """Обратная конвертация в базовые ModelParameters (6 полей).

        Агрегирует расширенные параметры обратно в базовый формат:
        - n0: сумма всех клеточных популяций
        - stem_cell_fraction: S0 / n0
        - macrophage_fraction: (M1_0 + M2_0) / n0
        - apoptotic_fraction: оценка из inflammation_level
        - c0: среднее цитокинов
        - inflammation_level: сохраняется

        Returns:
            ModelParameters для совместимости с существующим кодом

        Подробное описание: Description/Phase1/description_parameter_extraction.md#to_basic_parameters
        """
        # n0: сумма всех клеточных популяций
        n0 = (self.P0 + self.Ne0 + self.M1_0 + self.M2_0
              + self.F0 + self.Mf0 + self.E0 + self.S0)
        if n0 <= 0:
            n0 = 1.0

        stem_cell_fraction = float(np.clip(self.S0 / n0, 0, 1))
        macrophage_fraction = float(
            np.clip((self.M1_0 + self.M2_0) / n0, 0, 1)
        )
        apoptotic_fraction = float(
            np.clip(self.inflammation_level * 0.1, 0, 1)
        )

        cytokines = [
            self.C_TNF, self.C_IL10, self.C_PDGF, self.C_VEGF,
            self.C_TGFb, self.C_MCP1, self.C_IL8,
        ]
        c0 = float(np.clip(np.mean(cytokines), 1.0, 100.0))

        return ModelParameters(
            n0=float(n0),
            stem_cell_fraction=stem_cell_fraction,
            macrophage_fraction=macrophage_fraction,
            apoptotic_fraction=apoptotic_fraction,
            c0=c0,
            inflammation_level=self.inflammation_level,
            source_file=self.source_file,
            total_events=self.total_events,
        )

    def to_sde_state_vector(self) -> np.ndarray:
        """Конвертирует в вектор состояния для SDE решателя.

        Возвращает NumPy массив длиной 20 в фиксированном порядке:
        [P, Ne, M1, M2, F, Mf, E, S,
         C_TNF, C_IL10, C_PDGF, C_VEGF, C_TGFb, C_MCP1, C_IL8,
         rho_collagen, C_MMP, rho_fibrin, D, O2]

        Returns:
            np.ndarray shape=(20,) с float64

        Подробное описание: Description/Phase1/description_parameter_extraction.md#to_sde_state_vector
        """
        return np.array([
            self.P0, self.Ne0, self.M1_0, self.M2_0,
            self.F0, self.Mf0, self.E0, self.S0,
            self.C_TNF, self.C_IL10, self.C_PDGF, self.C_VEGF,
            self.C_TGFb, self.C_MCP1, self.C_IL8,
            self.rho_collagen, self.C_MMP, self.rho_fibrin,
            self.D, self.O2,
        ], dtype=np.float64)


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

    # === Расширенные референсные значения ===

    # Доли новых популяций
    ref_neutrophil_fraction: float = 0.05  # Референс нейтрофилов
    ref_endothelial_fraction: float = 0.03  # Референс эндотелиальных
    ref_m1_m2_ratio: float = 2.33  # Начальное M1/M2 = 70/30
    ref_platelet_density: float = 250000.0  # Тромбоциты (клеток/мкл)

    # Референсные концентрации цитокинов (нг/мл)
    ref_TNF: float = 0.1
    ref_IL10: float = 0.05
    ref_PDGF: float = 5.0
    ref_VEGF: float = 0.5
    ref_TGFb: float = 1.0
    ref_MCP1: float = 0.2
    ref_IL8: float = 0.1

    # Коэффициенты для новых популяций
    neutrophil_cytokine_factor: float = 1.0  # Вклад нейтрофилов в цитокины
    endothelial_cytokine_factor: float = 0.5  # Вклад эндотелиальных


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

    def extract_extended(
        self,
        gating_results: GatingResults,
        source_file: str | None = None,
    ) -> ExtendedModelParameters:
        """Извлекает все 20 переменных из расширенных результатов гейтирования.

        Требует GatingResults от apply_extended() (с гейтами neutrophils
        и endothelial). Использует клеточные доли для оценки начальных
        концентраций цитокинов и состояния ECM.

        Args:
            gating_results: Результаты GatingStrategy.apply_extended()
            source_file: Имя исходного файла

        Returns:
            ExtendedModelParameters с 20 переменными

        Подробное описание: Description/Phase1/description_parameter_extraction.md#extract_extended
        """
        if "neutrophils" not in gating_results.gates:
            raise KeyError("Gate 'neutrophils' not found in gating results")
        if "endothelial" not in gating_results.gates:
            raise KeyError("Gate 'endothelial' not found in gating results")

        n0 = self.extract_n0(gating_results)
        fractions = self._calculate_cell_fractions(gating_results)
        inflammation = self.extract_inflammation_level(gating_results)
        config = self._config

        # Клеточные популяции
        P0 = config.ref_platelet_density
        Ne0 = fractions.get("neutrophils", 0.0) * n0
        M_total = fractions.get("macrophages", 0.0) * n0
        M1_0 = 0.7 * M_total
        M2_0 = 0.3 * M_total
        S0 = fractions.get("cd34_positive", 0.0) * n0
        E0 = fractions.get("endothelial", 0.0) * n0
        Mf0 = 0.0
        remaining = max(
            1.0 - fractions.get("cd34_positive", 0)
            - fractions.get("macrophages", 0)
            - fractions.get("apoptotic", 0)
            - fractions.get("neutrophils", 0)
            - fractions.get("endothelial", 0),
            0.0,
        )
        F0 = n0 * remaining * 0.1

        # Цитокины и ECM
        cytokines = self.estimate_cytokine_profile(gating_results)
        ecm = self.estimate_ecm_state(gating_results)

        return ExtendedModelParameters(
            P0=P0, Ne0=Ne0, M1_0=M1_0, M2_0=M2_0,
            F0=F0, Mf0=Mf0, E0=E0, S0=S0,
            C_TNF=cytokines["TNF"], C_IL10=cytokines["IL10"],
            C_PDGF=cytokines["PDGF"], C_VEGF=cytokines["VEGF"],
            C_TGFb=cytokines["TGFb"], C_MCP1=cytokines["MCP1"],
            C_IL8=cytokines["IL8"],
            rho_collagen=ecm["rho_collagen"], C_MMP=ecm["C_MMP"],
            rho_fibrin=ecm["rho_fibrin"],
            D=1.0, O2=0.95,
            source_file=source_file,
            total_events=gating_results.total_events,
            inflammation_level=inflammation,
        )

    def extract_neutrophil_fraction(
        self,
        gating_results: GatingResults,
    ) -> float:
        """Извлекает долю CD66b+ нейтрофилов из результатов гейтирования.

        Ищет гейт "neutrophils" в GatingResults и возвращает его fraction.

        Args:
            gating_results: Результаты с гейтом "neutrophils"

        Returns:
            Доля нейтрофилов от общего числа событий (0-1)

        Raises:
            KeyError: Если гейт "neutrophils" отсутствует

        Подробное описание: Description/Phase1/description_parameter_extraction.md#extract_neutrophil_fraction
        """
        if "neutrophils" not in gating_results.gates:
            raise KeyError("Gate 'neutrophils' not found")
        return float(gating_results.gates["neutrophils"].fraction)

    def extract_endothelial_fraction(
        self,
        gating_results: GatingResults,
    ) -> float:
        """Извлекает долю CD31+ эндотелиальных клеток из гейтирования.

        Ищет гейт "endothelial" в GatingResults и возвращает его fraction.

        Args:
            gating_results: Результаты с гейтом "endothelial"

        Returns:
            Доля эндотелиальных от общего числа событий (0-1)

        Raises:
            KeyError: Если гейт "endothelial" отсутствует

        Подробное описание: Description/Phase1/description_parameter_extraction.md#extract_endothelial_fraction
        """
        if "endothelial" not in gating_results.gates:
            raise KeyError("Gate 'endothelial' not found")
        return float(gating_results.gates["endothelial"].fraction)

    def estimate_cytokine_profile(
        self,
        gating_results: GatingResults,
    ) -> dict[str, float]:
        """Оценивает начальные концентрации 7 цитокинов из клеточного состава.

        Использует состав популяций для косвенной оценки:
        - TNF-α: пропорционален M1 + Ne
        - IL-10: пропорционален M2
        - PDGF: пропорционален P + макрофаги
        - VEGF: пропорционален M2 + E
        - TGF-β: пропорционален P + M2
        - MCP-1: пропорционален D (сигнал повреждения)
        - IL-8: пропорционален D + M1

        Args:
            gating_results: Результаты гейтирования

        Returns:
            Словарь с ключами: TNF, IL10, PDGF, VEGF, TGFb, MCP1, IL8.
            Значения в нг/мл.

        Подробное описание: Description/Phase1/description_parameter_extraction.md#estimate_cytokine_profile
        """
        config = self._config
        fractions = self._calculate_cell_fractions(gating_results)
        infl = self.extract_inflammation_level(gating_results)

        f_macro = fractions.get("macrophages", 0.0)
        f_neutro = fractions.get("neutrophils", 0.0)
        f_endo = fractions.get("endothelial", 0.0)

        # TNF-α: пропорционален M1 + нейтрофилы
        TNF = config.ref_TNF * (1.0 + (f_macro + f_neutro) * 20.0)
        # IL-10: пропорционален M2 (противовоспалительный)
        IL10 = config.ref_IL10 * (1.0 + f_macro * 10.0 * (1.0 - infl))
        # PDGF: пропорционален тромбоцитам + макрофаги
        PDGF = config.ref_PDGF * (1.0 + f_macro * 5.0)
        # VEGF: пропорционален M2 + эндотелиальные
        VEGF = config.ref_VEGF * (1.0 + (f_macro + f_endo) * 10.0)
        # TGF-β: пропорционален тромбоцитам + M2
        TGFb = config.ref_TGFb * (1.0 + f_macro * 5.0)
        # MCP-1: пропорционален сигналу повреждения
        MCP1 = config.ref_MCP1 * (1.0 + infl * 5.0)
        # IL-8: пропорционален повреждению + M1
        IL8 = config.ref_IL8 * (1.0 + (f_macro + f_neutro) * 15.0)

        return {
            "TNF": float(max(TNF, 0)),
            "IL10": float(max(IL10, 0)),
            "PDGF": float(max(PDGF, 0)),
            "VEGF": float(max(VEGF, 0)),
            "TGFb": float(max(TGFb, 0)),
            "MCP1": float(max(MCP1, 0)),
            "IL8": float(max(IL8, 0)),
        }

    def estimate_ecm_state(
        self,
        gating_results: GatingResults,
    ) -> dict[str, float]:
        """Оценивает начальное состояние внеклеточного матрикса (ECM).

        Возвращает начальные значения для трёх ECM-переменных.
        По умолчанию: rho_collagen=0.1, C_MMP=0.5, rho_fibrin=0.8
        (ранняя фаза заживления — много фибрина, мало коллагена).

        Args:
            gating_results: Результаты гейтирования

        Returns:
            Словарь с ключами: rho_collagen, C_MMP, rho_fibrin

        Подробное описание: Description/Phase1/description_parameter_extraction.md#estimate_ecm_state
        """
        return {
            "rho_collagen": 0.1,
            "C_MMP": 0.5,
            "rho_fibrin": 0.8,
        }


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


def extract_extended_parameters(
    gating_results: GatingResults,
    config: ExtractionConfig | None = None,
    source_file: str | None = None,
) -> ExtendedModelParameters:
    """Удобная функция для извлечения расширенных параметров (20 переменных).

    Создаёт ParameterExtractor и вызывает extract_extended().
    Требует GatingResults от apply_extended() (с 8 популяциями).

    Args:
        gating_results: Результаты расширенного гейтирования
        config: Конфигурация (опционально)
        source_file: Имя исходного файла

    Returns:
        ExtendedModelParameters

    Подробное описание: Description/Phase1/description_parameter_extraction.md#extract_extended_parameters
    """
    extractor = ParameterExtractor(config=config)
    return extractor.extract_extended(gating_results, source_file=source_file)
