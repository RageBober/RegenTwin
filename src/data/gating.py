"""Стратегии гейтирования для flow cytometry данных.

Реализует автоматическое выделение популяций:
- Debris (мусор)
- Live cells (живые клетки)
- CD34+ стволовые клетки
- Макрофаги (CD14+/CD68+)
- Апоптотические клетки (Annexin-V+)

Подробное описание: Description/description_gating.md
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Optional import for Otsu thresholding
try:
    from skimage.filters import threshold_otsu

    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


@dataclass
class GateResult:
    """Результат применения гейта.

    Подробное описание: Description/description_gating.md#GateResult
    """

    name: str
    mask: np.ndarray  # Boolean mask [n_events]
    n_events: int
    fraction: float  # % от родительской популяции
    parent: str | None = None
    statistics: dict[str, float] = field(default_factory=dict)


@dataclass
class GatingResults:
    """Результаты полного гейтирования.

    Подробное описание: Description/description_gating.md#GatingResults
    """

    total_events: int
    gates: dict[str, GateResult]  # name -> GateResult

    def get_population(self, name: str) -> np.ndarray:
        """Получить маску популяции по имени.

        Подробное описание: Description/description_gating.md#get_population
        """
        if name not in self.gates:
            raise KeyError(f"Population '{name}' not found")
        return self.gates[name].mask

    def get_statistics(self) -> dict[str, Any]:
        """Получить сводную статистику.

        Подробное описание: Description/description_gating.md#get_statistics
        """
        stats: dict[str, Any] = {"total_events": self.total_events}
        for name, gate in self.gates.items():
            stats[f"{name}_fraction"] = gate.fraction
        return stats


class GatingStrategy:
    """Автоматическая стратегия гейтирования для RegenTwin.

    Иерархия гейтов:
    1. All events
       └── Non-debris (FSC/SSC threshold)
           └── Singlets (FSC-A vs FSC-H)
               └── Live cells (low Annexin-V)
                   ├── CD34+ stem cells
                   └── Macrophages (CD14+/CD68+)

    Подробное описание: Description/description_gating.md#GatingStrategy
    """

    # Конфигурация по умолчанию для каналов
    DEFAULT_CHANNELS: dict[str, str] = {
        "fsc_area": "FSC-A",
        "fsc_height": "FSC-H",
        "ssc_area": "SSC-A",
        "cd34": "CD34-APC",
        "cd14": "CD14-PE",
        "cd68": "CD68-FITC",
        "annexin": "Annexin-V-Pacific Blue",
        "cd66b": "CD66b-PE-Cy7",  # Нейтрофилы
        "cd31": "CD31-BV421",  # Эндотелиальные клетки
    }

    def __init__(
        self,
        channel_mapping: dict[str, str] | None = None,
    ) -> None:
        """Инициализация стратегии гейтирования.

        Args:
            channel_mapping: Маппинг логических имён -> реальные названия каналов

        Подробное описание: Description/description_gating.md#__init__
        """
        self._channels = self.DEFAULT_CHANNELS.copy()
        if channel_mapping:
            self._channels.update(channel_mapping)

    def _find_channel(self, columns: list | pd.Index, pattern: str) -> str:
        """Find column matching pattern (exact or substring).

        Args:
            columns: Available column names
            pattern: Pattern to match

        Returns:
            Matching column name or pattern as-is if not found
        """
        for col in columns:
            if pattern == col or pattern in col:
                return col
        return pattern  # Return as-is if not found

    # Standard column order for ndarray input
    STANDARD_COLUMN_ORDER: list[str] = [
        "fsc_area",    # index 0
        "fsc_height",  # index 1
        "ssc_area",    # index 2
        "cd34",        # index 3
        "cd14",        # index 4
        "cd68",        # index 5
        "annexin",     # index 6
        "cd66b",       # index 7 — нейтрофилы
        "cd31",        # index 8 — эндотелиальные клетки
    ]

    def apply(self, data: pd.DataFrame | np.ndarray) -> GatingResults:
        """Применение полной стратегии гейтирования.

        Args:
            data: DataFrame или ndarray с данными flow cytometry.
                  Для ndarray используется стандартный порядок колонок:
                  [FSC-A, FSC-H, SSC-A, CD34, CD14, CD68, Annexin-V]

        Returns:
            GatingResults со всеми популяциями

        Подробное описание: Description/description_gating.md#apply
        """
        if isinstance(data, np.ndarray):
            # Use standard column order for ndarray
            fsc_a = data[:, 0]
            fsc_h = data[:, 1]
            ssc_a = data[:, 2]
            cd34 = data[:, 3]
            cd14 = data[:, 4]
            cd68 = data[:, 5]
            annexin = data[:, 6]
            n_total = data.shape[0]
        else:
            # Get channel data using mapping for DataFrame
            fsc_a = data[self._channels["fsc_area"]].values
            fsc_h = data[self._channels["fsc_height"]].values
            ssc_a = data[self._channels["ssc_area"]].values

            # Find matching channels for markers
            cd34_col = self._find_channel(data.columns, self._channels["cd34"])
            cd14_col = self._find_channel(data.columns, self._channels["cd14"])
            cd68_col = self._find_channel(data.columns, self._channels["cd68"])
            annexin_col = self._find_channel(data.columns, self._channels["annexin"])

            cd34 = data[cd34_col].values
            cd14 = data[cd14_col].values
            cd68 = data[cd68_col].values
            annexin = data[annexin_col].values
            n_total = len(data)

        # Hierarchical gating
        non_debris = self.debris_gate(fsc_a, ssc_a)
        singlets = self.singlets_gate(fsc_a, fsc_h) & non_debris
        live = self.live_cells_gate(annexin) & singlets
        cd34_pos = self.cd34_gate(cd34) & live
        macrophages = self.macrophage_gate(cd14, cd68) & live
        apoptotic = self.apoptotic_gate(annexin) & singlets

        gates = {
            "non_debris": GateResult(
                name="non_debris",
                mask=non_debris,
                n_events=int(non_debris.sum()),
                fraction=non_debris.sum() / n_total,
                parent=None,
            ),
            "singlets": GateResult(
                name="singlets",
                mask=singlets,
                n_events=int(singlets.sum()),
                fraction=singlets.sum() / n_total,
                parent="non_debris",
            ),
            "live_cells": GateResult(
                name="live_cells",
                mask=live,
                n_events=int(live.sum()),
                fraction=live.sum() / n_total,
                parent="singlets",
            ),
            "cd34_positive": GateResult(
                name="cd34_positive",
                mask=cd34_pos,
                n_events=int(cd34_pos.sum()),
                fraction=cd34_pos.sum() / n_total,
                parent="live_cells",
            ),
            "macrophages": GateResult(
                name="macrophages",
                mask=macrophages,
                n_events=int(macrophages.sum()),
                fraction=macrophages.sum() / n_total,
                parent="live_cells",
            ),
            "apoptotic": GateResult(
                name="apoptotic",
                mask=apoptotic,
                n_events=int(apoptotic.sum()),
                fraction=apoptotic.sum() / n_total,
                parent="singlets",
            ),
        }

        return GatingResults(total_events=n_total, gates=gates)

    def debris_gate(
        self,
        fsc: np.ndarray,
        ssc: np.ndarray,
        fsc_threshold: float | None = None,
        ssc_threshold: float | None = None,
    ) -> np.ndarray:
        """Гейт для удаления debris (мусора).

        Debris имеет низкие значения FSC и SSC.

        Args:
            fsc: Массив FSC-A значений
            ssc: Массив SSC-A значений
            fsc_threshold: Порог FSC (авто если None)
            ssc_threshold: Порог SSC (авто если None)

        Returns:
            Boolean маска (True = NOT debris = живые/целые клетки)

        Подробное описание: Description/description_gating.md#debris_gate
        """
        if fsc_threshold is None:
            fsc_threshold = np.percentile(fsc, 15)
        if ssc_threshold is None:
            ssc_threshold = np.percentile(ssc, 10)

        return (fsc > fsc_threshold) & (ssc > ssc_threshold)

    def singlets_gate(
        self,
        fsc_a: np.ndarray,
        fsc_h: np.ndarray,
        tolerance: float = 0.1,
    ) -> np.ndarray:
        """Гейт для выделения синглетов (одиночных клеток).

        Удаляет дублеты по соотношению FSC-A/FSC-H.

        Args:
            fsc_a: FSC-A (Area)
            fsc_h: FSC-H (Height)
            tolerance: Допустимое отклонение от диагонали

        Returns:
            Boolean маска (True = синглеты)

        Подробное описание: Description/description_gating.md#singlets_gate
        """
        ratio = fsc_a / (fsc_h + 1e-10)
        expected_ratio = np.median(ratio)
        return np.abs(ratio - expected_ratio) < (expected_ratio * tolerance)

    def live_cells_gate(
        self,
        annexin: np.ndarray,
        threshold: float | None = None,
    ) -> np.ndarray:
        """Гейт для выделения живых клеток.

        Живые клетки имеют низкий Annexin-V (не апоптотические).

        Args:
            annexin: Массив значений Annexin-V
            threshold: Порог Annexin-V (авто если None)

        Returns:
            Boolean маска (True = живые клетки)

        Подробное описание: Description/description_gating.md#live_cells_gate
        """
        if threshold is None:
            threshold = np.percentile(annexin, 85)
        return annexin < threshold

    def cd34_gate(
        self,
        cd34: np.ndarray,
        threshold: float | None = None,
        percentile: float = 90.0,
    ) -> np.ndarray:
        """Гейт для CD34+ стволовых клеток.

        Args:
            cd34: Массив значений CD34
            threshold: Порог позитивности (авто если None)
            percentile: Перцентиль для автопорога

        Returns:
            Boolean маска (True = CD34+)

        Подробное описание: Description/description_gating.md#cd34_gate
        """
        if threshold is None:
            threshold = np.percentile(cd34, percentile)
        return cd34 > threshold

    def macrophage_gate(
        self,
        cd14: np.ndarray,
        cd68: np.ndarray,
        cd14_threshold: float | None = None,
        cd68_threshold: float | None = None,
    ) -> np.ndarray:
        """Гейт для макрофагов (CD14+/CD68+ или CD14+CD68+).

        Args:
            cd14: Массив значений CD14
            cd68: Массив значений CD68
            cd14_threshold: Порог CD14 (авто если None)
            cd68_threshold: Порог CD68 (авто если None)

        Returns:
            Boolean маска (True = макрофаги)

        Подробное описание: Description/description_gating.md#macrophage_gate
        """
        if cd14_threshold is None:
            cd14_threshold = np.percentile(cd14, 95)
        if cd68_threshold is None:
            cd68_threshold = np.percentile(cd68, 95)
        # OR logic: CD14+ OR CD68+
        return (cd14 > cd14_threshold) | (cd68 > cd68_threshold)

    def apoptotic_gate(
        self,
        annexin: np.ndarray,
        threshold: float | None = None,
    ) -> np.ndarray:
        """Гейт для апоптотических клеток.

        Args:
            annexin: Массив значений Annexin-V
            threshold: Порог позитивности

        Returns:
            Boolean маска (True = апоптотические)

        Подробное описание: Description/description_gating.md#apoptotic_gate
        """
        if threshold is None:
            threshold = np.percentile(annexin, 95)
        return annexin > threshold

    def _auto_threshold(
        self,
        data: np.ndarray,
        method: str = "otsu",
    ) -> float:
        """Автоматическое определение порога.

        Args:
            data: Данные канала
            method: Метод ('otsu', 'percentile', 'gmm')

        Returns:
            Значение порога

        Подробное описание: Description/description_gating.md#_auto_threshold
        """
        if method == "otsu":
            if HAS_SKIMAGE:
                # Use valley-finding between histogram peaks for bimodal data
                from scipy import ndimage

                nbins = 256
                hist, bin_edges = np.histogram(data, bins=nbins)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                # Smooth histogram to find valleys
                smooth_hist = ndimage.gaussian_filter1d(hist.astype(float), sigma=3)

                # Find local minima (valleys) where derivative changes from - to +
                deriv = np.diff(smooth_hist)
                valleys = []
                for i in range(len(deriv) - 1):
                    if deriv[i] <= 0 and deriv[i + 1] > 0:
                        valleys.append(i + 1)

                # Find the deepest valley (lowest histogram value)
                if valleys:
                    min_idx = min(valleys, key=lambda v: smooth_hist[v])
                    return float(bin_centers[min_idx])

                # Fallback to standard Otsu if no valleys found
                return float(threshold_otsu(data))
            else:
                # Fallback: find threshold between two peaks using percentiles
                return float(np.percentile(data, 70))
        elif method == "percentile":
            return float(np.percentile(data, 95))
        else:
            raise ValueError(f"Unknown method: {method}")

    def _density_gate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        fraction: float = 0.85,
    ) -> np.ndarray:
        """Гейтирование на основе плотности (2D).

        Args:
            x: Данные по X
            y: Данные по Y
            fraction: Доля событий для захвата

        Returns:
            Boolean маска

        Подробное описание: Description/description_gating.md#_density_gate
        """
        from scipy.stats import gaussian_kde

        # Stack coordinates
        xy = np.vstack([x, y])

        # Calculate density
        kde = gaussian_kde(xy)
        density = kde(xy)

        # Find threshold that captures desired fraction
        threshold = np.percentile(density, (1 - fraction) * 100)

        return density >= threshold

    def neutrophil_gate(
        self,
        cd66b: np.ndarray,
        threshold: float | None = None,
        percentile: float = 95.0,
    ) -> np.ndarray:
        """Выделяет CD66b+ нейтрофилы из массива данных.

        Применяет пороговое значение к каналу CD66b (CEACAM8) для идентификации
        нейтрофилов. Возвращает boolean маску где True = CD66b+.
        В контексте заживления раны нейтрофилы рекрутируются по градиенту
        IL-8 в первые 24-48 часов (популяция Ne в модели).
        Ожидаемая доля: ~5-10% от живых клеток.

        Args:
            cd66b: Массив значений CD66b
            threshold: Порог позитивности (авто если None)
            percentile: Перцентиль для автопорога (по умолчанию 95-й)

        Returns:
            Boolean маска (True = CD66b+ нейтрофилы)

        Подробное описание: Description/Phase1/description_gating.md#neutrophil_gate
        """
        if len(cd66b) == 0:
            return np.array([], dtype=bool)
        if threshold is None:
            threshold = np.percentile(cd66b, percentile)
        return cd66b > threshold

    def endothelial_gate(
        self,
        cd31: np.ndarray,
        threshold: float | None = None,
        percentile: float = 95.0,
    ) -> np.ndarray:
        """Выделяет CD31+ эндотелиальные клетки из массива данных.

        Применяет пороговое значение к каналу CD31 (PECAM-1) для идентификации
        эндотелиальных клеток. Возвращает boolean маску где True = CD31+.
        Эндотелиальные клетки участвуют в ангиогенезе — критическом процессе
        фазы пролиферации (популяция E в модели).
        Ожидаемая доля: ~2-5% от живых клеток.

        Args:
            cd31: Массив значений CD31
            threshold: Порог позитивности (авто если None)
            percentile: Перцентиль для автопорога (по умолчанию 95-й)

        Returns:
            Boolean маска (True = CD31+ эндотелиальные клетки)

        Подробное описание: Description/Phase1/description_gating.md#endothelial_gate
        """
        if len(cd31) == 0:
            return np.array([], dtype=bool)
        if threshold is None:
            threshold = np.percentile(cd31, percentile)
        return cd31 > threshold

    def apply_extended(
        self,
        data: pd.DataFrame | np.ndarray,
    ) -> GatingResults:
        """Расширенное гейтирование с 9 каналами (включая CD66b и CD31).

        Расширяет стандартный apply() дополнительными гейтами для полной
        модели регенерации. Возвращает GatingResults с 8 популяциями:
        non_debris, singlets, live_cells, cd34_positive, macrophages,
        apoptotic, neutrophils (CD66b+), endothelial (CD31+).

        Иерархия гейтов:
        All events
          └── Non-debris (FSC/SSC threshold)
              └── Singlets (FSC-A vs FSC-H)
                  └── Live cells (low Annexin-V)
                      ├── CD34+ стволовые клетки
                      ├── CD66b+ нейтрофилы
                      ├── CD31+ эндотелиальные
                      └── Макрофаги (CD14+/CD68+)
                  └── Апоптотические (high Annexin-V)

        Args:
            data: DataFrame или ndarray с 9 каналами данных flow cytometry.
                  Для ndarray порядок: [FSC-A, FSC-H, SSC-A, CD34, CD14,
                  CD68, Annexin-V, CD66b, CD31]

        Returns:
            GatingResults с 8 популяциями (включая neutrophils и endothelial)

        Подробное описание: Description/Phase1/description_gating.md#apply_extended
        """
        if isinstance(data, np.ndarray):
            if data.shape[1] < 9:
                raise ValueError(
                    f"Expected at least 9 columns, got {data.shape[1]}"
                )
            fsc_a = data[:, 0]
            fsc_h = data[:, 1]
            ssc_a = data[:, 2]
            cd34 = data[:, 3]
            cd14 = data[:, 4]
            cd68 = data[:, 5]
            annexin = data[:, 6]
            cd66b = data[:, 7]
            cd31 = data[:, 8]
            n_total = data.shape[0]
        else:
            fsc_a = data[self._channels["fsc_area"]].values
            fsc_h = data[self._channels["fsc_height"]].values
            ssc_a = data[self._channels["ssc_area"]].values

            cd34_col = self._find_channel(data.columns, self._channels["cd34"])
            cd14_col = self._find_channel(data.columns, self._channels["cd14"])
            cd68_col = self._find_channel(data.columns, self._channels["cd68"])
            annexin_col = self._find_channel(
                data.columns, self._channels["annexin"]
            )
            cd66b_col = self._find_channel(
                data.columns, self._channels["cd66b"]
            )
            cd31_col = self._find_channel(
                data.columns, self._channels["cd31"]
            )

            cd34 = data[cd34_col].values
            cd14 = data[cd14_col].values
            cd68 = data[cd68_col].values
            annexin = data[annexin_col].values
            cd66b = data[cd66b_col].values
            cd31 = data[cd31_col].values
            n_total = len(data)

        # Иерархическое гейтирование
        non_debris = self.debris_gate(fsc_a, ssc_a)
        singlets = self.singlets_gate(fsc_a, fsc_h) & non_debris
        live = self.live_cells_gate(annexin) & singlets
        cd34_pos = self.cd34_gate(cd34) & live
        macrophages = self.macrophage_gate(cd14, cd68) & live
        apoptotic = self.apoptotic_gate(annexin) & singlets
        neutrophils = self.neutrophil_gate(cd66b) & live
        endothelial = self.endothelial_gate(cd31) & live

        gates = {
            "non_debris": GateResult(
                name="non_debris", mask=non_debris,
                n_events=int(non_debris.sum()),
                fraction=non_debris.sum() / n_total, parent=None,
            ),
            "singlets": GateResult(
                name="singlets", mask=singlets,
                n_events=int(singlets.sum()),
                fraction=singlets.sum() / n_total, parent="non_debris",
            ),
            "live_cells": GateResult(
                name="live_cells", mask=live,
                n_events=int(live.sum()),
                fraction=live.sum() / n_total, parent="singlets",
            ),
            "cd34_positive": GateResult(
                name="cd34_positive", mask=cd34_pos,
                n_events=int(cd34_pos.sum()),
                fraction=cd34_pos.sum() / n_total, parent="live_cells",
            ),
            "macrophages": GateResult(
                name="macrophages", mask=macrophages,
                n_events=int(macrophages.sum()),
                fraction=macrophages.sum() / n_total, parent="live_cells",
            ),
            "apoptotic": GateResult(
                name="apoptotic", mask=apoptotic,
                n_events=int(apoptotic.sum()),
                fraction=apoptotic.sum() / n_total, parent="singlets",
            ),
            "neutrophils": GateResult(
                name="neutrophils", mask=neutrophils,
                n_events=int(neutrophils.sum()),
                fraction=neutrophils.sum() / n_total, parent="live_cells",
            ),
            "endothelial": GateResult(
                name="endothelial", mask=endothelial,
                n_events=int(endothelial.sum()),
                fraction=endothelial.sum() / n_total, parent="live_cells",
            ),
        }

        return GatingResults(total_events=n_total, gates=gates)
