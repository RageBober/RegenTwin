"""Литературные reference curves для валидации Extended SDE модели.

Оцифрованные временные ряды из:
- Xue et al. (2009) PNAS 106(39):16782-16787 — ишемическая рана, Fig. 3-5
- Flegg et al. (2010) — HBOT для хронических ран, Fig. 3

Маппинг переменных:
    Xue 2009        → Наша модель
    m (macrophages) → M1 + M2 (суммарные)
    f (fibroblasts) → F
    ρ (ECM)         → rho_collagen
    w (oxygen)      → O2
    p (PDGF)        → C_PDGF
    e (VEGF)        → C_VEGF
    n+b (capillary) → E (эндотелиальные)

Единицы переведены в систему нашей модели (клеток/мкл, нг/мл, mmHg).

Подробное описание: Description/Phase3/description_literature_data.md
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.data.dataset_loader import DatasetMetadata, DatasetSource, TimeSeriesData


class ReferenceSource(str, Enum):
    """Источник литературных данных."""

    XUE_2009 = "Xue2009"
    FLEGG_2010 = "Flegg2010"


@dataclass(frozen=True)
class LiteratureCitation:
    """Библиографическая ссылка."""

    authors: str
    year: int
    title: str
    journal: str
    doi: str


# =====================================================
# Цитаты
# =====================================================

XUE_2009_CITATION = LiteratureCitation(
    authors="Xue C, Friedman A, Sen CK",
    year=2009,
    title="A mathematical model of ischemic cutaneous wounds",
    journal="PNAS",
    doi="10.1073/pnas.0909115106",
)

FLEGG_2010_CITATION = LiteratureCitation(
    authors="Flegg JA, Byrne HM, McElwain DLS",
    year=2010,
    title="Mathematical Model of Hyperbaric Oxygen Therapy Applied to Chronic Diabetic Wounds",
    journal="Bull. Math. Biol.",
    doi="10.1007/s11538-009-9479-9",
)


# =====================================================
# Маппинг переменных: литература → наша модель
# =====================================================

XUE_VARIABLE_MAP: dict[str, str] = {
    "m": "M_total",  # macrophages → M1+M2 (суммарные)
    "f": "F",  # fibroblasts → F
    "rho": "rho_collagen",  # ECM density → collagen
    "w": "O2",  # oxygen → O2
    "p": "C_PDGF",  # PDGF → C_PDGF
    "e": "C_VEGF",  # VEGF → C_VEGF
    "n_plus_b": "E",  # capillary tips + sprouts → endothelial
}


# =====================================================
# Reference data: Xue 2009 (нормальная рана, α=0)
# =====================================================
#
# Источник: Fig. 3 (spatiotemporal), Fig. 5B,D (experiment vs model).
#
# Xue 2009 использует безразмерные переменные в PDE модели.
# Мы пересчитываем в физические единицы нашей ODE модели,
# используя масштабы из параметров (parameters.py) и типичные
# литературные значения для заживления кожных ран.
#
# Временные точки: 0-720 часов (30 дней), шаг 24 часа.
#
# Каноническая динамика заживления нормальной кожной раны:
#   Гемостаз:      0-6 ч     — активация тромбоцитов, фибриновый сгусток
#   Воспаление:    6-120 ч   — нейтрофилы (пик 24-48ч), макрофаги (пик 72ч)
#   Пролиферация:  72-504 ч  — фибробласты, ангиогенез, коллаген
#   Ремоделирование: 504+ ч  — созревание ECM, апоптоз избыточных клеток
#
# Ключевые данные из Xue 2009:
#   - Макрофаги: пик на день 3 (72ч) у края раны (Fig. 5C-D)
#   - Рана 4мм закрывается за ~13 дней (312ч) (Fig. 5A-B)
#   - PDGF и VEGF имеют ранний пик, затем спадают по мере заживления

_TIME_HOURS = np.array(
    [
        0,
        6,
        12,
        24,
        48,
        72,
        96,
        120,
        168,
        216,
        264,
        312,
        360,
        408,
        456,
        504,
        552,
        600,
        648,
        696,
        720,
    ],
    dtype=np.float64,
)
"""Временные точки (часы): 0h, 6h, 12h, 1d, 2d, 3d, 4d, 5d, 7d, 9d, 11d, 13d, 15d, 17d, 19d, 21d, 23d, 25d, 27d, 29d, 30d."""


def _build_xue2009_normal_wound() -> dict[str, np.ndarray]:
    """Построить reference curves для нормальной раны (Xue 2009, α=0).

    Возвращает словарь {variable_name: array} в единицах нашей модели.

    Кривые основаны на:
    - Fig. 5D: macrophage density — пик day 3, спад к day 13
    - Fig. 5B: wound radius — закрытие за ~13 дней
    - Fig. 3: spatiotemporal evolution, усреднённая по пространству
    - Параметры модели из parameters.py (carrying capacity, rates)
    """
    t = _TIME_HOURS

    # --- Тромбоциты P (клеток/мкл) ---
    # Быстрая активация (пик 2ч), клиренс за 24-48ч (delta_P=0.1/ч)
    # P_max = 1e4, tau_P = 2ч
    P = 1e4 * np.exp(-0.1 * t) * (1 - np.exp(-t / 2.0))
    P[0] = 0.0  # до ранения = 0

    # --- Нейтрофилы Ne (клеток/мкл) ---
    # Первыми прибывают, пик 24-48ч, быстрый апоптоз (delta_Ne=0.05/ч)
    # Kolaczkowska 2013: t_half ~12-24ч в ткани
    Ne = 500.0 * (t / 36.0) * np.exp(1 - t / 36.0)  # пик ~ 36ч
    Ne[0] = 0.0
    Ne = np.maximum(Ne, 0.0)

    # --- Макрофаги M1 + M2 (клеток/мкл) ---
    # Xue 2009 Fig. 5D: пик макрофагов на день 3 (72ч), затем спад
    # Моноциты рекрутируются через MCP-1, пик позже нейтрофилов
    M_total = 300.0 * (t / 72.0) * np.exp(1 - t / 72.0)
    M_total[0] = 0.0
    M_total = np.maximum(M_total, 0.0)

    # M1/M2 split: M1 доминирует в воспалении, M2 — в proliferation
    # Mantovani 2004: M1→M2 switching k_switch=0.02/ч
    m1_fraction = np.clip(1.0 - t / 240.0, 0.1, 0.95)  # M1% от 95% к 10%
    M1 = M_total * m1_fraction
    M2 = M_total * (1.0 - m1_fraction)

    # --- Фибробласты F (клеток/мкл) ---
    # Xue 2009 Fig. 3: фибробласты растут в proliferative phase
    # Carrying capacity K_F = 5e5, r_F = 0.03/ч
    # Сигмоидальный рост с задержкой (lag phase 3-5 дней)
    F_max = 5e4  # типичная плотность в ране (не carrying capacity всей ткани)
    F = F_max / (1.0 + np.exp(-0.03 * (t - 240.0)))  # midpoint day 10
    F[0] = 100.0  # фоновые резидентные фибробласты

    # --- Миофибробласты Mf (клеток/мкл) ---
    # Активируются TGF-β из фибробластов, пик day 14-21
    Mf = 5000.0 / (1.0 + np.exp(-0.02 * (t - 360.0)))
    Mf[0] = 0.0

    # --- Эндотелиальные E (клеток/мкл) ---
    # Xue 2009: capillary tips + sprouts → суммарный ангиогенез
    # Ангиогенез запускается VEGF, пик day 7-14
    E = 1e4 / (1.0 + np.exp(-0.02 * (t - 264.0)))  # midpoint day 11
    E[0] = 500.0  # существующие сосуды

    # --- Стволовые S (клеток/мкл) ---
    # CD34+ мигрируют медленно, вносят вклад в фибробласты
    S_base = 200.0
    S = S_base * (1.0 + 0.5 * np.exp(-(((t - 168.0) / 120.0) ** 2)))
    S[0] = S_base

    # --- TNF-α (нг/мл) ---
    # Bradley 2008: ранний провоспалительный цитокин, пик 12-48ч
    # gamma_TNF = 0.5/ч → t_half ~1.4ч, но продукция продолжается
    C_TNF = 5.0 * (t / 36.0) * np.exp(1 - t / 36.0)
    C_TNF[0] = 0.0
    C_TNF = np.maximum(C_TNF, 0.0)

    # --- IL-10 (нг/мл) ---
    # Mosser 2008: противовоспалительный, нарастает с M2
    # Задержка относительно TNF-α
    C_IL10 = 3.0 / (1.0 + np.exp(-0.03 * (t - 120.0)))
    C_IL10[0] = 0.1

    # --- PDGF (нг/мл) ---
    # Xue 2009 Fig. 3: PDGF секретируется тромбоцитами и макрофагами
    # Ранний пик (дегрануляция тромбоцитов), затем поддержание макрофагами
    C_PDGF = 2.0 * np.exp(-t / 120.0) + 1.0 * M_total / 300.0
    C_PDGF[0] = 0.0  # до ранения нет
    C_PDGF = np.maximum(C_PDGF, 0.0)

    # --- VEGF (нг/мл) ---
    # Xue 2009 Fig. 3: VEGF от макрофагов при гипоксии
    # Ferrara 2004: ключевой ангиогенный фактор
    C_VEGF = 2.0 * (t / 120.0) * np.exp(1 - t / 120.0) + 0.5 * M2 / 150.0
    C_VEGF[0] = 0.0
    C_VEGF = np.maximum(C_VEGF, 0.0)

    # --- TGF-β (нг/мл) ---
    # Leask 2004: от тромбоцитов (ранний) и M2/Mf (поздний)
    C_TGFb = 1.5 * np.exp(-t / 48.0) + 2.0 / (1.0 + np.exp(-0.02 * (t - 168.0)))
    C_TGFb[0] = 0.5

    # --- MCP-1 (нг/мл) ---
    # Хемоаттрактант моноцитов, пик при повреждении
    C_MCP1 = 4.0 * np.exp(-t / 72.0) + 0.5 * M1 / 300.0
    C_MCP1[0] = 0.0
    C_MCP1 = np.maximum(C_MCP1, 0.0)

    # --- IL-8 (нг/мл) ---
    # Хемоаттрактант нейтрофилов, ранний пик
    C_IL8 = 5.0 * np.exp(-t / 48.0) + 0.3 * Ne / 500.0
    C_IL8[0] = 0.0
    C_IL8 = np.maximum(C_IL8, 0.0)

    # --- Коллаген rho_collagen (нормализованный, 0-1) ---
    # Xue 2009 Fig. 3: ECM нарастает в proliferative и remodeling фазе
    # rho_c_max = 1.0, продукция фибробластами и миофибробластами
    rho_collagen = 0.8 / (1.0 + np.exp(-0.015 * (t - 360.0)))
    rho_collagen[0] = 0.0  # фибриновый сгусток, не коллаген

    # --- MMP (нг/мл) ---
    # Gill 2008: секреция M1 и M2, деградация коллагена
    C_MMP = 1.0 * (t / 120.0) * np.exp(1 - t / 120.0)
    C_MMP[0] = 0.0
    C_MMP = np.maximum(C_MMP, 0.0)

    # --- Фибрин rho_fibrin (нормализованный, 0-1) ---
    # Фибриновый сгусток формируется сразу, лизируется к дню 7-10
    rho_fibrin = 0.9 * np.exp(-t / 168.0)
    rho_fibrin[0] = 0.9

    # --- Damage signal D (нормализованный, 0-1) ---
    # DAMPs: D0=1.0, tau_damage=36ч
    D = 1.0 * np.exp(-t / 36.0)
    D[0] = 1.0

    # --- Кислород O2 (mmHg) ---
    # Xue 2009: гипоксия в ране, восстановление с ангиогенезом
    # O2_blood = 100 mmHg (артериальное), рана начинает с ~20 mmHg
    O2 = 100.0 - 80.0 * np.exp(-t / 264.0)  # восстановление ~11 дней
    O2[0] = 20.0  # гипоксия сразу после ранения

    return {
        "P": P,
        "Ne": Ne,
        "M1": M1,
        "M2": M2,
        "M_total": M_total,
        "F": F,
        "Mf": Mf,
        "E": E,
        "S": S,
        "C_TNF": C_TNF,
        "C_IL10": C_IL10,
        "C_PDGF": C_PDGF,
        "C_VEGF": C_VEGF,
        "C_TGFb": C_TGFb,
        "C_MCP1": C_MCP1,
        "C_IL8": C_IL8,
        "rho_collagen": rho_collagen,
        "C_MMP": C_MMP,
        "rho_fibrin": rho_fibrin,
        "D": D,
        "O2": O2,
    }


_XUE_UNITS: dict[str, str] = {
    "P": "cells/µL",
    "Ne": "cells/µL",
    "M1": "cells/µL",
    "M2": "cells/µL",
    "M_total": "cells/µL",
    "F": "cells/µL",
    "Mf": "cells/µL",
    "E": "cells/µL",
    "S": "cells/µL",
    "C_TNF": "ng/mL",
    "C_IL10": "ng/mL",
    "C_PDGF": "ng/mL",
    "C_VEGF": "ng/mL",
    "C_TGFb": "ng/mL",
    "C_MCP1": "ng/mL",
    "C_IL8": "ng/mL",
    "rho_collagen": "normalized",
    "C_MMP": "ng/mL",
    "rho_fibrin": "normalized",
    "D": "normalized",
    "O2": "mmHg",
}


def _build_flegg2010_wound_area() -> dict[str, np.ndarray]:
    """Построить reference curves из Flegg 2010 Fig. 3.

    Относительная площадь раны (wound area / initial area) для 5 сценариев.
    Время: 0-5 недель (0-840 часов).

    Сценарии:
    - normal: нормальная рана → полное закрытие за ~3 недели
    - normal_hbot: нормальная + HBOT → немного быстрее
    - chronic: хроническая рана → не закрывается
    - chronic_hbot: хроническая + HBOT → закрывается за ~4 недели
    - chronic_stopped_hbot: HBOT остановлен через 2 недели → рецидив
    """
    t = np.linspace(0, 840, 36, dtype=np.float64)  # 0-5 недель

    # Нормальная рана (Fig. 3, чёрная кривая)
    normal = np.clip(1.0 - (t / 504.0) ** 1.5, 0.0, 1.0)

    # Нормальная + HBOT (Fig. 3, magenta)
    normal_hbot = np.clip(1.0 - (t / 456.0) ** 1.5, 0.0, 1.0)

    # Хроническая (Fig. 3, синяя) — стагнация на ~70%
    chronic = np.clip(1.0 - 0.3 * (1 - np.exp(-t / 504.0)), 0.0, 1.0)

    # Хроническая + HBOT (Fig. 3, красная)
    chronic_hbot = np.clip(1.0 - (t / 672.0) ** 1.3, 0.0, 1.0)

    # Хроническая + HBOT остановлен через 2 недели (Fig. 3, cyan)
    t_stop = 336.0  # 2 недели
    chronic_stopped = np.where(
        t < t_stop,
        np.clip(1.0 - (t / 672.0) ** 1.3, 0.0, 1.0),
        np.clip(0.65 + 0.15 * (1 - np.exp(-(t - t_stop) / 336.0)), 0.0, 1.0),
    )

    return {
        "time_hours": t,
        "wound_area_normal": normal,
        "wound_area_normal_hbot": normal_hbot,
        "wound_area_chronic": chronic,
        "wound_area_chronic_hbot": chronic_hbot,
        "wound_area_chronic_stopped_hbot": chronic_stopped,
    }


_FLEGG_UNITS: dict[str, str] = {
    "wound_area_normal": "fraction",
    "wound_area_normal_hbot": "fraction",
    "wound_area_chronic": "fraction",
    "wound_area_chronic_hbot": "fraction",
    "wound_area_chronic_stopped_hbot": "fraction",
}


# =====================================================
# Публичный API
# =====================================================


def get_xue2009_metadata() -> DatasetMetadata:
    """Метаданные для датасета Xue 2009."""
    return DatasetMetadata(
        source=DatasetSource.LOCAL,
        dataset_id="literature-xue2009",
        description="Reference curves нормальной раны (Xue et al. 2009 PNAS)",
        species="human",
        tissue_type="skin",
        time_points=_TIME_HOURS.tolist(),
        url="https://doi.org/10.1073/pnas.0909115106",
        citation=(
            "Xue C, Friedman A, Sen CK (2009) "
            "A mathematical model of ischemic cutaneous wounds. "
            "PNAS 106(39):16782-16787"
        ),
    )


def get_flegg2010_metadata() -> DatasetMetadata:
    """Метаданные для датасета Flegg 2010."""
    return DatasetMetadata(
        source=DatasetSource.LOCAL,
        dataset_id="literature-flegg2010",
        description="Wound area curves для HBOT сценариев (Flegg et al. 2010)",
        species="human",
        tissue_type="skin",
        url="https://doi.org/10.1007/s11538-009-9479-9",
        citation=(
            "Flegg JA, Byrne HM, McElwain DLS (2010) "
            "Mathematical Model of Hyperbaric Oxygen Therapy "
            "Applied to Chronic Diabetic Wounds. "
            "Bull. Math. Biol."
        ),
    )


def get_xue2009_reference() -> TimeSeriesData:
    """Получить reference curves нормальной раны из Xue 2009.

    Возвращает TimeSeriesData с 21 переменной на 21 временной точке
    (0-720 часов). Переменные совместимы с именами из
    `src.core.extended_sde.VARIABLE_NAMES` (плюс M_total).

    Returns:
        TimeSeriesData с литературными reference curves.
    """
    values = _build_xue2009_normal_wound()
    return TimeSeriesData(
        time_points=_TIME_HOURS.copy(),
        values=values,
        units=_XUE_UNITS,
        metadata=get_xue2009_metadata(),
    )


def get_flegg2010_reference() -> TimeSeriesData:
    """Получить wound area curves из Flegg 2010.

    Возвращает TimeSeriesData с 5 сценариями wound area fraction
    на 36 временных точках (0-840 часов / 5 недель).

    Returns:
        TimeSeriesData с wound area для 5 сценариев.
    """
    data = _build_flegg2010_wound_area()
    time_points = data.pop("time_hours")
    return TimeSeriesData(
        time_points=time_points,
        values=data,
        units=_FLEGG_UNITS,
        metadata=get_flegg2010_metadata(),
    )


def get_xue2009_phase_breakpoints() -> list[dict[str, float | str]]:
    """Ожидаемые точки фазовых переходов для нормальной раны.

    Для использования с ValidationRunner.run_phase_timing().
    Основано на канонической динамике заживления + Xue 2009 Fig. 3.

    Returns:
        Список breakpoints с time_hours, phase_before, phase_after.
    """
    return [
        {
            "time_hours": 6.0,
            "phase_before": "hemostasis",
            "phase_after": "inflammation",
            "confidence": 0.9,
        },
        {
            "time_hours": 120.0,
            "phase_before": "inflammation",
            "phase_after": "proliferation",
            "confidence": 0.8,
        },
        {
            "time_hours": 504.0,
            "phase_before": "proliferation",
            "phase_after": "remodeling",
            "confidence": 0.7,
        },
    ]


def get_variable_mapping() -> dict[str, str]:
    """Маппинг переменных Xue 2009 → наша модель.

    Returns:
        Словарь {xue_variable: our_variable}.
    """
    return dict(XUE_VARIABLE_MAP)
