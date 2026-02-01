"""
Генераторы mock данных для тестирования flow cytometry модулей.

Используется для создания реалистичных FCS-подобных данных
с известными характеристиками популяций.
"""

import numpy as np
import pandas as pd
from typing import Optional


# =============================================================================
# Константы из документации
# =============================================================================

DEFAULT_CHANNELS = [
    "FSC-A",
    "FSC-H",
    "SSC-A",
    "CD34-APC",
    "CD14-PE",
    "CD68-FITC",
    "Annexin-V-Pacific Blue",
]

# Характеристики каналов для генерации данных
CHANNEL_CHARACTERISTICS = {
    "FSC-A": {"mean": 100000, "std": 20000, "range": (0, 262144)},
    "FSC-H": {"mean": 90000, "std": 18000, "range": (0, 262144)},
    "SSC-A": {"mean": 50000, "std": 15000, "range": (0, 262144)},
    "CD34-APC": {"negative_mean": 5000, "positive_mean": 150000, "positive_std": 30000},
    "CD14-PE": {"negative_mean": 8000, "positive_mean": 100000, "positive_std": 20000},
    "CD68-FITC": {"negative_mean": 3000, "positive_mean": 80000, "positive_std": 15000},
    "Annexin-V-Pacific Blue": {"live_mean": 2000, "apoptotic_mean": 120000, "apoptotic_std": 20000},
}

# Ожидаемые фракции для разных сценариев
SCENARIO_FRACTIONS = {
    "normal": {
        "debris": 0.20,
        "stem": 0.05,
        "macro": 0.03,
        "apopt": 0.02,
    },
    "inflamed": {
        "debris": 0.20,
        "stem": 0.03,
        "macro": 0.08,
        "apopt": 0.05,
    },
    "regenerating": {
        "debris": 0.20,
        "stem": 0.10,
        "macro": 0.02,
        "apopt": 0.01,
    },
}


# =============================================================================
# Генераторы данных
# =============================================================================

def generate_normal_fcs_data(
    n_events: int = 10000,
    seed: Optional[int] = 42,
    channels: Optional[list] = None,
) -> pd.DataFrame:
    """
    Генерирует mock FCS данные с нормальными распределениями популяций.

    Популяции:
    - Debris: ~20%
    - Live cells: ~70%
    - CD34+ стволовые: ~5%
    - Макрофаги: ~3%
    - Апоптотические: ~2%

    Parameters
    ----------
    n_events : int
        Количество событий для генерации
    seed : int, optional
        Seed для воспроизводимости
    channels : list, optional
        Список каналов. По умолчанию DEFAULT_CHANNELS

    Returns
    -------
    pd.DataFrame
        DataFrame с mock FCS данными
    """
    if channels is None:
        channels = DEFAULT_CHANNELS

    rng = np.random.default_rng(seed)
    fracs = SCENARIO_FRACTIONS["normal"]

    # Расчет количества событий для каждой популяции
    n_debris = int(n_events * fracs["debris"])
    n_cd34 = int(n_events * fracs["stem"])
    n_macro = int(n_events * fracs["macro"])
    n_apopt = int(n_events * fracs["apopt"])
    n_live_other = n_events - n_debris - n_cd34 - n_macro - n_apopt

    data = _generate_population_data(
        rng, n_debris, n_cd34, n_macro, n_apopt, n_live_other, channels
    )

    df = pd.DataFrame(data, columns=channels)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def generate_inflamed_fcs_data(
    n_events: int = 10000,
    seed: Optional[int] = 43,
    channels: Optional[list] = None,
) -> pd.DataFrame:
    """
    Генерирует mock FCS данные с повышенным воспалением.

    Характеристики:
    - Макрофаги: ~8% (повышено)
    - Апоптоз: ~5% (повышено)
    - Ожидаемый inflammation_level: ~0.7

    Parameters
    ----------
    n_events : int
        Количество событий
    seed : int, optional
        Seed для воспроизводимости
    channels : list, optional
        Список каналов

    Returns
    -------
    pd.DataFrame
        DataFrame с mock FCS данными
    """
    if channels is None:
        channels = DEFAULT_CHANNELS

    rng = np.random.default_rng(seed)
    fracs = SCENARIO_FRACTIONS["inflamed"]

    n_debris = int(n_events * fracs["debris"])
    n_cd34 = int(n_events * fracs["stem"])
    n_macro = int(n_events * fracs["macro"])
    n_apopt = int(n_events * fracs["apopt"])
    n_live_other = n_events - n_debris - n_cd34 - n_macro - n_apopt

    data = _generate_population_data(
        rng, n_debris, n_cd34, n_macro, n_apopt, n_live_other, channels
    )

    df = pd.DataFrame(data, columns=channels)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def generate_regenerating_fcs_data(
    n_events: int = 10000,
    seed: Optional[int] = 44,
    channels: Optional[list] = None,
) -> pd.DataFrame:
    """
    Генерирует mock FCS данные с высокой регенерацией.

    Характеристики:
    - CD34+ стволовые: ~10% (повышено)
    - Макрофаги: ~2% (низко)
    - Апоптоз: ~1% (низко)
    - Ожидаемый inflammation_level: ~0.2

    Parameters
    ----------
    n_events : int
        Количество событий
    seed : int, optional
        Seed для воспроизводимости
    channels : list, optional
        Список каналов

    Returns
    -------
    pd.DataFrame
        DataFrame с mock FCS данными
    """
    if channels is None:
        channels = DEFAULT_CHANNELS

    rng = np.random.default_rng(seed)
    fracs = SCENARIO_FRACTIONS["regenerating"]

    n_debris = int(n_events * fracs["debris"])
    n_cd34 = int(n_events * fracs["stem"])
    n_macro = int(n_events * fracs["macro"])
    n_apopt = int(n_events * fracs["apopt"])
    n_live_other = n_events - n_debris - n_cd34 - n_macro - n_apopt

    data = _generate_population_data(
        rng, n_debris, n_cd34, n_macro, n_apopt, n_live_other, channels
    )

    df = pd.DataFrame(data, columns=channels)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def _generate_population_data(
    rng: np.random.Generator,
    n_debris: int,
    n_cd34: int,
    n_macro: int,
    n_apopt: int,
    n_live_other: int,
    channels: list,
) -> dict:
    """
    Внутренняя функция для генерации данных всех популяций.

    Parameters
    ----------
    rng : np.random.Generator
        Генератор случайных чисел
    n_debris, n_cd34, n_macro, n_apopt, n_live_other : int
        Количества событий для каждой популяции
    channels : list
        Список каналов

    Returns
    -------
    dict
        Словарь с данными для каждого канала
    """
    data = {}

    # === FSC-A ===
    data["FSC-A"] = np.concatenate([
        rng.uniform(5000, 30000, n_debris),           # Debris - низкий
        rng.normal(100000, 20000, n_live_other),      # Обычные живые
        rng.normal(95000, 18000, n_cd34),             # CD34+
        rng.normal(105000, 22000, n_macro),           # Макрофаги
        rng.normal(70000, 25000, n_apopt),            # Апоптотические
    ])

    # === FSC-H (пропорционально FSC-A) ===
    n_singlets = n_live_other + n_cd34 + n_macro + n_apopt
    fsc_h_ratio = np.concatenate([
        rng.normal(0.9, 0.15, n_debris),              # Debris - вариабельно
        rng.normal(0.95, 0.03, n_singlets - n_apopt), # Синглеты
        rng.normal(0.95, 0.05, n_apopt),              # Апоптотические
    ])
    data["FSC-H"] = data["FSC-A"] * fsc_h_ratio

    # === SSC-A ===
    data["SSC-A"] = np.concatenate([
        rng.uniform(3000, 20000, n_debris),           # Debris - низкий
        rng.normal(50000, 15000, n_live_other),       # Обычные живые
        rng.normal(40000, 12000, n_cd34),             # CD34+ - низкий SSC
        rng.normal(70000, 20000, n_macro),            # Макрофаги - высокий SSC
        rng.normal(40000, 20000, n_apopt),            # Апоптотические
    ])

    # === CD34-APC ===
    data["CD34-APC"] = np.concatenate([
        rng.exponential(3000, n_debris),
        rng.exponential(5000, n_live_other),
        rng.normal(150000, 30000, n_cd34),            # CD34+ - высокий
        rng.exponential(5000, n_macro),
        rng.exponential(4000, n_apopt),
    ])

    # === CD14-PE ===
    data["CD14-PE"] = np.concatenate([
        rng.exponential(5000, n_debris),
        rng.exponential(8000, n_live_other),
        rng.exponential(7000, n_cd34),
        rng.normal(100000, 20000, n_macro),           # Макрофаги - высокий
        rng.exponential(6000, n_apopt),
    ])

    # === CD68-FITC ===
    data["CD68-FITC"] = np.concatenate([
        rng.exponential(2000, n_debris),
        rng.exponential(3000, n_live_other),
        rng.exponential(2500, n_cd34),
        rng.normal(80000, 15000, n_macro),            # Макрофаги - высокий
        rng.exponential(2500, n_apopt),
    ])

    # === Annexin-V-Pacific Blue ===
    data["Annexin-V-Pacific Blue"] = np.concatenate([
        rng.exponential(3000, n_debris),
        rng.exponential(2000, n_live_other + n_cd34 + n_macro),  # Живые - низкий
        rng.normal(120000, 20000, n_apopt),           # Апоптотические - высокий
    ])

    # Клипируем значения к допустимому диапазону FCS
    for key in data:
        data[key] = np.clip(data[key], 0, 262144)

    return data


def generate_bimodal_data(
    n_events: int = 1000,
    low_mean: float = 5000,
    low_std: float = 1000,
    high_mean: float = 100000,
    high_std: float = 20000,
    high_fraction: float = 0.1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Генерирует бимодальные данные для тестирования автопорогов.

    Parameters
    ----------
    n_events : int
        Общее количество событий
    low_mean, low_std : float
        Параметры низкой популяции
    high_mean, high_std : float
        Параметры высокой популяции
    high_fraction : float
        Доля высокой популяции
    seed : int, optional
        Seed для воспроизводимости

    Returns
    -------
    np.ndarray
        1D массив с бимодальными данными
    """
    rng = np.random.default_rng(seed)

    n_high = int(n_events * high_fraction)
    n_low = n_events - n_high

    low_data = rng.normal(low_mean, low_std, n_low)
    high_data = rng.normal(high_mean, high_std, n_high)

    data = np.concatenate([low_data, high_data])
    rng.shuffle(data)

    return np.clip(data, 0, 262144)


def generate_singlets_doublets_data(
    n_events: int = 1000,
    doublet_fraction: float = 0.1,
    seed: Optional[int] = None,
) -> tuple:
    """
    Генерирует данные FSC-A/FSC-H с синглетами и дублетами.

    Parameters
    ----------
    n_events : int
        Общее количество событий
    doublet_fraction : float
        Доля дублетов
    seed : int, optional
        Seed для воспроизводимости

    Returns
    -------
    tuple
        (fsc_a, fsc_h) - два numpy массива
    """
    rng = np.random.default_rng(seed)

    n_doublets = int(n_events * doublet_fraction)
    n_singlets = n_events - n_doublets

    # Синглеты: FSC-A ~ FSC-H
    fsc_h_singlets = rng.normal(80000, 15000, n_singlets)
    fsc_a_singlets = fsc_h_singlets * rng.normal(1.05, 0.03, n_singlets)

    # Дублеты: FSC-A >> FSC-H
    fsc_h_doublets = rng.normal(80000, 15000, n_doublets)
    fsc_a_doublets = fsc_h_doublets * rng.normal(1.8, 0.15, n_doublets)

    fsc_a = np.concatenate([fsc_a_singlets, fsc_a_doublets])
    fsc_h = np.concatenate([fsc_h_singlets, fsc_h_doublets])

    # Перемешиваем
    indices = rng.permutation(n_events)
    return fsc_a[indices], fsc_h[indices]
