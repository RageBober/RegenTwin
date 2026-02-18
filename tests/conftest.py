"""
Общие фикстуры для тестирования модулей flow cytometry.

Содержит:
- Константы каналов и ожидаемых фракций
- Фикстуры для mock FCS данных
- Фикстуры для mock FlowKit объектов
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock


# =============================================================================
# Константы
# =============================================================================

CHANNELS = [
    "FSC-A",
    "FSC-H",
    "SSC-A",
    "CD34-APC",
    "CD14-PE",
    "CD68-FITC",
    "Annexin-V-Pacific Blue",
]

# Ожидаемые фракции из документации description_gating.md
EXPECTED_FRACTIONS = {
    "debris": 0.20,
    "live_cells": 0.70,
    "cd34_positive": 0.05,
    "macrophages": 0.03,
    "apoptotic": 0.02,
}

# Расширенные каналы (9 каналов: 7 базовых + CD66b + CD31)
EXTENDED_CHANNELS = [
    "FSC-A",
    "FSC-H",
    "SSC-A",
    "CD34-APC",
    "CD14-PE",
    "CD68-FITC",
    "Annexin-V-Pacific Blue",
    "CD66b-PE-Cy7",
    "CD31-BV421",
]

# Ожидаемые фракции расширенного гейтирования (8 популяций)
EXTENDED_EXPECTED_FRACTIONS = {
    **EXPECTED_FRACTIONS,
    "neutrophils": 0.05,
    "endothelial": 0.03,
}

# Ожидаемые диапазоны параметров модели из description_parameter_extraction.md
EXPECTED_PARAMETER_RANGES = {
    "n0": {"min": 1000, "max": 50000},
    "c0": {"min": 1, "max": 100},
    "inflammation_level": {"min": 0, "max": 1},
    "stem_cell_fraction": {"min": 0.01, "max": 0.15},
    "macrophage_fraction": {"min": 0.01, "max": 0.10},
    "apoptotic_fraction": {"min": 0.01, "max": 0.10},
}


# =============================================================================
# Базовые фикстуры
# =============================================================================

@pytest.fixture
def sample_channels():
    """Список каналов для тестирования."""
    return CHANNELS.copy()


@pytest.fixture
def expected_fractions():
    """Ожидаемые фракции популяций."""
    return EXPECTED_FRACTIONS.copy()


@pytest.fixture
def expected_parameter_ranges():
    """Ожидаемые диапазоны параметров модели."""
    return EXPECTED_PARAMETER_RANGES.copy()


# =============================================================================
# Фикстуры для mock FCS данных
# =============================================================================

@pytest.fixture
def mock_fcs_data_normal(sample_channels):
    """
    Генерирует нормальный mock DataFrame с реалистичными распределениями.

    Популяции (из description_gating.md):
    - Debris: ~20% (низкий FSC/SSC)
    - Non-debris: ~80%
    - Live cells: ~70% от всех (низкий Annexin-V)
    - CD34+: ~5% (высокий CD34)
    - Macrophages: ~3% (высокий CD14/CD68)
    - Apoptotic: ~2% (высокий Annexin-V)
    """
    rng = np.random.default_rng(42)
    n_events = 10000

    # Распределение событий по популяциям
    n_debris = 2000      # 20%
    n_live = 7000        # 70%
    n_apoptotic = 200    # 2%
    n_other = 800        # Остаток (singlets, но не live и не apoptotic)

    # Из живых клеток:
    n_cd34 = 500         # 5% от всех = ~7% от живых
    n_macro = 300        # 3% от всех = ~4% от живых
    n_regular_live = n_live - n_cd34 - n_macro  # Обычные живые

    data = {}

    # === FSC-A ===
    fsc_debris = rng.uniform(5000, 30000, n_debris)
    fsc_live = rng.normal(100000, 20000, n_live)
    fsc_apopt = rng.normal(70000, 25000, n_apoptotic)
    fsc_other = rng.normal(90000, 25000, n_other)
    data["FSC-A"] = np.concatenate([fsc_debris, fsc_live, fsc_apopt, fsc_other])

    # === FSC-H (пропорционально FSC-A для синглетов) ===
    fsc_h_debris = fsc_debris * rng.normal(0.9, 0.1, n_debris)
    fsc_h_live = fsc_live * rng.normal(0.95, 0.03, n_live)  # Синглеты
    fsc_h_apopt = fsc_apopt * rng.normal(0.95, 0.05, n_apoptotic)
    fsc_h_other = fsc_other * rng.normal(1.5, 0.2, n_other)  # Дублеты
    data["FSC-H"] = np.concatenate([fsc_h_debris, fsc_h_live, fsc_h_apopt, fsc_h_other])

    # === SSC-A ===
    ssc_debris = rng.uniform(3000, 20000, n_debris)
    ssc_live = rng.normal(50000, 15000, n_live)
    ssc_apopt = rng.normal(40000, 20000, n_apoptotic)
    ssc_other = rng.normal(55000, 18000, n_other)
    data["SSC-A"] = np.concatenate([ssc_debris, ssc_live, ssc_apopt, ssc_other])

    # === CD34-APC (стволовые клетки) ===
    cd34_debris = rng.exponential(3000, n_debris)
    cd34_negative = rng.exponential(5000, n_regular_live + n_macro)
    cd34_positive = rng.normal(150000, 30000, n_cd34)
    cd34_apopt = rng.exponential(4000, n_apoptotic)
    cd34_other = rng.exponential(4500, n_other)
    data["CD34-APC"] = np.concatenate([
        cd34_debris,
        cd34_negative[:n_regular_live],
        cd34_positive,
        cd34_negative[n_regular_live:],  # macro
        cd34_apopt,
        cd34_other
    ])

    # === CD14-PE (макрофаги) ===
    cd14_debris = rng.exponential(5000, n_debris)
    cd14_negative = rng.exponential(8000, n_regular_live + n_cd34)
    cd14_positive = rng.normal(100000, 20000, n_macro)
    cd14_apopt = rng.exponential(6000, n_apoptotic)
    cd14_other = rng.exponential(7000, n_other)
    data["CD14-PE"] = np.concatenate([
        cd14_debris,
        cd14_negative[:n_regular_live],
        cd14_negative[n_regular_live:],  # cd34
        cd14_positive,
        cd14_apopt,
        cd14_other
    ])

    # === CD68-FITC (макрофаги) ===
    cd68_debris = rng.exponential(2000, n_debris)
    cd68_negative = rng.exponential(3000, n_regular_live + n_cd34)
    cd68_positive = rng.normal(80000, 15000, n_macro)
    cd68_apopt = rng.exponential(2500, n_apoptotic)
    cd68_other = rng.exponential(2800, n_other)
    data["CD68-FITC"] = np.concatenate([
        cd68_debris,
        cd68_negative[:n_regular_live],
        cd68_negative[n_regular_live:],  # cd34
        cd68_positive,
        cd68_apopt,
        cd68_other
    ])

    # === Annexin-V-Pacific Blue (апоптоз) ===
    annex_debris = rng.exponential(3000, n_debris)
    annex_live = rng.exponential(2000, n_live)  # Низкий = живые
    annex_apopt = rng.normal(120000, 20000, n_apoptotic)  # Высокий = апоптоз
    annex_other = rng.exponential(2500, n_other)
    data["Annexin-V-Pacific Blue"] = np.concatenate([
        annex_debris, annex_live, annex_apopt, annex_other
    ])

    # Клипируем отрицательные значения
    for key in data:
        data[key] = np.clip(data[key], 0, 262144)

    # Перемешиваем данные
    df = pd.DataFrame(data, columns=sample_channels)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


@pytest.fixture
def mock_fcs_data_inflamed(sample_channels):
    """
    Mock данные с повышенным воспалением.

    Характеристики (из description_parameter_extraction.md):
    - macrophage_fraction: 0.08 (повышено)
    - apoptotic_fraction: 0.05 (повышено)
    - Ожидаемый inflammation_level: ~0.7
    """
    rng = np.random.default_rng(43)
    n_events = 10000

    # Повышенные макрофаги и апоптоз
    n_debris = 2000
    n_macro = 800        # 8% (повышено)
    n_apopt = 500        # 5% (повышено)
    n_cd34 = 300         # 3%
    n_live_other = n_events - n_debris - n_macro - n_apopt - n_cd34

    data = {}

    # FSC-A
    data["FSC-A"] = np.concatenate([
        rng.uniform(5000, 30000, n_debris),
        rng.normal(100000, 20000, n_live_other),
        rng.normal(95000, 18000, n_cd34),
        rng.normal(105000, 22000, n_macro),
        rng.normal(70000, 25000, n_apopt),
    ])

    # FSC-H
    data["FSC-H"] = data["FSC-A"] * np.concatenate([
        rng.normal(0.9, 0.1, n_debris),
        rng.normal(0.95, 0.03, n_live_other),
        rng.normal(0.95, 0.03, n_cd34),
        rng.normal(0.95, 0.03, n_macro),
        rng.normal(0.95, 0.05, n_apopt),
    ])

    # SSC-A
    data["SSC-A"] = np.concatenate([
        rng.uniform(3000, 20000, n_debris),
        rng.normal(50000, 15000, n_live_other),
        rng.normal(45000, 12000, n_cd34),
        rng.normal(70000, 20000, n_macro),  # Высокий SSC для макрофагов
        rng.normal(40000, 20000, n_apopt),
    ])

    # CD34-APC
    data["CD34-APC"] = np.concatenate([
        rng.exponential(3000, n_debris),
        rng.exponential(5000, n_live_other),
        rng.normal(150000, 30000, n_cd34),
        rng.exponential(5000, n_macro),
        rng.exponential(4000, n_apopt),
    ])

    # CD14-PE
    data["CD14-PE"] = np.concatenate([
        rng.exponential(5000, n_debris),
        rng.exponential(8000, n_live_other),
        rng.exponential(7000, n_cd34),
        rng.normal(100000, 20000, n_macro),
        rng.exponential(6000, n_apopt),
    ])

    # CD68-FITC
    data["CD68-FITC"] = np.concatenate([
        rng.exponential(2000, n_debris),
        rng.exponential(3000, n_live_other),
        rng.exponential(2500, n_cd34),
        rng.normal(80000, 15000, n_macro),
        rng.exponential(2500, n_apopt),
    ])

    # Annexin-V
    data["Annexin-V-Pacific Blue"] = np.concatenate([
        rng.exponential(3000, n_debris),
        rng.exponential(2000, n_live_other + n_cd34 + n_macro),
        rng.normal(120000, 20000, n_apopt),
    ])

    for key in data:
        data[key] = np.clip(data[key], 0, 262144)

    df = pd.DataFrame(data, columns=sample_channels)
    return df.sample(frac=1, random_state=43).reset_index(drop=True)


@pytest.fixture
def mock_fcs_data_regenerating(sample_channels):
    """
    Mock данные с высокой регенерацией.

    Характеристики (из description_parameter_extraction.md):
    - stem_cell_fraction: 0.10 (повышено)
    - macrophage_fraction: 0.02 (низко)
    - apoptotic_fraction: 0.01 (низко)
    - Ожидаемый inflammation_level: ~0.2
    """
    rng = np.random.default_rng(44)
    n_events = 10000

    n_debris = 2000
    n_cd34 = 1000        # 10% (повышено)
    n_macro = 200        # 2% (низко)
    n_apopt = 100        # 1% (низко)
    n_live_other = n_events - n_debris - n_cd34 - n_macro - n_apopt

    data = {}

    # FSC-A
    data["FSC-A"] = np.concatenate([
        rng.uniform(5000, 30000, n_debris),
        rng.normal(100000, 20000, n_live_other),
        rng.normal(95000, 18000, n_cd34),
        rng.normal(105000, 22000, n_macro),
        rng.normal(70000, 25000, n_apopt),
    ])

    # FSC-H
    data["FSC-H"] = data["FSC-A"] * np.concatenate([
        rng.normal(0.9, 0.1, n_debris),
        rng.normal(0.95, 0.03, n_live_other + n_cd34 + n_macro),
        rng.normal(0.95, 0.05, n_apopt),
    ])

    # SSC-A
    data["SSC-A"] = np.concatenate([
        rng.uniform(3000, 20000, n_debris),
        rng.normal(50000, 15000, n_live_other),
        rng.normal(40000, 12000, n_cd34),  # Низкий SSC для стволовых
        rng.normal(70000, 20000, n_macro),
        rng.normal(40000, 20000, n_apopt),
    ])

    # CD34-APC
    data["CD34-APC"] = np.concatenate([
        rng.exponential(3000, n_debris),
        rng.exponential(5000, n_live_other),
        rng.normal(150000, 30000, n_cd34),
        rng.exponential(5000, n_macro),
        rng.exponential(4000, n_apopt),
    ])

    # CD14-PE
    data["CD14-PE"] = np.concatenate([
        rng.exponential(5000, n_debris),
        rng.exponential(8000, n_live_other + n_cd34),
        rng.normal(100000, 20000, n_macro),
        rng.exponential(6000, n_apopt),
    ])

    # CD68-FITC
    data["CD68-FITC"] = np.concatenate([
        rng.exponential(2000, n_debris),
        rng.exponential(3000, n_live_other + n_cd34),
        rng.normal(80000, 15000, n_macro),
        rng.exponential(2500, n_apopt),
    ])

    # Annexin-V
    data["Annexin-V-Pacific Blue"] = np.concatenate([
        rng.exponential(3000, n_debris),
        rng.exponential(2000, n_live_other + n_cd34 + n_macro),
        rng.normal(120000, 20000, n_apopt),
    ])

    for key in data:
        data[key] = np.clip(data[key], 0, 262144)

    df = pd.DataFrame(data, columns=sample_channels)
    return df.sample(frac=1, random_state=44).reset_index(drop=True)


@pytest.fixture
def mock_fcs_data_empty(sample_channels):
    """Пустой DataFrame для тестирования edge cases."""
    return pd.DataFrame(columns=sample_channels)


@pytest.fixture
def mock_fcs_data_single_event(sample_channels):
    """DataFrame с одним событием."""
    return pd.DataFrame({
        "FSC-A": [100000],
        "FSC-H": [95000],
        "SSC-A": [50000],
        "CD34-APC": [5000],
        "CD14-PE": [8000],
        "CD68-FITC": [3000],
        "Annexin-V-Pacific Blue": [2000],
    }, columns=sample_channels)


# =============================================================================
# Фикстуры для mock FlowKit
# =============================================================================

@pytest.fixture
def mock_flowkit_sample(sample_channels):
    """Mock объект FlowKit Sample для тестирования без реальных FCS файлов."""
    mock_sample = MagicMock()
    mock_sample.pnn_labels = sample_channels
    mock_sample.event_count = 10000

    # Mock для get_events
    mock_data = np.random.rand(10000, 7) * 100000
    mock_sample.get_events.return_value = mock_data

    # Mock для get_metadata
    mock_sample.get_metadata.return_value = {
        '$CYT': 'MockCytometer',
        '$DATE': '22-JAN-2026',
        'FCSversion': '3.1',
    }

    # Mock для as_dataframe
    mock_sample.as_dataframe.return_value = pd.DataFrame(
        mock_data,
        columns=sample_channels
    )

    # Mock для get_channel_events
    def get_channel_events(channel, source='raw'):
        idx = sample_channels.index(channel)
        return mock_data[:, idx]
    mock_sample.get_channel_events.side_effect = get_channel_events

    return mock_sample


@pytest.fixture
def mock_fcs_file(tmp_path):
    """Путь к временному mock FCS файлу."""
    fcs_path = tmp_path / "test_sample.fcs"
    fcs_path.touch()  # Создаем пустой файл
    return fcs_path


# =============================================================================
# Фикстуры для GatingResults (используются в test_parameter_extraction.py)
# =============================================================================

@pytest.fixture
def mock_gating_results_normal():
    """
    Mock GatingResults с нормальными фракциями.
    Импортируем локально чтобы избежать циклических импортов.
    """
    from src.data.gating import GatingResults, GateResult

    n_total = 10000

    gates = {
        "non_debris": GateResult(
            name="non_debris",
            mask=np.array([True] * 8000 + [False] * 2000),
            n_events=8000,
            fraction=0.80,
            parent=None,
        ),
        "singlets": GateResult(
            name="singlets",
            mask=np.array([True] * 7500 + [False] * 2500),
            n_events=7500,
            fraction=0.75,
            parent="non_debris",
        ),
        "live_cells": GateResult(
            name="live_cells",
            mask=np.array([True] * 7000 + [False] * 3000),
            n_events=7000,
            fraction=0.70,
            parent="singlets",
        ),
        "cd34_positive": GateResult(
            name="cd34_positive",
            mask=np.array([True] * 500 + [False] * 9500),
            n_events=500,
            fraction=0.05,
            parent="live_cells",
        ),
        "macrophages": GateResult(
            name="macrophages",
            mask=np.array([True] * 300 + [False] * 9700),
            n_events=300,
            fraction=0.03,
            parent="live_cells",
        ),
        "apoptotic": GateResult(
            name="apoptotic",
            mask=np.array([True] * 200 + [False] * 9800),
            n_events=200,
            fraction=0.02,
            parent="singlets",
        ),
    }

    return GatingResults(total_events=n_total, gates=gates)


@pytest.fixture
def mock_gating_results_inflamed():
    """
    Mock GatingResults с высоким воспалением.
    macro=0.08, apopt=0.05 -> expected inflammation ~0.7
    """
    from src.data.gating import GatingResults, GateResult

    n_total = 10000

    gates = {
        "non_debris": GateResult(
            name="non_debris",
            mask=np.array([True] * 8000 + [False] * 2000),
            n_events=8000,
            fraction=0.80,
        ),
        "singlets": GateResult(
            name="singlets",
            mask=np.array([True] * 7500 + [False] * 2500),
            n_events=7500,
            fraction=0.75,
            parent="non_debris",
        ),
        "live_cells": GateResult(
            name="live_cells",
            mask=np.array([True] * 6000 + [False] * 4000),
            n_events=6000,
            fraction=0.60,
            parent="singlets",
        ),
        "cd34_positive": GateResult(
            name="cd34_positive",
            mask=np.array([True] * 300 + [False] * 9700),
            n_events=300,
            fraction=0.03,
            parent="live_cells",
        ),
        "macrophages": GateResult(
            name="macrophages",
            mask=np.array([True] * 800 + [False] * 9200),
            n_events=800,
            fraction=0.08,
            parent="live_cells",
        ),
        "apoptotic": GateResult(
            name="apoptotic",
            mask=np.array([True] * 500 + [False] * 9500),
            n_events=500,
            fraction=0.05,
            parent="singlets",
        ),
    }

    return GatingResults(total_events=n_total, gates=gates)


@pytest.fixture
def mock_gating_results_regenerating():
    """
    Mock GatingResults с высокой регенерацией.
    stem=0.10, macro=0.02, apopt=0.01 -> expected inflammation ~0.2
    """
    from src.data.gating import GatingResults, GateResult

    n_total = 10000

    gates = {
        "non_debris": GateResult(
            name="non_debris",
            mask=np.array([True] * 8000 + [False] * 2000),
            n_events=8000,
            fraction=0.80,
        ),
        "singlets": GateResult(
            name="singlets",
            mask=np.array([True] * 7800 + [False] * 2200),
            n_events=7800,
            fraction=0.78,
            parent="non_debris",
        ),
        "live_cells": GateResult(
            name="live_cells",
            mask=np.array([True] * 7700 + [False] * 2300),
            n_events=7700,
            fraction=0.77,
            parent="singlets",
        ),
        "cd34_positive": GateResult(
            name="cd34_positive",
            mask=np.array([True] * 1000 + [False] * 9000),
            n_events=1000,
            fraction=0.10,
            parent="live_cells",
        ),
        "macrophages": GateResult(
            name="macrophages",
            mask=np.array([True] * 200 + [False] * 9800),
            n_events=200,
            fraction=0.02,
            parent="live_cells",
        ),
        "apoptotic": GateResult(
            name="apoptotic",
            mask=np.array([True] * 100 + [False] * 9900),
            n_events=100,
            fraction=0.01,
            parent="singlets",
        ),
    }

    return GatingResults(total_events=n_total, gates=gates)


# =============================================================================
# Фикстуры для модулей математического ядра (Phase 2)
# =============================================================================

@pytest.fixture
def default_sde_config():
    """SDEConfig с параметрами по умолчанию."""
    from src.core.sde_model import SDEConfig
    return SDEConfig()


@pytest.fixture
def custom_sde_config():
    """SDEConfig с кастомными параметрами для быстрых тестов."""
    from src.core.sde_model import SDEConfig
    return SDEConfig(r=0.5, K=1e7, dt=0.05, t_max=5.0)


@pytest.fixture
def prp_therapy_protocol():
    """TherapyProtocol с включенной PRP терапией."""
    from src.core.sde_model import TherapyProtocol
    return TherapyProtocol(
        prp_enabled=True,
        prp_start_time=1.0,
        prp_duration=7.0,
        prp_intensity=1.5,
        prp_initial_concentration=10.0,
    )


@pytest.fixture
def pemf_therapy_protocol():
    """TherapyProtocol с включенной PEMF терапией."""
    from src.core.sde_model import TherapyProtocol
    return TherapyProtocol(
        pemf_enabled=True,
        pemf_start_time=0.0,
        pemf_duration=14.0,
        pemf_frequency=60.0,
        pemf_intensity=1.0,
    )


@pytest.fixture
def combined_therapy_protocol():
    """TherapyProtocol с комбинированной PRP+PEMF терапией."""
    from src.core.sde_model import TherapyProtocol
    return TherapyProtocol(
        prp_enabled=True,
        prp_start_time=1.0,
        prp_duration=7.0,
        prp_intensity=1.0,
        pemf_enabled=True,
        pemf_start_time=0.0,
        pemf_duration=14.0,
        pemf_frequency=50.0,
        synergy_factor=1.3,
    )


@pytest.fixture
def sample_model_parameters():
    """ModelParameters для тестирования симуляций."""
    from src.data.parameter_extraction import ModelParameters
    return ModelParameters(
        n0=5000.0,
        c0=10.0,
        stem_cell_fraction=0.05,
        macrophage_fraction=0.03,
        apoptotic_fraction=0.02,
        inflammation_level=0.3,
    )


@pytest.fixture
def default_abm_config():
    """ABMConfig с параметрами по умолчанию."""
    from src.core.abm_model import ABMConfig
    return ABMConfig()


@pytest.fixture
def small_abm_config():
    """ABMConfig для быстрых тестов (маленькое пространство, короткое время)."""
    from src.core.abm_model import ABMConfig
    return ABMConfig(
        space_size=(50.0, 50.0),
        dt=0.5,
        t_max=24.0,  # 1 день
        initial_stem_cells=10,
        initial_macrophages=5,
        initial_fibroblasts=5,
        max_agents=100,
    )


@pytest.fixture
def sample_rng():
    """NumPy random generator с фиксированным seed для воспроизводимости."""
    return np.random.default_rng(42)


@pytest.fixture
def default_integration_config():
    """IntegrationConfig с параметрами по умолчанию."""
    from src.core.integration import IntegrationConfig
    return IntegrationConfig()


@pytest.fixture
def bidirectional_integration_config():
    """IntegrationConfig с двусторонней связью."""
    from src.core.integration import IntegrationConfig
    return IntegrationConfig(
        mode="bidirectional",
        coupling_strength=0.7,
        correction_rate=0.2,
        sync_interval=2.0,
    )


@pytest.fixture
def sde_only_integration_config():
    """IntegrationConfig только с SDE."""
    from src.core.integration import IntegrationConfig
    return IntegrationConfig(mode="sde_only")


@pytest.fixture
def abm_only_integration_config():
    """IntegrationConfig только с ABM."""
    from src.core.integration import IntegrationConfig
    return IntegrationConfig(mode="abm_only")


@pytest.fixture
def sde_monte_carlo_config():
    """MonteCarloConfig для SDE модели."""
    from src.core.sde_model import SDEConfig
    from src.core.monte_carlo import MonteCarloConfig
    return MonteCarloConfig(
        n_trajectories=10,
        model_type="sde",
        sde_config=SDEConfig(t_max=5.0),
        base_seed=42,
    )


@pytest.fixture
def abm_monte_carlo_config(small_abm_config):
    """MonteCarloConfig для ABM модели."""
    from src.core.monte_carlo import MonteCarloConfig
    return MonteCarloConfig(
        n_trajectories=5,
        model_type="abm",
        abm_config=small_abm_config,
        base_seed=42,
    )


# =============================================================================
# Фикстуры для Image Loader (Phase 1)
# =============================================================================

@pytest.fixture
def mock_scatter_plot_image():
    """Mock scatter plot изображение как numpy array.

    Создаёт 400x400 RGB изображение с белым фоном и случайными
    цветными точками для тестирования ScatterPlotExtractor.
    """
    rng = np.random.default_rng(42)

    # Белый фон
    image = np.ones((400, 400, 3), dtype=np.uint8) * 255

    # Добавляем случайные точки
    n_points = 100
    for _ in range(n_points):
        x = rng.integers(50, 350)
        y = rng.integers(50, 350)
        color = rng.integers(0, 200, 3)
        # Рисуем круг радиусом 3
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx * dx + dy * dy <= 9:
                    px, py = x + dx, y + dy
                    if 0 <= px < 400 and 0 <= py < 400:
                        image[py, px] = color

    return image


@pytest.fixture
def mock_grayscale_image():
    """Mock grayscale изображение."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (200, 200), dtype=np.uint8)


@pytest.fixture
def mock_image_with_axes():
    """Mock изображение scatter plot с осями.

    Создаёт изображение с чёрными осями и цветными точками
    для тестирования детекции осей.
    """
    rng = np.random.default_rng(42)

    # Белый фон
    image = np.ones((400, 400, 3), dtype=np.uint8) * 255

    # Рисуем оси (чёрные линии)
    # Y-axis (вертикальная линия x=50)
    image[50:350, 48:52] = [0, 0, 0]
    # X-axis (горизонтальная линия y=350)
    image[348:352, 50:350] = [0, 0, 0]

    # Добавляем точки в области графика
    n_points = 50
    for _ in range(n_points):
        x = rng.integers(70, 330)
        y = rng.integers(70, 330)
        color = [rng.integers(50, 200) for _ in range(3)]
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx * dx + dy * dy <= 4:
                    px, py = x + dx, y + dy
                    image[py, px] = color

    return image


@pytest.fixture
def sample_image_path(tmp_path):
    """Путь к временному изображению scatter plot.

    Создаёт реальный PNG файл для тестирования ImageLoader.load().
    """
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow not installed")

    img_path = tmp_path / "scatter_plot.png"

    # Создаём простое изображение
    rng = np.random.default_rng(42)
    img_array = np.ones((200, 200, 3), dtype=np.uint8) * 255

    # Добавляем несколько точек
    for i in range(20):
        x, y = rng.integers(20, 180, 2)
        img_array[y - 2 : y + 3, x - 2 : x + 3] = [
            rng.integers(50, 200),
            rng.integers(50, 200),
            rng.integers(50, 200),
        ]

    img = Image.fromarray(img_array, mode="RGB")
    img.save(img_path)

    return img_path


@pytest.fixture
def sample_jpeg_path(tmp_path):
    """Путь к временному JPEG изображению."""
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow not installed")

    img_path = tmp_path / "scatter_plot.jpg"

    rng = np.random.default_rng(42)
    img_array = rng.integers(100, 200, (150, 150, 3), dtype=np.uint8)

    img = Image.fromarray(img_array, mode="RGB")
    img.save(img_path, format="JPEG")

    return img_path


@pytest.fixture
def mock_image_metadata():
    """Mock ImageMetadata для тестов."""
    from src.data.image_loader import ImageMetadata

    return ImageMetadata(
        filename="test_scatter.png",
        width=400,
        height=400,
        channels=3,
        format="PNG",
        bit_depth=8,
        file_size_bytes=50000,
        has_alpha=False,
    )


@pytest.fixture
def default_image_config():
    """ImageConfig с параметрами по умолчанию."""
    from src.data.image_loader import ImageConfig

    return ImageConfig()


@pytest.fixture
def custom_image_config():
    """ImageConfig с кастомными параметрами."""
    from src.data.image_loader import ImageConfig

    return ImageConfig(
        max_dimension=1024,
        point_detection_method="contour",
        min_point_radius=5,
        max_point_radius=30,
        dominant_colors_count=3,
        auto_detect_axes=False,
    )


@pytest.fixture
def mock_scatter_plot_data():
    """Mock ScatterPlotData для тестов."""
    from src.data.image_loader import ScatterPlotData

    rng = np.random.default_rng(42)
    n_points = 50

    return ScatterPlotData(
        points=rng.integers(50, 350, (n_points, 2)),
        points_normalized=rng.random((n_points, 2)),
        colors=rng.integers(0, 256, (n_points, 3), dtype=np.uint8),
        color_labels=rng.integers(0, 3, n_points),
        n_points=n_points,
        detection_confidence=0.85,
        plot_bounds=(50, 50, 350, 350),
        axis_labels=("X-axis", "Y-axis"),
    )


@pytest.fixture
def mock_image_analysis_result():
    """Mock ImageAnalysisResult для тестов."""
    from src.data.image_loader import ImageAnalysisResult

    rng = np.random.default_rng(42)

    return ImageAnalysisResult(
        histogram_r=rng.integers(0, 1000, 256),
        histogram_g=rng.integers(0, 1000, 256),
        histogram_b=rng.integers(0, 1000, 256),
        histogram_gray=rng.integers(0, 1000, 256),
        dominant_colors=np.array([[255, 255, 255], [100, 50, 50], [50, 100, 50]]),
        dominant_colors_percentages=np.array([0.7, 0.2, 0.1]),
        mean_color=(180.0, 175.0, 170.0),
        std_color=(45.0, 50.0, 48.0),
        brightness=175.0,
        contrast=48.0,
        regions=None,
    )


# =============================================================================
# Фикстуры для расширенного гейтирования (9 каналов, 8 популяций)
# =============================================================================

@pytest.fixture
def extended_channels():
    """Список расширенных каналов (9 каналов)."""
    return EXTENDED_CHANNELS.copy()


@pytest.fixture
def extended_expected_fractions():
    """Ожидаемые фракции расширенного гейтирования (8 популяций)."""
    return EXTENDED_EXPECTED_FRACTIONS.copy()


@pytest.fixture
def mock_fcs_data_extended_normal(extended_channels):
    """
    Генерирует расширенный mock DataFrame с 9 каналами.

    Популяции (из description_gating.md):
    - Debris: ~20%
    - Live cells: ~70%
    - CD34+: ~5%
    - Macrophages: ~3%
    - Apoptotic: ~2%
    - Neutrophils (CD66b+): ~5%
    - Endothelial (CD31+): ~3%
    """
    rng = np.random.default_rng(45)
    n_events = 10000

    n_debris = 2000
    n_live = 7000
    n_apoptotic = 200
    n_other = 800

    n_cd34 = 500
    n_macro = 300
    n_neutro = 500
    n_endo = 300
    n_regular_live = n_live - n_cd34 - n_macro - n_neutro - n_endo

    data = {}

    # === FSC-A ===
    data["FSC-A"] = np.concatenate([
        rng.uniform(5000, 30000, n_debris),
        rng.normal(100000, 20000, n_regular_live),
        rng.normal(95000, 18000, n_cd34),
        rng.normal(105000, 22000, n_macro),
        rng.normal(98000, 20000, n_neutro),
        rng.normal(92000, 18000, n_endo),
        rng.normal(70000, 25000, n_apoptotic),
        rng.normal(90000, 25000, n_other),
    ])

    # === FSC-H ===
    data["FSC-H"] = data["FSC-A"] * np.concatenate([
        rng.normal(0.9, 0.1, n_debris),
        rng.normal(0.95, 0.03, n_regular_live + n_cd34 + n_macro + n_neutro + n_endo),
        rng.normal(0.95, 0.05, n_apoptotic),
        rng.normal(1.5, 0.2, n_other),
    ])

    # === SSC-A ===
    data["SSC-A"] = np.concatenate([
        rng.uniform(3000, 20000, n_debris),
        rng.normal(50000, 15000, n_regular_live),
        rng.normal(40000, 12000, n_cd34),
        rng.normal(70000, 20000, n_macro),
        rng.normal(65000, 18000, n_neutro),
        rng.normal(45000, 14000, n_endo),
        rng.normal(40000, 20000, n_apoptotic),
        rng.normal(55000, 18000, n_other),
    ])

    # === CD34-APC ===
    data["CD34-APC"] = np.concatenate([
        rng.exponential(3000, n_debris),
        rng.exponential(5000, n_regular_live),
        rng.normal(150000, 30000, n_cd34),
        rng.exponential(5000, n_macro),
        rng.exponential(4500, n_neutro),
        rng.exponential(4500, n_endo),
        rng.exponential(4000, n_apoptotic),
        rng.exponential(4500, n_other),
    ])

    # === CD14-PE ===
    data["CD14-PE"] = np.concatenate([
        rng.exponential(5000, n_debris),
        rng.exponential(8000, n_regular_live + n_cd34 + n_neutro + n_endo),
        rng.normal(100000, 20000, n_macro),
        rng.exponential(6000, n_apoptotic),
        rng.exponential(7000, n_other),
    ])

    # === CD68-FITC ===
    data["CD68-FITC"] = np.concatenate([
        rng.exponential(2000, n_debris),
        rng.exponential(3000, n_regular_live + n_cd34 + n_neutro + n_endo),
        rng.normal(80000, 15000, n_macro),
        rng.exponential(2500, n_apoptotic),
        rng.exponential(2800, n_other),
    ])

    # === Annexin-V-Pacific Blue ===
    data["Annexin-V-Pacific Blue"] = np.concatenate([
        rng.exponential(3000, n_debris),
        rng.exponential(2000, n_live),
        rng.normal(120000, 20000, n_apoptotic),
        rng.exponential(2500, n_other),
    ])

    # === CD66b-PE-Cy7 (нейтрофилы) ===
    data["CD66b-PE-Cy7"] = np.concatenate([
        rng.exponential(3000, n_debris),
        rng.exponential(4000, n_regular_live + n_cd34 + n_macro + n_endo),
        rng.normal(120000, 25000, n_neutro),
        rng.exponential(3500, n_apoptotic),
        rng.exponential(3800, n_other),
    ])

    # === CD31-BV421 (эндотелий) ===
    data["CD31-BV421"] = np.concatenate([
        rng.exponential(2000, n_debris),
        rng.exponential(3000, n_regular_live + n_cd34 + n_macro + n_neutro),
        rng.normal(100000, 20000, n_endo),
        rng.exponential(2500, n_apoptotic),
        rng.exponential(2800, n_other),
    ])

    for key in data:
        data[key] = np.clip(data[key], 0, 262144)

    df = pd.DataFrame(data, columns=extended_channels)
    return df.sample(frac=1, random_state=45).reset_index(drop=True)


@pytest.fixture
def mock_gating_results_extended_normal():
    """
    Mock GatingResults с 8 гейтами (расширенное гейтирование).
    Нормальные фракции: neutrophils=0.05, endothelial=0.03.
    """
    from src.data.gating import GatingResults, GateResult

    n_total = 10000

    gates = {
        "non_debris": GateResult(
            name="non_debris",
            mask=np.array([True] * 8000 + [False] * 2000),
            n_events=8000,
            fraction=0.80,
            parent=None,
        ),
        "singlets": GateResult(
            name="singlets",
            mask=np.array([True] * 7500 + [False] * 2500),
            n_events=7500,
            fraction=0.75,
            parent="non_debris",
        ),
        "live_cells": GateResult(
            name="live_cells",
            mask=np.array([True] * 7000 + [False] * 3000),
            n_events=7000,
            fraction=0.70,
            parent="singlets",
        ),
        "cd34_positive": GateResult(
            name="cd34_positive",
            mask=np.array([True] * 500 + [False] * 9500),
            n_events=500,
            fraction=0.05,
            parent="live_cells",
        ),
        "macrophages": GateResult(
            name="macrophages",
            mask=np.array([True] * 300 + [False] * 9700),
            n_events=300,
            fraction=0.03,
            parent="live_cells",
        ),
        "apoptotic": GateResult(
            name="apoptotic",
            mask=np.array([True] * 200 + [False] * 9800),
            n_events=200,
            fraction=0.02,
            parent="singlets",
        ),
        "neutrophils": GateResult(
            name="neutrophils",
            mask=np.array([True] * 500 + [False] * 9500),
            n_events=500,
            fraction=0.05,
            parent="live_cells",
        ),
        "endothelial": GateResult(
            name="endothelial",
            mask=np.array([True] * 300 + [False] * 9700),
            n_events=300,
            fraction=0.03,
            parent="live_cells",
        ),
    }

    return GatingResults(total_events=n_total, gates=gates)


@pytest.fixture
def mock_gating_results_extended_inflamed():
    """
    Mock GatingResults расширенные с высоким воспалением.
    neutrophils=0.10, endothelial=0.02.
    """
    from src.data.gating import GatingResults, GateResult

    n_total = 10000

    gates = {
        "non_debris": GateResult(
            name="non_debris",
            mask=np.array([True] * 8000 + [False] * 2000),
            n_events=8000,
            fraction=0.80,
        ),
        "singlets": GateResult(
            name="singlets",
            mask=np.array([True] * 7500 + [False] * 2500),
            n_events=7500,
            fraction=0.75,
            parent="non_debris",
        ),
        "live_cells": GateResult(
            name="live_cells",
            mask=np.array([True] * 6000 + [False] * 4000),
            n_events=6000,
            fraction=0.60,
            parent="singlets",
        ),
        "cd34_positive": GateResult(
            name="cd34_positive",
            mask=np.array([True] * 300 + [False] * 9700),
            n_events=300,
            fraction=0.03,
            parent="live_cells",
        ),
        "macrophages": GateResult(
            name="macrophages",
            mask=np.array([True] * 800 + [False] * 9200),
            n_events=800,
            fraction=0.08,
            parent="live_cells",
        ),
        "apoptotic": GateResult(
            name="apoptotic",
            mask=np.array([True] * 500 + [False] * 9500),
            n_events=500,
            fraction=0.05,
            parent="singlets",
        ),
        "neutrophils": GateResult(
            name="neutrophils",
            mask=np.array([True] * 1000 + [False] * 9000),
            n_events=1000,
            fraction=0.10,
            parent="live_cells",
        ),
        "endothelial": GateResult(
            name="endothelial",
            mask=np.array([True] * 200 + [False] * 9800),
            n_events=200,
            fraction=0.02,
            parent="live_cells",
        ),
    }

    return GatingResults(total_events=n_total, gates=gates)


@pytest.fixture
def mock_gating_results_extended_regenerating():
    """
    Mock GatingResults расширенные с высокой регенерацией.
    neutrophils=0.03, endothelial=0.05 (повышенный ангиогенез).
    """
    from src.data.gating import GatingResults, GateResult

    n_total = 10000

    gates = {
        "non_debris": GateResult(
            name="non_debris",
            mask=np.array([True] * 8000 + [False] * 2000),
            n_events=8000,
            fraction=0.80,
        ),
        "singlets": GateResult(
            name="singlets",
            mask=np.array([True] * 7800 + [False] * 2200),
            n_events=7800,
            fraction=0.78,
            parent="non_debris",
        ),
        "live_cells": GateResult(
            name="live_cells",
            mask=np.array([True] * 7700 + [False] * 2300),
            n_events=7700,
            fraction=0.77,
            parent="singlets",
        ),
        "cd34_positive": GateResult(
            name="cd34_positive",
            mask=np.array([True] * 1000 + [False] * 9000),
            n_events=1000,
            fraction=0.10,
            parent="live_cells",
        ),
        "macrophages": GateResult(
            name="macrophages",
            mask=np.array([True] * 200 + [False] * 9800),
            n_events=200,
            fraction=0.02,
            parent="live_cells",
        ),
        "apoptotic": GateResult(
            name="apoptotic",
            mask=np.array([True] * 100 + [False] * 9900),
            n_events=100,
            fraction=0.01,
            parent="singlets",
        ),
        "neutrophils": GateResult(
            name="neutrophils",
            mask=np.array([True] * 300 + [False] * 9700),
            n_events=300,
            fraction=0.03,
            parent="live_cells",
        ),
        "endothelial": GateResult(
            name="endothelial",
            mask=np.array([True] * 500 + [False] * 9500),
            n_events=500,
            fraction=0.05,
            parent="live_cells",
        ),
    }

    return GatingResults(total_events=n_total, gates=gates)


# =============================================================================
# Фикстуры для ExtendedModelParameters (Phase 1 — расширенная модель)
# =============================================================================

@pytest.fixture
def mock_extended_model_parameters():
    """ExtendedModelParameters с 20 переменными для тестов."""
    from src.data.parameter_extraction import ExtendedModelParameters

    return ExtendedModelParameters(
        P0=250000.0,
        Ne0=500.0,
        M1_0=105.0,
        M2_0=45.0,
        F0=500.0,
        Mf0=0.0,
        E0=300.0,
        S0=250.0,
        C_TNF=0.16,
        C_IL10=0.065,
        C_PDGF=5.5,
        C_VEGF=0.55,
        C_TGFb=1.1,
        C_MCP1=0.22,
        C_IL8=0.12,
        rho_collagen=0.1,
        C_MMP=0.5,
        rho_fibrin=0.8,
        D=1.0,
        O2=0.95,
        source_file="test.fcs",
        total_events=10000,
        inflammation_level=0.3,
    )


# =============================================================================
# Фикстуры для DatasetLoader (Phase 1 — загрузчик данных)
# =============================================================================

@pytest.fixture
def mock_time_series_data():
    """TimeSeriesData для тестов dataset_loader."""
    from src.data.dataset_loader import TimeSeriesData

    return TimeSeriesData(
        time_points=np.array([0.0, 6.0, 24.0, 48.0, 72.0]),
        values={
            "Ne": np.array([100.0, 500.0, 800.0, 400.0, 200.0]),
            "M1": np.array([50.0, 200.0, 300.0, 150.0, 80.0]),
        },
        units={"Ne": "cells/ul", "M1": "cells/ul"},
    )


@pytest.fixture
def mock_dataset_metadata():
    """DatasetMetadata для тестов dataset_loader."""
    from src.data.dataset_loader import DatasetMetadata, DatasetSource

    return DatasetMetadata(
        source=DatasetSource.LOCAL,
        dataset_id="test-mock",
        description="Тестовый датасет для модульных тестов",
    )


@pytest.fixture
def mock_validation_dataset(mock_dataset_metadata, mock_time_series_data):
    """ValidationDataset для тестов dataset_loader."""
    from src.data.dataset_loader import ValidationDataset

    return ValidationDataset(
        metadata=mock_dataset_metadata,
        cell_counts=mock_time_series_data,
    )


# =============================================================================
# Фикстуры Phase 2: Новые типы агентов ABM
# =============================================================================


@pytest.fixture
def neutrophil_agent():
    """NeutrophilAgent для тестов Phase 2."""
    from src.core.abm_model import NeutrophilAgent

    rng = np.random.default_rng(42)
    return NeutrophilAgent(agent_id=1, x=50.0, y=50.0, age=0.0, rng=rng)


@pytest.fixture
def endothelial_agent():
    """EndothelialAgent для тестов Phase 2."""
    from src.core.abm_model import EndothelialAgent

    rng = np.random.default_rng(42)
    return EndothelialAgent(agent_id=1, x=50.0, y=50.0, age=0.0, rng=rng)


@pytest.fixture
def myofibroblast_agent():
    """MyofibroblastAgent для тестов Phase 2."""
    from src.core.abm_model import MyofibroblastAgent

    rng = np.random.default_rng(42)
    return MyofibroblastAgent(agent_id=1, x=50.0, y=50.0, age=0.0, rng=rng)


@pytest.fixture
def kdtree_spatial_index():
    """KDTreeSpatialIndex для тестов Phase 2."""
    from src.core.abm_model import KDTreeSpatialIndex

    return KDTreeSpatialIndex(space_size=(100.0, 100.0), periodic=True)


@pytest.fixture
def sample_abm_snapshot():
    """ABM snapshot для тестов integration Phase 2."""
    from src.core.abm_model import ABMSnapshot, AgentState

    agents = [
        AgentState(i, "stem", 10.0 * i, 10.0 * i, 0, 0, 1.0)
        for i in range(10)
    ]
    return ABMSnapshot(
        t=1.0,
        agents=agents,
        cytokine_field=np.ones((10, 10)) * 5.0,
        ecm_field=np.zeros((10, 10)),
    )


# =============================================================================
# Фикстуры Phase 2.5: Расширенная SDE система
# =============================================================================


@pytest.fixture
def default_parameter_set():
    """ParameterSet с литературными значениями по умолчанию."""
    from src.core.parameters import ParameterSet

    return ParameterSet()


@pytest.fixture
def default_extended_state():
    """ExtendedSDEState со всеми нулями."""
    from src.core.extended_sde import ExtendedSDEState

    return ExtendedSDEState()


@pytest.fixture
def wound_initial_state():
    """Начальное состояние раны t=0: тромбоциты, фибрин, damage, кислород."""
    from src.core.extended_sde import ExtendedSDEState

    return ExtendedSDEState(
        P=1e4, D=1.0, O2=100.0, rho_fibrin=1.0, t=0.0,
    )


@pytest.fixture
def inflammation_state():
    """Пик воспаления: Ne высокий, M1>M2, TNF/IL8 повышены."""
    from src.core.extended_sde import ExtendedSDEState

    return ExtendedSDEState(
        Ne=500.0, M1=200.0, M2=50.0,
        C_TNF=5.0, C_IL8=3.0, C_MCP1=2.0,
        D=0.5, t=24.0,
    )


@pytest.fixture
def proliferation_state():
    """Пролиферация: F высокий, M2>M1, VEGF/PDGF, коллаген растёт."""
    from src.core.extended_sde import ExtendedSDEState

    return ExtendedSDEState(
        F=1000.0, Mf=100.0, M2=300.0, M1=50.0, E=200.0, S=50.0,
        C_PDGF=3.0, C_VEGF=2.0, C_TGFb=2.0,
        rho_collagen=0.4, O2=50.0, t=168.0,
    )


@pytest.fixture
def remodeling_state():
    """Ремоделирование: высокий коллаген, MMP, мало клеток."""
    from src.core.extended_sde import ExtendedSDEState

    return ExtendedSDEState(
        F=200.0, Mf=10.0, M2=50.0, E=300.0,
        rho_collagen=0.9, C_MMP=0.5, rho_fibrin=0.01,
        O2=90.0, t=600.0,
    )


@pytest.fixture
def extended_sde_model():
    """ExtendedSDEModel с фиксированным seed для воспроизводимости."""
    from src.core.extended_sde import ExtendedSDEModel

    return ExtendedSDEModel(rng_seed=42)


@pytest.fixture
def wound_phase_detector():
    """WoundPhaseDetector с параметрами по умолчанию."""
    from src.core.wound_phases import WoundPhaseDetector

    return WoundPhaseDetector()


@pytest.fixture
def sample_extended_trajectory():
    """Маленькая траектория из 10 шагов для тестов."""
    from src.core.extended_sde import ExtendedSDEState, ExtendedSDETrajectory

    states = [
        ExtendedSDEState(P=100.0 - i * 5, Ne=float(i * 10), t=float(i))
        for i in range(10)
    ]
    return ExtendedSDETrajectory(
        times=np.arange(10, dtype=float),
        states=states,
    )
