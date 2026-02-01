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
