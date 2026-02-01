"""
Тесты производительности для RegenTwin data pipeline.

Проверяет соответствие требованиям из плана:
- Обработка одного файла < 30 секунд
- Гейтирование 10000 событий < 5 секунд
- Извлечение параметров < 1 секунда

Тесты используют pytest markers для группировки:
- @pytest.mark.performance - тесты производительности
- @pytest.mark.slow - медленные тесты (>1 сек)
"""

import time
import pytest
import numpy as np
import pandas as pd

from src.data.gating import GatingStrategy
from src.data.parameter_extraction import (
    ParameterExtractor,
    extract_model_parameters,
)


# =============================================================================
# Фикстуры для тестов производительности
# =============================================================================

@pytest.fixture
def large_mock_fcs_data():
    """
    Генерирует большой датасет (100000 событий) для тестов производительности.
    """
    rng = np.random.default_rng(42)
    n_events = 100000

    channels = [
        "FSC-A", "FSC-H", "SSC-A",
        "CD34-APC", "CD14-PE", "CD68-FITC",
        "Annexin-V-Pacific Blue",
    ]

    data = {
        "FSC-A": np.concatenate([
            rng.uniform(5000, 30000, n_events // 5),  # debris
            rng.normal(100000, 20000, 4 * n_events // 5),  # cells
        ]),
        "FSC-H": np.concatenate([
            rng.uniform(4000, 28000, n_events // 5),
            rng.normal(95000, 18000, 4 * n_events // 5),
        ]),
        "SSC-A": np.concatenate([
            rng.uniform(3000, 20000, n_events // 5),
            rng.normal(50000, 15000, 4 * n_events // 5),
        ]),
        "CD34-APC": np.concatenate([
            rng.exponential(3000, 9 * n_events // 10),
            rng.normal(150000, 30000, n_events // 10),
        ]),
        "CD14-PE": np.concatenate([
            rng.exponential(5000, 97 * n_events // 100),
            rng.normal(100000, 20000, 3 * n_events // 100),
        ]),
        "CD68-FITC": np.concatenate([
            rng.exponential(2000, 97 * n_events // 100),
            rng.normal(80000, 15000, 3 * n_events // 100),
        ]),
        "Annexin-V-Pacific Blue": np.concatenate([
            rng.exponential(2000, 98 * n_events // 100),
            rng.normal(120000, 20000, 2 * n_events // 100),
        ]),
    }

    for key in data:
        data[key] = np.clip(data[key], 0, 262144)

    df = pd.DataFrame(data, columns=channels)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# =============================================================================
# Тесты производительности гейтирования
# =============================================================================

@pytest.mark.performance
class TestGatingPerformance:
    """Тесты производительности модуля гейтирования."""

    def test_gating_10000_events_under_5_seconds(self, mock_fcs_data_normal):
        """
        Тест: гейтирование 10000 событий < 5 секунд.

        Требование из плана разработки.
        """
        strategy = GatingStrategy()

        start_time = time.perf_counter()
        results = strategy.apply(mock_fcs_data_normal)
        elapsed = time.perf_counter() - start_time

        assert elapsed < 5.0, \
            f"Гейтирование 10000 событий заняло {elapsed:.2f} сек (лимит: 5 сек)"

        # Проверка что результат валиден
        assert results.total_events == len(mock_fcs_data_normal)

    @pytest.mark.slow
    def test_gating_100000_events_under_30_seconds(self, large_mock_fcs_data):
        """
        Тест: гейтирование 100000 событий < 30 секунд.

        Проверка масштабируемости для больших файлов.
        """
        strategy = GatingStrategy()

        start_time = time.perf_counter()
        results = strategy.apply(large_mock_fcs_data)
        elapsed = time.perf_counter() - start_time

        assert elapsed < 30.0, \
            f"Гейтирование 100000 событий заняло {elapsed:.2f} сек (лимит: 30 сек)"

        assert results.total_events == 100000

    def test_debris_gate_performance(self, mock_fcs_data_normal):
        """Тест производительности debris_gate."""
        strategy = GatingStrategy()
        fsc = mock_fcs_data_normal["FSC-A"].values
        ssc = mock_fcs_data_normal["SSC-A"].values

        start_time = time.perf_counter()
        for _ in range(100):  # 100 итераций
            mask = strategy.debris_gate(fsc, ssc)
        elapsed = time.perf_counter() - start_time

        # 100 итераций < 1 сек
        assert elapsed < 1.0, \
            f"100 вызовов debris_gate заняли {elapsed:.2f} сек"

    def test_singlets_gate_performance(self, mock_fcs_data_normal):
        """Тест производительности singlets_gate."""
        strategy = GatingStrategy()
        fsc_a = mock_fcs_data_normal["FSC-A"].values
        fsc_h = mock_fcs_data_normal["FSC-H"].values

        start_time = time.perf_counter()
        for _ in range(100):
            mask = strategy.singlets_gate(fsc_a, fsc_h)
        elapsed = time.perf_counter() - start_time

        assert elapsed < 1.0, \
            f"100 вызовов singlets_gate заняли {elapsed:.2f} сек"


# =============================================================================
# Тесты производительности извлечения параметров
# =============================================================================

@pytest.mark.performance
class TestParameterExtractionPerformance:
    """Тесты производительности извлечения параметров."""

    def test_parameter_extraction_under_1_second(self, mock_gating_results_normal):
        """
        Тест: извлечение параметров < 1 секунда.

        Требование для real-time обработки.
        """
        extractor = ParameterExtractor()

        start_time = time.perf_counter()
        params = extractor.extract(mock_gating_results_normal)
        elapsed = time.perf_counter() - start_time

        assert elapsed < 1.0, \
            f"Извлечение параметров заняло {elapsed:.2f} сек (лимит: 1 сек)"

        assert params.validate() is True

    def test_extract_n0_performance(self, mock_gating_results_normal):
        """Тест производительности extract_n0."""
        extractor = ParameterExtractor()

        start_time = time.perf_counter()
        for _ in range(1000):
            n0 = extractor.extract_n0(mock_gating_results_normal)
        elapsed = time.perf_counter() - start_time

        # 1000 вызовов < 0.5 сек
        assert elapsed < 0.5, \
            f"1000 вызовов extract_n0 заняли {elapsed:.2f} сек"

    def test_extract_inflammation_performance(self, mock_gating_results_normal):
        """Тест производительности extract_inflammation_level."""
        extractor = ParameterExtractor()

        start_time = time.perf_counter()
        for _ in range(1000):
            inflammation = extractor.extract_inflammation_level(mock_gating_results_normal)
        elapsed = time.perf_counter() - start_time

        assert elapsed < 0.5, \
            f"1000 вызовов extract_inflammation_level заняли {elapsed:.2f} сек"


# =============================================================================
# Тесты производительности полного pipeline
# =============================================================================

@pytest.mark.performance
class TestFullPipelinePerformance:
    """Тесты производительности полного pipeline."""

    def test_full_pipeline_under_30_seconds(self, mock_fcs_data_normal):
        """
        Тест: полный pipeline < 30 секунд.

        Требование из RegenTwin_Development_Plan.md:
        "обработка одного файла < 30 сек"
        """
        start_time = time.perf_counter()

        # Step 1: Gating
        strategy = GatingStrategy()
        gating_results = strategy.apply(mock_fcs_data_normal)

        # Step 2: Parameter extraction
        params = extract_model_parameters(gating_results)

        # Step 3: Validation
        params.validate()

        elapsed = time.perf_counter() - start_time

        assert elapsed < 30.0, \
            f"Полный pipeline занял {elapsed:.2f} сек (лимит: 30 сек)"

    @pytest.mark.slow
    def test_large_dataset_pipeline_under_60_seconds(self, large_mock_fcs_data):
        """
        Тест: pipeline для 100000 событий < 60 секунд.

        Проверка масштабируемости.
        """
        start_time = time.perf_counter()

        strategy = GatingStrategy()
        gating_results = strategy.apply(large_mock_fcs_data)
        params = extract_model_parameters(gating_results)
        params.validate()

        elapsed = time.perf_counter() - start_time

        assert elapsed < 60.0, \
            f"Pipeline для 100000 событий занял {elapsed:.2f} сек (лимит: 60 сек)"

    def test_multiple_files_sequential(self, mock_fcs_data_normal):
        """
        Тест: обработка 10 файлов последовательно < 60 сек.

        Симуляция batch обработки.
        """
        strategy = GatingStrategy()

        start_time = time.perf_counter()

        for i in range(10):
            gating_results = strategy.apply(mock_fcs_data_normal)
            params = extract_model_parameters(
                gating_results,
                source_file=f"file_{i}.fcs"
            )
            params.validate()

        elapsed = time.perf_counter() - start_time

        # 10 файлов * 6 сек максимум каждый = 60 сек
        assert elapsed < 60.0, \
            f"Обработка 10 файлов заняла {elapsed:.2f} сек (лимит: 60 сек)"


# =============================================================================
# Benchmarks для CI/CD
# =============================================================================

@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Benchmarks для мониторинга регрессий производительности."""

    def test_benchmark_gating_apply(self, mock_fcs_data_normal, benchmark=None):
        """
        Benchmark для GatingStrategy.apply().

        Используется для отслеживания регрессий производительности в CI/CD.
        """
        strategy = GatingStrategy()

        # Warmup
        _ = strategy.apply(mock_fcs_data_normal)

        # Measure
        times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = strategy.apply(mock_fcs_data_normal)
            times.append(time.perf_counter() - start)

        avg_time = sum(times) / len(times)
        max_time = max(times)

        # Логируем для анализа
        print(f"\nGating apply benchmark:")
        print(f"  Average: {avg_time*1000:.1f} ms")
        print(f"  Max:     {max_time*1000:.1f} ms")

        # Baseline: среднее < 500ms для 10000 событий
        assert avg_time < 0.5, f"Средняя скорость {avg_time:.2f}s превышает baseline 0.5s"

    def test_benchmark_parameter_extraction(self, mock_gating_results_normal):
        """Benchmark для extract_model_parameters."""
        # Warmup
        _ = extract_model_parameters(mock_gating_results_normal)

        # Measure
        times = []
        for _ in range(100):
            start = time.perf_counter()
            _ = extract_model_parameters(mock_gating_results_normal)
            times.append(time.perf_counter() - start)

        avg_time = sum(times) / len(times)

        print(f"\nParameter extraction benchmark:")
        print(f"  Average: {avg_time*1000:.2f} ms")

        # Baseline: среднее < 10ms
        assert avg_time < 0.01, f"Средняя скорость {avg_time*1000:.2f}ms превышает baseline 10ms"
