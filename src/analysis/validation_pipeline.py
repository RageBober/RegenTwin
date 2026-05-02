"""End-to-end pipeline валидации Extended SDE модели на литературных данных.

Объединяет:
- DatasetLoader → загрузка reference data
- ExtendedSDE → симуляция с reference начальными условиями
- MonteCarloSimulator → ансамблевые предсказания (для PPC)
- ValidationRunner → DTW/CRPS, PPC, phase timing, Kendall τ

Результат: ValidationReport с overall score и диагностикой.

Подробное описание: Description/Phase3/description_validation_pipeline.md
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.analysis.validation import (
    PhaseBreakpoint,
    ValidationConfig,
    ValidationResult,
    ValidationRunner,
)
from src.core.extended_sde import (
    VARIABLE_NAMES,
    ExtendedSDEModel,
    ExtendedSDEState,
    ExtendedSDETrajectory,
)
from src.core.parameters import ParameterSet
from src.data.dataset_loader import (
    DatasetLoader,
    TimeSeriesData,
    ValidationDataset,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Результат полного пайплайна валидации.

    Содержит validation metrics, конфигурацию, и диагностику.
    """

    dataset_id: str
    validation_result: ValidationResult | None = None
    overall_score: float = 0.0
    trajectory: ExtendedSDETrajectory | None = None
    observed: TimeSeriesData | None = None
    parameters_used: dict[str, float] = field(default_factory=dict)
    initial_conditions: dict[str, float] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Сериализация отчёта в словарь (JSON-совместимый)."""
        result: dict[str, Any] = {
            "dataset_id": self.dataset_id,
            "overall_score": self.overall_score,
            "elapsed_seconds": self.elapsed_seconds,
            "initial_conditions": self.initial_conditions,
            "errors": self.errors,
        }
        if self.validation_result is not None:
            result["validation"] = self.validation_result.get_summary()
        return result

    def to_json(self, indent: int = 2) -> str:
        """Сериализация в JSON строку."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, path: str | Path) -> None:
        """Сохранение отчёта в JSON файл."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")


@dataclass
class PipelineConfig:
    """Конфигурация пайплайна валидации."""

    # Симуляция
    t_max: float = 720.0  # Макс. время (ч)
    dt: float = 0.1  # Шаг времени (ч) — грубее чем default 0.01 для скорости
    rng_seed: int = 42

    # Monte Carlo (для PPC)
    run_monte_carlo: bool = False
    n_mc_samples: int = 50  # Число ансамблевых реализаций

    # Validation metrics
    validation_config: ValidationConfig = field(
        default_factory=ValidationConfig,
    )

    # Output
    save_trajectory: bool = False  # Сохранять ли полную траекторию в отчёт


class ValidationPipeline:
    """End-to-end пайплайн валидации модели на литературных данных.

    Порядок:
    1. Загрузить reference dataset через DatasetLoader
    2. Извлечь начальные условия и observed time series
    3. Запустить ExtendedSDE симуляцию
    4. (Опционально) Запустить Monte Carlo для PPC
    5. Прогнать ValidationRunner.run_all()
    6. Сформировать ValidationReport
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        params: ParameterSet | None = None,
    ) -> None:
        self._config = config or PipelineConfig()
        self._params = params or ParameterSet()
        self._loader = DatasetLoader()

    def run(
        self,
        dataset_id: str = "literature-xue2009",
    ) -> ValidationReport:
        """Запустить полный пайплайн валидации.

        Args:
            dataset_id: Идентификатор датасета из AVAILABLE_DATASETS.
                        По умолчанию "literature-xue2009".

        Returns:
            ValidationReport с результатами.
        """
        t0 = time.perf_counter()
        report = ValidationReport(dataset_id=dataset_id)
        cfg = self._config

        # === 1. Загрузка данных ===
        try:
            dataset = self._loader.load(dataset_id)
        except (KeyError, FileNotFoundError) as e:
            report.errors.append(f"Data loading failed: {e}")
            report.elapsed_seconds = time.perf_counter() - t0
            return report

        # === 2. Извлечение observed и начальных условий ===
        observed = self._extract_observed(dataset)
        report.observed = observed

        initial_state = self._build_initial_state(dataset)
        report.initial_conditions = {
            name: getattr(initial_state, name)
            for name in VARIABLE_NAMES
            if hasattr(initial_state, name)
        }

        # === 3. Симуляция ExtendedSDE ===
        try:
            params = self._prepare_params()
            model = ExtendedSDEModel(
                params=params,
                rng_seed=cfg.rng_seed,
            )
            trajectory = model.simulate(
                initial_state=initial_state,
                t_span=(0.0, cfg.t_max),
            )
            report.parameters_used = params.to_dict()
            if cfg.save_trajectory:
                report.trajectory = trajectory
        except Exception as e:
            report.errors.append(f"Simulation failed: {e}")
            logger.warning(f"SDE simulation failed: {e}")
            report.elapsed_seconds = time.perf_counter() - t0
            return report

        # === 4. Monte Carlo (опционально) ===
        mc_results = None
        if cfg.run_monte_carlo:
            try:
                mc_results = self._run_monte_carlo(initial_state, params)
            except Exception as e:
                report.errors.append(f"Monte Carlo failed: {e}")
                logger.warning(f"Monte Carlo failed: {e}")

        # === 5. Phase breakpoints ===
        observed_breakpoints = self._get_phase_breakpoints(dataset_id)

        # === 6. ValidationRunner ===
        try:
            runner = ValidationRunner(config=cfg.validation_config)
            validation_result = runner.run_all(
                trajectory=trajectory,
                mc_results=mc_results,
                observed=observed,
                observed_breakpoints=observed_breakpoints,
            )
            report.validation_result = validation_result
            report.overall_score = validation_result.overall_score
        except Exception as e:
            report.errors.append(f"Validation failed: {e}")
            logger.warning(f"Validation failed: {e}")

        report.elapsed_seconds = time.perf_counter() - t0
        return report

    def _extract_observed(
        self,
        dataset: ValidationDataset,
    ) -> TimeSeriesData | None:
        """Извлечь observed TimeSeriesData из ValidationDataset.

        Объединяет cell_counts и cytokine_levels в один TimeSeriesData,
        используя имена переменных совместимые с VARIABLE_NAMES.
        """
        # Если есть raw_data с полным набором переменных — используем его
        if dataset.raw_data is not None:
            return dataset.raw_data  # type: ignore[return-value]

        # Иначе объединяем cell_counts + cytokine_levels
        all_values: dict[str, np.ndarray] = {}
        all_units: dict[str, str] = {}
        time_points: np.ndarray | None = None

        for ts in [dataset.cell_counts, dataset.cytokine_levels]:
            if ts is not None:
                if time_points is None:
                    time_points = ts.time_points
                all_values.update(ts.values)
                all_units.update(ts.units)

        if time_points is None or not all_values:
            return None

        return TimeSeriesData(
            time_points=time_points,
            values=all_values,
            units=all_units,
            metadata=dataset.metadata,
        )

    def _build_initial_state(
        self,
        dataset: ValidationDataset,
    ) -> ExtendedSDEState:
        """Построить начальное состояние SDE из данных.

        Использует initial_conditions из dataset если доступны,
        иначе — defaults из литературы.
        """
        ic = dataset.get_initial_conditions()

        # Defaults для переменных, которых нет в dataset
        defaults = {
            "P": 0.0,
            "Ne": 0.0,
            "M1": 0.0,
            "M2": 0.0,
            "F": 100.0,
            "Mf": 0.0,
            "E": 500.0,
            "S": 200.0,
            "C_TNF": 0.0,
            "C_IL10": 0.1,
            "C_PDGF": 0.0,
            "C_VEGF": 0.0,
            "C_TGFb": 0.5,
            "C_MCP1": 0.0,
            "C_IL8": 0.0,
            "rho_collagen": 0.0,
            "C_MMP": 0.0,
            "rho_fibrin": 0.9,
            "D": 1.0,
            "O2": 20.0,
        }

        # Merge: dataset IC overrides defaults
        state_dict = {**defaults}
        for key, value in ic.items():
            if key in defaults:
                state_dict[key] = value

        return ExtendedSDEState(**state_dict)

    def _prepare_params(self) -> ParameterSet:
        """Подготовить параметры для симуляции."""
        import dataclasses

        params_dict = dataclasses.asdict(self._params)
        params_dict["dt"] = self._config.dt
        params_dict["t_max"] = self._config.t_max
        return ParameterSet.from_dict(params_dict)

    def _run_monte_carlo(
        self,
        initial_state: ExtendedSDEState,
        params: ParameterSet,
    ) -> Any:
        """Запустить Monte Carlo для ансамблевых предсказаний."""
        from src.core.monte_carlo import MonteCarloConfig, MonteCarloSimulator
        from src.data.parameter_extraction import ModelParameters

        # Конвертация ExtendedSDEState → ModelParameters для MC
        model_params = ModelParameters(
            n0=initial_state.F + initial_state.Ne,
            stem_cell_fraction=initial_state.S
            / max(
                initial_state.F + initial_state.Ne + initial_state.S,
                1.0,
            ),
            macrophage_fraction=initial_state.M1
            / max(
                initial_state.F + initial_state.Ne + initial_state.M1,
                1.0,
            ),
            apoptotic_fraction=0.05,
            c0=initial_state.C_TNF + initial_state.C_IL10,
            inflammation_level=0.5,
        )

        mc_config = MonteCarloConfig(
            n_trajectories=self._config.n_mc_samples,
            extended_params=params,
            extended_initial_state=initial_state,
        )
        simulator = MonteCarloSimulator(config=mc_config)
        return simulator.run(model_params)

    def _get_phase_breakpoints(
        self,
        dataset_id: str,
    ) -> list[PhaseBreakpoint] | None:
        """Получить ожидаемые breakpoints для phase timing."""
        if dataset_id == "literature-xue2009":
            from src.data.literature_data import get_xue2009_phase_breakpoints

            raw = get_xue2009_phase_breakpoints()
            return [
                PhaseBreakpoint(
                    time_hours=float(bp["time_hours"]),
                    phase_before=str(bp["phase_before"]),
                    phase_after=str(bp["phase_after"]),
                    confidence=float(bp.get("confidence", 0.8)),
                )
                for bp in raw
            ]
        return None


def run_validation(
    dataset_id: str = "literature-xue2009",
    config: PipelineConfig | None = None,
    params: ParameterSet | None = None,
) -> ValidationReport:
    """Convenience функция для запуска валидации.

    Args:
        dataset_id: Идентификатор датасета.
        config: Конфигурация пайплайна.
        params: Параметры модели.

    Returns:
        ValidationReport с результатами.
    """
    pipeline = ValidationPipeline(config=config, params=params)
    return pipeline.run(dataset_id)
