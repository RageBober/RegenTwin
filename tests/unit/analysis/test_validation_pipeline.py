"""Тесты для модуля validation_pipeline.py.

Проверяет:
- PipelineConfig
- ValidationReport (to_dict, to_json, save)
- ValidationPipeline.run() end-to-end
- run_validation() convenience функция
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.analysis.validation import ValidationConfig
from src.analysis.validation_pipeline import (
    PipelineConfig,
    ValidationPipeline,
    ValidationReport,
    run_validation,
)

# =============================================================================
# PipelineConfig
# =============================================================================


class TestPipelineConfig:
    def test_default_values(self):
        cfg = PipelineConfig()
        assert cfg.t_max == 720.0
        assert cfg.dt == 0.1
        assert cfg.rng_seed == 42
        assert cfg.run_monte_carlo is False
        assert cfg.n_mc_samples == 50
        assert cfg.save_trajectory is False

    def test_custom_values(self):
        cfg = PipelineConfig(t_max=360.0, dt=0.5, rng_seed=123)
        assert cfg.t_max == 360.0
        assert cfg.dt == 0.5
        assert cfg.rng_seed == 123

    def test_validation_config_default(self):
        cfg = PipelineConfig()
        assert isinstance(cfg.validation_config, ValidationConfig)


# =============================================================================
# ValidationReport
# =============================================================================


class TestValidationReport:
    @pytest.fixture
    def report(self) -> ValidationReport:
        return ValidationReport(
            dataset_id="test-dataset",
            overall_score=0.75,
            elapsed_seconds=1.5,
            initial_conditions={"F": 100.0, "M1": 0.0},
        )

    def test_to_dict(self, report: ValidationReport):
        d = report.to_dict()
        assert d["dataset_id"] == "test-dataset"
        assert d["overall_score"] == 0.75
        assert d["elapsed_seconds"] == 1.5
        assert "errors" in d

    def test_to_json(self, report: ValidationReport):
        j = report.to_json()
        parsed = json.loads(j)
        assert parsed["dataset_id"] == "test-dataset"
        assert parsed["overall_score"] == 0.75

    def test_save(self, report: ValidationReport, tmp_path: Path):
        path = tmp_path / "report.json"
        report.save(path)
        assert path.exists()
        content = json.loads(path.read_text(encoding="utf-8"))
        assert content["dataset_id"] == "test-dataset"

    def test_empty_errors(self, report: ValidationReport):
        assert report.errors == []

    def test_report_with_errors(self):
        report = ValidationReport(
            dataset_id="bad",
            errors=["Something failed"],
        )
        d = report.to_dict()
        assert "Something failed" in d["errors"]


# =============================================================================
# ValidationPipeline end-to-end
# =============================================================================


class TestValidationPipelineEndToEnd:
    @pytest.fixture
    def fast_config(self) -> PipelineConfig:
        """Быстрая конфигурация для тестов."""
        return PipelineConfig(
            t_max=720.0,
            dt=1.0,  # грубый шаг для скорости
            rng_seed=42,
            run_monte_carlo=False,
        )

    def test_run_xue2009(self, fast_config: PipelineConfig):
        pipeline = ValidationPipeline(config=fast_config)
        report = pipeline.run("literature-xue2009")
        assert isinstance(report, ValidationReport)
        assert report.dataset_id == "literature-xue2009"
        assert len(report.errors) == 0, f"Errors: {report.errors}"
        assert report.overall_score >= 0.0
        assert report.elapsed_seconds > 0.0

    def test_run_xue2009_has_validation_result(self, fast_config: PipelineConfig):
        pipeline = ValidationPipeline(config=fast_config)
        report = pipeline.run("literature-xue2009")
        assert report.validation_result is not None

    def test_run_xue2009_has_initial_conditions(self, fast_config: PipelineConfig):
        pipeline = ValidationPipeline(config=fast_config)
        report = pipeline.run("literature-xue2009")
        assert len(report.initial_conditions) > 0

    def test_run_unknown_dataset(self, fast_config: PipelineConfig):
        pipeline = ValidationPipeline(config=fast_config)
        report = pipeline.run("nonexistent-dataset")
        assert len(report.errors) > 0

    def test_run_validation_convenience(self, fast_config: PipelineConfig):
        report = run_validation(
            dataset_id="literature-xue2009",
            config=fast_config,
        )
        assert isinstance(report, ValidationReport)
        assert report.dataset_id == "literature-xue2009"


# =============================================================================
# ValidationPipeline с HPA
# =============================================================================


class TestValidationPipelineHPA:
    def test_run_hpa_baseline(self):
        """HPA — одна точка, pipeline должен вернуть report без краша."""
        config = PipelineConfig(t_max=720.0, dt=1.0)
        pipeline = ValidationPipeline(config=config)
        report = pipeline.run("HPA-skin-baseline")
        assert isinstance(report, ValidationReport)
        assert report.dataset_id == "HPA-skin-baseline"
