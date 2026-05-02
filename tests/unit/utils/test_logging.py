"""Tests for src.utils.logging."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pytest
from loguru import logger

import src.utils.logging as logging_module
from src.api.config import settings


@pytest.fixture
def isolated_log_dir(tmp_path, monkeypatch):
    """Point settings.log_dir to a temp dir and reset setup flag."""
    monkeypatch.setattr(settings, "log_dir", str(tmp_path))
    monkeypatch.setattr(logging_module, "_configured", False)
    yield tmp_path
    logger.remove()
    monkeypatch.setattr(logging_module, "_configured", False)


def _flush_sinks() -> None:
    # enqueue=True пишет в файл через отдельный поток — даём ему время
    logger.complete()
    time.sleep(0.05)


def test_setup_logging_creates_log_dir(isolated_log_dir: Path) -> None:
    target = isolated_log_dir / "nested"
    # Если путь не существует, setup_logging должна его создать
    settings.log_dir = str(target)
    logging_module.setup_logging(force=True)
    assert target.exists() and target.is_dir()


def test_setup_logging_idempotent(isolated_log_dir: Path) -> None:
    logging_module.setup_logging()
    handler_count_after_first = len(logger._core.handlers)  # type: ignore[attr-defined]
    logging_module.setup_logging()
    handler_count_after_second = len(logger._core.handlers)  # type: ignore[attr-defined]
    assert handler_count_after_first == handler_count_after_second


def test_setup_logging_force_reconfigures(isolated_log_dir: Path) -> None:
    logging_module.setup_logging()
    first = len(logger._core.handlers)  # type: ignore[attr-defined]
    logging_module.setup_logging(force=True)
    second = len(logger._core.handlers)  # type: ignore[attr-defined]
    assert first == second  # не должно накапливаться даже при force


def test_stdlib_intercept_routes_to_loguru(isolated_log_dir: Path) -> None:
    captured: list[str] = []
    logging_module.setup_logging(force=True)
    sink_id = logger.add(lambda msg: captured.append(str(msg)), level="DEBUG")
    try:
        logging.getLogger("uvicorn.error").warning("intercepted-uvicorn")
        _flush_sinks()
    finally:
        logger.remove(sink_id)
    assert any("intercepted-uvicorn" in line for line in captured)


def test_jsonl_sink_writes_valid_json(isolated_log_dir: Path) -> None:
    logging_module.setup_logging(force=True)
    logger.bind(scenario="test").info("structured-probe")
    _flush_sinks()

    jsonl = isolated_log_dir / "regentwin.jsonl"
    assert jsonl.exists(), f"JSONL sink not created in {isolated_log_dir}"
    lines = [ln for ln in jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert lines, "JSONL sink is empty"
    payload = json.loads(lines[-1])
    assert payload["record"]["level"]["name"] == "INFO"
    assert payload["record"]["message"] == "structured-probe"
    assert payload["record"]["extra"].get("scenario") == "test"


def test_error_sink_isolated_from_info(isolated_log_dir: Path) -> None:
    logging_module.setup_logging(force=True)
    logger.info("info-noise")
    logger.error("error-signal")
    _flush_sinks()

    errors_log = isolated_log_dir / "errors.log"
    assert errors_log.exists()
    content = errors_log.read_text(encoding="utf-8")
    assert "error-signal" in content
    assert "info-noise" not in content


def test_jsonl_disabled_when_flag_false(isolated_log_dir: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "log_json", False)
    logging_module.setup_logging(force=True)
    logger.info("no-jsonl")
    _flush_sinks()
    assert not (isolated_log_dir / "regentwin.jsonl").exists()
