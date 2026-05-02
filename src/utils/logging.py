"""Loguru configuration for RegenTwin.

Sinks:
- stderr: human-readable (или JSON при ``log_serialize_console=True``)
- logs/regentwin.log: DEBUG+ human-readable, rotation 10 MB / 7 days
- logs/regentwin.jsonl: INFO+ structured JSON, rotation 50 MB / 14 days (опц.)
- logs/errors.log: ERROR+ с backtrace/diagnose, retention 30 days

Stdlib logging (uvicorn, fastapi, sqlalchemy, celery) перехватывается через
:class:`InterceptHandler` и направляется в тот же loguru-пайплайн.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from loguru import logger

from src.api.config import settings

_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{process.name}:{thread.name}</cyan> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)
_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
    "{process.name}:{thread.name} | {name}:{function}:{line} | {message}"
)

_INTERCEPTED_LOGGERS = (
    "uvicorn",
    "uvicorn.error",
    "fastapi",
    "sqlalchemy.engine",
    "celery",
    "celery.task",
    "celery.worker",
)

_configured = False


class InterceptHandler(logging.Handler):
    """Route stdlib logging records into loguru preserving level and frame."""

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - thin shim
        """Translate a stdlib record into a loguru call."""
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def _resolve_log_dir() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    log_dir = Path(settings.log_dir)
    if not log_dir.is_absolute():
        log_dir = project_root / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def setup_logging(force: bool = False) -> None:
    """Configure loguru sinks and intercept stdlib logging.

    Idempotent: повторные вызовы без ``force=True`` игнорируются, что позволяет
    безопасно вызывать из FastAPI lifespan и Celery worker_process_init.
    """
    global _configured
    if _configured and not force:
        return

    logger.remove()
    log_dir = _resolve_log_dir()

    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=_CONSOLE_FORMAT,
        serialize=settings.log_serialize_console,
        backtrace=True,
        diagnose=True,
        colorize=not settings.log_serialize_console,
    )

    logger.add(
        log_dir / "regentwin.log",
        level="DEBUG",
        format=_FILE_FORMAT,
        rotation="10 MB",
        retention="7 days",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )

    if settings.log_json:
        logger.add(
            log_dir / "regentwin.jsonl",
            level="INFO",
            rotation="50 MB",
            retention="14 days",
            enqueue=True,
            serialize=True,
        )

    logger.add(
        log_dir / "errors.log",
        level="ERROR",
        format=_FILE_FORMAT,
        rotation="10 MB",
        retention="30 days",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )

    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    for name in _INTERCEPTED_LOGGERS:
        stdlib_logger = logging.getLogger(name)
        stdlib_logger.handlers = [InterceptHandler()]
        stdlib_logger.propagate = False

    # Access logs are already covered by our request middleware.
    uvicorn_access = logging.getLogger("uvicorn.access")
    uvicorn_access.handlers = []
    uvicorn_access.propagate = False
    uvicorn_access.disabled = True

    _configured = True
