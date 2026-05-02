"""Helper-функции для проверки здоровья зависимостей API.

Каждая проверка возвращает `HealthCheckDetail`:
- `ok` — зависимость работает
- `unhealthy` — зависимость недоступна
- `skipped` — проверка не применима к текущей конфигурации
- `degraded` — зависимость работает, но с замедлением (зарезервировано)

Все sync-операции оборачиваются через `asyncio.to_thread`, чтобы не блокировать
event loop FastAPI.
"""

from __future__ import annotations

import asyncio
import time

from sqlalchemy import text

from src.api.config import settings
from src.api.models.schemas import HealthCheckDetail
from src.db.session import engine

DB_TIMEOUT_SEC: float = 2.0
CELERY_TIMEOUT_SEC: float = 1.0
REDIS_TIMEOUT_SEC: float = 1.0


def _measure_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 2)


async def check_db() -> HealthCheckDetail:
    """`SELECT 1` через текущий SQLAlchemy engine."""

    def _sync_select_one() -> None:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

    started = time.perf_counter()
    try:
        await asyncio.wait_for(asyncio.to_thread(_sync_select_one), timeout=DB_TIMEOUT_SEC)
        return HealthCheckDetail(status="ok", latency_ms=_measure_ms(started))
    except TimeoutError:
        return HealthCheckDetail(
            status="unhealthy",
            latency_ms=_measure_ms(started),
            message=f"DB SELECT 1 timed out after {DB_TIMEOUT_SEC}s",
        )
    except Exception as exc:  # pragma: no cover — путь через mock в тестах
        return HealthCheckDetail(
            status="unhealthy",
            latency_ms=_measure_ms(started),
            message=f"{type(exc).__name__}: {exc}",
        )


async def check_celery() -> HealthCheckDetail:
    """`celery_app.control.ping()` если Celery включён."""
    if not settings.use_celery:
        return HealthCheckDetail(status="skipped", message="use_celery=False")

    started = time.perf_counter()
    try:
        from src.tasks.celery_app import celery_app  # ленивый импорт
    except Exception as exc:
        return HealthCheckDetail(
            status="unhealthy",
            latency_ms=_measure_ms(started),
            message=f"Celery import failed: {exc}",
        )

    def _ping() -> list[dict[str, str]]:
        return celery_app.control.ping(timeout=CELERY_TIMEOUT_SEC) or []

    try:
        replies = await asyncio.wait_for(asyncio.to_thread(_ping), timeout=CELERY_TIMEOUT_SEC + 0.5)
    except TimeoutError:
        return HealthCheckDetail(
            status="unhealthy",
            latency_ms=_measure_ms(started),
            message="celery ping timed out",
        )
    except Exception as exc:
        return HealthCheckDetail(
            status="unhealthy",
            latency_ms=_measure_ms(started),
            message=f"{type(exc).__name__}: {exc}",
        )

    if not replies:
        return HealthCheckDetail(
            status="unhealthy",
            latency_ms=_measure_ms(started),
            message="no celery workers responded",
        )
    return HealthCheckDetail(
        status="ok",
        latency_ms=_measure_ms(started),
        message=f"{len(replies)} worker(s) online",
    )


async def check_redis() -> HealthCheckDetail:
    """`redis.from_url(...).ping()` если Celery (значит и Redis) включён."""
    if not settings.use_celery:
        return HealthCheckDetail(status="skipped", message="use_celery=False")

    started = time.perf_counter()
    try:
        import redis
    except Exception as exc:
        return HealthCheckDetail(
            status="unhealthy",
            latency_ms=_measure_ms(started),
            message=f"redis import failed: {exc}",
        )

    def _ping() -> bool:
        client = redis.from_url(settings.celery_broker_url, socket_timeout=REDIS_TIMEOUT_SEC)
        return bool(client.ping())

    try:
        ok = await asyncio.wait_for(asyncio.to_thread(_ping), timeout=REDIS_TIMEOUT_SEC + 0.5)
    except TimeoutError:
        return HealthCheckDetail(
            status="unhealthy",
            latency_ms=_measure_ms(started),
            message="redis PING timed out",
        )
    except Exception as exc:
        return HealthCheckDetail(
            status="unhealthy",
            latency_ms=_measure_ms(started),
            message=f"{type(exc).__name__}: {exc}",
        )

    return HealthCheckDetail(
        status="ok" if ok else "unhealthy",
        latency_ms=_measure_ms(started),
    )


def aggregate_status(checks: dict[str, HealthCheckDetail]) -> str:
    """Агрегированный статус по dict проверок.

    - Любая `unhealthy` → общий `unhealthy`.
    - Все `skipped` или часть `skipped` + остальные `ok` → `ok`.
    - Любая `degraded` без unhealthy → `degraded`.
    """
    statuses = {check.status for check in checks.values()}
    if "unhealthy" in statuses:
        return "unhealthy"
    if "degraded" in statuses:
        return "degraded"
    return "ok"
