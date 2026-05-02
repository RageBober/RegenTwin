"""Database session management.

Поддерживаются два диалекта:
- DuckDB (`duckdb://...`) — embedded колоночная БД, default для приложения.
- SQLite (`sqlite://...`) — legacy режим для совместимости со старыми тестами.

Параметры engine выбираются по диалекту через `_engine_kwargs_for`.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from src.api.config import settings
from src.db.models import Base


def _detect_dialect(url: str) -> str:
    """Возвращает имя диалекта (duckdb / sqlite / postgresql / ...) из URL."""
    scheme = url.split(":", 1)[0]
    return scheme.split("+", 1)[0]


def _engine_kwargs_for(dialect: str) -> dict[str, Any]:
    """Параметры create_engine, специфичные для диалекта."""
    if dialect == "sqlite":
        return {
            "connect_args": {"check_same_thread": False},
            "pool_size": 1,
            "max_overflow": 2,
            "pool_timeout": 60,
        }
    if dialect == "duckdb":
        # DuckDB single-writer; pool > 1 безопасен для read, write идёт через SessionLocal
        return {
            "connect_args": {},
            "pool_size": 5,
            "max_overflow": 10,
            "pool_timeout": 30,
        }
    return {"pool_size": 5, "max_overflow": 10, "pool_timeout": 30}


_dialect = _detect_dialect(settings.database_url)
engine = create_engine(settings.database_url, echo=False, **_engine_kwargs_for(_dialect))


if _dialect == "sqlite":

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):  # type: ignore[no-untyped-def]
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False)


def create_tables() -> None:
    """Создать все таблицы (для dev/startup)."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency для получения DB session."""
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
