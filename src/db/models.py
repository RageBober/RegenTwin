"""SQLAlchemy ORM models для хранения записей симуляций, загрузок и анализов."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Index, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class SimulationRecord(Base):
    """Запись о запущенной симуляции."""

    __tablename__ = "simulations"
    __table_args__ = (
        Index("ix_simulations_status", "status"),
        Index("ix_simulations_created_at", "created_at"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    mode: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    message: Mapped[str | None] = mapped_column(String, nullable=True)
    params_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    result_path: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)


class UploadRecord(Base):
    """Запись о загруженном файле."""

    __tablename__ = "uploads"
    __table_args__ = (
        Index("ix_uploads_status", "status"),
        Index("ix_uploads_created_at", "created_at"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    file_path: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, default="processing")
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class AnalysisRecord(Base):
    """Запись о запущенном анализе (sensitivity / estimation)."""

    __tablename__ = "analyses"
    __table_args__ = (
        Index("ix_analyses_status", "status"),
        Index("ix_analyses_created_at", "created_at"),
        Index("ix_analyses_simulation_id", "simulation_id"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    analysis_type: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, default="pending")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    params_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    result_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    simulation_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("simulations.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
