"""SQLAlchemy ORM models для хранения записей симуляций, загрузок и анализов."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, ForeignKey, Index, JSON, String
from sqlalchemy.orm import DeclarativeBase


def _uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class SimulationRecord(Base):
    """Запись о запущенной симуляции."""

    __tablename__ = "simulations"
    __table_args__ = (
        Index("ix_simulations_status", "status"),
        Index("ix_simulations_created_at", "created_at"),
    )

    id = Column(String, primary_key=True, default=_uuid)
    mode = Column(String, nullable=False)
    status = Column(String, nullable=False, default="pending")
    progress = Column(Float, default=0.0)
    message = Column(String, nullable=True)
    params_json = Column(JSON, nullable=False)
    result_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=_utcnow)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(String, nullable=True)


class UploadRecord(Base):
    """Запись о загруженном файле."""

    __tablename__ = "uploads"
    __table_args__ = (
        Index("ix_uploads_status", "status"),
        Index("ix_uploads_created_at", "created_at"),
    )

    id = Column(String, primary_key=True, default=_uuid)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    status = Column(String, default="processing")
    description = Column(String, nullable=True)
    created_at = Column(DateTime, default=_utcnow)
    metadata_json = Column(JSON, nullable=True)


class AnalysisRecord(Base):
    """Запись о запущенном анализе (sensitivity / estimation)."""

    __tablename__ = "analyses"
    __table_args__ = (
        Index("ix_analyses_status", "status"),
        Index("ix_analyses_created_at", "created_at"),
        Index("ix_analyses_simulation_id", "simulation_id"),
    )

    id = Column(String, primary_key=True, default=_uuid)
    analysis_type = Column(String, nullable=False)
    status = Column(String, default="pending")
    progress = Column(Float, default=0.0)
    params_json = Column(JSON, nullable=False)
    result_json = Column(JSON, nullable=True)
    simulation_id = Column(String, ForeignKey("simulations.id"), nullable=True)
    created_at = Column(DateTime, default=_utcnow)
    completed_at = Column(DateTime, nullable=True)
