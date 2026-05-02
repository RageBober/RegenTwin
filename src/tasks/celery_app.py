"""Celery application factory for RegenTwin background tasks."""

from __future__ import annotations

from celery import Celery
from celery.signals import worker_process_init

from src.api.config import settings

celery_app = Celery(
    "regentwin",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    result_expires=3600,
    worker_hijack_root_logger=False,
)


@worker_process_init.connect
def _init_worker_logging(**_: object) -> None:
    """Attach loguru sinks and stdlib interceptor to each Celery worker process."""
    from src.utils.logging import setup_logging

    setup_logging()
