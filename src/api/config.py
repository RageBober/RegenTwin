"""Конфигурация API через переменные окружения."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки приложения. Переопределяются через REGENTWIN_* env vars."""

    model_config = SettingsConfigDict(env_prefix="REGENTWIN_")

    host: str = "127.0.0.1"
    port: int = 8000
    database_url: str = "duckdb:///data/regentwin.duckdb"
    upload_dir: str = "data/uploads"
    results_dir: str = "data/results"
    cors_origins: list[str] = Field(
        default=[
            "http://localhost:1420",
            "http://localhost:3000",
            "http://localhost:5173",
            "http://localhost:5174",
            "http://localhost:5175",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:5174",
            "http://127.0.0.1:5175",
            "https://tauri.localhost",
            "http://tauri.localhost",
            "tauri://localhost",
        ],
        description="Override via REGENTWIN_CORS_ORIGINS for production",
    )
    log_level: str = "INFO"
    log_dir: str = "logs"
    log_json: bool = True
    log_serialize_console: bool = False
    max_upload_bytes: int = 500 * 1024 * 1024  # 500 MB
    simulation_timeout: int = 3600  # seconds
    app_version: str = "0.1.0"

    # Celery — при use_celery=False (по умолчанию) используется asyncio fallback
    use_celery: bool = False
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"


settings = Settings()
