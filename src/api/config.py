"""Конфигурация API через переменные окружения."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки приложения. Переопределяются через REGENTWIN_* env vars."""

    model_config = SettingsConfigDict(env_prefix="REGENTWIN_")

    host: str = "127.0.0.1"
    port: int = 8000
    database_url: str = "sqlite:///data/regentwin.db"
    upload_dir: str = "data/uploads"
    results_dir: str = "data/results"
    cors_origins: list[str] = [
        "http://localhost:1420",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5175",
        "https://tauri.localhost",
    ]
    log_level: str = "INFO"
    max_upload_bytes: int = 500 * 1024 * 1024  # 500 MB
    simulation_timeout: int = 3600  # seconds
    app_version: str = "0.1.0"


settings = Settings()
