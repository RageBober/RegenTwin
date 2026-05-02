# Changelog

Все значимые изменения проекта документируются в этом файле.

Формат: [Keep a Changelog](https://keepachangelog.com/ru/1.1.0/),
версионирование: [Semantic Versioning](https://semver.org/lang/ru/).

## [Unreleased]

### Added
- Phase 8: Интеграция и деплой
  - Миграция БД на DuckDB (embedded, колоночный)
  - Переход на uv (PEP 735 dependency groups)
  - Pre-commit hooks (ruff, mypy, eslint)
  - Бенчмаркинг: pytest-benchmark + py-spy + scalene + auto-генерация Markdown отчёта
  - Health Check endpoints: `/health`, `/health/live`, `/health/ready` с проверками БД/Celery/Redis
  - GitHub Actions CI (lint + test + frontend build)
  - GitHub Actions Release (Tauri MSI auto-build по тегу)
  - Dockerfile (multi-stage, backend-only)
  - MkDocs Material документация (RU/EN, mkdocstrings, glightbox)

## [0.1.0] — 2026-04-30 (планируется)

### Added
- Phase 1-3: Mathematical core (Extended SDE, ABM, Monte Carlo, Sobol, validation)
- FastAPI backend с DuckDB
- React 19 + Tauri 2 frontend
- Phase 3.4: Validation metrics (DTW, CRPS, PPC, Changepoint, Kendall)
- Phase 3.6: Analysis визуализация (Sobol/Morris/Posterior/Convergence)
