# Changelog

Все значимые изменения проекта документируются в этом файле.

Формат: [Keep a Changelog](https://keepachangelog.com/ru/1.1.0/),
версионирование: [Semantic Versioning](https://semver.org/lang/ru/).

## [Unreleased]

### Added — Phase 8 (Интеграция и деплой)
- Миграция БД на DuckDB (embedded, колоночный, нативный JSON).
- Переход на uv (PEP 735 dependency groups: `dev`, `docs`).
- Pre-commit hooks: ruff, ruff-format, mypy (pre-push), eslint (pre-push).
- Бенчмаркинг: pytest-benchmark + py-spy + scalene + auto-генерация Markdown отчёта
  с matplotlib-графиками для сравнения «laptop vs powerful CPU».
- Health Check endpoints: `/api/v1/health`, `/health/live`, `/health/ready` с
  проверками БД/Celery/Redis и aggregated статусом (ok/degraded/unhealthy).
- GitHub Actions CI (`ci.yml`): ruff + mypy + pytest matrix (3.11, 3.12) +
  ESLint + Vitest + frontend build, Codecov upload.
- GitHub Actions Release (`release.yml`): по push'у тега `v*.*.*` собирает Tauri
  `.msi` и `.exe` под Windows, публикует GitHub Release.
- GitHub Actions Docs (`docs.yml`): MkDocs Material → gh-pages автодеплой.
- Dockerfile (multi-stage, backend-only) + `.dockerignore`.
- MkDocs полная документация (RU/EN, mkdocstrings auto API ref, glightbox).
- Tauri bundle: `.venv` бандлится в `resources/` (приложение-инсталлятор без
  необходимости установки Python пользователем).

### Changed
- `src/db/session.py`: ветвление по диалекту (DuckDB / SQLite), убран hardcoded SQLite-PRAGMA.
- `alembic.ini`: URL читается из `settings.database_url` через `env.py`.

## [0.1.0] — 2026-04-30 (планируется)

### Added
- Phase 1-3: Mathematical core (Extended SDE 20-var, ABM, Monte Carlo, Sobol, validation metrics).
- FastAPI backend с DuckDB.
- React 19 + Tauri 2 frontend.
- Phase 3.4: Validation metrics (DTW, CRPS, PPC, Changepoint, Kendall).
- Phase 3.6: Analysis визуализация (Sobol/Morris/Posterior/Convergence).
