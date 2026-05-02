# RegenTwin

**RegenTwin** — мультимасштабная платформа симуляции регенерации тканей с поддержкой
терапий PRP (platelet-rich plasma) и PEMF (pulsed electromagnetic fields).

## Что это такое

Цифровой двойник процесса заживления раны, объединяющий:

- **20-переменную SDE-модель** клеточно-цитокиновой динамики (Phase 2 Mathematical Framework)
- **Agent-Based модель** пространственного поведения клеток
- **Monte Carlo** ансамблевые прогоны (с параллелизацией)
- **Sensitivity analysis** (Sobol)
- **Parameter estimation** (PyMC / emcee)
- Импорт реальных данных flow cytometry (`.fcs`) → начальные условия

## Стек

| Слой | Технологии |
|---|---|
| Math core | NumPy, SciPy, PyMC, emcee, SALib |
| Backend  | FastAPI, SQLAlchemy 2.0, Alembic, Celery (опционально), DuckDB |
| Frontend | React 19 + TypeScript, Tauri 2.0, Vite, Plotly, Three.js |
| DevOps   | uv, ruff, mypy, pytest, pytest-benchmark, MkDocs Material, GitHub Actions |

## Куда дальше

- **[Установка](getting-started/installation.md)** — как поднять проект локально
- **[Первая симуляция](getting-started/first-simulation.md)** — запустить SDE через CLI / API
- **[Архитектура](architecture/overview.md)** — обзор слоёв и потока данных
- **[Туториалы](tutorials/run-simulation.md)** — практические гайды
- **[API Reference](api-reference/index.md)** — автоматическая документация модулей
- **[Бенчмарки](benchmarks/index.md)** — производительность на разном железе

!!! note "Дипломный проект"
    RegenTwin разработан как дипломный проект. Документация ориентирована на
    научного руководителя, рецензентов и потенциальных продолжателей разработки.
