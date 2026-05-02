# RegenTwin

[![CI](https://github.com/RageBober/RegenTwin/actions/workflows/ci.yml/badge.svg)](https://github.com/RageBober/RegenTwin/actions/workflows/ci.yml)
[![Docs](https://github.com/RageBober/RegenTwin/actions/workflows/docs.yml/badge.svg)](https://github.com/RageBober/RegenTwin/actions/workflows/docs.yml)
[![codecov](https://codecov.io/gh/RageBober/RegenTwin/graph/badge.svg)](https://codecov.io/gh/RageBober/RegenTwin)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tauri 2](https://img.shields.io/badge/Tauri-2.0-24C8DB.svg)](https://v2.tauri.app/)

RegenTwin — инструмент для моделирования регенерации тканей на основе данных flow cytometry, расширенной SDE-модели, ABM и визуализации результатов через FastAPI + React/Tauri.

## Что сейчас поддерживается

### Режимы симуляции

| Режим | Статус | Что делает |
| --- | --- | --- |
| `extended` | поддерживается | 20-переменная SDE-модель, полный набор графиков и экспортов |
| `mvp` | поддерживается | упрощенная SDE-модель, результаты и CSV-экспорт |
| `abm` | поддерживается | агентная модель, population dynamics, spatial snapshots, CSV-экспорт |
| `integrated` | недоступен | UI скрыт, API возвращает `501 Not Implemented` |

### Аналитика

| Возможность | Статус | Примечание |
| --- | --- | --- |
| Sensitivity analysis | поддерживается | только Sobol |
| Morris | недоступен | больше не рекламируется как рабочий метод |
| Parameter estimation | недоступен | API возвращает `501 Not Implemented` |

### Экспорт результатов

| Источник | `csv` | `png` | `svg` | `pdf` |
| --- | --- | --- | --- | --- |
| `extended` result | да | да | да | да |
| `mvp` result | да | нет | нет | нет |
| `abm` result | да | нет | нет | нет |
| live visualization preview | да | да | нет | да |

## Ключевые изменения текущего состояния

- `upload_id` теперь реально влияет на симуляцию: upload-derived initial conditions сохраняются в metadata и применяются сервером как authoritative source.
- WebSocket статусы симуляции выдают корректные terminal events: `complete`, `cancelled`, `failed`, `not_found`.
- Results/export/viz стали mode-aware: `extended`, `mvp`, `abm` больше не смешиваются через ложную реконструкцию extended trajectory.
- Results page больше не показывает spatial tabs для не-ABM режимов.
- Desktop/dev запуск переведен на `uv`; Tauri launcher пытается `uv`, затем локальный `.venv`, затем fallback `python`.

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- Node.js 18+
- Rust toolchain для Tauri desktop build

## Установка

```bash
git clone https://github.com/RageBober/RegenTwin.git
cd RegenTwin
uv sync                       # ставит main + dev-group (PEP 735)
uv run pre-commit install     # включает pre-commit (ruff/mypy/eslint)
uv run alembic upgrade head   # создаёт data/regentwin.duckdb
cd ui
npm ci
```

> Минимальные требования: `uv >= 0.5` (поддержка PEP 735 dependency groups).
> Чтобы установить документационные зависимости: `uv sync --group docs`.

## Запуск

### Быстрый запуск

```bash
# Windows
setup_and_run.bat

# Linux / macOS
bash setup_and_run.sh
```

### Раздельный запуск

```bash
# backend
uv run uvicorn src.api.main:app --host 127.0.0.1 --port 8000

# frontend
cd ui
npm run dev
```

### Tauri desktop

```bash
cd ui
npm run tauri:dev
```

### Docker (опционально, backend-only)

API можно запустить в контейнере без Tauri-фронтенда — удобно для воспроизводимого
ревью кода или удалённого запуска.

```bash
docker build -t regentwin-api .
docker run -d -p 8000:8000 -v "$(pwd)/data:/app/data" --name regentwin regentwin-api
curl http://localhost:8000/api/v1/health
```

`data/regentwin.duckdb` пробрасывается через volume, чтобы выжить `docker rm`.
Frontend в Docker НЕ собирается — это отдельный Tauri-инсталлятор `.msi` (см. release workflow).

Tauri launcher ищет backend в таком порядке:
1. `uv run uvicorn ...`
2. project-local `.venv`
3. `python` в PATH

`cargo` target-dir для desktop сборки теперь локальный: `ui/src-tauri/target-tauri`.

## REST API

### Health

| Метод | Путь | Описание |
| --- | --- | --- |
| `GET` | `/api/v1/health` | Статус API |

### Upload

| Метод | Путь | Описание |
| --- | --- | --- |
| `POST` | `/api/v1/upload` | Загрузка файла |
| `GET` | `/api/v1/upload/{upload_id}` | Статус upload |

`POST /api/v1/upload` для `.fcs` пытается извлечь initial conditions и пишет их в `metadata.initial_conditions`.

### Simulation

| Метод | Путь | Описание |
| --- | --- | --- |
| `GET` | `/api/v1/simulations` | Список симуляций |
| `POST` | `/api/v1/simulate` | Старт симуляции |
| `GET` | `/api/v1/simulate/{simulation_id}` | Статус симуляции |
| `POST` | `/api/v1/simulate/{simulation_id}/cancel` | Кооперативная отмена |
| `WS` | `/api/v1/simulate/{simulation_id}/ws` | Прогресс и terminal events |

### Results and export

| Метод | Путь | Описание |
| --- | --- | --- |
| `GET` | `/api/v1/results/{simulation_id}` | Загруженные результаты completed run |
| `POST` | `/api/v1/export/{simulation_id}` | Export completed run |

### Analysis

| Метод | Путь | Описание |
| --- | --- | --- |
| `POST` | `/api/v1/analysis/sensitivity` | Sobol sensitivity analysis |
| `POST` | `/api/v1/analysis/estimation` | Сейчас всегда `501` |
| `GET` | `/api/v1/analysis/{analysis_id}` | Статус анализа |

### Visualization

| Метод | Путь | Описание |
| --- | --- | --- |
| `POST` | `/api/viz/populations` | Population plot |
| `POST` | `/api/viz/cytokines` | Cytokine plot |
| `POST` | `/api/viz/ecm` | ECM plot |
| `POST` | `/api/viz/phases` | Wound phases plot |
| `POST` | `/api/viz/comparison` | Сравнение сценариев терапии |
| `GET` | `/api/viz/from-result/{simulation_id}/populations` | Cached populations |
| `GET` | `/api/viz/from-result/{simulation_id}/cytokines` | Cached cytokines |
| `GET` | `/api/viz/from-result/{simulation_id}/ecm` | Cached ECM |
| `GET` | `/api/viz/from-result/{simulation_id}/phases` | Cached phases |
| `POST` | `/api/viz/export/csv` | CSV export из preview |
| `POST` | `/api/viz/export/png` | PNG export из preview |
| `POST` | `/api/viz/export/pdf` | PDF export из preview |
| `POST` | `/api/viz/spatial/heatmap` | ABM heatmap |
| `POST` | `/api/viz/spatial/scatter` | ABM scatter |
| `POST` | `/api/viz/spatial/inflammation` | ABM inflammation map |

## Пользовательские сценарии

### Upload -> simulate

1. Загрузить `.fcs` через `/api/v1/upload`.
2. В ответе получить `upload_id` и `metadata.initial_conditions`.
3. Передать `upload_id` в `/api/v1/simulate`.
4. Backend применит upload-derived initial conditions как server truth.

### Results -> spatial

- Для `abm` spatial endpoints могут использовать `simulation_id` и отрисовывать сохраненные snapshots completed run.
- Для `mvp` и `extended` spatial tabs в UI не отображаются.

## Phase 8: Интеграция и деплой

| Компонент | Команда |
|---|---|
| Линт + типы локально | `uv run pre-commit run --all-files` |
| Pytest + покрытие | `uv run pytest --cov=src --cov-report=term` |
| Бенчмарки на текущей машине | `uv run python scripts/benchmark.py --label "<имя>"` |
| Профили (py-spy/scalene) | `uv run python scripts/profile_hotspots.py` |
| Markdown-отчёт по бенчам | `uv run python scripts/generate_benchmark_report.py` |
| MkDocs preview | `uv sync --group docs && uv run mkdocs serve` |
| Docker backend | `docker build -t regentwin-api . && docker run -p 8000:8000 regentwin-api` |
| Tauri MSI локально | `cd ui && npm run tauri:build` |
| GitHub Release | `git tag v0.1.0 && git push origin v0.1.0` (триггерит `release.yml`) |

## Проверки, подтвержденные в текущем worktree

- `pytest -q`: `2358 passed`, coverage `87%`, `36 warnings`
- `ui`: `npm run lint` — проходит
- `ui`: `tsc -b --pretty false` — проходит
- `ui/src-tauri`: `cargo check` — проходит

## Что еще не доведено до green quality gates

- `mypy src` пока не проходит: остаются исторические ошибки типизации в `api`, `data`, `core`
- `ruff check src tests` пока не проходит: остается крупный backlog по style/static cleanup
- В test suite остаются `ResourceWarning: unclosed database`
- В `src/core/sde_numerics.py` и `src/core/robustness.py` сохраняются stubs, которые не выведены в активный product surface

## Структура репозитория

```text
RegenTwin/
├── src/                 # backend, scientific core, data pipeline, DB, visualization
├── ui/                  # React + TypeScript + Tauri frontend
├── tests/               # unit / integration / performance
├── scripts/             # утилиты для локальной разработки
├── Doks/                # проектная документация
├── data/                # runtime данные: uploads, results, sqlite
├── setup_and_run.bat
├── setup_and_run.sh
└── pyproject.toml
```

## Ограничения

- `integrated` режим намеренно выключен до полноценного подключения integration layer.
- `Morris` и parameter estimation намеренно не показаны как рабочие фичи.
- Для `abm`/`mvp` cached export ограничен CSV.
- Часть quality cleanup задач остается отдельным треком и не должна путаться с функциональной готовностью.
