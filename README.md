# RegenTwin

Программный инструмент для симуляции регенерации тканей с использованием данных flow cytometry и моделирования терапий PRP/PEMF.

## Обзор

RegenTwin — это десктопное/веб-приложение для:
- Загрузки и анализа данных flow cytometry (.fcs файлы, форматы 2.0/3.0/3.1)
- Моделирования регенерации тканей с использованием расширенной системы SDE (20 переменных) + ABM (6 типов клеток)
- Симуляции терапий PRP (Platelet-Rich Plasma) и PEMF (Pulsed Electromagnetic Field) с механистическими моделями
- Визуализации результатов (Plotly) и генерации PDF-отчётов
- Анализа чувствительности параметров (метод Sobol)

## Технологический стек

| Компонент | Технология |
|-----------|------------|
| Язык | Python 3.11+ |
| Пакетный менеджер | UV |
| Flow cytometry | FlowKit |
| Численные методы | NumPy, SciPy |
| Визуализация | Plotly, kaleido |
| PDF экспорт | fpdf2 |
| Backend | FastAPI + Uvicorn |
| Desktop frontend | Tauri + React + TypeScript |
| Streamlit интерфейс | frontend/app.py |
| База данных | SQLite (dev) → PostgreSQL (prod) |
| ORM / Миграции | SQLAlchemy + Alembic |
| Анализ чувствительности | SALib (Sobol) |

## Установка

### Требования

- Python 3.11+
- UV (пакетный менеджер)

### Шаги установки

```bash
# Клонирование репозитория
git clone https://github.com/RageBober/RegenTwin.git
cd RegenTwin

# Установка зависимостей через UV
uv sync

# Установка dev-зависимостей
uv sync --extra dev
```

## Запуск

### Быстрый старт (рекомендуется)

```bash
# Linux / macOS
bash setup_and_run.sh

# Windows
setup_and_run.bat
```

### Ручной запуск

```bash
# Backend API (порт 8000)
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Streamlit интерфейс (порт 8501)
streamlit run frontend/app.py

# Tauri desktop frontend (dev режим, требует Node.js)
cd ui && npm install && npm run dev
```

API будет доступен по адресу: http://localhost:8000
Swagger UI: http://localhost:8000/docs

## REST API

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/health` | Статус сервера |
| POST | `/api/upload/fcs` | Загрузка FCS файла |
| GET | `/api/upload/list` | Список загруженных файлов |
| POST | `/api/simulate` | Запуск симуляции (EXTENDED / MVP / ABM) |
| GET | `/api/simulate/{id}/status` | Статус симуляции |
| WS | `/api/simulate/{id}/ws` | WebSocket прогресс |
| GET | `/api/results/{id}` | Результаты симуляции |
| GET | `/api/results/` | Список всех симуляций |
| POST | `/api/analysis/sensitivity` | Анализ чувствительности Sobol |
| GET | `/api/analysis/{id}` | Результаты анализа |
| POST | `/api/viz/populations` | График популяций клеток |
| POST | `/api/viz/cytokines` | График цитокинов |
| POST | `/api/viz/ecm` | График ECM компонентов |
| POST | `/api/viz/phases` | График фаз заживления |
| POST | `/api/viz/spatial/heatmap` | ABM тепловая карта |
| POST | `/api/viz/spatial/scatter` | ABM диаграмма рассеяния |
| POST | `/api/viz/spatial/inflammation` | ABM воспалительная карта |
| POST | `/api/export/png` | Экспорт в PNG |
| POST | `/api/export/pdf` | Экспорт PDF-отчёта |

## Структура проекта

```
RegenTwin/
├── src/
│   ├── core/               # Математическое ядро
│   │   ├── sde_model.py    # ✔ MVP SDE (2 переменных)
│   │   ├── extended_sde.py # ✔ Расширенная SDE (20 переменных)
│   │   ├── abm_model.py    # ◐ ABM (6 типов клеток)
│   │   ├── integration.py  # ✔ SDE+ABM operator splitting
│   │   ├── monte_carlo.py  # ✔ Monte Carlo + параллелизация
│   │   ├── therapy_models.py # ✔ PRP/PEMF механистические модели
│   │   ├── wound_phases.py # ✔ Определение фаз заживления
│   │   └── parameters.py   # ✔ ParameterSet (105 полей)
│   ├── data/               # Data Pipeline
│   │   ├── fcs_parser.py   # ✔ Парсинг FCS 2.0/3.0/3.1
│   │   ├── gating.py       # ✔ Стратегии гейтирования клеток
│   │   ├── parameter_extraction.py # ✔ FCS → начальные условия модели
│   │   └── validation.py   # ✔ Валидация входных данных
│   ├── api/                # ✔ FastAPI Backend
│   │   ├── main.py         # App + CORS + Middleware
│   │   ├── config.py       # Конфигурация (хост, порт, CORS)
│   │   ├── models/schemas.py # Pydantic схемы валидации
│   │   ├── routes/         # 7 групп маршрутов
│   │   └── services/       # simulation, analysis, file сервисы
│   ├── visualization/      # ✔ Визуализация (Plotly)
│   │   ├── plots.py        # Популяции, цитокины, ECM, фазы
│   │   ├── spatial.py      # ABM тепловые карты, scatter
│   │   ├── export.py       # PNG/SVG/PDF экспорт
│   │   └── theme.py        # Цветовые схемы
│   └── db/                 # ✔ Database Layer
│       ├── models.py       # SQLAlchemy ORM модели
│       ├── session.py      # Управление сессиями
│       └── migrations/     # Alembic миграции
├── ui/                     # ◐ Tauri + React frontend
├── frontend/               # Streamlit интерфейс
│   └── app.py
├── tests/                  # 1680 тестов, 87% покрытие
│   ├── unit/               # Юнит-тесты по модулям
│   ├── integration/        # E2E тесты пайплайна
│   └── performance/        # Тесты производительности
├── scripts/                # Утилиты
│   ├── full_api_test.py    # E2E тест API (22/22)
│   ├── generate_test_fcs.py # Генерация тестовых FCS файлов
│   └── kill_port.py        # Освобождение порта 8000
├── Doks/                   # Документация
│   ├── RegenTwin_Mathematical_Framework.md
│   ├── RegenTwin_Update_Implemention_Plan.md
│   └── RegenTwin_Diploma_Report.md
├── Description/            # Описания функционала по фазам
├── data/                   # Данные (FCS файлы, результаты)
├── alembic.ini             # Конфигурация миграций
├── pyproject.toml          # Зависимости и конфигурация
├── setup_and_run.sh        # Быстрый запуск (Linux/macOS)
└── setup_and_run.bat       # Быстрый запуск (Windows)
```

## Тесты

```bash
# Запуск всех тестов
uv run pytest

# С отчётом покрытия
uv run pytest --cov=src --cov-report=term-missing

# Только быстрые юнит-тесты
uv run pytest tests/unit/ -m "not slow"

# E2E тест API (требует запущенного сервера)
python scripts/full_api_test.py
```

**Текущее состояние:** 1680 passed, 87% code coverage

## Математическая модель

### Расширенная SDE (20 переменных)

Состояние системы: `(P, Ne, M1, M2, F, Mf, E, S, C_TNF, C_IL10, C_PDGF, C_VEGF, C_TGFb, C_MCP1, C_IL8, ρ_collagen, C_MMP, ρ_fibrin, D, O₂)`

Уравнение Ланжевена для каждой переменной xᵢ:
```
dxᵢ = fᵢ(x, θ, t)dt + σᵢ(x)dWᵢ(t)
```

### ABM (6 типов агентов)

Агенты: стволовые клетки, макрофаги M1/M2, фибробласты, нейтрофилы, эндотелий

### Терапии

- **PRP** — двухфазная кинетика высвобождения факторов роста (PDGF, VEGF, TGF-β)
- **PEMF** — многопутевые эффекты: аденозиновый путь, Ca²⁺-CaM/NO, MAPK/ERK

## Документация

- [Математический фреймворк](Doks/RegenTwin_Mathematical_Framework.md)
- [Обновлённый план разработки](Doks/RegenTwin_Update_Implemention_Plan.md)
- [Дипломный отчёт](Doks/RegenTwin_Diploma_Report.md)
- [Описания функционала](Description/)

## Лицензия

MIT License

## Авторы

RegenTwin Team
