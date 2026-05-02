# CLAUDE.md — Инструкции для Claude Code

## Контекст проекта

RegenTwin — мультимасштабная платформа симуляции регенерации тканей.
- **Backend:** Python 3.11+, FastAPI, NumPy, SciPy, ~32K LOC
- **Frontend:** Tauri 2.0 (Rust shell) + React 19 / TypeScript (ui/)
- **Math core:** 20+ связанных SDE + Agent-Based Model + Monte Carlo
- **DB:** SQLite через SQLAlchemy + Alembic migrations

## КРИТИЧНО: Проект сломан

Проект не работает при запуске. Качество реализации под вопросом.
Твоя задача — найти ВСЕ проблемы и исправить их.

## Шаг 1: Запусти диагностику

```bash
cd C:\Users\dzume\OneDrive\Документы\projects\RegenTwin
python scripts/diagnose.py
```

Этот скрипт НЕ запускает юнит-тесты. Он РЕАЛЬНО ИСПОЛЬЗУЕТ проект:
- Импортирует ВСЕ 35+ модулей
- Запускает SDE симуляцию (MVP 2-var и Extended 20-var на 720 часов)
- Запускает ABM, интеграцию SDE+ABM, Monte Carlo
- Проверяет биологическую корректность (M1/M2 ratio)
- Генерирует визуализации через Plotly
- Экспортирует CSV
- Создаёт таблицы в БД, делает CRUD
- Бьёт по КАЖДОМУ FastAPI endpoint через TestClient:
  - health, simulate (4 режима), results, simulations list
  - visualization API (5 endpoint-ов)
  - sensitivity analysis API
- Проверяет файловую структуру

Результат: `output/diagnosis_report.json` с каждым тестом и его статусом.

## Шаг 2: Исправь ВСЕ FAIL-ы

Прочитай diagnosis_report.json. Для каждого FAIL:
1. Определи корневую причину (import error? runtime crash? API 500? schema mismatch?)
2. Найди файл с проблемой
3. Исправь МИНИМАЛЬНО — не переписывай модули с нуля

## Шаг 3: Проверь API вручную

```bash
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Swagger: http://127.0.0.1:8000/docs
Попробуй полный цикл: simulate → poll status → get results

## Шаг 4: Проверь фронтенд

```bash
cd ui
npm install
npm run dev
```

http://localhost:5173 — должен загрузиться, показать sidebar, dashboard.

## Шаг 5: Перезапусти диагностику

```bash
python scripts/diagnose.py
```

Повторяй Шаги 2-5 пока все FAIL не станут OK.

## Приоритет починки

1. **Import errors** — если модуль не импортируется, всё остальное в нём не работает
2. **Core runtime** — SDE/ABM/Monte Carlo должны запускаться
3. **API endpoints** — simulate, results, viz должны возвращать 200
4. **DB** — таблицы должны создаваться, CRUD работать
5. **Visualization** — Plotly фигуры должны генерироваться
6. **Frontend** — только после того как бэкенд работает

## Правила

- НЕ переписывай модули с нуля — чини точечно
- НЕ трогай математические уравнения (extended_sde.py, therapy_models.py) без явной причины
- Создавай missing dirs (data/uploads/, data/results/) если их нет
- После КАЖДОГО исправления перезапусти diagnose.py
- Если зависимость не установлена: `pip install -e ".[dev]"`

## Структура проекта

```
src/
├── core/           # Мат. ядро (SDE, ABM, Monte Carlo, therapies)
├── data/           # FCS парсинг, гейтинг
├── api/            # FastAPI (routes/, services/, models/)
├── visualization/  # Plotly графики, export
├── db/             # SQLAlchemy модели, sessions
└── utils/          # (не реализовано)

ui/                 # Tauri + React frontend
├── src-tauri/      # Rust shell
└── src/            # React/TypeScript
    ├── routes/     # 6 страниц
    ├── components/ # 16+ компонентов
    ├── hooks/      # 6 hooks (useSimulation, useVisualization, etc.)
    ├── stores/     # Zustand (simulationStore, uiStore)
    └── types/      # api.ts — типы для всех endpoints

tests/              # pytest (2000+ тестов)
```
