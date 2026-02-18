# План реализации RegenTwin MVP

## Обзор проекта

**RegenTwin** — веб/десктоп-приложение для симуляции регенерации тканей с использованием:
- Данных flow cytometry (.fcs файлы)
- Моделирования терапий PRP (Platelet-Rich Plasma) и PEMF (Pulsed Electromagnetic Field)
- Математического ядра: SDE + ABM

**Стек:**
- **Backend:** Python 3.11+, FastAPI, NumPy, SciPy, FlowKit
- **Frontend:** Tauri + React + TypeScript
- **Визуализация:** Plotly.js, D3.js, Three.js (3D)

---

## Архитектура

```
┌─────────────────────────────────────────────────┐
│      Tauri + React (Desktop/Web)                │
│  - TypeScript, Plotly.js, D3.js                 │
│  - Компоненты: Upload, Parameters, Results      │
└────────────────────┬────────────────────────────┘
                     │ HTTP / WebSocket
                     ↓
┌─────────────────────────────────────────────────┐
│      FastAPI Backend (localhost:8000)           │
│  - REST API для симуляций                       │
│  - WebSocket для прогресса                      │
└────────────────────┬────────────────────────────┘
                     │ Python imports
                     ↓
┌─────────────────────────────────────────────────┐
│      Mathematical Core (src/)                   │
│  - src/core/ (SDE, ABM) ◐ 85%                   │
│  - src/data/ (FCS парсинг) ◐ 95%                │
└─────────────────────────────────────────────────┘
```

---

## Структура проекта

```
RegenTwin/
├── src/                        # Python backend
│   ├── core/                   # ◐ Математическое ядро (85%)
│   │   ├── sde_model.py        # ✔ Реализовано
│   │   ├── abm_model.py        # ◐ Частично (нет хемотаксиса)
│   │   ├── integration.py      # ◐ Частично (базовая связь)
│   │   └── monte_carlo.py      # ◐ Частично (нет параллелизации)
│   ├── data/                   # ◐ FCS парсинг (95%)
│   │   ├── fcs_parser.py       # ✔ Реализовано
│   │   ├── gating.py           # ✔ Реализовано
│   │   └── parameter_extraction.py  # ✔ Реализовано
│   ├── api/                    # ✖ FastAPI endpoints (только структура)
│   │   ├── routes/             # ✖
│   │   └── models/             # ✖
│   ├── visualization/          # ✖ Не реализовано
│   └── utils/                  # ✔
│
├── ui/                         # ✖ Tauri + React frontend (НЕ СОЗДАНО)
│   ├── src-tauri/              # Tauri (Rust)
│   └── src/                    # React (TypeScript)
│
├── tests/                      # ✔ Тесты (2100+ строк)
│   ├── unit/                   # ✔ Unit-тесты
│   ├── integration/            # ✔ Integration-тесты
│   └── performance/            # ✔ Performance-тесты
│
├── Description/                # ✔ Описания функционала (9 файлов)
├── Doks/                       # ✔ Документация проекта
├── data/mock/                  # ✔ Мок-данные
├── docker/                     # ✖ Пустая директория
├── pyproject.toml              # ✔
├── ruff.toml                   # ✔
└── .gitignore                  # ✔
```

---

## Условные обозначения

| Символ | Значение |
|--------|----------|
| ✔ | Полностью реализовано |
| ◐ | Частично реализовано |
| ✖ | Не реализовано |

---

## Фаза 0: Инфраструктура ◐ ЧАСТИЧНО (70%)

| Задача | Статус | Примечание |
|--------|--------|------------|
| pyproject.toml | ✔ | Все зависимости настроены |
| Структура директорий | ✔ | src/, tests/, data/ |
| __init__.py файлы | ✔ | Созданы |
| .gitignore, ruff.toml | ✔ | Настроены |
| Мок-данные | ✔ | generate_mock_data.py |
| **Docker настройка** | ✖ | Папка docker/ пуста |
| **CI/CD (GitHub Actions)** | ✖ | Нет workflow файлов |
| **Реальные датасеты FlowRepository** | ✖ | Не загружены |

---

## Фаза 1: Data Pipeline ✔ РЕАЛИЗОВАННО  (100%)

| Файл | Классы/Функции | Статус | LOC |
|------|----------------|--------|-----|
| `src/data/fcs_parser.py` | `FCSLoader`, `FCSMetadata`, `load_fcs()` | ✔ | 235 |
| `src/data/gating.py` | `GatingStrategy`, `GateResult`, `GatingResults` | ✔ | 460 |
| `src/data/parameter_extraction.py` | `ParameterExtractor`, `ModelParameters` | ✔ | 295 |
| `Description/Phase1/*.md` | Описания | ✔ | 3 файла |

### Недостающий функционал:

| Задача | Статус | Описание |
|--------|--------|----------|
| **Поддержка изображений (PNG/JPEG)** | ✔ | Загрузка scatter plots |
| **Базовый анализ изображений** | ✔ | Опционально для MVP |

#### Предложения по улучшению


---

## Фаза 2: Математическое ядро ◐ ЧАСТИЧНО (85%)

| Файл | Код | Description | Статус |
|------|-----|-------------|--------|
| `src/core/sde_model.py` | ✔ | ✔ | Полностью |
| `src/core/abm_model.py` | ◐ | ✔ | Частично |
| `src/core/integration.py` | ◐ | ✔ | Частично |
| `src/core/monte_carlo.py` | ◐ | ✔ | Частично |

### SDE модель ✔ ГОТОВО

| Компонент | Статус | Описание |
|-----------|--------|----------|
| Уравнение Ланжевена | ✔ | dNₜ = [rNₜ(1 - Nₜ/K) + αf(PRP) + βg(PEMF)]dt + σNₜdWₜ |
| Метод Эйлера-Маруямы | ✔ | Численное решение SDE |
| PRP терапия | ✔ | f(PRP) = C₀e^(-λt) — экспоненциальное затухание |
| PEMF терапия | ✔ | g(PEMF) = 1/(1 + e^(-k(f-f₀))) — сигмоидальный отклик |
| Граничные условия | ✔ | Отражающая граница Nₜ ≥ 0 |
| Синергия PRP+PEMF | ✔ | Коэффициент 1.2 |

### ABM модель ◐ ЧАСТИЧНО

| Компонент | Статус | Описание |
|-----------|--------|----------|
| 2D сетка | ✔ | 100×100 мкм, настраиваемая |
| Стволовые клетки (CD34+) | ✔ | Пролиферация, дифференциация |
| Макрофаги (CD14+/CD68+) | ✔ | M0/M1/M2 поляризация |
| Фибробласты | ✔ | Продукция ECM |
| Случайное блуждание | ✔ | 2D броуновское движение |
| Деление клеток | ✔ | Вероятностное с лимитами |
| Дифференциация | ✔ | Стволовые → Фибробласты |
| Поле цитокинов | ✔ | Диффузия, затухание |
| Поле ECM | ✔ | Продукция фибробластами |
| **Хемотаксис** | ✖ | Параметр определён, код отсутствует |
| **Контактное ингибирование** | ✖ | Параметр определён, код отсутствует |
| **Взаимодействие клетка-клетка** | ✖ | Нет коллизий, нет сил |

### Интеграция SDE+ABM ◐ ЧАСТИЧНО

| Компонент | Статус | Описание |
|-----------|--------|----------|
| Operator splitting | ✔ | Последовательное выполнение |
| Синхронизация состояния | ◐ | Только N, не C |
| **Пространственное масштабирование** | ✖ | SDE mean-field ↔ ABM spatial |
| **Передача терапий в ABM** | ✖ | PRP/PEMF не видны агентам |

### Монте-Карло ◐ ЧАСТИЧНО

| Компонент | Статус | Описание |
|-----------|--------|----------|
| Генерация ансамбля | ✔ | 100-1000 траекторий |
| Статистика | ✔ | Mean, std, квантили, CI |
| Parameter sweep | ✔ | Вариация параметров |
| Сравнение терапий | ✔ | Оценка протоколов |
| **Параллелизация** | ✖ | Framework есть, не используется |

---

## Фаза 3: Визуализация ✖ НЕ НАЧАТО (0%)

> **Примечание:** Эта фаза присутствует в плане разработки, но отсутствовала в предыдущей версии плана реализации.

| Задача | Статус | Описание |
|--------|--------|----------|
| Кривые роста N(t) с CI | ✖ | Matplotlib/Plotly |
| Динамика цитокинов | ✖ | PDGF, VEGF, TNF-α |
| Сравнение сценариев | ✖ | PRP vs PEMF vs комбинация |
| 2D heatmap плотности | ✖ | Пространственная карта |
| Карта воспаления | ✖ | Уровень воспаления |
| Анимация эволюции | ✖ | GIF/видео |
| 3D визуализация | ✖ | Plotly (опционально для MVP) |
| Экспорт PNG/SVG | ✖ | Графики |
| Экспорт CSV | ✖ | Данные симуляции |
| Экспорт PDF | ✖ | Базовый отчёт |

### Файлы для создания

| Файл | Описание | Статус |
|------|----------|--------|
| `src/visualization/plots.py` | Графики динамики | ✖ |
| `src/visualization/spatial.py` | Пространственная визуализация | ✖ |
| `src/visualization/export.py` | Экспорт результатов | ✖ |

---

## Фаза 4: FastAPI Backend ✖ НЕ НАЧАТО (0%)

### API Endpoints

```
POST /api/v1/upload              # Загрузка .fcs файла
GET  /api/v1/upload/{id}         # Статус загрузки

POST /api/v1/simulate            # Запуск симуляции
GET  /api/v1/simulate/{id}       # Статус симуляции
WS   /api/v1/simulate/{id}/ws    # WebSocket для прогресса

GET  /api/v1/results/{id}        # Результаты симуляции
POST /api/v1/export/{id}         # Экспорт (PDF/CSV)
```

### Файлы для создания

| Файл | Описание | Статус |
|------|----------|--------|
| `src/api/main.py` | FastAPI app, CORS, роутеры | ✖ |
| `src/api/routes/upload.py` | Загрузка FCS файлов | ✖ |
| `src/api/routes/simulate.py` | Запуск симуляций | ✖ |
| `src/api/routes/results.py` | Получение результатов | ✖ |
| `src/api/models/schemas.py` | Pydantic модели | ✖ |
| `src/api/services/simulation.py` | Бизнес-логика симуляций | ✖ |

**Примечание:** Структура директорий API создана (\_\_init\_\_.py файлы)

---

## Фаза 5: Tauri + React Frontend ✖ НЕ НАЧАТО (0%)

### 5.1 Инициализация проекта

```bash
cd RegenTwin
npm create tauri-app@latest ui -- --template react-ts
cd ui
npm install
```

### 5.2 Зависимости (package.json)

```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.20.0",
    "plotly.js": "^2.27.0",
    "react-plotly.js": "^2.6.0",
    "d3": "^7.8.0",
    "axios": "^1.6.0",
    "zustand": "^4.4.0",
    "@tanstack/react-query": "^5.8.0",
    "tailwindcss": "^3.3.0",
    "@headlessui/react": "^1.7.0"
  },
  "devDependencies": {
    "@tauri-apps/api": "^1.5.0",
    "@tauri-apps/cli": "^1.5.0",
    "typescript": "^5.3.0",
    "vite": "^5.0.0",
    "@types/d3": "^7.4.0",
    "@types/plotly.js": "^2.12.0"
  }
}
```

### 5.3 Компоненты React

| Компонент | Файл | Функционал | Статус |
|-----------|------|------------|--------|
| **Upload** | `components/Upload/UploadFCS.tsx` | Drag-drop загрузка | ✖ |
| **Parameters** | `components/Parameters/TherapyConfig.tsx` | Слайдеры PRP/PEMF | ✖ |
| **Simulation** | `components/Simulation/SimulationRunner.tsx` | Запуск, прогресс-бар | ✖ |
| **Charts** | `components/Visualization/GrowthChart.tsx` | Кривые роста | ✖ |
| **Heatmap** | `components/Visualization/CellHeatmap.tsx` | 2D карта плотности | ✖ |
| **3D View** | `components/Visualization/SpatialView3D.tsx` | Three.js | ✖ |
| **Results** | `components/Results/ExportPanel.tsx` | Экспорт PDF, CSV | ✖ |

### 5.4 Страницы

| Страница | Route | Описание | Статус |
|----------|-------|----------|--------|
| Home | `/` | Описание, Quick Start | ✖ |
| Dashboard | `/dashboard` | Загрузка → Параметры → Симуляция | ✖ |
| Results | `/results/:id` | Визуализация результатов | ✖ |
| History | `/history` | История симуляций | ✖ |
| Settings | `/settings` | Настройки приложения | ✖ |

---

## Фаза 6: Тестирование ✔ РЕАЛИЗОВАНО (90%)

> **Примечание:** Тесты реализованы, но не были отражены в предыдущей версии плана.

| Категория | Статус | Файлы | LOC |
|-----------|--------|-------|-----|
| Unit-тесты (core) | ✔ | `tests/unit/core/*.py` | ~1500 |
| Unit-тесты (data) | ✔ | `tests/unit/data/*.py` | ~600 |
| Integration-тесты | ✔ | `tests/integration/test_pipeline_e2e.py` | 307 |
| Performance-тесты | ✔ | `tests/performance/test_processing_time.py` | ~100 |
| Fixtures (conftest) | ✔ | `tests/conftest.py` | ~500 |

### Покрытие модулей

| Модуль | Тесты | Статус |
|--------|-------|--------|
| `sde_model.py` | `test_sde_model.py` | ✔ |
| `abm_model.py` | `test_abm_model.py` | ✔ |
| `integration.py` | `test_integration.py` | ✔ |
| `monte_carlo.py` | `test_monte_carlo.py` | ✔ |
| `fcs_parser.py` | `test_fcs_parser.py` | ✔ |
| `gating.py` | `test_gating.py` | ✔ |
| `parameter_extraction.py` | `test_parameter_extraction.py` | ✔ |
| **API endpoints** | - | ✖ |
| **Visualization** | - | ✖ |

### Недостающее

| Задача | Статус |
|--------|--------|
| Тесты API endpoints | ✖ |
| Тесты визуализации | ✖ |
| Jupyter notebooks с примерами | ✖ |

---

## Фаза 7: Интеграция и деплой ✖ НЕ НАЧАТО (0%)

### Docker

| Задача | Статус |
|--------|--------|
| Dockerfile | ✖ |
| docker-compose.yml | ✖ |
| .dockerignore | ✖ |

### CI/CD

| Задача | Статус |
|--------|--------|
| GitHub Actions workflow | ✖ |
| Автоматический запуск тестов | ✖ |
| Линтинг (ruff) в CI | ✖ |
| Сборка Docker образа | ✖ |

### Деплой

| Задача | Статус |
|--------|--------|
| Деплой на Heroku/Railway | ✖ |
| Tauri сборка (desktop) | ✖ |
| Базовый мониторинг | ✖ |

---

## Порядок реализации (обновлённый)

| # | Задача | Зависимости | Статус |
|---|--------|-------------|--------|
| 1 | Завершить ABM (хемотаксис, контактное ингибирование) | - | ✖ |
| 2 | Завершить интеграцию SDE↔ABM | 1 | ✖ |
| 3 | Включить параллелизацию Monte Carlo | - | ✖ |
| 4 | Реализовать визуализацию | 1, 2 | ✖ |
| 5 | FastAPI: базовый сервер + CORS | - | ✖ |
| 6 | API: /upload endpoint | 5 | ✖ |
| 7 | API: /simulate endpoint | 5, 4 | ✖ |
| 8 | API: /results + WebSocket | 7 | ✖ |
| 9 | Tauri: инициализация проекта | - | ✖ |
| 10 | React: базовая структура, роутинг | 9 | ✖ |
| 11 | React: Upload компонент | 10, 6 | ✖ |
| 12 | React: Parameters + Simulation | 11, 7 | ✖ |
| 13 | React: Visualization (Plotly, D3) | 12, 8 | ✖ |
| 14 | Docker + CI/CD | 8 | ✖ |
| 15 | Tauri: сборка desktop | 13 | ✖ |

---

## Критические файлы

| Файл | Описание | Статус |
|------|----------|--------|
| `pyproject.toml` | Зависимости проекта | ✔ |
| `src/core/abm_model.py` | Требует доработки (хемотаксис) | ◐ |
| `src/core/monte_carlo.py` | Требует доработки (параллелизация) | ◐ |
| `src/api/main.py` | FastAPI app с CORS | ✖ |
| `src/visualization/plots.py` | Графики | ✖ |
| `ui/src/App.tsx` | Роутинг страниц | ✖ |
| `ui/src/services/api.ts` | API клиент | ✖ |
| `ui/src-tauri/tauri.conf.json` | Конфигурация Tauri | ✖ |
| `Dockerfile` | Контейнеризация | ✖ |
| `.github/workflows/ci.yml` | CI/CD | ✖ |

---

## Сводка по прогрессу

| Фаза | Название | Статус | Прогресс |
|------|----------|--------|----------|
| 0 | Инфраструктура | ◐ Частично | 70% |
| 1 | Data Pipeline | ◐ Частично | 95% |
| 2 | Математическое ядро | ◐ Частично | 85% |
| 3 | Визуализация | ✖ Не начато | 0% |
| 4 | FastAPI Backend | ✖ Не начато | 0% (только структура) |
| 5 | Tauri + React Frontend | ✖ Не начато | 0% |
| 6 | Тестирование | ✔ Реализовано | 90% |
| 7 | Интеграция и деплой | ✖ Не начато | 0% |

### Общий прогресс: ~45%

### Созданные файлы (проверено):

**Python Backend:**
- `src/core/` — 4 файла (sde_model, abm_model, integration, monte_carlo) ◐
- `src/data/` — 3 файла (fcs_parser, gating, parameter_extraction) ✔
- `src/api/` — только \_\_init\_\_.py файлы ✖
- `src/visualization/` — только \_\_init\_\_.py ✖

**Тесты:**
- `tests/unit/` — 7 файлов тестов ✔
- `tests/integration/` — 1 файл ✔
- `tests/performance/` — 1 файл ✔
- `tests/conftest.py` — fixtures ✔

**Документация:**
- `Description/` — 9 файлов описаний ✔
- `data/mock/` — generate_mock_data.py + README.md ✔

**Конфигурация:**
- `pyproject.toml` ✔
- `ruff.toml` ✔
- `.gitignore` ✔

---

## Недостающие компоненты (приоритет)

### Высокий приоритет (блокеры)

1. **ABM: хемотаксис** — критично для реалистичной симуляции
2. **Визуализация** — без неё невозможен MVP
3. **FastAPI backend** — связь frontend ↔ core
4. **Frontend (минимум)** — хотя бы базовый UI

### Средний приоритет

5. **Monte Carlo параллелизация** — производительность
6. **Интеграция SDE↔ABM** — пространственная связь
7. **Docker + CI/CD** — деплой и автоматизация

### Низкий приоритет (после MVP)

8. **Поддержка изображений** — опционально
9. **3D визуализация** — опционально
10. **Реальные датасеты FlowRepository** — для валидации

---

## Верификация

```bash
# Backend
cd RegenTwin
uvicorn src.api.main:app --reload
# http://localhost:8000/docs — Swagger UI

# Тесты
pytest -v --cov=src --cov-report=term-missing

# Frontend (dev)
cd ui
npm run tauri dev
# Откроется нативное окно с React приложением

# Frontend (web only)
npm run dev
# http://localhost:5173

# Build desktop
npm run tauri build
```

---

*Документ обновлён: 5 февраля 2026*
*Версия: 3.0 (полный аудит реализации)*
