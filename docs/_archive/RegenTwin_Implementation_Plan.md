# План разработки RegenTwin MVP

## Обзор проекта

**RegenTwin** — веб-приложение для симуляции регенерации тканей с использованием:
- Данных flow cytometry (.fcs файлы)
- Моделирования терапий PRP (Platelet-Rich Plasma) и PEMF (Pulsed Electromagnetic Field)
- Математического ядра: SDE + ABM

**Стек:** Python 3.11+, UV, FastAPI, Streamlit, NumPy, SciPy, FlowKit, Plotly

---

## Методология разработки (из instructions.md)

Для каждого модуля/файла:

1. **Stub** — пишем скелет-заглушку с сигнатурами функций/классов
2. **Description** — создаём `Description/description_<filename>.md` с описанием функционала
3. **Docstrings** — краткие описания со ссылками на файл описания

---

## Структура проекта

```
regentwin/
├── src/
│   ├── core/           # Математическое ядро (SDE, ABM)
│   ├── data/           # Парсинг и обработка данных
│   ├── api/            # FastAPI backend
│   ├── visualization/  # Графики и визуализация
│   └── utils/          # Вспомогательные функции
├── frontend/           # Streamlit интерфейс
├── Description/        # Файлы описания функционала
├── notebooks/          # Jupyter для экспериментов
├── data/               # Примеры датасетов
├── docs/               # Документация
├── docker/
├── pyproject.toml
└── README.md
```

---

## Фаза 0: Подготовка инфраструктуры ✅

### Задачи:

| # | Задача | Файлы | Статус |
|---|--------|-------|--------|
| 0.1 | Инициализация репозитория | `.gitignore`, `README.md` | ✅ |
| 0.2 | Настройка UV и pyproject.toml | `pyproject.toml` | ✅ |
| 0.3 | Создание структуры директорий | все `__init__.py` | ✅ |
| 0.4 | Настройка линтеров/форматтеров | `ruff.toml` | ✅ |
| 0.5 | Мок-данные | `data/mock/` | ✅ |
| 0.6 | CI/CD (GitHub Actions) | `.github/workflows/ci.yml` | ⏳ |
| 0.7 | Docker окружение | `Dockerfile`, `docker-compose.yml` | ⏳ |

---

## Фаза 1: Data Pipeline ✅ (Stubs)

### Модули для создания:

#### 1.1 `src/data/fcs_parser.py` ✅
- Stub: класс `FCSLoader` с методами `load()`, `get_channels()`, `get_events()`, `get_metadata()`
- Description: `Description/description_fcs_parser.md` ✅

#### 1.2 `src/data/gating.py` ✅
- Stub: класс `GatingStrategy` с методами `debris_gate()`, `live_cells_gate()`, `cd34_gate()`, `macrophage_gate()`
- Description: `Description/description_gating.md` ✅

#### 1.3 `src/data/parameter_extraction.py` ✅
- Stub: класс `ParameterExtractor` с методами `extract_n0()`, `extract_c0()`, `extract_inflammation_level()`
- Description: `Description/description_parameter_extraction.md` ✅

### Статус: Stubs и Description файлы созданы. Требуется реализация.

---

## Фаза 2: Математическое ядро

### Модули для создания:

#### 2.1 `src/core/sde_model.py`
- Stub: класс `SDEModel` с методами `euler_maruyama()`, `growth_function()`, `prp_effect()`, `pemf_effect()`, `simulate()`
- Description: `Description/description_sde_model.md`

#### 2.2 `src/core/abm_model.py`
- Stub: классы `Agent`, `StemCell`, `Macrophage`, `Fibroblast`, `ABMModel`
- Description: `Description/description_abm_model.md`

#### 2.3 `src/core/integration.py`
- Stub: класс `IntegratedModel` — связка SDE и ABM
- Description: `Description/description_integration.md`

#### 2.4 `src/core/monte_carlo.py`
- Stub: класс `MonteCarloSimulator` с методами `run_trajectories()`, `aggregate_results()`
- Description: `Description/description_monte_carlo.md`


### Модули для создания:

#### 2.1 `src/core/sde_model.py` ✅
- Stub: класс `SDEModel` с методами `simulate()`, `_calculate_drift()`, `_calculate_diffusion()`, `_logistic_growth()`, `_prp_effect()`, `_pemf_effect()`
- Dataclasses: `SDEConfig`, `TherapyProtocol`, `SDEState`, `SDETrajectory`
- Description: `Description/description_sde_model.md` ✅

#### 2.2 `src/core/abm_model.py` ✅
- Stub: классы `Agent`, `StemCell`, `Macrophage`, `Fibroblast`, `ABMModel`
- Dataclasses: `ABMConfig`, `AgentState`, `ABMSnapshot`, `ABMTrajectory`
- Description: `Description/description_abm_model.md` ✅

#### 2.3 `src/core/integration.py` ✅
- Stub: класс `IntegratedModel` — связка SDE и ABM через operator splitting
- Dataclasses: `IntegrationConfig`, `IntegratedState`, `IntegratedTrajectory`
- Description: `Description/description_integration.md` ✅

#### 2.4 `src/core/monte_carlo.py` ✅
- Stub: класс `MonteCarloSimulator` с методами `run()`, `_run_single_trajectory()`, `_aggregate_trajectories()`
- Dataclasses: `MonteCarloConfig`, `TrajectoryResult`, `MonteCarloResults`
- Description: `Description/description_monte_carlo.md` ✅
---

## Фаза 3: Визуализация

### Модули для создания:

#### 3.1 `src/visualization/plots.py`
- Stub: функции `plot_growth_curve()`, `plot_cytokines()`, `plot_population_dynamics()`
- Description: `Description/description_plots.md`

#### 3.2 `src/visualization/spatial.py`
- Stub: функции `heatmap_density()`, `inflammation_map()`, `animate_evolution()`
- Description: `Description/description_spatial.md`

#### 3.3 `src/visualization/export.py`
- Stub: класс `ReportExporter` с методами `to_png()`, `to_csv()`, `to_pdf()`
- Description: `Description/description_export.md`

---

## Фаза 4: Backend API

### Модули для создания:

#### 4.1 `src/api/main.py`
- Stub: FastAPI app с роутерами
- Description: `Description/description_api_main.md`

#### 4.2 `src/api/routes/upload.py`
- Stub: эндпоинты `POST /upload`, `GET /upload/{id}`
- Description: `Description/description_routes_upload.md`

#### 4.3 `src/api/routes/simulate.py`
- Stub: эндпоинты `POST /simulate`, `GET /simulate/{id}`, `GET /results/{id}`
- Description: `Description/description_routes_simulate.md`

#### 4.4 `src/api/models/`
- Stub: Pydantic модели для Request/Response
- Description: `Description/description_api_models.md`

---

## Фаза 5: Frontend (Streamlit)

### Страницы для создания:

| Страница | Файл | Функционал |
|----------|------|------------|
| Главная | `frontend/pages/home.py` | Описание, quick start |
| Загрузка | `frontend/pages/upload.py` | Upload .fcs, превью |
| Параметры | `frontend/pages/parameters.py` | Слайдеры PRP/PEMF |
| Симуляция | `frontend/pages/simulation.py` | Запуск, прогресс |
| Результаты | `frontend/pages/results.py` | Графики, экспорт |

---

## Фаза 6: Интеграция и деплой

- Валидация модели на реальных датасетах
- Документация (README, API docs, Jupyter notebooks)
- Docker-образ production
- Деплой (Heroku/Railway)

---

## Верификация

После каждой фазы:
- Проверка линтера: `ruff check src/`
- Проверка типов: `mypy src/`
- Для API: `uvicorn src.api.main:app --reload`
- Для frontend: `streamlit run frontend/app.py`

---

*Документ создан: Январь 2026*
*Версия: 1.0*
