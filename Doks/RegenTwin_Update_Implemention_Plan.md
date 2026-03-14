# План разработки RegenTwin v4.0 — Полная математическая модель регенерации тканей

## Обзор проекта

**RegenTwin** — веб/десктоп-приложение для мультимасштабного моделирования регенерации тканей с использованием:
- Данных flow cytometry (.fcs файлы)
- Полной системы стохастических дифференциальных уравнений (20+ переменных)
- Agent-Based модели с расширенным набором клеточных типов
- Моделирования терапий PRP и PEMF с механистическими моделями
- Параметрической идентификации и анализа чувствительности

**Стек:**
- **Backend:** Python 3.11+, FastAPI, NumPy, SciPy, PyMC/emcee, SALib
- **Frontend:** Tauri + React + TypeScript
- **Визуализация:** Plotly.js, D3.js, Three.js (3D)
- **Инфраструктура:** Docker, GitHub Actions CI/CD, PostgreSQL, Alembic

---

## Архитектура

```
┌───────────────────────────────────────────────────┐
│      Tauri + React (Desktop/Web)                  │
│  - TypeScript, Plotly.js, D3.js, Three.js         │
│  - Компоненты: Upload, Parameters, Results, 3D    │
└────────────────────────┬──────────────────────────┘
                         │ HTTP / WebSocket
                         ▼
┌───────────────────────────────────────────────────┐
│      FastAPI Backend (localhost:8000)             │
│  - REST API для симуляций                         │
│  - WebSocket для прогресса                        │
│  - Celery/Background Tasks                        │
└────────────────────────┬──────────────────────────┘
                         │ Python imports
                         ▼
┌───────────────────────────────────────────────────┐
│      Mathematical Core (src/)                     │
│  - src/core/extended_sde.py  (20+ SDE)       100% │
│  - src/core/sde_model.py     (MVP 2-var)     100% │
│  - src/core/abm_model.py     (расширенная ABM) 99%│
│  - src/core/integration.py   (SDE+ABM)       100% │
│  - src/core/monte_carlo.py   (параллелизация) 100%│
│  - src/analysis/             (SALib, PyMC)   [NEW]│
│  - src/data/                 (FCS парсинг)   100% │
└───────────────────────────────────────────────────┘
                         │
                         ▼
┌───────────────────────────────────────────────────┐
│      Data & Validation Layer                      │
│  - PostgreSQL + Alembic миграции                  │
│  - FlowRepository данные для валидации            │
│  - Публичные датасеты (GEO, WHS)                  │
└───────────────────────────────────────────────────┘
```

---

## Структура проекта

```
RegenTwin/
├── src/                          # Python backend
│   ├── core/                     # Математическое ядро
│   │   ├── sde_model.py          # ✔ MVP SDE (2 переменных, 575 LOC)
│   │   ├── extended_sde.py       # ✔ Расширенная SDE (20+ перем., 1104 LOC)
│   │   ├── sde_numerics.py       # ✔ EM, Milstein, IMEX, Adaptive, SRK реализованы (880 LOC)
│   │   ├── abm_model.py          # ◐ ABM реализована, интегрирована с API, 2 стаба движения (2335 LOC)
│   │   ├── abm_spatial.py        # ✖ KD-Tree, хемотаксис
│   │   ├── integration.py        # ✔ SDE+ABM operator splitting (836 LOC)
│   │   ├── equation_free.py      # ✔ Equation-Free Framework (570 LOC)
│   │   ├── monte_carlo.py        # ✔ Monte Carlo + параллелизация (872 LOC)
│   │   ├── therapy_models.py     # ✔ PRP/PEMF механистические (124 тестов, 583 LOC)
│   │   ├── numerical_utils.py    # ✔ NumericalGuard, клиппинг, адапт. шаг (382 LOC)
│   │   ├── wound_phases.py       # ✔ Фазы заживления (331 LOC)
│   │   ├── parameters.py         # ✔ ParameterSet (80+ полей, 334 LOC)
│   │   └── robustness.py         # ✔ Верификация реализована: позитивность, NaN, conservation, MMS, SDE vs ABM (583 LOC)
│   ├── data/                     # Data Pipeline
│   │   ├── fcs_parser.py         # ✔ Реализовано (исправлены subsample, нормализация ключей метаданных)
│   │   ├── gating.py             # ✔ Реализовано
│   │   ├── parameter_extraction.py # ✔ Реализовано
│   │   ├── image_loader.py       # ✔ Реализовано
│   │   ├── validation.py         # ✔ Реализовано
│   │   └── dataset_loader.py     # ✔ Реализовано
│   ├── analysis/                 # ◐ ЧАСТИЧНО (Sobol реализован в api/services)
│   │   ├── sensitivity.py        # ✖ Выделенный модуль SobolAnalyzer (SALib)
│   │   ├── parameter_estimation.py # ✖ Bayesian (PyMC/emcee)
│   │   ├── validation.py         # ✖ Метрики: R², phase timing
│   │   └── benchmarking.py       # ✖ Сравнение с Flegg/Xue/Vodovotz
│   ├── api/                      # ✔ FastAPI endpoints
│   │   ├── main.py               # ✔ App + CORS + Middleware + Exception handlers
│   │   ├── config.py             # ✔ Конфигурация (хост, порт, CORS)
│   │   ├── routes/               # ✔ health, upload, simulate, results, analysis, viz, spatial
│   │   ├── models/schemas.py     # ✔ Pydantic схемы (SimulationRequest, SensitivityRequest, …)
│   │   └── services/             # ✔ simulation_service (EXTENDED/MVP/ABM), analysis_service (Sobol), file_service
│   ├── visualization/            # ✔ Визуализация (Plotly)
│   │   ├── plots.py              # ✔ Популяции, цитокины, ECM, фазы
│   │   ├── spatial.py            # ✔ ABM тепловые карты, scatter, inflammation
│   │   ├── export.py             # ✔ PNG/SVG/PDF (kaleido + fpdf2)
│   │   └── theme.py              # ✔ Цветовые схемы и константы
│   ├── db/                       # ✔ Database Layer
│   │   ├── models.py             # ✔ SQLAlchemy ORM: SimulationRecord, UploadRecord, AnalysisRecord
│   │   ├── session.py            # ✔ SessionLocal, get_db(), create_tables()
│   │   └── migrations/           # ✔ 001_initial_tables, 002_add_indexes_and_fk
│   └── utils/                    # ◐
│       ├── logging.py            # ✖ Loguru конфигурация
│       └── error_handling.py     # ✖ Обработка ошибок
│
├── ui/                           # ◐ Tauri + React frontend (в разработке)
│   ├── src-tauri/                # Tauri (Rust)
│   └── src/                      # React (TypeScript)
│
├── tests/                        # ◐ Тесты (расширение)
│   ├── unit/                     # ✔ + новые модули
│   ├── integration/              # ✔
│   ├── performance/              # ✔
│   └── validation/               # ✖ Тесты на данных
│
├── Description/                  # ✔ Описания функционала
├── Doks/                         # ✔ Документация проекта
├── data/                         # Данные
│   ├── mock/                     # ✔ Мок-данные
│   └── validation/               # ✖ Реальные датасеты
├── docker/                       # ✖ Docker конфигурация
├── .github/workflows/            # ✖ CI/CD
├── notebooks/                    # ✖ Jupyter примеры
├── pyproject.toml                # ✔
├── ruff.toml                     # ✔
└── .gitignore                    # ✔
```

---

## Условные обозначения

| Символ | Значение |
|--------|----------|
| ✔ | Полностью реализовано |
| ◐ | Частично реализовано |
| ✖ | Не реализовано |

---

## Фаза 0: Инфраструктура и DevOps ◐ ЧАСТИЧНО (70% → цель 100%)

| Задача | Статус | Описание |
|--------|--------|----------|
| pyproject.toml | ✔ | Все зависимости включая PyMC, SALib, emcee, celery — **уже добавлены** |
| Структура директорий | ✔ | src/, tests/, data/ |
| __init__.py файлы | ✔ | Созданы |
| .gitignore, ruff.toml | ✔ | Настроены |
| Мок-данные | ✔ | generate_mock_data.py |
| **Dockerfile** | ✖ | Мультистейдж: builder + runtime, Python 3.11 slim |
| **docker-compose.yml** | ✖ | app + postgres + redis (для Celery) |
| **.dockerignore** | ✖ | Исключения: .venv, __pycache__, .git |
| **CI/CD: lint+test** | ✖ | `.github/workflows/ci.yml`: ruff, mypy, pytest, coverage |
| **CI/CD: Docker build** | ✖ | `.github/workflows/docker.yml`: build + push |
| **CI/CD: coverage badge** | ✖ | Codecov или coveralls интеграция |
| **Loguru конфигурация** | ✖ | `src/utils/logging.py`: structured logging для всех модулей |
| **Alembic init** | ✔ | `alembic.ini` + `src/db/migrations/versions/001_initial_tables, 002_add_indexes_and_fk` |
| **SQLAlchemy модели** | ✔ | `src/db/models.py`: SimulationRecord, UploadRecord, AnalysisRecord |
| **Загрузка реальных данных** | ✖ | Скрипт загрузки с FlowRepository (FR-FCM-*) |
| **pre-commit hooks** | ✖ | `.pre-commit-config.yaml`: ruff, black, mypy |

### Новые зависимости для pyproject.toml

```toml
# Уже добавлены в dependencies:
"pymc>=5.10.0",           # ✔ добавлено
"emcee>=3.1.0",           # ✔ добавлено
"SALib>=1.4.0",           # ✔ добавлено
"celery>=5.3.0",          # ✔ добавлено (без redis extras)

# Ещё нужно добавить:
"redis>=5.0.0",           # ✖ отсутствует
"psycopg2-binary>=2.9.0", # ✖ отсутствует

# Добавить в dev:
"pytest-xdist>=3.5.0",   # ✖ отсутствует
"pytest-benchmark>=4.0.0", # ✖ отсутствует
```

---

## Фаза 1: Data Pipeline ✔ РЕАЛИЗОВАНО (100%)

| Файл | Классы/Функции | Статус | LOC |
|------|----------------|--------|-----|
| `src/data/fcs_parser.py` | `FCSLoader`, `FCSMetadata`, `load_fcs()` | ✔ | 234 |
| `src/data/gating.py` | `GatingStrategy`, `GateResult`, `GatingResults` | ✔ | 651 |
| `src/data/parameter_extraction.py` | `ParameterExtractor`, `ModelParameters` | ✔ | 813 |
| `src/data/image_loader.py` | `ImageLoader`, `ImageAnalyzer`, `ScatterPlotExtractor` | ✔ | 1260 |
| `src/data/validation.py` | `ValidationResult`, `DataValidator` | ✔ | 448 |
| `src/data/dataset_loader.py` | `DatasetLoader`, `DatasetMetadata` — FlowRepository, GEO, кэш | ✔ | 562 |
| `Description/Phase1/*.md` | Описания | ✔ | 6 файлов |
| `tests/unit/data/` | 532 теста, покрытие 93-100% | ✔ | ~2500 |

### Доработки для полной модели

| Задача | Статус | Описание |
|--------|--------|----------|
| Расширить `ModelParameters` для 20+ переменных | ✖ | Начальные условия P0, Ne0, M1_0, M2_0, F0, Mf0, E0, S0 |
| Расширить гейтинг для новых популяций | ✖ | CD66b+ (нейтрофилы), CD31+ (эндотелий) |
| `src/data/dataset_loader.py` | ✔ | Загрузка публичных датасетов (FlowRepository, GEO) — реализовано |
| Валидационные данные | ✖ | `data/validation/` с реальными .fcs и временными рядами |

---

## Фаза 2: Математическое ядро MVP ✔ РЕАЛИЗОВАНО (95%)

> **Цель:** Довести существующую 2-переменную модель до production-ready состояния.

| Файл | Код | Description | Тесты | Статус |
|------|-----|-------------|-------|--------|
| `src/core/sde_model.py` | ✔ | ✔ | ✔ 100 тестов | Полностью реализован (575 LOC) |
| `src/core/abm_model.py` | ◐ | ✔ | ✔ 214 тестов | Реализована, 2 стаба движения/взаимодействия (2335 LOC) |
| `src/core/integration.py` | ✔ | ✔ | ✔ 100 тестов | Полностью реализован — operator splitting, SDE+ABM (836 LOC) |
| `src/core/monte_carlo.py` | ✔ | ✔ | ✔ 97 тестов | Полностью реализован + параллелизация (872 LOC) |
| `src/core/numerical_utils.py` | ✔ | ✔ | ✔ 68 тестов | Полностью реализован: clip, detect_divergence, adaptive_dt, NumericalGuard (382 LOC) |

### 2.1 ABM: завершение базовой функциональности

| Компонент | Статус | Описание |
|-----------|--------|----------|
| 2D сетка, агенты, поля | ✔ | Работает |
| **Хемотаксис** | ✖ | Градиент цитокинов → направленное движение |
| **Контактное ингибирование** | ✖ | Подавление деления при высокой плотности |
| **Взаимодействие клетка-клетка** | ✖ | Отталкивание, адгезия |
| **KD-Tree для поиска соседей** | ✖ | `scipy.spatial.cKDTree` вместо SpatialHash |

### 2.2 Интеграция SDE+ABM: завершение

| Компонент | Статус | Описание |
|-----------|--------|----------|
| Operator splitting | ✔ | Реализовано в integration.py (836 LOC) |
| Синхронизация N | ✔ | Бидиректциональная синхронизация реализована в integration.py |
| **Передача терапий в ABM** | ✖ | PRP/PEMF эффекты на уровне агентов |
| **Пространственное масштабирование** | ✖ | SDE mean-field ↔ ABM spatial маппинг |

### 2.3 Monte Carlo: параллелизация

| Компонент | Статус | Описание |
|-----------|--------|----------|
| Генерация ансамбля | ✔ | Работает |
| Статистика | ✔ | Mean, std, CI, квантили |
| **Параллелизация** | ✔ | `concurrent.futures` реализовано в monte_carlo.py |
| **Прогресс callback** | ✖ | Для WebSocket отображения в UI |

### 2.4 Численная робастность

| Компонент | Статус | Описание |
|-----------|--------|----------|
| **Клиппинг отрицательных концентраций** | ✔ | `clip_negative_concentrations()` в numerical_utils.py |
| **NaN/Inf detection** | ✔ | `detect_divergence()` — остановка + fallback в numerical_utils.py |
| **Адаптивный временной шаг** | ✔ | `adaptive_timestep()` + `NumericalGuard` в numerical_utils.py |
| **Loguru логирование в core** | ✖ | Предупреждения при отрицательных значениях — src/utils/logging.py не создан |

---

## Фаза 2.5: Расширенная SDE система ✔ РЕАЛИЗОВАНО (100%)

> **Цель:** Реализация полной 20+ переменной системы SDE из математического фреймворка.
> **Приоритет:** КРИТИЧЕСКИЙ для публикации
> **Зависимости:** Фаза 2 (MVP должна быть стабильной)
> **Результат:** 142 тестов PASSED (test_extended_sde.py), coverage 99-100%, верификация формул с Mathematical Framework пройдена

### 2.5.1 Клеточные популяции (8 уравнений)

| Переменная | Уравнение | Файл | Статус |
|------------|-----------|------|--------|
| P(t) — тромбоциты | dP = [S_P - δ_P·P - k_deg·P]dt + σ_P·P·dW | `extended_sde.py` | ✔ |
| Nₑ(t) — нейтрофилы | dNe = [R_Ne(C_IL8) - δ_Ne·Ne - k_phag·M·Ne/(Ne+K)]dt + ... | `extended_sde.py` | ✔ |
| M₁(t) — M1 макрофаги | dM1 = [R_M·φ₁ - k_switch·ψ·M1 + k_rev·ζ·M2 - δ_M·M1]dt + ... | `extended_sde.py` | ✔ |
| M₂(t) — M2 макрофаги | dM2 = [R_M·φ₂ + k_switch·ψ·M1 - k_rev·ζ·M2 - δ_M·M2]dt + ... | `extended_sde.py` | ✔ |
| F(t) — фибробласты | dF = [r_F·F·(1-(F+Mf)/K_F)·H + k_diff_S·S·g - k_act·F·A - δ_F·F]dt + ... | `extended_sde.py` | ✔ |
| Mf(t) — миофибробласты | dMf = [k_act·F·A - δ_Mf·Mf·(1-TGF/(K_surv+TGF))]dt + ... | `extended_sde.py` | ✔ |
| E(t) — эндотелиальные | dE = [r_E·E·(1-E/K_E)·V·(1-θ) - δ_E·E]dt + ... | `extended_sde.py` | ✔ |
| S(t) — стволовые (CD34+) | dS = [r_S·S·(1-S/K_S)·(1+α_PRP·Θ) - k_diff·S·g - δ_S·S]dt + ... | `extended_sde.py` | ✔ |

### 2.5.2 Сигнальные молекулы (7 уравнений)

| Переменная | Продуценты | Файл | Статус |
|------------|-----------|------|--------|
| C_TNF(t) | M1, Ne; ингибирование IL-10 | `extended_sde.py` | ✔ |
| C_IL10(t) | M2, эффероцитоз; противовоспалительный | `extended_sde.py` | ✔ |
| C_PDGF(t) | Тромбоциты, макрофаги, PRP | `extended_sde.py` | ✔ |
| C_VEGF(t) | M2, фибробласты, PRP; гипоксия-зависимый | `extended_sde.py` | ✔ |
| C_TGFβ(t) | Тромбоциты, M2, Mf (положительная обр. связь!) | `extended_sde.py` | ✔ |
| C_MCP1(t) | DAMPs, M1; хемоаттрактант моноцитов | `extended_sde.py` | ✔ |
| C_IL8(t) | DAMPs, M1, Ne (аутокринная петля + IL-10 suppression) | `extended_sde.py` | ✔ |

### 2.5.3 Внеклеточный матрикс (3 уравнения)

| Переменная | Описание | Файл | Статус |
|------------|----------|------|--------|
| ρ_collagen(t) | Продукция F/Mf, деградация MMP, насыщение | `extended_sde.py` | ✔ |
| C_MMP(t) | Секреция M1/M2/F, ингибирование TIMP | `extended_sde.py` | ✔ |
| ρ_fibrin(t) | Фибринолиз + замещение коллагеном | `extended_sde.py` | ✔ |

### 2.5.4 Вспомогательные переменные (2)

| Переменная | Описание | Файл | Статус |
|------------|----------|------|--------|
| D(t) — сигнал повреждения | D₀·exp(-t/τ_damage), DAMPs | `extended_sde.py` | ✔ |
| O₂(t) — кислород | Диффузия + потребление + ангиогенез | `extended_sde.py` | ✔ |

### 2.5.5 Файлы

| Файл | Описание | Статус |
|------|----------|--------|
| `src/core/extended_sde.py` | `StateIndex(20)`, `ExtendedSDEState`, `ExtendedSDETrajectory`, `ExtendedSDEModel` (50+ методов) | ✔ 99% coverage |
| `src/core/wound_phases.py` | `WoundPhase(4)`, `PhaseIndicators`, `WoundPhaseDetector` (8 методов) | ✔ 100% coverage |
| `src/core/parameters.py` | `ParameterSet` dataclass (105 полей из §8), 4 метода | ✔ 100% coverage |
| `Description/Phase2/description_extended_sde.md` | Описание функционала + TDD секции | ✔ |
| `Description/Phase2/description_wound_phases.md` | Описание функционала + TDD секции | ✔ |
| `Description/Phase2/description_parameters.md` | Описание функционала + TDD секции | ✔ |
| `tests/unit/core/test_extended_sde.py` | TDD тесты: conservation laws, positivity, фазовые переходы | ✔ 142 passed |
| `tests/unit/core/test_wound_phases.py` | TDD тесты: детекция фаз, переходы | ✔ |
| `tests/unit/core/test_parameters.py` | TDD тесты: ParameterSet, валидация, экспорт | ✔ |

### 2.5.6 Ключевые биологические свойства — верифицировано

| Свойство | Тест | Описание | Статус |
|----------|------|----------|--------|
| Позитивность | `test_all_populations_nonnegative` | Все переменные ≥ 0 на всем интервале | ✔ |
| Фазовый переход M1→M2 | `test_macrophage_switch` | M1 пик на 24-48ч, M2 доминирует после 72ч | ✔ |
| Бистабильность TGF-β | `test_tgfb_bistability` | Два устойчивых состояния: заживление vs фиброз | ✔ |
| Ангиогенез при гипоксии | `test_hypoxia_driven_angiogenesis` | Низкий O₂ → рост E(t) | ✔ |
| Фибриновый каркас | `test_fibrin_to_collagen` | ρ_fibrin падает, ρ_collagen растёт | ✔ |
| Нейтрофильный пик | `test_neutrophil_peak` | Пик Ne на 12-24ч, затухание к 48ч | ✔ |

### 2.5.7 Верификация формул с Mathematical Framework

Все 20 SDE уравнений, 13 вспомогательных функций и 105 параметров сверены с `Doks/RegenTwin_Mathematical_Framework.md`.

| Компонент | Результат |
|-----------|-----------|
| 20 drift-методов | 20/20 совпадают с §2.1-2.4 |
| 13 helper-функций (Hill, polarization, switching и др.) | 13/13 корректны |
| 105 параметров ParameterSet | Совпадают с §8 |
| IL-8 расширение (IL-10 suppression) | Биологически обосновано (Fiorentino 1991), фреймворк обновлён |
| PRP двухфазная кинетика (§3.1) | TODO: интеграция через therapy_models.py |
| PEMF механизмы (§3.2) | TODO: интеграция через therapy_models.py |

---

## Фаза 2.6: Механистические модели терапий ✔ РЕАЛИЗОВАНО (100%)

> **Цель:** Замена феноменологических f(PRP), g(PEMF) на механистические модели.
> **Зависимости:** Фаза 2.5

### PRP — многофакторная двухфазная кинетика

| Компонент | Описание | Статус |
|-----------|----------|--------|
| Двухфазное высвобождение | burst (τ~1-2ч) + sustained (τ~24-72ч) для каждого фактора | ✔ |
| PDGF из PRP | Θ_PRP_PDGF(t) → уравнение C_PDGF | ✔ |
| VEGF из PRP | Θ_PRP_VEGF(t) → уравнение C_VEGF | ✔ |
| TGF-β из PRP | Θ_PRP_TGF(t) → уравнение C_TGFβ | ✔ |
| EGF из PRP | Θ_PRP_EGF(t) → пролиферация | ✔ |
| Рекрутирование стволовых | α_PRP_S · Θ_PRP(t) → уравнение S(t) | ✔ |
| Дозозависимость | PRP_dose (3-5x) как параметр | ✔ |

### PEMF — 3 механизма

| Механизм | Мишень | Эффект | Статус |
|----------|--------|--------|--------|
| Аденозиновый A₂A/A₃ | s_TNF_M1 | Снижение TNF-α на 30-50% | ✔ |
| Ca²⁺-CaM | r_F, r_E | Усиление пролиферации через NO | ✔ |
| MAPK/ERK | D_cell (ABM) | Усиление миграции клеток | ✔ |

### Синергия PRP+PEMF

| Компонент | Формула | Статус |
|-----------|---------|--------|
| Супер-аддитивный эффект | synergy = 1 + β_synergy · Θ_PRP · PEMF_active | ✔ |

### Файлы

| Файл | Описание | Статус |
|------|----------|--------|
| `src/core/therapy_models.py` | `PRPModel`, `PEMFModel`, `SynergyModel` — механистические модели (124/124 тестов, 99% coverage) | ✔ |
| `Description/Phase2/description_therapy_models.md` | Описание + TDD секции | ✔ |
| `tests/unit/core/test_therapy_models.py` | TDD: двухфазная кинетика PRP, PEMF частотная зависимость, синергия, биологическая валидация | ✔ |

---

## Фаза 2.7: Численные методы и робастность ✔ РЕАЛИЗОВАНО (95%)

> **Цель:** Upgrade с Euler-Maruyama на Milstein + IMEX для стиффной системы
> **Зависимости:** Фаза 2.5

### Численные схемы

| Метод | Применение | Порядок сходимости | Статус |
|-------|-----------|-------------------|--------|
| Euler-Maruyama (EM) | ✔ Текущий | 0.5 (сильная) | ✔ — реализован в sde_numerics.py |
| **Milstein** | Upgrade для скалярного шума | 1.0 (сильная) | ✔ — реализован в sde_numerics.py |
| **IMEX splitting** | Для стиффных систем (быстрые цитокины + медленный ECM) | Зависит от компонент | ✔ — реализован в sde_numerics.py |
| **Адаптивный шаг** | PI-контроллер + Richardson extrapolation в sde_numerics.py | - | ✔ — AdaptiveTimestepper в sde_numerics.py |
| **Stochastic RK (SRI2W1)** | Опционально для мультимерных перекрёстных термов | 1.0 | ✔ — реализован в sde_numerics.py |

### Робастность

| Компонент | Описание | Статус |
|-----------|----------|--------|
| Клиппинг отрицательных значений | `clip_negative_concentrations()` в numerical_utils.py | ✔ |
| NaN/Inf обнаружение | `detect_divergence()` в numerical_utils.py | ✔ |
| Conservation checks | ConservationChecker: mass_balance + cytokine_balance в robustness.py | ✔ |
| Method of Manufactured Solutions | ConvergenceVerifier: compute_order + manufactured_solution в robustness.py | ✔ |
| Сравнение SDE vs ABM | SDEvsABMComparator: Wasserstein + KS test в robustness.py | ✔ |

### Файлы для создания

| Файл | Описание | Статус |
|------|----------|--------|
| `src/core/sde_numerics.py` | EM, Milstein, IMEX, Adaptive (PI), SRK — все реализованы (880 LOC) | ✔ |
| `src/core/robustness.py` | PositivityEnforcer, NaNHandler, ConservationChecker, ConvergenceVerifier, SDEvsABMComparator (583 LOC) | ✔ |
| `Description/Phase2/description_sde_numerics.md` | Описание функционала (512 LOC) — численные схемы, IMEX, адаптивность | ✔ |
| `Description/Phase2/description_robustness.md` | Описание функционала (492 LOC) — верификация, conservation laws, MMS | ✔ |
| `tests/unit/core/test_sde_numerics.py` | 76 тестов: порядок сходимости Milstein, IMEX корректность, адаптивность | ✔ |
| `tests/unit/core/test_robustness.py` | 72 теста: клиппинг, NaN обработка, conservation, MMS, SDE vs ABM | ✔ |

---

## Фаза 2.8: Расширенная ABM ✔ РЕАЛИЗОВАНО (100%)

> **Цель:** Расширить ABM для поддержки полной системы из математического фреймворка.
> **Зависимости:** Фаза 2.5, Фаза 2 (базовая ABM)

### Новые типы агентов

| Тип агента | Поведение | Статус |
|------------|-----------|--------|
| StemCell (CD34+) | PRP-зависимая мобилизация (Michaelis-Menten) | ✔ |
| Macrophage | Continuous polarization_state ∈ [0,1], эффероцитоз | ✔ |
| Fibroblast | TGF-β-зависимая активация в MyofibroblastAgent | ✔ |
| **Neutrophil** | Хемотаксис по IL-8, апоптоз, фагоцитоз | ✔ |
| **Endothelial** | VEGF-зависимый спраутинг, junction formation | ✔ |
| **Myofibroblast** | Продукция коллагена (кумулятивная), TGF-β-зависимый апоптоз | ✔ |
| **Platelet** | Дегрануляция, высвобождение PDGF/TGFb/VEGF | ✔ |

### Новые механики ABM

| Механика | Описание | Статус |
|----------|----------|--------|
| **Хемотаксис** | ChemotaxisEngine: градиент цитокинов → направленное движение | ✔ |
| **Контактное ингибирование** | ContactInhibitionEngine: подавление деления при > threshold | ✔ |
| **Эффероцитоз** | EfferocytosisEngine: фагоцитоз → IL-10 + поляризация M2 | ✔ |
| **Механотрансдукция** | MechanotransductionEngine: мех. стресс → активация миофибробластов | ✔ |
| **Multiple cytokine fields** | MultiCytokineField: 7 полей (TNF, IL10, PDGF, VEGF, TGFb, MCP1, IL8) | ✔ |
| **KD-Tree** | KDTreeNeighborSearch: адаптер cKDTree с query_radius/query_nearest | ✔ |
| **Subcycling** | SubcyclingManager: разные dt для полей и агентов | ✔ |

### Файлы

| Файл | Описание | Статус |
|------|----------|--------|
| `src/core/abm_model.py` | Macrophage continuous polarization, StemCell.prp_mobilization, Fibroblast.tgfb_activation, EndothelialAgent VEGF | ✔ |
| `src/core/abm_spatial.py` | 8 классов: PlateletAgent, ChemotaxisEngine, ContactInhibitionEngine, EfferocytosisEngine, MechanotransductionEngine, MultiCytokineField, KDTreeNeighborSearch, SubcyclingManager | ✔ |
| `Description/Phase2/description_abm_extended.md` | Описание расширенной ABM | ✔ |
| `tests/unit/core/test_abm_extended.py` | 215 TDD тестов (все GREEN) | ✔ |

---

## Фаза 2.9: Мультимасштабная интеграция ✖ НЕ НАЧАТО (0%)

> **Цель:** Equation-Free Framework для связи расширенной SDE и ABM
> **Зависимости:** Фаза 2.5, Фаза 2.8

### Компоненты

| Компонент | Описание | Статус |
|-----------|----------|--------|
| **Lifting** (макро→микро) | Распределение агентов по ExtendedSDEState | ✖ |
| **Restricting** (микро→макро) | Агрегация: X_macro = Σ(agent_states) / volume | ✖ |
| **Синхронизация 20-мерного вектора** | Все популяции + цитокины + ECM | ✖ |
| **Subcycling интеграция** | Разные dt для быстрых и медленных переменных | ✖ |
| **Терапии на обоих уровнях** | PRP/PEMF эффекты в SDE И ABM | ✖ |

### Сопряжение шкал

| Уровень | Масштаб | Модель | Переменные |
|---------|---------|--------|-----------|
| Макро (ткань) | мм–см, часы–дни | Расширенная SDE (20+) | P, Ne, M1, M2, F, Mf, E, S, цитокины, ECM |
| Мезо (клеточная популяция) | 10-100 мкм, мин–часы | Расширенная ABM | Позиции агентов, локальные поля |
| Микро (внутриклеточный) | нм–мкм, сек–мин | **Опционально:** ODE сигнальных путей | NF-κB, Smad2/3, ERK1/2 |

### Файлы для создания

| Файл | Описание | Статус |
|------|----------|--------|
| `src/core/equation_free.py` | `EquationFreeIntegrator`, `Lifter`, `Restrictor` | ✔ |
| `tests/unit/core/test_equation_free.py` | TDD: lifting/restricting consistency, conservation | ✔ (114 passed) |

---

## Фаза 3: Анализ и валидация ✖ НЕ НАЧАТО (0%)

> **Цель:** Параметрическая идентификация, анализ чувствительности, валидация на данных
> **Приоритет:** КРИТИЧЕСКИЙ для публикации
> **Зависимости:** Фаза 2.5

### 3.1 Параметрическая идентификация

| Компонент | Библиотека | Описание | Статус |
|-----------|-----------|----------|--------|
| Bayesian estimation | PyMC 5 | Posterior распределения 40+ параметров | ✖ |
| MCMC sampling | emcee | Ансамблевый сэмплер как альтернатива | ✖ |
| Maximum Likelihood | scipy.optimize | Точечные оценки для быстрой проверки | ✖ |
| Prior specification | PyMC | Информативные priors из литературы (табл. §8) | ✖ |
| Confidence intervals | PyMC/emcee | 95% CI для всех параметров | ✖ |
| Convergence diagnostics | ArviZ | R-hat, ESS, trace plots | ✖ |

### 3.2 Анализ чувствительности

| Метод | Библиотека | Описание | Статус |
|-------|-----------|----------|--------|
| Sobol indices | SALib | Глобальная чувствительность (first-order + total) | ✖ |
| Morris screening | SALib | Скрининг 40+ параметров для отбора ключевых | ✖ |
| Local sensitivity | SciPy | Частные производные вблизи номинальных значений | ✖ |
| Tornado diagrams | Matplotlib | Визуализация ранжированной чувствительности | ✖ |

### 3.3 Валидация на данных

| Датасет | Источник | Применение | Статус |
|---------|----------|-----------|--------|
| FlowRepository (FR-FCM-*) | flowrepository.org | Начальные условия: CD34+%, CD14+%, апоптоз | ✖ |
| GEO (NCBI) | ncbi.nlm.nih.gov | Транскриптомы ран по времени | ✖ |
| Wound Healing Society | WHS | Клинические данные заживления | ✖ |
| Human Protein Atlas | proteinatlas.org | Базовые уровни белков в коже | ✖ |

### 3.4 Метрики валидации

| Метрика | Описание | Статус |
|---------|----------|--------|
| Temporal R² | Корреляция предсказанных и наблюдаемых траекторий | ✖ |
| Phase timing | Правильность длительности фаз заживления | ✖ |
| Monte Carlo envelopes | Наблюдения внутри 95% CI ансамбля | ✖ |
| Sensitivity ranking | Согласованность Sobol indices с экспертным знанием | ✖ |

### 3.5 Бенчмаркинг

| Модель-референция | Публикация | Сравнение | Статус |
|-------------------|-----------|-----------|--------|
| Flegg 2015 | Bull. Math. Biol. | Гипербарическая O₂ терапия, diabetic wounds | ✖ |
| Xue 2009 | PLoS Comput. Biol. | Ишемические раны, ECM динамика | ✖ |
| Vodovotz 2006 | Curr. Opin. Crit. Care | Острое воспаление, macrophage dynamics | ✖ |

### 3.6 Визуализация анализа (объединение с Фазой 4)

> **Примечание:** Графики анализа были отложены из Фазы 4 до реализации Фазы 3,
> так как требуют реальных выходов от `SobolAnalyzer` и `BayesianEstimator`.

| Задача | Описание | Статус |
|--------|----------|--------|
| `plot_sobol()` | Tornado bar chart — S1 и ST Sobol indices | ✖ |
| `plot_posterior()` | Marginal histograms / corner plots (ArviZ) | ✖ |
| `plot_convergence()` | Сходимость MC/MCMC метрик по итерациям | ✖ |
| `plot_morris()` | Morris screening: mu_star vs sigma scatter | ✖ |

**Файл:** `src/visualization/analysis_plots.py` — принимает generic dict/ndarray, чтобы не зависеть от конкретных типов Phase 3.

### Файлы для создания

| Файл | Описание | Статус |
|------|----------|--------|
| `src/analysis/__init__.py` | Инициализация модуля | ✖ |
| `src/analysis/sensitivity.py` | `SobolAnalyzer`, `MorrisScreener`, `LocalSensitivity` | ✖ |
| `src/analysis/parameter_estimation.py` | `BayesianEstimator`, `MCMCRunner`, `MLEstimator` | ✖ |
| `src/analysis/validation.py` | `ValidationRunner`, `TemporalR2`, `PhaseTimingMetric` | ✖ |
| `src/analysis/benchmarking.py` | `BenchmarkSuite`, сравнение с Flegg/Xue/Vodovotz | ✖ |
| `src/visualization/analysis_plots.py` | `plot_sobol()`, `plot_posterior()`, `plot_convergence()`, `plot_morris()` | ✖ |
| `Description/Phase3/description_analysis.md` | Описание | ✖ |
| `tests/unit/analysis/test_sensitivity.py` | TDD тесты | ✖ |
| `tests/unit/analysis/test_parameter_estimation.py` | TDD тесты | ✖ |
| `tests/unit/visualization/test_analysis_plots.py` | TDD тесты analysis_plots | ✖ |
| `tests/validation/test_on_real_data.py` | Интеграционные тесты на реальных данных | ✖ |

---

## Фаза 4: Визуализация ✔ РЕАЛИЗОВАНО (90%)

> **Цель:** Полный набор графиков для расширенной модели
> **Зависимости:** Фаза 2.5 (расширенная SDE)
> **Примечание:** analysis_plots.py (Sobol, posterior) отложен до Фазы 3 (пункт 3.6). 3D Three.js — до Фазы 6.

| Задача | Статус | Описание |
|--------|--------|----------|
| Кривые роста 8 популяций | ✔ | P, Ne, M1, M2, F, Mf, E, S с CI (plot_populations) |
| Динамика 7 цитокинов | ✔ | TNF, IL-10, PDGF, VEGF, TGF-β, MCP-1, IL-8 (plot_cytokines) |
| Динамика ECM | ✔ | Коллаген, MMP, фибрин — dual axes (plot_ecm) |
| Детекция фаз заживления | ✔ | Цветовая полоса + популяции (plot_phases) |
| Сравнение сценариев | ✔ | 4 сценария, single var / all 8 (plot_comparison) |
| 2D heatmap плотности | ✔ | Пространственная карта из ABM (heatmap_density) |
| Карта воспаления | ✔ | Цитокиновое поле с diverging colorscale (inflammation_map) |
| Анимация эволюции | ✔ | Plotly animation + GIF через matplotlib (animate_evolution) |
| Sensitivity tornado | ⏳ | Отложено до Фазы 3.6 (analysis_plots.py) |
| Posterior distributions | ⏳ | Отложено до Фазы 3.6 (analysis_plots.py) |
| 3D визуализация | ⏳ | Отложено до Фазы 6 (Three.js в React) |
| Экспорт PNG/SVG | ✔ | Kaleido (ReportExporter.to_png/to_svg) |
| Экспорт CSV | ✔ | 21 колонка: time + 20 переменных (ReportExporter.to_csv) |
| Экспорт PDF | ✔ | fpdf2: титул + метаданные + графики + сводка (ReportExporter.to_pdf) |
| API endpoints | ✔ | 8 FastAPI endpoints (/api/viz/*) → Plotly JSON для React |

### Файлы

| Файл | Описание | Статус |
|------|----------|--------|
| `src/visualization/theme.py` | Цветовая тема, layout defaults, группировка 20 переменных | ✔ |
| `src/visualization/plots.py` | `plot_populations()`, `plot_cytokines()`, `plot_ecm()`, `plot_phases()`, `plot_comparison()` | ✔ |
| `src/visualization/spatial.py` | `heatmap_density()`, `scatter_agents()`, `inflammation_map()`, `field_heatmap()`, `animate_evolution()` | ✔ |
| `src/visualization/analysis_plots.py` | `plot_sobol()`, `plot_posterior()`, `plot_convergence()` | ⏳ Фаза 3.6 |
| `src/visualization/export.py` | `ReportExporter`: `to_png()`, `to_svg()`, `to_csv()`, `to_pdf()` | ✔ |
| `src/api/routes/visualization.py` | FastAPI: 8 endpoints → Plotly JSON + экспорт файлов | ✔ |
| `Description/Phase4/description_visualization.md` | Описание модуля | ✔ |
| `tests/unit/visualization/test_theme.py` | 19 тестов — консистентность констант | ✔ |
| `tests/unit/visualization/test_plots.py` | 26 тестов — smoke + structural | ✔ |
| `tests/unit/visualization/test_spatial.py` | 21 тест — spatial viz + GIF export | ✔ |
| `tests/unit/visualization/test_export.py` | 18 тестов — PNG/SVG/CSV/PDF экспорт | ✔ |
| `tests/unit/api/test_visualization_routes.py` | 17 тестов — API endpoints | ✔ |

---

## Фаза 5: FastAPI Backend ✖ НЕ НАЧАТО (0%)

> **Зависимости:** Фаза 2 (MVP core), Фаза 4 (визуализация)

### API Endpoints

```
POST /api/v1/upload              # Загрузка .fcs файла
GET  /api/v1/upload/{id}         # Статус загрузки

POST /api/v1/simulate            # Запуск симуляции (MVP или Extended)
GET  /api/v1/simulate/{id}       # Статус симуляции
WS   /api/v1/simulate/{id}/ws    # WebSocket для прогресса

GET  /api/v1/results/{id}        # Результаты симуляции
POST /api/v1/export/{id}         # Экспорт (PDF/CSV)

POST /api/v1/analysis/sensitivity     # Запуск анализа чувствительности
POST /api/v1/analysis/estimation      # Запуск параметрической идентификации
GET  /api/v1/analysis/{id}            # Результаты анализа

GET  /api/v1/health              # Health check
```

### Файлы для создания

| Файл | Описание | Статус |
|------|----------|--------|
| `src/api/main.py` | FastAPI app, CORS, роутеры, Loguru middleware | ✖ |
| `src/api/routes/upload.py`  | Upload FCS + images endpoints | ✖ |
| `src/api/routes/simulate.py`| Simulate + WebSocket progress | ✖ |
| `src/api/routes/results.py` | Results + Export endpoints | ✖ |
| `src/api/routes/analysis.py`| Sensitivity + Estimation endpoints | ✖ |
| `src/api/models/schemas.py` | Pydantic модели (расширенные для 20+ перем.) | ✖ |
| `src/api/services/simulation_service.py` | Бизнес-логика: выбор MVP vs Extended, фоновое выполнение | ✖ |
| `src/api/services/file_service.py` | Обработка загруженных файлов | ✖ |
| `src/db/models.py` | SQLAlchemy: Simulation, Upload, Result, AnalysisRun | ✖ |
| `src/db/migrations/` | Alembic: начальная миграция | ✖ |

### Конфигурация

| Файл | Описание | Статус |
|------|----------|--------|
| `src/api/config.py` | Pydantic Settings: DB URL, Redis, CORS origins | ✖ |
| `.env.example` | Шаблон переменных окружения | ✖ |

---

## Фаза 6: Tauri + React Frontend ✖ НЕ НАЧАТО (0%)

> **Зависимости:** Фаза 5 (API)

### 6.1 Инициализация проекта

```bash
cd RegenTwin
npm create tauri-app@latest ui -- --template react-ts
cd ui
npm install react-router-dom axios zustand @tanstack/react-query
npm install plotly.js react-plotly.js d3 three @react-three/fiber
npm install tailwindcss postcss autoprefixer @headlessui/react
npm install -D vitest @testing-library/react @testing-library/jest-dom jsdom
```

### 6.2 Интеграция визуализации из Фазы 4

> **Важно:** API endpoints из Фазы 4 (`/api/viz/*`) возвращают Plotly JSON,
> который React-компоненты потребляют через `react-plotly.js`.
> Все 8 endpoints готовы — нужно только создать React-обёртки.

```tsx
// Пример интеграции:
import Plot from 'react-plotly.js';

const response = await fetch('/api/viz/populations', {
  method: 'POST',
  body: JSON.stringify({ simulation: { t_max_hours: 720, prp_enabled: true } })
});
const plotData = await response.json();
<Plot data={plotData.data} layout={plotData.layout} />
```

**Готовые API endpoints (Фаза 4):**
- `POST /api/viz/populations` → Plotly JSON (8 популяций)
- `POST /api/viz/cytokines` → Plotly JSON (7 цитокинов)
- `POST /api/viz/ecm` → Plotly JSON (ECM)
- `POST /api/viz/phases` → Plotly JSON (фазы заживления)
- `POST /api/viz/comparison` → Plotly JSON (4 сценария)
- `POST /api/viz/export/csv` → CSV файл
- `POST /api/viz/export/png` → PNG файл
- `POST /api/viz/export/pdf` → PDF отчёт

### 6.3 Компоненты React

| Компонент | Файл | API endpoint | Статус |
|-----------|------|-------------|--------|
| **Upload** | `components/Upload/UploadFCS.tsx` | POST /api/v1/upload | ✖ |
| **Parameters** | `components/Parameters/TherapyConfig.tsx` | — (local state) | ✖ |
| **ModelSelector** | `components/Parameters/ModelSelector.tsx` | — (local state) | ✖ |
| **Simulation** | `components/Simulation/SimulationRunner.tsx` | POST /api/v1/simulate | ✖ |
| **PopulationCharts** | `components/Visualization/PopulationCharts.tsx` | POST /api/viz/populations | ✖ |
| **CytokineCharts** | `components/Visualization/CytokineCharts.tsx` | POST /api/viz/cytokines | ✖ |
| **ECMCharts** | `components/Visualization/ECMCharts.tsx` | POST /api/viz/ecm | ✖ |
| **PhaseTimeline** | `components/Visualization/PhaseTimeline.tsx` | POST /api/viz/phases | ✖ |
| **TherapyComparison** | `components/Visualization/TherapyComparison.tsx` | POST /api/viz/comparison | ✖ |
| **Heatmap** | `components/Visualization/CellHeatmap.tsx` | (будущий spatial API) | ✖ |
| **InflammationMap** | `components/Visualization/InflammationMap.tsx` | (будущий spatial API) | ✖ |
| **AnimationPlayer** | `components/Visualization/AnimationPlayer.tsx` | (будущий spatial API) | ✖ |
| **3D View** | `components/Visualization/SpatialView3D.tsx` | Three.js ABM | ✖ |
| **SensitivityView** | `components/Analysis/SensitivityView.tsx` | (Фаза 3.6 API) | ✖ |
| **ExportPanel** | `components/Results/ExportPanel.tsx` | POST /api/viz/export/* | ✖ |

### 6.3 Страницы

| Страница | Route | Описание | Статус |
|----------|-------|----------|--------|
| Home | `/` | Описание, Quick Start | ✖ |
| Dashboard | `/dashboard` | Upload → Parameters → Simulation | ✖ |
| Results | `/results/:id` | Полная визуализация | ✖ |
| Analysis | `/analysis/:id` | Sensitivity, estimation | ✖ |
| History | `/history` | История симуляций | ✖ |
| Settings | `/settings` | Настройки | ✖ |

---

## Фаза 7: Тестирование ◐ РЕАЛИЗОВАНО (55% → цель 98%)

> **Примечание:** 1467 тестов. Покрыты: data pipeline (6 модулей, 532 теста), core MVP (5 модулей, 100+214+100+97+68 тестов), therapy_models (124 теста), extended_sde/wound_phases/parameters (полностью реализованы — 142+54+36 тестов).

| Категория | Текущий статус | Целевое состояние |
|-----------|---------------|-------------------|
| Unit-тесты (core MVP) | ✔ ~1500 LOC | ✔ Сохранить |
| Unit-тесты (data) | ✔ ~600 LOC | ✔ Сохранить |
| Integration-тесты | ✔ 307 LOC | Расширить для Extended SDE |
| Performance-тесты | ✔ ~100 LOC | Добавить бенчмарки Extended SDE |
| **Unit-тесты (Extended SDE)** | ✔ 142 теста | ✔ |
| **Unit-тесты (therapy models)** | ✔ (124 теста) | ~1060 LOC |
| **Unit-тесты (numerics)** | ✖ | ~300 LOC |
| **Unit-тесты (analysis)** | ✖ | ~500 LOC |
| **Unit-тесты (API)** | ◐ 17 тестов (visualization routes) | ~600 LOC |
| **Unit-тесты (visualization)** | ✔ 84 теста (theme+plots+spatial+export) | ~800 LOC |
| **Validation-тесты (реальные данные)** | ✖ | ~300 LOC |
| **E2E тесты (API + Frontend)** | ✖ | ~400 LOC |

### Покрытие модулей (целевое)

| Модуль | Тесты | Текущий | Целевой |
|--------|-------|---------|---------|
| `sde_model.py` | `test_sde_model.py` | ✔ | ✔ |
| `abm_model.py` | `test_abm_model.py` | ✔ | ✔ |
| `integration.py` | `test_integration.py` | ✔ | ✔ |
| `monte_carlo.py` | `test_monte_carlo.py` | ✔ | ✔ |
| `fcs_parser.py` | `test_fcs_parser.py` | ✔ | ✔ |
| `gating.py` | `test_gating.py` | ✔ | ✔ |
| `parameter_extraction.py` | `test_parameter_extraction.py` | ✔ | ✔ |
| `image_loader.py` | `test_image_loader.py` | ✔ | ✔ |
| **`extended_sde.py`** | `test_extended_sde.py` | ✔ 142 теста | ✔ |
| **`therapy_models.py`** | `test_therapy_models.py` | ✔ 124 теста | ✔ |
| **`sde_numerics.py`** | `test_sde_numerics.py` | ✔ 76 тестов | ✔ |
| **`robustness.py`** | `test_robustness.py` | ✔ 72 теста | ✔ |
| **`abm_spatial.py`** | `test_abm_spatial.py` | ✖ | ✖ → ✔ |
| **`equation_free.py`** | `test_equation_free.py` | ✔ | ✔ (114 passed) |
| **`sensitivity.py`** | `test_sensitivity.py` | ✖ | ✖ → ✔ |
| **`parameter_estimation.py`** | `test_parameter_estimation.py` | ✖ | ✖ → ✔ |
| **API endpoints** | `test_api_*.py` | ✖ | ✖ → ✔ |
| **Visualization** | `test_plots.py` | ✖ | ✖ → ✔ |

---

## Фаза 8: Интеграция и деплой ✖ НЕ НАЧАТО (0%)

### Docker

| Задача | Статус | Описание |
|--------|--------|----------|
| Dockerfile (multi-stage) | ✖ | Builder: UV + deps; Runtime: slim, non-root |
| docker-compose.yml | ✖ | app + postgres + redis |
| .dockerignore | ✖ | .venv, __pycache__, .git, tests, .coverage |
| docker-compose.dev.yml | ✖ | Hot-reload, volume mounts |

### CI/CD (GitHub Actions)

| Задача | Статус | Описание |
|--------|--------|----------|
| `.github/workflows/ci.yml` | ✖ | On push/PR: ruff, mypy, pytest, coverage |
| `.github/workflows/docker.yml` | ✖ | Build + push Docker image |
| `.github/workflows/release.yml` | ✖ | Tauri build artifacts (Windows, macOS, Linux) |
| Coverage badge | ✖ | Codecov / coveralls |
| Pre-commit hooks | ✖ | `.pre-commit-config.yaml` |

### Деплой

| Задача | Статус | Описание |
|--------|--------|----------|
| Деплой API (Railway/Fly.io) | ✖ | FastAPI + PostgreSQL |
| Tauri desktop build | ✖ | Windows MSI, macOS DMG |
| Мониторинг | ✖ | Health check + Sentry (опционально) |
| Документация (MkDocs) | ✖ | API docs + Math framework + User guide |

---

## Порядок реализации (обновлённый)

| # | Задача | Зависимости | Приоритет | Статус |
|---|--------|-------------|-----------|--------|
| 1 | Завершить ABM MVP (хемотаксис, контактное ингиб.) | — | Высокий | ◐ стабы |
| 2 | Завершить интеграцию MVP SDE↔ABM (синхронизация C) | 1 | Высокий | ✔ |
| 3 | Параллелизация Monte Carlo | — | Средний | ✔ |
| 4 | Численная робастность MVP (клиппинг, NaN) | — | Высокий | ✔ |
| 5 | **Расширенная SDE (20+ переменных)** | 4 | КРИТИЧЕСКИЙ | ✔ |
| 6 | Механистические модели терапий (PRP/PEMF) | 5 | КРИТИЧЕСКИЙ | ✔ |
| 7 | Milstein + IMEX для расширенной SDE | 5 | Высокий | ◐ стабы |
| 8 | Расширенная ABM (Neutrophil, Endothelial, Mf, KD-Tree) | 1, 5 | Высокий | ◐ стабы |
| 9 | Equation-Free интеграция (расширенная) | 5, 8 | Высокий | ✔ |
| 10 | Параметрическая идентификация (PyMC/emcee) | 5 | КРИТИЧЕСКИЙ | ✖ |
| 11 | Анализ чувствительности (SALib) | 5 | КРИТИЧЕСКИЙ | ✖ |
| 12 | Загрузка реальных данных + валидация | 10, 11 | КРИТИЧЕСКИЙ | ✖ |
| 13 | Бенчмаркинг (Flegg, Xue, Vodovotz) | 5, 12 | Высокий | ✖ |
| 14 | Визуализация (полная) | 5, 11 | Высокий | ✖ |
| 15 | FastAPI: базовый сервер + CORS + health | — | Средний | ✖ |
| 16 | API: /upload endpoint | 15 | Средний | ✖ |
| 17 | API: /simulate endpoint + WebSocket | 15, 14 | Средний | ✖ |
| 18 | API: /results + /export | 17 | Средний | ✖ |
| 19 | API: /analysis endpoints | 17, 11 | Средний | ✖ |
| 20 | DB: SQLAlchemy модели + Alembic | 15 | Средний | ✖ |
| 21 | Tauri: инициализация проекта | — | Средний | ✖ |
| 22 | React: структура, роутинг, API client | 21, 15 | Средний | ✖ |
| 23 | React: Upload + Parameters | 22, 16 | Средний | ✖ |
| 24 | React: Simulation + Progress | 23, 17 | Средний | ✖ |
| 25 | React: Visualization (Plotly, D3, Three.js) | 24, 18 | Средний | ✖ |
| 26 | React: Analysis views | 25, 19 | Средний | ✖ |
| 27 | Docker (Dockerfile, compose) | 18 | Средний | ✖ |
| 28 | CI/CD (GitHub Actions: lint+test+coverage) | 27 | Средний | ✖ |
| 29 | Tauri desktop build | 25 | Низкий | ✖ |
| 30 | Документация (MkDocs, Jupyter notebooks) | 12, 13 | Средний | ✖ |
| 31 | Препринт bioRxiv (Methods section) | 12, 13, 30 | КРИТИЧЕСКИЙ | ✖ |

---

## Критические файлы

| Файл | Описание | Статус |
|------|----------|--------|
| `pyproject.toml` | Зависимости включая PyMC, SALib, emcee, celery | ✔ |
| `src/core/extended_sde.py` | **КЛЮЧЕВОЙ:** 20+ переменных SDE система | ✔ Реализован (1104 LOC) |
| `src/core/therapy_models.py` | Механистические PRP/PEMF модели | ✔ Реализован (583 LOC) |
| `src/core/sde_numerics.py` | EM, Milstein, IMEX, Adaptive, SRK — все реализованы (880 LOC) | ✔ |
| `src/core/abm_model.py` | Расширение ABM, стаб-классы агентов | ◐ |
| `src/core/abm_spatial.py` | KD-Tree, хемотаксис, контактное ингибирование | ✖ |
| `src/core/equation_free.py` | Equation-Free мультимасштабная интеграция | ✔ |
| `src/core/parameters.py` | 80+ параметров из литературы | ✔ Реализован (334 LOC) |
| `src/core/robustness.py` | Верификация реализована: 5 классов, 148 тестов (583 LOC) | ✔ |
| `src/analysis/sensitivity.py` | Sobol, Morris (SALib) | ✖ |
| `src/analysis/parameter_estimation.py` | Bayesian estimation (PyMC) | ✖ |
| `src/analysis/validation.py` | Метрики валидации на реальных данных | ✖ |
| `src/api/main.py` | FastAPI app | ✖ |
| `src/visualization/plots.py` | Графики для 20+ переменных | ✖ |
| `ui/src/App.tsx` | React роутинг | ✖ |
| `Dockerfile` | Контейнеризация | ✖ |
| `.github/workflows/ci.yml` | CI/CD pipeline | ✖ |

---

## Сводка по прогрессу

| Фаза | Название | Статус | Прогресс |
|------|----------|--------|----------|
| 0 | Инфраструктура и DevOps | ◐ Частично | 70% |
| 1 | Data Pipeline | ✔ Реализовано | 100% |
| 2 | Математическое ядро MVP | ✔ Реализовано | 95% |
| 2.5 | Расширенная SDE (20+ переменных) | ✔ Реализовано (1104 LOC, 142 теста) | 100% |
| 2.6 | Механистические модели терапий | ✔ Реализовано (124/124 тестов) | 100% |
| 2.7 | Численные методы и робастность | ✔ Реализовано (148 тестов, simulate() → Фаза 3) | 95% |
| 2.8 | Расширенная ABM | ✔ Реализовано (429 тестов, 8 классов, 7 механик) | 100% |
| 2.9 | Мультимасштабная интеграция | ✖ Не начато | 0% |
| 3 | Анализ и валидация | ✖ Не начато | 0% |
| 4 | Визуализация | ✖ Не начато | 0% |
| 5 | FastAPI Backend | ✖ Не начато | 0% |
| 6 | Tauri + React Frontend | ✖ Не начато | 0% |
| 7 | Тестирование | ◐ Частично | 60% (1467 тестов) |
| 8 | Интеграция и деплой | ✖ Не начато | 0% |

### Общий прогресс: ~30% (от полной модели)

### Созданные файлы (проверено)

**Python Backend — `src/core/` (11 файлов):**
- `sde_model.py` ✔ (реализован, 575 LOC)
- `extended_sde.py` ✔ (ПОЛНОСТЬЮ РЕАЛИЗОВАН — StateIndex(20), 50+ методов, все 20 SDE, 1104 LOC)
- `abm_model.py` ◐ (2335 LOC — базовая ABM реализована, стаб-классы Neutrophil/Endothelial/Myofibroblast, 2 стаба движения)
- `integration.py` ✔ (реализован — bidirectional SDE+ABM sync, operator splitting, 836 LOC)
- `monte_carlo.py` ✔ (реализован — ensemble + concurrent.futures параллелизация, 872 LOC)
- `numerical_utils.py` ✔ (реализован — clip, detect_divergence, adaptive_timestep, NumericalGuard, 382 LOC)
- `therapy_models.py` ✔ (реализован — PRPModel, PEMFModel, SynergyModel, 124 теста, 99% coverage, 583 LOC)
- `parameters.py` ✔ (реализован — ParameterSet 80+ полей, 334 LOC)
- `wound_phases.py` ✔ (реализован — WoundPhaseDetector, 4 фазы, 8 методов, 331 LOC)
- `sde_numerics.py` ✔ (880 LOC — EM, Milstein, IMEX, Adaptive PI, SRK реализованы; simulate() оставлен для Фазы 3)
- `robustness.py` ✔ (583 LOC — PositivityEnforcer, NaNHandler, ConservationChecker, ConvergenceVerifier, SDEvsABMComparator)

**Python Backend — `src/data/` (6 файлов):** ✔ ВСЕ РЕАЛИЗОВАНЫ
- `fcs_parser.py`, `gating.py`, `parameter_extraction.py`, `image_loader.py`, `validation.py`, `dataset_loader.py`

**Python Backend — другие:**
- `src/api/` — только __init__.py файлы ✖
- `src/visualization/` — только __init__.py ✖
- `src/analysis/` — не создан ✖
- `src/db/` — не создан ✖

**Тесты (1615 тестов):**
- `tests/unit/data/` — 6 файлов, 532 теста ✔
- `tests/unit/core/` — 9 файлов (sde, abm, integration, monte_carlo, therapy_models, parameters, extended_sde, wound_phases, numerical_utils) ✔
- `tests/integration/` — 1 файл ✔
- `tests/performance/` — 1 файл ✔
- `tests/conftest.py` — fixtures ✔

**Документация:**
- `Description/Phase1/` — 6 файлов описаний ✔
- `Description/Phase2/` — 9 файлов описаний (sde, abm, integration, monte_carlo, numerical_utils, therapy_models, parameters, extended_sde, wound_phases) ✔
- `Doks/RegenTwin_Mathematical_Framework.md` — полная математическая модель ✔
- `developers_plans/` — phase2_plan.md, stub_template.md, phase1_completed.md ✔
- `data/mock/` — generate_mock_data.py + README.md ✔

**Конфигурация:**
- `pyproject.toml` ✔ (PyMC, SALib, emcee, celery уже добавлены)
- `ruff.toml` ✔
- `.gitignore` ✔

---

## Приоритетная дорожная карта

### Milestone 1: «Расширенная модель»
- [x] Фаза 2 — завершение MVP (интеграция, параллелизация, робастность) ✔
- [x] Фаза 2.5 — Расширенная SDE (20+ переменных) ✔
- [x] Фаза 2.6 — Механистические терапии ✔
- [x] Фаза 2.7 — Milstein + IMEX + Adaptive + SRK + Robustness (✔ 148 тестов)
- [x] Фаза 2.8 — Расширенная ABM (✔ 429 тестов, 8 классов, 7 механик)
- [x] Фаза 2.9 — Equation-Free интеграция (114 тестов, black/ruff/mypy чисто)

### Milestone 2: «Валидация для публикации»
- [ ] Фаза 3 — Анализ чувствительности + параметрическая идентификация
- [ ] Загрузка реальных данных
- [ ] Валидация на публичных датасетах
- [ ] Бенчмаркинг vs Flegg/Xue/Vodovotz

### Milestone 3: «Демонстрируемый продукт»
- [ ] Фаза 4 — Визуализация
- [ ] Фаза 5 — FastAPI Backend
- [ ] Фаза 6 — Tauri + React Frontend

### Milestone 4: «Production-ready»
- [ ] Фаза 0 (завершение) — Docker, CI/CD, DB
- [ ] Фаза 8 — Деплой, документация
- [ ] Подготовка препринта bioRxiv

---

## Таблица параметров модели (из математического фреймворка)

### Клеточные параметры

| Параметр | Значение | Единицы | Описание | Источник |
|----------|----------|---------|----------|----------|
| r_F | 0.03 | ч⁻¹ | Пролиферация фибробластов | Vodovotz 2006 |
| r_E | 0.02 | ч⁻¹ | Пролиферация эндотелия | Anderson 1998 |
| r_S | 0.01 | ч⁻¹ | Самообновление стволовых | Badiavas 2003 |
| δ_P | 0.1 | ч⁻¹ | Клиренс тромбоцитов | Nurden 2008 |
| δ_Ne | 0.05 | ч⁻¹ | Апоптоз нейтрофилов | Kolaczkowska 2013 |
| δ_M | 0.01 | ч⁻¹ | Апоптоз макрофагов | Murray 2017 |
| δ_F | 0.003 | ч⁻¹ | Апоптоз фибробластов | Hinz 2007 |
| k_switch | 0.02 | ч⁻¹ | M1→M2 переключение | Mantovani 2004 |
| k_act | 0.01 | ч⁻¹ | F→Mf активация | Hinz 2007 |
| K_F | 5×10⁵ | кл/мкл | Carrying capacity F+Mf | Flegg 2015 |

### Параметры цитокинов

| Параметр | Значение | Единицы | Описание | Источник |
|----------|----------|---------|----------|----------|
| γ_TNF | 0.5 | ч⁻¹ | Деградация TNF-α | Bradley 2008 |
| γ_IL10 | 0.3 | ч⁻¹ | Деградация IL-10 | Mosser 2008 |
| γ_PDGF | 0.2 | ч⁻¹ | Деградация PDGF | Heldin 1999 |
| γ_VEGF | 0.3 | ч⁻¹ | Деградация VEGF | Ferrara 2004 |
| γ_TGF | 0.15 | ч⁻¹ | Деградация TGF-β | Leask 2004 |
| s_TNF_M1 | 0.01 | нг/(мл·кл·ч) | Секреция TNF M1 | Bradley 2008 |
| s_IL10_M2 | 0.008 | нг/(мл·кл·ч) | Секреция IL-10 M2 | Mosser 2008 |

### Параметры ECM

| Параметр | Значение | Единицы | Описание | Источник |
|----------|----------|---------|----------|----------|
| q_F | 0.005 | ед/ч | Продукция коллагена F | Xue 2009 |
| q_Mf | 0.015 | ед/ч | Продукция коллагена Mf | Desmouliere 2005 |
| k_MMP | 0.02 | ч⁻¹ | Деградация коллагена MMP | Gill 2008 |

### Параметры PRP

| Параметр | Значение | Единицы | Описание | Источник |
|----------|----------|---------|----------|----------|
| PRP_dose | 3-5x | кратность | Концентрация тромбоцитов | Marx 2004 |
| τ_burst | 1-2 | ч | Быстрое высвобождение | Giusti 2009 |
| τ_sustained | 24-72 | ч | Замедленное высвобождение | Giusti 2009 |

---

## Верификация

```bash
# Backend
cd RegenTwin
uvicorn src.api.main:app --reload
# http://localhost:8000/docs — Swagger UI

# Тесты (все)
pytest -v --cov=src --cov-report=term-missing --cov-report=html

# Тесты (только расширенная модель)
pytest tests/unit/core/test_extended_sde.py -v
pytest tests/unit/core/test_therapy_models.py -v
pytest tests/unit/core/test_sde_numerics.py -v

# Тесты (анализ)
pytest tests/unit/analysis/ -v
pytest tests/validation/ -v -m validation

# Линтинг
ruff check src/
mypy src/
black --check src/

# Frontend (dev)
cd ui
npm run tauri dev

# Frontend (web only)
npm run dev
# http://localhost:5173

# Frontend тесты
npm run test

# Docker
docker compose up --build
# http://localhost:8000/docs

# Build desktop
npm run tauri build

# Sensitivity analysis (standalone)
python -m src.analysis.sensitivity --config config/sensitivity.yaml

# Parameter estimation (standalone)
python -m src.analysis.parameter_estimation --data data/validation/ --output results/
```

---

*Документ обновлён: 8 марта 2026*
*Версия: 4.4 (Phase 2–2.8 реализованы; 2044+ тестов)*
*Основан на: RegenTwin_Mathematical_Framework.md, RegenTwin_Implementation_Plan.md v3.0*
