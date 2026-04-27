# RegenTwin — Project Audit (2026-04-27)

> Read-only audit для диплома + планирования НИРС/патент/PhD. Без правок кода.
> Источники: live read репозитория `master @ b0bc8ca`, `output/diagnosis_report.json` (2026-04-20), 3 параллельных read-only sub-agent passes + ручная верификация конфликтов.

## 1. Executive Summary

- **Проект НЕ сломан.** `output/diagnosis_report.json` — все 95 проверок OK (импорты 35+ модулей, 4 режима симуляции, 41 API route, M1/M2 биология `early=1.44 / late=0.24`, БД, Plotly, экспорт). README устарел в части «нужно чинить».
- **Зрелость по компонентам (моя оценка):** SDE-ядро ~80 %, ABM ~55 %, SDE↔ABM coupling ~40 % и **отключено в API (501)**, FastAPI/Tauri/Frontend ~75 % (production-grade pipeline), валидация vs published models **0 %**, тесты по строкам 31 979 при src 25 980 (тесты структурные, не научные).
- **Главный риск для публикации.** Нет ни одного численного сравнения с опубликованной моделью раны (Flegg/Xue/Vodovotz/Reynolds). Файл `src/analysis/benchmarking.py` отсутствует, упомянутый в плане; `literature_data.py` хранит **fitted curves**, а не extracted numerical solutions из статей. Без бенчмарка любой рецензент Computational Biology / J Theor Biol заблокирует submit.
- **Главный технический долг.** «Equation-Free Framework» (`equation_free.py`) — это **lift-evolve-restrict шаблон**, а не Kevrekidis equation-free в строгом смысле (нет coarse-equation discovery, нет closure problem). Маркетинг как «equation-free» — уязвимость на защите перед знакомой панелью.
- **Что Альфер реально может защищать сегодня без доделок.** 20-переменную extended-SDE модель с цитированной биологией (M1↔M2, TGF-β bistability, oxygen-angiogenesis), PRP-кинетику (биофазная, Marx/Giusti/Eppley), полнофункциональный FastAPI+Tauri+R3F стэк. Это сильный CS/инженерный диплом, но не computational biology paper.
- **AI-ассистированность видна, но дисциплинированная.** Русские domain-specific комментарии, инлайн-математика, ссылки `Description/PhaseX/description_*.md#func` к внешним описаниям, отсутствие AI-boilerplate. Скорее «AI с серьёзным review автором», чем «vibe-coded».

## 2. Codebase Statistics

| Область | Файлов | LoC | Комментарий |
|---|---:|---:|---|
| `src/` (Python backend) | 66 | **25 980** | core 17 модулей, api 4 router-а + 3 service, data 9, db, tasks, analysis, viz |
| `tests/` (pytest) | 80 | **31 979** | тестов больше чем кода (1.23×); coverage 87 % |
| `scripts/` | — | 1 448 | diagnose.py, e2e_*, benchmark_abm_mc.py |
| `ui/src/` (React/TS) | 57 | **7 029** | 7 страниц, R3F-вьюер, Zustand, TanStack Query |
| `ui/src-tauri/src/` (Rust) | — | **168** | пустой launcher, **никакой симуляции** |

```
RegenTwin/
├── src/
│   ├── core/              # 17 модулей: extended_sde, abm_*, equation_free, monte_carlo, therapy_*, sde_numerics, robustness, sensitivity_analysis, parameter_estimation, ...
│   ├── api/               # FastAPI: routes/{simulate,results,analysis,visualization,spatial,parameters,upload,health}, services/, models/schemas.py
│   ├── data/              # fcs_parser, gating, parameter_extraction (813 LoC), literature_data (473), hpa_client (240), gene_mapping, dataset_loader
│   ├── db/                # SQLAlchemy sync (SQLite), 3 таблицы; alembic.ini есть, но миграции пустые
│   ├── tasks/             # celery_app + simulation_tasks (опциональный путь, по умолчанию выключен)
│   ├── analysis/          # validation, validation_pipeline (DTW/CRPS/PPC/Pelt+BIC/Kendall — каркас)
│   └── visualization/     # plots, spatial, analysis_plots, theme, export (Plotly+kaleido)
├── ui/
│   ├── src/               # 7 routes, Three.js/R3F, Zustand, vitest, Playwright e2e
│   └── src-tauri/         # Rust launcher 168 LoC: uv → .venv → python
├── tests/{unit,integration,performance,e2e,fixtures}
├── data/{uploads(130M real FCS), results(177M .npz), validation(5K placeholder), mock(stub)}
├── Doks/                  # 6 markdown-плана/верификация (PhD plan, Therapy optim, ...)
└── scripts/               # diagnose.py, e2e_math_checks.py, benchmark_abm_mc.py
```

**Ключевые зависимости (`pyproject.toml`):** numpy, scipy, pandas, **flowkit** (FCS parsing — реальная библиотека), plotly+kaleido, fastapi+uvicorn, sqlalchemy+alembic, **pymc + emcee + arviz** (для Bayesian — установлены, но estimation API возвращает 501), **SALib** (Sobol работает, Morris отключён в API), **celery + redis** (опционально, выключено по умолчанию), dtaidistance / properscoring / ruptures (метрики валидации).

**CI/CD:** **отсутствует**. Нет `.github/workflows/`, нет `.gitlab-ci.yml`, нет `.pre-commit-config.yaml`. Тесты никогда не запускаются автоматически — только локально.

## 3. Biological Core

### 3.1 SDE System — статус: **работает, средне-публикуема**

- **20 переменных** (`StateIndex`, `extended_sde.py`): 8 клеточных (P, Ne, M1, M2, F, Mf, E, S) + 7 цитокинов (TNF, IL10, PDGF, VEGF, TGFβ, MCP1, IL8) + 3 ECM (collagen, MMP, fibrin) + 2 вспомогательных (D — damage, O₂).
- **Solver:** **Euler-Maruyama 1-го порядка**, шаг dt=0.01, мультипликативный шум `σ_i × X_i` для клеточных, **нулевой шум для ECM/auxiliary** ([extended_sde.py:422-449](src/core/extended_sde.py#L422-L449)). Никакого Milstein / Stratonovich / IMEX в активном коде нет, несмотря на каркас в `sde_numerics.py`.
- **Параметры (`parameters.py`):** ~80 значений, **все с inline-цитатами author-year** — Xue 2009, Flegg 2010/2015, Anderson 1998, Bradley 2008, Mosser 2008, Hinz 2007, Mantovani 2004, Gurtner 2008, Eming 2014. **Ни одного DOI/PMID** во всём `src/core/`.
- **Не упомянуты вообще:** Vodovotz 2004 (sepsis ODE), Reynolds endotoxin, Mi 2007, Nagaraja 2014, Buganza Tepole, Day-Rubin, Menon. Это четыре конкретные модели, с которыми обычно сравнивают wound healing работы.
- **Биологическая валидация в коде:** только проверка ratio M1/M2 в `diagnose.py` (early > 1, late < 1). Других инвариантов (mass conservation, монотонность collagen на phase remodeling, clearance цитокинов поздно) **нет**.
- **Bounds:** `_apply_boundary_conditions()` — `np.maximum(x, 0)` ([extended_sde.py:1076-1095](src/core/extended_sde.py#L1076-L1095)). Верхних физиологических насыщений нет, кроме `K_F` carrying capacity.

### 3.2 Agent-Based Model — статус: **работает, не калибрована**

- **Off-lattice 2D**, 100×100 µm, периодик/reflective/absorbing границы ([abm_model.py:34-35](src/core/abm_model.py#L34-L35)).
- **7 типов агентов** (`abm_spatial.py`): Platelet, Neutrophil, Macrophage (с `polarization_state` M1/M2), Fibroblast, Myofibroblast, Endothelial, StemCell.
- **Соседи:** SpatialHash O(1) или KDTree O(log n); interaction radius 5 µm, contact inhibition 2 µm.
- **Update:** синхронный per-agent внутри timestep ([equation_free.py:484-496](src/core/equation_free.py#L484-L496)) — `update(dt, env)` затем `divide()`. Никакого event-driven queueing.
- **Производительность:** заявлено `max_agents=10000`, **бенчмарков нет** в коде; `scripts/benchmark_abm_mc.py` меряет MC fan-out, не ABM scale.
- **Биологическая калибровка отсутствует.** Lifespans, division rates, chemotaxis sensitivity — все из конфига, никаких ссылок на лабораторные измерения.

### 3.3 SDE↔ABM Coupling — статус: **частично, отключено в API**

- **Файлы:** [integration.py](src/core/integration.py), [equation_free.py](src/core/equation_free.py).
- **Механика:**
  1. **Lift (macro→micro):** SDE-state → агенты. Для каждого клеточного типа `n_i = round(conc × volume × n_agents_scale)`, агенты равномерно распределены ([equation_free.py:153-200](src/core/equation_free.py#L153-L200)).
  2. **Micro-step:** ABM эволюционирует `n_micro_steps` (default 10).
  3. **Restrict (micro→macro):** count alive agents per type / scale → новый SDE-state ([equation_free.py:278-332](src/core/equation_free.py#L278-L332)). Цитокины усредняются по агентам; ECM сохраняется в `_macro_context`.
- **Двунаправленный** (по структуре).
- **«Equation-free»? Нет, не в строгом смысле.** Это lift-evolve-restrict шаблон. **Нет coarse-equation discovery, нет closure problem solution, нет timestepper acceleration.** На защите перед панелью, знающей Kevrekidis (PNAS 2009), это уязвимость. Рекомендую переименовать во внутренней документации в «multiscale operator splitting» или «patch dynamics-style coupling».
- **API:** режим `integrated` возвращает 501 ([README.md:14](README.md#L14)). Код выполняется в `_run_integrated()` внутри `simulation_service.py`, но роутер блокирует. То есть это «реализовано, но скрыто», а не «сломано».
- **Валидации coupling нет.** Никакой проверки, что restrict воспроизводит SDE density (mass conservation теста), что lift→evolve→restrict сохраняет M1/M2 ratio. Без этого «multiscale» — заявка, не результат.

### 3.4 Therapies (`therapy_models.py`)

- **PRP — публикуема.** Биофазная кинетика релиза `Θ_PRP_i(t) = dose · c0_i · (exp(-t/τ_burst) - exp(-t/τ_sustained)) / (τ_burst-τ_sustained) · decay` ([therapy_models.py:199-244](src/core/therapy_models.py#L199-L244)). 4 фактора: PDGF/VEGF/TGFβ/EGF. burst 0.5–2h (α-granules), sustained 12–72h (fibrin matrix). Цитаты Marx 2004, Giusti 2009, Eppley 2006, Anitua 2004. Это механистическая модель.
- **PEMF — phenomenological/toy.** Frequency-response множители (опт. 27 Hz anti-inflam, 75 Hz prolif, 50 Hz migration). Цит. Pilla 2013, Varani 2017, Onstenk 2015. **Нет реальной Ca²⁺/CaM/NO динамики** — линейные multiplier-ы. На защите безопасно подавать как «phenomenological response surface», не как mechanism.
- **Synergy** PRP+PEMF — hardcoded β=0.2, не валидирован.

### 3.5 Numerical / robustness

- **`sde_numerics.py`:** 0 `NotImplementedError`. Каркасы для Milstein/IMEX/adaptive есть, но в основном пути симуляции (`extended_sde.py:simulate`) **используется только Euler-Maruyama**. README прав в части «stubs не выведены в active product surface».
- **`robustness.py`:** датаклассы `ViolationStats`, `ConservationReport`, `ConvergenceResult`. Тесты сходимости есть, но требуют reference solutions, которых в репозитории **нет**.
- **`bounds.py`:** ~80 параметрических границ для sensitivity analysis — это рабочий артефакт.

## 4. Engineering Layer

### 4.1 FastAPI Backend — production-capable

- **41 route**, все возвращают 200 в диагностике.
- **POST `/api/v1/simulate` НЕ блокирует event loop.** Async endpoint → создаёт DB-запись → запускает `threading.Thread` через `simulation_service._run_in_background()` → возвращает 200 с `simulation_id`. Клиент опрашивает `/simulate/{id}` или WebSocket `/{id}/ws`.
- **Persistence:** результат сохраняется в `data/results/{id}/*.npz`, БД хранит метаданные.
- **Cancellation:** кооперативная через `SimulationCancelledError` + `cancel_event`. Реальная отмена работает.
- **501-роуты:**
  - `POST /api/v1/simulate` mode=`integrated` → 501.
  - `POST /api/v1/analysis/estimation` → 501 (хотя `parameter_estimation.py` импортирует pymc, emcee, реализован — отключение преднамеренное).
  - Morris в `analysis/sensitivity` — отключён.

### 4.2 Tauri Desktop — рабочий launcher без симуляции

- `ui/src-tauri/src/lib.rs`: 168 строк, единственная команда `greet()` (заглушка). Spawn-логика:
  1. `uv run uvicorn src.api.main:app` (если найден `uv`)
  2. `.venv/Scripts/python.exe` или `.venv/bin/python`
  3. `python` из PATH
- Polls TCP port 8000 (5ms checks, 10s timeout). Kills child on shutdown.
- **В Rust ноль научного кода.** Заявка «переписать hot loops на Rust» — это greenfield-работа.

### 4.3 Frontend — впечатляющий уровень

- **7 страниц** (`Home, Dashboard, Analysis, Results, History, Settings, About`), все потребляют реальные backend-данные через TanStack Query + кастомные хуки (`useSimulation`, `useAnalysis`, `useResults`, `useSimulationWS`, `useSpatialData`, `useVisualization`).
- **R3F 3D viewer** (`SpatialView3D.tsx`) рендерит **реальные позиции ABM-агентов** из `/api/viz/spatial/scatter`, не demo-кубики.
- **Zustand** — `simulationStore` (mode, params, run id; persisted localStorage), `uiStore` (theme, sidebar).
- **WebSocket** consume для прогресса.
- **Tests:** vitest unit + Playwright e2e (7 spec-ов: mvp/extended/abm flow, cancel, websocket, errors, navigation), запускаются против Tauri dev-port 1420.

### 4.4 Tests

- **pytest:** 2358 passed, coverage 87 % (заявлено в README). Структура `tests/{unit,integration,performance,e2e,fixtures}`.
- **Качество тестов:** агент 3 семплировал 5 модулей `tests/unit/core/` — `test_extended_sde.py`, `test_wound_phases.py`, `test_abm_model.py`, `test_therapy_models.py`, `test_sde_numerics.py`. Все 5 — **структурные** (проверяют схемы dataclass-ов, enum-значения, дефолты конфига). **Ни одного fixed-seed regression теста на SDE-траекторию**. Биоинвариант (M1/M2) проверяется только в `diagnose.py`, не в pytest.
- **e2e:** `tests/e2e/` — против live uvicorn (markers `e2e`, `e2e_slow`).
- **Coverage 87 %** означает «строчки выполняются», а не «логика правильна».

### 4.5 CI/CD

**Отсутствует полностью.** Это — самый дешёвый quick-win.

### 4.6 Build/launch

- `setup_and_run.{bat,sh}` — рабочие, не скелеты: `uv sync` → `npm install` → kill port 8000 → uvicorn + npm run dev.
- `ui/build_tauri.bat` — есть, не проверял глубоко.
- `cargo target-dir = ui/src-tauri/target-tauri` — локально, не глобально.

### 4.7 DB

- 3 таблицы: `simulations`, `uploads`, `analyses`. JSON-блобы для params/result.
- **Sync SQLAlchemy + SQLite + WAL.** Async ORM нет.
- **Alembic — `alembic.ini` есть, миграции пустые.** Схема создаётся через `Base.metadata.create_all()` в lifespan.
- README упоминает `ResourceWarning: unclosed database` в тестах — мелочь, но индикатор недоведения.

## 5. Mock Data Inventory

| Артефакт | Размер | Реальность | Комментарий |
|---|---:|---|---|
| `data/uploads/` | 130 МБ | ✅ реальные | 60+ FCS-файлов от пользователя, parsed через `flowkit` |
| `data/results/` | 177 МБ | ✅ реальный runtime | 16 `.npz` симуляций |
| `data/regentwin.db` | 287 КБ | ✅ runtime | SQLite metadata |
| `data/validation/FR-FCM-wound-healing/time_series/baseline_fractions.csv` | 1 строка | ❌ placeholder | Заявлен FlowRepository download, на деле — одна синтетическая запись (t=0) |
| `data/mock/generate_mock_data.py` | — | ❌ полный stub | **9 `NotImplementedError`** во всех методах генерации |
| `src/data/literature_data.py` | 473 LoC | ⚠️ hardcoded fitted curves | Параметрические `1e4 * exp(-0.1*t) * (1 - exp(-t/2))` для платок, нейтрофилов, макрофагов. **Не extracted** из Xue 2009 figures — это аппроксимации автора, выглядящие как Xue-shape |
| `src/data/hpa_client.py` | 240 LoC | ⚠️ hardcoded lookup | Заявка «HPA v25.0 API» на деле — 7 жёстко прописанных `HPASkinExpression`-ов с RNA nTPM и приблизительной конверсией в ng/mL. **Никакого HTTP-запроса к proteinatlas.org** в коде |
| `src/data/parameter_extraction.py` | 813 LoC | ✅ настоящая | Реальный FCS-gating → `n0`, `c0`, `inflammation_level`, `stem_cell_fraction` через `flowkit.Sample` + `gating.py`. Это **самый сильный data-модуль** в проекте |
| `src/data/gating.py` | — | ✅ настоящая | 16 функций, 0 stub |

**TODO/FIXME/MOCK hunt в `src/`:** `MOCK`, `FIXME`, `XXX`, `HACK`, `PLACEHOLDER` — **0 совпадений** в `src/`. `stub` — 3 совпадения, все в смысле «stub data для тестов» / «stub sensitivity result когда SALib не установлен». То есть **техдолга «не доделал и забыл» в production-коде минимум** — что подтверждает дисциплинированность.

**Утверждение Альфера «all simulations run on mock data, blocking publication» — частично верно:**
- Симуляции работают на **реальных IC из FCS** или на дефолтах из `parameters.py`.
- Дефолты `parameters.py` — citation-аннотированные, но **author-year без DOI**, и не было независимой верификации значений.
- Сравнения с published time-series **нет ни одного**.
- Отсюда восприятие «всё mock» — корректное в смысле «нет real-world validation», но не в смысле «генерируем fake numbers».

## 6. Validation Gap Analysis

**Что есть:**
- Каркас `src/analysis/validation.py` + `validation_pipeline.py`: dataclass-ы для DTW+CRPS, Pelt+BIC changepoint, PPC через ArviZ, Kendall τ. Pipeline-структура (`DatasetLoader → ExtendedSDE → MonteCarlo → ValidationRunner`) выглядит корректно.
- `scripts/e2e_math_checks.py` — реальные инвариант-чеки (NaN, success_rate, биологическая позитивность, therapy delta vs baseline).
- `scripts/benchmark_abm_mc.py` — реальный perf-бенчмарк (serial vs parallel MC).

**Что отсутствует — критично для публикации:**
- **`src/analysis/benchmarking.py` — файл НЕ существует**, хотя упомянут в `Doks/RegenTwin_Update_Implemention_Plan.md`. Это и есть «сравнение vs Flegg/Xue/Vodovotz/Reynolds», которого нет.
- Numerical reference solutions (Xue Figure 3-5 как CSV) не оцифрованы.
- ValidationRunner вычисляет метрики **на model output vs model output**, то есть self-referential — это не валидация.
- Никаких аналитических test-кейсов (например, single-cytokine decay, single-population logistic — должно сходиться к close-form).
- Нет fixed-seed SDE regression test-а на хеш траектории.

**Минимум для submit-able paper:**
1. Численный сайд-бай-сайд vs **одной** опубликованной модели (Xue 2009 PNAS — самый дешёвый, биология та же).
2. 5–7 биологических инвариантов (M1/M2 timing, монотонность collagen в remodeling, clearance TNF-α к t≈300h, mass conservation в coupling, и т.д.) как pytest-cases.
3. Numerical convergence study (dt → 0) для Euler-Maruyama хотя бы на 2-var MVP — стандартная reviewer-проверка.

## 7. Top 10 Technical Debts

| # | Severity | Долг | Effort |
|---|---|---|---|
| 1 | **Critical** | Нет валидации vs published wound-healing models. `benchmarking.py` отсутствует. Без артефакта «Xue 2009 vs RegenTwin» — paper не пройдёт первый круг review. | 3–5 дней |
| 2 | **Critical** | `hpa_client.py` маркетируется как HPA-клиент, но это hardcoded lookup. `literature_data.py` — fitted curves, не extracted data. На защите вопрос «откуда n_TPM=1.4» вскроется быстро. | 2–4 дня (либо реальный API, либо честно переименовать в `..._baseline_constants.py`) |
| 3 | **High** | Заявленная новизна — SDE↔ABM coupling — отключена в API (501 на `integrated`). Код есть, но скрыт. На защите демо-сценарий должен показывать именно это. | 1–2 дня (раскрыть за feature-flag, ограничить N_micro для стабильности) |
| 4 | **High** | «Equation-Free» — это не Kevrekidis equation-free. Терминологическая уязвимость. | 0.5 дня (rename в коде/доках в «multiscale operator splitting» или «patch dynamics coupling») |
| 5 | **High** | 87% coverage, но тесты структурные. Нет fixed-seed SDE regression hash, нет conservation invariants, нет phase-timing assertions. | 2 дня (5–10 научных тестов) |
| 6 | **High** | Параметры с author-year цитатами без DOI/PMID. Любой reviewer попросит trace. | 1 день (вручную добавить DOI к ~30 ключевым параметрам) |
| 7 | **Medium** | Нет CI/CD. Тесты не гонятся на push. mypy/ruff backlog не enforced. | 0.5 дня (GitHub Actions: pytest + tsc + cargo check) |
| 8 | **Medium** | `data/validation/FR-FCM-wound-healing/` — пустой placeholder с 1 синтетической строкой. Альфер сам описал FlowRepository как источник. | 1 день (скачать реальный FR-FCM датасет и прогнать через `parameter_extraction`) |
| 9 | **Medium** | Parameter estimation API возвращает 501, при том что `parameter_estimation.py` подключает pymc+emcee и реализован. Disconnect между кодом и продуктовой поверхностью. | 1 день (либо вывести в API, либо удалить из README — сейчас вводит в заблуждение) |
| 10 | **Low** | Alembic подключён (`alembic.ini`), миграций 0 — `Base.metadata.create_all()`. Не блокирует, но индикатор недоведения. SQLite + sync ORM не масштабируется, но для desktop-app это ОК. | 0.5 дня — если стоит делать вообще |

## 8. Quick Wins (1–2 дня каждый, непропорциональный value)

1. **Сравнение vs Xue 2009.** В `literature_data.py` уже есть Xue-shape curves. Запустить extended-SDE с подобранными IC, сохранить overlay-plot `populations_xue_vs_regentwin.png`, записать DTW + CRPS. **Один артефакт превращает диплом из «работающий код» в «валидированную модель».**
2. **Раскрыть `integrated` mode за feature-flag для демо защиты.** Даже если на больших N нестабильно — на 1000 шагов 100 агентов хватит для визуализации coupling.
3. **5 научных pytest-тестов:** `test_m1m2_temporal_dynamics`, `test_collagen_monotone_in_remodeling`, `test_tnf_clearance_late_phase`, `test_mass_conservation_lift_restrict`, `test_therapy_delta_positive`. Делает заявку «validated» технически защитимой.
4. **GitHub Actions.** Один `.github/workflows/ci.yml` с `pytest`, `npm run lint`, `tsc -b`, `cargo check`. На защите «у нас зелёный CI» — стандартная гигиена.
5. **DOI/PMID в `parameters.py` для топ-30 параметров.** `Xue 2009` → `Xue 2009 (PMID:19809093)`. Часовая работа, защитное преимущество перед панелью.
6. **Включить Morris в API.** SALib его поддерживает, в коде он есть — просто снять 501. Расширяет analysis surface бесплатно.

## 9. Open Questions for Альфер

1. **`integrated` mode 501 — почему?** Не реализовано до конца, или решение скрыть из-за стабильности на больших N? Если второе — feature-flag решает за час.
2. **`hpa_client.py` — задумывался реальный REST-клиент к HPA или сразу hardcoded baseline?** Если первое — был ли неудачный attempt? Если второе — стоит переименовать модуль.
3. **`parameter_extraction.py` (813 LoC) — кто его писал?** Если AI: можешь ли ты сейчас защитить gate thresholds (FSC/SSC, CD34, % apoptotic)? Эту часть будут спрашивать.
4. **Альфер действительно понимает Euler-Maruyama vs Milstein?** Или это «вкл solver=EM по умолчанию»? Reviewer спросит про strong/weak convergence — ответ должен звучать осмысленно.
5. **«Equation-free» — это маркетинговое решение или Альфер реально читал Kevrekidis (PNAS 2009 / Annu Rev Phys Chem 2009)?** Если первое — переименовать заранее, не на защите. Если второе — стоит явно объяснить, какие части framework пропущены и почему.
6. **План на 6 месяцев vs 24 месяца** — где приоритет: защита диплома (валидация + 1 paper draft) или PhD pre-work (PINN, Fokker-Planck, MLMC, optimal control)? Это разные tradeoff-ы.
7. **Целевой журнал** — Computers in Biology and Medicine? J Theor Biol? Bull Math Biol? PLOS Comput Biol? У каждого свой минимум валидации, его стоит зафиксировать до выбора что делать.

## 10. Suggested Next Steps (priority-ordered)

1. **(Приоритет 1, к диплому)** Бенчмарк vs Xue 2009: один overlay-плот, DTW/CRPS-метрики, 1 страница prose в pdf-репорт. **Превращает проект в валидированный.**
2. **(Приоритет 1)** Раскрыть `integrated` режим за `RT_ENABLE_INTEGRATED=1` и записать demo-видео для защиты. SDE↔ABM визуализация — главный hook narrative.
3. **(Приоритет 2)** 5–7 научных pytest-тестов + GitHub Actions. Даёт «green CI + scientific assertions» статус.
4. **(Приоритет 2)** Переименовать `equation_free.py` или явно ограничить заявку до «multiscale operator splitting». Защитная мера.
5. **(Приоритет 3, для НИРС-статьи)** Скачать реальный FlowRepository wound-healing датасет (FR-FCM-Z2YP или эквивалент), прогнать через `parameter_extraction`, зафиксировать «реальные IC → реальная траектория» pipeline. Закроет «mock data» восприятие.
6. **(Отложить до PhD)** Rust hot loops, neural surrogate (sbi/SBI), PINN, SBML/MultiCellDS. Это работа 2027 года, не диплома.

---

## Notes for the other Claude (в чате claude.ai)

1. **«Diagnostics OK» ≠ «модель валидна».** Не покупайся на 95/95 OK в `output/diagnosis_report.json` или 2358 passed pytest. Это про runtime, не про science. Реальный gap — отсутствие external validation.
2. **Frontend и Tauri — неожиданно сильные.** 7 страниц, R3F-вьюер, WebSocket, Zustand, Playwright e2e, реальные данные. Не списывай на «ну там просто React». Это production-grade инженерная часть.
3. **Rust-часть — пустая (168 LoC launcher).** «Переписать hot loops на Rust» — greenfield work. Реалистичный effort: 3–4 недели только на bindings + Maturin/PyO3 + benchmarking, не считая собственно симуляции. Для диплома это **не quick win**.
4. **«Equation-free» — слабое место в защите.** Это lift-evolve-restrict цикл, без coarse-equation discovery. Перед панелью, знающей Kevrekidis, надо либо переименовать, либо явно объяснить ограничения. Я бы переименовал.
5. **AI-flavored, но дисциплинированный.** Inline-комментарии в `extended_sde.py` ссылаются на конкретные биологические механизмы (M1↔M2 switching, IL-10 inhibition Hill, TGF-β bistability). Cтиль docstring'ов унифицирован с pattern-ом `Подробное описание: Description/PhaseN/description_*.md#funcname`. Тестовые docstring'и TDD-стиля. Это **не vibe-coded** — здесь видна structured collaboration с AI и явный review.
6. **Самое уязвимое место — `hpa_client.py` и `literature_data.py`.** Они выглядят как «загружаем данные», а на деле — hardcoded lookup-таблицы. Если рецензент откроет код — это обнажится за минуту. Стоит либо реализовать настоящий fetch, либо переименовать в `_constants.py` / `_baseline.py` чтобы убрать ложное обещание.
7. **Тесты больше кода (32k vs 26k LoC), но почти все структурные.** Coverage 87 % — это покрытие строк, а не покрытие свойств. Перед НИРС-статьёй надо добавить хотя бы 5 property-based / invariant tests, иначе `methods` секция paper будет тонкой.
8. **Парадокс «mock data».** Альфер говорит «всё на mock». На самом деле: input-FCS реальный (130MB), parameter_extraction реальный (813 LoC), simulator реальный, **output не сравнивается ни с какими real-world wound-healing time-series**. Mock в смысле «нет ground truth для outputs», не в смысле «генерим случайные числа».
9. **PhD plan (`Doks/RegenTwin_PhD_Enhancement_Plan.md`, 60K) — амбициозный.** Fokker-Planck, MLMC, optimal control, ABC, PINN, virtual trials, TDA. **Ничего из этого ещё не реализовано** — это roadmap. На текущей кодовой базе реалистичен 1 пункт за 6 месяцев, не 8.
10. **Стоит спросить Альфера, на какой журнал он целится.** «Computational Biology paper» — слишком общо. Computers in Biology & Medicine примет с одним бенчмарком vs Xue. PLOS Comput Biol потребует ~3 валидации + uncertainty propagation. J Theor Biol — analytic insight. Эти три tradeoff-а определяют приоритеты следующих 3 месяцев.

---

*Audit conducted 2026-04-27 by Claude Code via 3-agent parallel exploration + manual cross-verification of findings on `master @ b0bc8ca`.*
