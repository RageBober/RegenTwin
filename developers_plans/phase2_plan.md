# Phase 2: Математическое ядро — Stub Code + TDD Descriptions

## Обзор

Phase 2 расширяет математическое ядро RegenTwin до production-ready состояния.
Реализуется по **Этап 1** (stubs + descriptions): все новые методы содержат
`raise NotImplementedError("Stub: требуется реализация в Этап 3")`.

**Модули:**
- `sde_model.py` — 100% done, НЕ модифицируется
- `abm_model.py` — +3 новых типа агентов, KD-Tree, улучшенные механики
- `integration.py` — +5 методов (двусторонняя синхронизация, Equation-Free)
- `monte_carlo.py` — +3 метода (параллелизация)
- `numerical_utils.py` — новый модуль (робастность)

---

## Task 1: Новые типы агентов — `abm_model.py`

### NeutrophilAgent (CD66b+)
- AGENT_TYPE = "neutro", LIFESPAN = 24h, MAX_DIVISIONS = 0
- Методы: `__init__`, `update`, `divide`, `phagocytose`, `secrete_cytokines`, `is_apoptotic`
- Биология: хемотаксис по IL-8, фагоцитоз debris, TNF-α/IL-8 секреция

### EndothelialAgent (CD31+)
- AGENT_TYPE = "endo", LIFESPAN = 480h, DIVISION_PROBABILITY = 0.01
- Методы: `__init__`, `update`, `divide`, `form_junction`, `secrete_cytokines`
- Биология: VEGF-зависимый ангиогенез, адгезия клетка-клетка

### MyofibroblastAgent (α-SMA+)
- AGENT_TYPE = "myofibro", LIFESPAN = 480h, ECM_PRODUCTION_RATE = 1.0
- Методы: `__init__`, `update`, `divide`, `produce_ecm`, `contract`, `should_apoptose_tgfb`
- Биология: усиленная продукция ECM, контракция раны, TGF-β-зависимое выживание

### Обновления
- ABMConfig: +initial_neutrophils, +initial_endothelial, +initial_myofibroblasts
- _create_agent: +"neutro", +"endo", +"myofibro" ветки

---

## Task 2: Улучшенные ABM механики — `abm_model.py`

### KDTreeSpatialIndex
- Альтернатива SpatialHash на scipy.spatial.cKDTree
- Методы: `build`, `query_radius`, `query_nearest`

### Новые методы ABMModel
| Метод | Что делает |
|-------|------------|
| `_chemotaxis_displacement` | Мульти-градиентный хемотаксис по типу агента |
| `_apply_contact_inhibition` | Модификатор пролиферации [0,1] по плотности |
| `_calculate_adhesion_force` | Адгезия endo-endo, myofibro-myofibro |

### ABMConfig расширение
- adhesion_strength, adhesion_equilibrium_distance
- use_multi_chemotaxis, spatial_index_type

---

## Task 3: Расширение интеграции — `integration.py`

| Метод | Что делает |
|-------|------------|
| `_synchronize_cytokines` | Двусторонняя синхронизация C (ABM→SDE + SDE→ABM) |
| `_transfer_therapy_to_abm` | PRP/PEMF флаги в ABM environment |
| `_spatial_scaling` | SDE scalar C ↔ ABM 2D field |
| `_lifting` | Equation-Free: macro→micro |
| `_restricting` | Equation-Free: micro→macro |

---

## Task 4: Параллелизация — `monte_carlo.py`

| Метод | Что делает |
|-------|------------|
| `_run_parallel` | ProcessPoolExecutor для n_jobs процессов |
| `_progress_callback_wrapper` | Thread-safe прогресс через Lock |
| `_validate_parallel_config` | Проверка multiprocessing доступности |

---

## Task 5: Численная робастность — `numerical_utils.py` (новый)

| Элемент | Что делает |
|---------|------------|
| `DivergenceInfo` | Dataclass: has_nan, has_inf, nan_variables, is_diverged |
| `clip_negative_concentrations` | Отсечение отрицательных значений |
| `detect_divergence` | NaN/Inf/overflow детекция |
| `adaptive_timestep` | dt ↔ скорость изменения состояния |
| `NumericalGuard` | Context manager для безопасных вычислений |

---

## Task 6: TDD Description файлы

Обновить:
- `Description/Phase2/description_abm_model.md` — TDD для агентов и механик
- `Description/Phase2/description_integration.md` — TDD для синхронизации
- `Description/Phase2/description_monte_carlo.md` — TDD для параллелизации
- Создать: `Description/Phase2/description_numerical_utils.md`

---

## Task 7: Экспорты — `__init__.py`

Добавить: NeutrophilAgent, EndothelialAgent, MyofibroblastAgent, KDTreeSpatialIndex,
DivergenceInfo, NumericalGuard, clip_negative_concentrations, detect_divergence, adaptive_timestep

---

## Порядок: Task 5 → 1 → 2 → 3 → 4 → 6 → 7

## Верификация

```bash
python -c "from src.core import *"
pytest tests/unit/core/ -v
ruff check src/core/
```
