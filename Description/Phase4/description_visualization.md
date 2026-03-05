# Фаза 4: Визуализация — Описание модуля

## Обзор

Модуль `src/visualization/` предоставляет полный набор функций для визуализации результатов 20-переменной SDE модели, ABM модели и терапевтических сценариев. Все функции возвращают `plotly.graph_objects.Figure` и не зависят от Streamlit — могут быть использованы в любом контексте (API, CLI, Jupyter).

## Архитектура

```
src/visualization/
├── __init__.py          # Публичный API: экспорт констант и функций
├── theme.py             # Цветовая тема, layout defaults, группировка переменных
├── plots.py             # Временные графики: популяции, цитокины, ECM, фазы, сравнение
├── spatial.py           # Пространственные: heatmap, scatter, inflammation map, анимация
└── export.py            # Экспорт: PNG/SVG, CSV, PDF

src/api/routes/
└── visualization.py     # FastAPI endpoints → Plotly JSON для React фронтенда

tests/unit/visualization/
├── conftest.py          # Mock-фикстуры (траектории, snapshot, MC results)
├── test_theme.py        # Консистентность констант
├── test_plots.py        # Smoke tests для plot_*()
├── test_spatial.py      # Smoke tests для spatial viz
└── test_export.py       # Тесты экспорта (PNG, CSV, PDF)

tests/unit/api/
└── test_visualization_routes.py  # API endpoint тесты
```

## Модули

### theme.py — Цветовая тема

Централизованные константы для единообразия визуализаций:

| Константа | Описание | Количество |
|-----------|----------|-----------|
| `POPULATION_COLORS` | Цвета 8 популяций (P, Ne, M1, M2, F, Mf, E, S) | 8 |
| `CYTOKINE_COLORS` | Цвета 7 цитокинов | 7 |
| `ECM_COLORS` | Цвета ECM (коллаген, MMP, фибрин) | 3 |
| `THERAPY_COLORS` | Цвета сценариев (Control, PRP, PEMF, PRP+PEMF) | 4 |
| `PHASE_COLORS` | Цвета фаз заживления | 4 |
| `VARIABLE_LABELS` | Человекочитаемые подписи для 20 переменных | 20 |

Функция `apply_default_layout(fig, height, **kwargs)` применяет стандартный шаблон `plotly_white` ко всем фигурам.

### plots.py — Временные графики

| Функция | Описание | Входные данные |
|---------|----------|---------------|
| `plot_populations()` | 8 кривых роста клеток с CI | ExtendedSDETrajectory + MonteCarloResults |
| `plot_cytokines()` | 7 цитокинов (overlay/subplots) | ExtendedSDETrajectory |
| `plot_ecm()` | Коллаген, MMP, фибрин (dual axes) | ExtendedSDETrajectory |
| `plot_phases()` | Цветовая полоса фаз + популяции | ExtendedSDETrajectory + PhaseIndicators |
| `plot_comparison()` | 4 сценария на одном графике | dict[str, ExtendedSDETrajectory] |

### spatial.py — Пространственная визуализация

| Функция | Описание | Входные данные |
|---------|----------|---------------|
| `heatmap_density()` | 2D гистограмма позиций агентов | ABMSnapshot |
| `scatter_agents()` | Scatter по типу/энергии/возрасту | ABMSnapshot |
| `inflammation_map()` | TNF-α/IL-10 ratio heatmap | ABMSnapshot |
| `field_heatmap()` | Generic heatmap (cytokine/ecm) | ABMSnapshot |
| `animate_evolution()` | Plotly animation / GIF export | ABMTrajectory |

### export.py — Экспорт результатов

Класс `ReportExporter` с методами:
- `to_png(output_dir)` → list[Path] — через kaleido
- `to_svg(output_dir)` → list[Path]
- `to_csv(output_dir)` → list[Path] — 21 колонка (time + 20 переменных)
- `to_pdf(output_path)` → Path — через fpdf2: титул + метаданные + графики + сводка

### visualization.py (API) — FastAPI endpoints

| Endpoint | Method | Описание |
|----------|--------|----------|
| `/api/viz/populations` | POST | Plotly JSON для 8 популяций |
| `/api/viz/cytokines` | POST | Plotly JSON для 7 цитокинов |
| `/api/viz/ecm` | POST | Plotly JSON для ECM |
| `/api/viz/phases` | POST | Plotly JSON фаз заживления |
| `/api/viz/comparison` | POST | Plotly JSON сравнения 4 сценариев |
| `/api/viz/export/csv` | POST | CSV файл (StreamingResponse) |
| `/api/viz/export/png` | POST | PNG файл (StreamingResponse) |
| `/api/viz/export/pdf` | POST | PDF отчёт (StreamingResponse) |

Формат ответа: `fig.to_json()` — React Plotly.js потребляет напрямую.

## Зависимости

- `plotly >= 5.18.0` — все графики
- `matplotlib >= 3.8.0` — GIF-анимация (только `animate_evolution` с output_path)
- `kaleido >= 1.0.0` — экспорт PNG/SVG
- `fpdf2 >= 2.7.0` — генерация PDF
- `fastapi >= 0.100.0` — API endpoints
- `numpy >= 1.24.0` — обработка данных

## Тестирование

- **101 тест** в `tests/unit/visualization/` + `tests/unit/api/`
- Smoke tests: каждая функция возвращает `go.Figure`
- Structural tests: число traces, axis labels, colorscales
- Export tests: файлы создаются, CSV имеет 21 колонку, PDF не пуст
- API tests: status 200, корректный content-type, Plotly JSON structure

## Интеграция с Фазой 6 (React + Tauri)

API endpoints возвращают Plotly JSON, который React-компоненты потребляют через:
```tsx
import Plot from 'react-plotly.js';

const response = await fetch('/api/viz/populations', { method: 'POST', body: JSON.stringify({...}) });
const plotData = await response.json();
<Plot data={plotData.data} layout={plotData.layout} />
```

Планируемые React-компоненты (Фаза 6):
- `<PopulationChart>` — кривые роста 8 популяций
- `<CytokineChart>` — динамика 7 цитокинов
- `<ECMChart>` — ECM компоненты
- `<PhaseTimeline>` — цветовая полоса фаз
- `<TherapyComparison>` — сравнение 4 сценариев
- `<SpatialHeatmap>` — пространственная карта ABM
- `<InflammationMap>` — карта воспаления
- `<AnimationPlayer>` — анимация эволюции ABM
- 3D визуализация ABM через Three.js
