# Описание: analysis_plots.py — Визуализация анализа чувствительности и идентификации

## Обзор

Модуль `src/visualization/analysis_plots.py` предоставляет 4 функции визуализации результатов
анализа чувствительности (Sobol, Morris) и параметрической идентификации (posterior, convergence).
Все функции возвращают `plotly.graph_objects.Figure` и не зависят от matplotlib/ArviZ.

Архитектура:

```
SensitivityAnalyzer                    ParameterEstimator
  │                                       │
  ├── run_sobol()  → SobolResult          ├── run_bayesian() → EstimationResult
  │                                       │     ├── .posterior_samples
  └── run_morris() → MorrisResult         │     ├── .diagnostics (ConvergenceDiagnostics)
                                          │     └── .point_estimates, .ci_lower, .ci_upper
                                          │
              ┌───────────────────────────┐
              │  analysis_plots.py        │
              │                           │
              │  plot_sobol(SobolResult)   │ ──► go.Figure (tornado bar chart)
              │  plot_morris(MorrisResult) │ ──► go.Figure (μ* vs σ scatter)
              │  plot_posterior(EstResult) │ ──► go.Figure (marginals / corner)
              │  plot_convergence(EstRes) │ ──► go.Figure (R-hat, ESS, trace)
              └───────────────────────────┘
                         │
                         ▼
                  theme.py (ANALYSIS_COLORS, apply_default_layout)
```

**Зависимости модуля:**
- `src.core.sensitivity_analysis.SobolResult` — результат Sobol анализа
- `src.core.sensitivity_analysis.MorrisResult` — результат Morris скрининга
- `src.core.parameter_estimation.EstimationResult` — результат идентификации
- `src.core.parameter_estimation.ConvergenceDiagnostics` — диагностика сходимости
- `src.visualization.theme.ANALYSIS_COLORS` — цветовая палитра анализа
- `src.visualization.theme.VARIABLE_LABELS` — человекочитаемые подписи
- `src.visualization.theme.apply_default_layout` — стандартный layout Plotly
- Внешние: `numpy`, `plotly`

---

## plot_sobol

**Назначение:** Tornado bar chart для визуализации Sobol sensitivity indices (S1, ST или оба). Горизонтальные столбцы, отсортированные по убыванию метрики, с опциональными error bars (95% CI).

**Сигнатура:**
```python
def plot_sobol(
    result: SobolResult,
    metric: str = "both",
    top_n: int | None = 15,
    show_confidence: bool = True,
    height: int = 500,
) -> go.Figure:
```

**Параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `result` | `SobolResult` | (обязательно) | Результат `SensitivityAnalyzer.run_sobol()` |
| `metric` | `str` | `"both"` | `"S1"` — first-order, `"ST"` — total-effect, `"both"` — обе метрики рядом |
| `top_n` | `int \| None` | `15` | Показать только top N параметров по ST. `None` — все |
| `show_confidence` | `bool` | `True` | Показать error bars из `S1_conf`/`ST_conf` |
| `height` | `int` | `500` | Высота фигуры в пикселях |

**Возвращает:** `go.Figure` — Plotly Figure с горизонтальными bar traces.

**Алгоритм:**

1. Валидация: `metric` ∈ {`"S1"`, `"ST"`, `"both"`}; `result.parameter_names` не пуст.
2. Вычислить порядок сортировки: `order = np.argsort(-result.ST)`.
3. Если `top_n` задан и < len(parameter_names): обрезать `order` до `top_n`.
4. Извлечь имена параметров, значения S1/ST и CI по отсортированным индексам.
5. Инвертировать порядок для `go.Bar(orientation='h')` — наименьший сверху, наибольший снизу.
6. Если `metric == "both"`: создать два trace (`go.Bar`) с `barmode="group"`:
   - S1 — цвет `ANALYSIS_COLORS["S1"]` (#2e86c1)
   - ST — цвет `ANALYSIS_COLORS["ST"]` (#e74c3c)
7. Если `metric == "S1"` или `"ST"`: один trace соответствующего цвета.
8. Если `show_confidence`: добавить `error_x` с `array=conf_values`, `type="data"`.
9. Оси: Y — имена параметров (через `VARIABLE_LABELS.get(name, name)`), X — "Sobol Index".
10. Title: `f"Sobol Sensitivity — {result.output_variable}"`.
11. `apply_default_layout(fig, height=height, title=title, barmode="group")`.

**Граничные случаи и ошибки:**

| Ситуация | Поведение |
|----------|-----------|
| `metric` ∉ {S1, ST, both} | `ValueError` |
| `parameter_names` пуст | `ValueError` |
| `top_n > len(parameters)` | Показать все (не обрезать) |
| `top_n <= 0` | Показать все |
| `S1` содержит отрицательные значения | Отобразить как есть (возможно при малой выборке) |
| `S1 > ST` для параметра | Отобразить (теоретически невозможно, но не кидать ошибку) |
| Все индексы ≈ 0 | Столбцы при нуле, фигура валидна |

**Зависимости:**
- `SobolResult` — поля: `parameter_names`, `S1`, `ST`, `S1_conf`, `ST_conf`, `output_variable`
- `ANALYSIS_COLORS`, `VARIABLE_LABELS`, `apply_default_layout`

**Тестовые сценарии:**

| Сценарий | Вход | Ожидание |
|----------|------|----------|
| Smoke test (both) | SobolResult с 5 параметрами | `go.Figure` с 2 bar traces |
| Smoke test (S1 only) | `metric="S1"` | `go.Figure` с 1 bar trace |
| top_n=3 | 10 параметров, `top_n=3` | Только 3 параметра на графике |
| top_n=None | 10 параметров | Все 10 параметров |
| error bars | `show_confidence=True` | Traces содержат `error_x` |
| no error bars | `show_confidence=False` | Traces без `error_x` |
| invalid metric | `metric="invalid"` | `ValueError` |
| empty result | 0 параметров | `ValueError` |
| сортировка | Разные значения ST | Y-axis отсортирована по убыванию ST |

---

## plot_posterior

**Назначение:** Маргинальные гистограммы или corner plot (треугольная матрица) апостериорного распределения из MCMC/Bayesian estimation. Plotly-аналог ArviZ `plot_posterior()` / `plot_pair()`.

**Сигнатура:**
```python
def plot_posterior(
    result: EstimationResult,
    parameters: list[str] | None = None,
    layout: str = "marginals",
    show_ci: bool = True,
    show_point_estimate: bool = True,
    n_bins: int = 40,
    height: int = 600,
) -> go.Figure:
```

**Параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `result` | `EstimationResult` | (обязательно) | Результат идентификации с `posterior_samples` |
| `parameters` | `list[str] \| None` | `None` | Подмножество параметров. None — все из `posterior_samples` |
| `layout` | `str` | `"marginals"` | `"marginals"` — столбец гистограмм, `"corner"` — NxN матрица |
| `show_ci` | `bool` | `True` | Вертикальные линии 95% CI |
| `show_point_estimate` | `bool` | `True` | Вертикальная линия точечной оценки |
| `n_bins` | `int` | `40` | Число бинов гистограммы |
| `height` | `int` | `600` | Высота фигуры в пикселях |

**Возвращает:** `go.Figure` — Plotly Figure с subplots.

**Алгоритм (layout="marginals"):**

1. Валидация: `result.posterior_samples is not None`; запрошенные `parameters` есть в `posterior_samples`.
2. Определить список параметров: `parameters or list(result.posterior_samples.keys())`.
3. `N = len(parameters)`. Создать `make_subplots(rows=N, cols=1, subplot_titles=param_labels)`.
4. Для каждого параметра `i`:
   a. `samples = result.posterior_samples[param]` (1D np.ndarray).
   b. Добавить `go.Histogram(x=samples, nbinsx=n_bins, marker_color=ANALYSIS_COLORS["posterior"])` в subplot (i+1, 1).
   c. Если `show_ci` и `result.ci_lower` содержит param:
      - Добавить вертикальную линию (go.Scatter) при `result.ci_lower[param]` и `result.ci_upper[param]`,
        цвет `ANALYSIS_COLORS["ci"]`, dash="dash".
   d. Если `show_point_estimate` и `result.point_estimates` содержит param:
      - Вертикальная линия при `result.point_estimates[param]`,
        цвет `ANALYSIS_COLORS["point_est"]`, dash="solid".
5. `apply_default_layout(fig, height=height, title="Posterior Distributions", showlegend=False)`.

**Алгоритм (layout="corner"):**

1. Шаги 1–2 аналогичны.
2. `N = len(parameters)`. Создать `make_subplots(rows=N, cols=N)`.
3. Диагональ `(i, i)`: `go.Histogram` маргинального распределения параметра i.
4. Нижний треугольник `(i, j)` где `i > j`:
   - `go.Scatter(x=samples_j, y=samples_i, mode="markers", opacity=0.3, marker_size=2)`.
   - Если `len(samples) > 5000`: случайная подвыборка 5000 точек для производительности.
5. Верхний треугольник: оставить пустым.
6. Оси: для последней строки — подписи X, для первого столбца — подписи Y.
7. `apply_default_layout(fig, height=height, title="Corner Plot", showlegend=False)`.

**Граничные случаи и ошибки:**

| Ситуация | Поведение |
|----------|-----------|
| `posterior_samples is None` (MLE) | `ValueError`: "MLE метод не предоставляет posterior samples. Используйте bayesian_pymc или mcmc_emcee." |
| Параметр не в posterior_samples | `ValueError` с перечислением доступных |
| `layout` ∉ {marginals, corner} | `ValueError` |
| Один параметр, corner mode | Вырождается в 1×1: одна гистограмма |
| >50000 samples, corner mode | Подвыборка 5000 для scatter |
| Нулевая дисперсия samples | Гистограмма коллапсирует в один бин — отобразить annotation-предупреждение |
| `n_bins` > len(samples) | Авто-уменьшить: `min(n_bins, int(np.sqrt(len(samples))))` |
| `ci_lower`/`ci_upper` не содержат параметр | Пропустить CI-линии для этого параметра |

**Зависимости:**
- `EstimationResult` — поля: `posterior_samples`, `ci_lower`, `ci_upper`, `point_estimates`
- `ANALYSIS_COLORS`, `VARIABLE_LABELS`, `apply_default_layout`, `make_subplots`

**Тестовые сценарии:**

| Сценарий | Вход | Ожидание |
|----------|------|----------|
| Smoke marginals | EstimationResult с 3 параметрами | Figure с 3 subplot-rows |
| Smoke corner | `layout="corner"`, 3 параметра | Figure с 3×3 subplots |
| MLE (no samples) | `posterior_samples=None` | `ValueError` |
| Фильтр параметров | `parameters=["r_F"]` из 5 | Figure с 1 subplot |
| Несуществующий параметр | `parameters=["nonexistent"]` | `ValueError` |
| CI-линии | `show_ci=True` | Traces с vertical lines на ci_lower/ci_upper |
| Без CI | `show_ci=False` | Нет CI-линий |
| Corner downsample | 100000 samples | Off-diagonal scatter ≤ 5000 точек |

---

## plot_convergence

**Назначение:** Многопанельная визуализация диагностики сходимости MCMC: R-hat по параметрам, ESS (bulk/tail), trace plots. Позволяет быстро оценить качество MCMC сэмплирования.

**Сигнатура:**
```python
def plot_convergence(
    result: EstimationResult,
    metrics: list[str] | None = None,
    show_rhat_threshold: bool = True,
    height: int = 500,
) -> go.Figure:
```

**Параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `result` | `EstimationResult` | (обязательно) | Результат идентификации с `diagnostics` |
| `metrics` | `list[str] \| None` | `None` | Панели: `"rhat"`, `"ess"`, `"trace"`. None — все доступные |
| `show_rhat_threshold` | `bool` | `True` | Горизонтальная линия R-hat = 1.05 |
| `height` | `int` | `500` | Высота фигуры в пикселях |

**Возвращает:** `go.Figure` — Plotly Figure с subplots (1-3 панели).

**Алгоритм:**

1. Валидация: `result.diagnostics is not None`.
2. Определить набор панелей:
   - `metrics or ["rhat", "ess", "trace"]`
   - Фильтр: `"trace"` только если `result.posterior_samples` доступен.
   - Фильтр: `"rhat"` только если `diagnostics.rhat` не пуст.
   - Фильтр: `"ess"` только если `diagnostics.ess_bulk` не пуст.
3. `n_panels = len(filtered_metrics)`. Создать `make_subplots(rows=n_panels, cols=1, subplot_titles=[...])`.

**Панель "rhat":**

4. `param_names = list(diagnostics.rhat.keys())`.
5. `rhat_values = list(diagnostics.rhat.values())`.
6. `colors = [ANALYSIS_COLORS["point_est"] if v < 1.05 else ANALYSIS_COLORS["ci"] for v in rhat_values]`.
7. `go.Bar(x=param_names, y=rhat_values, marker_color=colors)`.
8. Если `show_rhat_threshold`: горизонтальная линия `go.Scatter(y=[1.05, 1.05], ...)`, dash, цвет `ANALYSIS_COLORS["threshold"]`.

**Панель "ess":**

9. `go.Bar(name="ESS bulk", x=param_names, y=ess_bulk_values)`.
10. `go.Bar(name="ESS tail", x=param_names, y=ess_tail_values)`.
11. `barmode="group"`. Горизонтальная линия при 100 (минимум рекомендуемый ESS).

**Панель "trace":**

12. Для каждого параметра: `samples = result.posterior_samples[param]`.
13. Если `result.config` и `result.config.n_chains` известен:
    - Reshape samples в `(n_chains, samples_per_chain)`.
    - Одна линия на цепочку (`go.Scatter` с opacity=0.7).
14. Иначе: одна линия (assume 1 chain).
15. X-axis = номер итерации, Y-axis = значение параметра.

16. `apply_default_layout(fig, height=height * n_panels, title="Convergence Diagnostics")`.

**Граничные случаи и ошибки:**

| Ситуация | Поведение |
|----------|-----------|
| `diagnostics is None` (MLE) | `ValueError`: "MLE метод не предоставляет диагностику сходимости." |
| `rhat` пуст | Пропустить панель rhat |
| `ess_bulk`/`ess_tail` пуст | Пропустить панель ess |
| `posterior_samples is None` | Пропустить панель trace |
| Все панели отфильтрованы | `ValueError`: "Нет доступных метрик для визуализации." |
| `metrics` содержит невалидное значение | `ValueError` |
| Все R-hat < 1.05 | Все столбцы зелёные |
| `config is None` (n_chains неизвестен) | Trace: 1 линия на параметр |
| Один параметр | Одна группа столбцов / одна trace-линия |

**Зависимости:**
- `EstimationResult` — поля: `diagnostics`, `posterior_samples`, `config`
- `ConvergenceDiagnostics` — поля: `rhat`, `ess_bulk`, `ess_tail`
- `ANALYSIS_COLORS`, `apply_default_layout`, `make_subplots`
- `_CONVERGENCE_PANEL_TITLES` — заголовки панелей (внутренняя константа)

**Тестовые сценарии:**

| Сценарий | Вход | Ожидание |
|----------|------|----------|
| Smoke (все панели) | EstimationResult полный | Figure с 3 subplot-rows |
| Только rhat | `metrics=["rhat"]` | Figure с 1 subplot |
| MLE (no diagnostics) | `diagnostics=None` | `ValueError` |
| Пустой rhat | `rhat={}` | Панель rhat пропущена |
| Threshold line | `show_rhat_threshold=True` | Горизонтальная линия y=1.05 |
| Цвета R-hat | rhat: {a: 1.01, b: 1.10} | a — зелёный, b — красный |
| Trace без chains | `config=None` | Одна линия на параметр |
| ESS grouped bar | 3 параметра | 2 группы столбцов (bulk, tail) |

---

## plot_morris

**Назначение:** Morris screening scatter plot в пространстве (μ*, σ). Каждая точка — параметр модели. Позволяет одновременно оценить важность (μ*) и степень нелинейности/взаимодействий (σ) каждого параметра. Стандартная визуализация (Morris 1991, Campolongo et al. 2007).

**Ключевое отличие:** Это **scatter plot**, а НЕ tornado/bar chart. Tornado для Morris уже есть в `TornadoPlotter.from_morris()` (matplotlib). Данная функция — комплементарная двумерная визуализация в Plotly.

**Сигнатура:**
```python
def plot_morris(
    result: MorrisResult,
    highlight_influential: bool = True,
    threshold_ratio: float = 0.1,
    show_labels: bool = True,
    show_wedge: bool = True,
    height: int = 500,
) -> go.Figure:
```

**Параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `result` | `MorrisResult` | (обязательно) | Результат `SensitivityAnalyzer.run_morris()` |
| `highlight_influential` | `bool` | `True` | Выделить влиятельные параметры цветом |
| `threshold_ratio` | `float` | `0.1` | Порог для `result.get_influential()` (доля от max(μ*)) |
| `show_labels` | `bool` | `True` | Подписи параметров рядом с точками |
| `show_wedge` | `bool` | `True` | Диагональ σ = μ* (граница линейности) |
| `height` | `int` | `500` | Высота фигуры в пикселях |

**Возвращает:** `go.Figure` — Plotly Figure со Scatter trace(s).

**Алгоритм:**

1. Валидация: `result.parameter_names` не пуст.
2. Получить множество влиятельных параметров: `influential = set(result.get_influential(threshold_ratio))`.
3. Разделить параметры на две группы: influential и non-influential.
4. **Non-influential group** (если есть):
   - `go.Scatter(x=mu_star_non, y=sigma_non, mode="markers", marker=dict(color=ANALYSIS_COLORS["threshold"], size=8))`.
   - Если `show_labels` и `len(parameters) <= 20`: добавить `text` и `textposition="top center"`.
5. **Influential group** (если есть):
   - `go.Scatter(x=mu_star_inf, y=sigma_inf, mode="markers+text", marker=dict(color=ANALYSIS_COLORS["influential"], size=12))`.
   - Если `show_labels`: текстовые подписи всегда (через `VARIABLE_LABELS.get(name, name)`).
   - Если `result.mu_star_conf` не нулевой: горизонтальные error bars (`error_x`).
6. Если `show_wedge`:
   - `max_val = max(max(result.mu_star), max(result.sigma)) * 1.1`.
   - `go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines", line=dict(dash="dash", color=ANALYSIS_COLORS["threshold"]), name="σ = μ*")`.
7. Оси:
   - X: `"μ* (среднее |элементарных эффектов|)"`.
   - Y: `"σ (СКО элементарных эффектов)"`.
8. Title: `f"Morris Screening — {result.output_variable}"`.
9. `apply_default_layout(fig, height=height, title=title)`.

**Интерпретация:**
- **Ниже** диагонали σ = μ*: параметр влияет линейно (σ мало → эффект стабилен).
- **Выше** диагонали: сильные нелинейности или взаимодействия с другими параметрами.
- **Правее** на оси X: параметр более важен в целом.

**Граничные случаи и ошибки:**

| Ситуация | Поведение |
|----------|-----------|
| `parameter_names` пуст | `ValueError` |
| Все параметры influential | Только красные точки, нет серых |
| Нет influential (порог слишком высок) | Только серые точки, нет красных |
| Один параметр | Одна точка + wedge line |
| `mu_star_conf` все нули | Пропустить error bars |
| >20 параметров, show_labels | Подписи только для influential |
| `threshold_ratio=0` | Все параметры influential |
| `threshold_ratio=1` | Только параметр(ы) с max μ* influential |

**Зависимости:**
- `MorrisResult` — поля: `parameter_names`, `mu_star`, `sigma`, `mu_star_conf`, `output_variable`
- Метод: `MorrisResult.get_influential(threshold_ratio)` → `list[str]`
- `ANALYSIS_COLORS`, `VARIABLE_LABELS`, `apply_default_layout`

**Тестовые сценарии:**

| Сценарий | Вход | Ожидание |
|----------|------|----------|
| Smoke test | MorrisResult с 10 параметрами | Figure с scatter traces |
| Highlight on | `highlight_influential=True` | 2 trace-группы (красная + серая) |
| Highlight off | `highlight_influential=False` | 1 trace, все одного цвета |
| Wedge line | `show_wedge=True` | Диагональная dashed линия от (0,0) |
| No wedge | `show_wedge=False` | Нет диагонали |
| Labels | `show_labels=True`, 5 параметров | Текстовые подписи у точек |
| Error bars | mu_star_conf > 0 | Горизонтальные error bars |
| Empty result | 0 параметров | `ValueError` |
| Custom threshold | `threshold_ratio=0.5` | Меньше параметров в influential группе |

---

## Примеры использования

```python
from src.core.sensitivity_analysis import SensitivityAnalyzer, SensitivityConfig
from src.core.parameter_estimation import ParameterEstimator, EstimationConfig
from src.visualization.analysis_plots import (
    plot_sobol, plot_morris, plot_posterior, plot_convergence,
)

# ── Sobol ──
analyzer = SensitivityAnalyzer(model, params, bounds)
sobol_result = analyzer.run_sobol(n_samples=1024)
fig = plot_sobol(sobol_result, metric="both", top_n=10)
fig.show()

# ── Morris ──
morris_result = analyzer.run_morris(n_trajectories=50)
fig = plot_morris(morris_result, highlight_influential=True)
fig.show()

# ── Posterior ──
estimator = ParameterEstimator(model, observed_data, config)
est_result = estimator.run_bayesian()
fig = plot_posterior(est_result, layout="corner", parameters=["r_F", "r_M1", "K_F"])
fig.show()

# ── Convergence ──
fig = plot_convergence(est_result, metrics=["rhat", "ess"])
fig.show()
```
