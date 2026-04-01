# description_validation.md — Phase 3.4: Validation Metrics

## Назначение

Модуль `src/analysis/validation.py` предоставляет четыре научно строгих метрики
для оценки качества предсказаний 20-переменной SDE-модели раневого заживления.

Заменяет «Temporal R²» из раздела 6.2 Mathematical Framework — R² неприменим к
биологическим временным рядам, где пики клеточных популяций могут быть смещены
по времени между пациентами (фазовый сдвиг ≠ ошибка модели).

---

## Архитектура

```
ValidationConfig.validate()
        │
        ▼
ValidationRunner
    ├── run_dtw_crps(trajectory, observed) → DTWCRPSResult
    ├── run_ppc(mc_results, observed, estimation_result?) → PPCResult
    ├── run_phase_timing(trajectory, observed_breakpoints?) → PhaseTimingResult
    ├── run_sensitivity_ranking(sobol, morris) → SensitivityRankingResult
    └── run_all(...) → ValidationResult
            │
            ▼
    _compute_overall_score() → float ∈ [0, 1]

validate_model(trajectory, observed, ...)  # convenience wrapper
```

---

## Метрики

### DTWCRPSResult

**Dynamic Time Warping (DTW)** — расстояние между формами временных рядов.
В отличие от MSE/R², не штрафует корректную форму со смещением по времени.
Библиотека: `dtaidistance>=2.3.0`.

**CRPS** (Continuous Ranked Probability Score) — standard probabilistic forecast score.
Для гауссового приближения: `properscoring.crps_gaussian(obs, mu, sigma)`.
σ = 10% от среднего |μ| (консервативная оценка неопределённости).

Поля:
- `variable_names: list[str]` — переменные, по которым вычислены метрики
- `dtw_distances: dict[str, float]` — {var: dtw_distance ≥ 0}
- `crps_scores: dict[str, float]` — {var: mean_crps ≥ 0}
- `mean_dtw: float` — среднее DTW по переменным
- `mean_crps: float` — среднее CRPS по переменным
- `n_observations: int` — число временных точек наблюдений
- `elapsed_seconds: float`

### PPCResult

Posterior Predictive Check — проверяет, покрывает ли предсказанный ансамбль наблюдаемые данные.

**ArviZ path** (только PyMC, `inference_data` содержит `posterior_predictive`):
- `az.loo(idata)` → `loo_elpd`, `loo_se`
- `az.hdi(idata.posterior_predictive, hdi_prob)` → coverage per variable

**MC envelope fallback** (любые MC данные):
- Использует `mc_results.variable_quantiles[var][q]`
- Вычисляет долю наблюдений внутри [q₀.₀₂₅, q₀.₉₇₅]

Поля:
- `loo_elpd: float | None` — LOO ELPD (только ArviZ)
- `loo_se: float | None` — SE LOO
- `coverage_95: dict[str, float]` — {var: fraction ∈ [0,1]}
- `mean_coverage: float`
- `backend: str` — "arviz" или "mc_envelope"
- `elapsed_seconds: float`

### PhaseTimingResult

Changepoint detection для автоматического обнаружения фаз заживления.

Библиотека: `ruptures>=1.1.0`, алгоритм `Pelt` с `rbf`-ядром, BIC-штраф:
```python
pen = n_features * log(n_samples) * scale
```

Многоканальный сигнал: [Ne, M1, M2, F] (нормализованный по std).

Биологические ожидаемые точки разрыва:
- HEMOSTASIS → INFLAMMATION: ~6 ч
- INFLAMMATION → PROLIFERATION: ~4–6 дней (96–144 ч)
- PROLIFERATION → REMODELING: ~3 недели (504 ч)

Поля:
- `detected_breakpoints: list[PhaseBreakpoint]`
- `expected_breakpoints: list[PhaseBreakpoint] | None`
- `timing_mae_hours: float | None` — средняя ошибка (если есть expected)
- `n_phases_detected: int`
- `algorithm: str` — "Pelt+BIC"
- `elapsed_seconds: float`

### SensitivityRankingResult

Kendall's τ между рейтингами Sobol (по ST) и Morris (по μ*).
Вычисляется только на пересечении `parameter_names` обоих методов.

```python
tau, pvalue = scipy.stats.kendalltau(sobol_ranks, morris_ranks)
```

τ ≈ 1 → методы согласуются; τ ≈ 0 → нет консенсуса; τ < 0 → противоречие.

Поля:
- `kendall_tau: float` — ∈ [-1, 1]
- `p_value: float` — ∈ [0, 1]
- `ranking_comparisons: list[RankingComparison]`
- `n_parameters: int` — число общих параметров
- `elapsed_seconds: float`

### ValidationResult

Агрегирует все четыре метрики. `overall_score ∈ [0, 1]`:

| Метрика | Score formula | Weight (default) |
|---------|--------------|-----------------|
| DTW/CRPS | exp(−dtw / scale) | 0.30 |
| PPC | mean_coverage | 0.30 |
| Phase timing | exp(−mae / 24h) или 0.5 без MAE | 0.20 |
| Kendall's τ | (τ + 1) / 2 | 0.20 |

Веса ренормализуются если метрика недоступна (нет входных данных).

---

## ValidationConfig

```python
@dataclass
class ValidationConfig:
    run_dtw_crps: bool = True
    run_ppc: bool = True
    run_phase_timing: bool = True
    run_sensitivity_ranking: bool = True
    dtw_variables: list[str] = ["F", "M1", "M2", "Ne"]
    ppc_variables: list[str] = ["F", "M1", "M2"]
    hdi_prob: float = 0.95
    ruptures_model: str = "rbf"
    ruptures_penalty_scale: float = 1.0
    weight_dtw: float = 0.3    # сумма весов = 1.0
    weight_ppc: float = 0.3
    weight_timing: float = 0.2
    weight_ranking: float = 0.2
    rng_seed: int | None = 42
```

`config.validate()` бросает `ValueError` если сумма весов ≠ 1.0.

---

## Тесты

`tests/unit/analysis/test_validation.py` — 81 тест, 12 классов:

| Класс | Кол-во | Что проверяет |
|-------|--------|--------------|
| TestValidationConfig | 5 | validate(), веса, defaults |
| TestDTWCRPSResult | 8 | Поля dataclass, mean_dtw = mean(dict) |
| TestPPCResult | 8 | coverage ∈ [0,1], backend values |
| TestPhaseTimingResult | 8 | Breakpoints, timing_mae, confidence |
| TestSensitivityRankingResult | 8 | τ ∈ [-1,1], p ∈ [0,1], ranks |
| TestValidationResult | 6 | overall_score, get_summary() keys |
| TestRunnerDTWCRPS | 8 | Runner.run_dtw_crps() с фикстурами |
| TestRunnerPPC | 8 | MC fallback, coverage, loo=None |
| TestRunnerPhaseTiming | 6 | Pelt detection, MAE, algorithm name |
| TestRunnerSensRanking | 6 | τ=1 для идентичных рейтингов |
| TestRunnerRunAll | 5 | Агрегация, нет входных = None |
| TestValidateModelFunc | 5 | Convenience wrapper |

---

## Зависимости

```toml
"dtaidistance>=2.3.0",
"properscoring>=0.1",
"ruptures>=1.1.0",
```

`arviz`, `scipy` — уже в зависимостях проекта.
