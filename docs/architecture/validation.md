# Validation Metrics

Покрытие в `src/analysis/validation.py` и `src/core/wound_phases.py`:

- **DTW** (Dynamic Time Warping) — сравнение траекторий по форме
- **CRPS** (Continuous Ranked Probability Score) — для probabilistic forecasts
- **PPC** (Posterior Predictive Checks) — для bayesian inference
- **Changepoint detection** — `ruptures` библиотека
- **Kendall tau** — корреляция упорядоченных метрик
- **M1/M2 ratio** — биологическая корректность фазового перехода макрофагов

См. также: `tests/integration/test_validation_metrics.py`.
