# Sensitivity Analysis

Глобальный Sobol-анализ чувствительности через SALib. Реализация:
[src/core/sensitivity_analysis.py](https://github.com/RageBober/RegenTwin/blob/master/src/core/sensitivity_analysis.py).

## Sobol indices

Для модели $Y = f(X_1, \ldots, X_k)$:

- $S_i = V_i / V$ — first-order index (доля дисперсии от одного $X_i$)
- $S_T^i = E[V(Y|X_{-i})] / V$ — total-order (включает взаимодействия)

Стоимость вычисления: $N \cdot (2k + 2)$ запусков модели, где $N$ — base sample size.

## Через REST API

```bash
curl -X POST http://localhost:8000/api/v1/analysis/sensitivity \
  -H "Content-Type: application/json" \
  -d '{
    "n_samples": 256,
    "parameters": [
      {"name": "r_F", "lower": 0.01, "upper": 0.1, "nominal": 0.05},
      {"name": "delta_M", "lower": 0.01, "upper": 0.05, "nominal": 0.02}
    ]
  }'
```

## Morris (отключён)

Morris elementary effects временно отключён в текущей сборке — см. README, раздел
«Что сейчас поддерживается». Возможно вернётся в Phase 9 после оптимизации
тяжёлого ABM функционала.
