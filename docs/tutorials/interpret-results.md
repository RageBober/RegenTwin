# Туториал: интерпретация результатов

## Структура `ExtendedSDETrajectory`

```python
trajectory.times     # ndarray (n_steps + 1,)  — времена в часах
trajectory.states    # list[ExtendedSDEState]  — состояния системы
```

Конвертация в массив:
```python
import numpy as np
arr = np.array([s.to_array() for s in trajectory.states])  # (n_steps+1, 20)
```

Колонки соответствуют `StateIndex` (см. `src/core/extended_sde.py`).

## Биологические индикаторы

| Метрика | Что значит | Где смотреть |
|---|---|---|
| `M2/M1` ratio | M2 > M1 → переход к репаративной фазе | состояние при $t \geq 72$ ч |
| `rho_collagen` | заполнение раны соединительной тканью | при $t = T_\max$ |
| `Mf` (миофибробласты) | риск фиброза если плато не падает | при $t \geq 168$ ч |
| `O_2` | возврат к норме (~80 mmHg) | при $t = T_\max$ |
| `D` (DAMPs) | $D \to 0$ → разрешение воспаления | при $t \geq 96$ ч |

## Через визуализацию API

```bash
curl -X POST http://localhost:8000/api/viz/populations \
  -H "Content-Type: application/json" \
  -d '{"simulation_id": "..."}'
```

См. [REST API в README](https://github.com/RageBober/RegenTwin#rest-api).

## Quick sanity checks

- Все клеточные плотности неотрицательные
- $M_1$ пик в $\sim 24-48$ ч, $M_2$ растёт после 72 ч
- $\rho_\text{collagen}$ монотонно растёт после 96 ч
- $O_2$ восстанавливается к плато $\sim 80$ mmHg

Если что-то из этого не выполняется — это сигнал либо для пересмотра параметров,
либо для проверки `tests/unit/core/test_extended_sde.py`.
