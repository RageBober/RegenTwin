# Первая симуляция

Запустим extended-SDE модель на 24 часа через REST API.

## Старт API

```bash
uv run uvicorn src.api.main:app --port 8000
```

## Запуск симуляции

```bash
curl -X POST http://localhost:8000/api/v1/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "extended",
    "params": {"t_max": 24.0, "dt": 0.01},
    "therapy": null
  }'
```

Ответ содержит `simulation_id`. Опросите статус:

```bash
SIM_ID="..."
curl http://localhost:8000/api/v1/simulate/$SIM_ID
```

Когда `"status": "completed"` — забирайте результаты:

```bash
curl http://localhost:8000/api/v1/results/$SIM_ID > result.json
```

## Через Python напрямую

```python
from src.core.extended_sde import ExtendedSDEModel, ExtendedSDEState
from src.core.parameters import ParameterSet

params = ParameterSet()
params.t_max = 24.0
state = ExtendedSDEState(P=200, Ne=100, M1=50, M2=10, F=80, E=20, S=10,
                         O2=40.0, D=1.0, t=0.0)
model = ExtendedSDEModel(params=params, rng_seed=42)
trajectory = model.simulate(state)
print(trajectory.times[-1], trajectory.states[-1].M1, trajectory.states[-1].M2)
```

См. также: [как запустить свою симуляцию](../tutorials/run-simulation.md).
