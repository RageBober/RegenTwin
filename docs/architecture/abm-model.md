# Agent-Based модель

Агентная модель пространственного поведения клеток. Реализация:
- [src/core/abm_model.py](https://github.com/RageBober/RegenTwin/blob/master/src/core/abm_model.py) — главный класс `ABMModel`
- [src/core/abm_spatial.py](https://github.com/RageBober/RegenTwin/blob/master/src/core/abm_spatial.py) — пространственные движки

## Конфигурация

```python
from src.core.abm_model import ABMConfig, ABMModel
from src.data.parameter_extraction import ModelParameters

cfg = ABMConfig(
    space_size=(100.0, 100.0),  # мкм × мкм
    t_max=24.0,
    max_agents=500,
)
params = ModelParameters(n0=10000, stem_cell_fraction=0.05,
                         macrophage_fraction=0.03, apoptotic_fraction=0.02,
                         c0=10.0, inflammation_level=0.5)
model = ABMModel(config=cfg, random_seed=42)
trajectory = model.simulate(params, snapshot_interval=24.0)
```

## Spatial structures

- **`SpatialHash`** — O(1) поиск соседей по grid
- **`KDTreeNeighborSearch`** — O(log n) для разрежённых конфигураций
- **`MultiCytokineField`** — discrete grid для цитокинов и ECM

## Авто-документация

::: src.core.abm_model.ABMModel
    options:
      members:
        - simulate
        - step
