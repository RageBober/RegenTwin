# Monte Carlo

Ансамблевые прогоны для quantifying uncertainty. Реализация:
[src/core/monte_carlo.py](https://github.com/RageBober/RegenTwin/blob/master/src/core/monte_carlo.py).

```python
from src.core.monte_carlo import MonteCarloConfig, MonteCarloSimulator

cfg = MonteCarloConfig(
    n_trajectories=100,
    model_type="extended",
    extended_params=params,
    extended_initial_state=state,
    n_jobs=4,
    use_multiprocessing=True,
    base_seed=42,
)
sim = MonteCarloSimulator(config=cfg)
results = sim.run(model_params)
```

## Параллелизация

`use_multiprocessing=True` + `n_jobs > 1` → задача делегируется
`concurrent.futures.ProcessPoolExecutor`. GIL обходится за счёт separate
worker processes. Для маленьких симуляций (N < 10 траекторий) overhead
fork'а может перевесить выигрыш — используйте serial.

## Авто-документация

::: src.core.monte_carlo.MonteCarloSimulator
    options:
      members:
        - run
        - __init__
