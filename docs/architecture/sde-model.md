# SDE-модель

20-переменная Эйлер-Маруяма SDE-модель. Реализация — [src/core/extended_sde.py](https://github.com/RageBober/RegenTwin/blob/master/src/core/extended_sde.py).

## Класс `ExtendedSDEModel`

```python
from src.core.extended_sde import ExtendedSDEModel, ExtendedSDEState
from src.core.parameters import ParameterSet
from src.core.therapy_models import TherapyProtocol

model = ExtendedSDEModel(
    params=ParameterSet(),
    therapy=TherapyProtocol(prp_enabled=True, prp_intensity=0.7),
    rng_seed=42,
)
trajectory = model.simulate(initial_state)
```

## Численная схема

Эйлер-Маруяма 1-го порядка с шагом `dt = 0.01` ч (по умолчанию):

$$
X_{n+1} = X_n + \mu(X_n) \cdot \Delta t + \sigma(X_n) \cdot \sqrt{\Delta t} \cdot \xi_n
$$

где $\xi_n \sim \mathcal{N}(0, I)$.

## Граничные условия

- Все клеточные плотности и концентрации цитокинов ограничены $\geq 0$ (clamping).
- ECM плотности также неотрицательные.

## Терапии

- **PRP** — добавка к источниковым членам для PDGF/TGF-β/VEGF (см. `src/core/therapy_models.py`)
- **PEMF** — буст пролиферации фибробластов и хемотаксиса

## Авто-документация

::: src.core.extended_sde.ExtendedSDEModel
    options:
      members:
        - simulate
        - __init__
