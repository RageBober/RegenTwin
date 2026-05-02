# Туториал: запуск своей симуляции

## Сценарий 1: чистый Python

```python
from src.core.extended_sde import ExtendedSDEModel, ExtendedSDEState
from src.core.parameters import ParameterSet
from src.core.therapy_models import TherapyProtocol

# 1. Параметры (можно модифицировать любые поля ParameterSet)
params = ParameterSet()
params.t_max = 168.0          # 7 дней
params.dt = 0.05              # шаг 3 минуты

# 2. Начальное состояние
state = ExtendedSDEState(
    P=200.0, Ne=100.0, M1=50.0, M2=10.0,
    F=80.0, E=20.0, S=10.0,
    O2=40.0, D=1.0,
)

# 3. Опционально — терапия
therapy = TherapyProtocol(prp_enabled=True, prp_intensity=0.7)

# 4. Запуск
model = ExtendedSDEModel(params=params, therapy=therapy, rng_seed=42)
trajectory = model.simulate(state)

# 5. Анализ
print(f"Final M1: {trajectory.states[-1].M1:.1f}")
print(f"Final M2: {trajectory.states[-1].M2:.1f}")
print(f"Collagen: {trajectory.states[-1].rho_collagen:.3f}")
```

## Сценарий 2: загрузка `.fcs` → начальные условия → симуляция

```python
from src.data.fcs_parser import parse_fcs_file
from src.data.parameter_extraction import extract_parameters
from src.core.extended_sde import ExtendedSDEModel
from src.core.parameters import ParameterSet

# 1. Парсим FCS
events = parse_fcs_file("path/to/sample.fcs")

# 2. Извлекаем initial conditions
extracted = extract_parameters(events)

# 3. Конвертируем в ExtendedSDEState (есть helper в src.api.services)
# ...

# 4. Запускаем как обычно
```

## Сценарий 3: через REST API

См. [Первая симуляция](../getting-started/first-simulation.md).
