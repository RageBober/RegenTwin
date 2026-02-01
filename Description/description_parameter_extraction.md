# Описание: parameter_extraction.py

## Обзор

Модуль извлечения параметров для математических моделей SDE/ABM
из результатов гейтирования flow cytometry данных.

---

## Теоретическое обоснование

### Параметры модели

| Параметр | Обозначение | Единицы | Описание |
|----------|-------------|---------|----------|
| N0 | N₀ | клеток/мкл | Начальная плотность клеток |
| C0 | C₀ | нг/мл | Концентрация факторов роста |
| Inflammation | I | 0-1 | Уровень воспаления |

### Связь с flow cytometry данными

**N0 (плотность клеток):**
- Количество живых клеток после гейтирования
- Масштабируется на объём и разведение

**C0 (факторы роста):**
- Косвенная оценка по клеточному составу
- CD34+ клетки → высокая секреция PDGF, VEGF
- Макрофаги → секреция TNF-α, IL-1β (воспаление)

**Inflammation level:**
- Доля макрофагов (повышенная = воспаление)
- Уровень апоптоза (повышенный = повреждение)
- Комбинированный индекс

---

## Классы

### ModelParameters

**Назначение:** Dataclass для хранения параметров модели.

**Атрибуты:**

| Атрибут | Тип | Описание | Диапазон |
|---------|-----|----------|----------|
| n0 | float | Плотность клеток | 1000-50000 кл/мкл |
| stem_cell_fraction | float | Доля CD34+ | 0.01-0.15 |
| macrophage_fraction | float | Доля макрофагов | 0.01-0.10 |
| apoptotic_fraction | float | Доля апоптотических | 0.01-0.10 |
| c0 | float | Концентрация цитокинов | 1-100 нг/мл |
| inflammation_level | float | Уровень воспаления | 0-1 |
| source_file | str \| None | Исходный файл | - |
| total_events | int | Всего событий | - |

**Методы:**

#### to_dict

```python
def to_dict(self) -> dict[str, Any]
```

Возвращает:
```python
{
    "n0": 5000.0,
    "stem_cell_fraction": 0.05,
    "macrophage_fraction": 0.03,
    "apoptotic_fraction": 0.02,
    "c0": 10.0,
    "inflammation_level": 0.3,
    "source_file": "sample.fcs",
    "total_events": 10000
}
```

#### validate

```python
def validate(self) -> bool
```

Проверяет:
1. n0 > 0
2. Все фракции в диапазоне [0, 1]
3. Сумма фракций < 1 (непересекающиеся популяции)
4. c0 >= 0
5. inflammation_level в [0, 1]

Raises ValueError с описанием при нарушении.

---

### ExtractionConfig

**Назначение:** Конфигурация для настройки процесса извлечения.

**Атрибуты:**

| Атрибут | Тип | По умолчанию | Описание |
|---------|-----|--------------|----------|
| volume_ul | float | 1.0 | Объём образца (мкл) |
| dilution_factor | float | 1.0 | Фактор разведения |
| ref_cell_density | float | 5000.0 | Референс плотности |
| ref_cytokine_conc | float | 10.0 | Референс концентрации |
| stem_cell_cytokine_factor | float | 2.0 | Вклад CD34+ в C0 |
| macrophage_cytokine_factor | float | 1.5 | Вклад макрофагов в C0 |

---

### ParameterExtractor

**Назначение:** Основной класс для извлечения параметров.

---

## Методы

### __init__

**Сигнатура:**

```python
def __init__(self, config: ExtractionConfig | None = None) -> None
```

**Алгоритм:**

1. Если config = None, создать ExtractionConfig()
2. Сохранить self._config = config

---

### extract

**Назначение:** Полное извлечение всех параметров.

**Сигнатура:**

```python
def extract(
    self,
    gating_results: GatingResults,
    source_file: str | None = None,
) -> ModelParameters
```

**Алгоритм:**

1. Рассчитать доли популяций: `fractions = self._calculate_cell_fractions(gating_results)`
2. Извлечь N0: `n0 = self.extract_n0(gating_results)`
3. Извлечь C0: `c0 = self.extract_c0(gating_results)`
4. Извлечь inflammation: `infl = self.extract_inflammation_level(gating_results)`
5. Создать ModelParameters:
   ```python
   params = ModelParameters(
       n0=n0,
       stem_cell_fraction=fractions["cd34_positive"],
       macrophage_fraction=fractions["macrophages"],
       apoptotic_fraction=fractions["apoptotic"],
       c0=c0,
       inflammation_level=infl,
       source_file=source_file,
       total_events=gating_results.total_events,
   )
   ```
6. Валидировать: `params.validate()`
7. Вернуть params

---

### extract_n0

**Назначение:** Расчёт начальной плотности клеток.

**Сигнатура:**

```python
def extract_n0(self, gating_results: GatingResults) -> float
```

**Формула:**

```
N₀ = (n_live_cells / volume) × dilution_factor
```

**Алгоритм:**

1. Получить количество живых клеток: `n_live = gating_results.gates["live_cells"].n_events`
2. Применить масштабирование:
   ```python
   n0 = n_live / self._config.volume_ul * self._config.dilution_factor
   ```
3. Нормализовать относительно референса (опционально):
   ```python
   n0 = self._normalize_to_reference(n0, self._config.ref_cell_density)
   ```

**Пример:**
- 7000 живых клеток, volume=1 мкл, dilution=1
- N₀ = 7000 клеток/мкл

---

### extract_c0

**Назначение:** Оценка концентрации факторов роста.

**Сигнатура:**

```python
def extract_c0(self, gating_results: GatingResults) -> float
```

**Принцип:**
- CD34+ клетки активно секретируют факторы роста (PDGF, VEGF)
- Макрофаги секретируют цитокины
- C0 пропорционален этим популяциям

**Формула:**

```
C₀ = C_ref × (α × f_stem + β × f_macro)
```

Где:
- C_ref — референсная концентрация (10 нг/мл)
- α — stem_cell_cytokine_factor (2.0)
- β — macrophage_cytokine_factor (1.5)
- f_stem — доля стволовых клеток
- f_macro — доля макрофагов

**Алгоритм:**

```python
f_stem = gating_results.gates["cd34_positive"].fraction
f_macro = gating_results.gates["macrophages"].fraction

c0 = self._config.ref_cytokine_conc * (
    self._config.stem_cell_cytokine_factor * f_stem +
    self._config.macrophage_cytokine_factor * f_macro
)

# Ограничить разумным диапазоном
c0 = np.clip(c0, 1.0, 100.0)
```

**Пример:**
- f_stem = 0.05, f_macro = 0.03
- C₀ = 10 × (2.0 × 0.05 + 1.5 × 0.03) = 10 × 0.145 = 1.45 нг/мл

---

### extract_inflammation_level

**Назначение:** Расчёт уровня воспаления.

**Сигнатура:**

```python
def extract_inflammation_level(
    self,
    gating_results: GatingResults,
    data: pd.DataFrame | None = None,
) -> float
```

**Формула:**

```
I = w₁ × (f_macro / f_macro_ref) + w₂ × (f_apopt / f_apopt_ref)
```

Где:
- w₁, w₂ — веса (0.6, 0.4)
- f_macro_ref = 0.03 (нормальный уровень макрофагов)
- f_apopt_ref = 0.02 (нормальный уровень апоптоза)

**Алгоритм:**

```python
f_macro = gating_results.gates["macrophages"].fraction
f_apopt = gating_results.gates["apoptotic"].fraction

# Референсные значения для "нормального" состояния
REF_MACRO = 0.03
REF_APOPT = 0.02

# Взвешенная сумма
inflammation = (
    0.6 * min(f_macro / REF_MACRO, 3.0) +  # макс 3x от нормы
    0.4 * min(f_apopt / REF_APOPT, 3.0)
) / 3.0  # нормализация к [0, 1]

inflammation = np.clip(inflammation, 0.0, 1.0)
```

**Интерпретация:**
- I < 0.3: низкое воспаление (норма)
- 0.3 ≤ I < 0.6: умеренное воспаление
- I ≥ 0.6: сильное воспаление

---

### _calculate_cell_fractions

**Назначение:** Расчёт долей всех популяций.

**Сигнатура:**

```python
def _calculate_cell_fractions(
    self,
    gating_results: GatingResults,
) -> dict[str, float]
```

**Возвращает:**

```python
{
    "debris": 0.20,
    "non_debris": 0.80,
    "singlets": 0.75,
    "live_cells": 0.70,
    "cd34_positive": 0.05,
    "macrophages": 0.03,
    "apoptotic": 0.02,
}
```

**Алгоритм:**

```python
total = gating_results.total_events
fractions = {}
for name, gate in gating_results.gates.items():
    fractions[name] = gate.n_events / total
return fractions
```

---

### _normalize_to_reference

**Назначение:** Нормализация значения.

**Сигнатура:**

```python
def _normalize_to_reference(
    self,
    value: float,
    ref_value: float,
    scale: str = "linear",
) -> float
```

**Алгоритм:**

```python
if scale == "linear":
    return value / ref_value
elif scale == "log":
    return np.log10(value) / np.log10(ref_value)
else:
    raise ValueError(f"Unknown scale: {scale}")
```

---

## Функции

### extract_model_parameters

**Назначение:** Convenience функция.

**Сигнатура:**

```python
def extract_model_parameters(
    gating_results: GatingResults,
    config: ExtractionConfig | None = None,
    source_file: str | None = None,
) -> ModelParameters
```

**Алгоритм:**

```python
extractor = ParameterExtractor(config=config)
return extractor.extract(gating_results, source_file=source_file)
```

---

## Примеры использования

```python
from src.data.fcs_parser import load_fcs
from src.data.gating import GatingStrategy
from src.data.parameter_extraction import (
    ParameterExtractor,
    ExtractionConfig,
    extract_model_parameters,
)

# Загрузка и гейтирование
loader = load_fcs("data/sample.fcs")
df = loader.to_dataframe()

strategy = GatingStrategy()
gating_results = strategy.apply(df)

# Извлечение параметров (способ 1)
extractor = ParameterExtractor()
params = extractor.extract(gating_results, source_file="sample.fcs")

print(f"N0 = {params.n0:.0f} клеток/мкл")
print(f"C0 = {params.c0:.2f} нг/мл")
print(f"Воспаление = {params.inflammation_level:.2f}")

# Извлечение параметров (способ 2 - с конфигурацией)
config = ExtractionConfig(
    volume_ul=10.0,
    dilution_factor=2.0,
)
params = extract_model_parameters(gating_results, config=config)

# Конвертация в словарь для симуляции
sim_params = params.to_dict()
```

---

## Валидация

### Ожидаемые диапазоны параметров

| Параметр | Мин | Типичное | Макс |
|----------|-----|----------|------|
| N0 | 1,000 | 5,000 | 50,000 |
| stem_cell_fraction | 0.01 | 0.05 | 0.15 |
| macrophage_fraction | 0.01 | 0.03 | 0.10 |
| apoptotic_fraction | 0.01 | 0.02 | 0.10 |
| C0 | 1 | 10 | 100 |
| inflammation_level | 0 | 0.3 | 1.0 |

### Тестовые сценарии

| Сценарий | stem | macro | apopt | Ожидаемый inflammation |
|----------|------|-------|-------|------------------------|
| Норма | 0.05 | 0.03 | 0.02 | ~0.3 |
| Воспаление | 0.03 | 0.08 | 0.05 | ~0.7 |
| Регенерация | 0.10 | 0.02 | 0.01 | ~0.2 |

---

## Зависимости

- numpy
- pandas
- src.data.gating (GatingResults)
