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

---

## Расширенные параметры модели

### ExtendedModelParameters

**Назначение:** Dataclass для хранения всех 20 переменных полной модели регенерации.

**Атрибуты — Клеточные популяции (клеток/мкл):**

| Атрибут | Тип | Диапазон | Единицы | Описание |
|---------|-----|----------|---------|----------|
| P0 | float | 0 - 500000 | клеток/мкл | Тромбоциты |
| Ne0 | float | 0 - 50000 | клеток/мкл | Нейтрофилы (CD66b+) |
| M1_0 | float | 0 - 10000 | клеток/мкл | M1 макрофаги (провоспалительные) |
| M2_0 | float | 0 - 10000 | клеток/мкл | M2 макрофаги (репаративные) |
| F0 | float | 0 - 50000 | клеток/мкл | Фибробласты |
| Mf0 | float | 0 - 10000 | клеток/мкл | Миофибробласты (α-SMA+) |
| E0 | float | 0 - 20000 | клеток/мкл | Эндотелиальные клетки (CD31+) |
| S0 | float | 0 - 10000 | клеток/мкл | Стволовые клетки (CD34+) |

**Атрибуты — Цитокины (нг/мл):**

| Атрибут | Тип | Диапазон | Референс | Описание |
|---------|-----|----------|----------|----------|
| C_TNF | float | 0 - 100 | 0.1 | TNF-α (провоспалительный) |
| C_IL10 | float | 0 - 50 | 0.05 | IL-10 (противовоспалительный) |
| C_PDGF | float | 0 - 50 | 5.0 | PDGF (фактор роста) |
| C_VEGF | float | 0 - 10 | 0.5 | VEGF (ангиогенез) |
| C_TGFb | float | 0 - 100 | 1.0 | TGF-β (фиброз/заживление) |
| C_MCP1 | float | 0 - 50 | 0.2 | MCP-1/CCL2 (рекрутинг моноцитов) |
| C_IL8 | float | 0 - 50 | 0.1 | IL-8/CXCL8 (рекрутинг нейтрофилов) |

**Атрибуты — ECM и вспомогательные:**

| Атрибут | Тип | Диапазон | Описание |
|---------|-----|----------|----------|
| rho_collagen | float | [0, 1] | Плотность коллагена (нормализованная) |
| C_MMP | float | >= 0 | Матриксные металлопротеиназы (нг/мл) |
| rho_fibrin | float | [0, 1] | Плотность фибрина (нормализованная) |
| D | float | >= 0 | Сигнал повреждения (DAMPs) |
| O2 | float | [0, 1] | Уровень кислорода (нормализованный) |

**Атрибуты — Метаданные:**

| Атрибут | Тип | По умолчанию | Описание |
|---------|-----|--------------|----------|
| source_file | str \| None | None | Исходный файл |
| total_events | int | 0 | Всего событий flow cytometry |
| inflammation_level | float | 0.0 | Совместимость с ModelParameters |

---

#### ExtendedModelParameters.to_dict

**Сигнатура:**

```python
def to_dict(self) -> dict[str, Any]
```

**Поведение:** Возвращает словарь со всеми 20 переменными + метаданные.

**Тестовые сценарии:**

| Сценарий | Ожидаемый результат |
|----------|---------------------|
| Валидный объект | dict с >= 23 ключей (20 переменных + 3 метаданных) |
| Проверка ключей | Содержит "P0", "Ne0", "C_TNF", "rho_collagen", "D", "O2" |
| Типы значений | Все переменные — float или int |

**Инварианты:**
- Длина словаря >= 23
- Все 20 переменных модели присутствуют как ключи

---

#### ExtendedModelParameters.validate

**Сигнатура:**

```python
def validate(self) -> bool
```

**Поведение:** Проверяет все ограничения. Возвращает True или raises ValueError.

**Тестовые сценарии:**

| Сценарий | Входное значение | Ожидание |
|----------|-----------------|----------|
| Все корректные | P0=250000, Ne0=1000, ... | True |
| Отрицательные клетки | P0 = -1 | ValueError("P0 must be non-negative") |
| Отрицательные цитокины | C_TNF = -0.5 | ValueError("C_TNF must be non-negative") |
| O2 > 1 | O2 = 1.5 | ValueError("O2 must be in [0, 1]") |
| O2 < 0 | O2 = -0.1 | ValueError("O2 must be in [0, 1]") |
| rho_collagen > 1 | rho_collagen = 1.5 | ValueError("rho_collagen must be in [0, 1]") |
| rho_collagen < 0 | rho_collagen = -0.1 | ValueError |
| rho_fibrin > 1 | rho_fibrin = 2.0 | ValueError |
| D < 0 | D = -1.0 | ValueError("D must be non-negative") |
| C_MMP < 0 | C_MMP = -0.1 | ValueError |

**Граничные случаи:**
- P0 = 0 → True (допустимо)
- O2 = 0.0 и O2 = 1.0 → True (границы включены)
- rho_collagen = 0.0 и rho_collagen = 1.0 → True

**Инварианты:**
- Если validate() возвращает True, ни одно ограничение не нарушено
- Если ограничение нарушено, ValueError содержит имя поля

---

#### from_basic_parameters

**Сигнатура:**

```python
@classmethod
def from_basic_parameters(
    cls,
    basic: ModelParameters,
    total_cells: float | None = None,
) -> ExtendedModelParameters
```

**Поведение:** Конвертирует 6 параметров в 20. Недостающие оценивает эвристически.

**Алгоритм конвертации:**

```python
n0 = total_cells if total_cells else basic.n0
P0 = config.ref_platelet_density  # по умолчанию 250000
Ne0 = 0.0  # нет CD66b данных в базовом гейтинге
M_total = basic.macrophage_fraction * n0
M1_0 = M_total * 0.7  # 70% M1 (начальное воспаление)
M2_0 = M_total * 0.3  # 30% M2
S0 = basic.stem_cell_fraction * n0
E0 = 0.0  # нет CD31 данных
F0 = n0 * (1 - basic.stem_cell_fraction - basic.macrophage_fraction - basic.apoptotic_fraction) * 0.1
Mf0 = 0.0  # начальное состояние

# Цитокины масштабируются по inflammation_level
C_TNF = config.ref_TNF * (1 + 2 * basic.inflammation_level)
C_IL10 = config.ref_IL10 * (1 + basic.inflammation_level)
# ... аналогично для остальных
```

**Тестовые сценарии:**

| Сценарий | Вход | Ожидание |
|----------|------|----------|
| Нормальные параметры | n0=5000, macro=0.03, stem=0.05 | M1_0 ≈ 105, M2_0 ≈ 45, S0 = 250 |
| Высокое воспаление | inflammation=0.8 | C_TNF > ref_TNF |
| Ne0 без CD66b | стандартный гейтинг | Ne0 = 0.0 |
| E0 без CD31 | стандартный гейтинг | E0 = 0.0 |
| total_cells задано | total_cells=10000 | M1_0 рассчитан от 10000 |

**Инварианты:**
- Результат проходит validate()
- M1_0 + M2_0 ≈ macrophage_fraction × n0 (с погрешностью округления)
- S0 = stem_cell_fraction × n0

---

#### to_basic_parameters

**Сигнатура:**

```python
def to_basic_parameters(self) -> ModelParameters
```

**Поведение:** Агрегирует 20 параметров обратно в 6.

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Валидный Extended | ModelParameters с n0 > 0 |
| Round-trip | basic → from_basic → to_basic ≈ исходный basic (приближённо) |

**Инварианты:**
- Результат проходит ModelParameters.validate()
- n0 > 0

---

#### to_sde_state_vector

**Сигнатура:**

```python
def to_sde_state_vector(self) -> np.ndarray
```

**Поведение:** Возвращает массив из 20 элементов в фиксированном порядке.

**Порядок элементов:**

| Индекс | Переменная | Описание |
|--------|-----------|----------|
| 0 | P0 | Тромбоциты |
| 1 | Ne0 | Нейтрофилы |
| 2 | M1_0 | M1 макрофаги |
| 3 | M2_0 | M2 макрофаги |
| 4 | F0 | Фибробласты |
| 5 | Mf0 | Миофибробласты |
| 6 | E0 | Эндотелиальные |
| 7 | S0 | Стволовые |
| 8 | C_TNF | TNF-α |
| 9 | C_IL10 | IL-10 |
| 10 | C_PDGF | PDGF |
| 11 | C_VEGF | VEGF |
| 12 | C_TGFb | TGF-β |
| 13 | C_MCP1 | MCP-1 |
| 14 | C_IL8 | IL-8 |
| 15 | rho_collagen | Коллаген |
| 16 | C_MMP | MMP |
| 17 | rho_fibrin | Фибрин |
| 18 | D | Повреждение |
| 19 | O2 | Кислород |

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Валидный объект | ndarray shape=(20,), dtype=float64 |
| Порядок | vec[0] == P0, vec[7] == S0, vec[19] == O2 |
| Все нули | ndarray из нулей |

**Инварианты:**
- `len(result) == 20`
- `result.dtype == np.float64`
- `result[i] == getattr(self, VARIABLE_ORDER[i])`

---

### Расширенные методы ParameterExtractor

#### extract_extended

**Сигнатура:**

```python
def extract_extended(
    self,
    gating_results: GatingResults,
    source_file: str | None = None,
) -> ExtendedModelParameters
```

**Поведение:** Извлекает все 20 переменных из расширенных результатов гейтирования.

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| GatingResults с 8 гейтами | ExtendedModelParameters, validate() = True |
| GatingResults без "neutrophils" | KeyError или значение по умолчанию |

---

#### extract_neutrophil_fraction

**Сигнатура:**

```python
def extract_neutrophil_fraction(self, gating_results: GatingResults) -> float
```

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Гейт "neutrophils" есть, fraction=0.05 | 0.05 |
| Гейт "neutrophils" отсутствует | KeyError |

**Инварианты:** `0 <= result <= 1`

---

#### extract_endothelial_fraction

**Сигнатура:**

```python
def extract_endothelial_fraction(self, gating_results: GatingResults) -> float
```

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Гейт "endothelial" есть, fraction=0.03 | 0.03 |
| Гейт "endothelial" отсутствует | KeyError |

**Инварианты:** `0 <= result <= 1`

---

#### estimate_cytokine_profile

**Сигнатура:**

```python
def estimate_cytokine_profile(self, gating_results: GatingResults) -> dict[str, float]
```

**Поведение:** Оценивает 7 цитокинов из клеточного состава.

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Стандартные гейты | dict с 7 ключами: TNF, IL10, PDGF, VEGF, TGFb, MCP1, IL8 |
| Все значения | >= 0 |
| Высокая доля макрофагов | TNF > ref_TNF |

**Инварианты:**
- Ровно 7 ключей
- Все значения >= 0

---

#### estimate_ecm_state

**Сигнатура:**

```python
def estimate_ecm_state(self, gating_results: GatingResults) -> dict[str, float]
```

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Любые гейты | dict с 3 ключами: rho_collagen, C_MMP, rho_fibrin |
| rho_collagen | 0 <= value <= 1 |
| rho_fibrin | 0 <= value <= 1 |
| C_MMP | >= 0 |

**Инварианты:**
- Ровно 3 ключа
- rho_collagen и rho_fibrin в [0, 1]

---

### extract_extended_parameters

**Сигнатура:**

```python
def extract_extended_parameters(
    gating_results: GatingResults,
    config: ExtractionConfig | None = None,
    source_file: str | None = None,
) -> ExtendedModelParameters
```

**Поведение:** Convenience-обёртка. Создаёт ParameterExtractor и вызывает extract_extended().

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| С конфигом | ExtendedModelParameters |
| Без конфига (None) | ExtendedModelParameters с дефолтной конфигурацией |

---

### Расширенные поля ExtractionConfig

| Атрибут | Тип | По умолчанию | Описание |
|---------|-----|--------------|----------|
| ref_neutrophil_fraction | float | 0.05 | Референс доли нейтрофилов |
| ref_endothelial_fraction | float | 0.03 | Референс доли эндотелиальных |
| ref_m1_m2_ratio | float | 2.33 | M1/M2 соотношение (70/30) |
| ref_platelet_density | float | 250000.0 | Плотность тромбоцитов (клеток/мкл) |
| ref_TNF | float | 0.1 | Референс TNF-α (нг/мл) |
| ref_IL10 | float | 0.05 | Референс IL-10 (нг/мл) |
| ref_PDGF | float | 5.0 | Референс PDGF (нг/мл) |
| ref_VEGF | float | 0.5 | Референс VEGF (нг/мл) |
| ref_TGFb | float | 1.0 | Референс TGF-β (нг/мл) |
| ref_MCP1 | float | 0.2 | Референс MCP-1 (нг/мл) |
| ref_IL8 | float | 0.1 | Референс IL-8 (нг/мл) |
| neutrophil_cytokine_factor | float | 1.0 | Вклад нейтрофилов |
| endothelial_cytokine_factor | float | 0.5 | Вклад эндотелиальных |

---

## Зависимости

- numpy
- pandas
- src.data.gating (GatingResults)
