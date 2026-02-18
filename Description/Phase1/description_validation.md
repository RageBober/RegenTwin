# Описание: validation.py

## Обзор

Модуль валидации данных для RegenTwin. Обеспечивает проверку целостности
и формата данных на всех этапах пайплайна: от загрузки FCS файлов до
проверки параметров модели и результатов гейтирования.

---

## Теоретическое обоснование

Качество входных данных критически важно для корректности симуляции.
Невалидные данные (отрицательные концентрации, нарушенная иерархия
гейтов, немонотонные временные ряды) приводят к нефизичным результатам.

Модуль реализует трёхуровневую систему валидации:
- **STRICT** — для production: любое отклонение = ошибка
- **NORMAL** — для разработки: отклонения диапазонов = warning
- **LENIENT** — для прототипирования: только критические проверки

---

## Классы

### ValidationLevel

**Назначение:** Enum уровней строгости валидации.

**Значения:**

| Значение | Описание |
|----------|----------|
| STRICT | Любое отклонение → error |
| NORMAL | Отсутствие required → error, нарушение диапазона → warning |
| LENIENT | Только критические проверки (наличие обязательных колонок) |

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| `ValidationLevel("strict")` | ValidationLevel.STRICT |
| `ValidationLevel("invalid")` | ValueError |
| `ValidationLevel.NORMAL.value` | "normal" |

---

### ValidationResult

**Назначение:** Хранит результат валидации: статус, ошибки, предупреждения.

**Атрибуты:**

| Атрибут | Тип | Описание |
|---------|-----|----------|
| is_valid | bool | True если нет ошибок |
| errors | list[str] | Список ошибок |
| warnings | list[str] | Список предупреждений |
| metadata | dict[str, Any] | Доп. информация (schema_name, n_rows, etc.) |

**Инварианты:**
- `is_valid == (len(errors) == 0)` — ВСЕГДА должно выполняться

#### summary

**Сигнатура:**

```python
def summary(self) -> str
```

**Поведение:** Возвращает строку вида:
- `"Validation: PASS. 0 errors, 2 warnings."`
- `"Validation: FAIL. 3 errors, 1 warnings. Errors: [...]"`

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| is_valid=True, 0 errors, 0 warnings | содержит "PASS" |
| is_valid=False, 2 errors | содержит "FAIL" и "2 errors" |
| 5+ errors | показывает первые 5 |

---

### ColumnSchema

**Назначение:** Описание одной колонки данных.

**Атрибуты:**

| Атрибут | Тип | По умолчанию | Описание |
|---------|-----|--------------|----------|
| name | str | — | Имя колонки |
| dtype | str | — | Тип: 'float', 'int', 'str', 'bool' |
| required | bool | True | Обязательная? |
| min_value | float \| None | None | Минимум (None = без ограничений) |
| max_value | float \| None | None | Максимум |
| allowed_values | list \| None | None | Допустимые значения (категории) |
| description | str | "" | Описание |

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Создание с именем и типом | Объект с required=True по умолчанию |
| min_value=0, max_value=1 | Валидация проверяет диапазон |

---

### DataSchema

**Назначение:** Описание таблицы данных.

**Атрибуты:**

| Атрибут | Тип | По умолчанию | Описание |
|---------|-----|--------------|----------|
| name | str | — | Имя схемы |
| columns | list[ColumnSchema] | — | Описания колонок |
| min_rows | int | 1 | Мин. строк |
| max_rows | int \| None | None | Макс. строк |
| description | str | "" | Описание |

#### get_required_columns

**Сигнатура:**

```python
def get_required_columns(self) -> list[str]
```

**Поведение:** Возвращает имена колонок с `required=True`.

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| FCS_DATA_SCHEMA | ["FSC-A", "FSC-H", "SSC-A", "CD34", "Annexin-V"] |
| Все optional | [] |
| Все required | все имена |

**Инварианты:**
- Результат — подмножество имён всех колонок
- Длина <= len(columns)

---

## Предопределённые схемы

### FCS_DATA_SCHEMA

| Колонка | Тип | Required | Min | Max | Описание |
|---------|-----|----------|-----|-----|----------|
| FSC-A | float | Да | 0 | — | Forward Scatter Area |
| FSC-H | float | Да | 0 | — | Forward Scatter Height |
| SSC-A | float | Да | 0 | — | Side Scatter Area |
| CD34 | float | Да | 0 | — | CD34-APC (стволовые) |
| CD14 | float | Нет | 0 | — | CD14-PE (моноциты) |
| CD68 | float | Нет | 0 | — | CD68-FITC (макрофаги) |
| Annexin-V | float | Да | 0 | — | Annexin-V (апоптоз) |
| CD66b | float | Нет | 0 | — | CD66b (нейтрофилы) |
| CD31 | float | Нет | 0 | — | CD31 (эндотелий) |

min_rows = 100

### TIME_SERIES_SCHEMA

| Колонка | Тип | Required | Min | Max | Описание |
|---------|-----|----------|-----|-----|----------|
| time | float | Да | 0 | — | Время (часы) |
| cell_count | float | Нет | 0 | — | Кол-во клеток |
| wound_area | float | Нет | 0 | 1 | Площадь раны (норм.) |

min_rows = 2

### CYTOKINE_TIMESERIES_SCHEMA

| Колонка | Тип | Required | Min | Max | Описание |
|---------|-----|----------|-----|-----|----------|
| time | float | Да | 0 | — | Время (часы) |
| TNF_alpha | float | Нет | 0 | — | TNF-α (нг/мл) |
| IL_10 | float | Нет | 0 | — | IL-10 (нг/мл) |
| PDGF | float | Нет | 0 | — | PDGF (нг/мл) |
| VEGF | float | Нет | 0 | — | VEGF (нг/мл) |
| TGF_beta | float | Нет | 0 | — | TGF-β (нг/мл) |
| MCP_1 | float | Нет | 0 | — | MCP-1 (нг/мл) |
| IL_8 | float | Нет | 0 | — | IL-8 (нг/мл) |

min_rows = 2

---

## Методы DataValidator

### __init__

**Сигнатура:**

```python
def __init__(self, level: ValidationLevel | str = ValidationLevel.NORMAL) -> None
```

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| `DataValidator()` | level = NORMAL |
| `DataValidator("strict")` | level = STRICT |
| `DataValidator(ValidationLevel.LENIENT)` | level = LENIENT |
| `DataValidator("invalid")` | ValueError |

---

### validate_dataframe

**Сигнатура:**

```python
def validate_dataframe(self, data: pd.DataFrame, schema: DataSchema) -> ValidationResult
```

**Алгоритм:**

1. Проверить наличие обязательных колонок
2. Проверить min_rows / max_rows
3. Для каждой существующей колонки со schema:
   - Проверить min_value / max_value
   - Уровень строгости определяет error vs warning
4. Собрать ValidationResult

**Тестовые сценарии:**

| Сценарий | Level | Ожидание |
|----------|-------|----------|
| Все колонки корректны | any | is_valid=True |
| Отсутствует required колонка | any | is_valid=False, errors содержит имя |
| Значение < min_value | STRICT | is_valid=False, error |
| Значение < min_value | NORMAL | is_valid=True, warning |
| 0 строк при min_rows=1 | any | is_valid=False |
| 1000 строк при max_rows=100 | any | is_valid=False (или warning) |
| Лишние колонки | any | игнорируются (is_valid=True) |
| Пустой DataFrame | any | is_valid=False (min_rows >= 1) |

**Инварианты:**
- `result.is_valid == (len(result.errors) == 0)`

---

### validate_fcs_data

**Сигнатура:**

```python
def validate_fcs_data(self, data: pd.DataFrame) -> ValidationResult
```

**Поведение:** Вызывает `validate_dataframe(data, FCS_DATA_SCHEMA)`.

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| DataFrame с FSC-A, FSC-H, SSC-A, CD34, Annexin-V | is_valid=True |
| Нет FSC-A | is_valid=False |
| Нет CD66b (optional) | is_valid=True |
| 50 строк (< min 100) | is_valid=False |

---

### validate_time_series

**Сигнатура:**

```python
def validate_time_series(self, data: pd.DataFrame) -> ValidationResult
```

**Дополнительные проверки:**
- Колонка "time" монотонно возрастает: `np.all(np.diff(time) > 0)`
- Числовые значения неотрицательны

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| time=[0, 6, 24, 48] | is_valid=True |
| time=[0, 6, 3, 48] (не монотонно) | is_valid=False, error содержит "monoton" |
| time=[0] (1 точка при min_rows=2) | is_valid=False |
| cell_count=-5 | warning или error |

---

### validate_model_parameters

**Сигнатура:**

```python
def validate_model_parameters(self, parameters: Any) -> ValidationResult
```

**Поведение:** Вызывает `parameters.validate()`. Если ValueError → errors.

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Валидный ModelParameters | is_valid=True |
| ModelParameters с n0=-1 | is_valid=False, error содержит "n0" |
| Валидный ExtendedModelParameters | is_valid=True |
| ExtendedModelParameters с O2=1.5 | is_valid=False |

---

### validate_gating_results

**Сигнатура:**

```python
def validate_gating_results(self, gating_results: Any) -> ValidationResult
```

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Корректные гейты | is_valid=True |
| fraction > 1 | is_valid=False |
| fraction < 0 | is_valid=False |
| n_events < 0 | is_valid=False |
| total_events = 0 | is_valid=False или warning |
| Дочерний n_events > родительский | warning |

**Инварианты:**
- Все fraction в [0, 1]
- n_events >= 0

---

### validate_data (convenience)

**Сигнатура:**

```python
def validate_data(
    data: pd.DataFrame,
    schema: DataSchema | None = None,
    level: str = "normal",
) -> ValidationResult
```

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| С явной schema | использует переданную схему |
| schema=None, колонки содержат "FSC-A" | автоопределяет FCS_DATA_SCHEMA |
| schema=None, колонка "time" | автоопределяет TIME_SERIES_SCHEMA |

---

## Примеры использования

```python
from src.data.validation import DataValidator, FCS_DATA_SCHEMA, validate_data
import pandas as pd

# Валидация FCS данных
validator = DataValidator(level="strict")
data = pd.DataFrame({"FSC-A": [1.0], "FSC-H": [1.0], ...})
result = validator.validate_fcs_data(data)
print(result.summary())

# Валидация параметров модели
from src.data.parameter_extraction import ModelParameters
params = ModelParameters(n0=5000, ...)
result = validator.validate_model_parameters(params)
assert result.is_valid

# Convenience функция
result = validate_data(my_dataframe, level="normal")
```

---

## Зависимости

- numpy
- pandas
