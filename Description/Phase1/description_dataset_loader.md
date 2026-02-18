# Описание: dataset_loader.py

## Обзор

Модуль загрузки публичных датасетов для валидации моделей RegenTwin.
Поддерживает flow cytometry данные (FlowRepository), транскриптомные
временные ряды (GEO/NCBI) и локальные данные из data/validation/.

---

## Теоретическое обоснование

Валидация математической модели регенерации требует сравнения предсказаний
с экспериментальными данными (раздел 6 Mathematical Framework):

- **Flow cytometry (t=0):** CD34+%, CD14+%, CD66b+% → начальные условия
- **Временные ряды клеток:** динамика Ne, M1, M2, F, E → валидация траекторий
- **Цитокины:** TNF-α, IL-10, PDGF, VEGF, TGF-β → валидация сигналинга
- **Заживление раны:** площадь раны vs время → интегральная метрика

---

## Классы

### DatasetSource

**Назначение:** Enum источников данных.

**Значения:**

| Значение | Описание |
|----------|----------|
| FLOW_REPOSITORY | FlowRepository.org (FR-FCM-*) |
| GEO | Gene Expression Omnibus (GSE*) |
| LOCAL | Локальные файлы data/validation/ |

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| `DatasetSource("flow_repository")` | DatasetSource.FLOW_REPOSITORY |
| `DatasetSource("invalid")` | ValueError |
| `DatasetSource.LOCAL.value` | "local" |

---

### DatasetMetadata

**Назначение:** Метаданные датасета (без самих данных).

**Атрибуты:**

| Атрибут | Тип | По умолчанию | Описание |
|---------|-----|--------------|----------|
| source | DatasetSource | — | Источник данных |
| dataset_id | str | — | Уникальный ID |
| description | str | — | Описание |
| file_paths | list[str] | [] | Пути к файлам |
| species | str | "human" | Вид организма |
| tissue_type | str | "skin" | Тип ткани |
| n_samples | int | 0 | Количество образцов |
| time_points | list[float] \| None | None | Временные точки (часы) |
| url | str \| None | None | URL источника |
| citation | str \| None | None | Ссылка на публикацию |

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Создание с обязательными полями | Корректный объект |
| Проверка defaults | species="human", tissue_type="skin" |

---

### TimeSeriesData

**Назначение:** Временной ряд с несколькими переменными.

**Атрибуты:**

| Атрибут | Тип | Описание |
|---------|-----|----------|
| time_points | np.ndarray | Временные точки (часы), shape=(n,) |
| values | dict[str, np.ndarray] | {имя_переменной: массив}, каждый shape=(n,) |
| units | dict[str, str] | {имя_переменной: единица_измерения} |
| metadata | DatasetMetadata \| None | Источник данных |

#### to_dataframe

**Сигнатура:**

```python
def to_dataframe(self) -> pd.DataFrame
```

**Поведение:** Создаёт DataFrame с колонкой "time" и колонками для каждой переменной.

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| time=[0,6,24], values={"Ne": [100,500,200]} | DataFrame 3 строки, колонки ["time", "Ne"] |
| Пустой values | DataFrame с одной колонкой "time" |
| 3 переменные | 4 колонки (time + 3 переменных) |

**Инварианты:**
- "time" всегда первая колонка
- Число строк == len(time_points)
- Число колонок == 1 + len(values)

#### get_variable

**Сигнатура:**

```python
def get_variable(self, name: str) -> np.ndarray
```

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Существующая переменная "Ne" | np.ndarray со значениями |
| Несуществующая "XYZ" | KeyError |

#### interpolate

**Сигнатура:**

```python
def interpolate(self, new_time_points: np.ndarray) -> TimeSeriesData
```

**Поведение:** Линейная интерполяция всех переменных на новую сетку.

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Те же точки | Значения не меняются |
| Промежуточные точки | Интерполированные значения |
| Больше точек | len(result.time_points) == len(new_time_points) |

**Инварианты:**
- len(result.time_points) == len(new_time_points)
- Ключи values сохраняются
- units сохраняются

---

### ValidationDataset

**Назначение:** Полный датасет для валидации (может содержать разные типы данных).

**Атрибуты:**

| Атрибут | Тип | Описание |
|---------|-----|----------|
| metadata | DatasetMetadata | Метаданные |
| cell_counts | TimeSeriesData \| None | Временной ряд клеточных популяций |
| cytokine_levels | TimeSeriesData \| None | Временной ряд цитокинов |
| fcs_data | pd.DataFrame \| None | Raw FCS данные |
| wound_closure | TimeSeriesData \| None | Динамика заживления раны |
| raw_data | dict[str, Any] | Прочие данные |

#### get_initial_conditions

**Сигнатура:**

```python
def get_initial_conditions(self) -> dict[str, float]
```

**Поведение:** Извлекает значения при t=0 из cell_counts и cytokine_levels.

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| cell_counts с t=0 | dict с ключами переменных модели |
| Нет cell_counts | пустой dict или только cytokine данные |
| cell_counts и cytokine_levels | объединённый dict |

**Инварианты:**
- Все значения — float
- Ключи — имена переменных модели

#### get_validation_targets

**Сигнатура:**

```python
def get_validation_targets(self) -> dict[str, TimeSeriesData]
```

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Все данные есть | dict с 3 ключами |
| Только cell_counts | dict с 1 ключом "cell_counts" |
| Ничего нет | пустой dict |

---

## Реестр AVAILABLE_DATASETS

| ID | Source | Описание | Time Points |
|----|--------|----------|-------------|
| FR-FCM-wound-healing | FLOW_REPOSITORY | Flow cytometry раны | — |
| GSE28914 | GEO | Транскриптомика раны | [0, 1, 3, 5, 7, 14] дней |
| local-mock | LOCAL | Мок-данные для тестов | — |

**Инварианты:**
- `AVAILABLE_DATASETS` — непустой dict
- Все значения — DatasetMetadata
- Минимум 3 записи

---

## Методы DatasetLoader

### __init__

**Сигнатура:**

```python
def __init__(self, cache_dir: str | Path | None = None) -> None
```

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| `DatasetLoader()` | cache_dir = Path("data/validation") |
| `DatasetLoader("custom/path")` | cache_dir = Path("custom/path") |
| `DatasetLoader(Path("/tmp"))` | cache_dir = Path("/tmp") |

---

### list_available

**Сигнатура:**

```python
def list_available(self) -> list[DatasetMetadata]
```

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Стандартный реестр | list, len >= 3 |
| Все элементы | isinstance(item, DatasetMetadata) |

**Инварианты:**
- len(result) == len(AVAILABLE_DATASETS)
- Все элементы — DatasetMetadata

---

### load

**Сигнатура:**

```python
def load(self, dataset_id: str) -> ValidationDataset
```

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Несуществующий id | KeyError |
| Данные недоступны | FileNotFoundError |
| Повторный вызов с тем же id | тот же объект из кэша |
| Валидный локальный id | ValidationDataset |

**Инварианты:**
- Результат — ValidationDataset
- metadata.dataset_id == переданный dataset_id

---

### download

**Сигнатура:**

```python
def download(self, dataset_id: str, target_dir: str | Path | None = None) -> Path
```

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Несуществующий id | KeyError |
| Нет подключения | ConnectionError |
| Успешная загрузка | Path к директории с файлами |

---

### validate_dataset

**Сигнатура:**

```python
def validate_dataset(self, dataset: ValidationDataset) -> bool
```

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Корректный датасет | True |
| Немонотонные time_points | ValueError |
| Отрицательные значения | ValueError |
| Пустой metadata.dataset_id | ValueError |

---

### _load_local

**Сигнатура:**

```python
def _load_local(self, metadata: DatasetMetadata) -> ValidationDataset
```

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Директория с .fcs файлами | fcs_data != None |
| Директория с .csv файлами | cell_counts или cytokine_levels != None |
| Пустая директория | ValidationDataset с None полями |

---

### _load_fcs_files

**Сигнатура:**

```python
def _load_fcs_files(self, directory: Path) -> pd.DataFrame | None
```

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Директория с 3 .fcs файлами | DataFrame (конкатенация) |
| Пустая директория | None |
| Несуществующая директория | None |

---

### _load_time_series

**Сигнатура:**

```python
def _load_time_series(self, file_path: Path) -> TimeSeriesData | None
```

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Корректный CSV файл | TimeSeriesData |
| Корректный JSON файл | TimeSeriesData |
| Неподдерживаемый формат (.xlsx) | None |
| Несуществующий файл | None или FileNotFoundError |

---

### load_dataset (convenience)

**Сигнатура:**

```python
def load_dataset(dataset_id: str, cache_dir: str | Path | None = None) -> ValidationDataset
```

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Валидный id | ValidationDataset |
| Невалидный id | KeyError |

---

## Примеры использования

```python
from src.data.dataset_loader import DatasetLoader, load_dataset, AVAILABLE_DATASETS

# Список доступных датасетов
loader = DatasetLoader()
datasets = loader.list_available()
for ds in datasets:
    print(f"{ds.dataset_id}: {ds.description}")

# Загрузка датасета
dataset = loader.load("local-mock")

# Извлечение начальных условий
initial = dataset.get_initial_conditions()
print(f"Начальные условия: {initial}")

# Получение временных рядов для валидации
targets = dataset.get_validation_targets()
for name, ts in targets.items():
    df = ts.to_dataframe()
    print(f"{name}: {len(df)} точек")

# Convenience функция
dataset = load_dataset("GSE28914")
```

---

## Зависимости

- numpy
- pandas
- pathlib (стандартная библиотека)
- scipy (для интерполяции)
- requests (опционально, для download)
