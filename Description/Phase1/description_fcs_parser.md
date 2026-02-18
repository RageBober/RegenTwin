# Описание: fcs_parser.py

## Обзор

Модуль для загрузки и парсинга .fcs файлов flow cytometry.
Использует библиотеку FlowKit как backend.

---

## Классы

### FCSMetadata

**Назначение:** Dataclass для хранения метаданных FCS файла.

**Атрибуты:**

| Атрибут | Тип | Описание |
|---------|-----|----------|
| filename | str | Имя файла |
| n_events | int | Количество событий (клеток) |
| n_channels | int | Количество каналов |
| channels | list[str] | Список названий каналов |
| cytometer | str \| None | Название цитометра |
| date | str \| None | Дата создания файла |
| fcs_version | str \| None | Версия FCS формата (2.0, 3.0, 3.1) |

---

### FCSLoader

**Назначение:** Основной класс для загрузки и работы с FCS данными.

**Внутренние атрибуты:**

| Атрибут | Тип | Описание |
|---------|-----|----------|
| _sample | flowkit.Sample | Объект FlowKit Sample |
| _file_path | Path | Путь к загруженному файлу |
| _subsample | int \| None | Лимит событий |

---

## Методы

### __init__

**Назначение:** Инициализация загрузчика с опциональным субсэмплированием.

**Сигнатура:**

```python
def __init__(self, subsample: int | None = None) -> None
```

**Параметры:**

| Параметр | Тип | Описание | По умолчанию |
|----------|-----|----------|--------------|
| subsample | int \| None | Макс. количество событий | None (все) |

**Алгоритм:**

1. Сохранить параметр subsample
2. Инициализировать _sample = None, _file_path = None

---

### load

**Назначение:** Загрузка .fcs файла с диска.

**Сигнатура:**

```python
def load(self, file_path: str | Path) -> "FCSLoader"
```

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| file_path | str \| Path | Путь к .fcs файлу |

**Возвращает:** self (для цепочки вызовов)

**Алгоритм:**

1. Конвертировать file_path в Path
2. Проверить существование файла
3. Создать FlowKit Sample:
   ```python
   from flowkit import Sample
   self._sample = Sample(
       fcs_path_or_data=file_path,
       subsample=self._subsample,
       ignore_offset_error=True,
       ignore_offset_discrepancy=True
   )
   ```
4. Сохранить _file_path
5. Вернуть self

**Обработка ошибок:**
- FileNotFoundError: файл не существует
- ValueError: некорректный FCS формат (FlowKit выбросит исключение)

---

### get_channels

**Назначение:** Получение списка названий каналов.

**Сигнатура:**

```python
def get_channels(self) -> list[str]
```

**Возвращает:** Список строк с PnN идентификаторами каналов.

**Алгоритм:**

1. Проверить что _sample загружен
2. Вернуть self._sample.pnn_labels

**Пример результата:**
```python
['FSC-A', 'FSC-H', 'SSC-A', 'CD34-APC', 'CD14-PE', 'CD68-FITC', 'Annexin-V-Pacific Blue']
```

---

### get_events

**Назначение:** Получение матрицы событий (сырых или обработанных).

**Сигнатура:**

```python
def get_events(
    self,
    source: str = "raw",
    channels: list[str] | None = None,
) -> np.ndarray
```

**Параметры:**

| Параметр | Тип | Описание | По умолчанию |
|----------|-----|----------|--------------|
| source | str | Тип данных: 'raw', 'comp', 'xform' | 'raw' |
| channels | list[str] \| None | Фильтр каналов | None (все) |

**Возвращает:** NumPy массив shape [n_events, n_channels]

**Алгоритм:**

1. Проверить что _sample загружен
2. Получить данные: `events = self._sample.get_events(source=source)`
3. Если channels указаны:
   - Получить индексы нужных каналов
   - Отфильтровать столбцы
4. Вернуть массив

**Примечания:**
- 'raw' — сырые некомпенсированные данные
- 'comp' — компенсированные данные
- 'xform' — трансформированные данные (logicle и т.д.)

---

### get_metadata

**Назначение:** Извлечение метаданных из FCS файла.

**Сигнатура:**

```python
def get_metadata(self) -> FCSMetadata
```

**Возвращает:** FCSMetadata dataclass

**Алгоритм:**

1. Получить raw metadata: `meta = self._sample.get_metadata()`
2. Извлечь ключевые поля:
   - filename из _file_path.name
   - n_events из self._sample.event_count
   - n_channels из len(self._sample.pnn_labels)
   - channels из self._sample.pnn_labels
   - cytometer из meta.get('$CYT', None)
   - date из meta.get('$DATE', None)
   - fcs_version из meta.get('FCSversion', None)
3. Вернуть FCSMetadata

---

### to_dataframe

**Назначение:** Конвертация данных в pandas DataFrame.

**Сигнатура:**

```python
def to_dataframe(
    self,
    source: str = "raw",
    channels: list[str] | None = None,
) -> pd.DataFrame
```

**Возвращает:** DataFrame с колонками-каналами

**Алгоритм:**

1. Использовать self._sample.as_dataframe(source=source)
2. Если channels указаны — отфильтровать колонки
3. Вернуть DataFrame

---

### get_channel_data

**Назначение:** Получение данных одного канала.

**Сигнатура:**

```python
def get_channel_data(self, channel: str, source: str = "raw") -> np.ndarray
```

**Возвращает:** 1D NumPy массив

**Алгоритм:**

1. Использовать self._sample.get_channel_events(channel, source=source)

---

### validate_required_channels

**Назначение:** Проверка наличия обязательных каналов для RegenTwin.

**Сигнатура:**

```python
def validate_required_channels(self, required: list[str]) -> bool
```

**Обязательные каналы для RegenTwin:**
- FSC-A (Forward Scatter)
- SSC-A (Side Scatter)
- CD34 (стволовые клетки)
- CD14 или CD68 (макрофаги)
- Annexin-V (апоптоз)

**Алгоритм:**

1. Получить список каналов файла
2. Для каждого required канала:
   - Проверить точное совпадение
   - Если нет — поиск по подстроке (CD34 matches CD34-APC)
3. Собрать список отсутствующих
4. Если есть отсутствующие — raise ValueError с перечислением
5. Иначе return True

---

## Функции

### load_fcs

**Назначение:** Convenience функция для быстрой загрузки.

**Сигнатура:**

```python
def load_fcs(file_path: str | Path, subsample: int | None = None) -> FCSLoader
```

**Алгоритм:**

```python
return FCSLoader(subsample=subsample).load(file_path)
```

---

## Примеры использования

```python
from src.data.fcs_parser import FCSLoader, load_fcs

# Способ 1: через класс
loader = FCSLoader(subsample=10000)
loader.load("data/sample.fcs")

channels = loader.get_channels()
# ['FSC-A', 'FSC-H', 'SSC-A', 'CD34-APC', 'CD14-PE', 'CD68-FITC', 'Annexin-V']

metadata = loader.get_metadata()
# FCSMetadata(filename='sample.fcs', n_events=10000, ...)

events = loader.get_events()  # [10000, 7] ndarray

df = loader.to_dataframe()  # pandas DataFrame

# Способ 2: convenience функция
loader = load_fcs("data/sample.fcs")

# Проверка обязательных каналов
loader.validate_required_channels(['FSC-A', 'SSC-A', 'CD34'])
```

---

## Зависимости

- flowkit>=1.0.0
- numpy
- pandas
- pathlib (stdlib)
