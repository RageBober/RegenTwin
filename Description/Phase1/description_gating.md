# Описание: gating.py

## Обзор

Модуль автоматического гейтирования flow cytometry данных.
Реализует иерархическую стратегию выделения клеточных популяций.

---

## Теоретическое обоснование

### Иерархия гейтирования

```
All Events (100%)
    │
    ├─ Debris (низкий FSC/SSC) ──→ Исключается
    │
    └─ Non-Debris
         │
         ├─ Дублеты (FSC-A ≠ FSC-H) ──→ Исключается
         │
         └─ Singlets (одиночные клетки)
              │
              ├─ Апоптотические (высокий Annexin-V)
              │
              └─ Живые клетки (низкий Annexin-V)
                   │
                   ├─ CD34+ стволовые клетки
                   │
                   └─ CD14+/CD68+ макрофаги
```

### Характеристики популяций

| Популяция | FSC | SSC | CD34 | CD14 | CD68 | Annexin-V |
|-----------|-----|-----|------|------|------|-----------|
| Debris | Низкий | Низкий | - | - | - | - |
| Живые клетки | Средний | Средний | Низкий | Низкий | Низкий | Низкий |
| CD34+ стволовые | Средний | Низкий | **Высокий** | Низкий | Низкий | Низкий |
| Макрофаги | Средний | **Высокий** | Низкий | **Высокий** | **Высокий** | Варьирует |
| Апоптотические | Сниженный | - | - | - | - | **Высокий** |

---

## Классы

### GateResult

**Назначение:** Результат применения одного гейта.

**Атрибуты:**

| Атрибут | Тип | Описание |
|---------|-----|----------|
| name | str | Название гейта |
| mask | np.ndarray | Boolean маска [n_events] |
| n_events | int | Количество событий в гейте |
| fraction | float | Процент от родителя (0-1) |
| parent | str \| None | Название родительского гейта |
| statistics | dict | Дополнительная статистика |

---

### GatingResults

**Назначение:** Контейнер для всех результатов гейтирования.

**Атрибуты:**

| Атрибут | Тип | Описание |
|---------|-----|----------|
| total_events | int | Общее количество событий |
| gates | dict[str, GateResult] | Словарь гейтов |

**Методы:**

#### get_population

```python
def get_population(self, name: str) -> np.ndarray
```

Возвращает boolean маску для указанной популяции.

#### get_statistics

```python
def get_statistics(self) -> dict[str, Any]
```

Возвращает сводку:
```python
{
    "total_events": 10000,
    "debris_fraction": 0.20,
    "live_cells_fraction": 0.70,
    "cd34_positive_fraction": 0.05,
    "macrophage_fraction": 0.03,
    "apoptotic_fraction": 0.02,
}
```

---

### GatingStrategy

**Назначение:** Основной класс стратегии гейтирования.

---

## Методы

### __init__

**Сигнатура:**

```python
def __init__(
    self,
    channel_mapping: dict[str, str] | None = None,
) -> None
```

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| channel_mapping | dict \| None | Маппинг названий каналов |

**Алгоритм:**

1. Если channel_mapping = None, использовать DEFAULT_CHANNELS
2. Иначе объединить с DEFAULT_CHANNELS (user values override)
3. Сохранить в self._channels

**Пример маппинга:**
```python
{
    "fsc_area": "FSC-A",      # Forward Scatter Area
    "fsc_height": "FSC-H",    # Forward Scatter Height
    "ssc_area": "SSC-A",      # Side Scatter Area
    "cd34": "CD34-APC",       # CD34 в канале APC
    "cd14": "CD14-PE",        # CD14 в канале PE
    "cd68": "CD68-FITC",      # CD68 в канале FITC
    "annexin": "Annexin-V-Pacific Blue",
}
```

---

### apply

**Назначение:** Применение полной стратегии гейтирования.

**Сигнатура:**

```python
def apply(self, data: pd.DataFrame | np.ndarray) -> GatingResults
```

**Алгоритм:**

1. Конвертировать данные в DataFrame если ndarray
2. Извлечь каналы по маппингу
3. **Шаг 1: Debris gate**
   - `non_debris = self.debris_gate(fsc, ssc)`
4. **Шаг 2: Singlets gate** (на non_debris)
   - `singlets = self.singlets_gate(fsc_a, fsc_h) & non_debris`
5. **Шаг 3: Live cells gate** (на singlets)
   - `live = self.live_cells_gate(annexin) & singlets`
6. **Шаг 4: CD34+ gate** (на live)
   - `cd34_pos = self.cd34_gate(cd34) & live`
7. **Шаг 5: Macrophage gate** (на live)
   - `macrophages = self.macrophage_gate(cd14, cd68) & live`
8. **Шаг 6: Apoptotic gate** (на singlets)
   - `apoptotic = self.apoptotic_gate(annexin) & singlets`
9. Собрать GatingResults

---

### debris_gate

**Назначение:** Исключение debris (мусора, обломков клеток).

**Сигнатура:**

```python
def debris_gate(
    self,
    fsc: np.ndarray,
    ssc: np.ndarray,
    fsc_threshold: float | None = None,
    ssc_threshold: float | None = None,
) -> np.ndarray
```

**Алгоритм:**

1. Если threshold = None:
   - `fsc_threshold = np.percentile(fsc, 15)` (нижние 15%)
   - `ssc_threshold = np.percentile(ssc, 10)` (нижние 10%)
2. Создать маску:
   ```python
   mask = (fsc > fsc_threshold) & (ssc > ssc_threshold)
   ```
3. Вернуть mask (True = НЕ debris)

**Типичные пороги (мок-данные):**
- FSC threshold: ~30,000
- SSC threshold: ~20,000

---

### singlets_gate

**Назначение:** Удаление дублетов (слипшихся клеток).

**Сигнатура:**

```python
def singlets_gate(
    self,
    fsc_a: np.ndarray,
    fsc_h: np.ndarray,
    tolerance: float = 0.1,
) -> np.ndarray
```

**Принцип:**
- Для одиночных клеток: FSC-A ≈ FSC-H (пропорционально)
- Для дублетов: FSC-A > FSC-H (площадь увеличена)

**Алгоритм:**

1. Вычислить соотношение: `ratio = fsc_a / (fsc_h + 1e-10)`
2. Вычислить медиану ratio для синглетов
3. Создать маску:
   ```python
   expected_ratio = np.median(ratio)
   mask = np.abs(ratio - expected_ratio) < (expected_ratio * tolerance)
   ```
4. Альтернатива — линейная регрессия:
   ```python
   from scipy import stats
   slope, intercept, r, p, se = stats.linregress(fsc_h, fsc_a)
   predicted = slope * fsc_h + intercept
   residuals = np.abs(fsc_a - predicted)
   mask = residuals < np.percentile(residuals, 90)
   ```

---

### live_cells_gate

**Назначение:** Выделение живых клеток (низкий Annexin-V).

**Сигнатура:**

```python
def live_cells_gate(
    self,
    annexin: np.ndarray,
    threshold: float | None = None,
) -> np.ndarray
```

**Алгоритм:**

1. Если threshold = None:
   - Использовать метод Оцу или GMM для бимодального распределения
   - Или `threshold = np.percentile(annexin, 85)`
2. Создать маску:
   ```python
   mask = annexin < threshold
   ```
3. Вернуть mask (True = живые)

---

### cd34_gate

**Назначение:** Выделение CD34+ стволовых клеток.

**Сигнатура:**

```python
def cd34_gate(
    self,
    cd34: np.ndarray,
    threshold: float | None = None,
    percentile: float = 95.0,
) -> np.ndarray
```

**Алгоритм:**

1. Если threshold = None:
   - `threshold = np.percentile(cd34, percentile)` (топ 5%)
   - Или использовать _auto_threshold с методом Оцу
2. Создать маску:
   ```python
   mask = cd34 > threshold
   ```

**Ожидаемый результат:** ~5% от живых клеток (референс из мок-данных)

---

### macrophage_gate

**Назначение:** Выделение макрофагов по CD14 и CD68.

**Сигнатура:**

```python
def macrophage_gate(
    self,
    cd14: np.ndarray,
    cd68: np.ndarray,
    cd14_threshold: float | None = None,
    cd68_threshold: float | None = None,
) -> np.ndarray
```

**Алгоритм:**

1. Автопороги если не указаны:
   - `cd14_threshold = np.percentile(cd14, 90)`
   - `cd68_threshold = np.percentile(cd68, 90)`
2. Два варианта гейтирования:
   - **OR логика:** `mask = (cd14 > cd14_threshold) | (cd68 > cd68_threshold)`
   - **AND логика:** `mask = (cd14 > cd14_threshold) & (cd68 > cd68_threshold)`
3. По умолчанию используем OR (более чувствительно)

---

### apoptotic_gate

**Назначение:** Выделение апоптотических клеток.

**Сигнатура:**

```python
def apoptotic_gate(
    self,
    annexin: np.ndarray,
    threshold: float | None = None,
) -> np.ndarray
```

**Алгоритм:**

1. threshold = инверсия от live_cells_gate
2. `mask = annexin > threshold`

---

### _auto_threshold

**Назначение:** Автоматическое определение порога методами ML.

**Сигнатура:**

```python
def _auto_threshold(
    self,
    data: np.ndarray,
    method: str = "otsu",
) -> float
```

**Методы:**

1. **Оцу (otsu):** Минимизация внутриклассовой дисперсии
   ```python
   from skimage.filters import threshold_otsu
   return threshold_otsu(data)
   ```

2. **Перцентиль (percentile):**
   ```python
   return np.percentile(data, 95)
   ```

3. **GMM (gmm):** Gaussian Mixture Model
   ```python
   from sklearn.mixture import GaussianMixture
   gmm = GaussianMixture(n_components=2)
   gmm.fit(data.reshape(-1, 1))
   # Порог = пересечение двух гауссиан
   ```

---

### _density_gate

**Назначение:** 2D гейтирование на основе плотности событий.

**Сигнатура:**

```python
def _density_gate(
    self,
    x: np.ndarray,
    y: np.ndarray,
    fraction: float = 0.85,
) -> np.ndarray
```

**Алгоритм (аналог FlowCal density2d):**

1. Построить 2D гистограмму / KDE
2. Найти область максимальной плотности
3. Расширять область пока не захватит `fraction` событий
4. Вернуть маску для событий в области

---

## Примеры использования

```python
from src.data.gating import GatingStrategy
from src.data.fcs_parser import load_fcs

# Загрузка данных
loader = load_fcs("data/sample.fcs")
df = loader.to_dataframe()

# Настройка маппинга каналов
channel_mapping = {
    "cd34": "CD34-APC",
    "cd14": "CD14-PE",
    "cd68": "CD68-FITC",
    "annexin": "Annexin-V-Pacific Blue",
}

# Применение гейтирования
strategy = GatingStrategy(channel_mapping=channel_mapping)
results = strategy.apply(df)

# Получение статистики
stats = results.get_statistics()
print(f"Живые клетки: {stats['live_cells_fraction']*100:.1f}%")
print(f"CD34+ стволовые: {stats['cd34_positive_fraction']*100:.1f}%")

# Получение маски популяции
cd34_mask = results.get_population("cd34_positive")
cd34_cells = df[cd34_mask]
```

---

## Валидация

Ожидаемые результаты для мок-данных:

| Популяция | Ожидаемая доля |
|-----------|----------------|
| Debris | ~20% |
| Live cells | ~70% |
| CD34+ | ~5% (от живых) |
| Macrophages | ~3% (от живых) |
| Apoptotic | ~2% (от singlets) |

---

---

## Расширенное гейтирование (новые популяции)

### Обновлённая иерархия гейтов

```
All Events (100%)
    │
    ├─ Debris (низкий FSC/SSC) ──→ Исключается
    │
    └─ Non-Debris
         │
         ├─ Дублеты (FSC-A ≠ FSC-H) ──→ Исключается
         │
         └─ Singlets (одиночные клетки)
              │
              ├─ Апоптотические (высокий Annexin-V)
              │
              └─ Живые клетки (низкий Annexin-V)
                   │
                   ├─ CD34+ стволовые клетки (S)
                   ├─ CD66b+ нейтрофилы (Ne)        ← НОВЫЙ
                   ├─ CD31+ эндотелиальные (E)       ← НОВЫЙ
                   └─ CD14+/CD68+ макрофаги (M1+M2)
```

### Обновлённая таблица характеристик популяций

| Популяция | FSC | SSC | CD34 | CD14 | CD68 | Annexin-V | CD66b | CD31 |
|-----------|-----|-----|------|------|------|-----------|-------|------|
| Debris | Низкий | Низкий | - | - | - | - | - | - |
| Живые клетки | Средний | Средний | Низкий | Низкий | Низкий | Низкий | Низкий | Низкий |
| CD34+ стволовые | Средний | Низкий | **Высокий** | Низкий | Низкий | Низкий | Низкий | Низкий |
| CD66b+ нейтрофилы | Средний | **Высокий** | Низкий | Низкий | Низкий | Варьирует | **Высокий** | Низкий |
| CD31+ эндотелий | Средний | Средний | Низкий | Низкий | Низкий | Низкий | Низкий | **Высокий** |
| Макрофаги | Средний | **Высокий** | Низкий | **Высокий** | **Высокий** | Варьирует | Низкий | Низкий |
| Апоптотические | Сниженный | - | - | - | - | **Высокий** | - | - |

### Обновлённый DEFAULT_CHANNELS

```python
DEFAULT_CHANNELS = {
    "fsc_area": "FSC-A",
    "fsc_height": "FSC-H",
    "ssc_area": "SSC-A",
    "cd34": "CD34-APC",
    "cd14": "CD14-PE",
    "cd68": "CD68-FITC",
    "annexin": "Annexin-V-Pacific Blue",
    "cd66b": "CD66b-PE-Cy7",    # НОВЫЙ
    "cd31": "CD31-BV421",       # НОВЫЙ
}
```

---

### neutrophil_gate

**Назначение:** Выделение CD66b+ нейтрофилов (популяция Ne в модели).

**Сигнатура:**

```python
def neutrophil_gate(
    self,
    cd66b: np.ndarray,
    threshold: float | None = None,
    percentile: float = 95.0,
) -> np.ndarray
```

**Параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| cd66b | np.ndarray | — | Массив значений CD66b канала |
| threshold | float \| None | None | Порог позитивности. None = автопорог |
| percentile | float | 95.0 | Перцентиль для автопорога |

**Возвращает:** `np.ndarray` (dtype=bool), True = CD66b+ нейтрофил

**Алгоритм:**

1. Если threshold = None:
   - `threshold = np.percentile(cd66b, percentile)`
2. `mask = cd66b > threshold`
3. Вернуть mask

**Тестовые сценарии:**

| Сценарий | Входные данные | Ожидаемый результат |
|----------|---------------|---------------------|
| Нормальные данные | Массив 1000 элементов, 5% имеют значения > percentile | маска с ~50 True значениями |
| Пустой массив | `np.array([])` | пустая маска `shape=(0,)`, dtype=bool |
| Все одинаковые | `np.full(100, 5.0)` | все False (ни одно > percentile) |
| Ручной порог | `cd66b=[1,2,3,4,5]`, threshold=3 | `[False, False, False, True, True]` |
| Один элемент | `np.array([100.0])` | `[False]` при percentile=95 |
| percentile=0 | любой массив | все True (порог = минимум) |
| percentile=100 | любой массив | все False (порог = максимум) |

**Граничные случаи:**
- Пустой массив → пустая маска
- NaN значения → поведение зависит от np.percentile (может вернуть NaN порог)
- Все значения нулевые → все False

**Ошибки:**
- Нет специфичных исключений (numpy обрабатывает пустые массивы)

**Инварианты:**
- `mask.shape == cd66b.shape`
- `mask.dtype == np.bool_`
- `0 <= mask.sum() <= len(cd66b)`

---

### endothelial_gate

**Назначение:** Выделение CD31+ эндотелиальных клеток (популяция E в модели).

**Сигнатура:**

```python
def endothelial_gate(
    self,
    cd31: np.ndarray,
    threshold: float | None = None,
    percentile: float = 95.0,
) -> np.ndarray
```

**Параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| cd31 | np.ndarray | — | Массив значений CD31 канала |
| threshold | float \| None | None | Порог позитивности. None = автопорог |
| percentile | float | 95.0 | Перцентиль для автопорога |

**Возвращает:** `np.ndarray` (dtype=bool), True = CD31+ эндотелий

**Алгоритм:**

Идентичен `neutrophil_gate`, но для CD31:
1. Если threshold = None:
   - `threshold = np.percentile(cd31, percentile)`
2. `mask = cd31 > threshold`
3. Вернуть mask

**Тестовые сценарии:**

| Сценарий | Входные данные | Ожидаемый результат |
|----------|---------------|---------------------|
| Нормальные данные | Массив 1000, ~3% высоких | маска с ~30 True |
| Пустой массив | `np.array([])` | пустая маска `shape=(0,)` |
| Все одинаковые | `np.full(100, 5.0)` | все False |
| Ручной порог | `cd31=[10,20,30,40,50]`, threshold=25 | `[False, False, True, True, True]` |

**Граничные случаи:** аналогичны `neutrophil_gate`

**Инварианты:**
- `mask.shape == cd31.shape`
- `mask.dtype == np.bool_`

---

### apply_extended

**Назначение:** Расширенное гейтирование с 9 каналами, включая нейтрофилы и эндотелий.

**Сигнатура:**

```python
def apply_extended(
    self,
    data: pd.DataFrame | np.ndarray,
) -> GatingResults
```

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| data | pd.DataFrame \| np.ndarray | Данные flow cytometry с 9 каналами |

Для ndarray порядок колонок: `[FSC-A, FSC-H, SSC-A, CD34, CD14, CD68, Annexin-V, CD66b, CD31]`

**Возвращает:** `GatingResults` с 8 популяциями

**Алгоритм:**

1. Извлечь все 9 каналов (как в `apply()` + cd66b, cd31)
2. Шаги 1-6 аналогичны `apply()`
3. **Шаг 7:** `neutrophils = self.neutrophil_gate(cd66b) & live`
4. **Шаг 8:** `endothelial = self.endothelial_gate(cd31) & live`
5. Собрать GatingResults с 8 ключами:
   - non_debris, singlets, live_cells, cd34_positive, macrophages,
     apoptotic, **neutrophils**, **endothelial**

**Тестовые сценарии:**

| Сценарий | Входные данные | Ожидаемый результат |
|----------|---------------|---------------------|
| 9-колоночный ndarray | shape=(10000, 9) | GatingResults с 8 ключами |
| DataFrame с 9 каналами | правильные имена колонок | GatingResults с 8 ключами |
| 7-колоночный ndarray | shape=(10000, 7) | ошибка (нет данных CD66b/CD31) |

**Ожидаемые доли популяций (мок-данные):**

| Популяция | Ожидаемая доля | Ключ в gates |
|-----------|----------------|--------------|
| Debris | ~20% | — (исключается) |
| Live cells | ~70% | "live_cells" |
| CD34+ | ~5% | "cd34_positive" |
| CD66b+ нейтрофилы | ~5% | "neutrophils" |
| CD31+ эндотелий | ~3% | "endothelial" |
| Макрофаги | ~3% | "macrophages" |
| Апоптотические | ~2% | "apoptotic" |

**Инварианты:**
- Все ключи присутствуют: "non_debris", "singlets", "live_cells", "cd34_positive", "macrophages", "apoptotic", "neutrophils", "endothelial"
- Для каждого GateResult: `0 <= fraction <= 1`
- `neutrophils.parent == "live_cells"`
- `endothelial.parent == "live_cells"`
- `n_events >= 0` для каждого гейта

---

## Зависимости

- numpy
- pandas
- scipy (для статистики)
- scikit-learn (опционально, для GMM)
- scikit-image (опционально, для Оцу)
