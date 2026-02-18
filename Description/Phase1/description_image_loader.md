# Описание: image_loader.py

## Обзор

Модуль для загрузки и анализа изображений scatter plots в форматах PNG и JPEG.
Предоставляет функционал для извлечения координат точек, их цветов и базового
анализа изображений для интеграции с pipeline обработки данных RegenTwin.

---

## Классы

### ImageConfig

**Назначение:** Dataclass конфигурации обработки изображений.

**Атрибуты:**

| Атрибут | Тип | Описание | По умолчанию |
|---------|-----|----------|--------------|
| max_dimension | int | Максимальный размер стороны (px) | 4096 |
| color_space | str | Цветовое пространство: 'RGB', 'RGBA', 'grayscale' | "RGB" |
| point_detection_method | str | Метод детекции точек: 'threshold', 'contour', 'hough' | "threshold" |
| min_point_radius | int | Минимальный радиус точки (px) | 2 |
| max_point_radius | int | Максимальный радиус точки (px) | 50 |
| color_quantization_bins | int | Количество бинов гистограммы | 256 |
| dominant_colors_count | int | Количество доминантных цветов | 5 |
| auto_detect_axes | bool | Автоопределение осей графика | True |
| axis_color_threshold | int | Порог яркости для детекции осей | 30 |

---

### ImageMetadata

**Назначение:** Dataclass для хранения метаданных изображения.

**Атрибуты:**

| Атрибут | Тип | Описание |
|---------|-----|----------|
| filename | str | Имя файла |
| width | int | Ширина изображения (px) |
| height | int | Высота изображения (px) |
| channels | int | Количество каналов (1, 3, или 4) |
| format | str | Формат файла: 'PNG', 'JPEG' |
| bit_depth | int | Битовая глубина (обычно 8) |
| file_size_bytes | int | Размер файла в байтах |
| has_alpha | bool | Наличие альфа-канала |

---

### ScatterPlotData

**Назначение:** Dataclass результата извлечения данных из scatter plot.

**Атрибуты:**

| Атрибут | Тип | Описание |
|---------|-----|----------|
| points | np.ndarray | Координаты точек [n_points, 2] в пикселях |
| points_normalized | np.ndarray | Нормализованные координаты [n_points, 2] в [0, 1] |
| colors | np.ndarray | Цвета точек [n_points, 3] в RGB |
| color_labels | np.ndarray \| None | Метки кластеров цветов [n_points] |
| n_points | int | Общее количество точек |
| detection_confidence | float | Уверенность детекции [0, 1] |
| plot_bounds | tuple[int, int, int, int] \| None | Границы области графика |
| axis_labels | tuple[str \| None, str \| None] | Подписи осей (x_label, y_label) |

---

### ImageAnalysisResult

**Назначение:** Dataclass результата анализа изображения.

**Атрибуты:**

| Атрибут | Тип | Описание |
|---------|-----|----------|
| histogram_r | np.ndarray | Гистограмма красного канала [256] |
| histogram_g | np.ndarray | Гистограмма зелёного канала [256] |
| histogram_b | np.ndarray | Гистограмма синего канала [256] |
| histogram_gray | np.ndarray | Гистограмма grayscale [256] |
| dominant_colors | np.ndarray | Доминантные цвета [n_colors, 3] |
| dominant_colors_percentages | np.ndarray | Доли доминантных цветов [n_colors] |
| mean_color | tuple[float, float, float] | Средний цвет (R, G, B) |
| std_color | tuple[float, float, float] | Std цвета (R, G, B) |
| brightness | float | Средняя яркость [0, 255] |
| contrast | float | Контраст изображения |
| regions | list[dict] \| None | Обнаруженные регионы |

---

### ImageLoader

**Назначение:** Основной класс для загрузки изображений PNG/JPEG.

**Внутренние атрибуты:**

| Атрибут | Тип | Описание |
|---------|-----|----------|
| _config | ImageConfig | Конфигурация обработки |
| _image | np.ndarray \| None | Загруженное изображение |
| _file_path | Path \| None | Путь к файлу |
| _metadata | ImageMetadata \| None | Кэшированные метаданные |

**Константы класса:**

| Константа | Значение | Описание |
|-----------|----------|----------|
| SUPPORTED_FORMATS | [".png", ".jpg", ".jpeg"] | Поддерживаемые форматы |

---

### ScatterPlotExtractor

**Назначение:** Экстрактор данных из изображений scatter plots.

**Внутренние атрибуты:**

| Атрибут | Тип | Описание |
|---------|-----|----------|
| _config | ImageConfig | Конфигурация обработки |

---

### ImageAnalyzer

**Назначение:** Базовый анализатор изображений для расчёта статистик и детекции регионов.

**Внутренние атрибуты:**

| Атрибут | Тип | Описание |
|---------|-----|----------|
| _config | ImageConfig | Конфигурация обработки |

---

## Методы ImageConfig

### validate

**Назначение:** Валидация параметров конфигурации.

**Сигнатура:**

```python
def validate(self) -> bool
```

**Возвращает:** True если все параметры корректны.

**Алгоритм:**

1. Проверить max_dimension > 0
2. Проверить color_space в ['RGB', 'RGBA', 'grayscale']
3. Проверить point_detection_method в ['threshold', 'contour', 'hough']
4. Проверить 0 < min_point_radius < max_point_radius
5. Проверить color_quantization_bins > 0
6. Проверить dominant_colors_count > 0
7. Если всё корректно — return True
8. Иначе — raise ValueError с описанием

---

## Методы ImageLoader

### __init__

**Назначение:** Инициализация загрузчика с конфигурацией.

**Сигнатура:**

```python
def __init__(self, config: ImageConfig | None = None) -> None
```

**Параметры:**

| Параметр | Тип | Описание | По умолчанию |
|----------|-----|----------|--------------|
| config | ImageConfig \| None | Конфигурация обработки | None (по умолчанию) |

**Алгоритм:**

1. Если config is None — создать ImageConfig()
2. Сохранить _config = config
3. Инициализировать _image = None, _file_path = None, _metadata = None

---

### load

**Назначение:** Загрузка изображения с диска.

**Сигнатура:**

```python
def load(self, file_path: str | Path) -> "ImageLoader"
```

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| file_path | str \| Path | Путь к файлу изображения |

**Возвращает:** self (для цепочки вызовов)

**Алгоритм:**

1. Конвертировать file_path в Path
2. Проверить существование файла
3. Проверить расширение в SUPPORTED_FORMATS
4. Загрузить изображение с помощью PIL/Pillow:
   ```python
   from PIL import Image
   pil_image = Image.open(file_path)
   self._image = np.array(pil_image)
   ```
5. Конвертировать в нужное цветовое пространство (RGB)
6. Если размер > max_dimension — масштабировать
7. Сохранить _file_path
8. Вернуть self

**Обработка ошибок:**
- FileNotFoundError: файл не существует
- ValueError: неподдерживаемый формат

---

### get_image

**Назначение:** Получение загруженного изображения.

**Сигнатура:**

```python
def get_image(self) -> np.ndarray
```

**Возвращает:** NumPy массив shape [H, W, C] или [H, W]

**Алгоритм:**

1. Проверить что _image is not None
2. Вернуть копию self._image.copy()

---

### get_metadata

**Назначение:** Получение метаданных изображения.

**Сигнатура:**

```python
def get_metadata(self) -> ImageMetadata
```

**Возвращает:** ImageMetadata dataclass

**Алгоритм:**

1. Проверить что _image is not None
2. Если _metadata кэширована — вернуть
3. Определить format по расширению
4. Определить channels из shape
5. Получить file_size_bytes через os.path.getsize
6. Создать ImageMetadata и сохранить в _metadata
7. Вернуть _metadata

---

### to_grayscale

**Назначение:** Конвертация изображения в grayscale.

**Сигнатура:**

```python
def to_grayscale(self) -> np.ndarray
```

**Возвращает:** NumPy массив shape [H, W]

**Алгоритм:**

1. Проверить что _image is not None
2. Если уже grayscale (2D) — вернуть копию
3. Иначе использовать формулу: gray = 0.299*R + 0.587*G + 0.114*B
4. Вернуть результат

---

### resize

**Назначение:** Изменение размера изображения.

**Сигнатура:**

```python
def resize(self, width: int, height: int) -> np.ndarray
```

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| width | int | Новая ширина (px) |
| height | int | Новая высота (px) |

**Возвращает:** NumPy массив с изменённым размером

**Алгоритм:**

1. Проверить что _image is not None
2. Проверить width > 0 и height > 0
3. Использовать PIL или skimage для resize с интерполяцией
4. Вернуть результат

---

### crop

**Назначение:** Обрезка области изображения.

**Сигнатура:**

```python
def crop(self, x: int, y: int, width: int, height: int) -> np.ndarray
```

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| x | int | X-координата левого верхнего угла |
| y | int | Y-координата левого верхнего угла |
| width | int | Ширина области |
| height | int | Высота области |

**Возвращает:** NumPy массив обрезанной области

**Алгоритм:**

1. Проверить что _image is not None
2. Проверить координаты в пределах изображения
3. Выполнить срез: self._image[y:y+height, x:x+width]
4. Вернуть результат

---

### validate_format

**Назначение:** Проверка поддерживаемого формата.

**Сигнатура:**

```python
def validate_format(self) -> bool
```

**Возвращает:** True если формат поддерживается

**Алгоритм:**

1. Проверить _file_path is not None
2. Получить расширение: _file_path.suffix.lower()
3. Проверить в SUPPORTED_FORMATS
4. Если нет — raise ValueError
5. Иначе return True

---

## Методы ScatterPlotExtractor

### extract

**Назначение:** Полное извлечение данных из scatter plot.

**Сигнатура:**

```python
def extract(self, image: np.ndarray) -> ScatterPlotData
```

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| image | np.ndarray | Изображение [H, W, C] |

**Возвращает:** ScatterPlotData

**Алгоритм:**

1. Вызвать detect_plot_region для определения границ
2. Вызвать detect_points для нахождения точек
3. Вызвать extract_point_colors для получения цветов
4. Вызвать normalize_coordinates для нормализации
5. Опционально: вызвать cluster_colors для кластеризации
6. Рассчитать detection_confidence
7. Создать и вернуть ScatterPlotData

---

### detect_points

**Назначение:** Детекция точек на scatter plot.

**Сигнатура:**

```python
def detect_points(self, image: np.ndarray, method: str = "threshold") -> np.ndarray
```

**Параметры:**

| Параметр | Тип | Описание | По умолчанию |
|----------|-----|----------|--------------|
| image | np.ndarray | Изображение [H, W, C] | - |
| method | str | Метод детекции | "threshold" |

**Возвращает:** Координаты точек [n_points, 2]

**Алгоритм (threshold):**

1. Конвертировать в grayscale
2. Применить пороговую бинаризацию (Otsu)
3. Найти контуры/connected components
4. Фильтровать по размеру (min_point_radius, max_point_radius)
5. Вычислить центроиды
6. Вернуть массив координат

**Алгоритм (contour):**

1. Применить Canny edge detection
2. Найти контуры
3. Фильтровать по округлости и размеру
4. Вычислить центроиды
5. Вернуть массив координат

**Алгоритм (hough):**

1. Применить Hough Circle Transform
2. Фильтровать по радиусу
3. Вернуть центры окружностей

---

### extract_point_colors

**Назначение:** Извлечение цветов точек.

**Сигнатура:**

```python
def extract_point_colors(self, image: np.ndarray, points: np.ndarray) -> np.ndarray
```

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| image | np.ndarray | Изображение [H, W, C] |
| points | np.ndarray | Координаты точек [n_points, 2] |

**Возвращает:** Цвета [n_points, 3] в RGB

**Алгоритм:**

1. Для каждой точки (x, y):
   - Извлечь пиксель image[y, x]
   - Или усреднить по окрестности (3x3)
2. Вернуть массив цветов

---

### detect_axes

**Назначение:** Детекция осей графика.

**Сигнатура:**

```python
def detect_axes(self, image: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]
```

**Возвращает:** (x_axis_coords, y_axis_coords) или (None, None)

**Алгоритм:**

1. Конвертировать в grayscale
2. Применить Hough Line Transform
3. Фильтровать линии по ориентации (горизонтальные, вертикальные)
4. Найти линии с наибольшей длиной
5. Вернуть координаты осей

---

### detect_plot_region

**Назначение:** Детекция области графика (без осей и подписей).

**Сигнатура:**

```python
def detect_plot_region(self, image: np.ndarray) -> tuple[int, int, int, int] | None
```

**Возвращает:** (x_min, y_min, x_max, y_max) или None

**Алгоритм:**

1. Если auto_detect_axes — найти оси
2. Определить область между осями
3. Исключить области с текстом (низкая дисперсия)
4. Вернуть границы или None

---

### normalize_coordinates

**Назначение:** Нормализация координат в [0, 1].

**Сигнатура:**

```python
def normalize_coordinates(
    self,
    points: np.ndarray,
    bounds: tuple[int, int, int, int],
) -> np.ndarray
```

**Возвращает:** Нормализованные координаты [n_points, 2]

**Алгоритм:**

1. Извлечь x_min, y_min, x_max, y_max из bounds
2. Для каждой точки:
   - x_norm = (x - x_min) / (x_max - x_min)
   - y_norm = (y - y_min) / (y_max - y_min)
3. Вернуть нормализованный массив

---

### cluster_colors

**Назначение:** Кластеризация цветов точек.

**Сигнатура:**

```python
def cluster_colors(self, colors: np.ndarray, n_clusters: int = 5) -> np.ndarray
```

**Возвращает:** Метки кластеров [n_points]

**Алгоритм:**

1. Использовать K-Means кластеризацию:
   ```python
   from sklearn.cluster import KMeans
   kmeans = KMeans(n_clusters=n_clusters)
   labels = kmeans.fit_predict(colors)
   ```
2. Вернуть метки

---

## Методы ImageAnalyzer

### analyze

**Назначение:** Полный анализ изображения.

**Сигнатура:**

```python
def analyze(self, image: np.ndarray) -> ImageAnalysisResult
```

**Возвращает:** ImageAnalysisResult

**Алгоритм:**

1. Вычислить гистограммы для R, G, B, gray
2. Найти доминантные цвета
3. Вычислить статистики (mean, std, brightness, contrast)
4. Опционально: детектировать регионы
5. Создать и вернуть ImageAnalysisResult

---

### compute_histogram

**Назначение:** Расчёт гистограммы канала.

**Сигнатура:**

```python
def compute_histogram(self, image: np.ndarray, channel: str = "gray") -> np.ndarray
```

**Параметры:**

| Параметр | Тип | Описание | По умолчанию |
|----------|-----|----------|--------------|
| image | np.ndarray | Изображение | - |
| channel | str | Канал: 'r', 'g', 'b', 'gray' | "gray" |

**Возвращает:** Гистограмма [256]

**Алгоритм:**

1. Если channel == 'gray' — конвертировать в grayscale
2. Иначе — извлечь канал (0=R, 1=G, 2=B)
3. Использовать np.histogram с bins=256, range=(0, 256)
4. Вернуть гистограмму

---

### find_dominant_colors

**Назначение:** Поиск доминантных цветов.

**Сигнатура:**

```python
def find_dominant_colors(
    self,
    image: np.ndarray,
    n_colors: int = 5,
) -> tuple[np.ndarray, np.ndarray]
```

**Возвращает:** (colors [n_colors, 3], percentages [n_colors])

**Алгоритм:**

1. Преобразовать изображение в список пикселей [H*W, 3]
2. Применить K-Means с n_clusters=n_colors
3. Получить центры кластеров (доминантные цвета)
4. Рассчитать процент пикселей в каждом кластере
5. Отсортировать по убыванию процента
6. Вернуть (colors, percentages)

---

### compute_statistics

**Назначение:** Расчёт базовых статистик.

**Сигнатура:**

```python
def compute_statistics(self, image: np.ndarray) -> dict[str, float]
```

**Возвращает:** Словарь статистик

**Алгоритм:**

1. Рассчитать mean для R, G, B
2. Рассчитать std для R, G, B
3. Конвертировать в grayscale
4. brightness = mean(gray)
5. contrast = std(gray)
6. Вернуть словарь

---

### detect_regions

**Назначение:** Детекция регионов интереса.

**Сигнатура:**

```python
def detect_regions(self, image: np.ndarray, method: str = "threshold") -> list[dict[str, Any]]
```

**Возвращает:** Список словарей с информацией о регионах

**Алгоритм:**

1. Бинаризация изображения
2. Поиск connected components
3. Для каждого компонента:
   - bounds = bounding box
   - area = количество пикселей
   - centroid = центр масс
4. Вернуть список регионов

---

### segment_by_color

**Назначение:** Сегментация по целевому цвету.

**Сигнатура:**

```python
def segment_by_color(
    self,
    image: np.ndarray,
    target_color: tuple[int, int, int],
    tolerance: int = 30,
) -> np.ndarray
```

**Возвращает:** Бинарная маска [H, W]

**Алгоритм:**

1. Вычислить расстояние каждого пикселя до target_color
2. Создать маску: distance <= tolerance
3. Вернуть маску

---

## Функции

### load_image

**Назначение:** Convenience функция для загрузки изображения.

**Сигнатура:**

```python
def load_image(file_path: str | Path, config: ImageConfig | None = None) -> ImageLoader
```

**Алгоритм:**

```python
return ImageLoader(config=config).load(file_path)
```

---

### extract_scatter_plot

**Назначение:** Convenience функция для извлечения данных из scatter plot.

**Сигнатура:**

```python
def extract_scatter_plot(
    file_path: str | Path,
    config: ImageConfig | None = None,
) -> ScatterPlotData
```

**Алгоритм:**

```python
loader = ImageLoader(config=config).load(file_path)
extractor = ScatterPlotExtractor(config=config)
return extractor.extract(loader.get_image())
```

---

### analyze_image

**Назначение:** Convenience функция для анализа изображения.

**Сигнатура:**

```python
def analyze_image(
    file_path: str | Path,
    config: ImageConfig | None = None,
) -> ImageAnalysisResult
```

**Алгоритм:**

```python
loader = ImageLoader(config=config).load(file_path)
analyzer = ImageAnalyzer(config=config)
return analyzer.analyze(loader.get_image())
```

---

## Примеры использования

```python
from src.data.image_loader import (
    ImageLoader,
    ScatterPlotExtractor,
    ImageAnalyzer,
    load_image,
    extract_scatter_plot,
)

# Способ 1: через классы
loader = ImageLoader()
loader.load("data/scatter_plot.png")

image = loader.get_image()
# [400, 400, 3] ndarray

metadata = loader.get_metadata()
# ImageMetadata(filename='scatter_plot.png', width=400, height=400, ...)

# Извлечение данных из scatter plot
extractor = ScatterPlotExtractor()
scatter_data = extractor.extract(image)
# ScatterPlotData(n_points=150, detection_confidence=0.92, ...)

df = scatter_data.to_dataframe()
# DataFrame с колонками: x, y, x_norm, y_norm, r, g, b

# Анализ изображения
analyzer = ImageAnalyzer()
analysis = analyzer.analyze(image)
# ImageAnalysisResult(brightness=128.5, contrast=45.2, ...)

# Способ 2: convenience функции
loader = load_image("data/scatter_plot.png")

scatter_data = extract_scatter_plot("data/scatter_plot.png")

# Пользовательская конфигурация
from src.data.image_loader import ImageConfig

config = ImageConfig(
    point_detection_method="contour",
    min_point_radius=5,
    dominant_colors_count=3,
)

scatter_data = extract_scatter_plot("data/scatter_plot.png", config=config)
```

---

## Зависимости

- Pillow>=10.0.0 — загрузка PNG/JPEG
- numpy — работа с массивами
- pandas — конвертация в DataFrame
- scikit-image>=0.21.0 — алгоритмы обработки изображений
- scikit-learn — K-Means кластеризация (опционально)
- opencv-python>=4.8.0 — продвинутая обработка (опционально)
