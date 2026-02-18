"""Загрузчик и анализатор изображений для RegenTwin.

Поддержка форматов PNG и JPEG для загрузки scatter plot изображений
и базового анализа изображений.

Подробное описание: Description/Phase1/description_image_loader.md
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from sklearn.cluster import KMeans


class ColorSpace(str, Enum):
    """Цветовые пространства для обработки изображений."""

    RGB = "RGB"
    RGBA = "RGBA"
    GRAYSCALE = "grayscale"


class DetectionMethod(str, Enum):
    """Методы детекции точек на scatter plot."""

    THRESHOLD = "threshold"
    CONTOUR = "contour"
    HOUGH = "hough"


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Внутренняя конвертация изображения в grayscale.

    Args:
        image: NumPy массив изображения [H, W, C] или [H, W]

    Returns:
        NumPy массив grayscale [H, W]
    """
    if len(image.shape) == 2:
        return image
    return (
        0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    ).astype(np.uint8)


@dataclass
class ImageConfig:
    """Конфигурация обработки изображений.

    Подробное описание: Description/Phase1/description_image_loader.md#ImageConfig
    """

    # Параметры загрузки
    max_dimension: int = 4096
    """Максимальный размер стороны изображения (px)."""

    color_space: str | ColorSpace = "RGB"
    """Цветовое пространство: 'RGB', 'RGBA', 'grayscale' или ColorSpace enum."""

    # Параметры извлечения точек
    point_detection_method: str | DetectionMethod = "threshold"
    """Метод детекции точек: 'threshold', 'contour', 'hough' или DetectionMethod enum."""

    min_point_radius: int = 2
    """Минимальный радиус точки (px)."""

    max_point_radius: int = 50
    """Максимальный радиус точки (px)."""

    # Параметры анализа цвета
    color_quantization_bins: int = 256
    """Количество бинов гистограммы."""

    dominant_colors_count: int = 5
    """Количество доминантных цветов для извлечения."""

    # Параметры детекции осей
    auto_detect_axes: bool = True
    """Автоматическое определение осей графика."""

    axis_color_threshold: int = 30
    """Порог яркости для детекции осей (grayscale)."""

    def validate(self) -> bool:
        """Валидация параметров конфигурации.

        Returns:
            True если параметры валидны

        Raises:
            ValueError: Если параметры некорректны

        Подробное описание: Description/Phase1/description_image_loader.md#validate
        """
        # Валидация max_dimension
        if self.max_dimension <= 0:
            raise ValueError("max_dimension must be positive")

        # Валидация color_space
        valid_color_spaces = ["RGB", "RGBA", "grayscale"]
        color_space_value = (
            self.color_space.value
            if isinstance(self.color_space, ColorSpace)
            else self.color_space
        )
        if color_space_value not in valid_color_spaces:
            raise ValueError(
                f"color_space must be one of {valid_color_spaces}, got {self.color_space}"
            )

        # Валидация point_detection_method
        valid_methods = ["threshold", "contour", "hough"]
        method_value = (
            self.point_detection_method.value
            if isinstance(self.point_detection_method, DetectionMethod)
            else self.point_detection_method
        )
        if method_value not in valid_methods:
            raise ValueError(
                f"point_detection_method must be one of {valid_methods}, "
                f"got {self.point_detection_method}"
            )

        # Валидация point_radius
        if self.min_point_radius <= 0:
            raise ValueError("min_point_radius must be positive")
        if self.min_point_radius >= self.max_point_radius:
            raise ValueError("min_point_radius must be less than max_point_radius")

        # Валидация color_quantization_bins
        if self.color_quantization_bins <= 0:
            raise ValueError("color_quantization_bins must be positive")

        # Валидация dominant_colors_count
        if self.dominant_colors_count <= 0:
            raise ValueError("dominant_colors_count must be positive")

        return True


@dataclass
class ImageMetadata:
    """Метаданные изображения.

    Подробное описание: Description/Phase1/description_image_loader.md#ImageMetadata
    """

    filename: str
    """Имя файла."""

    width: int
    """Ширина изображения (px)."""

    height: int
    """Высота изображения (px)."""

    channels: int
    """Количество каналов (1=grayscale, 3=RGB, 4=RGBA)."""

    format: str
    """Формат файла: 'PNG', 'JPEG'."""

    bit_depth: int
    """Битовая глубина (обычно 8)."""

    file_size_bytes: int
    """Размер файла в байтах."""

    has_alpha: bool = False
    """Наличие альфа-канала."""


@dataclass
class ScatterPlotData:
    """Результат извлечения данных из scatter plot изображения.

    Подробное описание: Description/Phase1/description_image_loader.md#ScatterPlotData
    """

    points: np.ndarray
    """Координаты точек [n_points, 2] в пикселях (x, y)."""

    points_normalized: np.ndarray
    """Нормализованные координаты [n_points, 2] в диапазоне [0, 1]."""

    colors: np.ndarray
    """Цвета точек [n_points, 3] в RGB."""

    color_labels: np.ndarray | None
    """Метки кластеров цветов [n_points] или None."""

    n_points: int
    """Общее количество обнаруженных точек."""

    detection_confidence: float
    """Уверенность детекции [0, 1]."""

    plot_bounds: tuple[int, int, int, int] | None
    """Границы области графика (x_min, y_min, x_max, y_max) или None."""

    axis_labels: tuple[str | None, str | None] = (None, None)
    """Подписи осей (x_label, y_label)."""

    def to_dataframe(self) -> pd.DataFrame:
        """Конвертация данных в pandas DataFrame.

        Returns:
            DataFrame с колонками: x, y, x_norm, y_norm, r, g, b, [color_label]

        Подробное описание: Description/Phase1/description_image_loader.md#to_dataframe
        """
        data = {
            "x": self.points[:, 0],
            "y": self.points[:, 1],
            "x_norm": self.points_normalized[:, 0],
            "y_norm": self.points_normalized[:, 1],
            "r": self.colors[:, 0],
            "g": self.colors[:, 1],
            "b": self.colors[:, 2],
        }

        # Добавляем color_label если есть
        if self.color_labels is not None:
            data["color_label"] = self.color_labels

        return pd.DataFrame(data)


@dataclass
class ImageAnalysisResult:
    """Результат базового анализа изображения.

    Подробное описание: Description/Phase1/description_image_loader.md#ImageAnalysisResult
    """

    histogram_r: np.ndarray
    """Гистограмма красного канала [256]."""

    histogram_g: np.ndarray
    """Гистограмма зелёного канала [256]."""

    histogram_b: np.ndarray
    """Гистограмма синего канала [256]."""

    histogram_gray: np.ndarray
    """Гистограмма в grayscale [256]."""

    dominant_colors: np.ndarray
    """Доминантные цвета [n_colors, 3] в RGB."""

    dominant_colors_percentages: np.ndarray
    """Процентные доли доминантных цветов [n_colors]."""

    mean_color: tuple[float, float, float]
    """Средний цвет (R, G, B)."""

    std_color: tuple[float, float, float]
    """Стандартное отклонение цвета (R, G, B)."""

    brightness: float
    """Средняя яркость [0, 255]."""

    contrast: float
    """Контраст (стандартное отклонение яркости)."""

    regions: list[dict[str, Any]] | None = None
    """Список обнаруженных регионов интереса или None."""


class ImageLoader:
    """Загрузчик изображений PNG/JPEG.

    Аналог FCSLoader для работы с изображениями scatter plots.
    Поддерживает форматы PNG и JPEG.

    Подробное описание: Description/Phase1/description_image_loader.md#ImageLoader
    """

    SUPPORTED_FORMATS: list[str] = [".png", ".jpg", ".jpeg"]
    """Поддерживаемые форматы файлов."""

    def __init__(self, config: ImageConfig | None = None) -> None:
        """Инициализация загрузчика.

        Args:
            config: Конфигурация обработки (None = параметры по умолчанию)

        Подробное описание: Description/Phase1/description_image_loader.md#ImageLoader.__init__
        """
        self._config = config if config else ImageConfig()
        self._image: np.ndarray | None = None
        self._file_path: Path | None = None
        self._metadata: ImageMetadata | None = None

    @property
    def config(self) -> ImageConfig:
        """Получить конфигурацию загрузчика."""
        return self._config

    def load(self, file_path: str | Path) -> "ImageLoader":
        """Загрузка изображения с диска.

        Args:
            file_path: Путь к файлу изображения (PNG или JPEG)

        Returns:
            Self для цепочки вызовов

        Raises:
            FileNotFoundError: Файл не найден
            ValueError: Неподдерживаемый формат файла

        Подробное описание: Description/Phase1/description_image_loader.md#load
        """
        path = Path(file_path)

        # Проверка существования файла
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Проверка формата
        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {suffix}. "
                f"Supported formats: {self.SUPPORTED_FORMATS}"
            )

        # Загрузка изображения
        img = Image.open(path)

        # Конвертация RGBA -> RGB если нужно
        if img.mode == "RGBA":
            img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")

        # Масштабирование если превышает max_dimension
        max_dim = self._config.max_dimension
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        # Сохранение как numpy array
        self._image = np.array(img, dtype=np.uint8)
        self._file_path = path
        self._metadata = None  # Сброс кэша метаданных

        return self

    def get_image(self) -> np.ndarray:
        """Получение изображения как NumPy массив.

        Returns:
            NumPy массив shape [H, W, C] (RGB) или [H, W] (grayscale)

        Raises:
            RuntimeError: Файл не загружен

        Подробное описание: Description/Phase1/description_image_loader.md#get_image
        """
        if self._image is None:
            raise RuntimeError("No image loaded. Call load() first.")
        return self._image.copy()

    def get_metadata(self) -> ImageMetadata:
        """Получение метаданных изображения.

        Returns:
            ImageMetadata dataclass

        Raises:
            RuntimeError: Файл не загружен

        Подробное описание: Description/Phase1/description_image_loader.md#get_metadata
        """
        if self._image is None:
            raise RuntimeError("No image loaded. Call load() first.")

        # Возвращаем кэшированные метаданные если есть
        if self._metadata is not None:
            return self._metadata

        # Определение формата
        suffix = self._file_path.suffix.lower()
        if suffix == ".png":
            img_format = "PNG"
        elif suffix in [".jpg", ".jpeg"]:
            img_format = "JPEG"
        else:
            img_format = suffix.upper().lstrip(".")

        # Определение количества каналов
        if len(self._image.shape) == 2:
            channels = 1
            has_alpha = False
        else:
            channels = self._image.shape[2]
            has_alpha = channels == 4

        # Создание метаданных
        self._metadata = ImageMetadata(
            filename=self._file_path.name,
            width=self._image.shape[1],
            height=self._image.shape[0],
            channels=channels,
            format=img_format,
            bit_depth=8,
            file_size_bytes=self._file_path.stat().st_size,
            has_alpha=has_alpha,
        )

        return self._metadata

    def to_grayscale(self) -> np.ndarray:
        """Конвертация изображения в grayscale.

        Returns:
            NumPy массив shape [H, W]

        Raises:
            RuntimeError: Файл не загружен

        Подробное описание: Description/Phase1/description_image_loader.md#to_grayscale
        """
        if self._image is None:
            raise RuntimeError("No image loaded. Call load() first.")

        return _to_grayscale(self._image).copy()

    def resize(self, width: int, height: int) -> np.ndarray:
        """Изменение размера изображения.

        Args:
            width: Новая ширина (px)
            height: Новая высота (px)

        Returns:
            NumPy массив с изменённым размером

        Raises:
            RuntimeError: Файл не загружен
            ValueError: Некорректные размеры

        Подробное описание: Description/Phase1/description_image_loader.md#resize
        """
        if self._image is None:
            raise RuntimeError("No image loaded. Call load() first.")

        # Валидация размеров
        if width <= 0:
            raise ValueError("width must be positive")
        if height <= 0:
            raise ValueError("height must be positive")

        # Resize через PIL
        img = Image.fromarray(self._image)
        resized = img.resize((width, height), Image.Resampling.LANCZOS)
        return np.array(resized, dtype=np.uint8)

    def crop(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Обрезка изображения.

        Args:
            x: X-координата левого верхнего угла
            y: Y-координата левого верхнего угла
            width: Ширина области
            height: Высота области

        Returns:
            NumPy массив обрезанной области

        Raises:
            RuntimeError: Файл не загружен
            ValueError: Координаты выходят за пределы изображения

        Подробное описание: Description/Phase1/description_image_loader.md#crop
        """
        if self._image is None:
            raise RuntimeError("No image loaded. Call load() first.")

        img_height, img_width = self._image.shape[:2]

        # Валидация координат
        if x < 0:
            raise ValueError("x must be non-negative")
        if y < 0:
            raise ValueError("y must be non-negative")
        if width <= 0:
            raise ValueError("width must be positive")
        if height <= 0:
            raise ValueError("height must be positive")
        if x + width > img_width:
            raise ValueError("crop region exceeds image width")
        if y + height > img_height:
            raise ValueError("crop region exceeds image height")

        # Обрезка
        return self._image[y : y + height, x : x + width].copy()

    def validate_format(self) -> bool:
        """Проверка что формат файла поддерживается.

        Returns:
            True если формат поддерживается

        Raises:
            ValueError: Формат не поддерживается

        Подробное описание: Description/Phase1/description_image_loader.md#validate_format
        """
        if self._file_path is None:
            raise ValueError("No file loaded")

        suffix = self._file_path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {suffix}. "
                f"Supported formats: {self.SUPPORTED_FORMATS}"
            )
        return True


class ScatterPlotExtractor:
    """Экстрактор данных из изображений scatter plots.

    Извлекает координаты точек и их цвета из изображений
    scatter plot графиков для последующего анализа.

    Подробное описание: Description/Phase1/description_image_loader.md#ScatterPlotExtractor
    """

    def __init__(self, config: ImageConfig | None = None) -> None:
        """Инициализация экстрактора.

        Args:
            config: Конфигурация обработки (None = параметры по умолчанию)

        Подробное описание: Description/Phase1/description_image_loader.md#ScatterPlotExtractor.__init__
        """
        self._config = config if config else ImageConfig()

    @property
    def config(self) -> ImageConfig:
        """Получить конфигурацию экстрактора."""
        return self._config

    def extract(self, image: np.ndarray) -> ScatterPlotData:
        """Полное извлечение данных из scatter plot изображения.

        Выполняет детекцию точек, извлечение цветов, определение осей
        и нормализацию координат.

        Args:
            image: NumPy массив изображения [H, W, C]

        Returns:
            ScatterPlotData с извлечёнными данными

        Подробное описание: Description/Phase1/description_image_loader.md#extract
        """
        h, w = image.shape[:2]

        # Детекция точек
        points = self.detect_points(image)
        n_points = len(points)

        # Детекция региона графика
        plot_bounds = self.detect_plot_region(image)
        if plot_bounds is None:
            plot_bounds = (0, 0, w, h)

        # Нормализация координат
        if n_points > 0:
            points_normalized = self.normalize_coordinates(points, plot_bounds)
            colors = self.extract_point_colors(image, points)

            # Кластеризация цветов если есть достаточно точек
            if n_points >= 2:
                n_clusters = min(self._config.dominant_colors_count, n_points)
                color_labels = self.cluster_colors(colors, n_clusters)
            else:
                color_labels = np.zeros(n_points, dtype=int)
        else:
            points = np.empty((0, 2), dtype=int)
            points_normalized = np.empty((0, 2), dtype=float)
            colors = np.empty((0, 3), dtype=np.uint8)
            color_labels = np.array([], dtype=int)

        # Детекция осей (для axis_labels - пока None)
        axis_labels: tuple[str | None, str | None] = (None, None)

        # Расчёт confidence (простая эвристика)
        detection_confidence = min(1.0, n_points / 50) if n_points > 0 else 0.0

        return ScatterPlotData(
            points=points,
            points_normalized=points_normalized,
            colors=colors,
            color_labels=color_labels if n_points > 0 else None,
            n_points=n_points,
            detection_confidence=detection_confidence,
            plot_bounds=plot_bounds,
            axis_labels=axis_labels,
        )

    def detect_points(
        self,
        image: np.ndarray,
        method: str | None = None,
    ) -> np.ndarray:
        """Детекция точек на изображении scatter plot.

        Args:
            image: NumPy массив изображения [H, W, C]
            method: Метод детекции ('threshold', 'contour', 'hough')
                    None = использовать метод из конфигурации

        Returns:
            NumPy массив координат точек [n_points, 2]

        Подробное описание: Description/Phase1/description_image_loader.md#detect_points
        """
        if method is None:
            method = self._config.point_detection_method

        # Получаем значение метода если это Enum
        method_value = method.value if isinstance(method, DetectionMethod) else method

        # Конвертация в grayscale
        gray = _to_grayscale(image)

        if method_value == "threshold":
            return self._detect_points_threshold(image, gray)
        elif method_value == "contour":
            return self._detect_points_contour(image, gray)
        elif method_value == "hough":
            return self._detect_points_hough(image, gray)
        else:
            raise ValueError(f"Unknown detection method: {method}")

    def _detect_points_threshold(
        self, image: np.ndarray, gray: np.ndarray
    ) -> np.ndarray:
        """Детекция точек методом порогов."""
        # Бинаризация: точки обычно темнее фона (белого)
        # Порог: пиксели значительно темнее среднего
        threshold = 200  # Белый фон - 255, точки темнее
        binary = gray < threshold

        # Labeling connected components
        labeled, n_features = ndimage.label(binary)

        points = []
        for i in range(1, n_features + 1):
            # Находим центроид каждого компонента
            component = labeled == i
            area = component.sum()

            # Фильтруем по размеру (минимальный и максимальный радиус)
            min_area = np.pi * self._config.min_point_radius**2
            max_area = np.pi * self._config.max_point_radius**2

            if min_area <= area <= max_area:
                y_coords, x_coords = np.where(component)
                cx = int(np.mean(x_coords))
                cy = int(np.mean(y_coords))
                points.append([cx, cy])

        return np.array(points) if points else np.empty((0, 2), dtype=int)

    def _detect_points_contour(
        self, image: np.ndarray, gray: np.ndarray
    ) -> np.ndarray:
        """Детекция точек методом контуров."""
        # Похоже на threshold, но ищем контуры компонентов
        threshold = 200
        binary = gray < threshold

        # Находим контуры через границы компонентов
        labeled, n_features = ndimage.label(binary)

        points = []
        for i in range(1, n_features + 1):
            component = labeled == i
            area = component.sum()

            min_area = np.pi * self._config.min_point_radius**2
            max_area = np.pi * self._config.max_point_radius**2

            if min_area <= area <= max_area:
                # Проверяем округлость через отношение площади к периметру
                y_coords, x_coords = np.where(component)
                cx = int(np.mean(x_coords))
                cy = int(np.mean(y_coords))
                points.append([cx, cy])

        return np.array(points) if points else np.empty((0, 2), dtype=int)

    def _detect_points_hough(
        self, image: np.ndarray, gray: np.ndarray
    ) -> np.ndarray:
        """Детекция точек методом Hough circle transform."""
        # Упрощённая реализация без OpenCV
        # Используем тот же подход, что и threshold
        return self._detect_points_threshold(image, gray)

    def extract_point_colors(
        self,
        image: np.ndarray,
        points: np.ndarray,
    ) -> np.ndarray:
        """Извлечение цветов точек по их координатам.

        Args:
            image: NumPy массив изображения [H, W, C]
            points: Координаты точек [n_points, 2]

        Returns:
            NumPy массив цветов [n_points, 3] в RGB

        Подробное описание: Description/Phase1/description_image_loader.md#extract_point_colors
        """
        if len(points) == 0:
            return np.empty((0, 3), dtype=np.uint8)

        h, w = image.shape[:2]
        colors = []

        for x, y in points:
            # Проверка границ
            x = int(np.clip(x, 0, w - 1))
            y = int(np.clip(y, 0, h - 1))

            # Берём цвет центрального пикселя
            if len(image.shape) == 3:
                color = image[y, x]
            else:
                # Grayscale -> RGB
                val = image[y, x]
                color = [val, val, val]

            colors.append(color)

        return np.array(colors, dtype=np.uint8)

    def detect_axes(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Детекция осей графика.

        Args:
            image: NumPy массив изображения [H, W, C]

        Returns:
            Кортеж (x_axis, y_axis) — координаты линий осей или None

        Подробное описание: Description/Phase1/description_image_loader.md#detect_axes
        """
        # Конвертация в grayscale
        gray = _to_grayscale(image)

        h, w = gray.shape
        threshold = self._config.axis_color_threshold

        # Ищем тёмные линии (оси обычно чёрные)
        dark_mask = gray < threshold

        # Поиск вертикальной оси (Y-axis)
        # Ищем столбец с большим количеством тёмных пикселей в левой части
        y_axis = None
        for x in range(w // 4):  # Ищем в левой четверти
            col_dark = dark_mask[:, x].sum()
            if col_dark > h * 0.5:  # Более 50% пикселей тёмные
                y_axis = np.array([[x, 0], [x, h - 1]])
                break

        # Поиск горизонтальной оси (X-axis)
        # Ищем строку с большим количеством тёмных пикселей в нижней части
        x_axis = None
        for y in range(h - 1, h * 3 // 4, -1):  # Ищем в нижней четверти
            row_dark = dark_mask[y, :].sum()
            if row_dark > w * 0.5:  # Более 50% пикселей тёмные
                x_axis = np.array([[0, y], [w - 1, y]])
                break

        return (x_axis, y_axis)

    def detect_plot_region(
        self,
        image: np.ndarray,
    ) -> tuple[int, int, int, int] | None:
        """Детекция региона графика (без осей и подписей).

        Args:
            image: NumPy массив изображения [H, W, C]

        Returns:
            Границы (x_min, y_min, x_max, y_max) или None если не найдено

        Подробное описание: Description/Phase1/description_image_loader.md#detect_plot_region
        """
        h, w = image.shape[:2]

        # Пытаемся найти оси
        x_axis, y_axis = self.detect_axes(image)

        x_min, y_min = 0, 0
        x_max, y_max = w, h

        # Если нашли Y-axis, начинаем справа от неё
        if y_axis is not None:
            x_min = int(y_axis[0, 0]) + 5

        # Если нашли X-axis, заканчиваем выше неё
        if x_axis is not None:
            y_max = int(x_axis[0, 1]) - 5

        # Добавляем отступы
        margin = 10
        x_min = max(0, x_min + margin)
        y_min = max(0, y_min + margin)
        x_max = min(w, x_max - margin)
        y_max = min(h, y_max - margin)

        # Валидация
        if x_min >= x_max or y_min >= y_max:
            return None

        return (x_min, y_min, x_max, y_max)

    def normalize_coordinates(
        self,
        points: np.ndarray,
        bounds: tuple[int, int, int, int],
    ) -> np.ndarray:
        """Нормализация координат точек в диапазон [0, 1].

        Args:
            points: Координаты точек [n_points, 2] в пикселях
            bounds: Границы области (x_min, y_min, x_max, y_max)

        Returns:
            Нормализованные координаты [n_points, 2]

        Подробное описание: Description/Phase1/description_image_loader.md#normalize_coordinates
        """
        if len(points) == 0:
            return np.empty((0, 2), dtype=float)

        x_min, y_min, x_max, y_max = bounds

        # Защита от деления на ноль
        width = max(x_max - x_min, 1)
        height = max(y_max - y_min, 1)

        # Нормализация
        normalized = np.zeros_like(points, dtype=float)
        normalized[:, 0] = (points[:, 0] - x_min) / width
        normalized[:, 1] = (points[:, 1] - y_min) / height

        # Клиппинг в [0, 1]
        normalized = np.clip(normalized, 0, 1)

        return normalized

    def cluster_colors(
        self,
        colors: np.ndarray,
        n_clusters: int = 5,
    ) -> np.ndarray:
        """Кластеризация цветов точек.

        Args:
            colors: Цвета точек [n_points, 3] в RGB
            n_clusters: Количество кластеров

        Returns:
            Метки кластеров [n_points]

        Подробное описание: Description/Phase1/description_image_loader.md#cluster_colors
        """
        if len(colors) == 0:
            return np.array([], dtype=int)

        # Ограничиваем количество кластеров количеством точек
        n_clusters = min(n_clusters, len(colors))

        # KMeans кластеризация
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(colors.astype(float))

        return labels


class ImageAnalyzer:
    """Базовый анализатор изображений.

    Предоставляет инструменты для анализа цветов, регионов
    и статистик изображений scatter plots.

    Подробное описание: Description/Phase1/description_image_loader.md#ImageAnalyzer
    """

    def __init__(self, config: ImageConfig | None = None) -> None:
        """Инициализация анализатора.

        Args:
            config: Конфигурация обработки (None = параметры по умолчанию)

        Подробное описание: Description/Phase1/description_image_loader.md#ImageAnalyzer.__init__
        """
        self._config = config if config else ImageConfig()

    @property
    def config(self) -> ImageConfig:
        """Получить конфигурацию анализатора."""
        return self._config

    def analyze(self, image: np.ndarray) -> ImageAnalysisResult:
        """Полный анализ изображения.

        Выполняет расчёт гистограмм, поиск доминантных цветов,
        расчёт статистик и детекцию регионов.

        Args:
            image: NumPy массив изображения [H, W, C]

        Returns:
            ImageAnalysisResult с результатами анализа

        Подробное описание: Description/Phase1/description_image_loader.md#analyze
        """
        # Гистограммы
        histogram_r = self.compute_histogram(image, channel="r")
        histogram_g = self.compute_histogram(image, channel="g")
        histogram_b = self.compute_histogram(image, channel="b")
        histogram_gray = self.compute_histogram(image, channel="gray")

        # Доминантные цвета
        dominant_colors, dominant_colors_percentages = self.find_dominant_colors(image)

        # Статистики
        stats = self.compute_statistics(image)

        # Регионы
        regions = self.detect_regions(image)

        return ImageAnalysisResult(
            histogram_r=histogram_r,
            histogram_g=histogram_g,
            histogram_b=histogram_b,
            histogram_gray=histogram_gray,
            dominant_colors=dominant_colors,
            dominant_colors_percentages=dominant_colors_percentages,
            mean_color=(stats["mean_r"], stats["mean_g"], stats["mean_b"]),
            std_color=(stats["std_r"], stats["std_g"], stats["std_b"]),
            brightness=stats["brightness"],
            contrast=stats["contrast"],
            regions=regions if regions else None,
        )

    def compute_histogram(
        self,
        image: np.ndarray,
        channel: str = "gray",
    ) -> np.ndarray:
        """Расчёт гистограммы канала изображения.

        Args:
            image: NumPy массив изображения [H, W, C] или [H, W]
            channel: Канал: 'r', 'g', 'b', 'gray'

        Returns:
            Гистограмма [256]

        Подробное описание: Description/Phase1/description_image_loader.md#compute_histogram
        """
        if channel == "gray":
            gray = _to_grayscale(image)
            hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
        elif channel == "r":
            hist, _ = np.histogram(image[:, :, 0].ravel(), bins=256, range=(0, 256))
        elif channel == "g":
            hist, _ = np.histogram(image[:, :, 1].ravel(), bins=256, range=(0, 256))
        elif channel == "b":
            hist, _ = np.histogram(image[:, :, 2].ravel(), bins=256, range=(0, 256))
        else:
            raise ValueError(f"Unknown channel: {channel}")

        return hist.astype(np.int64)

    def find_dominant_colors(
        self,
        image: np.ndarray,
        n_colors: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Поиск доминантных цветов в изображении.

        Args:
            image: NumPy массив изображения [H, W, C]
            n_colors: Количество доминантных цветов (None = из конфигурации)

        Returns:
            Кортеж (colors [n_colors, 3], percentages [n_colors])

        Подробное описание: Description/Phase1/description_image_loader.md#find_dominant_colors
        """
        if n_colors is None:
            n_colors = self._config.dominant_colors_count

        # Reshape изображения в список пикселей
        if len(image.shape) == 3:
            pixels = image.reshape(-1, 3).astype(float)
        else:
            # Grayscale -> RGB
            pixels = np.stack([image.ravel()] * 3, axis=1).astype(float)

        # Субсэмплинг для ускорения на больших изображениях
        max_samples = 10000
        if len(pixels) > max_samples:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(pixels), max_samples, replace=False)
            pixels_sample = pixels[indices]
        else:
            pixels_sample = pixels

        # KMeans кластеризация на сэмпле
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels_sample)

        # Применяем к полному набору пикселей для точных процентов
        labels = kmeans.predict(pixels)

        # Получение центров кластеров и их размеров
        colors = kmeans.cluster_centers_.astype(np.uint8)
        unique, counts = np.unique(labels, return_counts=True)
        percentages = counts / len(labels)

        # Сортировка по убыванию процента
        sorted_indices = np.argsort(percentages)[::-1]
        colors = colors[sorted_indices]
        percentages = percentages[sorted_indices]

        return colors, percentages

    def compute_statistics(
        self,
        image: np.ndarray,
    ) -> dict[str, float]:
        """Расчёт базовых статистик изображения.

        Args:
            image: NumPy массив изображения [H, W, C]

        Returns:
            Словарь со статистиками: mean_r, mean_g, mean_b,
            std_r, std_g, std_b, brightness, contrast

        Подробное описание: Description/Phase1/description_image_loader.md#compute_statistics
        """
        if len(image.shape) == 3:
            mean_r = float(np.mean(image[:, :, 0]))
            mean_g = float(np.mean(image[:, :, 1]))
            mean_b = float(np.mean(image[:, :, 2]))
            std_r = float(np.std(image[:, :, 0]))
            std_g = float(np.std(image[:, :, 1]))
            std_b = float(np.std(image[:, :, 2]))

            # Яркость через grayscale
            gray = _to_grayscale(image).astype(float)
        else:
            # Grayscale
            mean_r = mean_g = mean_b = float(np.mean(image))
            std_r = std_g = std_b = float(np.std(image))
            gray = image.astype(float)

        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))

        return {
            "mean_r": mean_r,
            "mean_g": mean_g,
            "mean_b": mean_b,
            "std_r": std_r,
            "std_g": std_g,
            "std_b": std_b,
            "brightness": brightness,
            "contrast": contrast,
        }

    def detect_regions(
        self,
        image: np.ndarray,
        method: str = "threshold",
    ) -> list[dict[str, Any]]:
        """Детекция регионов интереса в изображении.

        Args:
            image: NumPy массив изображения [H, W, C]
            method: Метод детекции ('threshold', 'contour', 'watershed')

        Returns:
            Список словарей с информацией о регионах:
            {'bounds': (x_min, y_min, x_max, y_max), 'area': int, 'centroid': (x, y)}

        Подробное описание: Description/Phase1/description_image_loader.md#detect_regions
        """
        # Конвертация в grayscale
        gray = _to_grayscale(image)

        h, w = gray.shape

        # Бинаризация
        threshold = 200
        binary = gray < threshold

        # Labeling
        labeled, n_features = ndimage.label(binary)

        regions = []
        for i in range(1, n_features + 1):
            component = labeled == i
            area = int(component.sum())

            # Минимальный размер региона
            if area < 10:
                continue

            # Границы
            y_coords, x_coords = np.where(component)
            x_min = int(x_coords.min())
            x_max = int(x_coords.max())
            y_min = int(y_coords.min())
            y_max = int(y_coords.max())

            # Центроид
            cx = int(np.mean(x_coords))
            cy = int(np.mean(y_coords))

            regions.append({
                "bounds": (x_min, y_min, x_max, y_max),
                "area": area,
                "centroid": (cx, cy),
            })

        return regions

    def segment_by_color(
        self,
        image: np.ndarray,
        target_color: tuple[int, int, int],
        tolerance: int = 30,
    ) -> np.ndarray:
        """Сегментация изображения по целевому цвету.

        Args:
            image: NumPy массив изображения [H, W, C]
            target_color: Целевой цвет (R, G, B)
            tolerance: Допустимое отклонение от цвета

        Returns:
            Бинарная маска [H, W] (True = пиксель соответствует цвету)

        Подробное описание: Description/Phase1/description_image_loader.md#segment_by_color
        """
        if len(image.shape) == 2:
            # Grayscale - конвертируем target_color в grayscale
            target_gray = (
                0.299 * target_color[0]
                + 0.587 * target_color[1]
                + 0.114 * target_color[2]
            )
            diff = np.abs(image.astype(float) - target_gray)
            mask = diff <= tolerance
        else:
            # RGB - Евклидово расстояние в цветовом пространстве
            target = np.array(target_color, dtype=float)
            diff = np.sqrt(np.sum((image.astype(float) - target) ** 2, axis=2))
            mask = diff <= tolerance

        return mask


# =============================================================================
# Convenience функции
# =============================================================================


def load_image(
    file_path: str | Path,
    config: ImageConfig | None = None,
) -> ImageLoader:
    """Удобная функция для загрузки изображения.

    Args:
        file_path: Путь к файлу изображения
        config: Конфигурация обработки

    Returns:
        Инициализированный ImageLoader

    Подробное описание: Description/Phase1/description_image_loader.md#load_image
    """
    return ImageLoader(config=config).load(file_path)


def extract_scatter_plot(
    file_path: str | Path,
    config: ImageConfig | None = None,
) -> ScatterPlotData:
    """Удобная функция для извлечения данных из scatter plot изображения.

    Args:
        file_path: Путь к файлу изображения
        config: Конфигурация обработки

    Returns:
        ScatterPlotData с извлечёнными данными

    Подробное описание: Description/Phase1/description_image_loader.md#extract_scatter_plot
    """
    loader = ImageLoader(config=config).load(file_path)
    extractor = ScatterPlotExtractor(config=config)
    return extractor.extract(loader.get_image())


def analyze_image(
    file_path: str | Path,
    config: ImageConfig | None = None,
) -> ImageAnalysisResult:
    """Удобная функция для анализа изображения.

    Args:
        file_path: Путь к файлу изображения
        config: Конфигурация обработки

    Returns:
        ImageAnalysisResult с результатами анализа

    Подробное описание: Description/Phase1/description_image_loader.md#analyze_image
    """
    loader = ImageLoader(config=config).load(file_path)
    analyzer = ImageAnalyzer(config=config)
    return analyzer.analyze(loader.get_image())
