"""Тесты для модуля image_loader."""

import numpy as np
import pytest

from src.data.image_loader import (
    ImageAnalyzer,
    ImageConfig,
    ImageLoader,
    ScatterPlotExtractor,
    analyze_image,
    extract_scatter_plot,
    load_image,
)


class TestImageConfig:
    """Тесты для ImageConfig dataclass."""

    def test_default_config(self, default_image_config):
        """Проверка параметров по умолчанию."""
        config = default_image_config

        assert config.max_dimension == 4096
        assert config.color_space == "RGB"
        assert config.point_detection_method == "threshold"
        assert config.min_point_radius == 2
        assert config.max_point_radius == 50
        assert config.dominant_colors_count == 5
        assert config.auto_detect_axes is True

    def test_custom_config(self, custom_image_config):
        """Проверка кастомной конфигурации."""
        config = custom_image_config

        assert config.max_dimension == 1024
        assert config.point_detection_method == "contour"
        assert config.min_point_radius == 5
        assert config.max_point_radius == 30
        assert config.dominant_colors_count == 3
        assert config.auto_detect_axes is False

    def test_validate_returns_true_for_valid_config(self, default_image_config):
        """Проверка что validate() возвращает True для валидной конфигурации."""
        assert default_image_config.validate() is True


class TestImageMetadata:
    """Тесты для ImageMetadata dataclass."""

    def test_metadata_attributes(self, mock_image_metadata):
        """Проверка атрибутов метаданных."""
        meta = mock_image_metadata

        assert meta.filename == "test_scatter.png"
        assert meta.width == 400
        assert meta.height == 400
        assert meta.channels == 3
        assert meta.format == "PNG"
        assert meta.bit_depth == 8
        assert meta.file_size_bytes == 50000
        assert meta.has_alpha is False


class TestScatterPlotData:
    """Тесты для ScatterPlotData dataclass."""

    def test_scatter_data_attributes(self, mock_scatter_plot_data):
        """Проверка атрибутов ScatterPlotData."""
        data = mock_scatter_plot_data

        assert data.n_points == 50
        assert data.points.shape == (50, 2)
        assert data.points_normalized.shape == (50, 2)
        assert data.colors.shape == (50, 3)
        assert data.color_labels.shape == (50,)
        assert 0 <= data.detection_confidence <= 1
        assert data.plot_bounds == (50, 50, 350, 350)
        assert data.axis_labels == ("X-axis", "Y-axis")

    def test_to_dataframe_returns_dataframe(self, mock_scatter_plot_data):
        """Проверка что to_dataframe() возвращает DataFrame."""
        df = mock_scatter_plot_data.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == mock_scatter_plot_data.n_points


class TestImageAnalysisResult:
    """Тесты для ImageAnalysisResult dataclass."""

    def test_analysis_result_attributes(self, mock_image_analysis_result):
        """Проверка атрибутов ImageAnalysisResult."""
        result = mock_image_analysis_result

        assert result.histogram_r.shape == (256,)
        assert result.histogram_g.shape == (256,)
        assert result.histogram_b.shape == (256,)
        assert result.histogram_gray.shape == (256,)
        assert result.dominant_colors.shape == (3, 3)
        assert result.dominant_colors_percentages.shape == (3,)
        assert len(result.mean_color) == 3
        assert len(result.std_color) == 3
        assert isinstance(result.brightness, float)
        assert isinstance(result.contrast, float)


class TestImageLoader:
    """Тесты для ImageLoader класса."""

    def test_init_default_config(self):
        """Проверка инициализации с конфигурацией по умолчанию."""
        loader = ImageLoader()
        assert isinstance(loader.config, ImageConfig)

    def test_init_custom_config(self, custom_image_config):
        """Проверка инициализации с кастомной конфигурацией."""
        loader = ImageLoader(config=custom_image_config)
        assert loader.config.max_dimension == 1024
        assert loader.config.point_detection_method == "contour"

    def test_load_returns_self(self, sample_image_path):
        """Проверка что load() возвращает self для chaining."""
        loader = ImageLoader()
        result = loader.load(sample_image_path)
        assert result is loader

    def test_get_image_raises_runtime_error_when_not_loaded(self):
        """Проверка что get_image() выбрасывает RuntimeError без загрузки."""
        loader = ImageLoader()
        with pytest.raises(RuntimeError, match="No image loaded"):
            loader.get_image()

    def test_get_metadata_raises_runtime_error_when_not_loaded(self):
        """Проверка что get_metadata() выбрасывает RuntimeError без загрузки."""
        loader = ImageLoader()
        with pytest.raises(RuntimeError, match="No image loaded"):
            loader.get_metadata()

    def test_to_grayscale_raises_runtime_error_when_not_loaded(self):
        """Проверка что to_grayscale() выбрасывает RuntimeError без загрузки."""
        loader = ImageLoader()
        with pytest.raises(RuntimeError, match="No image loaded"):
            loader.to_grayscale()

    def test_resize_raises_runtime_error_when_not_loaded(self):
        """Проверка что resize() выбрасывает RuntimeError без загрузки."""
        loader = ImageLoader()
        with pytest.raises(RuntimeError, match="No image loaded"):
            loader.resize(100, 100)

    def test_crop_raises_runtime_error_when_not_loaded(self):
        """Проверка что crop() выбрасывает RuntimeError без загрузки."""
        loader = ImageLoader()
        with pytest.raises(RuntimeError, match="No image loaded"):
            loader.crop(0, 0, 50, 50)

    def test_validate_format_raises_error_when_no_file_loaded(self):
        """Проверка что validate_format() выбрасывает ValueError без загруженного файла."""
        loader = ImageLoader()
        with pytest.raises(ValueError, match="No file loaded"):
            loader.validate_format()

    def test_supported_formats(self):
        """Проверка списка поддерживаемых форматов."""
        assert ".png" in ImageLoader.SUPPORTED_FORMATS
        assert ".jpg" in ImageLoader.SUPPORTED_FORMATS
        assert ".jpeg" in ImageLoader.SUPPORTED_FORMATS


class TestScatterPlotExtractor:
    """Тесты для ScatterPlotExtractor класса."""

    def test_init_default_config(self):
        """Проверка инициализации с конфигурацией по умолчанию."""
        extractor = ScatterPlotExtractor()
        assert isinstance(extractor.config, ImageConfig)

    def test_init_custom_config(self, custom_image_config):
        """Проверка инициализации с кастомной конфигурацией."""
        extractor = ScatterPlotExtractor(config=custom_image_config)
        assert extractor.config.point_detection_method == "contour"

    def test_extract_returns_scatter_plot_data(self, mock_scatter_plot_image):
        """Проверка что extract() возвращает ScatterPlotData."""
        extractor = ScatterPlotExtractor()
        result = extractor.extract(mock_scatter_plot_image)
        assert isinstance(result, ScatterPlotData)

    def test_detect_points_returns_array(self, mock_scatter_plot_image):
        """Проверка что detect_points() возвращает numpy массив."""
        extractor = ScatterPlotExtractor()
        points = extractor.detect_points(mock_scatter_plot_image)
        assert isinstance(points, np.ndarray)

    def test_extract_point_colors_returns_array(
        self, mock_scatter_plot_image
    ):
        """Проверка что extract_point_colors() возвращает numpy массив."""
        extractor = ScatterPlotExtractor()
        points = np.array([[100, 100], [200, 200]])
        colors = extractor.extract_point_colors(mock_scatter_plot_image, points)
        assert isinstance(colors, np.ndarray)
        assert colors.shape == (2, 3)

    def test_detect_axes_returns_tuple(self, mock_image_with_axes):
        """Проверка что detect_axes() возвращает tuple."""
        extractor = ScatterPlotExtractor()
        result = extractor.detect_axes(mock_image_with_axes)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_detect_plot_region_returns_tuple_or_none(self, mock_scatter_plot_image):
        """Проверка что detect_plot_region() возвращает tuple или None."""
        extractor = ScatterPlotExtractor()
        result = extractor.detect_plot_region(mock_scatter_plot_image)
        assert result is None or (isinstance(result, tuple) and len(result) == 4)

    def test_normalize_coordinates_returns_array(self):
        """Проверка что normalize_coordinates() возвращает numpy массив."""
        extractor = ScatterPlotExtractor()
        points = np.array([[100, 100], [200, 200]])
        bounds = (50, 50, 350, 350)
        result = extractor.normalize_coordinates(points, bounds)
        assert isinstance(result, np.ndarray)
        assert result.shape == points.shape

    def test_cluster_colors_returns_array(self):
        """Проверка что cluster_colors() возвращает numpy массив."""
        extractor = ScatterPlotExtractor()
        colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
        result = extractor.cluster_colors(colors)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(colors)


class TestImageAnalyzer:
    """Тесты для ImageAnalyzer класса."""

    def test_init_default_config(self):
        """Проверка инициализации с конфигурацией по умолчанию."""
        analyzer = ImageAnalyzer()
        assert isinstance(analyzer.config, ImageConfig)

    def test_init_custom_config(self, custom_image_config):
        """Проверка инициализации с кастомной конфигурацией."""
        analyzer = ImageAnalyzer(config=custom_image_config)
        assert analyzer.config.dominant_colors_count == 3

    def test_analyze_returns_result(self, mock_scatter_plot_image):
        """Проверка что analyze() возвращает ImageAnalysisResult."""
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(mock_scatter_plot_image)
        assert isinstance(result, ImageAnalysisResult)

    def test_compute_histogram_returns_array(self, mock_scatter_plot_image):
        """Проверка что compute_histogram() возвращает numpy массив."""
        analyzer = ImageAnalyzer()
        result = analyzer.compute_histogram(mock_scatter_plot_image)
        assert isinstance(result, np.ndarray)
        assert result.shape == (256,)

    def test_find_dominant_colors_returns_tuple(self, mock_scatter_plot_image):
        """Проверка что find_dominant_colors() возвращает tuple."""
        analyzer = ImageAnalyzer()
        result = analyzer.find_dominant_colors(mock_scatter_plot_image)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_compute_statistics_returns_dict(self, mock_scatter_plot_image):
        """Проверка что compute_statistics() возвращает dict."""
        analyzer = ImageAnalyzer()
        result = analyzer.compute_statistics(mock_scatter_plot_image)
        assert isinstance(result, dict)

    def test_detect_regions_returns_list(self, mock_scatter_plot_image):
        """Проверка что detect_regions() возвращает list."""
        analyzer = ImageAnalyzer()
        result = analyzer.detect_regions(mock_scatter_plot_image)
        assert isinstance(result, list)

    def test_segment_by_color_returns_mask(self, mock_scatter_plot_image):
        """Проверка что segment_by_color() возвращает маску."""
        analyzer = ImageAnalyzer()
        result = analyzer.segment_by_color(mock_scatter_plot_image, (255, 0, 0))
        assert isinstance(result, np.ndarray)
        assert result.dtype == bool


class TestConvenienceFunctions:
    """Тесты для convenience функций."""

    def test_load_image_returns_loader(self, sample_image_path):
        """Проверка что load_image() возвращает ImageLoader."""
        loader = load_image(sample_image_path)
        assert isinstance(loader, ImageLoader)

    def test_extract_scatter_plot_returns_data(self, sample_image_path):
        """Проверка что extract_scatter_plot() возвращает ScatterPlotData."""
        data = extract_scatter_plot(sample_image_path)
        assert isinstance(data, ScatterPlotData)

    def test_analyze_image_returns_result(self, sample_image_path):
        """Проверка что analyze_image() возвращает ImageAnalysisResult."""
        result = analyze_image(sample_image_path)
        assert isinstance(result, ImageAnalysisResult)


class TestImageFixtures:
    """Тесты для проверки корректности фикстур."""

    def test_mock_scatter_plot_image_shape(self, mock_scatter_plot_image):
        """Проверка формы mock scatter plot изображения."""
        assert mock_scatter_plot_image.shape == (400, 400, 3)
        assert mock_scatter_plot_image.dtype == np.uint8

    def test_mock_grayscale_image_shape(self, mock_grayscale_image):
        """Проверка формы mock grayscale изображения."""
        assert mock_grayscale_image.shape == (200, 200)
        assert mock_grayscale_image.dtype == np.uint8

    def test_mock_image_with_axes_shape(self, mock_image_with_axes):
        """Проверка формы mock изображения с осями."""
        assert mock_image_with_axes.shape == (400, 400, 3)
        assert mock_image_with_axes.dtype == np.uint8

    def test_sample_image_path_exists(self, sample_image_path):
        """Проверка что sample_image_path указывает на существующий файл."""
        assert sample_image_path.exists()
        assert sample_image_path.suffix == ".png"

    def test_sample_jpeg_path_exists(self, sample_jpeg_path):
        """Проверка что sample_jpeg_path указывает на существующий файл."""
        assert sample_jpeg_path.exists()
        assert sample_jpeg_path.suffix == ".jpg"


# =============================================================================
# TDD тесты для реализации (будут проходить после имплементации)
# =============================================================================

import pandas as pd
from src.data.image_loader import ImageMetadata, ScatterPlotData, ImageAnalysisResult


class TestImageConfigValidation:
    """TDD тесты для ImageConfig.validate()."""

    def test_validate_default_config_returns_true(self, default_image_config):
        """Валидация конфигурации по умолчанию должна вернуть True."""
        assert default_image_config.validate() is True

    def test_validate_custom_valid_config(self):
        """Валидация кастомной валидной конфигурации."""
        config = ImageConfig(
            max_dimension=2048,
            color_space="RGBA",
            min_point_radius=5,
            max_point_radius=20,
        )
        assert config.validate() is True

    def test_validate_grayscale_color_space(self):
        """Валидация с цветовым пространством grayscale."""
        config = ImageConfig(color_space="grayscale")
        assert config.validate() is True

    def test_validate_invalid_max_dimension_zero(self):
        """max_dimension=0 должен вызвать ValueError."""
        config = ImageConfig(max_dimension=0)
        with pytest.raises(ValueError, match="max_dimension"):
            config.validate()

    def test_validate_invalid_max_dimension_negative(self):
        """Отрицательный max_dimension должен вызвать ValueError."""
        config = ImageConfig(max_dimension=-100)
        with pytest.raises(ValueError, match="max_dimension"):
            config.validate()

    def test_validate_invalid_color_space(self):
        """Неподдерживаемое цветовое пространство должно вызвать ValueError."""
        config = ImageConfig(color_space="HSV")
        with pytest.raises(ValueError, match="color_space"):
            config.validate()

    def test_validate_invalid_point_detection_method(self):
        """Неподдерживаемый метод детекции должен вызвать ValueError."""
        config = ImageConfig(point_detection_method="neural_net")
        with pytest.raises(ValueError, match="point_detection_method"):
            config.validate()

    def test_validate_contour_method(self):
        """Метод contour валиден."""
        config = ImageConfig(point_detection_method="contour")
        assert config.validate() is True

    def test_validate_hough_method(self):
        """Метод hough валиден."""
        config = ImageConfig(point_detection_method="hough")
        assert config.validate() is True

    def test_validate_invalid_point_radius_order(self):
        """min_point_radius >= max_point_radius должен вызвать ValueError."""
        config = ImageConfig(min_point_radius=50, max_point_radius=10)
        with pytest.raises(ValueError, match="point_radius"):
            config.validate()

    def test_validate_invalid_point_radius_equal(self):
        """min_point_radius == max_point_radius должен вызвать ValueError."""
        config = ImageConfig(min_point_radius=10, max_point_radius=10)
        with pytest.raises(ValueError, match="point_radius"):
            config.validate()

    def test_validate_invalid_min_point_radius_zero(self):
        """min_point_radius=0 должен вызвать ValueError."""
        config = ImageConfig(min_point_radius=0)
        with pytest.raises(ValueError, match="point_radius"):
            config.validate()

    def test_validate_invalid_color_quantization_bins_zero(self):
        """color_quantization_bins=0 должен вызвать ValueError."""
        config = ImageConfig(color_quantization_bins=0)
        with pytest.raises(ValueError, match="color_quantization_bins"):
            config.validate()

    def test_validate_invalid_color_quantization_bins_negative(self):
        """Отрицательный color_quantization_bins должен вызвать ValueError."""
        config = ImageConfig(color_quantization_bins=-10)
        with pytest.raises(ValueError, match="color_quantization_bins"):
            config.validate()

    def test_validate_invalid_dominant_colors_count_zero(self):
        """dominant_colors_count=0 должен вызвать ValueError."""
        config = ImageConfig(dominant_colors_count=0)
        with pytest.raises(ValueError, match="dominant_colors_count"):
            config.validate()

    def test_validate_invalid_dominant_colors_count_negative(self):
        """Отрицательный dominant_colors_count должен вызвать ValueError."""
        config = ImageConfig(dominant_colors_count=-5)
        with pytest.raises(ValueError, match="dominant_colors_count"):
            config.validate()


class TestImageLoaderLoad:
    """TDD тесты для ImageLoader.load()."""

    def test_load_png_file_success(self, sample_image_path):
        """Успешная загрузка PNG файла."""
        loader = ImageLoader().load(sample_image_path)
        assert loader._image is not None
        assert loader._file_path == sample_image_path

    def test_load_jpeg_file_success(self, sample_jpeg_path):
        """Успешная загрузка JPEG файла."""
        loader = ImageLoader().load(sample_jpeg_path)
        assert loader._image is not None

    def test_load_returns_self_for_chaining(self, sample_image_path):
        """load() возвращает self для цепочки вызовов."""
        loader = ImageLoader()
        result = loader.load(sample_image_path)
        assert result is loader

    def test_load_file_not_found(self):
        """Несуществующий файл вызывает FileNotFoundError."""
        loader = ImageLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("nonexistent_file.png")

    def test_load_unsupported_format(self, tmp_path):
        """Неподдерживаемый формат вызывает ValueError."""
        bmp_file = tmp_path / "image.bmp"
        bmp_file.touch()
        loader = ImageLoader()
        with pytest.raises(ValueError, match="format"):
            loader.load(bmp_file)

    def test_load_converts_rgba_to_rgb(self, tmp_path):
        """RGBA изображение конвертируется в RGB."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        rgba_path = tmp_path / "rgba.png"
        img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        img.save(rgba_path)

        loader = ImageLoader().load(rgba_path)
        assert loader.get_image().shape[2] == 3  # RGB

    def test_load_respects_max_dimension(self, tmp_path):
        """Изображение масштабируется если превышает max_dimension."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        large_path = tmp_path / "large.png"
        img = Image.new("RGB", (5000, 5000), (128, 128, 128))
        img.save(large_path)

        config = ImageConfig(max_dimension=1000)
        loader = ImageLoader(config=config).load(large_path)
        image = loader.get_image()
        assert max(image.shape[:2]) <= 1000

    def test_load_preserves_aspect_ratio(self, tmp_path):
        """При масштабировании сохраняется соотношение сторон."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        # Создаём изображение 6000x3000 (2:1)
        large_path = tmp_path / "wide.png"
        img = Image.new("RGB", (6000, 3000), (128, 128, 128))
        img.save(large_path)

        config = ImageConfig(max_dimension=1000)
        loader = ImageLoader(config=config).load(large_path)
        image = loader.get_image()
        # После масштабирования должно быть 1000x500
        assert image.shape[1] == 1000  # width
        assert image.shape[0] == 500   # height

    def test_load_string_path(self, sample_image_path):
        """load() принимает строковый путь."""
        loader = ImageLoader().load(str(sample_image_path))
        assert loader._image is not None


class TestImageLoaderGetImage:
    """TDD тесты для ImageLoader.get_image()."""

    def test_get_image_returns_numpy_array(self, sample_image_path):
        """get_image() возвращает numpy массив."""
        loader = ImageLoader().load(sample_image_path)
        image = loader.get_image()
        assert isinstance(image, np.ndarray)

    def test_get_image_correct_shape(self, sample_image_path):
        """get_image() возвращает массив с правильной формой [H, W, C]."""
        loader = ImageLoader().load(sample_image_path)
        image = loader.get_image()
        assert len(image.shape) == 3
        assert image.shape[2] == 3  # RGB

    def test_get_image_correct_dtype(self, sample_image_path):
        """get_image() возвращает uint8."""
        loader = ImageLoader().load(sample_image_path)
        image = loader.get_image()
        assert image.dtype == np.uint8

    def test_get_image_returns_copy(self, sample_image_path):
        """get_image() возвращает копию, не оригинал."""
        loader = ImageLoader().load(sample_image_path)
        image1 = loader.get_image()
        image2 = loader.get_image()
        # Модификация первой копии не должна влиять на вторую
        original_value = image2[0, 0, 0]
        image1[0, 0, 0] = 255 if original_value == 0 else 0
        assert image2[0, 0, 0] == original_value


class TestImageLoaderMetadata:
    """TDD тесты для ImageLoader.get_metadata()."""

    def test_get_metadata_returns_correct_type(self, sample_image_path):
        """get_metadata() возвращает ImageMetadata."""
        loader = ImageLoader().load(sample_image_path)
        meta = loader.get_metadata()
        assert isinstance(meta, ImageMetadata)

    def test_get_metadata_correct_filename(self, sample_image_path):
        """Имя файла в метаданных корректное."""
        loader = ImageLoader().load(sample_image_path)
        meta = loader.get_metadata()
        assert meta.filename == sample_image_path.name

    def test_get_metadata_correct_dimensions(self, sample_image_path):
        """Размеры в метаданных корректные."""
        loader = ImageLoader().load(sample_image_path)
        meta = loader.get_metadata()
        image = loader.get_image()
        assert meta.height == image.shape[0]
        assert meta.width == image.shape[1]

    def test_get_metadata_correct_channels(self, sample_image_path):
        """Количество каналов в метаданных корректное."""
        loader = ImageLoader().load(sample_image_path)
        meta = loader.get_metadata()
        assert meta.channels == 3

    def test_get_metadata_correct_format_png(self, sample_image_path):
        """Формат PNG определяется корректно."""
        loader = ImageLoader().load(sample_image_path)
        meta = loader.get_metadata()
        assert meta.format == "PNG"

    def test_get_metadata_correct_format_jpeg(self, sample_jpeg_path):
        """Формат JPEG определяется корректно."""
        loader = ImageLoader().load(sample_jpeg_path)
        meta = loader.get_metadata()
        assert meta.format == "JPEG"

    def test_get_metadata_file_size(self, sample_image_path):
        """Размер файла в метаданных положительный."""
        loader = ImageLoader().load(sample_image_path)
        meta = loader.get_metadata()
        assert meta.file_size_bytes > 0

    def test_get_metadata_bit_depth(self, sample_image_path):
        """Битовая глубина в метаданных корректна."""
        loader = ImageLoader().load(sample_image_path)
        meta = loader.get_metadata()
        assert meta.bit_depth == 8

    def test_get_metadata_caches_result(self, sample_image_path):
        """get_metadata() кэширует результат."""
        loader = ImageLoader().load(sample_image_path)
        meta1 = loader.get_metadata()
        meta2 = loader.get_metadata()
        assert meta1 is meta2

    def test_get_metadata_has_alpha_false_for_rgb(self, sample_image_path):
        """has_alpha=False для RGB изображения."""
        loader = ImageLoader().load(sample_image_path)
        meta = loader.get_metadata()
        assert meta.has_alpha is False


class TestImageLoaderTransforms:
    """TDD тесты для преобразований ImageLoader."""

    # === to_grayscale ===

    def test_to_grayscale_returns_2d_array(self, sample_image_path):
        """to_grayscale() возвращает 2D массив."""
        loader = ImageLoader().load(sample_image_path)
        gray = loader.to_grayscale()
        assert len(gray.shape) == 2

    def test_to_grayscale_correct_shape(self, sample_image_path):
        """to_grayscale() сохраняет H и W."""
        loader = ImageLoader().load(sample_image_path)
        image = loader.get_image()
        gray = loader.to_grayscale()
        assert gray.shape == (image.shape[0], image.shape[1])

    def test_to_grayscale_values_in_range(self, sample_image_path):
        """to_grayscale() возвращает значения в диапазоне [0, 255]."""
        loader = ImageLoader().load(sample_image_path)
        gray = loader.to_grayscale()
        assert gray.min() >= 0
        assert gray.max() <= 255

    def test_to_grayscale_correct_dtype(self, sample_image_path):
        """to_grayscale() возвращает uint8."""
        loader = ImageLoader().load(sample_image_path)
        gray = loader.to_grayscale()
        assert gray.dtype == np.uint8

    def test_to_grayscale_uses_luminance_formula(self):
        """to_grayscale() использует формулу яркости (0.299R + 0.587G + 0.114B)."""
        loader = ImageLoader()
        # Создаём изображение с известными значениями
        loader._image = np.array([[[100, 150, 200]]], dtype=np.uint8)
        gray = loader.to_grayscale()
        # Ожидаемое значение: 0.299*100 + 0.587*150 + 0.114*200 = 29.9 + 88.05 + 22.8 = 140.75
        expected = int(0.299 * 100 + 0.587 * 150 + 0.114 * 200)
        assert abs(gray[0, 0] - expected) <= 1  # Допустимая погрешность округления

    # === resize ===

    def test_resize_correct_dimensions(self, sample_image_path):
        """resize() возвращает изображение с заданными размерами."""
        loader = ImageLoader().load(sample_image_path)
        resized = loader.resize(100, 50)
        assert resized.shape[0] == 50   # height
        assert resized.shape[1] == 100  # width

    def test_resize_preserves_channels(self, sample_image_path):
        """resize() сохраняет количество каналов."""
        loader = ImageLoader().load(sample_image_path)
        resized = loader.resize(100, 50)
        assert resized.shape[2] == 3

    def test_resize_invalid_zero_width(self):
        """resize() с width=0 вызывает ValueError."""
        loader = ImageLoader()
        loader._image = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            loader.resize(0, 50)

    def test_resize_invalid_negative_width(self):
        """resize() с отрицательной шириной вызывает ValueError."""
        loader = ImageLoader()
        loader._image = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            loader.resize(-50, 50)

    def test_resize_invalid_zero_height(self):
        """resize() с height=0 вызывает ValueError."""
        loader = ImageLoader()
        loader._image = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            loader.resize(50, 0)

    def test_resize_invalid_negative_height(self):
        """resize() с отрицательной высотой вызывает ValueError."""
        loader = ImageLoader()
        loader._image = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            loader.resize(50, -10)

    def test_resize_upscale(self, sample_image_path):
        """resize() может увеличивать изображение."""
        loader = ImageLoader().load(sample_image_path)
        original_shape = loader.get_image().shape
        resized = loader.resize(original_shape[1] * 2, original_shape[0] * 2)
        assert resized.shape[0] == original_shape[0] * 2
        assert resized.shape[1] == original_shape[1] * 2

    # === crop ===

    def test_crop_correct_region(self, sample_image_path):
        """crop() возвращает правильную область."""
        loader = ImageLoader().load(sample_image_path)
        cropped = loader.crop(10, 20, 50, 30)
        assert cropped.shape == (30, 50, 3)  # height, width, channels

    def test_crop_preserves_pixel_values(self):
        """crop() сохраняет значения пикселей."""
        loader = ImageLoader()
        # Создаём изображение с известным паттерном
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[20:50, 10:60] = [255, 128, 64]
        loader._image = image

        cropped = loader.crop(10, 20, 50, 30)
        assert np.all(cropped == [255, 128, 64])

    def test_crop_out_of_bounds_right(self):
        """crop() за пределами изображения (справа) вызывает ValueError."""
        loader = ImageLoader()
        loader._image = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            loader.crop(80, 10, 50, 50)  # x + width > image_width

    def test_crop_out_of_bounds_bottom(self):
        """crop() за пределами изображения (снизу) вызывает ValueError."""
        loader = ImageLoader()
        loader._image = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            loader.crop(10, 80, 50, 50)  # y + height > image_height

    def test_crop_negative_x(self):
        """crop() с отрицательной x координатой вызывает ValueError."""
        loader = ImageLoader()
        loader._image = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            loader.crop(-10, 10, 50, 50)

    def test_crop_negative_y(self):
        """crop() с отрицательной y координатой вызывает ValueError."""
        loader = ImageLoader()
        loader._image = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            loader.crop(10, -10, 50, 50)

    def test_crop_negative_width(self):
        """crop() с отрицательной шириной вызывает ValueError."""
        loader = ImageLoader()
        loader._image = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            loader.crop(10, 10, -50, 50)

    def test_crop_negative_height(self):
        """crop() с отрицательной высотой вызывает ValueError."""
        loader = ImageLoader()
        loader._image = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            loader.crop(10, 10, 50, -50)

    def test_crop_zero_width(self):
        """crop() с нулевой шириной вызывает ValueError."""
        loader = ImageLoader()
        loader._image = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            loader.crop(10, 10, 0, 50)

    def test_crop_zero_height(self):
        """crop() с нулевой высотой вызывает ValueError."""
        loader = ImageLoader()
        loader._image = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            loader.crop(10, 10, 50, 0)

    # === validate_format ===

    def test_validate_format_png_returns_true(self, sample_image_path):
        """validate_format() возвращает True для PNG."""
        loader = ImageLoader().load(sample_image_path)
        assert loader.validate_format() is True

    def test_validate_format_jpeg_returns_true(self, sample_jpeg_path):
        """validate_format() возвращает True для JPEG."""
        loader = ImageLoader().load(sample_jpeg_path)
        assert loader.validate_format() is True

    def test_validate_format_jpg_extension(self, sample_jpeg_path):
        """validate_format() работает с расширением .jpg."""
        loader = ImageLoader().load(sample_jpeg_path)
        assert loader.validate_format() is True


class TestScatterPlotExtractorDetection:
    """TDD тесты для детекции точек ScatterPlotExtractor."""

    def test_detect_points_returns_array(self, mock_scatter_plot_image):
        """detect_points() возвращает numpy массив."""
        extractor = ScatterPlotExtractor()
        points = extractor.detect_points(mock_scatter_plot_image)
        assert isinstance(points, np.ndarray)

    def test_detect_points_correct_shape(self, mock_scatter_plot_image):
        """detect_points() возвращает [n_points, 2]."""
        extractor = ScatterPlotExtractor()
        points = extractor.detect_points(mock_scatter_plot_image)
        assert len(points.shape) == 2
        assert points.shape[1] == 2

    def test_detect_points_finds_points(self, mock_scatter_plot_image):
        """detect_points() находит точки на изображении."""
        extractor = ScatterPlotExtractor()
        points = extractor.detect_points(mock_scatter_plot_image)
        assert len(points) > 0

    def test_detect_points_threshold_method(self, mock_scatter_plot_image):
        """detect_points с методом threshold работает."""
        extractor = ScatterPlotExtractor()
        points = extractor.detect_points(mock_scatter_plot_image, method="threshold")
        assert len(points) > 0

    def test_detect_points_contour_method(self, mock_scatter_plot_image):
        """detect_points с методом contour работает."""
        extractor = ScatterPlotExtractor()
        points = extractor.detect_points(mock_scatter_plot_image, method="contour")
        assert isinstance(points, np.ndarray)

    def test_detect_points_hough_method(self, mock_scatter_plot_image):
        """detect_points с методом hough работает."""
        extractor = ScatterPlotExtractor()
        points = extractor.detect_points(mock_scatter_plot_image, method="hough")
        assert isinstance(points, np.ndarray)

    def test_detect_points_uses_config_method(self, mock_scatter_plot_image):
        """detect_points() использует метод из конфигурации по умолчанию."""
        config = ImageConfig(point_detection_method="contour")
        extractor = ScatterPlotExtractor(config=config)
        # Метод должен быть contour по умолчанию
        points = extractor.detect_points(mock_scatter_plot_image)
        assert isinstance(points, np.ndarray)

    def test_detect_points_coordinates_in_image_bounds(self, mock_scatter_plot_image):
        """Координаты точек находятся в пределах изображения."""
        extractor = ScatterPlotExtractor()
        points = extractor.detect_points(mock_scatter_plot_image)
        h, w = mock_scatter_plot_image.shape[:2]
        if len(points) > 0:
            assert np.all(points[:, 0] >= 0)
            assert np.all(points[:, 0] < w)
            assert np.all(points[:, 1] >= 0)
            assert np.all(points[:, 1] < h)


class TestScatterPlotExtractorColors:
    """TDD тесты для извлечения цветов ScatterPlotExtractor."""

    def test_extract_point_colors_returns_array(self, mock_scatter_plot_image):
        """extract_point_colors() возвращает numpy массив."""
        extractor = ScatterPlotExtractor()
        points = np.array([[100, 100], [200, 200]])
        colors = extractor.extract_point_colors(mock_scatter_plot_image, points)
        assert isinstance(colors, np.ndarray)

    def test_extract_point_colors_correct_shape(self, mock_scatter_plot_image):
        """extract_point_colors() возвращает [n_points, 3]."""
        extractor = ScatterPlotExtractor()
        points = np.array([[100, 100], [200, 200]])
        colors = extractor.extract_point_colors(mock_scatter_plot_image, points)
        assert colors.shape == (2, 3)

    def test_extract_point_colors_values_in_range(self, mock_scatter_plot_image):
        """extract_point_colors() возвращает RGB в диапазоне [0, 255]."""
        extractor = ScatterPlotExtractor()
        points = np.array([[100, 100], [200, 200]])
        colors = extractor.extract_point_colors(mock_scatter_plot_image, points)
        assert colors.min() >= 0
        assert colors.max() <= 255

    def test_extract_point_colors_correct_values(self):
        """extract_point_colors() возвращает корректные значения цветов."""
        extractor = ScatterPlotExtractor()
        # Создаём изображение с известными цветами
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[50, 50] = [255, 128, 64]
        image[25, 75] = [0, 255, 0]

        points = np.array([[50, 50], [75, 25]])  # x, y -> image[y, x]
        colors = extractor.extract_point_colors(image, points)

        assert np.allclose(colors[0], [255, 128, 64], atol=10)  # Усреднение окрестности
        assert np.allclose(colors[1], [0, 255, 0], atol=10)

    def test_cluster_colors_returns_array(self):
        """cluster_colors() возвращает numpy массив."""
        extractor = ScatterPlotExtractor()
        colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 0, 0]])
        labels = extractor.cluster_colors(colors, n_clusters=3)
        assert isinstance(labels, np.ndarray)

    def test_cluster_colors_correct_shape(self):
        """cluster_colors() возвращает [n_points] labels."""
        extractor = ScatterPlotExtractor()
        colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
        labels = extractor.cluster_colors(colors, n_clusters=3)
        assert labels.shape == (3,)

    def test_cluster_colors_valid_labels(self):
        """cluster_colors() возвращает метки от 0 до n_clusters-1."""
        extractor = ScatterPlotExtractor()
        colors = np.random.randint(0, 256, (100, 3))
        labels = extractor.cluster_colors(colors, n_clusters=5)
        assert labels.min() >= 0
        assert labels.max() <= 4

    def test_cluster_colors_groups_similar(self):
        """cluster_colors() группирует похожие цвета."""
        extractor = ScatterPlotExtractor()
        # Создаём 3 явно различающиеся группы
        colors = np.array([
            [255, 0, 0], [250, 5, 5], [245, 10, 0],  # Красные
            [0, 255, 0], [5, 250, 5], [0, 245, 10],  # Зелёные
            [0, 0, 255], [5, 5, 250], [10, 0, 245],  # Синие
        ])
        labels = extractor.cluster_colors(colors, n_clusters=3)
        # Проверяем что красные сгруппированы
        assert labels[0] == labels[1] == labels[2]
        # Проверяем что зелёные сгруппированы
        assert labels[3] == labels[4] == labels[5]
        # Проверяем что синие сгруппированы
        assert labels[6] == labels[7] == labels[8]


class TestScatterPlotExtractorAxes:
    """TDD тесты для детекции осей ScatterPlotExtractor."""

    def test_detect_axes_returns_tuple(self, mock_image_with_axes):
        """detect_axes() возвращает tuple."""
        extractor = ScatterPlotExtractor()
        result = extractor.detect_axes(mock_image_with_axes)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_detect_axes_finds_axes_on_image_with_axes(self, mock_image_with_axes):
        """detect_axes() находит оси на изображении с осями."""
        extractor = ScatterPlotExtractor()
        x_axis, y_axis = extractor.detect_axes(mock_image_with_axes)
        # Хотя бы одна ось должна быть найдена
        assert x_axis is not None or y_axis is not None

    def test_detect_plot_region_returns_tuple_or_none(self, mock_image_with_axes):
        """detect_plot_region() возвращает tuple или None."""
        extractor = ScatterPlotExtractor()
        result = extractor.detect_plot_region(mock_image_with_axes)
        assert result is None or (isinstance(result, tuple) and len(result) == 4)

    def test_detect_plot_region_valid_bounds(self, mock_image_with_axes):
        """detect_plot_region() возвращает валидные границы."""
        extractor = ScatterPlotExtractor()
        bounds = extractor.detect_plot_region(mock_image_with_axes)
        if bounds:
            x_min, y_min, x_max, y_max = bounds
            assert x_min < x_max
            assert y_min < y_max
            assert x_min >= 0 and y_min >= 0


class TestScatterPlotExtractorNormalization:
    """TDD тесты для нормализации координат ScatterPlotExtractor."""

    def test_normalize_coordinates_returns_array(self):
        """normalize_coordinates() возвращает numpy массив."""
        extractor = ScatterPlotExtractor()
        points = np.array([[100, 100], [200, 200]])
        bounds = (50, 50, 350, 350)
        normalized = extractor.normalize_coordinates(points, bounds)
        assert isinstance(normalized, np.ndarray)

    def test_normalize_coordinates_correct_shape(self):
        """normalize_coordinates() сохраняет форму."""
        extractor = ScatterPlotExtractor()
        points = np.array([[100, 100], [200, 200], [300, 300]])
        bounds = (50, 50, 350, 350)
        normalized = extractor.normalize_coordinates(points, bounds)
        assert normalized.shape == points.shape

    def test_normalize_coordinates_values_in_range(self):
        """normalize_coordinates() возвращает значения в [0, 1]."""
        extractor = ScatterPlotExtractor()
        points = np.array([[100, 100], [200, 200]])
        bounds = (50, 50, 350, 350)
        normalized = extractor.normalize_coordinates(points, bounds)
        assert normalized.min() >= 0
        assert normalized.max() <= 1

    def test_normalize_coordinates_boundary_points(self):
        """normalize_coordinates() корректно обрабатывает граничные точки."""
        extractor = ScatterPlotExtractor()
        points = np.array([[50, 50], [350, 350]])  # на границах
        bounds = (50, 50, 350, 350)
        normalized = extractor.normalize_coordinates(points, bounds)
        assert np.allclose(normalized[0], [0, 0])
        assert np.allclose(normalized[1], [1, 1])

    def test_normalize_coordinates_middle_point(self):
        """normalize_coordinates() корректно нормализует середину."""
        extractor = ScatterPlotExtractor()
        points = np.array([[200, 200]])  # середина bounds (50, 50, 350, 350)
        bounds = (50, 50, 350, 350)
        normalized = extractor.normalize_coordinates(points, bounds)
        assert np.allclose(normalized[0], [0.5, 0.5])


class TestScatterPlotExtractorExtract:
    """TDD тесты для полного извлечения ScatterPlotExtractor.extract()."""

    def test_extract_returns_scatter_plot_data(self, mock_scatter_plot_image):
        """extract() возвращает ScatterPlotData."""
        extractor = ScatterPlotExtractor()
        result = extractor.extract(mock_scatter_plot_image)
        assert isinstance(result, ScatterPlotData)

    def test_extract_populates_points(self, mock_scatter_plot_image):
        """extract() заполняет поле points."""
        extractor = ScatterPlotExtractor()
        result = extractor.extract(mock_scatter_plot_image)
        assert result.points is not None
        assert isinstance(result.points, np.ndarray)

    def test_extract_populates_normalized_points(self, mock_scatter_plot_image):
        """extract() заполняет поле points_normalized."""
        extractor = ScatterPlotExtractor()
        result = extractor.extract(mock_scatter_plot_image)
        assert result.points_normalized is not None
        assert isinstance(result.points_normalized, np.ndarray)

    def test_extract_populates_colors(self, mock_scatter_plot_image):
        """extract() заполняет поле colors."""
        extractor = ScatterPlotExtractor()
        result = extractor.extract(mock_scatter_plot_image)
        assert result.colors is not None
        assert isinstance(result.colors, np.ndarray)

    def test_extract_populates_n_points(self, mock_scatter_plot_image):
        """extract() заполняет поле n_points."""
        extractor = ScatterPlotExtractor()
        result = extractor.extract(mock_scatter_plot_image)
        assert result.n_points >= 0

    def test_extract_populates_detection_confidence(self, mock_scatter_plot_image):
        """extract() заполняет поле detection_confidence."""
        extractor = ScatterPlotExtractor()
        result = extractor.extract(mock_scatter_plot_image)
        assert 0 <= result.detection_confidence <= 1

    def test_extract_consistent_shapes(self, mock_scatter_plot_image):
        """extract() возвращает согласованные размеры массивов."""
        extractor = ScatterPlotExtractor()
        result = extractor.extract(mock_scatter_plot_image)
        assert result.points.shape[0] == result.n_points
        assert result.points_normalized.shape[0] == result.n_points
        assert result.colors.shape[0] == result.n_points

    def test_extract_with_custom_config(self, mock_scatter_plot_image):
        """extract() работает с кастомной конфигурацией."""
        config = ImageConfig(
            point_detection_method="contour",
            min_point_radius=3,
            max_point_radius=20,
        )
        extractor = ScatterPlotExtractor(config=config)
        result = extractor.extract(mock_scatter_plot_image)
        assert isinstance(result, ScatterPlotData)


class TestScatterPlotDataToDataframe:
    """TDD тесты для ScatterPlotData.to_dataframe()."""

    def test_to_dataframe_returns_dataframe(self):
        """to_dataframe() возвращает pandas DataFrame."""
        data = ScatterPlotData(
            points=np.array([[100, 100], [200, 200]]),
            points_normalized=np.array([[0.2, 0.2], [0.6, 0.6]]),
            colors=np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8),
            color_labels=np.array([0, 1]),
            n_points=2,
            detection_confidence=0.9,
            plot_bounds=(50, 50, 350, 350),
            axis_labels=("X", "Y"),
        )
        df = data.to_dataframe()
        assert isinstance(df, pd.DataFrame)

    def test_to_dataframe_correct_columns(self):
        """to_dataframe() содержит правильные колонки."""
        data = ScatterPlotData(
            points=np.array([[100, 100]]),
            points_normalized=np.array([[0.5, 0.5]]),
            colors=np.array([[255, 0, 0]], dtype=np.uint8),
            color_labels=None,
            n_points=1,
            detection_confidence=0.9,
            plot_bounds=None,
            axis_labels=(None, None),
        )
        df = data.to_dataframe()
        expected_cols = ["x", "y", "x_norm", "y_norm", "r", "g", "b"]
        for col in expected_cols:
            assert col in df.columns

    def test_to_dataframe_correct_rows(self):
        """to_dataframe() содержит правильное количество строк."""
        n = 10
        data = ScatterPlotData(
            points=np.random.randint(0, 400, (n, 2)),
            points_normalized=np.random.rand(n, 2),
            colors=np.random.randint(0, 256, (n, 3), dtype=np.uint8),
            color_labels=None,
            n_points=n,
            detection_confidence=0.85,
            plot_bounds=None,
            axis_labels=(None, None),
        )
        df = data.to_dataframe()
        assert len(df) == n

    def test_to_dataframe_includes_color_labels_if_present(self):
        """to_dataframe() включает color_label если есть."""
        data = ScatterPlotData(
            points=np.array([[100, 100], [200, 200]]),
            points_normalized=np.array([[0.2, 0.2], [0.6, 0.6]]),
            colors=np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8),
            color_labels=np.array([0, 1]),
            n_points=2,
            detection_confidence=0.9,
            plot_bounds=None,
            axis_labels=(None, None),
        )
        df = data.to_dataframe()
        assert "color_label" in df.columns

    def test_to_dataframe_no_color_labels_column_if_none(self):
        """to_dataframe() не включает color_label если None."""
        data = ScatterPlotData(
            points=np.array([[100, 100]]),
            points_normalized=np.array([[0.5, 0.5]]),
            colors=np.array([[255, 0, 0]], dtype=np.uint8),
            color_labels=None,
            n_points=1,
            detection_confidence=0.9,
            plot_bounds=None,
            axis_labels=(None, None),
        )
        df = data.to_dataframe()
        assert "color_label" not in df.columns

    def test_to_dataframe_correct_values(self):
        """to_dataframe() содержит корректные значения."""
        data = ScatterPlotData(
            points=np.array([[100, 200]]),
            points_normalized=np.array([[0.25, 0.75]]),
            colors=np.array([[255, 128, 64]], dtype=np.uint8),
            color_labels=None,
            n_points=1,
            detection_confidence=0.9,
            plot_bounds=None,
            axis_labels=(None, None),
        )
        df = data.to_dataframe()
        assert df.iloc[0]["x"] == 100
        assert df.iloc[0]["y"] == 200
        assert df.iloc[0]["x_norm"] == 0.25
        assert df.iloc[0]["y_norm"] == 0.75
        assert df.iloc[0]["r"] == 255
        assert df.iloc[0]["g"] == 128
        assert df.iloc[0]["b"] == 64


class TestImageAnalyzerHistogram:
    """TDD тесты для гистограмм ImageAnalyzer."""

    def test_compute_histogram_returns_array(self, mock_scatter_plot_image):
        """compute_histogram() возвращает numpy массив."""
        analyzer = ImageAnalyzer()
        hist = analyzer.compute_histogram(mock_scatter_plot_image, channel="gray")
        assert isinstance(hist, np.ndarray)

    def test_compute_histogram_correct_shape(self, mock_scatter_plot_image):
        """compute_histogram() возвращает [256] bins."""
        analyzer = ImageAnalyzer()
        hist = analyzer.compute_histogram(mock_scatter_plot_image)
        assert hist.shape == (256,)

    def test_compute_histogram_all_channels(self, mock_scatter_plot_image):
        """compute_histogram() работает для всех каналов."""
        analyzer = ImageAnalyzer()
        for channel in ["r", "g", "b", "gray"]:
            hist = analyzer.compute_histogram(mock_scatter_plot_image, channel=channel)
            assert hist.shape == (256,)

    def test_compute_histogram_sum_equals_pixels(self, mock_scatter_plot_image):
        """Сумма гистограммы равна количеству пикселей."""
        analyzer = ImageAnalyzer()
        hist = analyzer.compute_histogram(mock_scatter_plot_image, channel="gray")
        h, w = mock_scatter_plot_image.shape[:2]
        assert hist.sum() == h * w

    def test_compute_histogram_non_negative(self, mock_scatter_plot_image):
        """Все значения гистограммы неотрицательные."""
        analyzer = ImageAnalyzer()
        hist = analyzer.compute_histogram(mock_scatter_plot_image)
        assert np.all(hist >= 0)


class TestImageAnalyzerColors:
    """TDD тесты для доминантных цветов ImageAnalyzer."""

    def test_find_dominant_colors_returns_tuple(self, mock_scatter_plot_image):
        """find_dominant_colors() возвращает tuple."""
        analyzer = ImageAnalyzer()
        result = analyzer.find_dominant_colors(mock_scatter_plot_image, n_colors=3)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_find_dominant_colors_correct_shapes(self, mock_scatter_plot_image):
        """find_dominant_colors() возвращает правильные формы."""
        analyzer = ImageAnalyzer()
        colors, percentages = analyzer.find_dominant_colors(
            mock_scatter_plot_image, n_colors=5
        )
        assert colors.shape == (5, 3)
        assert percentages.shape == (5,)

    def test_find_dominant_colors_percentages_sum_to_one(self, mock_scatter_plot_image):
        """Проценты доминантных цветов суммируются к ~1."""
        analyzer = ImageAnalyzer()
        _, percentages = analyzer.find_dominant_colors(mock_scatter_plot_image)
        assert np.isclose(percentages.sum(), 1.0, atol=0.01)

    def test_find_dominant_colors_values_in_range(self, mock_scatter_plot_image):
        """Значения цветов в диапазоне [0, 255]."""
        analyzer = ImageAnalyzer()
        colors, _ = analyzer.find_dominant_colors(mock_scatter_plot_image)
        assert colors.min() >= 0
        assert colors.max() <= 255

    def test_find_dominant_colors_sorted_by_percentage(self, mock_scatter_plot_image):
        """Доминантные цвета отсортированы по убыванию процента."""
        analyzer = ImageAnalyzer()
        _, percentages = analyzer.find_dominant_colors(mock_scatter_plot_image)
        # Проверяем что отсортировано по убыванию
        assert np.all(percentages[:-1] >= percentages[1:])


class TestImageAnalyzerStatistics:
    """TDD тесты для статистик ImageAnalyzer."""

    def test_compute_statistics_returns_dict(self, mock_scatter_plot_image):
        """compute_statistics() возвращает dict."""
        analyzer = ImageAnalyzer()
        stats = analyzer.compute_statistics(mock_scatter_plot_image)
        assert isinstance(stats, dict)

    def test_compute_statistics_contains_required_keys(self, mock_scatter_plot_image):
        """compute_statistics() содержит все необходимые ключи."""
        analyzer = ImageAnalyzer()
        stats = analyzer.compute_statistics(mock_scatter_plot_image)
        required_keys = [
            "mean_r",
            "mean_g",
            "mean_b",
            "std_r",
            "std_g",
            "std_b",
            "brightness",
            "contrast",
        ]
        for key in required_keys:
            assert key in stats

    def test_compute_statistics_brightness_in_range(self, mock_scatter_plot_image):
        """Яркость в допустимом диапазоне [0, 255]."""
        analyzer = ImageAnalyzer()
        stats = analyzer.compute_statistics(mock_scatter_plot_image)
        assert 0 <= stats["brightness"] <= 255

    def test_compute_statistics_contrast_non_negative(self, mock_scatter_plot_image):
        """Контраст неотрицательный."""
        analyzer = ImageAnalyzer()
        stats = analyzer.compute_statistics(mock_scatter_plot_image)
        assert stats["contrast"] >= 0

    def test_compute_statistics_std_non_negative(self, mock_scatter_plot_image):
        """Стандартные отклонения неотрицательные."""
        analyzer = ImageAnalyzer()
        stats = analyzer.compute_statistics(mock_scatter_plot_image)
        assert stats["std_r"] >= 0
        assert stats["std_g"] >= 0
        assert stats["std_b"] >= 0


class TestImageAnalyzerRegions:
    """TDD тесты для детекции регионов ImageAnalyzer."""

    def test_detect_regions_returns_list(self, mock_scatter_plot_image):
        """detect_regions() возвращает list."""
        analyzer = ImageAnalyzer()
        regions = analyzer.detect_regions(mock_scatter_plot_image)
        assert isinstance(regions, list)

    def test_detect_regions_dict_structure(self, mock_scatter_plot_image):
        """detect_regions() возвращает словари с правильной структурой."""
        analyzer = ImageAnalyzer()
        regions = analyzer.detect_regions(mock_scatter_plot_image)
        if regions:
            region = regions[0]
            assert "bounds" in region
            assert "area" in region
            assert "centroid" in region

    def test_detect_regions_valid_bounds(self, mock_scatter_plot_image):
        """Границы регионов валидны."""
        analyzer = ImageAnalyzer()
        regions = analyzer.detect_regions(mock_scatter_plot_image)
        h, w = mock_scatter_plot_image.shape[:2]
        for region in regions:
            bounds = region["bounds"]
            assert bounds[0] >= 0  # x_min
            assert bounds[1] >= 0  # y_min
            assert bounds[2] <= w  # x_max
            assert bounds[3] <= h  # y_max


class TestImageAnalyzerSegmentation:
    """TDD тесты для сегментации ImageAnalyzer."""

    def test_segment_by_color_returns_mask(self, mock_scatter_plot_image):
        """segment_by_color() возвращает бинарную маску."""
        analyzer = ImageAnalyzer()
        mask = analyzer.segment_by_color(
            mock_scatter_plot_image, target_color=(255, 255, 255)
        )
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool

    def test_segment_by_color_correct_shape(self, mock_scatter_plot_image):
        """segment_by_color() возвращает маску правильной формы."""
        analyzer = ImageAnalyzer()
        mask = analyzer.segment_by_color(
            mock_scatter_plot_image, target_color=(255, 0, 0)
        )
        assert mask.shape == mock_scatter_plot_image.shape[:2]

    def test_segment_by_color_respects_tolerance(self, mock_scatter_plot_image):
        """segment_by_color() учитывает tolerance."""
        analyzer = ImageAnalyzer()
        mask_tight = analyzer.segment_by_color(
            mock_scatter_plot_image, (128, 128, 128), tolerance=10
        )
        mask_loose = analyzer.segment_by_color(
            mock_scatter_plot_image, (128, 128, 128), tolerance=100
        )
        assert mask_loose.sum() >= mask_tight.sum()

    def test_segment_by_color_finds_exact_color(self):
        """segment_by_color() находит пиксели с точным цветом."""
        analyzer = ImageAnalyzer()
        # Создаём изображение с известным цветом
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[25:75, 25:75] = [255, 0, 0]  # Красный квадрат

        mask = analyzer.segment_by_color(image, target_color=(255, 0, 0), tolerance=10)
        # Маска должна покрывать красный квадрат
        assert mask[50, 50] == True  # noqa: E712
        assert mask[10, 10] == False  # noqa: E712


class TestImageAnalyzerAnalyze:
    """TDD тесты для полного анализа ImageAnalyzer.analyze()."""

    def test_analyze_returns_result(self, mock_scatter_plot_image):
        """analyze() возвращает ImageAnalysisResult."""
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(mock_scatter_plot_image)
        assert isinstance(result, ImageAnalysisResult)

    def test_analyze_populates_histograms(self, mock_scatter_plot_image):
        """analyze() заполняет все гистограммы."""
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(mock_scatter_plot_image)
        assert result.histogram_r.shape == (256,)
        assert result.histogram_g.shape == (256,)
        assert result.histogram_b.shape == (256,)
        assert result.histogram_gray.shape == (256,)

    def test_analyze_populates_colors(self, mock_scatter_plot_image):
        """analyze() заполняет доминантные цвета."""
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(mock_scatter_plot_image)
        assert result.dominant_colors is not None
        assert result.dominant_colors_percentages is not None

    def test_analyze_populates_statistics(self, mock_scatter_plot_image):
        """analyze() заполняет статистики."""
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(mock_scatter_plot_image)
        assert isinstance(result.brightness, float)
        assert isinstance(result.contrast, float)
        assert len(result.mean_color) == 3
        assert len(result.std_color) == 3


class TestConvenienceFunctionsTDD:
    """TDD тесты для convenience функций."""

    def test_load_image_returns_loader(self, sample_image_path):
        """load_image() возвращает инициализированный ImageLoader."""
        loader = load_image(sample_image_path)
        assert isinstance(loader, ImageLoader)
        assert loader._image is not None

    def test_load_image_with_config(self, sample_image_path):
        """load_image() принимает конфигурацию."""
        config = ImageConfig(max_dimension=500)
        loader = load_image(sample_image_path, config=config)
        assert loader.config.max_dimension == 500

    def test_extract_scatter_plot_returns_data(self, sample_image_path):
        """extract_scatter_plot() возвращает ScatterPlotData."""
        data = extract_scatter_plot(sample_image_path)
        assert isinstance(data, ScatterPlotData)

    def test_extract_scatter_plot_with_config(self, sample_image_path):
        """extract_scatter_plot() принимает конфигурацию."""
        config = ImageConfig(point_detection_method="contour")
        data = extract_scatter_plot(sample_image_path, config=config)
        assert isinstance(data, ScatterPlotData)

    def test_analyze_image_returns_result(self, sample_image_path):
        """analyze_image() возвращает ImageAnalysisResult."""
        result = analyze_image(sample_image_path)
        assert isinstance(result, ImageAnalysisResult)

    def test_analyze_image_with_config(self, sample_image_path):
        """analyze_image() принимает конфигурацию."""
        config = ImageConfig(dominant_colors_count=3)
        result = analyze_image(sample_image_path, config=config)
        assert isinstance(result, ImageAnalysisResult)


class TestEdgeCases:
    """TDD тесты для граничных случаев."""

    def test_empty_white_image(self):
        """Обработка полностью белого изображения."""
        white_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        extractor = ScatterPlotExtractor()
        result = extractor.extract(white_image)
        assert result.n_points == 0

    def test_single_pixel_point(self):
        """Обработка изображения с одной точкой."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        image[50, 50] = [255, 0, 0]
        extractor = ScatterPlotExtractor()
        # Одиночный пиксель может не определиться как точка
        points = extractor.detect_points(image)
        assert isinstance(points, np.ndarray)

    def test_grayscale_input_image(self):
        """Обработка grayscale изображения."""
        gray_image = np.ones((100, 100), dtype=np.uint8) * 128
        analyzer = ImageAnalyzer()
        hist = analyzer.compute_histogram(gray_image, channel="gray")
        assert hist.shape == (256,)

    def test_very_small_image(self):
        """Обработка очень маленького изображения."""
        small_image = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(small_image)
        assert isinstance(result, ImageAnalysisResult)

    def test_uniform_color_image(self):
        """Обработка изображения с одним цветом."""
        uniform_image = np.full((100, 100, 3), [128, 64, 32], dtype=np.uint8)
        analyzer = ImageAnalyzer()
        stats = analyzer.compute_statistics(uniform_image)
        assert np.isclose(stats["std_r"], 0, atol=0.01)
        assert np.isclose(stats["std_g"], 0, atol=0.01)
        assert np.isclose(stats["std_b"], 0, atol=0.01)


class TestIntegration:
    """Интеграционные TDD тесты."""

    def test_full_pipeline_png(self, sample_image_path):
        """Полный pipeline загрузки и анализа PNG."""
        loader = ImageLoader().load(sample_image_path)
        image = loader.get_image()
        meta = loader.get_metadata()

        extractor = ScatterPlotExtractor()
        scatter_data = extractor.extract(image)

        analyzer = ImageAnalyzer()
        analysis = analyzer.analyze(image)

        assert meta.format == "PNG"
        assert isinstance(scatter_data, ScatterPlotData)
        assert isinstance(analysis, ImageAnalysisResult)

    def test_full_pipeline_jpeg(self, sample_jpeg_path):
        """Полный pipeline загрузки и анализа JPEG."""
        loader = ImageLoader().load(sample_jpeg_path)
        image = loader.get_image()
        meta = loader.get_metadata()

        assert meta.format == "JPEG"
        assert image is not None

        analyzer = ImageAnalyzer()
        analysis = analyzer.analyze(image)
        assert isinstance(analysis, ImageAnalysisResult)

    def test_chained_operations(self, sample_image_path):
        """Цепочка операций на одном загрузчике."""
        loader = ImageLoader().load(sample_image_path)

        # Получаем изображение и метаданные
        image = loader.get_image()
        meta = loader.get_metadata()
        gray = loader.to_grayscale()
        resized = loader.resize(50, 50)
        cropped = loader.crop(0, 0, 50, 50)

        assert image is not None
        assert meta is not None
        assert gray is not None
        assert resized.shape[:2] == (50, 50)
        assert cropped.shape[:2] == (50, 50)

    def test_scatter_to_dataframe_integration(self, mock_scatter_plot_image):
        """Интеграция extract -> to_dataframe."""
        extractor = ScatterPlotExtractor()
        scatter_data = extractor.extract(mock_scatter_plot_image)
        df = scatter_data.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == scatter_data.n_points
        if scatter_data.n_points > 0:
            assert "x" in df.columns
            assert "y" in df.columns
