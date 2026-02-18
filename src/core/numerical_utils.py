"""Утилиты для численной робастности стохастических моделей.

Обеспечивают:
- Отсечение нефизичных значений (отрицательные концентрации)
- Детекцию дивергенции (NaN, Inf) с fallback-стратегией
- Адаптивный шаг времени
- Контекстный менеджер для безопасных вычислений
- Структурированное логирование через Loguru

Подробное описание: Description/Phase2/description_numerical_utils.md
"""

import math
import warnings
from dataclasses import dataclass, field

import numpy as np

try:
    from loguru import logger  # Структурированное логирование предупреждений
except ImportError:
    import logging as _logging

    logger = _logging.getLogger(__name__)  # type: ignore[assignment]


@dataclass
class DivergenceInfo:
    """Информация о детектированной дивергенции в численном решении.

    Содержит диагностику: какие переменные содержат NaN/Inf,
    максимальное значение, текстовое сообщение. Используется
    для принятия решения о fallback (уменьшение dt, остановка).

    Подробное описание: Description/Phase2/description_numerical_utils.md#DivergenceInfo
    """

    has_nan: bool = False  # Есть ли NaN в переменных
    has_inf: bool = False  # Есть ли Inf в переменных
    nan_variables: list[str] = field(default_factory=list)  # Имена NaN-переменных
    inf_variables: list[str] = field(default_factory=list)  # Имена Inf-переменных
    max_value: float = 0.0  # Максимальное абсолютное значение
    message: str = ""  # Диагностическое сообщение

    @property
    def is_diverged(self) -> bool:
        """True если обнаружена дивергенция (NaN или Inf).

        Подробное описание: Description/Phase2/description_numerical_utils.md#DivergenceInfo.is_diverged
        """
        return self.has_nan or self.has_inf


def clip_negative_concentrations(
    state: dict[str, float],
    variables: list[str] | None = None,
    min_value: float = 0.0,
) -> dict[str, float]:
    """Отсечение отрицательных значений концентраций и плотностей.

    Физическое обоснование: концентрации клеток и цитокинов не могут
    быть отрицательными. Метод Эйлера-Маруямы при сильном стохастическом
    шуме может генерировать отрицательные значения — их нужно обрезать.

    Применяется после каждого шага интегрирования SDE:
    state[var] = max(state[var], min_value) для каждой переменной.

    Args:
        state: Словарь переменных состояния {имя: значение}
        variables: Список переменных для отсечения (None = все)
        min_value: Минимальное допустимое значение (обычно 0.0)

    Returns:
        Скорректированный словарь состояния (новый dict)

    Подробное описание: Description/Phase2/description_numerical_utils.md#clip_negative_concentrations
    """
    result = dict(state)
    vars_to_check = variables if variables is not None else list(state.keys())
    clipped_vars: list[str] = []

    for var in vars_to_check:
        if var not in result:
            continue
        val = result[var]
        # NaN пропускаем — math.isnan проверяет, max с NaN не работает корректно
        if isinstance(val, float) and math.isnan(val):
            continue
        if val < min_value:
            clipped_vars.append(var)
            result[var] = min_value

    if clipped_vars:
        logger.warning(
            f"Клиппинг {len(clipped_vars)} переменных: {clipped_vars}"
        )

    return result


def detect_divergence(
    state: dict[str, float],
    max_allowed: float = 1e15,
) -> DivergenceInfo:
    """Детекция дивергенции: NaN, Inf и аномально большие значения.

    Проверяет каждую переменную состояния на:
    - np.isnan() → has_nan, nan_variables
    - np.isinf() → has_inf, inf_variables
    - |value| > max_allowed → is_diverged (через max_value)

    Используется для раннего обнаружения нестабильности и
    принятия решения о fallback (уменьшение dt или остановка).

    Args:
        state: Словарь переменных состояния
        max_allowed: Максимальное допустимое абсолютное значение

    Returns:
        DivergenceInfo с диагностической информацией

    Подробное описание: Description/Phase2/description_numerical_utils.md#detect_divergence
    """
    nan_vars: list[str] = []
    inf_vars: list[str] = []
    max_val = 0.0
    messages: list[str] = []

    for var, val in state.items():
        if np.isnan(val):
            nan_vars.append(var)
        elif np.isinf(val):
            inf_vars.append(var)
        else:
            abs_val = abs(val)
            if abs_val > max_val:
                max_val = abs_val

    has_nan = len(nan_vars) > 0
    has_inf = len(inf_vars) > 0

    if has_nan:
        messages.append(f"NaN в переменных: {nan_vars}")
    if has_inf:
        messages.append(f"Inf в переменных: {inf_vars}")
    if max_val > max_allowed:
        messages.append(f"Overflow: max_value={max_val:.2e} > max_allowed={max_allowed:.2e}")

    message = "; ".join(messages)

    if has_nan or has_inf or max_val > max_allowed:
        logger.warning(f"Дивергенция обнаружена: {message}")

    return DivergenceInfo(
        has_nan=has_nan,
        has_inf=has_inf,
        nan_variables=nan_vars,
        inf_variables=inf_vars,
        max_value=max_val,
        message=message,
    )


def handle_divergence(
    divergence_info: DivergenceInfo,
    state_current: dict[str, float],
    state_previous: dict[str, float],
    dt_current: float,
    dt_min: float = 1e-6,
    max_retries: int = 3,
) -> tuple[dict[str, float], float, bool]:
    """Стратегия реагирования на дивергенцию: остановка шага + fallback.

    При жёсткой дивергенции (NaN/Inf) — откат к state_previous с dt/2.
    При мягкой (overflow) — клиппинг текущего состояния с dt/2.
    Если dt < dt_min — сигнал остановки симуляции.

    Алгоритм:
    1. Логировать через logger.warning/error
    2. NaN/Inf → откат к state_previous, dt = dt/2
    3. Overflow → clip_negative_concentrations(state_current), dt = dt/2
    4. dt < dt_min → should_stop = True

    Args:
        divergence_info: Результат detect_divergence()
        state_current: Текущее (дивергировавшее) состояние
        state_previous: Предыдущее (безопасное) состояние
        dt_current: Текущий шаг времени
        dt_min: Минимальный допустимый шаг
        max_retries: Максимальное количество повторных попыток

    Returns:
        (safe_state, new_dt, should_stop):
        - safe_state: откат или клиппированное состояние
        - new_dt: уменьшенный шаг (≥ dt_min)
        - should_stop: True если дальнейшая симуляция невозможна

    Подробное описание: Description/Phase2/description_numerical_utils.md#handle_divergence
    """
    # Нет дивергенции — ничего не делаем
    if not divergence_info.is_diverged and divergence_info.max_value <= 1e15:
        return dict(state_current), dt_current, False

    new_dt = max(dt_current / 2.0, dt_min)
    should_stop = dt_current / 2.0 < dt_min

    if divergence_info.has_nan or divergence_info.has_inf:
        # Жёсткая дивергенция — откат к предыдущему состоянию
        safe_state = dict(state_previous)
        logger.warning(
            f"Откат к предыдущему состоянию: {divergence_info.message}, "
            f"dt: {dt_current} → {new_dt}"
        )
    else:
        # Мягкая дивергенция (overflow) — клиппинг текущего
        safe_state = clip_negative_concentrations(state_current)
        logger.warning(
            f"Клиппинг текущего состояния: {divergence_info.message}, "
            f"dt: {dt_current} → {new_dt}"
        )

    if should_stop:
        logger.error(
            f"Невозможно продолжить: dt={new_dt} < dt_min={dt_min}. "
            f"Причина: {divergence_info.message}"
        )

    return safe_state, new_dt, should_stop


def adaptive_timestep(
    state_current: dict[str, float],
    state_previous: dict[str, float],
    dt_current: float,
    tolerance: float = 0.1,
    dt_min: float = 1e-6,
    dt_max: float = 1.0,
) -> float:
    """Адаптивный шаг времени на основе скорости изменения состояния.

    Алгоритм:
    1. Вычислить max relative change: max(|x_new - x_old| / max(|x_old|, eps))
    2. Если change > tolerance → уменьшить dt (dt * tolerance / change)
    3. Если change < tolerance/4 → увеличить dt (dt * 2.0)
    4. Ограничить результат: [dt_min, dt_max]

    Используется для SDE/ABM при быстрых переходах (воспаление → разрешение).

    Args:
        state_current: Текущее состояние
        state_previous: Предыдущее состояние
        dt_current: Текущий шаг времени
        tolerance: Допустимое относительное изменение за шаг
        dt_min: Минимальный шаг времени
        dt_max: Максимальный шаг времени

    Returns:
        Новый шаг времени в диапазоне [dt_min, dt_max]

    Подробное описание: Description/Phase2/description_numerical_utils.md#adaptive_timestep
    """
    eps = 1e-30

    # Если нет переменных — максимальный шаг
    if not state_current or not state_previous:
        return min(dt_current * 2.0, dt_max)

    # Вычислить максимальное относительное изменение
    max_change = 0.0
    for key in state_current:
        if key not in state_previous:
            continue
        curr = state_current[key]
        prev = state_previous[key]
        denom = max(abs(prev), eps)
        change = abs(curr - prev) / denom
        if change > max_change:
            max_change = change

    # Адаптация шага
    if max_change < tolerance / 4.0:
        new_dt = dt_current * 2.0
    elif max_change > tolerance:
        new_dt = dt_current * (tolerance / max_change)
    else:
        new_dt = dt_current

    # Зажать в [dt_min, dt_max]
    return float(max(dt_min, min(new_dt, dt_max)))


class NumericalGuard:
    """Контекстный менеджер для безопасных численных операций.

    При входе: сохраняет numpy error settings (seterr).
    В процессе: перехватывает RuntimeWarning (overflow, underflow, invalid).
    При выходе: восстанавливает settings, собирает предупреждения.

    Использование:
        with NumericalGuard() as guard:
            result = risky_computation()
        if guard.had_warnings:
            handle_warnings(guard.warnings)

    Подробное описание: Description/Phase2/description_numerical_utils.md#NumericalGuard
    """

    def __init__(
        self,
        clip_on_overflow: bool = True,
        log_warnings: bool = True,
    ) -> None:
        """Инициализация NumericalGuard.

        Args:
            clip_on_overflow: Автоматически отсекать overflow значения
            log_warnings: Логировать предупреждения через loguru (если доступен)

        Подробное описание: Description/Phase2/description_numerical_utils.md#NumericalGuard.__init__
        """
        self._clip_on_overflow = clip_on_overflow
        self._log_warnings = log_warnings
        self._warnings_list: list[str] = []
        self._old_settings: dict[str, str] | None = None
        self._warning_catcher: warnings.catch_warnings | None = None
        self._caught_warnings: list[warnings.WarningMessage] | None = None

    def __enter__(self) -> "NumericalGuard":
        """Вход в контекст: сохранение numpy error settings, установка фильтров.

        Подробное описание: Description/Phase2/description_numerical_utils.md#NumericalGuard.__enter__
        """
        self._old_settings = np.geterr()
        np.seterr(all="warn")
        self._warning_catcher = warnings.catch_warnings(record=True)
        self._caught_warnings = self._warning_catcher.__enter__()
        warnings.simplefilter("always")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:  # type: ignore[type-arg]
        """Выход: восстановление settings, обработка перехваченных ошибок.

        Подробное описание: Description/Phase2/description_numerical_utils.md#NumericalGuard.__exit__
        """
        # Закрыть catch_warnings
        if self._warning_catcher is not None:
            self._warning_catcher.__exit__(exc_type, exc_val, exc_tb)

        # Собрать предупреждения
        if self._caught_warnings:
            for w in self._caught_warnings:
                if issubclass(w.category, RuntimeWarning):
                    self._warnings_list.append(str(w.message))

        # Восстановить numpy settings
        if self._old_settings is not None:
            np.seterr(**self._old_settings)

        # Логирование
        if self._log_warnings and self._warnings_list:
            logger.warning(
                f"NumericalGuard: {len(self._warnings_list)} предупреждений: "
                f"{self._warnings_list}"
            )

        return False

    @property
    def had_warnings(self) -> bool:
        """True если во время выполнения блока были предупреждения.

        Подробное описание: Description/Phase2/description_numerical_utils.md#NumericalGuard.had_warnings
        """
        return len(self._warnings_list) > 0

    @property
    def warnings(self) -> list[str]:
        """Список текстовых описаний всех перехваченных предупреждений.

        Подробное описание: Description/Phase2/description_numerical_utils.md#NumericalGuard.warnings
        """
        return list(self._warnings_list)
