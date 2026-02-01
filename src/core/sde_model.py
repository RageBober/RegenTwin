"""Стохастические дифференциальные уравнения для регенерации тканей.

Модель роста клеточных популяций с учётом:
- Логистического роста с carrying capacity
- Влияния PRP (Platelet-Rich Plasma) терапии
- Влияния PEMF (Pulsed Electromagnetic Field) терапии
- Динамики цитокинов

Численное интегрирование методом Эйлера-Маруямы.

Подробное описание: Description/description_sde_model.md
"""

from dataclasses import dataclass, field

import numpy as np

from src.data.parameter_extraction import ModelParameters


@dataclass
class SDEConfig:
    """Конфигурация SDE модели.

    Подробное описание: Description/description_sde_model.md#SDEConfig
    """

    # Параметры роста
    r: float = 0.3  # Скорость пролиферации (1/день)
    K: float = 1e6  # Carrying capacity (клеток/см²)
    delta: float = 0.05  # Скорость естественной гибели (1/день)

    # Параметры стохастичности
    sigma_n: float = 0.05  # Волатильность плотности клеток
    sigma_c: float = 0.02  # Волатильность цитокинов

    # Параметры цитокинов
    gamma: float = 0.5  # Скорость деградации цитокинов (1/день)
    eta: float = 0.001  # Производство цитокинов клетками (нг/мл на клетку)

    # Терапевтические коэффициенты
    alpha_prp: float = 0.5  # Коэффициент PRP эффекта (0.1-1.0)
    beta_pemf: float = 0.1  # Коэффициент PEMF эффекта (0.01-0.1)
    lambda_prp: float = 0.3  # Скорость затухания PRP (1/день)
    f0_pemf: float = 50.0  # Оптимальная частота PEMF (Гц)
    k_pemf: float = 0.1  # Крутизна сигмоиды PEMF

    # Численные параметры
    dt: float = 0.01  # Шаг времени (дни)
    t_max: float = 30.0  # Максимальное время симуляции (дни)

    def validate(self) -> bool:
        """Валидация физической осмысленности параметров.

        Returns:
            True если все параметры валидны

        Raises:
            ValueError: Если параметры некорректны

        Подробное описание: Description/description_sde_model.md#validate
        """
        if self.r <= 0:
            raise ValueError("r (скорость роста) должна быть положительной")
        if self.K <= 0:
            raise ValueError("K (carrying capacity) должна быть положительной")
        if self.delta < 0:
            raise ValueError("delta (скорость гибели) должна быть неотрицательной")
        if self.sigma_n < 0 or self.sigma_c < 0:
            raise ValueError("sigma должны быть неотрицательными")
        if self.gamma <= 0:
            raise ValueError("gamma должна быть положительной")
        if self.eta < 0:
            raise ValueError("eta должна быть неотрицательной")
        if self.alpha_prp < 0 or self.beta_pemf < 0:
            raise ValueError("alpha_prp и beta_pemf должны быть неотрицательными")
        if self.dt <= 0 or self.dt > 1.0:
            raise ValueError("dt должен быть в диапазоне (0, 1.0]")
        if self.t_max <= 0:
            raise ValueError("t_max должен быть положительным")
        return True


@dataclass
class TherapyProtocol:
    """Протокол терапевтического вмешательства.

    Подробное описание: Description/description_sde_model.md#TherapyProtocol
    """

    # PRP терапия
    prp_enabled: bool = False
    prp_start_time: float = 0.0  # дни
    prp_duration: float = 7.0  # дни
    prp_intensity: float = 1.0  # множитель интенсивности (0-2)
    prp_initial_concentration: float = 10.0  # нг/мл начальная концентрация

    # PEMF терапия
    pemf_enabled: bool = False
    pemf_start_time: float = 0.0  # дни
    pemf_duration: float = 14.0  # дни
    pemf_frequency: float = 50.0  # Hz
    pemf_intensity: float = 1.0  # множитель интенсивности (0-2)

    # Комбинированная терапия
    synergy_factor: float = 1.2  # Синергетический эффект при совместном применении


@dataclass
class SDEState:
    """Состояние SDE системы в момент времени.

    Подробное описание: Description/description_sde_model.md#SDEState
    """

    t: float  # Время (дни)
    N: float  # Плотность клеток (клеток/см²)
    C: float  # Концентрация цитокинов (нг/мл)
    prp_active: bool = False
    pemf_active: bool = False

    def to_dict(self) -> dict[str, float | bool]:
        """Конвертация в словарь.

        Returns:
            Словарь с состоянием системы

        Подробное описание: Description/description_sde_model.md#SDEState.to_dict
        """
        return {
            "t": self.t,
            "N": self.N,
            "C": self.C,
            "prp_active": self.prp_active,
            "pemf_active": self.pemf_active,
        }


@dataclass
class SDETrajectory:
    """Траектория SDE симуляции.

    Подробное описание: Description/description_sde_model.md#SDETrajectory
    """

    times: np.ndarray  # [n_steps] - временные точки
    N_values: np.ndarray  # [n_steps] - плотность клеток
    C_values: np.ndarray  # [n_steps] - концентрация цитокинов
    therapy_markers: dict[str, np.ndarray] = field(default_factory=dict)  # Boolean маски терапий

    config: SDEConfig = field(default_factory=SDEConfig)
    initial_state: SDEState = field(default_factory=lambda: SDEState(t=0.0, N=0.0, C=0.0))

    def get_final_state(self) -> SDEState:
        """Получить финальное состояние.

        Returns:
            SDEState в конце симуляции

        Подробное описание: Description/description_sde_model.md#SDETrajectory.get_final_state
        """
        prp_active = False
        pemf_active = False
        if self.therapy_markers:
            if "prp" in self.therapy_markers and len(self.therapy_markers["prp"]) > 0:
                prp_active = bool(self.therapy_markers["prp"][-1])
            if "pemf" in self.therapy_markers and len(self.therapy_markers["pemf"]) > 0:
                pemf_active = bool(self.therapy_markers["pemf"][-1])

        return SDEState(
            t=float(self.times[-1]),
            N=float(self.N_values[-1]),
            C=float(self.C_values[-1]),
            prp_active=prp_active,
            pemf_active=pemf_active,
        )

    def get_statistics(self) -> dict[str, float]:
        """Статистика траектории.

        Returns:
            Словарь со статистиками (final_N, final_C, max_N, growth_rate, etc.)

        Подробное описание: Description/description_sde_model.md#SDETrajectory.get_statistics
        """
        final_N = float(self.N_values[-1])
        final_C = float(self.C_values[-1])
        max_N = float(np.max(self.N_values))

        # Расчёт эффективной скорости роста
        t_total = float(self.times[-1] - self.times[0])
        if t_total > 0 and self.N_values[0] > 0:
            growth_rate = (final_N - self.N_values[0]) / t_total
        else:
            growth_rate = 0.0

        return {
            "final_N": final_N,
            "final_C": final_C,
            "max_N": max_N,
            "growth_rate": float(growth_rate),
        }


class SDEModel:
    """Модель стохастических дифференциальных уравнений для регенерации тканей.

    Реализует уравнение Ланжевена:
    dNₜ = [rNₜ(1 - Nₜ/K) + αf(PRP) + βg(PEMF) - δN]dt + σNₜdWₜ

    Использует метод Эйлера-Маруямы для численного интегрирования.

    Подробное описание: Description/description_sde_model.md#SDEModel
    """

    def __init__(
        self,
        config: SDEConfig | None = None,
        therapy: TherapyProtocol | None = None,
        random_seed: int | None = None,
    ) -> None:
        """Инициализация SDE модели.

        Args:
            config: Конфигурация модели
            therapy: Протокол терапии
            random_seed: Seed для воспроизводимости

        Подробное описание: Description/description_sde_model.md#SDEModel.__init__
        """
        self._config = config if config else SDEConfig()
        self._config.validate()

        self._therapy = therapy if therapy else TherapyProtocol()
        self._rng = np.random.default_rng(random_seed)

    @property
    def config(self) -> SDEConfig:
        """Получить конфигурацию модели."""
        return self._config

    @property
    def therapy(self) -> TherapyProtocol:
        """Получить протокол терапии."""
        return self._therapy

    def simulate(
        self,
        initial_params: ModelParameters,
    ) -> SDETrajectory:
        """Полная симуляция SDE методом Эйлера-Маруямы.

        Args:
            initial_params: Начальные параметры из parameter_extraction

        Returns:
            SDETrajectory с результатами

        Подробное описание: Description/description_sde_model.md#SDEModel.simulate
        """
        # Количество шагов и временная ось
        n_steps = int(self._config.t_max / self._config.dt)
        times = np.linspace(0, self._config.t_max, n_steps + 1)

        # Массивы для хранения траектории
        N_values = np.zeros(n_steps + 1)
        C_values = np.zeros(n_steps + 1)

        # Начальные условия
        N_values[0] = initial_params.n0
        C_values[0] = initial_params.c0

        # Параметр для стохастического члена
        sqrt_dt = np.sqrt(self._config.dt)

        # Цикл Эйлера-Маруямы
        for i in range(n_steps):
            t = times[i]
            N = N_values[i]
            C = C_values[i]

            # Расчёт drift и diffusion
            drift_N, drift_C = self._calculate_drift(t, N, C)
            diff_N, diff_C = self._calculate_diffusion(t, N, C)

            # Стохастический инкремент (Винеровский процесс)
            dW_N = self._rng.standard_normal() * sqrt_dt
            dW_C = self._rng.standard_normal() * sqrt_dt

            # Обновление по схеме Эйлера-Маруямы
            N_new = N + drift_N * self._config.dt + diff_N * dW_N
            C_new = C + drift_C * self._config.dt + diff_C * dW_C

            # Применение граничных условий
            N_values[i + 1], C_values[i + 1] = self._apply_boundary_conditions(N_new, C_new)

        # Создание траектории
        return SDETrajectory(
            times=times,
            N_values=N_values,
            C_values=C_values,
            therapy_markers={
                "prp": self._get_therapy_mask(times, "prp"),
                "pemf": self._get_therapy_mask(times, "pemf"),
            },
            config=self._config,
            initial_state=SDEState(
                t=0.0,
                N=initial_params.n0,
                C=initial_params.c0,
                prp_active=self._is_therapy_active(0.0, "prp"),
                pemf_active=self._is_therapy_active(0.0, "pemf"),
            ),
        )

    def _calculate_drift(
        self,
        t: float,
        N: float,
        C: float,
    ) -> tuple[float, float]:
        """Расчёт drift terms μ(N, C, t).

        dN/dt = rN(1 - N/K) + α·f_PRP(t, C) + β·g_PEMF(t) - δN
        dC/dt = η·N - γ·C + PRP_secretion(t)

        Args:
            t: Текущее время
            N: Плотность клеток
            C: Концентрация цитокинов

        Returns:
            (drift_N, drift_C) — скорости изменения

        Подробное описание: Description/description_sde_model.md#SDEModel._calculate_drift
        """
        # Drift для N: логистический рост + терапии - гибель
        drift_N = (
            self._logistic_growth(N)
            + self._prp_effect(t, N, C)
            + self._pemf_effect(t, N)
            - self._config.delta * N
        )

        # Drift для C: производство клетками - деградация + секреция из PRP
        drift_C = (
            self._config.eta * N
            - self._config.gamma * C
            + self._therapy_prp_secretion(t)
        )

        return (drift_N, drift_C)

    def _calculate_diffusion(
        self,
        t: float,
        N: float,
        C: float,
    ) -> tuple[float, float]:
        """Расчёт diffusion terms σ(N, C, t).

        Args:
            t: Текущее время
            N: Плотность клеток
            C: Концентрация цитокинов

        Returns:
            (diffusion_N, diffusion_C)

        Подробное описание: Description/description_sde_model.md#SDEModel._calculate_diffusion
        """
        # Диффузия пропорциональна текущему значению (геометрическое броуновское движение)
        diffusion_N = self._config.sigma_n * max(0.0, N)
        diffusion_C = self._config.sigma_c * max(0.0, C)
        return (diffusion_N, diffusion_C)

    def _logistic_growth(self, N: float) -> float:
        """Логистический рост r·N·(1 - N/K).

        Args:
            N: Плотность клеток

        Returns:
            Скорость логистического роста

        Подробное описание: Description/description_sde_model.md#SDEModel._logistic_growth
        """
        return self._config.r * N * (1.0 - N / self._config.K)

    def _prp_effect(self, t: float, N: float, C: float) -> float:
        """Эффект PRP терапии α·f_PRP(t, C).

        f(PRP) = C₀·e^(-λt) — экспоненциальное затухание
        PRP стимулирует рост через факторы роста (PDGF, VEGF).

        Args:
            t: Текущее время
            N: Плотность клеток
            C: Концентрация цитокинов

        Returns:
            Вклад PRP в скорость роста

        Подробное описание: Description/description_sde_model.md#SDEModel._prp_effect
        """
        if not self._is_therapy_active(t, "prp"):
            return 0.0

        # Время с начала терапии
        t_therapy = t - self._therapy.prp_start_time

        # Экспоненциальное затухание: C0 * e^(-lambda * t)
        effect = (
            self._config.alpha_prp
            * self._therapy.prp_initial_concentration
            * np.exp(-self._config.lambda_prp * t_therapy)
            * self._therapy.prp_intensity
        )

        # Синергия с PEMF
        if self._is_therapy_active(t, "pemf"):
            effect *= self._therapy.synergy_factor

        return float(effect)

    def _pemf_effect(self, t: float, N: float) -> float:
        """Эффект PEMF терапии β·g_PEMF(t).

        g(PEMF) = 1/(1 + e^(-k(f - f₀))) — сигмоидальный отклик на частоту
        PEMF стимулирует пролиферацию через электромагнитное поле.

        Args:
            t: Текущее время
            N: Плотность клеток

        Returns:
            Вклад PEMF в скорость роста

        Подробное описание: Description/description_sde_model.md#SDEModel._pemf_effect
        """
        if not self._is_therapy_active(t, "pemf"):
            return 0.0

        # Сигмоидальный отклик на частоту
        freq_diff = self._therapy.pemf_frequency - self._config.f0_pemf
        sigmoid = 1.0 / (1.0 + np.exp(-self._config.k_pemf * freq_diff))

        # Эффект пропорционален N
        effect = (
            self._config.beta_pemf
            * sigmoid
            * self._therapy.pemf_intensity
            * N
        )

        return float(effect)

    def _therapy_prp_secretion(self, t: float) -> float:
        """Секреция цитокинов из PRP.

        Дополнительный источник цитокинов при PRP терапии.

        Args:
            t: Текущее время

        Returns:
            Скорость секреции цитокинов из PRP

        Подробное описание: Description/description_sde_model.md#SDEModel._therapy_prp_secretion
        """
        if not self._is_therapy_active(t, "prp"):
            return 0.0

        # Время с начала терапии
        t_therapy = t - self._therapy.prp_start_time

        # Секреция цитокинов с экспоненциальным затуханием
        secretion = (
            self._therapy.prp_initial_concentration
            * np.exp(-self._config.lambda_prp * t_therapy)
            * self._therapy.prp_intensity
            * 0.1  # Коэффициент секреции
        )

        return float(secretion)

    def _is_therapy_active(self, t: float, therapy_type: str) -> bool:
        """Проверка активности терапии в момент времени.

        Args:
            t: Текущее время
            therapy_type: Тип терапии ('prp' или 'pemf')

        Returns:
            True если терапия активна

        Подробное описание: Description/description_sde_model.md#SDEModel._is_therapy_active
        """
        if therapy_type == "prp":
            if not self._therapy.prp_enabled:
                return False
            start = self._therapy.prp_start_time
            end = start + self._therapy.prp_duration
            return start <= t < end
        elif therapy_type == "pemf":
            if not self._therapy.pemf_enabled:
                return False
            start = self._therapy.pemf_start_time
            end = start + self._therapy.pemf_duration
            return start <= t < end
        return False

    def _apply_boundary_conditions(self, N: float, C: float) -> tuple[float, float]:
        """Применение граничных условий (отражающая граница N ≥ 0, C ≥ 0).

        Args:
            N: Плотность клеток
            C: Концентрация цитокинов

        Returns:
            (N_bounded, C_bounded)

        Подробное описание: Description/description_sde_model.md#SDEModel._apply_boundary_conditions
        """
        return (max(0.0, N), max(0.0, C))

    def _get_therapy_mask(
        self,
        times: np.ndarray,
        therapy_type: str,
    ) -> np.ndarray:
        """Boolean маска активности терапии.

        Args:
            times: Массив временных точек
            therapy_type: Тип терапии ('prp' или 'pemf')

        Returns:
            Boolean массив активности

        Подробное описание: Description/description_sde_model.md#SDEModel._get_therapy_mask
        """
        return np.array([self._is_therapy_active(t, therapy_type) for t in times])


def simulate_sde(
    initial_params: ModelParameters,
    config: SDEConfig | None = None,
    therapy: TherapyProtocol | None = None,
    random_seed: int | None = None,
) -> SDETrajectory:
    """Convenience функция для SDE симуляции.

    Args:
        initial_params: Начальные параметры из parameter_extraction
        config: Конфигурация модели (опционально)
        therapy: Протокол терапии (опционально)
        random_seed: Seed для воспроизводимости

    Returns:
        SDETrajectory с результатами

    Подробное описание: Description/description_sde_model.md#simulate_sde
    """
    model = SDEModel(config=config, therapy=therapy, random_seed=random_seed)
    return model.simulate(initial_params)
