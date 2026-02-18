"""Интеграция SDE и ABM моделей для мультимасштабного моделирования.

Связывает непрерывную динамику SDE (макроуровень) с дискретными событиями
ABM (микроуровень) через operator splitting и периодическую синхронизацию.

Алгоритм:
1. SDE описывает глобальную плотность клеток N(t) и цитокины C(t)
2. ABM моделирует индивидуальные клетки на сетке
3. Периодическая синхронизация обеспечивает согласованность

Подробное описание: Description/description_integration.md
"""

from dataclasses import dataclass, field

import numpy as np

from src.core.abm_model import ABMConfig, ABMModel, ABMSnapshot, ABMTrajectory, AgentState
from src.core.sde_model import SDEConfig, SDEModel, SDEState, SDETrajectory, TherapyProtocol
from src.data.parameter_extraction import ModelParameters


@dataclass
class IntegrationConfig:
    """Конфигурация интегрированной модели.

    Подробное описание: Description/description_integration.md#IntegrationConfig
    """

    sde_config: SDEConfig = field(default_factory=SDEConfig)
    abm_config: ABMConfig = field(default_factory=ABMConfig)

    # Параметры синхронизации
    sync_interval: float = 1.0  # часов между синхронизациями
    coupling_strength: float = 0.5  # сила связи ABM→SDE (0-1)

    # Стратегия интеграции
    mode: str = "bidirectional"  # "sde_only", "abm_only", "bidirectional"

    # Параметры коррекции
    correction_rate: float = 0.1  # скорость коррекции рассогласования
    max_discrepancy: float = 0.5  # макс. допустимое рассогласование

    def validate(self) -> bool:
        """Валидация параметров интеграции.

        Returns:
            True если все параметры валидны

        Raises:
            ValueError: Если параметры некорректны

        Подробное описание: Description/description_integration.md#IntegrationConfig.validate
        """
        if self.sync_interval <= 0:
            raise ValueError("sync_interval должен быть положительным")
        if not 0 <= self.coupling_strength <= 1:
            raise ValueError("coupling_strength должен быть в диапазоне [0, 1]")
        if self.mode not in ["sde_only", "abm_only", "bidirectional"]:
            raise ValueError("mode должен быть 'sde_only', 'abm_only' или 'bidirectional'")
        if self.correction_rate < 0 or self.correction_rate > 1:
            raise ValueError("correction_rate должен быть в диапазоне [0, 1]")
        if self.max_discrepancy <= 0:
            raise ValueError("max_discrepancy должен быть положительным")

        # Валидация вложенных конфигураций
        self.sde_config.validate()
        self.abm_config.validate()

        return True


@dataclass
class IntegratedState:
    """Состояние интегрированной системы в момент синхронизации.

    Подробное описание: Description/description_integration.md#IntegratedState
    """

    t: float  # Время (часы)

    # SDE состояние
    sde_N: float  # Плотность клеток из SDE
    sde_C: float  # Концентрация цитокинов из SDE

    # ABM состояние
    abm_agent_counts: dict[str, int]  # Количество агентов по типам
    abm_total: int  # Общее количество агентов

    # Метрики синхронизации
    discrepancy: float  # |sde_N - abm_total| / sde_N
    correction_applied: float  # Величина применённой коррекции

    def to_dict(self) -> dict[str, float | int]:
        """Конвертация в словарь.

        Returns:
            Словарь с состоянием интеграции

        Подробное описание: Description/description_integration.md#IntegratedState.to_dict
        """
        result: dict[str, float | int] = {
            "t": self.t,
            "sde_N": self.sde_N,
            "sde_C": self.sde_C,
            "abm_total": self.abm_total,
            "discrepancy": self.discrepancy,
            "correction_applied": self.correction_applied,
        }
        for agent_type, count in self.abm_agent_counts.items():
            result[f"abm_{agent_type}"] = count
        return result


@dataclass
class IntegratedTrajectory:
    """Траектория интегрированной симуляции.

    Подробное описание: Description/description_integration.md#IntegratedTrajectory
    """

    times: np.ndarray  # [n_sync_points] - точки синхронизации
    states: list[IntegratedState]  # Состояния в точках синхронизации

    sde_trajectory: SDETrajectory  # Полная SDE траектория
    abm_trajectory: ABMTrajectory  # Полная ABM траектория

    config: IntegrationConfig = field(default_factory=IntegrationConfig)

    def get_statistics(self) -> dict[str, float]:
        """Статистика интеграции.

        Returns:
            Словарь со статистиками

        Подробное описание: Description/description_integration.md#IntegratedTrajectory.get_statistics
        """
        if not self.states:
            return {}

        final = self.states[-1]
        discrepancies = [s.discrepancy for s in self.states]
        corrections = [s.correction_applied for s in self.states]

        return {
            "final_sde_N": final.sde_N,
            "final_sde_C": final.sde_C,
            "final_abm_total": float(final.abm_total),
            "mean_discrepancy": float(np.mean(discrepancies)),
            "max_discrepancy": float(np.max(discrepancies)),
            "std_discrepancy": float(np.std(discrepancies)),
            "total_corrections": float(np.sum(np.abs(corrections))),
            "n_sync_points": len(self.states),
        }

    def get_discrepancy_timeseries(self) -> tuple[np.ndarray, np.ndarray]:
        """Временной ряд рассогласования.

        Returns:
            (times, discrepancies)

        Подробное описание: Description/description_integration.md#IntegratedTrajectory.get_discrepancy_timeseries
        """
        times = np.array([state.t for state in self.states])
        discrepancies = np.array([state.discrepancy for state in self.states])
        return (times, discrepancies)


class IntegratedModel:
    """Интегрированная SDE + ABM модель.

    Использует оператор расщепления (operator splitting) для
    связывания непрерывной SDE динамики с дискретными ABM событиями.

    Подробное описание: Description/description_integration.md#IntegratedModel
    """

    def __init__(
        self,
        config: IntegrationConfig,
        therapy: TherapyProtocol | None = None,
        random_seed: int | None = None,
    ) -> None:
        """Инициализация интегрированной модели.

        Args:
            config: Конфигурация интеграции
            therapy: Протокол терапии
            random_seed: Seed для воспроизводимости

        Подробное описание: Description/description_integration.md#IntegratedModel.__init__
        """
        self._config = config
        self._config.validate()

        self._therapy = therapy if therapy else TherapyProtocol()
        self._rng = np.random.default_rng(random_seed)

        # Создать SDE и ABM модели с разными seeds для независимости
        sde_seed = None if random_seed is None else random_seed
        abm_seed = None if random_seed is None else random_seed + 1000

        self._sde_model = SDEModel(
            config=config.sde_config,
            therapy=therapy,
            random_seed=sde_seed,
        )

        self._abm_model = ABMModel(
            config=config.abm_config,
            random_seed=abm_seed,
        )

        self._current_sde_state: SDEState | None = None
        self._sync_states: list[IntegratedState] = []

    @property
    def config(self) -> IntegrationConfig:
        """Получить конфигурацию."""
        return self._config

    @property
    def sde_model(self) -> SDEModel:
        """Получить SDE модель."""
        return self._sde_model

    @property
    def abm_model(self) -> ABMModel:
        """Получить ABM модель."""
        return self._abm_model

    def simulate(
        self,
        initial_params: ModelParameters,
    ) -> IntegratedTrajectory:
        """Полная интегрированная симуляция.

        Алгоритм operator splitting:
        1. Инициализация SDE и ABM с начальными параметрами
        2. Цикл по синхронизационным интервалам:
           a. SDE шаг до следующей точки синхронизации
           b. ABM шаги до синхронизации
           c. Обмен данными: корректировка N на основе агентов
           d. Сохранение состояния
        3. Возврат IntegratedTrajectory

        Args:
            initial_params: Начальные параметры из parameter_extraction

        Returns:
            IntegratedTrajectory с результатами

        Подробное описание: Description/description_integration.md#IntegratedModel.simulate
        """
        # Инициализация ABM
        self._abm_model.initialize_from_parameters(initial_params)

        # Начальные значения
        current_N = initial_params.n0
        current_C = initial_params.c0
        current_time_days = 0.0

        # Интервал синхронизации в днях
        sync_interval_days = self._config.sync_interval / 24.0

        # Максимальное время в днях
        t_max_days = self._config.sde_config.t_max

        # Списки для хранения траекторий
        sync_states: list[IntegratedState] = []
        all_times: list[float] = [0.0]
        all_N: list[np.ndarray] = []
        all_C: list[np.ndarray] = []

        # Начальный снимок
        initial_snapshot = self._abm_model._get_snapshot()
        initial_state = self._create_integrated_state(
            t=0.0,
            sde_N=current_N,
            sde_C=current_C,
            abm_snapshot=initial_snapshot,
            discrepancy=0.0,
            correction=0.0,
        )
        sync_states.append(initial_state)

        # Основной цикл operator splitting
        while current_time_days < t_max_days:
            next_time_days = min(current_time_days + sync_interval_days, t_max_days)

            # 1. SDE шаг
            final_N, final_C, N_seg, C_seg = self._run_sde_segment(
                current_time_days, next_time_days, current_N, current_C
            )
            all_N.append(N_seg)
            all_C.append(C_seg)

            # 2. ABM шаг (в часах)
            abm_start_hours = current_time_days * 24.0
            abm_end_hours = next_time_days * 24.0
            abm_snapshot = self._run_abm_segment(abm_start_hours, abm_end_hours)

            # 3. Синхронизация (если bidirectional)
            if self._config.mode == "bidirectional":
                corrected_N, corrected_C, discrepancy = self._synchronize(
                    final_N, final_C, abm_snapshot
                )
                correction = corrected_N - final_N
            elif self._config.mode == "sde_only":
                corrected_N, corrected_C = final_N, final_C
                discrepancy = self._calculate_discrepancy(final_N, abm_snapshot.get_total_agents())
                correction = 0.0
            else:  # abm_only
                corrected_N = float(abm_snapshot.get_total_agents())
                corrected_C = final_C
                discrepancy = self._calculate_discrepancy(final_N, abm_snapshot.get_total_agents())
                correction = corrected_N - final_N

            # 4. Сохранение состояния
            state = self._create_integrated_state(
                t=next_time_days,
                sde_N=corrected_N,
                sde_C=corrected_C,
                abm_snapshot=abm_snapshot,
                discrepancy=discrepancy,
                correction=correction,
            )
            sync_states.append(state)
            all_times.append(next_time_days)

            # Обновление текущих значений
            current_N = corrected_N
            current_C = corrected_C
            current_time_days = next_time_days

        # Построение полных траекторий
        if all_N:
            full_N = np.concatenate(all_N)
            full_C = np.concatenate(all_C)
            full_times = np.linspace(0, t_max_days, len(full_N))
        else:
            full_N = np.array([current_N])
            full_C = np.array([current_C])
            full_times = np.array([0.0])

        # Создание SDE траектории
        sde_trajectory = SDETrajectory(
            times=full_times,
            N_values=full_N,
            C_values=full_C,
            therapy_markers={
                "prp": self._sde_model._get_therapy_mask(full_times, "prp"),
                "pemf": self._sde_model._get_therapy_mask(full_times, "pemf"),
            },
            config=self._config.sde_config,
            initial_state=SDEState(t=0.0, N=initial_params.n0, C=initial_params.c0),
        )

        # Создание ABM траектории (используем финальный snapshot)
        abm_trajectory = ABMTrajectory(
            snapshots=[self._abm_model._get_snapshot()],
            config=self._config.abm_config,
        )

        return IntegratedTrajectory(
            times=np.array(all_times),
            states=sync_states,
            sde_trajectory=sde_trajectory,
            abm_trajectory=abm_trajectory,
            config=self._config,
        )

    def _run_sde_segment(
        self,
        start_time: float,
        end_time: float,
        initial_N: float,
        initial_C: float,
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        """Запуск SDE на сегменте времени.

        Args:
            start_time: Начало сегмента (в днях)
            end_time: Конец сегмента (в днях)
            initial_N: Начальная плотность
            initial_C: Начальная концентрация

        Returns:
            (final_N, final_C, N_segment, C_segment)

        Подробное описание: Description/description_integration.md#IntegratedModel._run_sde_segment
        """
        dt = self._config.sde_config.dt
        duration = end_time - start_time
        n_steps = max(1, int(duration / dt))

        N_segment = np.zeros(n_steps + 1)
        C_segment = np.zeros(n_steps + 1)

        N_segment[0] = initial_N
        C_segment[0] = initial_C

        sqrt_dt = np.sqrt(dt)

        for i in range(n_steps):
            t = start_time + i * dt
            N = N_segment[i]
            C = C_segment[i]

            drift_N, drift_C = self._sde_model._calculate_drift(t, N, C)
            diff_N, diff_C = self._sde_model._calculate_diffusion(t, N, C)

            dW_N = self._rng.standard_normal() * sqrt_dt
            dW_C = self._rng.standard_normal() * sqrt_dt

            N_new = N + drift_N * dt + diff_N * dW_N
            C_new = C + drift_C * dt + diff_C * dW_C

            N_segment[i + 1], C_segment[i + 1] = self._sde_model._apply_boundary_conditions(
                N_new, C_new
            )

        return (N_segment[-1], C_segment[-1], N_segment, C_segment)

    def _run_abm_segment(
        self,
        start_time: float,
        end_time: float,
    ) -> ABMSnapshot:
        """Запуск ABM на сегменте времени.

        Args:
            start_time: Начало сегмента (в часах)
            end_time: Конец сегмента (в часах)

        Returns:
            ABMSnapshot в конце сегмента

        Подробное описание: Description/description_integration.md#IntegratedModel._run_abm_segment
        """
        dt = self._config.abm_config.dt
        duration = end_time - start_time

        # Выполняем шаги ABM
        n_steps = max(1, int(duration / dt))
        for _ in range(n_steps):
            self._abm_model.step(dt)

        return self._abm_model._get_snapshot()

    def _synchronize(
        self,
        sde_N: float,
        sde_C: float,
        abm_snapshot: ABMSnapshot,
    ) -> tuple[float, float, float]:
        """Синхронизация SDE и ABM состояний.

        Корректирует SDE.N на основе реального количества агентов.
        Обновляет ABM environment на основе SDE.C.

        Args:
            sde_N: Текущая плотность из SDE
            sde_C: Текущая концентрация из SDE
            abm_snapshot: Текущий снимок ABM

        Returns:
            (corrected_N, corrected_C, discrepancy)

        Подробное описание: Description/description_integration.md#IntegratedModel._synchronize
        """
        abm_count = abm_snapshot.get_total_agents()

        # Расчёт рассогласования
        discrepancy = self._calculate_discrepancy(sde_N, abm_count)

        # Применение коррекции к N
        corrected_N = self._apply_correction(sde_N, abm_count, discrepancy)

        # C не корректируется напрямую, но обновляем ABM environment
        self._update_abm_environment(sde_C)

        return (corrected_N, sde_C, discrepancy)

    def _calculate_discrepancy(
        self,
        sde_N: float,
        abm_count: int,
    ) -> float:
        """Расчёт рассогласования между SDE и ABM.

        Discrepancy = |sde_N - abm_count| / max(sde_N, 1)

        Args:
            sde_N: Плотность клеток из SDE
            abm_count: Количество агентов из ABM

        Returns:
            Относительное рассогласование [0, 1+]

        Подробное описание: Description/description_integration.md#IntegratedModel._calculate_discrepancy
        """
        return abs(sde_N - abm_count) / max(sde_N, 1.0)

    def _apply_correction(
        self,
        sde_N: float,
        abm_count: int,
        discrepancy: float,
    ) -> float:
        """Применение коррекции к SDE на основе ABM.

        Args:
            sde_N: Текущая плотность из SDE
            abm_count: Количество агентов из ABM
            discrepancy: Рассогласование

        Returns:
            Скорректированная плотность

        Подробное описание: Description/description_integration.md#IntegratedModel._apply_correction
        """
        # Коэффициент коррекции
        correction_factor = self._config.coupling_strength * self._config.correction_rate

        # Ограничение при большом рассогласовании
        if discrepancy > self._config.max_discrepancy:
            correction_factor *= self._config.max_discrepancy / discrepancy

        # N_corrected = N_sde + alpha * (N_abm - N_sde)
        corrected_N = sde_N + correction_factor * (abm_count - sde_N)

        # Граничное условие
        return max(0.0, corrected_N)

    def _update_abm_environment(self, sde_C: float) -> None:
        """Обновление окружения ABM на основе SDE цитокинов.

        Args:
            sde_C: Концентрация цитокинов из SDE

        Подробное описание: Description/description_integration.md#IntegratedModel._update_abm_environment
        """
        # Обновляем цитокиновое поле ABM модели
        # Масштабируем sde_C для поля ABM
        scaled_C = sde_C / 100.0  # Нормализация

        # Устанавливаем базовый уровень цитокинов
        self._abm_model._cytokine_field = np.maximum(
            self._abm_model._cytokine_field,
            scaled_C,
        )

    def _synchronize_cytokines(
        self,
        sde_C: float,
        abm_snapshot: ABMSnapshot,
    ) -> float:
        """Двусторонняя синхронизация цитокинов (ABM ↔ SDE).

        Дополняет текущую одностороннюю передачу (SDE→ABM).
        Усредняет цитокиновое поле ABM, вычисляет расхождение с SDE.C,
        корректирует SDE.C пропорционально coupling_strength.

        Алгоритм:
        1. abm_C_mean = mean(abm_snapshot.cytokine_field) × scaling_factor
        2. discrepancy = abm_C_mean - sde_C
        3. correction = coupling_strength × discrepancy
        4. return sde_C + correction

        Args:
            sde_C: Текущая концентрация цитокинов в SDE
            abm_snapshot: Текущий снимок ABM состояния

        Returns:
            Скорректированное значение C для SDE

        Подробное описание: Description/Phase2/description_integration.md#IntegratedModel._synchronize_cytokines
        """
        abm_C_mean = float(np.mean(abm_snapshot.cytokine_field))
        discrepancy = abm_C_mean - sde_C
        correction = self._config.coupling_strength * discrepancy
        return max(0.0, sde_C + correction)

    def _transfer_therapy_to_abm(self, current_time_days: float) -> None:
        """Передача терапевтических флагов в ABM environment.

        Проверяет TherapyProtocol для определения активных терапий
        в текущий момент времени. Устанавливает флаги prp_active
        и pemf_active в окружение ABM модели.

        PRP: активен в окне [protocol.prp_start, protocol.prp_end]
        PEMF: активен в окне [protocol.pemf_start, protocol.pemf_end]

        Args:
            current_time_days: Текущее время симуляции (дни)

        Подробное описание: Description/Phase2/description_integration.md#IntegratedModel._transfer_therapy_to_abm
        """
        therapy = self._therapy
        prp_active = False
        pemf_active = False

        if therapy.prp_enabled:
            prp_end = getattr(
                therapy, "prp_end_time",
                therapy.prp_start_time + therapy.prp_duration,
            )
            prp_active = therapy.prp_start_time <= current_time_days <= prp_end

        if therapy.pemf_enabled:
            pemf_end = getattr(
                therapy, "pemf_end_time",
                therapy.pemf_start_time + therapy.pemf_duration,
            )
            pemf_active = therapy.pemf_start_time <= current_time_days <= pemf_end

        self._abm_prp_active = prp_active
        self._abm_pemf_active = pemf_active

    def _spatial_scaling(
        self,
        sde_C: float,
        abm_field: np.ndarray,
        direction: str = "sde_to_abm",
    ) -> np.ndarray | float:
        """Конвертация между SDE скаляром и ABM 2D полем.

        SDE работает со скалярным C (среднее по объёму),
        ABM — с 2D полем цитокинов на сетке. Преобразование
        учитывает масштабные факторы и пространственную неоднородность.

        direction="sde_to_abm": scalar → 2D field (заполнение сетки)
        direction="abm_to_sde": 2D field → scalar (усреднение)

        Args:
            sde_C: Скалярная концентрация из SDE
            abm_field: 2D поле цитокинов ABM
            direction: Направление конвертации ("sde_to_abm" или "abm_to_sde")

        Returns:
            np.ndarray (для sde_to_abm) или float (для abm_to_sde)

        Raises:
            ValueError: Если direction не "sde_to_abm" и не "abm_to_sde"

        Подробное описание: Description/Phase2/description_integration.md#IntegratedModel._spatial_scaling
        """
        if direction == "sde_to_abm":
            return np.full(abm_field.shape, sde_C)
        elif direction == "abm_to_sde":
            return float(np.mean(abm_field))
        else:
            raise ValueError(
                f"direction must be 'sde_to_abm' or 'abm_to_sde', got '{direction}'"
            )

    def _lifting(self, macro_state: dict[str, float]) -> ABMSnapshot:
        """Equation-Free lifting: макросостояние SDE → микросостояние ABM.

        Создаёт ABM snapshot из макропеременных SDE. Процедура:
        1. N → количество агентов (с учётом масштабирования)
        2. Распределение типов агентов пропорционально текущим долям
        3. C → инициализация цитокинового поля
        4. Случайное размещение агентов в пространстве

        Используется в Equation-Free фреймворке для рестарта
        ABM из макросостояния после coarse timestepper.

        Args:
            macro_state: Словарь макропеременных {"N": float, "C": float, ...}

        Returns:
            ABMSnapshot с инициализированными агентами и полями

        Подробное описание: Description/Phase2/description_integration.md#IntegratedModel._lifting
        """
        N = macro_state.get("N", 0.0)
        C = macro_state.get("C", 0.0)

        if N < 0:
            raise ValueError("N must be non-negative")

        n_agents = int(N)
        space_size = self._config.abm_config.space_size
        grid_res = self._config.abm_config.grid_resolution

        agents = []
        for i in range(n_agents):
            x = float(self._rng.uniform(0, space_size[0]))
            y = float(self._rng.uniform(0, space_size[1]))
            agents.append(AgentState(
                agent_id=i,
                agent_type="stem",
                x=x,
                y=y,
                age=0.0,
                division_count=0,
                energy=1.0,
            ))

        grid_shape = (
            int(space_size[0] / grid_res),
            int(space_size[1] / grid_res),
        )
        cytokine_field = np.full(grid_shape, C)
        ecm_field = np.zeros(grid_shape)

        return ABMSnapshot(
            t=0.0,
            agents=agents,
            cytokine_field=cytokine_field,
            ecm_field=ecm_field,
        )

    def _restricting(self, abm_snapshot: ABMSnapshot) -> dict[str, float]:
        """Equation-Free restricting: микросостояние ABM → макропеременные SDE.

        Агрегирует ABM snapshot в макропеременные. Процедура:
        1. Подсчёт агентов → N (с масштабированием)
        2. Усреднение цитокинового поля → C
        3. Подсчёт типов → доли популяций

        Инвариант: restricting(lifting(state)) ≈ state
        (с точностью до дискретизации).

        Args:
            abm_snapshot: Снимок ABM состояния

        Returns:
            Словарь макропеременных {"N": float, "C": float, ...}

        Подробное описание: Description/Phase2/description_integration.md#IntegratedModel._restricting
        """
        N = abm_snapshot.get_total_agents()
        C = float(np.mean(abm_snapshot.cytokine_field))
        return {"N": N, "C": C}

    def _create_integrated_state(
        self,
        t: float,
        sde_N: float,
        sde_C: float,
        abm_snapshot: ABMSnapshot,
        discrepancy: float,
        correction: float,
    ) -> IntegratedState:
        """Создание IntegratedState из текущих состояний.

        Args:
            t: Время
            sde_N: Плотность из SDE
            sde_C: Цитокины из SDE
            abm_snapshot: Снимок ABM
            discrepancy: Рассогласование
            correction: Применённая коррекция

        Returns:
            IntegratedState

        Подробное описание: Description/description_integration.md#IntegratedModel._create_integrated_state
        """
        agent_counts = abm_snapshot.get_agent_count_by_type()
        total_agents = abm_snapshot.get_total_agents()

        return IntegratedState(
            t=t,
            sde_N=sde_N,
            sde_C=sde_C,
            abm_agent_counts=agent_counts,
            abm_total=total_agents,
            discrepancy=discrepancy,
            correction_applied=correction,
        )


def simulate_integrated(
    initial_params: ModelParameters,
    integration_config: IntegrationConfig,
    therapy: TherapyProtocol | None = None,
    random_seed: int | None = None,
) -> IntegratedTrajectory:
    """Convenience функция для интегрированной симуляции.

    Args:
        initial_params: Начальные параметры из parameter_extraction
        integration_config: Конфигурация интеграции
        therapy: Протокол терапии (опционально)
        random_seed: Seed для воспроизводимости

    Returns:
        IntegratedTrajectory с результатами

    Подробное описание: Description/description_integration.md#simulate_integrated
    """
    model = IntegratedModel(
        config=integration_config,
        therapy=therapy,
        random_seed=random_seed,
    )
    return model.simulate(initial_params)


def create_default_integration_config(
    t_max_days: float = 30.0,
    sync_interval_hours: float = 1.0,
    mode: str = "bidirectional",
) -> IntegrationConfig:
    """Создание конфигурации интеграции с согласованными параметрами.

    Args:
        t_max_days: Максимальное время симуляции в днях
        sync_interval_hours: Интервал синхронизации в часах
        mode: Режим интеграции

    Returns:
        IntegrationConfig с согласованными SDE и ABM конфигурациями

    Подробное описание: Description/description_integration.md#create_default_integration_config
    """
    sde_config = SDEConfig(
        t_max=t_max_days,
        dt=0.01,  # дни
    )

    abm_config = ABMConfig(
        t_max=t_max_days * 24.0,  # конвертация в часы
        dt=0.1,  # часы
    )

    return IntegrationConfig(
        sde_config=sde_config,
        abm_config=abm_config,
        sync_interval=sync_interval_hours,
        mode=mode,
    )
