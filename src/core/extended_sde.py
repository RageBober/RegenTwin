"""Расширенная 20-переменная SDE модель регенерации тканей.

Полная система стохастических дифференциальных уравнений:
- 8 клеточных популяций: P, Ne, M1, M2, F, Mf, E, S
- 7 цитокинов: TNF-α, IL-10, PDGF, VEGF, TGF-β, MCP-1, IL-8
- 3 ECM компонента: коллаген, MMP, фибрин
- 2 вспомогательных: D (damage signal), O₂

Численное интегрирование методом Эйлера-Маруямы.
Математическое обоснование: Doks/RegenTwin_Mathematical_Framework.md §2

Подробное описание: Description/Phase2/description_extended_sde.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from src.core.parameters import ParameterSet
from src.core.sde_model import TherapyProtocol


class StateIndex(IntEnum):
    """Индексы переменных в массиве состояния (20 переменных).

    Определяет порядок переменных в numpy-массиве для drift/diffusion.

    Подробное описание:
        Description/Phase2/description_extended_sde.md#StateIndex
    """

    P = 0            # Тромбоциты
    Ne = 1           # Нейтрофилы
    M1 = 2           # M1 макрофаги
    M2 = 3           # M2 макрофаги
    F = 4            # Фибробласты
    Mf = 5           # Миофибробласты
    E = 6            # Эндотелиальные
    S = 7            # Стволовые (CD34+)
    C_TNF = 8        # TNF-α
    C_IL10 = 9       # IL-10
    C_PDGF = 10      # PDGF
    C_VEGF = 11      # VEGF
    C_TGFb = 12      # TGF-β
    C_MCP1 = 13      # MCP-1
    C_IL8 = 14       # IL-8
    RHO_COLLAGEN = 15  # Плотность коллагена
    C_MMP = 16       # MMP
    RHO_FIBRIN = 17  # Плотность фибрина
    D = 18           # Сигнал повреждения
    O2 = 19          # Кислород


VARIABLE_NAMES: list[str] = [
    "P", "Ne", "M1", "M2", "F", "Mf", "E", "S",
    "C_TNF", "C_IL10", "C_PDGF", "C_VEGF", "C_TGFb",
    "C_MCP1", "C_IL8",
    "rho_collagen", "C_MMP", "rho_fibrin",
    "D", "O2",
]
"""Имена всех 20 переменных в порядке StateIndex."""

N_VARIABLES: int = 20
"""Общее число переменных в расширенной SDE системе."""


@dataclass
class ExtendedSDEState:
    """Состояние 20-переменной SDE системы в момент времени t.

    Содержит все клеточные популяции, цитокины, ECM компоненты
    и вспомогательные переменные.

    Подробное описание:
        Description/Phase2/description_extended_sde.md#ExtendedSDEState
    """

    # Клеточные популяции (клеток/мкл)
    P: float = 0.0    # Тромбоциты
    Ne: float = 0.0   # Нейтрофилы
    M1: float = 0.0   # M1 макрофаги (провоспалительные)
    M2: float = 0.0   # M2 макрофаги (репаративные)
    F: float = 0.0    # Фибробласты
    Mf: float = 0.0   # Миофибробласты
    E: float = 0.0    # Эндотелиальные клетки
    S: float = 0.0    # Стволовые/прогениторные (CD34+)

    # Цитокины (нг/мл)
    C_TNF: float = 0.0   # TNF-α
    C_IL10: float = 0.0  # IL-10
    C_PDGF: float = 0.0  # PDGF
    C_VEGF: float = 0.0  # VEGF
    C_TGFb: float = 0.0  # TGF-β
    C_MCP1: float = 0.0  # MCP-1
    C_IL8: float = 0.0   # IL-8

    # ECM
    rho_collagen: float = 0.0  # Плотность коллагена
    C_MMP: float = 0.0         # Концентрация MMP
    rho_fibrin: float = 0.0    # Плотность фибрина

    # Вспомогательные
    D: float = 0.0   # Сигнал повреждения (DAMPs)
    O2: float = 0.0  # Кислород (mmHg)

    # Время
    t: float = 0.0  # Текущее время (ч)

    def to_array(self) -> np.ndarray:
        """Конвертация состояния в numpy массив (20 элементов).

        Порядок элементов соответствует StateIndex:
        [P, Ne, M1, M2, F, Mf, E, S, C_TNF, C_IL10, C_PDGF,
         C_VEGF, C_TGFb, C_MCP1, C_IL8, rho_collagen, C_MMP,
         rho_fibrin, D, O2]

        Returns:
            np.ndarray shape (20,) с текущими значениями

        Подробное описание:
            Description/Phase2/description_extended_sde.md#to_array
        """
        return np.array([
            self.P, self.Ne, self.M1, self.M2, self.F, self.Mf, self.E,
            self.S, self.C_TNF, self.C_IL10, self.C_PDGF, self.C_VEGF,
            self.C_TGFb, self.C_MCP1, self.C_IL8, self.rho_collagen,
            self.C_MMP, self.rho_fibrin, self.D, self.O2,
        ])

    @classmethod
    def from_array(
        cls, arr: np.ndarray, t: float = 0.0,
    ) -> ExtendedSDEState:
        """Создание состояния из numpy массива.

        Args:
            arr: Массив shape (20,) в порядке StateIndex
            t: Текущее время (ч)

        Returns:
            Новый ExtendedSDEState

        Raises:
            ValueError: Если len(arr) != 20

        Подробное описание:
            Description/Phase2/description_extended_sde.md#from_array
        """
        if len(arr) != N_VARIABLES:
            raise ValueError(
                f"Ожидался массив длины {N_VARIABLES}, получен {len(arr)}"
            )
        return cls(
            P=float(arr[0]), Ne=float(arr[1]),
            M1=float(arr[2]), M2=float(arr[3]),
            F=float(arr[4]), Mf=float(arr[5]),
            E=float(arr[6]), S=float(arr[7]),
            C_TNF=float(arr[8]), C_IL10=float(arr[9]),
            C_PDGF=float(arr[10]), C_VEGF=float(arr[11]),
            C_TGFb=float(arr[12]), C_MCP1=float(arr[13]),
            C_IL8=float(arr[14]), rho_collagen=float(arr[15]),
            C_MMP=float(arr[16]), rho_fibrin=float(arr[17]),
            D=float(arr[18]), O2=float(arr[19]),
            t=t,
        )

    def to_dict(self) -> dict[str, float]:
        """Конвертация в словарь {имя_переменной: значение}.

        Включает все 20 переменных и время t.

        Returns:
            Словарь с 21 ключом (20 переменных + t)

        Подробное описание:
            Description/Phase2/description_extended_sde.md#to_dict
        """
        return {
            "P": self.P, "Ne": self.Ne, "M1": self.M1, "M2": self.M2,
            "F": self.F, "Mf": self.Mf, "E": self.E, "S": self.S,
            "C_TNF": self.C_TNF, "C_IL10": self.C_IL10,
            "C_PDGF": self.C_PDGF, "C_VEGF": self.C_VEGF,
            "C_TGFb": self.C_TGFb, "C_MCP1": self.C_MCP1,
            "C_IL8": self.C_IL8,
            "rho_collagen": self.rho_collagen, "C_MMP": self.C_MMP,
            "rho_fibrin": self.rho_fibrin,
            "D": self.D, "O2": self.O2,
            "t": self.t,
        }


@dataclass
class ExtendedSDETrajectory:
    """Траектория 20-переменной SDE симуляции.

    Хранит полную историю состояний для анализа и визуализации.

    Подробное описание:
        Description/Phase2/description_extended_sde.md#ExtendedSDETrajectory
    """

    times: np.ndarray = field(
        default_factory=lambda: np.array([]),
    )  # Временные точки (ч)
    states: list[ExtendedSDEState] = field(
        default_factory=list,
    )  # Состояния
    params: ParameterSet = field(
        default_factory=ParameterSet,
    )  # Параметры симуляции

    def get_variable(self, name: str) -> np.ndarray:
        """Извлечь временной ряд одной переменной.

        Args:
            name: Имя переменной из VARIABLE_NAMES
                  (e.g., "P", "C_TNF", "rho_collagen")

        Returns:
            np.ndarray shape (n_steps,) со значениями переменной

        Raises:
            KeyError: Если имя переменной неизвестно

        Подробное описание:
            Description/Phase2/description_extended_sde.md#get_variable
        """
        if name not in VARIABLE_NAMES:
            raise KeyError(f"Неизвестная переменная: {name}")
        if not self.states:
            return np.array([])
        return np.array([getattr(s, name) for s in self.states])

    def get_statistics(self) -> dict[str, dict[str, float]]:
        """Статистика траектории для каждой переменной.

        Вычисляет mean, std, min, max, final для всех 20 переменных.

        Returns:
            {имя_переменной: {"mean", "std", "min", "max", "final"}}

        Подробное описание:
            Description/Phase2/description_extended_sde.md#get_statistics
        """
        result: dict[str, dict[str, float]] = {}
        for name in VARIABLE_NAMES:
            values = self.get_variable(name)
            result[name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "final": float(values[-1]),
            }
        return result


class ExtendedSDEModel:
    """Расширенная 20-переменная SDE модель регенерации тканей.

    Реализует полную систему из Mathematical Framework §2:
    - 8 клеточных: логистический рост, рекрутирование, переключение
    - 7 цитокиновых: секреция-деградация с обратными связями
    - 3 ECM: продукция-деградация коллагена, MMP баланс
    - 2 вспомогательных: damage signal, oxygen dynamics

    Ключевые биологические свойства:
    - M1→M2 переключение макрофагов (фазовый переход)
    - TGF-β ↔ Mf бистабильность (заживление vs фиброз)
    - Гипоксия-зависимый ангиогенез
    - Эффероцитоз как драйвер разрешения воспаления

    Метод интегрирования: Эйлера-Маруяма.

    Подробное описание:
        Description/Phase2/description_extended_sde.md#ExtendedSDEModel
    """

    N_VARIABLES: int = 20

    def __init__(
        self,
        params: ParameterSet | None = None,
        therapy: TherapyProtocol | None = None,
        rng_seed: int | None = None,
    ) -> None:
        """Инициализация расширенной SDE модели.

        Args:
            params: Набор параметров (None → литературные defaults)
            therapy: Протокол терапии PRP/PEMF (None → без терапии)
            rng_seed: Seed для воспроизводимости (None → random)

        Подробное описание:
            Description/Phase2/description_extended_sde.md#__init__
        """
        self.params = params if params is not None else ParameterSet()
        self.therapy = therapy
        self._rng = np.random.default_rng(rng_seed)

    # ===== Основные методы =====

    def simulate(
        self,
        initial_state: ExtendedSDEState,
        t_span: tuple[float, float] | None = None,
    ) -> ExtendedSDETrajectory:
        """Полная симуляция методом Эйлера-Маруямы.

        Алгоритм:
        1. Инициализация X₀ из initial_state
        2. Для каждого шага n: X_{n+1} = X_n + μ(X_n)·dt + σ(X_n)·√dt·ξ
        3. Применение граничных условий (X >= 0)
        4. Сохранение траектории

        Args:
            initial_state: Начальное состояние (20 переменных)
            t_span: (t_start, t_end) в часах.
                    None → (0, params.t_max)

        Returns:
            ExtendedSDETrajectory с результатами

        Подробное описание:
            Description/Phase2/description_extended_sde.md#simulate
        """
        p = self.params
        if t_span is None:
            t_start, t_end = 0.0, p.t_max
        else:
            t_start, t_end = t_span

        n_steps = int((t_end - t_start) / p.dt)
        sqrt_dt = np.sqrt(p.dt)
        times = np.linspace(t_start, t_end, n_steps + 1)

        states: list[ExtendedSDEState] = []
        current_state = initial_state
        states.append(current_state)

        for i in range(n_steps):
            drift = self._compute_drift(current_state)
            diffusion = self._compute_diffusion(current_state)
            dW = self._rng.standard_normal(self.N_VARIABLES) * sqrt_dt
            x = current_state.to_array()
            x_new = x + drift * p.dt + diffusion * dW
            new_state = ExtendedSDEState.from_array(x_new, t=times[i + 1])
            new_state = self._apply_boundary_conditions(new_state)
            states.append(new_state)
            current_state = new_state

        return ExtendedSDETrajectory(
            times=times, states=states, params=p,
        )

    def _compute_drift(self, state: ExtendedSDEState) -> np.ndarray:
        """Вычисление 20-мерного вектора drift μ(X, t).

        Собирает drift из индивидуальных _drift_* компонентов.

        Args:
            state: Текущее состояние

        Returns:
            np.ndarray shape (20,) — drift для каждой переменной

        Подробное описание:
            Description/Phase2/description_extended_sde.md#_compute_drift
        """
        drift = np.zeros(self.N_VARIABLES)
        drift[StateIndex.P] = self._drift_platelets(state)
        drift[StateIndex.Ne] = self._drift_neutrophils(state)
        drift[StateIndex.M1] = self._drift_M1(state)
        drift[StateIndex.M2] = self._drift_M2(state)
        drift[StateIndex.F] = self._drift_fibroblasts(state)
        drift[StateIndex.Mf] = self._drift_myofibroblasts(state)
        drift[StateIndex.E] = self._drift_endothelial(state)
        drift[StateIndex.S] = self._drift_stem_cells(state)
        drift[StateIndex.C_TNF] = self._drift_C_TNF(state)
        drift[StateIndex.C_IL10] = self._drift_C_IL10(state)
        drift[StateIndex.C_PDGF] = self._drift_C_PDGF(state)
        drift[StateIndex.C_VEGF] = self._drift_C_VEGF(state)
        drift[StateIndex.C_TGFb] = self._drift_C_TGFb(state)
        drift[StateIndex.C_MCP1] = self._drift_C_MCP1(state)
        drift[StateIndex.C_IL8] = self._drift_C_IL8(state)
        drift[StateIndex.RHO_COLLAGEN] = self._drift_collagen(state)
        drift[StateIndex.C_MMP] = self._drift_MMP(state)
        drift[StateIndex.RHO_FIBRIN] = self._drift_fibrin(state)
        drift[StateIndex.D] = self._drift_damage(state)
        drift[StateIndex.O2] = self._drift_oxygen(state)
        return drift

    def _compute_diffusion(
        self, state: ExtendedSDEState,
    ) -> np.ndarray:
        """Вычисление 20-мерного вектора diffusion σ(X, t).

        Формула: σ_i = sigma_i * X_i (геометрическое броуновское движение).
        Диагональная матрица шума (независимые Винеровские процессы).

        Args:
            state: Текущее состояние

        Returns:
            np.ndarray shape (20,) — diffusion для каждой переменной

        Подробное описание:
            Description/Phase2/description_extended_sde.md#_compute_diffusion
        """
        p = self.params
        x = state.to_array()
        sigmas = np.array([
            p.sigma_P, p.sigma_Ne, p.sigma_M, p.sigma_M,
            p.sigma_F, p.sigma_Mf, p.sigma_E, p.sigma_S,
            p.sigma_TNF, p.sigma_IL10, p.sigma_PDGF, p.sigma_VEGF,
            p.sigma_TGF, p.sigma_MCP1, p.sigma_IL8,
            0.0, 0.0, 0.0,  # ECM: collagen, MMP, fibrin
            0.0, 0.0,        # auxiliary: D, O2
        ])
        return sigmas * x

    # ===== Drift клеточных популяций (§2.1) =====

    def _drift_platelets(self, state: ExtendedSDEState) -> float:
        """Drift тромбоцитов P(t).

        dP/dt = S_P(t) - δ_P·P - k_deg·P
        где S_P(t) = P_max·exp(-t/τ_P) — быстрая активация.

        Подробное описание:
            Description/Phase2/description_extended_sde.md#_drift_platelets
        """
        p = self.params
        source = p.P_max * np.exp(-state.t / p.tau_P)
        clearance = p.delta_P * state.P
        degranulation = p.k_deg * state.P
        return source - clearance - degranulation

    def _drift_neutrophils(self, state: ExtendedSDEState) -> float:
        """Drift нейтрофилов Nₑ(t).

        dNe/dt = R_Ne(C_IL8) - δ_Ne·Ne
                 - k_phag·M_total·Ne/(Ne + K_phag)
        где R_Ne = R_Ne_max · C_IL8ⁿ/(K_IL8ⁿ + C_IL8ⁿ).

        Подробное описание:
            Description/Phase2/description_extended_sde.md#_drift_neutrophils
        """
        p = self.params
        recruitment = p.R_Ne_max * self._hill(state.C_IL8, p.K_IL8, p.n_hill)
        apoptosis = p.delta_Ne * state.Ne
        m_total = state.M1 + state.M2
        denom = state.Ne + p.K_phag
        phagocytosis = (
            p.k_phag * m_total * state.Ne / denom if denom > 0 else 0.0
        )
        return recruitment - apoptosis - phagocytosis

    def _drift_M1(self, state: ExtendedSDEState) -> float:
        """Drift M1 макрофагов (провоспалительных).

        dM1/dt = R_M(C_MCP1)·φ₁ - k_switch·ψ·M1
                 + k_reverse·ζ·M2 - δ_M·M1
        Рекрутирование, переключение M1↔M2, апоптоз.

        Подробное описание:
            Description/Phase2/description_extended_sde.md#_drift_M1
        """
        p = self.params
        r_m = p.R_M_max * self._hill(state.C_MCP1, p.K_MCP1, p.n_hill)
        phi1 = self._polarization_M1(state)
        recruitment = r_m * phi1
        switching_out = p.k_switch * self._switching_function(state) * state.M1
        switching_in = p.k_reverse * self._reverse_switching(state) * state.M2
        death = p.delta_M * state.M1
        return recruitment - switching_out + switching_in - death

    def _drift_M2(self, state: ExtendedSDEState) -> float:
        """Drift M2 макрофагов (репаративных).

        dM2/dt = R_M(C_MCP1)·φ₂ + k_switch·ψ·M1
                 - k_reverse·ζ·M2 - δ_M·M2
        Зеркальное уравнение M1, с φ₂ = 1 - φ₁.

        Подробное описание:
            Description/Phase2/description_extended_sde.md#_drift_M2
        """
        p = self.params
        r_m = p.R_M_max * self._hill(state.C_MCP1, p.K_MCP1, p.n_hill)
        phi2 = self._polarization_M2(state)
        recruitment = r_m * phi2
        switching_in = p.k_switch * self._switching_function(state) * state.M1
        switching_out = (
            p.k_reverse * self._reverse_switching(state) * state.M2
        )
        death = p.delta_M * state.M2
        return recruitment + switching_in - switching_out - death

    def _drift_fibroblasts(self, state: ExtendedSDEState) -> float:
        """Drift фибробластов F(t).

        dF/dt = r_F·F·(1-(F+Mf)/K_F)·H(PDGF,TGFβ)
                + k_diff_S·S·g_diff(TGFβ)
                - k_act·F·A(TGFβ) - δ_F·F
        Логистический рост + дифференциация S - активация в Mf.

        Подробное описание:
            Description/Phase2/description_extended_sde.md#_drift_fibroblasts
        """
        p = self.params
        logistic = p.r_F * state.F * (1.0 - (state.F + state.Mf) / p.K_F)
        h = self._mitogenic_stimulation(state)
        growth = logistic * h
        diff_in = (
            p.k_diff_S * state.S * self._differentiation_probability(state)
        )
        activation_out = (
            p.k_act * state.F * self._activation_function(state)
        )
        death = p.delta_F * state.F
        return growth + diff_in - activation_out - death

    def _drift_myofibroblasts(self, state: ExtendedSDEState) -> float:
        """Drift миофибробластов Mf(t).

        dMf/dt = k_act·F·A(TGFβ)
                 - δ_Mf·Mf·(1 - TGFβ/(K_survival + TGFβ))
        Приток из F, TGF-β-зависимый апоптоз (бистабильность!).

        Подробное описание:
            Description/Phase2/description_extended_sde.md#_drift_myofibroblasts
        """
        p = self.params
        a = self._activation_function(state)
        influx = p.k_act * state.F * a
        denom = p.K_survival + state.C_TGFb
        survival_factor = (
            state.C_TGFb / denom if denom > 0 else 0.0
        )
        apoptosis = p.delta_Mf * state.Mf * (1.0 - survival_factor)
        return influx - apoptosis

    def _drift_endothelial(self, state: ExtendedSDEState) -> float:
        """Drift эндотелиальных клеток E(t) — ангиогенез.

        dE/dt = r_E·E·(1-E/K_E)·V(VEGF)·(1-θ(O₂)) - δ_E·E
        VEGF-зависимая пролиферация, гипоксия-стимуляция.

        Подробное описание:
            Description/Phase2/description_extended_sde.md#_drift_endothelial
        """
        p = self.params
        v = self._vegf_activation(state)
        theta = self._hypoxia_factor(state)
        growth = (
            p.r_E * state.E * (1.0 - state.E / p.K_E) * v * (1.0 - theta)
        )
        death = p.delta_E * state.E
        return growth - death

    def _drift_stem_cells(self, state: ExtendedSDEState) -> float:
        """Drift стволовых клеток S(t).

        dS/dt = r_S·S·(1-S/K_S)·(1+α_PRP·Θ_PRP)
                - k_diff_S·S·g_diff(TGFβ) - δ_S·S
        Логистический рост с PRP-стимуляцией, дифференциация.

        Подробное описание:
            Description/Phase2/description_extended_sde.md#_drift_stem_cells
        """
        p = self.params
        prp_factor = 0.0
        if self.therapy and self.therapy.prp_enabled:
            t_days = state.t / 24.0
            if self.therapy.prp_start_time <= t_days < self.therapy.prp_end_time:
                prp_factor = self.therapy.prp_intensity

        logistic = p.r_S * state.S * (1.0 - state.S / p.K_S)
        growth = logistic * (1.0 + p.alpha_PRP_S * prp_factor)
        diff_loss = (
            p.k_diff_S * state.S * self._differentiation_probability(state)
        )
        death = p.delta_S * state.S
        return growth - diff_loss - death

    # ===== Drift цитокинов (§2.2) =====

    def _drift_C_TNF(self, state: ExtendedSDEState) -> float:
        """Drift TNF-α.

        dC_TNF/dt = s_M1·M1 + s_Ne·Ne - γ·C_TNF
                    - k_inhib·C_IL10·C_TNF/(K_inhib + C_TNF)
        Секреция M1 и Ne, деградация, ингибирование IL-10.

        Подробное описание:
            Description/Phase2/description_extended_sde.md#_drift_C_TNF
        """
        p = self.params
        production = p.s_TNF_M1 * state.M1 + p.s_TNF_Ne * state.Ne
        degradation = p.gamma_TNF * state.C_TNF
        denom = p.K_inhib + state.C_TNF
        inhibition = (
            p.k_inhib_IL10 * state.C_IL10 * state.C_TNF / denom
            if denom > 0 else 0.0
        )
        return production - degradation - inhibition

    def _drift_C_IL10(self, state: ExtendedSDEState) -> float:
        """Drift IL-10.

        dC_IL10/dt = s_M2·M2 + s_efferocytosis·phagocytosis_rate
                     - γ·C_IL10
        Секреция M2, эффероцитоз-индуцированная продукция.

        Подробное описание:
            Description/Phase2/description_extended_sde.md#_drift_C_IL10
        """
        p = self.params
        m_total = state.M1 + state.M2
        denom = state.Ne + p.K_phag
        phag_rate = (
            p.k_phag * m_total * state.Ne / denom if denom > 0 else 0.0
        )
        production = (
            p.s_IL10_M2 * state.M2 + p.s_IL10_efferocytosis * phag_rate
        )
        degradation = p.gamma_IL10 * state.C_IL10
        return production - degradation

    def _drift_C_PDGF(self, state: ExtendedSDEState) -> float:
        """Drift PDGF.

        dC_PDGF/dt = s_P·k_deg·P + s_M·(M1+M2) + Θ_PRP_PDGF
                     - γ·C_PDGF - k_bind_F·F·C/(K+C)
        Тромбоциты, макрофаги, PRP; связывание фибробластами.

        Подробное описание:
            Description/Phase2/description_extended_sde.md#_drift_C_PDGF
        """
        p = self.params
        platelet_release = p.s_PDGF_P * p.k_deg * state.P
        macro_production = p.s_PDGF_M * (state.M1 + state.M2)
        degradation = p.gamma_PDGF * state.C_PDGF
        denom = p.K_PDGF + state.C_PDGF
        binding = (
            p.k_bind_F * state.F * state.C_PDGF / denom
            if denom > 0 else 0.0
        )
        # TODO: добавить Θ_PRP_PDGF(t) из therapy_models.py (§3.1)
        return platelet_release + macro_production - degradation - binding

    def _drift_C_VEGF(self, state: ExtendedSDEState) -> float:
        """Drift VEGF.

        dC_VEGF/dt = s_M2·M2·(1+α_hypoxia·(1-θ)) + s_F·F
                     + Θ_PRP_VEGF - γ·C_VEGF - k_bind_E·E·C/(K+C)
        M2 (гипоксия-усиленный), фибробласты, PRP.

        Подробное описание:
            Description/Phase2/description_extended_sde.md#_drift_C_VEGF
        """
        p = self.params
        theta = self._hypoxia_factor(state)
        m2_production = (
            p.s_VEGF_M2 * state.M2 * (1.0 + p.alpha_hypoxia * (1.0 - theta))
        )
        f_production = p.s_VEGF_F * state.F
        degradation = p.gamma_VEGF * state.C_VEGF
        denom = p.K_VEGF + state.C_VEGF
        binding = (
            p.k_bind_E * state.E * state.C_VEGF / denom
            if denom > 0 else 0.0
        )
        # TODO: добавить Θ_PRP_VEGF(t) из therapy_models.py (§3.1)
        return m2_production + f_production - degradation - binding

    def _drift_C_TGFb(self, state: ExtendedSDEState) -> float:
        """Drift TGF-β (критический для бистабильности!).

        dC_TGFβ/dt = s_P·k_deg·P + s_M2·M2 + s_Mf·Mf
                     + Θ_PRP_TGF - γ·C_TGFβ
        Положительная обратная связь: Mf → TGF-β → Mf.

        Подробное описание:
            Description/Phase2/description_extended_sde.md#_drift_C_TGFb
        """
        p = self.params
        platelet_release = p.s_TGF_P * p.k_deg * state.P
        m2_production = p.s_TGF_M2 * state.M2
        mf_production = p.s_TGF_Mf * state.Mf  # положительная обр. связь
        degradation = p.gamma_TGF * state.C_TGFb
        # TODO: добавить Θ_PRP_TGF(t) из therapy_models.py (§3.1)
        return platelet_release + m2_production + mf_production - degradation

    def _drift_C_MCP1(self, state: ExtendedSDEState) -> float:
        """Drift MCP-1.

        dC_MCP1/dt = s_damage·D(t) + s_M1·M1 - γ·C_MCP1
        DAMPs-зависимый, M1-амплифицированный хемоаттрактант.

        Подробное описание:
            Description/Phase2/description_extended_sde.md#_drift_C_MCP1
        """
        p = self.params
        damage_prod = p.s_MCP1_damage * state.D
        m1_prod = p.s_MCP1_M1 * state.M1
        degradation = p.gamma_MCP1 * state.C_MCP1
        return damage_prod + m1_prod - degradation

    def _drift_C_IL8(self, state: ExtendedSDEState) -> float:
        """Drift IL-8.

        dC_IL8/dt = (s_damage·D + s_M1·M1 + s_Ne·Ne)
                    / (1 + C_IL10/K_inhib) - γ·C_IL8
        DAMPs, M1, нейтрофильная аутокринная петля.

        Расширение §2.2.7: IL-10 подавляет продукцию хемокинов
        через ингибирование NF-κB (Fiorentino 1991, Moore 2001).

        Подробное описание:
            Description/Phase2/description_extended_sde.md#_drift_C_IL8
        """
        p = self.params
        production = (
            p.s_IL8_damage * state.D
            + p.s_IL8_M1 * state.M1
            + p.s_IL8_Ne * state.Ne
        )
        # Расширение §2.2.7: IL-10 подавляет хемокины через NF-κB
        il10_suppression = 1.0 + state.C_IL10 / p.K_inhib
        degradation = p.gamma_IL8 * state.C_IL8
        return production / il10_suppression - degradation

    # ===== Drift ECM (§2.3) =====

    def _drift_collagen(self, state: ExtendedSDEState) -> float:
        """Drift коллагена ρ_c(t).

        dρ_c/dt = (q_F·F + q_Mf·Mf)·(1-ρ_c/ρ_max)
                  - k_MMP·C_MMP·ρ_c/(K_MMP_sub + ρ_c)
        Продукция F и Mf с насыщением, MMP-деградация.

        Подробное описание:
            Description/Phase2/description_extended_sde.md#_drift_collagen
        """
        p = self.params
        production = (
            (p.q_F * state.F + p.q_Mf * state.Mf)
            * (1.0 - state.rho_collagen / p.rho_c_max)
        )
        denom = p.K_MMP_sub + state.rho_collagen
        degradation = (
            p.k_MMP_deg * state.C_MMP * state.rho_collagen / denom
            if denom > 0 else 0.0
        )
        return production - degradation

    def _drift_MMP(self, state: ExtendedSDEState) -> float:
        """Drift MMP.

        dC_MMP/dt = s_M1·M1 + s_M2·α·M2 + s_F·F
                    - k_TIMP·C_TIMP·C_MMP - γ·C_MMP
        M1, M2 (сниженный), F; TIMP-ингибирование.

        Подробное описание:
            Description/Phase2/description_extended_sde.md#_drift_MMP
        """
        p = self.params
        m1_secretion = p.s_MMP_M1 * state.M1
        m2_secretion = p.s_MMP_M2 * p.alpha_MMP_M2 * state.M2
        f_secretion = p.s_MMP_F * state.F
        timp_inhibition = p.k_TIMP * p.C_TIMP * state.C_MMP
        degradation = p.gamma_MMP * state.C_MMP
        return (
            m1_secretion + m2_secretion + f_secretion
            - timp_inhibition - degradation
        )

    def _drift_fibrin(self, state: ExtendedSDEState) -> float:
        """Drift фибрина ρ_f(t).

        dρ_f/dt = -k_fibrinolysis·C_MMP·ρ_f - k_remodel·F·ρ_f
        Только убыль: фибринолиз + ремоделирование в коллаген.

        Подробное описание:
            Description/Phase2/description_extended_sde.md#_drift_fibrin
        """
        p = self.params
        fibrinolysis = p.k_fibrinolysis * state.C_MMP * state.rho_fibrin
        remodeling = p.k_remodel * state.F * state.rho_fibrin
        return -fibrinolysis - remodeling

    # ===== Drift вспомогательных (§2.4) =====

    def _drift_damage(self, state: ExtendedSDEState) -> float:
        """Drift сигнала повреждения D(t).

        dD/dt = -D/τ_damage
        Аналитическое экспоненциальное затухание DAMPs.

        Подробное описание:
            Description/Phase2/description_extended_sde.md#_drift_damage
        """
        return -state.D / self.params.tau_damage

    def _drift_oxygen(self, state: ExtendedSDEState) -> float:
        """Drift кислорода O₂(t).

        dO₂/dt = D_O2·(O₂_blood - O₂)/L²
                 - k_consumption·cells·O₂/(K_consume + O₂)
                 + k_angio·E
        Диффузия, потребление клетками, перфузия от ангиогенеза.

        Подробное описание:
            Description/Phase2/description_extended_sde.md#_drift_oxygen
        """
        p = self.params
        l_sq = p.L_diffusion ** 2
        diffusion = p.D_O2 * (p.O2_blood - state.O2) / l_sq
        total_cells = (
            state.P + state.Ne + state.M1 + state.M2
            + state.F + state.Mf + state.E + state.S
        )
        denom = p.K_O2_consume + state.O2
        consumption = (
            p.k_consumption * total_cells * state.O2 / denom
            if denom > 0 else 0.0
        )
        perfusion = p.k_angio * state.E
        return diffusion - consumption + perfusion

    # ===== Вспомогательные функции =====

    def _hill(self, x: float, K: float, n: int = 2) -> float:
        """Hill функция: xⁿ / (Kⁿ + xⁿ).

        Стандартная кинетика насыщения с кооперативностью.
        Используется для рекрутирования, активации, переключения.

        Args:
            x: Концентрация лиганда
            K: Константа полунасыщения
            n: Коэффициент Хилла (кооперативность)

        Returns:
            Значение в [0, 1]

        Подробное описание:
            Description/Phase2/description_extended_sde.md#_hill
        """
        if x <= 0:
            return 0.0
        xn = x ** n
        kn = K ** n
        return xn / (kn + xn)

    def _polarization_M1(self, state: ExtendedSDEState) -> float:
        """Доля поляризации в M1: φ₁ = C_TNF / (C_TNF + C_IL10 + ε).

        TNF-α поддерживает M1 состояние, IL-10 — M2.

        Args:
            state: Текущее состояние

        Returns:
            φ₁ ∈ [0, 1] — доля M1 поляризации

        Подробное описание:
            Description/Phase2/description_extended_sde.md#_polarization_M1
        """
        denom = state.C_TNF + state.C_IL10 + self.params.epsilon
        return state.C_TNF / denom

    def _polarization_M2(self, state: ExtendedSDEState) -> float:
        """Доля поляризации в M2: φ₂ = 1 - φ₁.

        Баланс: φ₁ + φ₂ = 1 (M1/M2 — спектр поляризации).

        Args:
            state: Текущее состояние

        Returns:
            φ₂ ∈ [0, 1] — доля M2 поляризации

        Подробное описание:
            Description/Phase2/description_extended_sde.md#_polarization_M2
        """
        return 1.0 - self._polarization_M1(state)

    def _switching_function(self, state: ExtendedSDEState) -> float:
        """Функция M1→M2 переключения: ψ(C_IL10, C_TGFβ).

        ψ = Hill(C_IL10 + C_TGFβ, K_switch_half, n)
        IL-10 и TGF-β управляют M2 поляризацией.

        Args:
            state: Текущее состояние

        Returns:
            ψ ∈ [0, 1]

        Подробное описание:
            Description/Phase2/description_extended_sde.md#_switching_function
        """
        return self._hill(
            state.C_IL10 + state.C_TGFb,
            self.params.K_switch_half,
            self.params.n_hill,
        )

    def _reverse_switching(self, state: ExtendedSDEState) -> float:
        """Обратное M2→M1 переключение: ζ(C_TNF).

        ζ = Hill(C_TNF, K_reverse_half, n=2)
        Высокий TNF-α возвращает M2 в M1 состояние.

        Args:
            state: Текущее состояние

        Returns:
            ζ ∈ [0, 1]

        Подробное описание:
            Description/Phase2/description_extended_sde.md#_reverse_switching
        """
        return self._hill(state.C_TNF, self.params.K_reverse_half, 2)

    def _mitogenic_stimulation(
        self, state: ExtendedSDEState,
    ) -> float:
        """Митогенная стимуляция фибробластов H(C_PDGF, C_TGFβ).

        H = (C_PDGF/(K_PDGF+C_PDGF))
            · (1 + α_TGF · C_TGFβ/(K_TGFb_prolif+C_TGFβ))
        PDGF — первичный митоген, TGF-β усиливает.

        Args:
            state: Текущее состояние

        Returns:
            H >= 0 — митогенный фактор

        Подробное описание:
            Description/Phase2/description_extended_sde.md#_mitogenic_stimulation
        """
        p = self.params
        denom_pdgf = p.K_PDGF + state.C_PDGF
        pdgf_term = (
            state.C_PDGF / denom_pdgf if denom_pdgf > 0 else 0.0
        )
        denom_tgf = p.K_TGFb_prolif + state.C_TGFb
        tgf_term = (
            state.C_TGFb / denom_tgf if denom_tgf > 0 else 0.0
        )
        return pdgf_term * (1.0 + p.alpha_TGF * tgf_term)

    def _differentiation_probability(
        self, state: ExtendedSDEState,
    ) -> float:
        """Вероятность дифференциации стволовых: g_diff(C_TGFβ).

        g_diff = C_TGFβ / (K_diff + C_TGFβ)
        TGF-β-зависимая дифференциация S → F.

        Args:
            state: Текущее состояние

        Returns:
            g_diff ∈ [0, 1]

        Подробное описание:
            Description/Phase2/description_extended_sde.md#_differentiation_probability
        """
        denom = self.params.K_diff + state.C_TGFb
        if denom > 0:
            return state.C_TGFb / denom
        return 0.0

    def _activation_function(self, state: ExtendedSDEState) -> float:
        """Активация F→Mf: A(C_TGFβ) = Hill(C_TGFβ, K_activ, n=2).

        TGF-β через Smad2/3 активирует трансформацию
        фибробластов в миофибробласты (α-SMA экспрессия).

        Args:
            state: Текущее состояние

        Returns:
            A ∈ [0, 1]

        Подробное описание:
            Description/Phase2/description_extended_sde.md#_activation_function
        """
        return self._hill(state.C_TGFb, self.params.K_activ, 2)

    def _vegf_activation(self, state: ExtendedSDEState) -> float:
        """VEGF-зависимая активация ангиогенеза: V(C_VEGF).

        V = Hill(C_VEGF, K_VEGF, n=2)
        VEGFR2 димеризация определяет Hill n=2.

        Args:
            state: Текущее состояние

        Returns:
            V ∈ [0, 1]

        Подробное описание:
            Description/Phase2/description_extended_sde.md#_vegf_activation
        """
        return self._hill(state.C_VEGF, self.params.K_VEGF, 2)

    def _hypoxia_factor(self, state: ExtendedSDEState) -> float:
        """Гипоксический фактор: θ = O₂/(K_O2 + O₂).

        При низком O₂ → θ → 0 → (1-θ) → 1 → стимуляция ангиогенеза.
        При нормоксии → θ → 1 → (1-θ) → 0 → нет стимуляции.

        Args:
            state: Текущее состояние

        Returns:
            θ ∈ [0, 1]

        Подробное описание:
            Description/Phase2/description_extended_sde.md#_hypoxia_factor
        """
        denom = self.params.K_O2 + state.O2
        if denom > 0:
            return state.O2 / denom
        return 0.0

    # ===== Граничные условия и утилиты =====

    def _apply_boundary_conditions(
        self, state: ExtendedSDEState,
    ) -> ExtendedSDEState:
        """Граничные условия: все переменные >= 0.

        Отражающая граница: отрицательные значения обнуляются.
        Физически: концентрации и популяции не могут быть < 0.

        Args:
            state: Состояние (может содержать отрицательные)

        Returns:
            Скорректированное состояние

        Подробное описание:
            Description/Phase2/description_extended_sde.md#_apply_boundary
        """
        arr = state.to_array()
        arr = np.maximum(arr, 0.0)
        return ExtendedSDEState.from_array(arr, t=state.t)

    def validate_params(self) -> bool:
        """Проверка совместимости параметров с моделью.

        Делегирует к self.params.validate() с дополнительными
        проверками специфичными для SDE интеграции.

        Returns:
            True если параметры валидны

        Raises:
            ValueError: Если параметры некорректны

        Подробное описание:
            Description/Phase2/description_extended_sde.md#validate_params
        """
        return self.params.validate()

    def get_default_initial_state(self) -> ExtendedSDEState:
        """Начальное состояние для типичной раны (t=0).

        Начальные условия (Gurtner et al., Nature 2008):
        - P = P_max (максимальная активация тромбоцитов)
        - D = D0 (максимальный damage signal)
        - O2 = O2_blood (начальный кислород)
        - rho_fibrin = 1.0 (фибриновый сгусток)
        - Остальные ≈ 0 или малые базовые значения

        Returns:
            ExtendedSDEState с начальными условиями

        Подробное описание:
            Description/Phase2/description_extended_sde.md#get_default_initial
        """
        p = self.params
        return ExtendedSDEState(
            P=p.P_max, D=p.D0, O2=p.O2_blood, rho_fibrin=1.0,
            # Резидентные клетки раневого ложа
            F=10.0,
            # Ранние нейтрофилы из повреждённых сосудов (Kolaczkowska 2013)
            Ne=50.0,
            # DAMP-индуцированное начальное высвобождение цитокинов
            # C_IL8 = K_IL8 для 50% Hill-рекрутирования (немедленный DAMP ответ)
            C_IL8=p.K_IL8,
            C_MCP1=p.s_MCP1_damage * p.D0 / p.gamma_MCP1,
            t=0.0,
        )
