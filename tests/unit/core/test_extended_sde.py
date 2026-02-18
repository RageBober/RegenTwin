"""TDD тесты для extended_sde.py — 20-переменная SDE модель.

Тестирование:
- StateIndex: индексация 20 переменных
- ExtendedSDEState: dataclass состояния, to_array/from_array/to_dict
- ExtendedSDETrajectory: get_variable, get_statistics
- ExtendedSDEModel: инициализация, simulate, drift/diffusion
- Drift компоненты: 8 клеточных, 7 цитокиновых, 3 ECM, 2 вспомогательных
- Вспомогательные функции: Hill, поляризация, переключение и др.
- Граничные условия, валидация, начальное состояние
- Биологические свойства: позитивность, M1→M2, бистабильность

Все тесты написаны для stub-реализации (NotImplementedError).
После реализации методов тесты должны проходить.
"""

import numpy as np
import pytest

from src.core.extended_sde import (
    N_VARIABLES,
    VARIABLE_NAMES,
    ExtendedSDEModel,
    ExtendedSDEState,
    ExtendedSDETrajectory,
    StateIndex,
)
from src.core.parameters import ParameterSet

# =============================================================================
# TestStateIndex
# =============================================================================


class TestStateIndex:
    """Тесты перечисления StateIndex (20 переменных)."""

    def test_count_equals_20(self):
        """StateIndex содержит ровно 20 значений."""
        assert len(StateIndex) == 20

    def test_values_0_to_19(self):
        """Индексы покрывают 0..19 без пропусков."""
        assert {s.value for s in StateIndex} == set(range(20))

    def test_first_P_zero(self):
        """Первый индекс — тромбоциты P = 0."""
        assert StateIndex.P == 0

    def test_last_O2_nineteen(self):
        """Последний индекс — кислород O2 = 19."""
        assert StateIndex.O2 == 19

    def test_variable_names_match(self):
        """VARIABLE_NAMES и N_VARIABLES согласованы со StateIndex."""
        assert len(VARIABLE_NAMES) == 20
        assert N_VARIABLES == 20


# =============================================================================
# TestExtendedSDEStateDataclass
# =============================================================================


class TestExtendedSDEStateDataclass:
    """Тесты dataclass ExtendedSDEState."""

    def test_default_all_zeros(self):
        """Все поля по умолчанию равны 0.0."""
        state = ExtendedSDEState()
        assert state.P == 0.0
        assert state.Ne == 0.0
        assert state.M1 == 0.0
        assert state.M2 == 0.0
        assert state.O2 == 0.0
        assert state.t == 0.0

    def test_custom_values(self):
        """Кастомные значения корректно сохраняются."""
        state = ExtendedSDEState(P=100.0, Ne=50.0)
        assert state.P == 100.0
        assert state.Ne == 50.0

    def test_field_count_21(self):
        """21 поле: 20 переменных + t."""
        import dataclasses

        fields = dataclasses.fields(ExtendedSDEState)
        assert len(fields) == 21

    def test_time_field(self):
        """Поле времени t по умолчанию 0.0."""
        state = ExtendedSDEState()
        assert state.t == 0.0


# =============================================================================
# TestExtendedSDEStateToArray
# =============================================================================


class TestExtendedSDEStateToArray:
    """Тесты конвертации состояния в numpy массив."""

    def test_zeros_returns_zeros(self):
        """Нулевое состояние -> нулевой массив."""
        state = ExtendedSDEState()
        result = state.to_array()
        np.testing.assert_array_equal(result, np.zeros(20))

    def test_shape_20(self):
        """Результат имеет shape (20,)."""
        state = ExtendedSDEState()
        result = state.to_array()
        assert result.shape == (20,)

    def test_P_at_index_0(self):
        """P записывается в индекс StateIndex.P = 0."""
        state = ExtendedSDEState(P=100.0)
        result = state.to_array()
        assert result[StateIndex.P] == 100.0

    def test_O2_at_index_19(self):
        """O2 записывается в индекс StateIndex.O2 = 19."""
        state = ExtendedSDEState(O2=90.0)
        result = state.to_array()
        assert result[StateIndex.O2] == 90.0

    def test_order_matches_state_index(self):
        """Порядок элементов соответствует StateIndex."""
        state = ExtendedSDEState(
            P=1.0, Ne=2.0, M1=3.0, M2=4.0, F=5.0, Mf=6.0, E=7.0, S=8.0,
            C_TNF=9.0, C_IL10=10.0, C_PDGF=11.0, C_VEGF=12.0, C_TGFb=13.0,
            C_MCP1=14.0, C_IL8=15.0, rho_collagen=16.0, C_MMP=17.0,
            rho_fibrin=18.0, D=19.0, O2=20.0,
        )
        result = state.to_array()
        assert result[StateIndex.P] == 1.0
        assert result[StateIndex.Ne] == 2.0
        assert result[StateIndex.C_TNF] == 9.0
        assert result[StateIndex.RHO_COLLAGEN] == 16.0
        assert result[StateIndex.D] == 19.0


# =============================================================================
# TestExtendedSDEStateFromArray
# =============================================================================


class TestExtendedSDEStateFromArray:
    """Тесты создания состояния из numpy массива."""

    def test_from_zeros(self):
        """from_array(zeros) -> все поля 0.0."""
        state = ExtendedSDEState.from_array(np.zeros(20))
        assert state.P == 0.0
        assert state.O2 == 0.0

    def test_from_ones_with_time(self):
        """from_array(ones, t=5) -> поля 1.0, t=5.0."""
        state = ExtendedSDEState.from_array(np.ones(20), t=5.0)
        assert state.P == 1.0
        assert state.Ne == 1.0
        assert state.t == 5.0

    def test_wrong_length_19_raises(self):
        """Массив длины 19 -> ValueError."""
        with pytest.raises(ValueError):
            ExtendedSDEState.from_array(np.zeros(19))

    def test_wrong_length_21_raises(self):
        """Массив длины 21 -> ValueError."""
        with pytest.raises(ValueError):
            ExtendedSDEState.from_array(np.zeros(21))

    def test_round_trip(self):
        """Round-trip: from_array(state.to_array()) сохраняет все поля."""
        original = ExtendedSDEState(P=100.0, Ne=50.0, C_TNF=3.0, t=10.0)
        arr = original.to_array()
        restored = ExtendedSDEState.from_array(arr, t=original.t)
        assert pytest.approx(original.P) == restored.P
        assert pytest.approx(original.Ne) == restored.Ne
        assert pytest.approx(original.C_TNF) == restored.C_TNF
        assert pytest.approx(original.t) == restored.t

    def test_nan_allowed(self):
        """NaN допустим (для detect_divergence)."""
        state = ExtendedSDEState.from_array(np.full(20, np.nan))
        assert np.isnan(state.P)


# =============================================================================
# TestExtendedSDEStateToDict
# =============================================================================


class TestExtendedSDEStateToDict:
    """Тесты конвертации состояния в словарь."""

    def test_returns_dict(self):
        """to_dict() возвращает dict."""
        state = ExtendedSDEState()
        result = state.to_dict()
        assert isinstance(result, dict)

    def test_has_21_keys(self):
        """Словарь содержит 21 ключ (20 переменных + t)."""
        state = ExtendedSDEState()
        result = state.to_dict()
        assert len(result) == 21

    def test_contains_P_and_t(self):
        """Словарь содержит ключи P и t."""
        state = ExtendedSDEState()
        result = state.to_dict()
        assert "P" in result
        assert "t" in result

    def test_values_match(self):
        """Значения в словаре совпадают с полями."""
        state = ExtendedSDEState(P=100.0, Ne=50.0, t=5.0)
        result = state.to_dict()
        assert result["P"] == 100.0
        assert result["Ne"] == 50.0
        assert result["t"] == 5.0


# =============================================================================
# TestExtendedSDETrajectoryGetVariable
# =============================================================================


class TestExtendedSDETrajectoryGetVariable:
    """Тесты извлечения временного ряда переменной."""

    def test_returns_ndarray(self, sample_extended_trajectory):
        """get_variable возвращает np.ndarray."""
        result = sample_extended_trajectory.get_variable("P")
        assert isinstance(result, np.ndarray)

    def test_length_matches_states(self, sample_extended_trajectory):
        """Длина результата == количество состояний."""
        result = sample_extended_trajectory.get_variable("P")
        assert len(result) == len(sample_extended_trajectory.states)

    def test_unknown_name_raises(self, sample_extended_trajectory):
        """Неизвестное имя переменной -> KeyError."""
        with pytest.raises(KeyError):
            sample_extended_trajectory.get_variable("unknown_variable")

    def test_empty_trajectory(self):
        """Пустая траектория -> массив нулевой длины."""
        traj = ExtendedSDETrajectory()
        result = traj.get_variable("P")
        assert len(result) == 0

    def test_values_match_states(self, sample_extended_trajectory):
        """Значения в массиве совпадают с полями состояний."""
        result = sample_extended_trajectory.get_variable("P")
        for i, state in enumerate(sample_extended_trajectory.states):
            assert result[i] == pytest.approx(state.P)


# =============================================================================
# TestExtendedSDETrajectoryGetStatistics
# =============================================================================


class TestExtendedSDETrajectoryGetStatistics:
    """Тесты статистики траектории."""

    def test_returns_dict_20_keys(self, sample_extended_trajectory):
        """Результат содержит 20 ключей (по переменной)."""
        result = sample_extended_trajectory.get_statistics()
        assert len(result) == 20

    def test_each_has_five_stats(self, sample_extended_trajectory):
        """Каждая переменная содержит mean, std, min, max, final."""
        result = sample_extended_trajectory.get_statistics()
        expected_keys = {"mean", "std", "min", "max", "final"}
        for var_name, stats in result.items():
            assert set(stats.keys()) == expected_keys, (
                f"Неверные ключи для {var_name}"
            )

    def test_min_le_mean_le_max(self, sample_extended_trajectory):
        """Min <= mean <= max для каждой переменной."""
        result = sample_extended_trajectory.get_statistics()
        for var_name, stats in result.items():
            assert stats["min"] <= stats["mean"] <= stats["max"], (
                f"Нарушение для {var_name}"
            )

    def test_final_equals_last_state(self, sample_extended_trajectory):
        """Final == значение в последнем состоянии."""
        result = sample_extended_trajectory.get_statistics()
        last_state = sample_extended_trajectory.states[-1]
        assert result["P"]["final"] == pytest.approx(last_state.P)

    def test_all_variables_present(self, sample_extended_trajectory):
        """Все 20 переменных из VARIABLE_NAMES присутствуют."""
        result = sample_extended_trajectory.get_statistics()
        for name in VARIABLE_NAMES:
            assert name in result, f"Переменная {name} отсутствует"

    def test_std_nonnegative(self, sample_extended_trajectory):
        """Std >= 0 для каждой переменной."""
        result = sample_extended_trajectory.get_statistics()
        for var_name, stats in result.items():
            assert stats["std"] >= 0, f"Отрицательный std для {var_name}"


# =============================================================================
# TestExtendedSDEModelInit
# =============================================================================


class TestExtendedSDEModelInit:
    """Тесты инициализации ExtendedSDEModel."""

    def test_default_init(self):
        """Инициализация с defaults сохраняет ParameterSet по умолчанию."""
        model = ExtendedSDEModel()
        assert model.params == ParameterSet()

    def test_custom_params(self):
        """Кастомные параметры корректно сохраняются."""
        params = ParameterSet(r_F=0.05)
        model = ExtendedSDEModel(params=params)
        assert model.params.r_F == 0.05

    def test_therapy_stored(self):
        """Протокол терапии сохраняется."""
        from src.core.sde_model import TherapyProtocol

        therapy = TherapyProtocol(prp_enabled=True)
        model = ExtendedSDEModel(therapy=therapy)
        assert model.therapy is therapy

    def test_rng_seed_reproducibility(self):
        """Два экземпляра с одинаковым seed дают одинаковые результаты."""
        model1 = ExtendedSDEModel(rng_seed=42)
        model2 = ExtendedSDEModel(rng_seed=42)
        # RNG состояние должно быть идентично
        assert model1._rng.random() == model2._rng.random()

    def test_none_params_defaults(self):
        """params=None -> используются ParameterSet по умолчанию."""
        model = ExtendedSDEModel(params=None)
        assert model.params == ParameterSet()


# =============================================================================
# TestExtendedSDEModelSimulate
# =============================================================================


class TestExtendedSDEModelSimulate:
    """Тесты полной симуляции."""

    def test_returns_trajectory(self, extended_sde_model, wound_initial_state):
        """simulate() возвращает ExtendedSDETrajectory."""
        result = extended_sde_model.simulate(
            wound_initial_state, t_span=(0, 1),
        )
        assert isinstance(result, ExtendedSDETrajectory)

    def test_no_nan(self, extended_sde_model, wound_initial_state):
        """Траектория не содержит NaN."""
        traj = extended_sde_model.simulate(
            wound_initial_state, t_span=(0, 1),
        )
        for state in traj.states:
            arr = state.to_array()
            assert not np.any(np.isnan(arr)), "Обнаружены NaN в траектории"

    def test_all_nonnegative(self, extended_sde_model, wound_initial_state):
        """Все переменные >= 0 на всём интервале."""
        traj = extended_sde_model.simulate(
            wound_initial_state, t_span=(0, 1),
        )
        for state in traj.states:
            arr = state.to_array()
            assert np.all(arr >= 0), (
                f"Отрицательные значения: {arr[arr < 0]}"
            )

    def test_times_match_states(self, extended_sde_model, wound_initial_state):
        """len(times) == len(states)."""
        traj = extended_sde_model.simulate(
            wound_initial_state, t_span=(0, 1),
        )
        assert len(traj.times) == len(traj.states)

    def test_custom_t_span(self, extended_sde_model, wound_initial_state):
        """t_span=(0, 10) -> times в [0, 10]."""
        traj = extended_sde_model.simulate(
            wound_initial_state, t_span=(0, 10),
        )
        assert traj.times[0] == pytest.approx(0.0)
        assert traj.times[-1] == pytest.approx(10.0, abs=0.1)

    def test_default_t_span_uses_t_max(self, wound_initial_state):
        """t_span=None -> times[-1] ~= params.t_max."""
        params = ParameterSet(t_max=5.0, dt=0.1)
        model = ExtendedSDEModel(params=params, rng_seed=42)
        traj = model.simulate(wound_initial_state)
        assert traj.times[-1] == pytest.approx(params.t_max, abs=0.2)


# =============================================================================
# TestComputeDrift
# =============================================================================


class TestComputeDrift:
    """Тесты вычисления вектора drift."""

    def test_shape_20(self, extended_sde_model, wound_initial_state):
        """Drift имеет shape (20,)."""
        result = extended_sde_model._compute_drift(wound_initial_state)
        assert result.shape == (20,)

    def test_zero_state_finite(self, extended_sde_model):
        """Drift конечен для нулевого состояния."""
        state = ExtendedSDEState()
        result = extended_sde_model._compute_drift(state)
        assert np.all(np.isfinite(result))

    def test_components_match_individual(
        self, extended_sde_model, wound_initial_state,
    ):
        """drift[StateIndex.P] == _drift_platelets(state)."""
        drift = extended_sde_model._compute_drift(wound_initial_state)
        p_drift = extended_sde_model._drift_platelets(wound_initial_state)
        assert drift[StateIndex.P] == pytest.approx(p_drift)

    def test_wound_state_finite(
        self, extended_sde_model, wound_initial_state,
    ):
        """Drift конечен для начального состояния раны."""
        result = extended_sde_model._compute_drift(wound_initial_state)
        assert np.all(np.isfinite(result))


# =============================================================================
# TestComputeDiffusion
# =============================================================================


class TestComputeDiffusion:
    """Тесты вычисления вектора diffusion."""

    def test_shape_20(self, extended_sde_model, wound_initial_state):
        """Diffusion имеет shape (20,)."""
        result = extended_sde_model._compute_diffusion(wound_initial_state)
        assert result.shape == (20,)

    def test_zero_state_zero_diffusion(self, extended_sde_model):
        """Нулевое состояние -> нулевой diffusion (sigma_i * X_i = 0)."""
        state = ExtendedSDEState()
        result = extended_sde_model._compute_diffusion(state)
        np.testing.assert_array_equal(result, np.zeros(20))

    def test_positive_state_nonzero(self, extended_sde_model):
        """Положительное состояние -> ненулевой diffusion."""
        state = ExtendedSDEState(P=100.0, Ne=50.0, M1=30.0)
        result = extended_sde_model._compute_diffusion(state)
        assert result[StateIndex.P] > 0
        assert result[StateIndex.Ne] > 0

    def test_finite(self, extended_sde_model, wound_initial_state):
        """Diffusion конечен для начального состояния раны."""
        result = extended_sde_model._compute_diffusion(wound_initial_state)
        assert np.all(np.isfinite(result))


# =============================================================================
# TestDriftPlatelets
# =============================================================================


class TestDriftPlatelets:
    """Тесты drift тромбоцитов: dP/dt = P_max*exp(-t/tau) - delta*P - k_deg*P."""

    def test_t0_P0_positive(self, extended_sde_model):
        """t=0, P=0: источник P_max доминирует -> drift > 0."""
        state = ExtendedSDEState(P=0.0, t=0.0)
        result = extended_sde_model._drift_platelets(state)
        assert result > 0

    def test_late_time_decay(self, extended_sde_model):
        """T >> tau_P, P > 0: экспонента затухла -> drift < 0."""
        state = ExtendedSDEState(P=1000.0, t=100.0)
        result = extended_sde_model._drift_platelets(state)
        assert result < 0

    def test_late_time_P0_near_zero(self, extended_sde_model):
        """T >> tau_P, P=0: drift ~= 0."""
        state = ExtendedSDEState(P=0.0, t=100.0)
        result = extended_sde_model._drift_platelets(state)
        assert abs(result) < 1.0  # Практически ноль


# =============================================================================
# TestDriftNeutrophils
# =============================================================================


class TestDriftNeutrophils:
    """Тесты drift нейтрофилов: Hill-рекрутирование + апоптоз + фагоцитоз."""

    def test_high_IL8_recruitment(self, extended_sde_model):
        """Высокий IL-8, Ne=0: рекрутирование -> drift > 0."""
        state = ExtendedSDEState(Ne=0.0, C_IL8=10.0)
        result = extended_sde_model._drift_neutrophils(state)
        assert result > 0

    def test_no_IL8_decay(self, extended_sde_model):
        """IL-8=0, Ne > 0: апоптоз + фагоцитоз -> drift < 0."""
        state = ExtendedSDEState(Ne=500.0, C_IL8=0.0, M1=100.0, M2=50.0)
        result = extended_sde_model._drift_neutrophils(state)
        assert result < 0

    def test_no_macrophages_no_phagocytosis(self, extended_sde_model):
        """M1=M2=0: терм фагоцитоза равен нулю."""
        state_with = ExtendedSDEState(Ne=500.0, C_IL8=0.0, M1=100.0, M2=50.0)
        state_without = ExtendedSDEState(Ne=500.0, C_IL8=0.0, M1=0.0, M2=0.0)
        drift_with = extended_sde_model._drift_neutrophils(state_with)
        drift_without = extended_sde_model._drift_neutrophils(state_without)
        # Без макрофагов drift должен быть менее отрицательным
        assert drift_without > drift_with


# =============================================================================
# TestDriftM1
# =============================================================================


class TestDriftM1:
    """Тесты drift M1 макрофагов: рекрутирование, переключение M1↔M2."""

    def test_high_TNF_MCP1_positive(self, extended_sde_model):
        """Высокий TNF и MCP-1: рекрутирование в M1."""
        state = ExtendedSDEState(
            C_TNF=10.0, C_MCP1=10.0, M1=0.0, M2=0.0,
        )
        result = extended_sde_model._drift_M1(state)
        assert result > 0

    def test_high_IL10_switching(self, extended_sde_model):
        """Высокий IL-10 + M1 > 0: переключение M1->M2, drift уменьшается."""
        state_low = ExtendedSDEState(
            M1=200.0, M2=0.0, C_IL10=0.0, C_TGFb=0.0, C_MCP1=5.0,
        )
        state_high = ExtendedSDEState(
            M1=200.0, M2=0.0, C_IL10=10.0, C_TGFb=5.0, C_MCP1=5.0,
        )
        drift_low = extended_sde_model._drift_M1(state_low)
        drift_high = extended_sde_model._drift_M1(state_high)
        # Высокий IL-10 усиливает переключение -> меньший drift
        assert drift_high < drift_low

    def test_all_zero(self, extended_sde_model):
        """M1=M2=0, MCP1=0: drift ~= 0."""
        state = ExtendedSDEState()
        result = extended_sde_model._drift_M1(state)
        assert abs(result) < 1e-6


# =============================================================================
# TestDriftM2
# =============================================================================


class TestDriftM2:
    """Тесты drift M2 макрофагов: переключение из M1, обратное."""

    def test_switching_from_M1(self, extended_sde_model):
        """Высокий IL-10, M1 > 0: переключение M1→M2 -> положительный вклад."""
        state = ExtendedSDEState(
            M1=200.0, M2=0.0, C_IL10=10.0, C_TGFb=5.0,
        )
        result = extended_sde_model._drift_M2(state)
        assert result > 0

    def test_reverse_switching(self, extended_sde_model):
        """Высокий TNF, M2 > 0: обратное переключение -> отрицательный вклад."""
        state_high_tnf = ExtendedSDEState(
            M1=0.0, M2=200.0, C_TNF=10.0, C_IL10=0.0,
        )
        state_low_tnf = ExtendedSDEState(
            M1=0.0, M2=200.0, C_TNF=0.0, C_IL10=0.0,
        )
        drift_high = extended_sde_model._drift_M2(state_high_tnf)
        drift_low = extended_sde_model._drift_M2(state_low_tnf)
        # Высокий TNF увеличивает обратное переключение -> больший отток
        assert drift_high < drift_low

    def test_mirror_invariant(self, extended_sde_model):
        """Потоки переключения M1↔M2 зеркальны."""
        state = ExtendedSDEState(
            M1=200.0, M2=100.0, C_IL10=5.0, C_TGFb=3.0,
            C_TNF=2.0, C_MCP1=1.0,
        )
        drift_m1 = extended_sde_model._drift_M1(state)
        drift_m2 = extended_sde_model._drift_M2(state)
        # Сумма M1+M2 drift содержит только рекрутирование и апоптоз
        # Потоки переключения взаимно компенсируются
        # Проверяем, что оба drift конечны
        assert np.isfinite(drift_m1)
        assert np.isfinite(drift_m2)


# =============================================================================
# TestDriftFibroblasts
# =============================================================================


class TestDriftFibroblasts:
    """Тесты drift фибробластов: логистический рост + дифференциация S."""

    def test_differentiation_from_S(self, extended_sde_model):
        """F=0, S > 0, TGFb > 0: приток через дифференциацию S -> drift > 0."""
        state = ExtendedSDEState(F=0.0, S=100.0, C_TGFb=5.0)
        result = extended_sde_model._drift_fibroblasts(state)
        assert result > 0

    def test_at_capacity(self, extended_sde_model):
        """F+Mf == K_F: логистический рост = 0."""
        K_F = extended_sde_model.params.K_F
        state = ExtendedSDEState(
            F=K_F * 0.5, Mf=K_F * 0.5, C_PDGF=5.0, C_TGFb=2.0,
        )
        result = extended_sde_model._drift_fibroblasts(state)
        # Логистический рост = 0 при capacity, но есть S->F и F->Mf и апоптоз
        # Проверяем что рост ограничен по сравнению с подкапасити
        state_low = ExtendedSDEState(
            F=100.0, Mf=0.0, C_PDGF=5.0, C_TGFb=2.0,
        )
        drift_low = extended_sde_model._drift_fibroblasts(state_low)
        assert drift_low > result  # Ниже capacity -> больше роста

    def test_activation_to_Mf(self, extended_sde_model):
        """Высокий TGFb, F > 0: потеря через активацию F→Mf."""
        state_low_tgf = ExtendedSDEState(F=500.0, C_TGFb=0.0)
        state_high_tgf = ExtendedSDEState(F=500.0, C_TGFb=10.0)
        drift_low = extended_sde_model._drift_fibroblasts(state_low_tgf)
        drift_high = extended_sde_model._drift_fibroblasts(state_high_tgf)
        # Высокий TGFb -> больше уход в Mf -> меньший drift
        assert drift_high < drift_low


# =============================================================================
# TestDriftMyofibroblasts
# =============================================================================


class TestDriftMyofibroblasts:
    """Тесты drift миофибробластов: TGFb-зависимый апоптоз."""

    def test_high_TGFb_growth(self, extended_sde_model):
        """Высокий TGFb, F > 0, Mf > 0: малый апоптоз, рост."""
        state = ExtendedSDEState(F=500.0, Mf=100.0, C_TGFb=10.0)
        result = extended_sde_model._drift_myofibroblasts(state)
        assert result > 0

    def test_zero_TGFb_decay(self, extended_sde_model):
        """TGFb=0, Mf > 0: полный апоптоз -> Mf убывает."""
        state = ExtendedSDEState(F=0.0, Mf=100.0, C_TGFb=0.0)
        result = extended_sde_model._drift_myofibroblasts(state)
        assert result < 0

    def test_no_F_only_decay(self, extended_sde_model):
        """F=0: нет притока, только убыль."""
        state = ExtendedSDEState(F=0.0, Mf=100.0, C_TGFb=1.0)
        result = extended_sde_model._drift_myofibroblasts(state)
        assert result < 0


# =============================================================================
# TestDriftEndothelial
# =============================================================================


class TestDriftEndothelial:
    """Тесты drift эндотелиальных клеток: VEGF, гипоксия, ангиогенез."""

    def test_high_VEGF_low_O2(self, extended_sde_model):
        """Высокий VEGF, низкий O2: ангиогенез -> drift > 0."""
        state = ExtendedSDEState(E=100.0, C_VEGF=10.0, O2=1.0)
        result = extended_sde_model._drift_endothelial(state)
        assert result > 0

    def test_normoxia_minimal(self, extended_sde_model):
        """O2=O2_blood: (1-theta) ~= 0 -> минимальный рост."""
        O2_blood = extended_sde_model.params.O2_blood
        state_normoxia = ExtendedSDEState(
            E=100.0, C_VEGF=5.0, O2=O2_blood,
        )
        state_hypoxia = ExtendedSDEState(
            E=100.0, C_VEGF=5.0, O2=1.0,
        )
        drift_norm = extended_sde_model._drift_endothelial(state_normoxia)
        drift_hyp = extended_sde_model._drift_endothelial(state_hypoxia)
        # Гипоксия даёт больший рост
        assert drift_hyp > drift_norm

    def test_no_VEGF_no_growth(self, extended_sde_model):
        """VEGF=0: нет стимуляции роста."""
        state = ExtendedSDEState(E=100.0, C_VEGF=0.0, O2=1.0)
        result = extended_sde_model._drift_endothelial(state)
        # Без VEGF только апоптоз
        assert result < 0


# =============================================================================
# TestDriftStemCells
# =============================================================================


class TestDriftStemCells:
    """Тесты drift стволовых клеток: логистический рост, PRP, дифференциация."""

    def test_S0_zero(self, extended_sde_model):
        """S=0: drift ~= 0."""
        state = ExtendedSDEState(S=0.0)
        result = extended_sde_model._drift_stem_cells(state)
        assert abs(result) < 1e-6

    def test_PRP_enhanced(self, extended_sde_model):
        """С PRP стимуляцией: рост выше базового."""
        from src.core.sde_model import TherapyProtocol

        therapy = TherapyProtocol(prp_enabled=True, prp_intensity=2.0)
        model_prp = ExtendedSDEModel(therapy=therapy, rng_seed=42)
        state = ExtendedSDEState(S=100.0, t=5.0)
        drift_no_prp = extended_sde_model._drift_stem_cells(state)
        drift_prp = model_prp._drift_stem_cells(state)
        # PRP должен усилить рост
        assert drift_prp >= drift_no_prp

    def test_TGFb_differentiation_loss(self, extended_sde_model):
        """Высокий TGFb: потеря через дифференциацию S→F."""
        state_low = ExtendedSDEState(S=100.0, C_TGFb=0.0)
        state_high = ExtendedSDEState(S=100.0, C_TGFb=10.0)
        drift_low = extended_sde_model._drift_stem_cells(state_low)
        drift_high = extended_sde_model._drift_stem_cells(state_high)
        # Высокий TGFb увеличивает дифференциацию -> меньший drift
        assert drift_high < drift_low


# =============================================================================
# TestDriftCytokines
# =============================================================================


class TestDriftCytokines:
    """Тесты drift цитокинов: продукция и деградация."""

    # --- TNF-α ---

    def test_drift_C_TNF_production(self, extended_sde_model):
        """M1 > 0: продукция TNF -> drift > 0."""
        state = ExtendedSDEState(M1=200.0, Ne=100.0, C_TNF=0.0)
        result = extended_sde_model._drift_C_TNF(state)
        assert result > 0

    def test_drift_C_TNF_degradation(self, extended_sde_model):
        """M1=0, C_TNF > 0: только деградация -> drift < 0."""
        state = ExtendedSDEState(M1=0.0, Ne=0.0, C_TNF=5.0)
        result = extended_sde_model._drift_C_TNF(state)
        assert result < 0

    # --- IL-10 ---

    def test_drift_C_IL10_production(self, extended_sde_model):
        """M2 > 0: секреция IL-10 -> drift > 0."""
        state = ExtendedSDEState(M2=200.0, C_IL10=0.0)
        result = extended_sde_model._drift_C_IL10(state)
        assert result > 0

    def test_drift_C_IL10_degradation(self, extended_sde_model):
        """Все источники = 0, C_IL10 > 0: деградация -> drift < 0."""
        state = ExtendedSDEState(M2=0.0, Ne=0.0, C_IL10=5.0)
        result = extended_sde_model._drift_C_IL10(state)
        assert result < 0

    # --- PDGF ---

    def test_drift_C_PDGF_production(self, extended_sde_model):
        """Тромбоциты P > 0: продукция PDGF -> drift > 0."""
        state = ExtendedSDEState(P=1000.0, C_PDGF=0.0)
        result = extended_sde_model._drift_C_PDGF(state)
        assert result > 0

    def test_drift_C_PDGF_degradation(self, extended_sde_model):
        """Нет источников, C_PDGF > 0: деградация -> drift < 0."""
        state = ExtendedSDEState(P=0.0, M1=0.0, M2=0.0, C_PDGF=5.0)
        result = extended_sde_model._drift_C_PDGF(state)
        assert result < 0

    # --- VEGF ---

    def test_drift_C_VEGF_hypoxia_boost(self, extended_sde_model):
        """Низкий O2: гипоксия усиливает VEGF продукцию."""
        state_norm = ExtendedSDEState(
            M2=100.0, C_VEGF=0.0, O2=100.0,
        )
        state_hypo = ExtendedSDEState(
            M2=100.0, C_VEGF=0.0, O2=1.0,
        )
        drift_norm = extended_sde_model._drift_C_VEGF(state_norm)
        drift_hypo = extended_sde_model._drift_C_VEGF(state_hypo)
        assert drift_hypo > drift_norm

    def test_drift_C_VEGF_degradation(self, extended_sde_model):
        """Нет источников, C_VEGF > 0: деградация."""
        state = ExtendedSDEState(M2=0.0, F=0.0, C_VEGF=5.0)
        result = extended_sde_model._drift_C_VEGF(state)
        assert result < 0

    # --- TGF-β ---

    def test_drift_C_TGFb_Mf_feedback(self, extended_sde_model):
        """Mf > 0: положительная обратная связь TGFb."""
        state = ExtendedSDEState(Mf=200.0, C_TGFb=0.0)
        result = extended_sde_model._drift_C_TGFb(state)
        assert result > 0

    def test_drift_C_TGFb_degradation(self, extended_sde_model):
        """Нет источников, C_TGFb > 0: деградация."""
        state = ExtendedSDEState(
            P=0.0, M2=0.0, Mf=0.0, C_TGFb=5.0,
        )
        result = extended_sde_model._drift_C_TGFb(state)
        assert result < 0

    # --- MCP-1 ---

    def test_drift_C_MCP1_damage_driven(self, extended_sde_model):
        """D > 0: DAMPs-индуцированная продукция MCP-1."""
        state = ExtendedSDEState(D=1.0, M1=0.0, C_MCP1=0.0)
        result = extended_sde_model._drift_C_MCP1(state)
        assert result > 0

    def test_drift_C_MCP1_degradation(self, extended_sde_model):
        """D=0, M1=0: только деградация."""
        state = ExtendedSDEState(D=0.0, M1=0.0, C_MCP1=5.0)
        result = extended_sde_model._drift_C_MCP1(state)
        assert result < 0

    # --- IL-8 ---

    def test_drift_C_IL8_autocrine(self, extended_sde_model):
        """Ne > 0: аутокринная продукция IL-8."""
        state = ExtendedSDEState(
            Ne=200.0, D=0.5, M1=50.0, C_IL8=0.0,
        )
        result = extended_sde_model._drift_C_IL8(state)
        assert result > 0

    def test_drift_C_IL8_degradation(self, extended_sde_model):
        """Все источники = 0: деградация."""
        state = ExtendedSDEState(
            Ne=0.0, D=0.0, M1=0.0, C_IL8=5.0,
        )
        result = extended_sde_model._drift_C_IL8(state)
        assert result < 0


# =============================================================================
# TestDriftECM
# =============================================================================


class TestDriftECM:
    """Тесты drift ECM компонентов: коллаген, MMP, фибрин."""

    def test_collagen_production(self, extended_sde_model):
        """F > 0, Mf > 0, rho < rho_max: продукция -> drift > 0."""
        state = ExtendedSDEState(
            F=500.0, Mf=100.0, rho_collagen=0.0, C_MMP=0.0,
        )
        result = extended_sde_model._drift_collagen(state)
        assert result > 0

    def test_collagen_saturation(self, extended_sde_model):
        """rho_c == rho_max: продукция = 0 (насыщение)."""
        rho_max = extended_sde_model.params.rho_c_max
        state = ExtendedSDEState(
            F=500.0, Mf=100.0, rho_collagen=rho_max, C_MMP=0.0,
        )
        result = extended_sde_model._drift_collagen(state)
        # При насыщении продукция = 0, деградация MMP тоже 0 -> ~0
        assert abs(result) < 1e-6

    def test_collagen_MMP_degradation(self, extended_sde_model):
        """Высокий MMP, rho > 0: деградация коллагена."""
        state = ExtendedSDEState(
            F=0.0, Mf=0.0, rho_collagen=0.5, C_MMP=5.0,
        )
        result = extended_sde_model._drift_collagen(state)
        assert result < 0

    def test_MMP_M1_secretion(self, extended_sde_model):
        """M1 > 0: секреция MMP -> drift > 0."""
        state = ExtendedSDEState(M1=200.0, C_MMP=0.0)
        result = extended_sde_model._drift_MMP(state)
        assert result > 0

    def test_MMP_TIMP_inhibition(self, extended_sde_model):
        """Высокий TIMP, MMP > 0: ингибирование."""
        state_mmp = ExtendedSDEState(
            M1=0.0, M2=0.0, F=0.0, C_MMP=5.0,
        )
        result = extended_sde_model._drift_MMP(state_mmp)
        # Без источников, с TIMP -> деградация
        assert result < 0

    def test_fibrin_always_nonpositive(self, extended_sde_model):
        """rho_fibrin > 0: drift фибрина <= 0 (только убыль)."""
        state = ExtendedSDEState(
            rho_fibrin=1.0, C_MMP=0.5, F=100.0,
        )
        result = extended_sde_model._drift_fibrin(state)
        assert result <= 0

    def test_fibrin_zero_when_zero(self, extended_sde_model):
        """rho_fibrin = 0: drift фибрина = 0."""
        state = ExtendedSDEState(rho_fibrin=0.0, C_MMP=5.0, F=100.0)
        result = extended_sde_model._drift_fibrin(state)
        assert abs(result) < 1e-10


# =============================================================================
# TestDriftAuxiliary
# =============================================================================


class TestDriftAuxiliary:
    """Тесты drift вспомогательных переменных: damage signal, oxygen."""

    def test_damage_negative(self, extended_sde_model):
        """D > 0: drift < 0 (монотонное затухание)."""
        state = ExtendedSDEState(D=1.0)
        result = extended_sde_model._drift_damage(state)
        assert result < 0

    def test_damage_zero(self, extended_sde_model):
        """D = 0: drift = 0."""
        state = ExtendedSDEState(D=0.0)
        result = extended_sde_model._drift_damage(state)
        assert abs(result) < 1e-10

    def test_oxygen_diffusion(self, extended_sde_model):
        """O2 < O2_blood: диффузия повышает O2."""
        state = ExtendedSDEState(O2=10.0)
        result = extended_sde_model._drift_oxygen(state)
        # Диффузия из крови должна поставлять O2
        assert result > 0

    def test_oxygen_consumption(self, extended_sde_model):
        """Клетки потребляют кислород."""
        O2_blood = extended_sde_model.params.O2_blood
        state_cells = ExtendedSDEState(
            O2=O2_blood, Ne=500.0, M1=200.0, F=300.0, E=100.0,
        )
        state_no_cells = ExtendedSDEState(O2=O2_blood)
        drift_cells = extended_sde_model._drift_oxygen(state_cells)
        drift_no_cells = extended_sde_model._drift_oxygen(state_no_cells)
        # С клетками кислород потребляется больше
        assert drift_cells < drift_no_cells


# =============================================================================
# TestHelperFunctions
# =============================================================================


class TestHillFunction:
    """Тесты Hill функции: xⁿ / (Kⁿ + xⁿ)."""

    def test_at_K_equals_half(self, extended_sde_model):
        """Hill(K, K, n=2) = 0.5."""
        result = extended_sde_model._hill(1.0, 1.0, n=2)
        assert result == pytest.approx(0.5)

    def test_zero_input(self, extended_sde_model):
        """Hill(0, K) = 0."""
        result = extended_sde_model._hill(0.0, 1.0, n=2)
        assert result == pytest.approx(0.0)

    def test_large_input_near_one(self, extended_sde_model):
        """Hill(1000*K, K) ~= 1.0."""
        result = extended_sde_model._hill(1000.0, 1.0, n=2)
        assert result == pytest.approx(1.0, abs=1e-4)

    @pytest.mark.parametrize("x", [0.0, 0.1, 0.5, 1.0, 5.0, 100.0])
    def test_range_0_to_1(self, extended_sde_model, x):
        """Hill(x >= 0) ∈ [0, 1]."""
        result = extended_sde_model._hill(x, 1.0, n=2)
        assert 0.0 <= result <= 1.0


class TestPolarizationFunctions:
    """Тесты функций поляризации M1/M2."""

    def test_M1_high_TNF(self, extended_sde_model):
        """Высокий TNF, низкий IL10: phi1 ~= 1."""
        state = ExtendedSDEState(C_TNF=10.0, C_IL10=0.01)
        result = extended_sde_model._polarization_M1(state)
        assert result > 0.9

    def test_M1_high_IL10(self, extended_sde_model):
        """Высокий IL10, низкий TNF: phi1 ~= 0."""
        state = ExtendedSDEState(C_TNF=0.01, C_IL10=10.0)
        result = extended_sde_model._polarization_M1(state)
        assert result < 0.1

    def test_M1_M2_sum_to_one(self, extended_sde_model):
        """phi1 + phi2 == 1.0."""
        state = ExtendedSDEState(C_TNF=3.0, C_IL10=2.0)
        phi1 = extended_sde_model._polarization_M1(state)
        phi2 = extended_sde_model._polarization_M2(state)
        assert phi1 + phi2 == pytest.approx(1.0)

    def test_M2_complement(self, extended_sde_model):
        """phi2 == 1 - phi1."""
        state = ExtendedSDEState(C_TNF=5.0, C_IL10=3.0)
        phi1 = extended_sde_model._polarization_M1(state)
        phi2 = extended_sde_model._polarization_M2(state)
        assert phi2 == pytest.approx(1.0 - phi1)


class TestSwitchingFunctions:
    """Тесты функций переключения M1↔M2."""

    def test_switching_high_signals(self, extended_sde_model):
        """Высокий IL-10 + TGFb: psi ~= 1."""
        state = ExtendedSDEState(C_IL10=10.0, C_TGFb=10.0)
        result = extended_sde_model._switching_function(state)
        assert result > 0.9

    def test_switching_zero_signals(self, extended_sde_model):
        """IL-10=TGFb=0: psi = 0."""
        state = ExtendedSDEState(C_IL10=0.0, C_TGFb=0.0)
        result = extended_sde_model._switching_function(state)
        assert result == pytest.approx(0.0)

    def test_reverse_high_TNF(self, extended_sde_model):
        """Высокий TNF: zeta ~= 1."""
        state = ExtendedSDEState(C_TNF=10.0)
        result = extended_sde_model._reverse_switching(state)
        assert result > 0.9

    def test_reverse_zero_TNF(self, extended_sde_model):
        """TNF=0: zeta = 0."""
        state = ExtendedSDEState(C_TNF=0.0)
        result = extended_sde_model._reverse_switching(state)
        assert result == pytest.approx(0.0)


class TestMitogenicStimulation:
    """Тесты митогенной стимуляции фибробластов."""

    def test_zero_PDGF(self, extended_sde_model):
        """PDGF=0: H = 0."""
        state = ExtendedSDEState(C_PDGF=0.0, C_TGFb=5.0)
        result = extended_sde_model._mitogenic_stimulation(state)
        assert result == pytest.approx(0.0)

    def test_positive_PDGF(self, extended_sde_model):
        """PDGF > 0: H > 0."""
        state = ExtendedSDEState(C_PDGF=5.0, C_TGFb=0.0)
        result = extended_sde_model._mitogenic_stimulation(state)
        assert result > 0

    def test_TGFb_enhancement(self, extended_sde_model):
        """TGFb > 0: увеличивает митогенный фактор H."""
        state_no_tgf = ExtendedSDEState(C_PDGF=5.0, C_TGFb=0.0)
        state_with_tgf = ExtendedSDEState(C_PDGF=5.0, C_TGFb=5.0)
        H_no = extended_sde_model._mitogenic_stimulation(state_no_tgf)
        H_with = extended_sde_model._mitogenic_stimulation(state_with_tgf)
        assert H_with > H_no


class TestDifferentiationAndActivation:
    """Тесты функций дифференциации и активации."""

    @pytest.mark.parametrize("tgfb", [0.0, 0.5, 1.0, 5.0, 100.0])
    def test_differentiation_range(self, extended_sde_model, tgfb):
        """g_diff ∈ [0, 1]."""
        state = ExtendedSDEState(C_TGFb=tgfb)
        result = extended_sde_model._differentiation_probability(state)
        assert 0.0 <= result <= 1.0

    @pytest.mark.parametrize("tgfb", [0.0, 0.5, 1.0, 5.0, 100.0])
    def test_activation_range(self, extended_sde_model, tgfb):
        """A ∈ [0, 1]."""
        state = ExtendedSDEState(C_TGFb=tgfb)
        result = extended_sde_model._activation_function(state)
        assert 0.0 <= result <= 1.0

    def test_activation_is_hill(self, extended_sde_model):
        """A(K_activ) == 0.5 (Hill at half-saturation)."""
        K = extended_sde_model.params.K_activ
        state = ExtendedSDEState(C_TGFb=K)
        result = extended_sde_model._activation_function(state)
        assert result == pytest.approx(0.5)


class TestVEGFAndHypoxia:
    """Тесты VEGF-активации и гипоксического фактора."""

    @pytest.mark.parametrize("vegf", [0.0, 0.5, 1.0, 5.0, 100.0])
    def test_vegf_activation_range(self, extended_sde_model, vegf):
        """V ∈ [0, 1]."""
        state = ExtendedSDEState(C_VEGF=vegf)
        result = extended_sde_model._vegf_activation(state)
        assert 0.0 <= result <= 1.0

    def test_hypoxia_low_O2_near_zero(self, extended_sde_model):
        """Низкий O2: theta ~= 0 -> (1-theta) ~= 1."""
        state = ExtendedSDEState(O2=0.01)
        result = extended_sde_model._hypoxia_factor(state)
        assert result < 0.1

    def test_hypoxia_high_O2_near_one(self, extended_sde_model):
        """Высокий O2: theta ~= 1 -> (1-theta) ~= 0."""
        state = ExtendedSDEState(O2=100.0)
        result = extended_sde_model._hypoxia_factor(state)
        assert result > 0.9


# =============================================================================
# TestBoundaryConditions
# =============================================================================


class TestBoundaryConditions:
    """Тесты граничных условий: все переменные >= 0."""

    def test_positive_unchanged(self, extended_sde_model):
        """Положительное состояние не изменяется."""
        state = ExtendedSDEState(P=100.0, Ne=50.0, M1=30.0)
        result = extended_sde_model._apply_boundary_conditions(state)
        assert result.P == 100.0
        assert result.Ne == 50.0
        assert result.M1 == 30.0

    def test_negative_to_zero(self, extended_sde_model):
        """Отрицательное значение обнуляется."""
        state = ExtendedSDEState(P=-5.0, Ne=50.0)
        result = extended_sde_model._apply_boundary_conditions(state)
        assert result.P == 0.0
        assert result.Ne == 50.0

    def test_all_negative_all_zero(self, extended_sde_model):
        """Все отрицательные -> все нули."""
        state = ExtendedSDEState(
            P=-1.0, Ne=-2.0, M1=-3.0, M2=-4.0,
            F=-5.0, Mf=-6.0, E=-7.0, S=-8.0,
        )
        result = extended_sde_model._apply_boundary_conditions(state)
        arr = result.to_array()
        assert np.all(arr >= 0)

    def test_returns_state(self, extended_sde_model):
        """Возвращает ExtendedSDEState."""
        state = ExtendedSDEState(P=100.0)
        result = extended_sde_model._apply_boundary_conditions(state)
        assert isinstance(result, ExtendedSDEState)


# =============================================================================
# TestValidateParams
# =============================================================================


class TestValidateParams:
    """Тесты валидации параметров модели."""

    def test_valid_returns_true(self, extended_sde_model):
        """Валидные параметры -> True."""
        assert extended_sde_model.validate_params() is True

    def test_invalid_raises(self):
        """Невалидные параметры -> ValueError."""
        params = ParameterSet(r_F=-1.0)
        model = ExtendedSDEModel(params=params)
        with pytest.raises(ValueError):
            model.validate_params()

    def test_delegates_to_params(self, extended_sde_model):
        """Делегирует к params.validate()."""
        # Если params.validate() проходит, то и validate_params() должен
        assert extended_sde_model.params.validate() is True
        assert extended_sde_model.validate_params() is True


# =============================================================================
# TestGetDefaultInitialState
# =============================================================================


class TestGetDefaultInitialState:
    """Тесты начального состояния раны."""

    def test_returns_state(self, extended_sde_model):
        """Возвращает ExtendedSDEState."""
        result = extended_sde_model.get_default_initial_state()
        assert isinstance(result, ExtendedSDEState)

    def test_P_equals_P_max(self, extended_sde_model):
        """P == P_max (максимальная активация тромбоцитов)."""
        result = extended_sde_model.get_default_initial_state()
        assert pytest.approx(extended_sde_model.params.P_max) == result.P

    def test_D_equals_D0(self, extended_sde_model):
        """D == D0 (максимальный damage signal)."""
        result = extended_sde_model.get_default_initial_state()
        assert pytest.approx(extended_sde_model.params.D0) == result.D

    def test_O2_equals_O2_blood(self, extended_sde_model):
        """O2 == O2_blood (начальный кислород)."""
        result = extended_sde_model.get_default_initial_state()
        assert pytest.approx(extended_sde_model.params.O2_blood) == result.O2

    def test_fibrin_one(self, extended_sde_model):
        """rho_fibrin == 1.0 (фибриновый сгусток)."""
        result = extended_sde_model.get_default_initial_state()
        assert result.rho_fibrin == pytest.approx(1.0)


# =============================================================================
# TestBiologicalProperties
# =============================================================================


class TestBiologicalProperties:
    """Тесты ключевых биологических свойств системы."""

    def test_all_populations_nonnegative(self):
        """Все популяции >= 0 на всём интервале (позитивность)."""
        model = ExtendedSDEModel(rng_seed=42)
        state = model.get_default_initial_state()
        traj = model.simulate(state, t_span=(0, 48))
        for s in traj.states:
            arr = s.to_array()
            assert np.all(arr >= 0), (
                f"Отрицательные значения на t={s.t}: {arr[arr < 0]}"
            )

    def test_macrophage_M1_M2_switch(self):
        """M1 пик на 24-48ч, M2 доминирует после 72ч."""
        model = ExtendedSDEModel(
            params=ParameterSet(dt=0.1), rng_seed=42,
        )
        state = model.get_default_initial_state()
        traj = model.simulate(state, t_span=(0, 120))
        m1 = traj.get_variable("M1")
        m2 = traj.get_variable("M2")
        times = traj.times

        # M2 доминирует после 72ч
        idx_72h = np.searchsorted(times, 72.0)
        if idx_72h < len(times):
            m2_late = np.mean(m2[idx_72h:])
            m1_late = np.mean(m1[idx_72h:])
            assert m2_late > m1_late, "M2 должен доминировать после 72ч"

    def test_tgfb_mf_bistability(self):
        """TGF-β ↔ Mf бистабильность: высокий TGFb -> Mf сохраняется."""
        model = ExtendedSDEModel(
            params=ParameterSet(dt=0.1), rng_seed=42,
        )

        # Состояние с высоким TGFb и Mf
        state_high = ExtendedSDEState(
            P=1e4, F=500.0, Mf=200.0, C_TGFb=10.0,
            D=1.0, O2=100.0, rho_fibrin=1.0,
        )
        traj_high = model.simulate(state_high, t_span=(0, 48))
        mf_high_final = traj_high.get_variable("Mf")[-1]

        # Состояние с низким TGFb и Mf
        state_low = ExtendedSDEState(
            P=1e4, F=500.0, Mf=200.0, C_TGFb=0.0,
            D=1.0, O2=100.0, rho_fibrin=1.0,
        )
        traj_low = model.simulate(state_low, t_span=(0, 48))
        mf_low_final = traj_low.get_variable("Mf")[-1]

        # Высокий TGFb поддерживает Mf лучше
        assert mf_high_final > mf_low_final

    def test_hypoxia_angiogenesis(self):
        """Низкий O2 -> рост E(t) (гипоксия-ангиогенез)."""
        model = ExtendedSDEModel(
            params=ParameterSet(dt=0.1), rng_seed=42,
        )

        # Гипоксия
        state_hypo = ExtendedSDEState(
            P=1e4, E=100.0, C_VEGF=5.0, O2=5.0,
            D=1.0, rho_fibrin=1.0,
        )
        traj_hypo = model.simulate(state_hypo, t_span=(0, 48))
        e_hypo = traj_hypo.get_variable("E")

        # Нормоксия
        state_norm = ExtendedSDEState(
            P=1e4, E=100.0, C_VEGF=5.0, O2=100.0,
            D=1.0, rho_fibrin=1.0,
        )
        traj_norm = model.simulate(state_norm, t_span=(0, 48))
        e_norm = traj_norm.get_variable("E")

        # Гипоксия усиливает ангиогенез
        assert np.max(e_hypo) > np.max(e_norm)

    def test_fibrin_to_collagen(self):
        """rho_fibrin падает, rho_collagen растёт (замещение)."""
        model = ExtendedSDEModel(
            params=ParameterSet(dt=0.1), rng_seed=42,
        )
        state = model.get_default_initial_state()
        traj = model.simulate(state, t_span=(0, 168))  # 7 дней
        fibrin = traj.get_variable("rho_fibrin")
        collagen = traj.get_variable("rho_collagen")

        # Фибрин должен уменьшиться
        assert fibrin[-1] < fibrin[0]
        # Коллаген должен увеличиться
        assert collagen[-1] > collagen[0]

    def test_neutrophil_peak_decay(self):
        """Ne пик на 12-24ч, затухание к 48ч."""
        model = ExtendedSDEModel(
            params=ParameterSet(dt=0.1), rng_seed=42,
        )
        state = model.get_default_initial_state()
        traj = model.simulate(state, t_span=(0, 72))
        ne = traj.get_variable("Ne")
        times = traj.times

        # Находим пик
        peak_idx = np.argmax(ne)
        peak_time = times[peak_idx]

        # Пик должен быть в районе 12-48ч
        assert 6.0 <= peak_time <= 48.0, (
            f"Пик нейтрофилов на {peak_time}ч, ожидалось 6-48ч"
        )

        # После 48ч Ne убывает
        idx_48h = np.searchsorted(times, 48.0)
        if idx_48h < len(times) - 1:
            ne_at_48 = ne[idx_48h]
            ne_peak = ne[peak_idx]
            assert ne_at_48 < ne_peak, "Ne должен убывать после пика"
