# extended_sde.py — Расширенная 20-переменная SDE модель

## Назначение

Полная система стохастических дифференциальных уравнений для моделирования
регенерации тканей. 20 переменных покрывают все фазы заживления:
гемостаз, воспаление, пролиферацию, ремоделирование.

Математическое обоснование: `Doks/RegenTwin_Mathematical_Framework.md` §2.

Подробное описание: Description/Phase2/description_extended_sde.md

---

## StateIndex (IntEnum)

**Назначение:** Индексы 20 переменных в numpy-массиве состояния.

| Индекс | Имя | Описание |
|--------|-----|----------|
| 0 | P | Тромбоциты |
| 1 | Ne | Нейтрофилы |
| 2 | M1 | M1 макрофаги |
| 3 | M2 | M2 макрофаги |
| 4 | F | Фибробласты |
| 5 | Mf | Миофибробласты |
| 6 | E | Эндотелиальные |
| 7 | S | Стволовые (CD34+) |
| 8 | C_TNF | TNF-α |
| 9 | C_IL10 | IL-10 |
| 10 | C_PDGF | PDGF |
| 11 | C_VEGF | VEGF |
| 12 | C_TGFb | TGF-β |
| 13 | C_MCP1 | MCP-1 |
| 14 | C_IL8 | IL-8 |
| 15 | RHO_COLLAGEN | Коллаген |
| 16 | C_MMP | MMP |
| 17 | RHO_FIBRIN | Фибрин |
| 18 | D | Сигнал повреждения |
| 19 | O2 | Кислород |

**Инварианты:**
- len(StateIndex) == 20
- Значения 0..19 без пропусков

---

## ExtendedSDEState

**Назначение:** Состояние системы в момент времени t.

**Поля:** 20 переменных (float, default=0.0) + t (float, default=0.0).

### to_array

**Сигнатура:**
```python
def to_array(self) -> np.ndarray
```

**Поведение:** Вернуть np.array([P, Ne, ..., O2]) shape (20,).

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| ExtendedSDEState().to_array() | np.zeros(20) |
| State с P=100.to_array()[0] | 100.0 |
| Порядок элементов | Соответствует StateIndex |

**Инварианты:**
- result.shape == (20,)
- result[StateIndex.P] == self.P

### from_array

**Сигнатура:**
```python
@classmethod
def from_array(cls, arr: np.ndarray, t: float = 0.0) -> ExtendedSDEState
```

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| from_array(np.zeros(20)) | Все поля == 0.0 |
| from_array(np.ones(20), t=5.0) | Все поля == 1.0, t == 5.0 |
| from_array(np.zeros(19)) | ValueError |

**Edge cases:**
- len(arr) != 20 → ValueError
- arr содержит NaN → допустимо (для detect_divergence)

**Инварианты:**
- from_array(state.to_array(), state.t) ≈ state (round-trip)

### to_dict

**Сигнатура:**
```python
def to_dict(self) -> dict[str, float]
```

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| ExtendedSDEState().to_dict() | dict с 21 ключом |
| to_dict()["P"] | self.P |
| to_dict()["t"] | self.t |

**Инварианты:**
- len(result) == 21 (20 переменных + t)

---

## ExtendedSDETrajectory

**Назначение:** Полная траектория симуляции.

**Поля:**
- times: np.ndarray — временные точки
- states: list[ExtendedSDEState] — состояния
- params: ParameterSet — параметры

### get_variable

**Сигнатура:**
```python
def get_variable(self, name: str) -> np.ndarray
```

**Поведение:**
1. Проверить name ∈ VARIABLE_NAMES
2. Извлечь значения переменной из всех states
3. Вернуть np.ndarray

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| get_variable("P") | np.array P значений |
| get_variable("unknown") | KeyError |
| Пустая траектория | np.array([]) |

**Инварианты:**
- len(result) == len(self.states)

### get_statistics

**Сигнатура:**
```python
def get_statistics(self) -> dict[str, dict[str, float]]
```

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Непустая траектория | dict с 20 ключами |
| stats["P"]["mean"] | Среднее значение P |
| stats["P"]["final"] | Последнее значение P |

**Инварианты:**
- Каждый ключ содержит: mean, std, min, max, final
- stats[var]["min"] <= stats[var]["mean"] <= stats[var]["max"]

---

## ExtendedSDEModel

**Назначение:** Основной класс расширенной SDE модели.

### __init__

**Сигнатура:**
```python
def __init__(
    self,
    params: ParameterSet | None = None,
    therapy: TherapyProtocol | None = None,
    rng_seed: int | None = None,
) -> None
```

**Поведение:**
1. params → self.params (или ParameterSet() если None)
2. therapy → self.therapy
3. Инициализация numpy RNG с seed
4. Валидация параметров

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| ExtendedSDEModel() | params == defaults |
| ExtendedSDEModel(params=custom) | Кастомные параметры |
| ExtendedSDEModel(rng_seed=42) | Воспроизводимость |

### simulate

**Сигнатура:**
```python
def simulate(
    self,
    initial_state: ExtendedSDEState,
    t_span: tuple[float, float] | None = None,
) -> ExtendedSDETrajectory
```

**Поведение:**
1. Инициализация X₀ = initial_state.to_array()
2. Цикл по шагам: X_{n+1} = X_n + drift·dt + diffusion·√dt·ξ
3. _apply_boundary_conditions на каждом шаге
4. Сохранение в ExtendedSDETrajectory

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| simulate(default_state) | Траектория без NaN |
| simulate(state, t_span=(0, 10)) | times от 0 до ~10 |
| Все переменные | >= 0 на всем интервале |

**Инварианты:**
- len(trajectory.states) == len(trajectory.times)
- Все переменные >= 0 (граничные условия)

### _compute_drift

**Сигнатура:**
```python
def _compute_drift(self, state: ExtendedSDEState) -> np.ndarray
```

**Поведение:** Собрать 20-мерный вектор из _drift_* компонентов.

**Инварианты:**
- result.shape == (20,)
- result[i] == _drift_*() для соответствующей переменной

### _compute_diffusion

**Сигнатура:**
```python
def _compute_diffusion(self, state: ExtendedSDEState) -> np.ndarray
```

**Поведение:** σ_i = sigma_i * X_i (диагональный шум).

**Инварианты:**
- result.shape == (20,)
- Если X_i == 0 → σ_i == 0

---

## Drift компоненты — Клеточные (§2.1)

### _drift_platelets

**Формула:** dP/dt = P_max·exp(-t/τ_P) - δ_P·P - k_deg·P

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| t=0, P=0 | ≈ P_max (источник доминирует) |
| t >> τ_P, P > 0 | < 0 (затухание) |
| P=0, t >> τ_P | ≈ 0 |

### _drift_neutrophils

**Формула:** dNe/dt = R_Ne_max·C_IL8ⁿ/(K_IL8ⁿ+C_IL8ⁿ) - δ_Ne·Ne - k_phag·M_total·Ne/(Ne+K_phag)

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Высокий IL-8, Ne=0 | > 0 (рекрутирование) |
| IL-8=0, Ne > 0 | < 0 (апоптоз + фагоцитоз) |
| M_total = 0 | Нет фагоцитоза |

### _drift_M1

**Формула:** dM1/dt = R_M·φ₁ - k_switch·ψ·M1 + k_reverse·ζ·M2 - δ_M·M1

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Высокий TNF, MCP-1 | > 0 (рекрутирование в M1) |
| Высокий IL-10, TGF-β | < 0 (переключение в M2) |
| M1=0, M2=0, MCP-1=0 | ≈ 0 |

### _drift_M2

**Формула:** dM2/dt = R_M·φ₂ + k_switch·ψ·M1 - k_reverse·ζ·M2 - δ_M·M2

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Высокий IL-10, M1 > 0 | > 0 (переключение M1→M2) |
| Высокий TNF, M2 > 0 | < 0 (обратное переключение) |

**Инвариант:** Потоки M1→M2 в _drift_M1 и _drift_M2 зеркальны.

### _drift_fibroblasts

**Формула:** dF/dt = r_F·F·(1-(F+Mf)/K_F)·H + k_diff_S·S·g_diff - k_act·F·A - δ_F·F

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| F=0, S > 0, TGF-β > 0 | > 0 (дифференциация S) |
| F+Mf == K_F | Логистический рост = 0 |
| Высокий TGF-β, F > 0 | k_act·F·A отнимает (в Mf) |

### _drift_myofibroblasts

**Формула:** dMf/dt = k_act·F·A(TGFβ) - δ_Mf·Mf·(1 - TGFβ/(K_surv+TGFβ))

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Высокий TGF-β: F > 0, Mf > 0 | Mf растёт (малый апоптоз) |
| TGF-β → 0, Mf > 0 | Mf убывает (полный апоптоз) |
| F=0 | Нет притока, только убыль |

**Критическое свойство:** TGF-β ↔ Mf бистабильность.

### _drift_endothelial

**Формула:** dE/dt = r_E·E·(1-E/K_E)·V(VEGF)·(1-θ(O₂)) - δ_E·E

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Высокий VEGF, низкий O₂ | > 0 (ангиогенез) |
| O₂ = O₂_blood (нормоксия) | Малый рост (1-θ ≈ 0) |
| VEGF = 0 | Нет роста (V ≈ 0) |

### _drift_stem_cells

**Формула:** dS/dt = r_S·S·(1-S/K_S)·(1+α_PRP·Θ) - k_diff·S·g_diff - δ_S·S

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| PRP активен, S > 0 | Усиленный рост |
| TGF-β высокий | Потеря через дифференциацию |
| S=0 | ≈ 0 |

---

## Drift компоненты — Цитокины (§2.2)

### _drift_C_TNF

**Формула:** s_M1·M1 + s_Ne·Ne - γ·C_TNF - k_inhib·C_IL10·C_TNF/(K_inhib+C_TNF)

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| M1 > 0, IL-10 = 0 | > 0 (продукция) |
| M1 = 0, C_TNF > 0 | < 0 (деградация) |
| Высокий IL-10 | Ингибирование продукции |

### _drift_C_IL10

**Формула:** s_M2·M2 + s_efferocytosis·phagocytosis_rate - γ·C_IL10

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| M2 > 0 | > 0 (секреция) |
| Эффероцитоз активен | Дополнительный IL-10 |
| Всё = 0 кроме C_IL10 | < 0 (деградация) |

### _drift_C_PDGF

**Формула:** s_P·k_deg·P + s_M·(M1+M2) + Θ_PRP - γ·C - k_bind_F·F·C/(K+C)

### _drift_C_VEGF

**Формула:** s_M2·M2·(1+α_hypoxia·(1-θ)) + s_F·F + Θ_PRP - γ·C - k_bind_E·E·C/(K+C)

### _drift_C_TGFb

**Формула:** s_P·k_deg·P + s_M2·M2 + s_Mf·Mf + Θ_PRP - γ·C_TGFβ

**Критическое свойство:** s_Mf·Mf — положительная обратная связь!

### _drift_C_MCP1

**Формула:** s_damage·D(t) + s_M1·M1 - γ·C_MCP1

### _drift_C_IL8

**Формула:** s_damage·D(t) + s_M1·M1 + s_Ne·Ne - γ·C_IL8

---

## Drift компоненты — ECM (§2.3)

### _drift_collagen

**Формула:** (q_F·F + q_Mf·Mf)·(1-ρ_c/ρ_max) - k_MMP·C_MMP·ρ_c/(K_sub+ρ_c)

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| F > 0, Mf > 0, ρ < ρ_max | > 0 (продукция) |
| ρ_c = ρ_max | Продукция = 0 (насыщение) |
| Высокий MMP, ρ > 0 | < 0 (деградация) |

### _drift_MMP

**Формула:** s_M1·M1 + s_M2·α·M2 + s_F·F - k_TIMP·C_TIMP·C_MMP - γ·C_MMP

### _drift_fibrin

**Формула:** -k_fibrinolysis·C_MMP·ρ_f - k_remodel·F·ρ_f

**Инвариант:** Drift всегда <= 0 (только убыль фибрина).

---

## Drift компоненты — Вспомогательные (§2.4)

### _drift_damage

**Формула:** -D/τ_damage

**Инвариант:** Drift < 0 при D > 0 (монотонное затухание).

### _drift_oxygen

**Формула:** D_O2·(O₂_blood - O₂)/L² - k_consumption·cells·O₂/(K+O₂) + k_angio·E

---

## Вспомогательные функции

### _hill

**Сигнатура:** `def _hill(self, x: float, K: float, n: int = 2) -> float`
**Формула:** xⁿ / (Kⁿ + xⁿ)

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| x=K | 0.5 |
| x=0 | 0.0 |
| x >> K | → 1.0 |
| x < 0 | Зависит от n (допустимо) |

**Инварианты:** 0 <= result <= 1 (при x >= 0)

### _polarization_M1

**Формула:** φ₁ = C_TNF / (C_TNF + C_IL10 + ε)

**Инварианты:** 0 <= φ₁ <= 1, φ₁ + φ₂ ≈ 1

### _polarization_M2

**Формула:** φ₂ = 1 - φ₁

### _switching_function

**Формула:** ψ = Hill(C_IL10 + C_TGFβ, K_switch_half, n)

### _reverse_switching

**Формула:** ζ = Hill(C_TNF, K_reverse_half, n=2)

### _mitogenic_stimulation

**Формула:** H = PDGF/(K_PDGF+PDGF) · (1 + α_TGF·TGFβ/(K_prolif+TGFβ))

**Инвариант:** H >= 0

### _differentiation_probability

**Формула:** g = TGFβ / (K_diff + TGFβ)

**Инвариант:** 0 <= g <= 1

### _activation_function

**Формула:** A = Hill(TGFβ, K_activ, n=2)

**Инвариант:** 0 <= A <= 1

### _vegf_activation

**Формула:** V = Hill(VEGF, K_VEGF, n=2)

### _hypoxia_factor

**Формула:** θ = O₂/(K_O2 + O₂)

**Инвариант:** 0 <= θ <= 1

---

## _apply_boundary_conditions

**Сигнатура:**
```python
def _apply_boundary_conditions(
    self, state: ExtendedSDEState,
) -> ExtendedSDEState
```

**Поведение:** max(0, X_i) для всех 20 переменных.

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Все положительные | Без изменений |
| P = -5 | P = 0 |
| Все отрицательные | Все = 0 |

**Инвариант:** Все переменные в результате >= 0.

---

## validate_params / get_default_initial_state

### validate_params

Делегирует к params.validate() + доп. проверки для SDE.

### get_default_initial_state

Начальные условия раны: P=P_max, D=D0, O2=O2_blood,
rho_fibrin=1.0, остальные ≈ 0.

---

## Ключевые биологические свойства для тестирования

| Свойство | Тест | Описание |
|----------|------|----------|
| Позитивность | test_all_populations_nonneg | Все >= 0 |
| M1→M2 переход | test_macrophage_switch | M1 пик 24-48ч, M2 после 72ч |
| TGF-β бистабильность | test_tgfb_bistability | Два устойчивых состояния |
| Гипоксия→ангиогенез | test_hypoxia_angiogenesis | Низкий O₂ → рост E |
| Фибрин→коллаген | test_fibrin_to_collagen | ρ_fibrin ↓, ρ_collagen ↑ |
| Нейтрофильный пик | test_neutrophil_peak | Пик 12-24ч, затухание 48ч |
