# Claude Code: Инструкция по применению фиксов RegenTwin v2.0 → код

Этот файл — рабочая инструкция для Claude Code. Передай его вместе с `RegenTwin_Mathematical_Framework_v2.md` и `params.yaml` в новую сессию Claude Code, и он сможет систематически применить все 25 фиксов к существующей кодовой базе.

---

## Подготовка сессии Claude Code

Открой проект в директории RegenTwin и запусти Claude Code:

```bash
cd ~/path/to/RegenTwin
claude
```

В первом сообщении сессии передай контекст:

```
Я обновил математический фреймворк до v2.0. В корне проекта лежат три файла:
- RegenTwin_Mathematical_Framework_v2.md — полная спецификация v2.0
- params.yaml — машинно-читаемые параметры (источник правды)
- CLAUDE_CODE_INSTRUCTIONS.md — пошаговый план рефакторинга

Прочитай все три и сделай начальный план: что изменилось в архитектуре,
какие модули нужно тронуть, в каком порядке.
```

После этого работай по шагам ниже. **Не делай всё сразу** — на каждом шаге проверяй тесты перед переходом к следующему.

---

## Порядок применения фиксов (8 этапов)

Этапы упорядочены так, что каждый последующий не ломает предыдущие. Внутри этапа фиксы можно делать параллельно.

### Этап 1: Фундаментальные единицы и состояние (FIX-01, FIX-19)

**Цель**: устранить размерностную несогласованность cells/mkl vs cells/ml.

```
Промпт для Claude Code:

Прочитай раздел "ЕДИНИЦЫ" в фреймворке. ВСЕ клеточные плотности теперь в cells/ml (было cells/mkl). Найди все места, где:
1. Объявлена клеточная переменная (state vector, dataclass, тесты)
2. Параметры s_X используются для продукции цитокинов
3. Initial conditions

Сделай:
- Добавь явный type alias или комментарий: `# units: cells/ml` в state классе
- Создай unit_check тест в tests/test_units.py: при типичных M1 = 1e6 cells/ml продукция TNF должна давать стационар в диапазоне 0.1-20 ng/ml
- Загрузи params.yaml и убедись, что значения s_X читаются правильно

Покажи мне diff state-классов перед тем как тронуть параметры.
```

### Этап 2: Расширение state vector до 22 переменных

**Цель**: добавить новые переменные C_SDF1, C_TIMP, переименовать.

```
Промпт для Claude Code:

В фреймворке v2.0 раздел 7 указан расширенный state vector:
- 8 клеточных популяций: P, Ne, M1, M2, F, Mf, E, S
- 8 цитокинов: TNF, IL10, PDGF, VEGF, TGFβ, MCP1, IL8, **SDF1 (NEW)**
- 4 ECM: collagen, MMP, **TIMP (NEW)**, fibrin
- 2 вспомогательных: D, O2

Текущий state vector — какой у нас? Покажи. Потом добавь недостающее:
- C_SDF1 как новая переменная (FIX-06)
- C_TIMP как новая переменная (FIX-09)
- Соответствующие IC из params.yaml

Не трогай уравнения пока — только структуру state.
```

### Этап 3: Критические математические фиксы уравнений

Эти три фикса — disqualifying ошибки, которые останавливают модель от запуска или дают нефизические результаты.

**FIX-03: Mf carrying capacity** (CRITICAL)

```
Промпт:

В уравнении миофибробластов dMf/dt (раздел 2.1.6 фреймворка) был критический баг:
при высоком TGF-β миофибробласты росли неограниченно (численно подтверждено).

Применить фикс:
- Добавить `max(0, 1 - (F + Mf)/K_F)` к источнику (общая carrying capacity с F)
- Добавить δ_floor (= 0.1) как минимальную относительную смертность даже при высоком TGF-β

Старая формула:
  dMf = k_act·F·A - δ_Mf·Mf·(1 - C_TGFβ/(K_survival+C_TGFβ))

Новая формула (см. фреймворк):
  dMf = k_act·F·A·max(0, 1-(F+Mf)/K_F)
       - δ_Mf·Mf·(δ_floor + (1-δ_floor)·K_survival/(K_survival+C_TGFβ))

Найди функцию вычисления dMf в коде и обнови. Добавь тест: при высоком TGF (=100 ng/ml)
и F=1e5 cells/ml в стационаре Mf должен оставаться < 1e9 cells/ml (не уезжать в бесконечность).
```

**FIX-09: уравнение TIMP**

```
Промпт:

В уравнении MMP появлялась переменная C_TIMP без уравнения — модель математически
незамкнута. По умолчанию в v2.0 — добавить уравнение:

dC_TIMP = [s_TIMP_F·F·(1+α_TGF_TIMP·C_TGFβ/(K_TIMP+C_TGFβ)) + s_TIMP_M2·M2 - γ_TIMP·C_TIMP]dt

Параметры в params.yaml: ecm.s_TIMP_F, s_TIMP_M2, alpha_TGF_TIMP, K_TIMP, gamma_TIMP

Альтернатива (упрощение для MVP): C_TIMP = const = TIMP_0. Сделай это конфигурируемым
через флаг в params.yaml: ecm.timp_dynamic = true/false. По умолчанию true.
```

**FIX-10: фибринолиз через плазмин**

```
Промпт:

В уравнении фибрина (2.3.3) для деградации использовалась переменная C_MMP — это
биологически неверно. Фибрин разрушается ПЛАЗМИНОМ, не MMPs.

Минимальный фикс — переименование:
  В уравнении dρ_f: C_MMP → C_protease
  C_protease — обобщённая концентрация фибринолитических протеаз

В упрощённой реализации: C_protease ≈ C_MMP_total (просто переименовываем тип переменной для
семантической ясности). Документируй в коде, что это аппроксимация, и для полной модели
требуется отдельная переменная C_plasmin.

Добавь TODO-комментарий в коде об этом.
```

### Этап 4: Структурные фиксы клеточных уравнений

```
Промпт для Claude Code:

Применить FIX-04, FIX-05, FIX-06, FIX-21, FIX-22 одновременно (все — клеточные уравнения):

FIX-04 — dF: контактное ингибирование `(1-(F+Mf)/K_F)` → `max(0, 1-(F+Mf)/K_F)`
FIX-05 — dE: добавить J_sprouting · ξ(C_VEGF) как первый член источника. УБРАТЬ старый множитель (1-θ_hypoxia) — двойной счёт.
FIX-06 — dS: добавить J_homing · η(C_SDF1) · (1+α_PRP_homing·Θ_PRP_homing) как первый член. PRP-эффект ПЕРЕМЕСТИТЬ с пролиферации (r_S) на хоминг.
FIX-21 — dM1, dM2: формула φ₁ = (TNF + φ_baseline)/(TNF + IL10 + φ_baseline + ε). φ_baseline ≈ 0.1.
FIX-22 — dF: добавить J_F_migration · χ(C_PDGF) как первый член источника.

Все параметры есть в params.yaml. Каждое изменение — отдельный коммит для трассировки.
```

### Этап 5: Цитокиновые уравнения (FIX-07, FIX-08, FIX-25)

```
Промпт:

FIX-07 — dTNF: ингибиция IL-10 переехала с члена-стока на множитель источника.
Старая форма: ... - k_inhib·C_IL10·C_TNF/(K+C_TNF)
Новая форма: (s_TNF_M1·M1 + s_TNF_Ne·Ne) · ρ_inhib(C_IL10) - γ_TNF·C_TNF
где ρ_inhib(C_IL10) = 1/(1 + (C_IL10/K_inhib)^n_inhib)

FIX-08 — dTGFβ: добавить член рецепторного потребления:
  -k_bind_TGF·(F+Mf)·C_TGFβ/(K_TGF_bind+C_TGFβ)

FIX-25 — dNe: Hill коэффициент для R_Ne(C_IL8) с n=2 на n=1.

Аналогичную ингибицию (как в TNF) применить и к dIL8 (был старый множитель 1/(1+IL10/K_inhib) —
проверь, что он использует ρ_inhib теперь же).
```

### Этап 6: Терапевтические воздействия (FIX-12, FIX-13, FIX-14)

```
Промпт:

FIX-12 — PRP-кинетика. Добавить нормализацию V_wound в формулу Бэйтмена:
  Θ_PRP_i(t) = (D_PRP/V_wound) · φᵢ · k_burst/(k_burst-k_clear) · [exp(-k_clear·t) - exp(-k_burst·t)]
где k_burst = 1/τ_burst, k_clear = 1/τ_sustained.

Без V_wound размерность была неправильной.

FIX-13 — PEMF: разделить на два частотных окна:
  W_LF(f) — гауссиана 50-100 Гц
  W_RF(f) — лог-гауссиана ~27.12 МГц
Реализовать обе функции отдельно. Параметры в params.yaml: pemf.f_LF, sigma_LF, f_RF, sigma_RF_log.

FIX-14 — synergy: явно применить к ε_LF_prolif:
  ε_LF_prolif → ε_LF_prolif · synergy(t)
  synergy(t) = 1 + β_synergy · max_i(Θ_PRP_i(t))/Θ_PRP_ref · [PEMF active]
```

### Этап 7: Численные методы (FIX-11, FIX-15, FIX-16, FIX-17, FIX-18, FIX-24)

```
Промпт:

FIX-15 — В docstring и комментариях к coupling layer: заменить "Equation-Free Framework"
на "Hybrid SDE-ABM via operator splitting". Это не Equation-Free по Kevrekidis — это стандартный
operator splitting (Strang 1968).

FIX-16 — Multirate subcycling. Сейчас, скорее всего, один dt для всего. Нужно:
- dt_fast = 0.02 ч для цитокинов (γ ≈ 0.5 1/h требует устойчивости)
- dt_slow = 1.0 ч для клеток
- На каждый dt_slow выполнять N=50 субшагов цитокиновой динамики
Реализуй через subcycling в integrator.

FIX-17 — Заменить "Method of Manufactured Solutions" на "strong convergence test on benchmark SDE
(geometric Brownian motion)" в тестах сходимости.

FIX-18 — В docstring сходимости ABM→SDE: заменить "law of large numbers" на
"Kurtz diffusion approximation theorem (Kurtz 1971)". Шум σ должен масштабироваться как 1/√N.

FIX-24 — В docstring SDE solver явно указать: "All SDEs interpreted in Itô sense.
Numerical schemes (Euler-Maruyama, Milstein) directly correspond to Itô interpretation."

FIX-11 — В уравнении O2 убрать аддитивный член k_angio·E (двойной счёт). Реализовать L(E) = L_0/(1+α_E·E/K_E).
Также взвешенная сумма метаболизма с w_Ne, w_M, w_F, w_E, w_S из params.yaml.
```

### Этап 8: Шумы (FIX-20)

```
Промпт:

В params.yaml есть полная таблица noise.sigma_X. Сейчас в коде, скорее всего, σ либо нет,
либо они hardcoded. Сделай:

1. Загрузить все sigma_X из params.yaml.noise
2. Применить к соответствующим переменным в SDE step
3. ECM-уравнения (collagen, MMP, TIMP, fibrin) — БЕЗ шума, флаг noise.ecm_deterministic = true в params.yaml.

Добавь тест: симуляция с σ=0 (детерминистическая) и σ=0.1 (стохастическая) должны давать
разные траектории, но одинаковое среднее за множество запусков (ergodicity check).
```

---

## Тестовый набор для верификации

После применения всех фиксов запустить полный набор тестов:

```python
# tests/test_v2_fixes.py — обязательные тесты после рефакторинга

def test_units_consistency():
    """FIX-01: cells/ml everywhere, физиологические концентрации цитокинов."""
    state = simulate_to_steady_state(IC=default_IC, t_end=72)
    assert 0.1 < state.C_TNF < 20, f"TNF non-physiological: {state.C_TNF}"
    assert 0.1 < state.C_PDGF < 50
    assert 0.05 < state.C_VEGF < 5

def test_Mf_no_divergence():
    """FIX-03: Mf не должен уезжать в бесконечность при высоком TGF-β."""
    state = simulate(IC=default_IC, t_end=720, force_high_TGF=True)
    assert state.Mf < 1e9, f"Mf diverged: {state.Mf}"

def test_TIMP_equation_present():
    """FIX-09: C_TIMP — динамическая переменная, не пропадает."""
    state = simulate(IC=default_IC, t_end=240)
    assert state.C_TIMP > 0
    assert state.C_TIMP != initial_TIMP  # changed dynamically

def test_E_recovery_from_low():
    """FIX-05: E должен восстанавливаться из низких значений через J_sprouting."""
    IC_low_E = default_IC.copy()
    IC_low_E.E = 1.0  # very low
    state = simulate(IC=IC_low_E, t_end=240)
    assert state.E > 100  # recovered

def test_S_recovery_from_low():
    """FIX-06: аналогично для стволовых через J_homing."""
    IC_low_S = default_IC.copy()
    IC_low_S.S = 1.0
    state = simulate(IC=IC_low_S, t_end=240)
    assert state.S > 10  # recovered

def test_phase_dynamics():
    """Качественная проверка 4 фаз заживления."""
    history = simulate(IC=default_IC, t_end=504, save_history=True)

    # Inflammation peak (M1 max) at 24-72h
    M1_peak_time = history.t[np.argmax(history.M1)]
    assert 24 <= M1_peak_time <= 72, f"M1 peak: {M1_peak_time}h"

    # Proliferation: F should grow significantly
    F_at_120h = history.F[history.t == 120][0]
    F_initial = default_IC.F
    assert F_at_120h > 2 * F_initial

    # Mf activation in 4-21d
    Mf_peak_time = history.t[np.argmax(history.Mf)]
    assert 240 <= Mf_peak_time <= 504

def test_strong_convergence_GBM():
    """FIX-17: SDE solver сходится с правильным порядком на GBM benchmark."""
    errors = []
    dts = [0.1, 0.05, 0.025, 0.0125]
    for dt in dts:
        err = run_GBM_benchmark(dt, scheme="EulerMaruyama")
        errors.append(err)

    # Order should be ~0.5 for EM
    rates = [np.log(errors[i]/errors[i+1])/np.log(2) for i in range(len(errors)-1)]
    avg_rate = np.mean(rates)
    assert 0.3 < avg_rate < 0.7, f"EM convergence rate: {avg_rate}"

def test_no_double_hypoxia_counting():
    """FIX-05: гипоксия влияет на E только через VEGF, не напрямую."""
    # Run two simulations at same VEGF but different O2 — E dynamics should match
    state_normoxic = simulate(IC=default_IC, t_end=72, force_O2=80, force_VEGF=2.0)
    state_hypoxic = simulate(IC=default_IC, t_end=72, force_O2=20, force_VEGF=2.0)
    rel_diff = abs(state_normoxic.E - state_hypoxic.E) / state_normoxic.E
    assert rel_diff < 0.05  # E dynamics should be ≈ same when VEGF clamped
```

---

## Контрольный список (checklist) для самопроверки

После применения всех фиксов проверь:

- [ ] `params.yaml` загружается, все параметры доступны
- [ ] State vector содержит C_SDF1 и C_TIMP
- [ ] Все σ_X заданы (носят значимые величины)
- [ ] Тест `test_Mf_no_divergence` проходит
- [ ] Тест `test_units_consistency` проходит — TNF в диапазоне 0.1-20 ng/ml
- [ ] Тест `test_phase_dynamics` проходит — M1 пик в 24-72ч, Mf в 240-504ч
- [ ] Тест `test_strong_convergence_GBM` проходит — EM order ~0.5
- [ ] Все cells/mkl в комментариях/обозначениях заменены на cells/ml
- [ ] Документация (README, docstrings) ссылается на v2.0 фреймворка
- [ ] Equation-Free Framework нигде не упоминается (заменено на operator splitting)
- [ ] Method of Manufactured Solutions заменён на strong convergence test

---

## Известные ограничения v2.0 — что не нужно трогать

Некоторые проблемы оставлены как TODO для пост-защитной работы (см. раздел 10 фреймворка):

1. Бистабильность TGF-β/Mf — не доказана аналитически, не трогать в коде сейчас
2. Латентный vs активный TGF-β — не разделять, использовать единый C_TGFβ
3. Коллаген III/I — единый ρ_collagen
4. Plasmin/PA-PAI — обобщённый C_protease
5. Spatial dynamics — текущий volume-averaged подход

Если Claude Code попытается рефакторить эти места — останови. Они не в скоупе v2.0.

---

## Если что-то пойдёт не так

При проблемах:

1. **Тесты падают после фикса**: откати последний коммит (`git reset --hard HEAD~1`), проверь параметры в params.yaml против фреймворка v2.0
2. **Mf снова уезжает**: убедись, что carrying capacity с `max(0, ...)` применилась корректно, `δ_floor = 0.1` не ноль
3. **TNF нефизиологически высокий**: проверь, что клетки в cells/ml, а не cells/mkl
4. **E или S схлопываются в ноль**: проверь, что J_sprouting и J_homing > 0
5. **C_TIMP undefined error**: проверь, что добавлено уравнение и переменная в state vector

---

## После успешного рефакторинга

1. Тег версии: `git tag v2.0.0 -m "RegenTwin Mathematical Framework v2.0"`
2. Обновить README с указанием на v2.0
3. Запустить полную симуляцию на 21 день и сравнить графики с v1.0
4. Если есть данные Xue 2009 — провести внешний бенчмарк (раздел 6.3 фреймворка)
5. Обновить тезисы доклада на защиту с явным упоминанием исправленных проблем

---

*Этот документ — runbook для Claude Code.*
*Источник правды для математики: RegenTwin_Mathematical_Framework_v2.md*
*Источник правды для параметров: params.yaml*
