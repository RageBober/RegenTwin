# therapy_models.py — Механистические модели терапевтических вмешательств

## Назначение

Механистические модели PRP и PEMF терапий, заменяющие
феноменологические приближения (экспоненциальный спад, сигмоид)
на модели, основанные на биофизических механизмах.

Биологическое обоснование:
- Marx, J Oral Maxillofac Surg 2004 (PRP концентрации)
- Giusti et al., Exp Hematol 2009 (PRP кинетика)
- Eppley et al., Plast Reconstr Surg 2006 (факторы роста в PRP)
- Pilla, Ann Biomed Eng 2013 (PEMF механизмы)
- Varani et al., Mediators Inflamm 2017 (PEMF + аденозин)
- Onstenk et al., J Orthop Res 2015 (синергия PRP+PEMF)

Математический фреймворк: §3.1 (PRP), §3.2 (PEMF), §8.4 (параметры)

Подробное описание: Description/Phase2/description_therapy_models.md

---

## PRPConfig

**Назначение:** Параметры двухфазной кинетики PRP-терапии.

**Поля:**

| Поле | Тип | Default | Единицы | Описание | Источник |
|------|-----|---------|---------|----------|----------|
| dose | float | 4.0 | fold | Концентрация тромбоцитов (3-5x) | Marx 2004 |
| pdgf_c0 | float | 20.0 | нг/мл | Начальная концентрация PDGF-AB | Marx 2004 |
| vegf_c0 | float | 1.0 | нг/мл | Начальная концентрация VEGF | Everts 2006 |
| tgfb_c0 | float | 30.0 | нг/мл | Начальная концентрация TGF-β1 | Eppley 2006 |
| egf_c0 | float | 0.2 | нг/мл | Начальная концентрация EGF | Anitua 2004 |
| tau_burst_pdgf | float | 1.0 | ч | τ_burst для PDGF | Giusti 2009 |
| tau_burst_vegf | float | 1.0 | ч | τ_burst для VEGF | Giusti 2009 |
| tau_burst_tgfb | float | 2.0 | ч | τ_burst для TGF-β | Giusti 2009 |
| tau_burst_egf | float | 0.5 | ч | τ_burst для EGF | Giusti 2009 |
| tau_sustained_pdgf | float | 48.0 | ч | τ_sustained для PDGF | Giusti 2009 |
| tau_sustained_vegf | float | 24.0 | ч | τ_sustained для VEGF | Giusti 2009 |
| tau_sustained_tgfb | float | 72.0 | ч | τ_sustained для TGF-β | Giusti 2009 |
| tau_sustained_egf | float | 12.0 | ч | τ_sustained для EGF | Giusti 2009 |
| alpha_PRP_S | float | 0.5 | — | Коэффициент рекрутирования стволовых | §8 |

**Инварианты:**
- dose > 0
- Все c0 >= 0
- tau_burst < tau_sustained для каждого фактора
- alpha_PRP_S >= 0

---

## PEMFConfig

**Назначение:** Параметры 3 биофизических механизмов PEMF.

**Поля:**

| Поле | Тип | Default | Единицы | Описание |
|------|-----|---------|---------|----------|
| B_amplitude | float | 1.0 | мТ | Амплитуда магнитного поля |
| frequency | float | 50.0 | Гц | Частота PEMF |
| B0_threshold | float | 0.5 | мТ | Пороговая амплитуда (Hill) |
| n_B | float | 2.0 | — | Коэффициент Hill для B-поля |
| f_opt_anti_inflam | float | 27.12 | Гц | Оптимальная частота аденозинового пути |
| sigma_f_anti_inflam | float | 10.0 | Гц | Ширина частотного окна (Гаусс) |
| epsilon_max_anti_inflam | float | 0.4 | — | Макс. снижение TNF-α |
| f_center_prolif | float | 75.0 | Гц | Центр частотного окна Ca²⁺ |
| sigma_window_prolif | float | 25.0 | Гц | Ширина окна Ca²⁺ (Гаусс) |
| epsilon_prolif_max | float | 0.3 | — | Макс. усиление пролиферации |
| B_half_prolif | float | 0.5 | мТ | Полунасыщение B² Hill |
| epsilon_migration_max | float | 0.25 | — | Макс. усиление миграции |

**Инварианты:**
- B_amplitude >= 0
- frequency > 0
- Все epsilon ∈ [0, 1]
- B0_threshold > 0, B_half_prolif > 0
- n_B > 0

---

## SynergyConfig

**Назначение:** Параметры супер-аддитивного эффекта PRP+PEMF.

**Поля:**

| Поле | Тип | Default | Описание |
|------|-----|---------|----------|
| beta_synergy | float | 0.2 | Коэффициент синергии |

**Инварианты:**
- beta_synergy >= 0

---

## PRPReleaseState

**Назначение:** Результат вычисления PRP-релиза на момент t.

**Поля:**

| Поле | Тип | Default | Описание |
|------|-----|---------|----------|
| theta_pdgf | float | 0.0 | Θ_PRP_PDGF(t) — концентрация PDGF (нг/мл) |
| theta_vegf | float | 0.0 | Θ_PRP_VEGF(t) — концентрация VEGF (нг/мл) |
| theta_tgfb | float | 0.0 | Θ_PRP_TGF(t) — концентрация TGF-β (нг/мл) |
| theta_egf | float | 0.0 | Θ_PRP_EGF(t) — концентрация EGF (нг/мл) |
| theta_total | float | 0.0 | Суммарный нормализованный показатель |

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| PRPReleaseState() | Все поля = 0.0 |
| theta_pdgf=15.0, theta_total=0.5 | Валидный объект |

**Инварианты:**
- Все theta >= 0

---

## PEMFEffects

**Назначение:** Активные эффекты PEMF на момент t.

**Поля:**

| Поле | Тип | Default | Описание |
|------|-----|---------|----------|
| anti_inflammatory | float | 0.0 | ε для снижения s_TNF_M1 |
| proliferation | float | 0.0 | ε для усиления r_F, r_E |
| migration | float | 0.0 | ε для усиления D_cell (ABM) |

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| PEMFEffects() | Все поля = 0.0 |
| anti_inflammatory=0.35 | Валидный объект |

**Инварианты:**
- Все ε ∈ [0, 1]

---

## PRPModel

**Назначение:** Механистическая модель двухфазной кинетики PRP.

### __init__

**Сигнатура:**

```python
def __init__(self, config: PRPConfig | None = None) -> None
```

**Поведение:**
1. Если config is None → создать PRPConfig() с defaults
2. Сохранить self.config

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| PRPModel() | config == PRPConfig() |
| PRPModel(PRPConfig(dose=5.0)) | config.dose == 5.0 |

---

### _biphasic_release

**Сигнатура:**

```python
def _biphasic_release(
    self, t: float, c0: float, tau_burst: float, tau_sustained: float,
) -> float
```

**Поведение:**
1. Если t < 0 → вернуть 0.0
2. Вычислить burst = exp(-t / tau_burst)
3. Вычислить sustained = exp(-t / tau_sustained)
4. Вычислить denominator = tau_burst - tau_sustained
5. Если |denominator| < 1e-10 → использовать предельную формулу (L'Hôpital)
6. Вернуть dose · c0 · (burst - sustained) / denominator
7. Применить max(0.0, result) для защиты от отрицательных значений

**Математическая формула:**

```
Θ_PRP_i(t) = dose · c0_i · (e^(-t/τ_burst) - e^(-t/τ_sustained)) / (τ_burst - τ_sustained)
```

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| t=0, c0=20, tau_b=1, tau_s=48 | ~dose·c0·(1-1)/(1-48) = 0 |
| t=1, PDGF параметры | Положительное значение, близко к пику |
| t=48, PDGF параметры | Значение меньше пикового |
| t=200, любые параметры | Близко к 0 (экспоненциальное затухание) |
| t=-5 | 0.0 (до инъекции) |
| tau_burst == tau_sustained | Предельная формула, без деления на 0 |

**Edge cases:**
- t < 0 → return 0.0
- tau_burst ≈ tau_sustained → L'Hôpital или добавить epsilon
- c0 = 0 → return 0.0
- t → ∞ → return → 0

**Инварианты:**
- result >= 0 для всех t
- Результат монотонно убывает после пика
- Пик наступает при t_peak = τ_b·τ_s·ln(τ_s/τ_b) / (τ_s - τ_b)

---

### compute_release

**Сигнатура:**

```python
def compute_release(
    self, t: float, application_time: float = 0.0,
) -> PRPReleaseState
```

**Поведение:**
1. Вычислить t_rel = t - application_time
2. Вызвать _biphasic_release для каждого из 4 факторов:
   - PDGF: (t_rel, pdgf_c0, tau_burst_pdgf, tau_sustained_pdgf)
   - VEGF: (t_rel, vegf_c0, tau_burst_vegf, tau_sustained_vegf)
   - TGF-β: (t_rel, tgfb_c0, tau_burst_tgfb, tau_sustained_tgfb)
   - EGF: (t_rel, egf_c0, tau_burst_egf, tau_sustained_egf)
3. Вычислить theta_total = нормализованная сумма (сумма / (sum of c0 * dose))
4. Вернуть PRPReleaseState

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| t=0, app_time=0 | Все theta ≈ 0 (начальный момент) |
| t=2, app_time=0 | theta_pdgf > 0, theta_egf ~ пик |
| t=100, app_time=0 | Все theta → малые значения |
| t=5, app_time=10 | t_rel < 0 → все theta = 0 |

**Инварианты:**
- Все theta >= 0
- theta_total ∈ [0, 1]
- При t < application_time → все theta = 0

---

### compute_stem_cell_factor

**Сигнатура:**

```python
def compute_stem_cell_factor(
    self, t: float, application_time: float = 0.0,
) -> float
```

**Поведение:**
1. Вызвать compute_release(t, application_time) → получить theta_total
2. Вернуть alpha_PRP_S · theta_total

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| t=0 | 0.0 |
| t=2 (пик PRP) | alpha_PRP_S · theta_total > 0 |
| alpha_PRP_S=0 | 0.0 всегда |

**Инварианты:**
- result >= 0
- result <= alpha_PRP_S

---

### PRPModel.is_active

**Сигнатура:**

```python
def is_active(self, t: float, application_time: float = 0.0) -> bool
```

**Поведение:**
1. Вызвать compute_release(t, application_time) → получить theta_total
2. Вернуть theta_total > 0.01

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| t=0, app_time=0 | False (ещё не начался релиз) |
| t=2, app_time=0 | True (активный релиз) |
| t=500, app_time=0 | False (полностью затух) |
| t=5, app_time=10 | False (до инъекции) |

---

## PEMFModel

**Назначение:** Механистическая модель PEMF с 3 биофизическими путями.

### __init__

**Сигнатура:**

```python
def __init__(self, config: PEMFConfig | None = None) -> None
```

**Поведение:**
1. Если config is None → создать PEMFConfig() с defaults
2. Сохранить self.config

---

### compute_anti_inflammatory

**Сигнатура:**

```python
def compute_anti_inflammatory(self, t: float) -> float
```

**Поведение:**
1. Вычислить Hill-функцию для B: hill = (B/B₀)^n / (1 + (B/B₀)^n)
2. Вычислить Гауссово окно для f: gauss = exp(-(f - f_opt)² / (2·σ²))
3. Вернуть ε_max · hill · gauss

**Математическая формула:**

```
ε_PEMF_anti_inflam(f, B) = ε_max · (B/B₀)^n_B / (1 + (B/B₀)^n_B)
                           · exp(-(f - f_opt)² / (2·σ_f²))
```

**Применение к SDE:**

```
s_TNF_M1 → s_TNF_M1 · (1 - ε_PEMF_anti_inflam)
```

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| B=1.0, f=27.12 (оптимум) | ~ε_max · Hill(2) ≈ 0.32 |
| B=0 | 0.0 (нет поля) |
| B=10.0, f=27.12 | ~ε_max (насыщение Hill) |
| f=100 (далеко от f_opt) | ~0 (Гаусс → 0) |
| B=0.5, f=50 | Промежуточное значение |

**Edge cases:**
- B = 0 → Hill = 0 → result = 0
- B₀ = 0 → деление на 0 → защита
- σ = 0 → Гаусс вырождается → защита

**Инварианты:**
- result ∈ [0, ε_max]
- Монотонно растёт с B (при фиксированном f)
- Максимум при f = f_opt

---

### compute_proliferation_boost

**Сигнатура:**

```python
def compute_proliferation_boost(self, t: float) -> float
```

**Поведение:**
1. Вычислить Hill-функцию для B²: hill = B² / (B_half² + B²)
2. Вычислить частотное окно: W(f) = exp(-(f - f_center)² / (2·σ²))
3. Вернуть ε_prolif_max · hill · W(f)

**Математическая формула:**

```
ε_PEMF_prolif(f, B) = ε_prolif_max · B² / (B_half² + B²)
                      · exp(-(f - f_center)² / (2·σ_window²))
```

**Применение к SDE:**

```
r_F → r_F · (1 + ε_PEMF_prolif)
r_E → r_E · (1 + ε_PEMF_prolif)
```

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| B=1.0, f=75 (оптимум) | ~ε_max · 0.8 ≈ 0.24 |
| B=0 | 0.0 |
| B=5.0, f=75 | ~ε_max (насыщение) |
| f=200 (далеко) | ~0 |

**Инварианты:**
- result ∈ [0, ε_prolif_max]
- Монотонно растёт с B
- Максимум при f = f_center

---

### compute_migration_boost

**Сигнатура:**

```python
def compute_migration_boost(self, t: float) -> float
```

**Поведение:**
1. Вычислить Hill-функцию для B: hill = (B/B₀)^n / (1 + (B/B₀)^n)
2. Вернуть ε_migration_max · hill

**Математическая формула:**

```
ε_PEMF_migration(B) = ε_migration_max · (B/B₀)^n_B / (1 + (B/B₀)^n_B)
```

**Применение к ABM:**

```
D_cell → D_cell · (1 + ε_PEMF_migration)
```

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| B=1.0 | ~ε_max · Hill(2) ≈ 0.2 |
| B=0 | 0.0 |
| B=10.0 | ~ε_max (насыщение) |

**Инварианты:**
- result ∈ [0, ε_migration_max]
- Не зависит от частоты

---

### compute_effects

**Сигнатура:**

```python
def compute_effects(self, t: float) -> PEMFEffects
```

**Поведение:**
1. Вызвать compute_anti_inflammatory(t) → anti_inflam
2. Вызвать compute_proliferation_boost(t) → prolif
3. Вызвать compute_migration_boost(t) → migration
4. Вернуть PEMFEffects(anti_inflam, prolif, migration)

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| B=1.0, f=50 | Все 3 эффекта > 0 |
| B=0 | Все 3 эффекта = 0 |

**Инварианты:**
- Консистентность с индивидуальными compute_ методами

---

### PEMFModel.is_active

**Сигнатура:**

```python
def is_active(self, t: float) -> bool
```

**Поведение:**
1. Проверить B_amplitude > 0
2. Вернуть True/False

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| B_amplitude=1.0 | True |
| B_amplitude=0.0 | False |

---

## SynergyModel

**Назначение:** Супер-аддитивный эффект при одновременном PRP+PEMF.

### __init__

**Сигнатура:**

```python
def __init__(
    self,
    prp_model: PRPModel,
    pemf_model: PEMFModel,
    config: SynergyConfig | None = None,
) -> None
```

**Поведение:**
1. Сохранить ссылки на prp_model и pemf_model
2. Если config is None → SynergyConfig() с defaults
3. Сохранить self.config

---

### compute_synergy_factor

**Сигнатура:**

```python
def compute_synergy_factor(self, t: float) -> float
```

**Поведение:**
1. Проверить PRP активен: prp_active = prp_model.is_active(t)
2. Проверить PEMF активна: pemf_active = pemf_model.is_active(t)
3. Если обе активны: synergy = 1 + β_synergy · theta_total · 1.0
4. Иначе: synergy = 1.0

**Математическая формула:**

```
synergy(t) = 1 + β_synergy · Θ_PRP(t) · PEMF_active(t)
```

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| PRP активен + PEMF активна | > 1.0 |
| Только PRP | 1.0 |
| Только PEMF | 1.0 |
| Ни одна не активна | 1.0 |
| beta_synergy=0 | Всегда 1.0 |

**Инварианты:**
- synergy >= 1.0
- synergy = 1.0 если хотя бы одна терапия не активна

---

### apply_to_drift

**Сигнатура:**

```python
def apply_to_drift(self, drift_modifier: float, t: float) -> float
```

**Поведение:**
1. Вычислить synergy = compute_synergy_factor(t)
2. Вернуть drift_modifier · synergy

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| modifier=0.5, synergy=1.2 | 0.6 |
| modifier=0.5, synergy=1.0 | 0.5 (без изменений) |
| modifier=0.0 | 0.0 |

**Инварианты:**
- |result| >= |drift_modifier| (синергия только усиливает)

---

## Интеграция с SDE системой

### Точки интеграции в extended_sde.py

| Drift-терм | Модификатор | Формула |
|------------|-------------|---------|
| dC_PDGF | PRPModel | + Θ_PRP_PDGF(t) |
| dC_VEGF | PRPModel | + Θ_PRP_VEGF(t) |
| dC_TGFβ | PRPModel | + Θ_PRP_TGF(t) |
| dS | PRPModel | + α_PRP_S · θ_total |
| s_TNF_M1 | PEMFModel | · (1 - ε_anti_inflam) |
| r_F | PEMFModel | · (1 + ε_prolif) |
| r_E | PEMFModel | · (1 + ε_prolif) |
| D_cell (ABM) | PEMFModel | · (1 + ε_migration) |
| Все модификаторы | SynergyModel | · synergy(t) |

### Биологические свойства для валидации

| Свойство | Критерий | Источник |
|----------|----------|----------|
| PDGF пик | ~1-2 ч после инъекции | Marx 2004 |
| VEGF sustained | Значимый уровень до 24 ч | Everts 2006 |
| TGF-β длительный | Самый долгий релиз (72 ч) | Eppley 2006 |
| EGF быстрый | Пик < 1 ч, затухание к 12 ч | Anitua 2004 |
| PEMF anti-inflam | TNF снижение 30-50% | Varani 2017 |
| PEMF prolif | Частотное окно 50-100 Гц | Pilla 2013 |
| Синергия | 1.0 < synergy < 1.5 | Onstenk 2015 |
