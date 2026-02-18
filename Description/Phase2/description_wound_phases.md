# wound_phases.py — Детекция фаз заживления раны

## Назначение

Определение текущей фазы заживления (из 4) по состоянию
20-переменной SDE системы. Используется для анализа траекторий,
валидации биологической корректности и визуализации.

Биологическое обоснование:
- Gurtner et al., Nature 2008
- Eming et al., Science Translational Medicine 2014

Подробное описание: Description/Phase2/description_wound_phases.md

---

## WoundPhase (Enum)

**Назначение:** Перечисление 4 фаз заживления раны.

| Значение | Название | Временные рамки | Ключевые процессы |
|----------|----------|----------------|-------------------|
| HEMOSTASIS | Гемостаз | 0-6 ч | Тромбоциты, фибрин, PDGF/TGF-β |
| INFLAMMATION | Воспаление | 6 ч - 4-6 дней | Нейтрофилы, M1, TNF-α, IL-8 |
| PROLIFERATION | Пролиферация | 4-21 дней | Фибробласты, M2, VEGF, коллаген |
| REMODELING | Ремоделирование | 21 д - 1 год | MMP/TIMP, апоптоз Mf, коллаген I |

**Инварианты:**
- Ровно 4 значения
- Фазы идут в биологическом порядке

---

## PhaseIndicators

**Назначение:** Результат детекции фазы для одного момента времени.

**Поля:**

| Поле | Тип | Default | Описание |
|------|-----|---------|----------|
| phase | WoundPhase | HEMOSTASIS | Определённая фаза |
| confidence | float | 0.0 | Уверенность (0-1) |
| dominant_cells | list[str] | [] | Доминирующие клетки |
| dominant_cytokines | list[str] | [] | Доминирующие цитокины |
| phase_progress | float | 0.0 | Прогресс внутри фазы (0-1) |

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| PhaseIndicators() | phase=HEMOSTASIS, confidence=0.0 |
| confidence=0.85 | Валидный объект |
| dominant_cells=["P", "Ne"] | Валидный объект |

**Инварианты:**
- 0 <= confidence <= 1
- 0 <= phase_progress <= 1
- phase ∈ WoundPhase

---

## WoundPhaseDetector

**Назначение:** Определение фазы заживления по состоянию SDE системы.

### __init__

**Сигнатура:**

```python
def __init__(self, params: ParameterSet | None = None) -> None
```

**Поведение:**
1. Сохранить params (или ParameterSet() если None)
2. Инициализировать пороги детекции из параметров

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| WoundPhaseDetector() | params == defaults |
| WoundPhaseDetector(custom_params) | Кастомные пороги |

---

### detect_phase

**Сигнатура:**

```python
def detect_phase(self, state: ExtendedSDEState) -> PhaseIndicators
```

**Поведение:**
1. Вычислить confidence для каждой из 4 фаз
2. Выбрать фазу с максимальным confidence
3. Определить доминирующие клетки и цитокины
4. Вернуть PhaseIndicators

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| P=1000, D=1.0, rho_fibrin=1.0 | HEMOSTASIS |
| Ne=500, M1=200, M2=50, C_TNF=5 | INFLAMMATION |
| F=1000, M2=300, M1=50, rho_collagen=0.5 | PROLIFERATION |
| rho_collagen=0.9, C_MMP=0.5, Mf→0 | REMODELING |
| Все нули | HEMOSTASIS (default) |

**Edge cases:**
- Все переменные = 0 → вернуть минимальный confidence
- Два phase с одинаковым confidence → первый по порядку

**Инварианты:**
- Возвращает ровно одну фазу
- confidence ∈ [0, 1]

---

### detect_phase_trajectory

**Сигнатура:**

```python
def detect_phase_trajectory(
    self, trajectory: ExtendedSDETrajectory,
) -> list[PhaseIndicators]
```

**Поведение:**
1. Для каждого state в trajectory.states → detect_phase(state)
2. Вернуть список результатов

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Траектория 100 шагов | list из 100 PhaseIndicators |
| Пустая траектория | [] |

**Инварианты:**
- len(result) == len(trajectory.states)

---

### _is_hemostasis

**Сигнатура:**

```python
def _is_hemostasis(self, state: ExtendedSDEState) -> float
```

**Критерии:**
- Тромбоциты P выше порога (P > P_threshold)
- Фибрин rho_fibrin высокий (> 0.5)
- Damage signal D активен (D > 0.1)
- Мало нейтрофилов и макрофагов

**Поведение:**
1. Нормализовать P, rho_fibrin, D к [0, 1]
2. Взвешенная сумма индикаторов
3. Вернуть confidence ∈ [0, 1]

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| P=10000, D=1.0, rho_fibrin=1.0 | Высокий confidence |
| P=0, D=0, rho_fibrin=0 | Низкий confidence |

---

### _is_inflammation

**Сигнатура:**

```python
def _is_inflammation(self, state: ExtendedSDEState) -> float
```

**Критерии:**
- Нейтрофилы Ne высокие
- M1 > M2 (провоспалительная доминантность)
- TNF-α и IL-8 повышены
- Damage signal снижается (D < D0 но > 0)

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Ne=500, M1=200, M2=50 | Высокий confidence |
| Ne=0, M1=0, TNF=0 | Низкий confidence |
| M1=100, M2=300 | Умеренный (M2 > M1) |

---

### _is_proliferation

**Сигнатура:**

```python
def _is_proliferation(self, state: ExtendedSDEState) -> float
```

**Критерии:**
- M2 > M1 (репаративная доминантность)
- F высокие (фибробластная активность)
- Коллаген растёт (rho_collagen > 0)
- VEGF и PDGF повышены
- Ангиогенез (E растёт)

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| F=1000, M2=300, M1=50, rho_collagen=0.5 | Высокий |
| F=0, M2=0, collagen=0 | Низкий |

---

### _is_remodeling

**Сигнатура:**

```python
def _is_remodeling(self, state: ExtendedSDEState) -> float
```

**Критерии:**
- Высокий стабильный коллаген (rho_collagen > 0.7)
- MMP активен (C_MMP > 0)
- Клеточность снижена (Ne → 0, M1 → 0)
- Миофибробласты убывают (Mf → 0)
- Фибрин практически отсутствует

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| collagen=0.9, MMP=0.5, Ne=0, Mf=5 | Высокий |
| collagen=0.1, MMP=0 | Низкий |

---

### get_phase_boundaries

**Сигнатура:**

```python
def get_phase_boundaries(
    self, trajectory: ExtendedSDETrajectory,
) -> dict[WoundPhase, tuple[float, float]]
```

**Поведение:**
1. Вычислить detect_phase для всей траектории
2. Для каждой фазы найти первый и последний момент доминирования
3. Вернуть {phase: (t_start, t_end)}

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| 720ч нормальное заживление | 4 фазы с корректными рамками |
| Хроническое воспаление | INFLAMMATION до конца |
| Пустая траектория | {} |

**Инварианты:**
- t_start <= t_end для каждой фазы
- Фазы не обязаны покрывать весь интервал (возможны пропуски)

---

## Биологические свойства для тестирования

| Свойство | Тест | Описание |
|----------|------|----------|
| Последовательность | test_phase_order | Нормальное заживление: H→I→P→R |
| Временные рамки | test_phase_timing | H < 6ч, I пик 24-48ч, P 4-21д |
| M1/M2 соотношение | test_m1_m2_ratio | M1>M2 в I, M2>M1 в P |
| Коллаген динамика | test_collagen_phase | collagen ↑ в P, стабилен в R |
