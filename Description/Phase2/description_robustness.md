# robustness.py — Верификация робастности численных методов

## Назначение

Модуль обеспечивает high-level верификацию робастности SDE/ABM моделей:
- Контроль позитивности с накоплением статистики нарушений
- Обнаружение NaN/Inf и стратегия восстановления с счётчиком
- Проверка законов сохранения (баланс клеток, цитокинов, ECM)
- Верификация порядка сходимости (Method of Manufactured Solutions)
- Статистическое сравнение SDE vs ABM (Закон больших чисел)

**Отличие от numerical_utils.py:**
- `numerical_utils` — low-level утилиты (clip, detect NaN, adaptive dt)
- `robustness` — high-level верификация и диагностика

Математическое обоснование: Doks/RegenTwin_Mathematical_Framework.md §2

---

## ViolationStats

**Назначение:** Dataclass для накопления статистики нарушений позитивности.

**Поля:**

| Поле | Тип | Default | Описание |
|------|-----|---------|----------|
| count | int | 0 | Общее число нарушений |
| variables | dict[str, int] | {} | {имя: число_нарушений} |
| timestamps | list[float] | [] | Моменты времени нарушений |
| total_clipped | float | 0.0 | Суммарная величина отсечения |

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| ViolationStats() | count == 0, variables == {} |
| После enforce() с отриц. значениями | count > 0, variables заполнен |
| reset_stats() | Всё обнулено |

**Инварианты:**
- count == sum(variables.values())
- len(timestamps) ≤ count
- total_clipped ≥ 0

---

## ConservationReport

**Назначение:** Отчёт о проверке законов сохранения.

**Поля:**

| Поле | Тип | Default | Описание |
|------|-----|---------|----------|
| mass_error | float | 0.0 | Относительная ошибка клеток |
| cytokine_error | float | 0.0 | Относительная ошибка цитокинов |
| ecm_error | float | 0.0 | Относительная ошибка ECM |
| is_conserved | bool | True | Все ошибки < tolerance |
| tolerance | float | 0.05 | Допуск (5%) |
| details | str | "" | Текстовая диагностика |

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Точный баланс (error=0) | is_conserved == True |
| mass_error = 0.1 > tol=0.05 | is_conserved == False |
| Все ошибки < tolerance | is_conserved == True |

**Инварианты:**
- is_conserved == (mass_error < tol and cytokine_error < tol and ecm_error < tol)
- Все ошибки ≥ 0

---

## ConvergenceResult

**Назначение:** Результат верификации порядка сходимости.

**Поля:**

| Поле | Тип | Default | Описание |
|------|-----|---------|----------|
| estimated_order | float | 0.0 | Оценённый порядок p |
| errors | list[float] | [] | Strong errors для каждого dt |
| dt_sequence | list[float] | [] | Последовательность dt |
| reference_order | float | 0.0 | Теоретический порядок |
| is_valid | bool | False | |estimated - reference| < 0.2 |

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| EM solver | estimated_order ≈ 0.5 |
| Milstein solver | estimated_order ≈ 1.0 |
| errors убывают с dt | estimated_order > 0 |

**Инварианты:**
- len(errors) == len(dt_sequence)
- estimated_order > 0 для корректных солверов
- is_valid == (|estimated_order - reference_order| < 0.2)

---

## ComparisonMetrics

**Назначение:** Метрики сравнения SDE и ABM траекторий.

**Поля:**

| Поле | Тип | Default | Описание |
|------|-----|---------|----------|
| wasserstein_distance | float | 0.0 | W1 (Earth Mover's) расстояние |
| mean_diff | float | 0.0 | |mean_SDE - mean_ABM| |
| std_diff | float | 0.0 | |std_SDE - std_ABM| |
| ks_statistic | float | 0.0 | KS статистика |
| ks_pvalue | float | 0.0 | KS p-value |
| is_consistent | bool | False | p > significance_level |

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Одинаковые выборки | W1 ≈ 0, KS p > 0.05 |
| Разные распределения | W1 >> 0, KS p < 0.05 |
| N_agents → ∞ | ABM → SDE, is_consistent == True |

**Инварианты:**
- wasserstein_distance ≥ 0
- 0 ≤ ks_statistic ≤ 1
- 0 ≤ ks_pvalue ≤ 1
- is_consistent == (ks_pvalue > significance_level)

---

## PositivityEnforcer

**Назначение:** Контроль позитивности с накоплением статистики.

### PositivityEnforcer.__init__

**Сигнатура:**
```python
def __init__(self, variable_names=None, min_value=0.0)
```

### PositivityEnforcer.enforce

**Назначение:** Отсечь отрицательные значения + обновить статистику.

**Сигнатура:**
```python
def enforce(self, state: np.ndarray, t: float = 0.0,
            variable_names: list[str] | None = None) -> np.ndarray
```

**Поведение:**
1. result = state.copy()
2. Для каждого i: если result[i] < min_value:
   - total_clipped += abs(result[i] - min_value)
   - result[i] = min_value
   - Обновить variables[name] += 1, timestamps.append(t), count += 1
3. Если есть нарушения → logger.warning
4. Вернуть result

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Все положительные | Без изменений, count=0 |
| state[3] = -5 | state[3] → 0, count=1, total_clipped=5 |
| Несколько отрицательных | count == число отрицательных |
| Повторный вызов | count накапливается |

**Edge cases:**
- state с NaN → NaN остаётся (не клипуется)
- Пустой массив → пустой результат
- min_value < 0 → допустимо (отрицательный порог)

**Инварианты:**
- Возвращает НОВЫЙ ndarray (не мутирует вход)
- result[i] ≥ min_value для всех i (кроме NaN)
- stats.count монотонно возрастает

### get_violation_stats

**Сигнатура:**
```python
def get_violation_stats(self) -> ViolationStats
```

**Поведение:** Вернуть копию текущей статистики.

### reset_stats

**Сигнатура:**
```python
def reset_stats(self) -> None
```

**Поведение:** Обнулить stats (count=0, variables={}, timestamps=[], total_clipped=0).

---

## NaNHandler

**Назначение:** Обнаружение NaN/Inf + восстановление с лимитом.

### NaNHandler.__init__

**Сигнатура:**
```python
def __init__(self, max_recoveries=10, dt_reduction_factor=0.5)
```

### NaNHandler.check

**Назначение:** Проверить наличие NaN/Inf.

**Сигнатура:**
```python
def check(self, state: np.ndarray) -> bool
```

**Поведение:**
1. return not np.all(np.isfinite(state))

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Все finite | False |
| Один NaN | True |
| Один Inf | True |
| Все NaN | True |

### NaNHandler.recover

**Назначение:** Восстановление: откат + уменьшение dt.

**Сигнатура:**
```python
def recover(self, state, last_valid_state, dt) -> tuple[np.ndarray, float, bool]
```

**Поведение:**
1. recovery_count += 1
2. recovered_state = last_valid_state.copy()
3. new_dt = dt * dt_reduction_factor
4. should_stop = (recovery_count >= max_recoveries)
5. logger.warning(f"NaN recovery #{recovery_count}")
6. Вернуть (recovered_state, new_dt, should_stop)

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Первое восстановление | should_stop=False, dt уменьшен |
| max_recoveries-е восстановление | should_stop=True |
| После recover | recovery_count увеличен |

**Edge cases:**
- last_valid_state содержит NaN → проблема (должен быть валидным)
- dt_reduction_factor = 0 → new_dt = 0 (деградация)

**Инварианты:**
- recovery_count монотонно возрастает
- new_dt = dt * factor^(число восстановлений за шаг)

### get_recovery_count / reset

**Сигнатура:**
```python
def get_recovery_count(self) -> int
def reset(self) -> None
```

---

## ConservationChecker

**Назначение:** Проверка балансов рождения/смерти для биологических переменных.

### check_mass_balance

**Назначение:** Баланс клеточных популяций.

**Сигнатура:**
```python
def check_mass_balance(
    self, births, deaths, population_current,
    population_previous, dt
) -> ConservationReport
```

**Поведение:**
1. delta_actual = population_current - population_previous
2. delta_expected = (births - deaths) * dt
3. error = ||delta_actual - delta_expected|| / max(||population_current||, ε)
4. is_conserved = error < tolerance
5. Сохранить в reports

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Точный Euler (без шума) | mass_error ≈ 0 |
| С шумом (EM) | mass_error ~ O(√dt) |
| births=deaths=0, ΔN=0 | mass_error = 0 |
| Большое dt | mass_error может быть > tol |

**Edge cases:**
- Все популяции = 0 → error = 0 (деление на ε)
- Отрицательные births → биологически некорректно, но математически допустимо

**Инварианты:**
- mass_error ≥ 0
- Для детерминист. Euler с малым dt: mass_error → 0

### check_cytokine_balance

**Назначение:** Баланс цитокинов (продукция - деградация).

**Сигнатура:**
```python
def check_cytokine_balance(
    self, production, degradation,
    concentration_current, concentration_previous, dt
) -> ConservationReport
```

**Поведение:** Аналогично check_mass_balance, но для 7 цитокинов.

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Стационарное состояние (prod=deg) | ΔC ≈ 0, error ≈ 0 |
| Только деградация | ΔC < 0, error зависит от метода |

### report / reset

**Сигнатура:**
```python
def report(self) -> list[ConservationReport]
def reset(self) -> None
```

---

## ConvergenceVerifier

**Назначение:** Верификация порядка сходимости через MMS.

### compute_order

**Назначение:** Log-log регрессия для оценки порядка.

**Сигнатура:**
```python
def compute_order(self, errors: list[float], dt_sequence: list[float]) -> float
```

**Поведение:**
1. log_dt = [log(dt) for dt in dt_sequence]
2. log_err = [log(e) for e in errors]
3. p = slope of np.polyfit(log_dt, log_err, deg=1)
4. Вернуть p

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| errors = [1, 0.5, 0.25], dt = [1, 0.5, 0.25] | order ≈ 1.0 |
| errors = [1, 0.707, 0.5], dt = [1, 0.5, 0.25] | order ≈ 0.5 |
| Константные errors | order ≈ 0 |

**Edge cases:**
- errors содержит 0 → log(0) = -inf → обработка
- Один элемент → невозможно оценить → return 0.0
- Немонотонные errors → шумная оценка

**Инварианты:**
- Для корректного солвера: p > 0
- len(errors) == len(dt_sequence)

### verify_solver

**Назначение:** Полная верификация с geometric Brownian motion.

**Сигнатура:**
```python
def verify_solver(self, solver, reference_order,
                  dt_base=0.01, n_refinements=4) -> ConvergenceResult
```

**Поведение:**
1. Создать тестовую задачу: dX = μXdt + σXdW (GBM)
2. dt_sequence = [dt_base / 2^k for k in range(n_refinements)]
3. Для каждого dt:
   - Запустить n_realizations симуляций
   - Вычислить strong error = E[|X_num(T) - X_exact(T)|]
4. estimated_order = compute_order(errors, dt_sequence)
5. is_valid = |estimated_order - reference_order| < 0.2

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| EM + GBM | estimated_order ≈ 0.5, is_valid=True |
| Milstein + GBM | estimated_order ≈ 1.0, is_valid=True |
| SRK + GBM | estimated_order ≈ 1.0, is_valid=True |

### manufactured_solution

**Назначение:** Аналитическое решение GBM для MMS.

**Сигнатура:**
```python
def manufactured_solution(self, t, x0=1.0, mu=0.05, sigma=0.2) -> float
```

**Поведение:**
1. Детерминированная часть: x0 * exp((mu - sigma²/2) * t)
2. Полное решение (со стохастической частью) вычисляется отдельно

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| t=0 | x0 |
| mu=0, sigma=0 | x0 для всех t |
| t>0, mu>0 | > x0 (при sigma²/2 < mu) |

---

## SDEvsABMComparator

**Назначение:** Статистическое сравнение SDE и ABM при большом N.

### compare

**Назначение:** Вычислить все метрики сравнения.

**Сигнатура:**
```python
def compare(self, sde_values: np.ndarray, abm_values: np.ndarray) -> ComparisonMetrics
```

**Поведение:**
1. wasserstein = scipy.stats.wasserstein_distance(sde, abm)
2. ks_stat, ks_p = scipy.stats.ks_2samp(sde, abm)
3. mean_diff = |np.mean(sde) - np.mean(abm)|
4. std_diff = |np.std(sde) - np.std(abm)|
5. is_consistent = ks_p > significance_level

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Одинаковые массивы | W1=0, KS p=1.0, consistent=True |
| Один сдвинутый на 100 | W1 > 0, KS p < 0.05, consistent=False |
| N=1000, одинаковое распред. | consistent=True |

**Edge cases:**
- Пустые массивы → ValueError
- Один элемент → KS-тест некорректен
- Очень разные N → тест всё равно работает

**Инварианты:**
- W1 ≥ 0
- 0 ≤ ks_statistic ≤ 1
- is_consistent == (ks_pvalue > significance_level)

### wasserstein_distance

**Сигнатура:**
```python
def wasserstein_distance(self, sde_values, abm_values) -> float
```

**Поведение:** Обёртка над scipy.stats.wasserstein_distance.

### summary

**Сигнатура:**
```python
def summary(self, metrics: ComparisonMetrics) -> str
```

**Поведение:** Форматированный текстовый отчёт с W1, KS, mean/std diff.
