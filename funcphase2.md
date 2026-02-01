План реализации математического ядра RegenTwin
Обзор задачи
Реализовать функционал 4 модулей математического ядра для прохождения TDD тестов:

src/core/sde_model.py — SDE модель (уравнения Ланжевена)
src/core/abm_model.py — Agent-Based модель
src/core/integration.py — Интеграция SDE+ABM
src/core/monte_carlo.py — Monte Carlo симулятор
Всего: ~63 классов тестов, ~2200 строк тестового кода

Текущее состояние
Все модули содержат полные stub-реализации:

Dataclasses определены полностью (SDEConfig, ABMConfig, etc.)
Методы validate() уже реализованы
Все рабочие методы выбрасывают NotImplementedError
ModelParameters из parameter_extraction.py полностью готов
Порядок реализации

sde_model.py (независимый)
    ↓
abm_model.py (независимый)
    ↓
integration.py (зависит от sde + abm)
    ↓
monte_carlo.py (зависит от всех)
1. SDE Model (src/core/sde_model.py)
1.1 Методы для реализации (в порядке зависимостей)
Метод	Формула/Логика
_logistic_growth(N)	r * N * (1 - N/K)
_apply_boundary_conditions(N, C)	(max(0, N), max(0, C))
_is_therapy_active(t, type)	Проверка start <= t < start + duration
_get_therapy_mask(times, type)	Boolean массив активности
_prp_effect(t, N, C)	alpha * C0 * exp(-lambda*t_therapy) * intensity
_pemf_effect(t, N)	beta * sigmoid(freq) * intensity * N
_therapy_prp_secretion(t)	Дополнительные цитокины из PRP
_calculate_diffusion(t, N, C)	(sigma_n * N, sigma_c * C)
_calculate_drift(t, N, C)	Объединение всех компонентов
simulate(params)	Цикл Эйлера-Маруямы
SDETrajectory.get_final_state()	Последнее состояние
SDETrajectory.get_statistics()	{final_N, final_C, max_N, growth_rate}
1.2 Ключевые алгоритмы
Метод Эйлера-Маруямы:


N[i+1] = N[i] + drift_N * dt + diffusion_N * sqrt(dt) * xi
C[i+1] = C[i] + drift_C * dt + diffusion_C * sqrt(dt) * eta
PRP эффект (экспоненциальное затухание):


t_therapy = t - prp_start_time
effect = alpha_prp * C0 * exp(-lambda_prp * t_therapy) * intensity
if pemf_active: effect *= synergy_factor
PEMF эффект (сигмоида):


sigmoid = 1 / (1 + exp(-k_pemf * (frequency - f0_pemf)))
effect = beta_pemf * sigmoid * intensity * N
2. ABM Model (src/core/abm_model.py)
2.1 Агенты — методы для реализации
Agent (базовый класс):

Метод	Логика
_random_walk_displacement(D, dt)	dx = sqrt(2*D*dt) * N(0,1)
move(dx, dy, space, boundary)	periodic/reflective/absorbing
can_divide()	energy >= threshold && divisions < max
should_die(dt)	`age > lifespan
get_state()	Возвращает AgentState
StemCell:

Метод	Логика
update(dt, env)	age, energy, movement
divide(new_id)	Создание дочерней StemCell
should_differentiate()	random < diff_prob
differentiate(new_id)	Создание Fibroblast
secrete_cytokines(dt)	rate * dt
Macrophage:

Метод	Логика
update(dt, env)	age, energy, movement + polarization
divide(new_id)	Создание дочернего Macrophage
phagocytose(debris)	min(debris, capacity)
polarize(inflammation)	M0→M1 (>0.5) или M2 (<=0.5)
secrete_cytokines(dt)	Зависит от polarization_state
Fibroblast:

Метод	Логика
update(dt, env)	age, energy, movement + ECM
divide(new_id)	Создание дочернего Fibroblast
produce_ecm(dt)	rate * dt (больше если activated)
activate()	Переход в миофибробласт
2.2 ABMModel — методы
Метод	Логика
_create_agent(type, x, y)	Factory для агентов
initialize_from_parameters(params)	Создание начальной популяции
_get_environment(x, y)	cytokine_level, inflammation, etc.
_get_snapshot()	Создание ABMSnapshot
_update_agents(dt)	Вызов update() для всех
_handle_divisions()	Проверка can_divide(), создание потомков
_handle_differentiations()	Проверка should_differentiate()
_remove_dead_agents()	Фильтрация alive == False
_update_cytokine_field(dt)	Диффузия + секреция + деградация
_update_ecm_field(dt)	Обновление ECM поля
step(dt)	Один шаг симуляции
simulate(params, interval)	Полный цикл
2.3 Trajectory методы
Метод	Логика
ABMSnapshot.get_agent_count_by_type()	{stem: N, macro: N, fibro: N}
ABMSnapshot.get_total_agents()	sum(alive)
ABMTrajectory.get_times()	[s.t for s in snapshots]
ABMTrajectory.get_population_dynamics()	По типам во времени
ABMTrajectory.get_statistics()	Финальная статистика
3. Integration (src/core/integration.py)
3.1 Методы для реализации
Метод	Логика
IntegratedTrajectory.get_discrepancy_timeseries()	(times, [s.discrepancy for s])
_calculate_discrepancy(sde_N, abm_count)	`
_apply_correction(sde_N, abm_count, disc)	sde_N + alpha * (abm_count - sde_N)
_update_abm_environment(sde_C)	Обновление cytokine_field
_create_integrated_state(...)	Создание IntegratedState
_run_sde_segment(start, end, N, C)	Сегмент SDE симуляции
_run_abm_segment(start, end)	Сегмент ABM симуляции
_synchronize(sde_N, sde_C, snapshot)	Коррекция + discrepancy
simulate(params)	Operator splitting цикл
simulate_integrated(...)	Convenience функция
3.2 Алгоритм Operator Splitting

while current_time < t_max:
    # 1. SDE шаг
    final_N, final_C = _run_sde_segment(current, next, N, C)

    # 2. ABM шаг
    snapshot = _run_abm_segment(current*24, next*24)  # часы

    # 3. Синхронизация (если bidirectional)
    corrected_N, corrected_C, disc = _synchronize(final_N, final_C, snapshot)

    # 4. Сохранение состояния
    state = _create_integrated_state(...)

    current_time = next_time
4. Monte Carlo (src/core/monte_carlo.py)
4.1 Методы для реализации
Метод	Логика
TrajectoryResult.get_statistics()	Статистика из траектории
TrajectoryResult.get_timeseries(var)	(times, values)
MonteCarloResults.get_summary_statistics()	Агрегированная статистика
MonteCarloResults.get_confidence_interval(var, level)	Через квантили
MonteCarloResults.get_final_distribution(var)	Массив финальных значений
_run_sde_trajectory(params, seed)	Запуск SDE
_run_abm_trajectory(params, seed)	Запуск ABM
_run_integrated_trajectory(params, seed)	Запуск Integrated
_run_single_trajectory(id, params, seed)	Обёртка + timing
_extract_trajectories_array(results, var)	[n_traj, n_steps]
_calculate_quantiles(trajectories, qs)	np.quantile(axis=0)
_extract_summary_stats(result)	Статистика одной траектории
_aggregate_trajectories(results)	Создание MonteCarloResults
run(params)	Основной цикл
run_monte_carlo(...)	Convenience функция
run_parameter_sweep(...)	Sweep по параметру
compare_therapies(...)	Сравнение терапий
4.2 Формулы статистики

# Квантили
quantiles[q] = np.quantile(trajectories, q, axis=0)

# Доверительный интервал (95%)
lower = quantiles[0.025]
upper = quantiles[0.975]

# Среднее и std
mean_N = np.mean(trajectories, axis=0)
std_N = np.std(trajectories, axis=0)
Критические файлы
Файл	Действие
src/core/sde_model.py	Реализовать ~12 методов
src/core/abm_model.py	Реализовать ~25 методов
src/core/integration.py	Реализовать ~10 методов
src/core/monte_carlo.py	Реализовать ~16 методов
Верификация
Запуск тестов

# Все тесты математического ядра
pytest tests/unit/core/ -v

# По модулям
pytest tests/unit/core/test_sde_model.py -v
pytest tests/unit/core/test_abm_model.py -v
pytest tests/unit/core/test_integration.py -v
pytest tests/unit/core/test_monte_carlo.py -v

# С покрытием
pytest tests/unit/core/ --cov=src/core --cov-report=term-missing
Критерии успеха
Все тесты проходят (0 failures)
Воспроизводимость с random_seed
Граничные условия работают корректно
Численная стабильность (N, C >= 0)
Примечания
Временные масштабы: SDE в днях, ABM в часах (1 день = 24 часа)
Граничные условия: Отражающие для N, C >= 0
Seed management: Разные seeds для SDE и ABM в интеграции
Stub методы: Заменять raise NotImplementedError на реализацию
