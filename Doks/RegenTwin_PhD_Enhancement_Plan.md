# План разработки RegenTwin v5.0 — Научные расширения PhD+ уровня

## Обзор проекта

**RegenTwin** — мультимасштабная платформа моделирования регенерации тканей. Данный документ описывает расширения, поднимающие проект до уровня PhD-диссертации и серии публикаций в top-журналах (PLOS Computational Biology, Bioinformatics, Journal of Theoretical Biology, npj Systems Biology and Applications).

**Базис:** Полностью реализованные Фазы 1–2.8 (20-переменная SDE, ABM с 7 типами агентов, PRP/PEMF терапии, 1400+ тестов).

**Расширения:**
- Стохастический бифуркационный анализ (Fokker-Planck, Kramers, Freidlin-Wentzell)
- Multilevel Monte Carlo, Polynomial Chaos Expansion
- Оптимальное управление терапией (принцип Понтрягина)
- Patient-specific калибровка (Approximate Bayesian Computation)
- Physics-Informed Neural Networks, Neural SDE, Gaussian Process эмуляторы
- Внутриклеточная сигнализация (NF-κB, Smad2/3, YAP-TAZ)
- Виртуальные клинические испытания

**Новый стек (дополнение к существующему):**
- **UQ/Calibration:** chaospy, pyabc, arviz
- **ML:** PyTorch, torchsde, torch_geometric, GPyTorch
- **TDA:** giotto-tda, ripser
- **Solvers:** scipy.sparse, scipy.optimize (L-BFGS-B)

---

## Архитектура расширений

```
┌───────────────────────────────────────────────────┐
│      Существующее ядро (v4.0)                      │
│  - extended_sde.py (20 SDE, 1133 LOC)             │
│  - abm_model.py (7 агентов, 2335 LOC)             │
│  - therapy_models.py (PRP/PEMF, 583 LOC)          │
│  - monte_carlo.py (ансамбли, 872 LOC)             │
└────────────────────────┬──────────────────────────┘
                         │ Python imports
                         ▼
┌───────────────────────────────────────────────────┐
│      Научные расширения (v5.0)                     │
│  - src/core/bifurcation.py      (Fokker-Planck)   │
│  - src/core/large_deviations.py (инстантоны)      │
│  - src/core/mlmc.py             (MLMC)            │
│  - src/core/qmc.py              (Quasi-MC)        │
│  - src/core/intracellular.py    (NF-κB, Smad)     │
│  - src/core/abm_tau_leaping.py  (tau-leaping)     │
│  - src/core/convergence_analysis.py (Strang)      │
│  - src/analysis/pce.py          (PCE surrogate)   │
│  - src/analysis/optimal_control.py (Pontryagin)   │
│  - src/analysis/abc_calibration.py (SMC-ABC)      │
│  - src/analysis/model_selection.py (WAIC)         │
│  - src/analysis/validation.py   (R², RMSE)        │
│  - src/analysis/benchmarking.py (vs Flegg/Xue)    │
│  - src/analysis/decision_support.py (CDS)         │
│  - src/analysis/virtual_trials.py (in-silico)     │
│  - src/analysis/tda.py          (persistent H)    │
│  - src/ml/pinn.py               (PINNs)           │
│  - src/ml/neural_sde.py         (Neural SDE)      │
│  - src/ml/gp_emulator.py        (GP surrogate)    │
│  - src/ml/gnn_surrogate.py      (GNN for ABM)     │
└───────────────────────────────────────────────────┘
```

---

## Структура новых файлов

```
RegenTwin/
├── src/
│   ├── core/                          # Расширения ядра
│   │   ├── bifurcation.py             # ✖ Бифуркационный анализ, Fokker-Planck, Kramers
│   │   ├── large_deviations.py        # ✖ Freidlin-Wentzell, string method, инстантоны
│   │   ├── mlmc.py                    # ✖ Multilevel Monte Carlo
│   │   ├── qmc.py                     # ✖ Quasi-Monte Carlo + Brownian bridge
│   │   ├── intracellular.py           # ✖ NF-κB, Smad2/3, YAP-TAZ ODE модули
│   │   ├── abm_tau_leaping.py         # ✖ Adaptive tau-leaping для ABM
│   │   └── convergence_analysis.py    # ✖ Strang splitting, Richardson extrapolation
│   ├── analysis/                      # Научный анализ
│   │   ├── pce.py                     # ✖ Polynomial Chaos Expansion (chaospy)
│   │   ├── optimal_control.py         # ✖ Pontryagin, forward-backward sweep
│   │   ├── abc_calibration.py         # ✖ SMC-ABC (pyabc)
│   │   ├── model_selection.py         # ✖ WAIC, Bayes factors (arviz)
│   │   ├── validation.py              # ✖ R², RMSE, phase timing metrics
│   │   ├── benchmarking.py            # ✖ Flegg 2015, Xue 2009, Vodovotz 2006
│   │   ├── decision_support.py        # ✖ Risk scores, therapy recommendation
│   │   ├── virtual_trials.py          # ✖ In-silico clinical trials
│   │   └── tda.py                     # ✖ Topological Data Analysis
│   └── ml/                            # Machine Learning
│       ├── __init__.py                # ✖
│       ├── pinn.py                    # ✖ Physics-Informed Neural Networks
│       ├── neural_sde.py              # ✖ Neural SDE (hybrid drift)
│       ├── gp_emulator.py             # ✖ Gaussian Process emulators
│       └── gnn_surrogate.py           # ✖ Graph Neural Networks для ABM
├── tests/
│   ├── unit/core/
│   │   ├── test_bifurcation.py        # ✖
│   │   ├── test_large_deviations.py   # ✖
│   │   ├── test_mlmc.py               # ✖
│   │   ├── test_qmc.py                # ✖
│   │   ├── test_intracellular.py      # ✖
│   │   ├── test_tau_leaping.py        # ✖
│   │   └── test_convergence.py        # ✖
│   ├── unit/analysis/
│   │   ├── test_pce.py                # ✖
│   │   ├── test_optimal_control.py    # ✖
│   │   ├── test_abc_calibration.py    # ✖
│   │   ├── test_model_selection.py    # ✖
│   │   ├── test_validation.py         # ✖
│   │   ├── test_benchmarking.py       # ✖
│   │   └── test_virtual_trials.py     # ✖
│   ├── unit/ml/
│   │   ├── test_pinn.py               # ✖
│   │   ├── test_neural_sde.py         # ✖
│   │   ├── test_gp_emulator.py        # ✖
│   │   └── test_gnn_surrogate.py      # ✖
│   └── validation/
│       └── test_literature_data.py    # ✖ Валидация на опубликованных данных
├── data/
│   └── validation/                    # Литературные данные
│       ├── neutrophil_kinetics.json   # ✖ Kolaczkowska & Kubes 2013
│       ├── macrophage_polarization.json # ✖ Murray 2017
│       ├── collagen_deposition.json   # ✖ Xue et al. 2009
│       └── wound_closure.json         # ✖ Flegg et al. 2015
└── Description/
    └── Phase3/
        ├── description_bifurcation.md      # ✖
        ├── description_mlmc.md             # ✖
        ├── description_optimal_control.md  # ✖
        ├── description_pinn.md             # ✖
        └── description_intracellular.md    # ✖
```

---

## Условные обозначения

| Символ | Значение |
|--------|----------|
| ✔ | Полностью реализовано |
| ◐ | Частично реализовано |
| ✖ | Не реализовано |

---

## Фаза 3.0: Валидация на литературных данных ✖ НЕ НАЧАТО (0%)

> **Цель:** Валидация существующей 20-переменной SDE на опубликованных данных. Необходимо для любой публикации.
> **Приоритет:** КРИТИЧЕСКИЙ
> **Зависимости:** Фаза 2.5 (расширенная SDE)

### 3.0.1 Референсные данные

| Источник | Переменные | Временные точки | Применение |
|----------|-----------|-----------------|-----------|
| Kolaczkowska & Kubes 2013 | Нейтрофилы Ne(t) | 0, 6, 12, 24, 48, 72 ч | Пик на 12-24ч, клиренс к 72ч |
| Murray 2017 | M1/M2 ratio | 0, 24, 48, 72, 96, 120 ч | M1-доминант на 24ч, M2 к дню 4-5 |
| Xue et al. 2009 | Коллаген ρ_c(t) | Дни 0, 3, 5, 7, 10, 14, 21 | Сигмоидный рост |
| Flegg et al. 2015 | Площадь раны | Дни 0-28 | Экспоненциальное приближение к закрытию |
| Desmoulière et al. 1993 | Миофибробласты | Дни 5, 10, 15, 20 | Пик на 10-15 дней, апоптоз |
| Ferrara 2004 | VEGF | Часы 0-168 | Пик при гипоксии |

### 3.0.2 Метрики валидации

| Метрика | Формула | Описание | Статус |
|---------|---------|----------|--------|
| Temporal R² | 1 - Σ(y_obs - y_pred)² / Σ(y_obs - ȳ)² | Корреляция траекторий | ✖ |
| RMSE | √(mean((y_obs - y_pred)²)) | Абсолютная ошибка | ✖ |
| Phase timing error | \|t_transition_model - t_transition_literature\| | Точность фазовых переходов (ч) | ✖ |
| MC envelope coverage | % наблюдений внутри 95% CI ансамбля | Калибровка неопределённости | ✖ |

### 3.0.3 Бенчмаркинг

| Модель-референция | Публикация | Переменные | Сравнение | Статус |
|-------------------|-----------|-----------|-----------|--------|
| Vodovotz 2006 | Curr. Opin. Crit. Care | 2 (pathogen, damage) | R² на данных воспаления | ✖ |
| Xue 2009 | PNAS | PDE (O₂, клетки, ECM) | R² на данных коллагена | ✖ |
| Flegg 2015 | Bull. Math. Biol. | ODE (GF, F, коллаген) | R² на данных закрытия раны | ✖ |

**Тест:** 20-переменная RegenTwin должна давать статистически значимое улучшение (paired t-test, p < 0.05) по R² и RMSE на всех трёх датасетах.

### Файлы для создания

| Файл | Описание | Классы/Функции | Статус |
|------|----------|----------------|--------|
| `src/analysis/validation.py` | Метрики валидации | `ValidationRunner`, `TemporalR2`, `PhaseTimingMetric`, `MCEnvelopeCoverage` | ✖ |
| `src/analysis/benchmarking.py` | Бенчмаркинг | `BenchmarkSuite`, `VodovotzModel`, `XueModel`, `FleggModel`, `compare_models()` | ✖ |
| `data/validation/*.json` | Литературные данные | JSON: {time_hours: [], values: [], ci_lower: [], ci_upper: [], source: ""} | ✖ |
| `tests/unit/analysis/test_validation.py` | TDD тесты | ~30 тестов: R², RMSE, phase timing, coverage | ✖ |
| `tests/unit/analysis/test_benchmarking.py` | TDD тесты | ~25 тестов: упрощённые модели, сравнение | ✖ |
| `tests/validation/test_literature_data.py` | Интеграционные тесты | ~15 тестов: полная симуляция vs данные | ✖ |

---

## Фаза 3.1: Стохастический бифуркационный анализ ✖ НЕ НАЧАТО (0%)

> **Цель:** Строгий математический анализ бистабильности TGF-β/миофибробласт (нормальное заживление vs фиброз). Это главный научный вклад проекта — ни одна существующая модель заживления ран не проводила такой анализ.
> **Приоритет:** КРИТИЧЕСКИЙ
> **Зависимости:** Фаза 2.5 (extended_sde.py, drift-функции)
> **Публикация:** PLOS Computational Biology

### 3.1.1 Редуцированная 2D система

Подсистема TGF-β/миофибробласт из `extended_sde.py` (строки 717, 1025):

```
dMf = [k_act · F · H(TGFb, K_activ, 2) - δ_Mf · Mf · (1 - TGFb/(K_survival + TGFb))] dt + σ_Mf · Mf · dW_Mf
dTGFb = [s_TGF_P · k_deg · P + s_TGF_M2 · M₂ + s_TGF_Mf · Mf - γ_TGF · TGFb] dt + σ_TGF · TGFb · dW_TGF
```

**Биологический смысл:** Мf секретирует TGF-β → TGF-β активирует F→Mf переход + подавляет апоптоз Mf → положительная обратная связь → два устойчивых состояния.

### 3.1.2 Детерминированный анализ

| Компонент | Метод | Описание | Статус |
|-----------|-------|----------|--------|
| Нуллклины | scipy.optimize.brentq | dMf/dt=0, dTGFb/dt=0 — кривые в (Mf, TGFb) плоскости | ✖ |
| Фиксированные точки | scipy.optimize.fsolve | Пересечения нуллклин — 1 или 3 точки | ✖ |
| Якобиан 20×20 | Конечные разности | J_ij = ∂μ_i/∂x_j для полной 20-переменной системы | ✖ |
| Классификация устойчивости | numpy.linalg.eig | Собственные значения J(x*): Re(λ) < 0 ⟹ устойчив | ✖ |
| Бифуркационная диаграмма | Параметрическая развёртка | Mf* vs k_act (или s_TGF_Mf) — saddle-node бифуркация | ✖ |

### 3.1.3 Стохастический анализ

| Компонент | Метод | Формула | Статус |
|-----------|-------|---------|--------|
| Стационарный Fokker-Planck | FD + Scharfetter-Gummel | 0 = -∂/∂Mf[μ_Mf·p] - ∂/∂TGFb[μ_TGFb·p] + ½∂²/∂Mf²[(σ_Mf·Mf)²·p] + ½∂²/∂TGFb²[(σ_TGF·TGFb)²·p] | ✖ |
| Квази-потенциал | gMAM (L-BFGS-B) | V(x) = min_φ ½∫\|\|σ⁻¹(φ)·[dφ/dt - μ(φ)]\|\|² dt | ✖ |
| Скорость Крамерса | Аналитическая | r_escape ~ (ω_well·ω_saddle)/(2π) · exp(-2ΔV/ε) | ✖ |
| Стохастические экспоненты Ляпунова | Алгоритм Benettin | λ_max = lim_{t→∞} (1/t)·ln(\|\|Φ(t)\|\|) с QR-декомпозицией | ✖ |

### 3.1.4 Large Deviation Theory

| Компонент | Метод | Описание | Статус |
|-----------|-------|----------|--------|
| Функционал действия | Freidlin-Wentzell | I(φ) = ½·Σᵢ ∫[(dφᵢ/dt - μᵢ)/(σᵢ·φᵢ)]² dt | ✖ |
| Инстантон | String method | Наиболее вероятный путь normal→fibrosis | ✖ |
| Вероятность патологии | Аналитическая | P(fibrosis) ~ exp(-I*/ε), ε = max(σᵢ²) | ✖ |
| Флуктуационный детерминант | Численный | Префактор через отношение функциональных детерминантов | ✖ |

### Файлы для создания

| Файл | Описание | Классы/Функции | Статус |
|------|----------|----------------|--------|
| `src/core/bifurcation.py` | Бифуркационный анализ | `NullclineComputer`, `FixedPointFinder`, `JacobianComputer`, `StabilityClassifier`, `FokkerPlanckSolver`, `QuasiPotentialSolver`, `KramersRateCalculator`, `LyapunovExponentComputer` | ✖ |
| `src/core/large_deviations.py` | Теория больших отклонений | `ActionFunctional`, `InstantonSolver`, `StringMethod`, `FluctuationDeterminant` | ✖ |
| `tests/unit/core/test_bifurcation.py` | TDD тесты | ~60 тестов: нуллклины, фикс. точки, якобиан, FP, Kramers | ✖ |
| `tests/unit/core/test_large_deviations.py` | TDD тесты | ~30 тестов: действие, инстантон, string method | ✖ |
| `Description/Phase3/description_bifurcation.md` | Описание | Математика + TDD секции | ✖ |

### Ключевые ссылки

| Ссылка | Применение |
|--------|-----------|
| Horsthemke & Lefever, "Noise-Induced Transitions" (Springer, 1984) | Стохастические бифуркации |
| Freidlin & Wentzell, "Random Perturbations of Dynamical Systems" 3rd ed (2012) | Large deviations |
| Heymann & Vanden-Eijnden, Comm Pure Appl Math 2008; 61(8):1052 | gMAM метод |
| Scharfetter & Gummel, IEEE Trans Electron Devices 1969; 16(1):64 | Upwind FD схема |
| Benettin et al., Meccanica 1980; 15:9-20 | Ляпуновские экспоненты |
| Strogatz, "Nonlinear Dynamics and Chaos" 2nd ed (2015) | Бифуркационная теория |
| Lv et al., PNAS 2014; 111(29):10510-10515 | Switching mechanism |

---

## Фаза 3.2: Продвинутые численные методы ✖ НЕ НАЧАТО (0%)

> **Цель:** Снижение вычислительных затрат UQ в 10-100x. Без этого ABC-калибровка и виртуальные испытания вычислительно непрактичны.
> **Приоритет:** КРИТИЧЕСКИЙ
> **Зависимости:** Фаза 2.7 (sde_numerics.py — солверы)
> **Публикация:** Bioinformatics

### 3.2.1 Multilevel Monte Carlo (MLMC)

**Математическая формулировка:**

Стандартный MC для E[f(X_T)]: стоимость O(N·dt⁻¹), дисперсия O(1/N) → полная стоимость O(ε⁻³).

MLMC использует иерархию разрешений dt₀ > dt₁ > ... > dt_L с dt_l = M⁻ˡ·dt₀:

```
E[f(X_T^L)] = E[f(X_T^0)] + Σ_{l=1}^L E[f(X_T^l) - f(X_T^{l-1})]
```

На каждом уровне l: N_l связанных выборок, дисперсия V_l = O(dt_l).

Оптимальное распределение:
```
N_l = O(ε⁻² · √(V_l/C_l) · Σ_k √(V_k·C_k))
```

Итоговая стоимость: **O(ε⁻²)** — улучшение на O(ε⁻¹).

| Компонент | Описание | Статус |
|-----------|----------|--------|
| Связанные броуновские пути | Fine dW → суммирование M инкрементов → coarse dW | ✖ |
| Предгенерированные пути | Параметр `dW_sequence` в `ExtendedSDEModel.simulate` | ✖ |
| MLMC estimator | Алгоритм Giles 2015: пилот → V_l, C_l → оптим. N_l | ✖ |
| Адаптивный выбор уровней | Автоматическое определение L | ✖ |
| CI через CLT | Доверительные интервалы на каждом уровне | ✖ |

### 3.2.2 Polynomial Chaos Expansion (PCE)

**Математическая формулировка:**

```
X(t, ξ) = Σ_{α∈A} c_α(t) · Ψ_α(ξ)
```

где ξ = (ξ₁,...,ξ_d) — стандартизованные случайные величины параметрических неопределённостей, Ψ_α — ортогональные полиномы (Hermite для Gaussian, Legendre для uniform).

Для d=10 ключевых параметров, порядок p=3: |A| = C(13,3) = 286 базисных функций.

Индексы Соболя аналитически:
```
S_i = Σ_{α: α_i>0, α_j=0 ∀j≠i} c_α² / Σ_{α≠0} c_α²
```

| Компонент | Описание | Статус |
|-----------|----------|--------|
| Базис полиномов | chaospy: Hermite/Legendre, multivariate | ✖ |
| Sparse quadrature | Точки коллокации (Smolyak grid) | ✖ |
| Коэффициенты PCE | Регрессия (overdetermined, least squares) | ✖ |
| Sobol из PCE | Аналитическое извлечение из c_α | ✖ |
| Суррогатная модель | Для нового θ: оценка за O(\|A\|) операций | ✖ |

### 3.2.3 Quasi-Monte Carlo

| Компонент | Описание | Статус |
|-----------|----------|--------|
| Sobol sequences | scipy.stats.qmc.Sobol → dW = Φ⁻¹(u)·√dt | ✖ |
| Brownian bridge | W(T) → W(T/2) → W(T/4), W(3T/4) → ... | ✖ |
| QMCSolver wrapper | Обёртка EulerMaruyamaSolver с QMC-генератором | ✖ |

### 3.2.4 Adaptive Tau-Leaping для ABM

| Компонент | Описание | Статус |
|-----------|----------|--------|
| Propensity functions | a_j(X) для деления, смерти, миграции, switch | ✖ |
| Tau selection | τ = min_j { ε·a_j / \|da_j/dt\| } (Cao et al. 2006) | ✖ |
| Poisson sampling | K_j(τ) ~ Poisson(a_j·τ) для каждого канала | ✖ |
| Hybrid SSA fallback | Если τ < dt_min → exact Gillespie | ✖ |

### Файлы для создания

| Файл | Описание | Классы/Функции | LOC (ожид.) | Статус |
|------|----------|----------------|-------------|--------|
| `src/core/mlmc.py` | MLMC | `CoupledBrownianPaths`, `MLMCEstimator`, `MLMCLevelConfig`, `adaptive_level_selection()` | ~600 | ✖ |
| `src/analysis/pce.py` | PCE | `PCESurrogate`, `SparseQuadrature`, `PCESobolExtractor` | ~500 | ✖ |
| `src/core/qmc.py` | Quasi-MC | `BrownianBridge`, `QMCSolver`, `SobolBrownianGenerator` | ~300 | ✖ |
| `src/core/abm_tau_leaping.py` | Tau-leaping | `PropensityFunction`, `TauLeapingSolver`, `HybridSSA` | ~400 | ✖ |
| `tests/unit/core/test_mlmc.py` | | ~40 тестов: связанные пути, дисперсия, convergence rate | | ✖ |
| `tests/unit/analysis/test_pce.py` | | ~35 тестов: базис, регрессия, Sobol, surrogate | | ✖ |
| `tests/unit/core/test_qmc.py` | | ~20 тестов: Brownian bridge, discrepancy | | ✖ |
| `tests/unit/core/test_tau_leaping.py` | | ~25 тестов: propensity, tau selection, Poisson | | ✖ |

### Ключевые ссылки

| Ссылка | Применение |
|--------|-----------|
| Giles, Acta Numerica 2015; 24:259-328 | MLMC алгоритм |
| Giles, Operations Research 2008; 56(3):607 | MLMC для SDE |
| Xiu & Karniadakis, SIAM J Sci Comput 2002; 24(2):619 | PCE теория |
| Sudret, Reliability Eng & Sys Safety 2008; 93(7):964 | PCE → Sobol |
| Feinberg & Langtangen, J Comput Sci 2015; 11:46 | chaospy |
| Dick, Kuo & Sloan, Acta Numerica 2013; 22:133 | QMC теория |
| Cao, Gillespie & Petzold, J Chem Phys 2006; 124:044109 | Tau-leaping |

---

## Фаза 3.3: Оптимальное управление и калибровка ✖ НЕ НАЧАТО (0%)

> **Цель:** Математически оптимальный ответ на вопрос "когда применять PRP и PEMF?" + калибровка модели под конкретного пациента.
> **Приоритет:** КРИТИЧЕСКИЙ
> **Зависимости:** Фаза 3.1 (якобиан), Фаза 3.2 (MLMC)
> **Публикация:** Journal of Theoretical Biology

### 3.3.1 Оптимальное управление терапией (принцип Понтрягина)

**Управляющие переменные:** u(t) = (u_PRP(t), u_PEMF(t))

**Функционал цели:**
```
J[u] = ∫₀ᵀ L(X(t), u(t)) dt + Φ(X(T))
```

Текущие затраты (штраф за воспаление + стоимость терапии):
```
L = w₁·(M₁ + Nₑ) + w₂·(1 - ρ_collagen/ρ_max) + w₃·(u_PRP² + u_PEMF²)
```

Терминальные затраты (неполное заживление):
```
Φ = w₄·(1 - ρ_collagen(T)/ρ_max) + w₅·Mf(T)
```

**Сопряжённые уравнения (Pontryagin):**
```
dp/dt = -∂H/∂X = -(∂L/∂X + pᵀ · ∂μ/∂X)
```

**Оптимальное управление:**
```
u*(t) = -(1/(2w₃)) · pᵀ · ∂μ/∂u
```

| Компонент | Метод | Описание | Статус |
|-----------|-------|----------|--------|
| Forward state | scipy.integrate.solve_ivp | X(t) по детерминистическому скелету | ✖ |
| Backward costate | scipy.integrate.solve_ivp | p(t) от T к 0 | ✖ |
| Forward-backward sweep | Итеративный | X → p → u → X → p → ... до сходимости | ✖ |
| Direct transcription | scipy.optimize.minimize | L-BFGS-B на дискретизированной u(t) | ✖ |
| Constraint: u_PRP | Бинарный: {0, dose_standard} | Когда инъецировать PRP | ✖ |
| Constraint: u_PEMF | Непрерывный: [0, B_max] | Интенсивность PEMF | ✖ |

### 3.3.2 Patient-Specific калибровка (SMC-ABC)

**Approximate Bayesian Computation** — без вычисления likelihood:

```
1. θ* ~ p(θ)                  # Sample from prior
2. X_sim = Model(θ*)           # Simulate
3. Accept θ* if d(S(X_sim), S(D)) < ε   # Compare summary stats
```

SMC-ABC (Sequential Monte Carlo):
```
θ⁽ⁿ⁾ ~ K(θ | θ⁽ⁿ⁻¹⁾)    # Perturbation kernel
w⁽ⁿ⁾ = p(θ⁽ⁿ⁾) / Σⱼ w⁽ⁿ⁻¹⁾ⱼ · K(θ⁽ⁿ⁾ | θ⁽ⁿ⁻¹⁾ⱼ)    # Weights
```

Tolerance schedule: ε₁ > ε₂ > ... > ε_T

| Компонент | Описание | Статус |
|-----------|----------|--------|
| Summary statistics | Из WoundPhaseDetector: фазовые переходы, peak counts, final collagen | ✖ |
| Prior | Log-normal, центрированные на ParameterSet defaults, CV=0.3 | ✖ |
| Subset параметров | 5-15 из 105+ (отбор по Sobol indices) | ✖ |
| Параллелизация | pyabc distributed computation | ✖ |
| Posterior predictive checks | Forward simulation из calibrated posterior | ✖ |

### 3.3.3 Model Selection

| Метод | Формула | Описание | Статус |
|-------|---------|----------|--------|
| WAIC | -2·(lppd - p_WAIC) | Widely Applicable Information Criterion | ✖ |
| LOO-CV | Leave-One-Out Cross Validation | Via Pareto-smoothed IS (arviz) | ✖ |
| Bayes factors | BF₁₂ = p(D\|M₁)/p(D\|M₂) | Сравнение: 20-var vs 2-var, с/без NF-κB | ✖ |

**Тест:** Сравнить (a) 20-var SDE vs 2-var MVP, (b) с/без intracellular, (c) с/без microbiome.

### Файлы для создания

| Файл | Описание | Классы/Функции | Статус |
|------|----------|----------------|--------|
| `src/analysis/optimal_control.py` | Оптимальное управление | `CostFunctional`, `CostateEquations`, `ForwardBackwardSweep`, `DirectTranscription`, `TherapyScheduleOptimizer` | ✖ |
| `src/analysis/abc_calibration.py` | ABC калибровка | `SummaryStatisticsExtractor`, `ABCCalibrator`, `PosteriorPredictiveChecker` | ✖ |
| `src/analysis/model_selection.py` | Выбор модели | `WAICCalculator`, `BayesFactorEstimator`, `ModelComparer` | ✖ |
| `tests/unit/analysis/test_optimal_control.py` | | ~40 тестов | ✖ |
| `tests/unit/analysis/test_abc_calibration.py` | | ~35 тестов | ✖ |
| `tests/unit/analysis/test_model_selection.py` | | ~20 тестов | ✖ |

### Ключевые ссылки

| Ссылка | Применение |
|--------|-----------|
| Lenhart & Workman, "Optimal Control Applied to Biological Models" (Chapman & Hall, 2007) | Теория |
| Stengel, Kim & Day, Optimal Control Appl Methods 2013; 34(3):263 | Immune response OC |
| Toni et al., J Royal Soc Interface 2009; 6(31):187 | ABC алгоритм |
| Klinger et al., Bioinformatics 2018; 34(20):3591 | pyABC |
| Vehtari, Gelman & Gabry, Stat Comput 2017; 27(5):1413 | LOO-CV, WAIC |

---

## Фаза 3.4: Биологическая глубина ✖ НЕ НАЧАТО (0%)

> **Цель:** Мостик между молекулярным и клеточным масштабами. Добавление внутриклеточной сигнализации, метаболизма, тренированного иммунитета и микробиома.
> **Приоритет:** ВЫСОКИЙ
> **Зависимости:** Фаза 2.5 (extended_sde.py)
> **Публикация:** npj Systems Biology and Applications

### 3.4.1 Внутриклеточная сигнализация: NF-κB, Smad2/3, YAP-TAZ

**NF-κB модуль** (3 ODE, Hoffmann et al. 2002):
```
d[NF-κB_free]/dt = k_import·[NF-κB_cyto] - k_export·[NF-κB_nuc] - k_bind·[NF-κB_nuc]·[IκBα_nuc]
d[IκBα]/dt = s_IκBα·[NF-κB_nuc] - δ_IκBα·[IκBα] - k_IKK·[IKK]·[IκBα]
d[IKK]/dt = f_TNF(C_TNF) - δ_IKK·[IKK]
```

NF-κB модулирует продукцию TNF-α и IL-8 в SDE:
```
s_TNF_M1 → s_TNF_M1 · (1 + α_NFκB · [NF-κB_nuc])
```

**Smad2/3 модуль** (TGF-β сигнализация):
```
d[Smad_active]/dt = k_activate · C_TGFβ · [Smad_inactive] / (K_Smad + [Smad_inactive]) - k_deactivate · [Smad_active]
```

Заменяет эмпирическую Hill-функцию A(C_TGFβ) на механистическую:
```
k_act · F · [Smad_active] / (K_Smad_threshold + [Smad_active])
```

**YAP-TAZ модуль** (механочувствительность):
```
d[YAP_nuc]/dt = k_mech · σ_stress / (K_stress + σ_stress) · [YAP_cyto] - k_export · [YAP_nuc]
```

| Компонент | Входы | Выходы | Интеграция | Статус |
|-----------|-------|--------|-----------|--------|
| NF-κB | C_TNF от SDE | α_NFκB → _drift_C_TNF, _drift_C_IL8 | quasi-steady-state | ✖ |
| Smad2/3 | C_TGFβ от SDE | Заменяет _activation_function в extended_sde.py (строка 1010) | явная подстановка | ✖ |
| YAP-TAZ | σ_stress от MechanotransductionEngine | YAP_nuc → r_F, persistence Mf | модификация drift | ✖ |

### 3.4.2 Метаболические ограничения

Иммунометаболизм: M1/нейтрофилы → гликолиз (2 ATP), M2/фибробласты → окислительное фосфорилирование (36 ATP):

```
d[Glucose]/dt = D_glc·(Glc_blood - [Glucose])/L² - k_glyc·(Ne+M1)·[Glucose]/(K_glc+[Glucose]) - k_oxphos·(M2+F)·[Glucose]/(K_glc_ox+[Glucose])
d[Lactate]/dt = 2·k_glyc·(Ne+M1)·[Glucose]/(K_glc+[Glucose]) - k_clear·[Lactate]
```

Обратная связь: `r_F → r_F · ATP_available / (K_ATP + ATP_available)`

| Новые переменные | Описание | Единицы | Статус |
|-----------------|----------|---------|--------|
| Glucose(t) | Внеклеточная глюкоза | мМ | ✖ |
| Lactate(t) | Лактат (продукт гликолиза) | мМ | ✖ |

### 3.4.3 Тренированный иммунитет

```
d[Epi_M]/dt = k_imprint · [NF-κB_nuc] · (1 - [Epi_M]) - k_decay_epi · [Epi_M]
k_switch_effective = k_switch · (1 - β_epi · [Epi_M])
```

**Биологический смысл:** Высокий [Epi_M] (trained immunity) замедляет M1→M2 переключение → затяжное воспаление → хронические раны. Объясняет почему у пациентов с диабетом раны заживают хуже.

### 3.4.4 Микробиом-рана

```
dB = [r_B · B · (1-B/K_B) - k_kill_Ne · Ne · B/(K_phag_B+B) - k_kill_M1 · M1 · B/(K_phag_B+B)] dt + σ_B · B · dW_B
d[PAMPs]/dt = s_PAMP · B - γ_PAMP · [PAMPs]
s_TNF_M1 → s_TNF_M1 · (1 + α_PAMP · [PAMPs])
```

| Новые переменные | Описание | Статус |
|-----------------|----------|--------|
| B(t) — бактериальная нагрузка | Логистический рост + фагоцитоз Ne/M1 | ✖ |
| PAMPs(t) | Pathogen-associated molecular patterns | ✖ |
| Epi_M(t) | Эпигенетическое состояние макрофагов | ✖ |

### Файлы для создания

| Файл | Описание | Классы | Статус |
|------|----------|--------|--------|
| `src/core/intracellular.py` | Внутриклеточные модули | `NFkBModule`, `Smad23Module`, `YAPTAZModule`, `IntracellularManager` | ✖ |
| `tests/unit/core/test_intracellular.py` | TDD тесты | ~50 тестов: NF-κB осцилляции, Smad активация, YAP | ✖ |
| Модификация `src/core/extended_sde.py` | +5-7 переменных | Glucose, Lactate, Epi_M, B, PAMPs + intracellular | ✖ |
| Модификация `src/core/therapy_models.py` | Антибиотик | `AntibioticModel` в therapy_models.py | ✖ |

### Ключевые ссылки

| Ссылка | Применение |
|--------|-----------|
| Hoffmann et al., Science 2002; 298(5596):1241 | NF-κB модель |
| Schmierer & Hill, Nature Cell Bio 2007; 9:1008 | Smad2/3 |
| Dupont et al., Nature 2011; 474:179 | YAP/TAZ |
| O'Neill et al., Nature Rev Immunology 2016; 16:553 | Иммунометаболизм |
| Netea et al., Nature Rev Immunology 2020; 20:375 | Trained immunity |
| Kalan et al., J Clin Invest 2019; 129(5):2190 | Wound microbiome |

---

## Фаза 3.5: Machine Learning интеграция ✖ НЕ НАЧАТО (0%)

> **Цель:** ML-методы для параметрической идентификации, суррогатного моделирования и ускорения ABM.
> **Приоритет:** ВЫСОКИЙ
> **Зависимости:** Фаза 2.5 (extended_sde.py), Фаза 3.0 (данные для обучения)
> **Публикация:** npj Systems Biology and Applications

### 3.5.1 Physics-Informed Neural Networks (PINNs)

**Loss function:**
```
L(θ) = L_data + λ_PDE · L_PDE + λ_BC · L_BC

L_data = Σ_k ||u_θ(t_k) - X_observed(t_k)||²
L_PDE = Σ_j ||du_θ/dt(t_j) - μ(u_θ(t_j), p_unknown)||²
L_BC  = ||u_θ(0) - X₀||²
```

Неизвестные параметры p_unknown (k_switch, K_activ, γ_TGF и др.) обучаются совместно с весами нейросети.

| Компонент | Описание | Статус |
|-----------|----------|--------|
| Архитектура | MLP: 4 hidden layers, 128 neurons, tanh activation | ✖ |
| torch_drift | Реимплементация 20 drift-функций в PyTorch (differentiable) | ✖ |
| Auto-differentiation | torch.autograd.grad для du_θ/dt | ✖ |
| Обучение | Adam, 10000 epochs, Latin hypercube collocation | ✖ |
| Multi-fidelity | PCE surrogate (Фаза 3.2) для pretraining | ✖ |

### 3.5.2 Neural SDE (гибридный подход)

```
dX = f_θ(X, t) dt + g_φ(X, t) dW
f_θ(X, t) = μ_known(X) + Δf_θ(X, t)    # known drift + learned residual
```

| Компонент | Описание | Статус |
|-----------|----------|--------|
| Hybrid drift | Known μ из extended_sde.py + neural correction | ✖ |
| Diffusion learning | g_φ = diagonal positive matrix (Cholesky factor) | ✖ |
| Training | SDE likelihood: log p(X_{0:T}) = Σ_n log N(X_{n+1}; X_n + f_θ·dt, g_φ²·dt) | ✖ |
| Validation | На синтетических данных → затем на клинических | ✖ |

### 3.5.3 Gaussian Process эмуляторы

```
Y(θ) ~ GP(m(θ), k(θ, θ'))
k(θ, θ') = σ_f² · (1 + √5·r/l + 5r²/(3l²)) · exp(-√5·r/l)    # Matérn 5/2
```

| Компонент | Описание | Статус |
|-----------|----------|--------|
| Training data | LHS(parameters) → simulate → summary statistics | ✖ |
| Multi-output GP | ICM (Intrinsic Coregionalization Model) | ✖ |
| Active learning | Expected Improvement для добавления training points | ✖ |
| Real-time prediction | 200-500 runs → ms-предсказания с калиброванной неопределённостью | ✖ |

### 3.5.4 Graph Neural Networks для ABM

```
h_i^{l+1} = UPDATE(h_i^l, AGGREGATE({h_j^l : j ∈ N(i)}))
```

ABM как граф G=(V,E): V = {agents}, E = {pairs within interaction_radius} из cKDTree.

| Компонент | Описание | Статус |
|-----------|----------|--------|
| Graph construction | cKDTree.query_ball_point → edge_index | ✖ |
| Architecture | 3-layer MPNN. Node: [type_one_hot, x, y, energy, age]. Edge: [distance, rel_pos] | ✖ |
| Training | ABM trajectories → (G(t), h(t), h(t+dt)) | ✖ |
| Speedup | 10-100x vs direct ABM | ✖ |

### Файлы для создания

| Файл | Описание | Классы | LOC | Статус |
|------|----------|--------|-----|--------|
| `src/ml/__init__.py` | Init | — | ~10 | ✖ |
| `src/ml/pinn.py` | PINNs | `TorchDrift`, `PINNTrainer`, `ParameterDiscovery` | ~600 | ✖ |
| `src/ml/neural_sde.py` | Neural SDE | `HybridDrift`, `LearnedDiffusion`, `NeuralSDETrainer` | ~500 | ✖ |
| `src/ml/gp_emulator.py` | GP | `GPEmulator`, `ActiveLearner`, `MultiOutputGP` | ~400 | ✖ |
| `src/ml/gnn_surrogate.py` | GNN | `ABMGraph`, `MPNNSurrogate`, `GNNTrainer` | ~500 | ✖ |
| `tests/unit/ml/test_pinn.py` | | ~35 тестов | | ✖ |
| `tests/unit/ml/test_neural_sde.py` | | ~30 тестов | | ✖ |
| `tests/unit/ml/test_gp_emulator.py` | | ~25 тестов | | ✖ |
| `tests/unit/ml/test_gnn_surrogate.py` | | ~25 тестов | | ✖ |

### Ключевые ссылки

| Ссылка | Применение |
|--------|-----------|
| Raissi et al., J Comput Phys 2019; 378:686 | PINNs |
| Yazdani et al., PNAS 2020; 117(47):29571 | PINNs + biology |
| Kidger et al., ICML 2021 | Neural SDE |
| Tzen & Raginsky, ICLR 2019 | Variational SDE |
| Rasmussen & Williams, "GP for ML" (MIT Press, 2006) | GP теория |
| Kennedy & O'Hagan, JRSS B 2001; 63(3):425 | GP calibration |
| Sanchez-Gonzalez et al., ICML 2020 | GNN + physics |

---

## Фаза 3.6: Клиническая трансляция ✖ НЕ НАЧАТО (0%)

> **Цель:** Перевод математической модели в инструмент клинической поддержки решений.
> **Приоритет:** ВЫСОКИЙ
> **Зависимости:** Фаза 3.3 (ABC калибровка, оптимальное управление), Фаза 3.5 (GP эмулятор)

### 3.6.1 Система поддержки клинических решений

```
P(нормальное заживление | θ_patient) = ∫ I(X(T,θ) ∈ Ω_normal) · p(θ | D_patient) dθ
≈ (1/N) · Σᵢ I(X(T,θᵢ) ∈ Ω_normal),   θᵢ ~ p(θ | D_patient)
```

| Компонент | Описание | Статус |
|-----------|----------|--------|
| GP fast forward | GP эмулятор (Фаза 3.5.3) для ms-предсказаний | ✖ |
| Risk scores | P(хроническая рана), P(гипертрофический рубец), E[время до закрытия] | ✖ |
| Therapy comparison | P(good\|no therapy) vs P(good\|PRP) vs P(good\|PEMF) vs P(good\|PRP+PEMF) | ✖ |
| Decision rule | P(good) > threshold → стандартное лечение; иначе → optimal control protocol | ✖ |

### 3.6.2 Виртуальные клинические испытания

```
θ_patient ~ p_population(θ) = N(μ_pop, Σ_pop)
ATE = E[Y(therapy)] - E[Y(control)]
N_required = 2·(z_α + z_β)²·σ²/δ²
```

| Компонент | Описание | Статус |
|-----------|----------|--------|
| Population sampling | θ ~ N(μ_pop, Σ_pop) из aggregate clinical data | ✖ |
| Virtual patient | Sample θ → calibrate (optional) → simulate → evaluate | ✖ |
| MLMC variance reduction | Фаза 3.2.1 для быстрых ансамблей | ✖ |
| Statistical analysis | t-test, Mann-Whitney U, Kaplan-Meier | ✖ |
| Power analysis | Sample size для planned real trial | ✖ |
| CDISC export | Regulatory-compatible format | ✖ |

### 3.6.3 Topological Data Analysis

```
H₀ = connected components (cell clusters)
H₁ = loops (vascular loops, ring structures)
d_B(D₁, D₂) = inf_η sup_p ||p - η(p)||_∞    # Bottleneck distance
```

| Компонент | Описание | Статус |
|-----------|----------|--------|
| Rips complex | ABM позиции → filtration → persistence diagram | ✖ |
| H₀ tracking | Cluster count vs time (↓ = wound closing) | ✖ |
| H₁ tracking | Vascular network topology | ✖ |
| Persistence landscapes | Feature vectors для ML классификации исходов | ✖ |

### Файлы для создания

| Файл | Описание | Классы | Статус |
|------|----------|--------|--------|
| `src/analysis/decision_support.py` | CDS | `RiskCalculator`, `TherapyComparator`, `DecisionEngine` | ✖ |
| `src/analysis/virtual_trials.py` | In-silico trials | `VirtualPopulation`, `TrialSimulator`, `TrialAnalyzer`, `CDISCExporter` | ✖ |
| `src/analysis/tda.py` | TDA | `PersistenceComputer`, `TopologicalTracker`, `BottleneckDistance` | ✖ |
| `tests/unit/analysis/test_decision_support.py` | | ~20 тестов | ✖ |
| `tests/unit/analysis/test_virtual_trials.py` | | ~30 тестов | ✖ |
| `tests/unit/analysis/test_tda.py` | | ~15 тестов | ✖ |

### Ключевые ссылки

| Ссылка | Применение |
|--------|-----------|
| Vodovotz et al., PLoS Comp Bio 2008; 4(4):e1000014 | Translational systems biology |
| Pappalardo et al., Trends Biotechnol 2019; 37(8):808 | In-silico trials |
| Viceconti et al., Nature BME 2021; 5:450 | In-silico trials framework |
| FDA Guidance 2016 | Computational modeling in devices |
| Topaz et al., PLoS ONE 2015; 10(5):e0126383 | TDA + biological aggregation |
| Carlsson, Bull AMS 2009; 46(2):255 | TDA теория |

---

## Мультимасштабная интеграция: Strang Splitting и Equation-Free ✖ НЕ НАЧАТО (0%)

> **Цель:** Модификация существующего integration.py для строгой сходимости O(τ²) и полного Equation-Free framework.
> **Приоритет:** ВЫСОКИЙ
> **Зависимости:** Фаза 2.8 (ABM), Фаза 3.1 (якобиан)

### Strang Splitting

Текущий: Lie-Trotter SDE(τ) → ABM(τ) — порядок O(τ).

Upgrade: **Strang** SDE(τ/2) → ABM(τ) → SDE(τ/2) — порядок O(τ²).

| Компонент | Описание | Статус |
|-----------|----------|--------|
| Strang splitting | Модификация `SDEABMIntegrator._step()` | ✖ |
| Richardson extrapolation | Проверка порядка: τ, τ/2, τ/4 | ✖ |
| SDE-ABM consistency | 1/√N_agents scaling Wasserstein distance (100, 1000, 10000) | ✖ |
| Lipschitz constant | L = max_x \|\|J(x)\|\|_2 по физиологической области | ✖ |
| Adaptive sync interval | A-posteriori error estimator → динамический τ | ✖ |

### Файлы для создания

| Файл | Описание | Статус |
|------|----------|--------|
| `src/core/convergence_analysis.py` | `StrangSplitter`, `RichardsonExtrapolator`, `ConsistencyVerifier`, `AdaptiveSyncController` | ✖ |
| Модификация `src/core/integration.py` | Strang splitting в _step(), полный lifting/restricting | ✖ |
| `tests/unit/core/test_convergence.py` | ~30 тестов: порядок, consistency, adaptive sync | ✖ |

---

## Новые зависимости для pyproject.toml

```toml
[project.optional-dependencies]
phd = [
    # UQ / Calibration
    "chaospy>=4.3.0",           # PCE (Фаза 3.2)
    "pyabc>=0.12.0",            # ABC calibration (Фаза 3.3)
    "arviz>=0.16.0",            # Model selection, WAIC (Фаза 3.3)

    # Machine Learning
    "torch>=2.0.0",             # PINNs, Neural SDE, GNN (Фаза 3.5)
    "torchsde>=0.2.5",          # Neural SDE (Фаза 3.5.2)
    "torch_geometric>=2.0",     # GNN (Фаза 3.5.4)
    "gpytorch>=1.10",           # GP emulators (Фаза 3.5.3)

    # TDA
    "giotto-tda>=0.6.0",        # Topological Data Analysis (Фаза 3.6.3)
    "ripser>=0.6.0",            # TDA alternative (Фаза 3.6.3)
]
```

---

## Порядок реализации

### Milestone 3: «Научная валидация» (недели 1-8)

| # | Задача | Зависимости | Приоритет | Статус |
|---|--------|-------------|-----------|--------|
| 1 | Фаза 3.0 — Валидация на литературных данных | Фаза 2.5 | КРИТИЧЕСКИЙ | ✖ |
| 2 | Фаза 3.1.2 — Якобиан + Lyapunov stability | Фаза 2.5 | КРИТИЧЕСКИЙ | ✖ |
| 3 | Фаза 3.1.3 — Fokker-Planck + Kramers | #2 | КРИТИЧЕСКИЙ | ✖ |
| 4 | Фаза 3.2.1 — Multilevel Monte Carlo | Фаза 2.7 | КРИТИЧЕСКИЙ | ✖ |
| 5 | Фаза 3.3.1 — Оптимальное управление | #2 | КРИТИЧЕСКИЙ | ✖ |

### Milestone 4: «Калибровка и суррогаты» (недели 9-16)

| # | Задача | Зависимости | Приоритет | Статус |
|---|--------|-------------|-----------|--------|
| 6 | Фаза 3.3.2 — ABC калибровка | #4 (MLMC) | КРИТИЧЕСКИЙ | ✖ |
| 7 | Фаза 3.2.2 — PCE surrogate | — | ВЫСОКИЙ | ✖ |
| 8 | Фаза 3.5.1 — PINNs | — | ВЫСОКИЙ | ✖ |
| 9 | Фаза 3.4.1 — Intracellular (NF-κB, Smad, YAP) | — | ВЫСОКИЙ | ✖ |
| 10 | Фаза 3.5.3 — GP emulators | #7 (training data) | ВЫСОКИЙ | ✖ |
| 11 | Фаза 3.3.3 — Model selection | #6 (posteriors) | ВЫСОКИЙ | ✖ |

### Milestone 5: «Расширения» (недели 17-24)

| # | Задача | Зависимости | Приоритет | Статус |
|---|--------|-------------|-----------|--------|
| 12 | Фаза 3.1.4 — Large deviations | #3 (fixed points) | ВЫСОКИЙ | ✖ |
| 13 | Strang splitting + convergence | #2 (Jacobian) | ВЫСОКИЙ | ✖ |
| 14 | Фаза 3.4.2 — Метаболизм (Glucose, Lactate) | — | СРЕДНИЙ | ✖ |
| 15 | Фаза 3.5.2 — Neural SDE | #8 (PyTorch) | СРЕДНИЙ | ✖ |
| 16 | Фаза 3.2.4 — Tau-leaping для ABM | — | СРЕДНИЙ | ✖ |
| 17 | Фаза 3.6.2 — Virtual trials | #4, #6 | ВЫСОКИЙ | ✖ |
| 18 | Фаза 3.0.3 — Benchmarking vs Flegg/Xue | #1 | ВЫСОКИЙ | ✖ |
| 19 | Фаза 3.4.3 — Trained immunity | #9 | СРЕДНИЙ | ✖ |

### Milestone 6: «Продвинутые расширения» (недели 25-30)

| # | Задача | Зависимости | Приоритет | Статус |
|---|--------|-------------|-----------|--------|
| 20 | Фаза 3.5.4 — GNN surrogate | #8 | СРЕДНИЙ | ✖ |
| 21 | Фаза 3.4.4 — Микробиом | — | СРЕДНИЙ | ✖ |
| 22 | Фаза 3.2.3 — Quasi-Monte Carlo | #4 | СРЕДНИЙ | ✖ |
| 23 | Фаза 3.6.3 — TDA | — | НИЗКИЙ | ✖ |
| 24 | Фаза 3.6.1 — Decision support | #6, #10 | ВЫСОКИЙ | ✖ |

---

## Стратегия публикаций

### Paper 1: PLOS Computational Biology

**Заголовок:** "Stochastic Bifurcation Analysis Reveals TGF-β Bistability as a Predictor of Fibrotic Wound Outcomes"

| Компонент | Фаза | Ключевые результаты |
|-----------|------|-------------------|
| Бифуркационный анализ | 3.1.2-3.1.3 | Нуллклины, Fokker-Planck, Kramers rate |
| Large deviations | 3.1.4 | Инстантоны, quasi-potential |
| Lyapunov analysis | 3.1.2 | Спектр собственных значений, timescales |
| Валидация | 3.0 | R² > 0.85 на литературных данных |

### Paper 2: Bioinformatics

**Заголовок:** "Multilevel Monte Carlo and Polynomial Chaos Expansion for Efficient UQ in Multiscale Wound Healing Models"

| Компонент | Фаза | Ключевые результаты |
|-----------|------|-------------------|
| MLMC | 3.2.1 | 10-100x speedup vs standard MC |
| PCE surrogate | 3.2.2 | ms-оценки с Sobol indices |
| ABC calibration | 3.3.2 | Patient-specific posteriors |
| Model selection | 3.3.3 | WAIC: 20-var >> 2-var |

### Paper 3: Journal of Theoretical Biology

**Заголовок:** "Optimal Therapy Scheduling for PRP/PEMF Wound Healing via Pontryagin Maximum Principle and Patient-Specific Calibration"

| Компонент | Фаза | Ключевые результаты |
|-----------|------|-------------------|
| Optimal control | 3.3.1 | Оптимальные u*(t) для PRP/PEMF |
| ABC calibration | 3.3.2 | Patient-specific параметры |
| Decision support | 3.6.1 | Risk scores, therapy recommendation |
| Virtual trials | 3.6.2 | In-silico trial design |

### Paper 4: npj Systems Biology and Applications

**Заголовок:** "Physics-Informed Neural Networks for Parameter Discovery in Multiscale Tissue Regeneration Models"

| Компонент | Фаза | Ключевые результаты |
|-----------|------|-------------------|
| PINNs | 3.5.1 | Обнаружение k_switch, K_activ из данных |
| Neural SDE | 3.5.2 | Hybrid drift: known + learned correction |
| GP emulators | 3.5.3 | Real-time digital twin operation |
| Intracellular | 3.4.1 | NF-κB, Smad2/3 модули |

---

## Таблица параметров новых модулей

### Параметры NF-κB модуля

| Параметр | Значение | Единицы | Описание | Источник |
|----------|----------|---------|----------|----------|
| k_import | 0.5 | ч⁻¹ | NF-κB ядерный импорт | Hoffmann 2002 |
| k_export | 0.3 | ч⁻¹ | NF-κB ядерный экспорт | Hoffmann 2002 |
| k_bind | 1.0 | ч⁻¹·нМ⁻¹ | NF-κB/IκBα связывание | Hoffmann 2002 |
| s_IκBα | 0.5 | нМ/ч | Транскрипция IκBα (NF-κB-зависимая) | Hoffmann 2002 |
| δ_IκBα | 0.1 | ч⁻¹ | Деградация IκBα | Hoffmann 2002 |
| k_IKK | 0.5 | ч⁻¹·нМ⁻¹ | IKK-опосредованная деградация IκBα | Hoffmann 2002 |
| α_NFκB | 2.0 | безразм. | Усиление TNF-α продукции | Hoffmann 2002 |

### Параметры метаболизма

| Параметр | Значение | Единицы | Описание | Источник |
|----------|----------|---------|----------|----------|
| D_glc | 0.5 | мм²/ч | Коэффициент диффузии глюкозы | Casciari 1992 |
| Glc_blood | 5.0 | мМ | Глюкоза крови (норма) | Стандарт |
| k_glyc | 0.1 | ч⁻¹ | Скорость гликолиза | O'Neill 2016 |
| k_oxphos | 0.02 | ч⁻¹ | Скорость окислительного фосфорилирования | Kelly 2015 |
| K_ATP | 1.0 | мМ | Half-max ATP для пролиферации | Оценка |

### Параметры микробиома

| Параметр | Значение | Единицы | Описание | Источник |
|----------|----------|---------|----------|----------|
| r_B | 0.5 | ч⁻¹ | Удвоение бактерий | Стандарт |
| K_B | 10⁸ | КОЕ/мл | Carrying capacity | Kalan 2019 |
| k_kill_Ne | 0.05 | ч⁻¹ | Фагоцитоз нейтрофилами | Kolaczkowska 2013 |
| k_kill_M1 | 0.03 | ч⁻¹ | Фагоцитоз M1 макрофагами | Murray 2017 |
| s_PAMP | 0.01 | нг/(мл·КОЕ·ч) | Секреция PAMPs | Оценка |
| α_PAMP | 3.0 | безразм. | Усиление TNF-α от PAMPs | Оценка |

### Параметры оптимального управления

| Параметр | Значение | Единицы | Описание |
|----------|----------|---------|----------|
| w₁ | 1.0 | безразм. | Вес штрафа за воспаление |
| w₂ | 0.5 | безразм. | Вес штрафа за незавершённый коллаген |
| w₃ | 0.1 | безразм. | Вес стоимости терапии |
| w₄ | 2.0 | безразм. | Вес терминального штрафа (коллаген) |
| w₅ | 1.0 | безразм. | Вес терминального штрафа (Mf) |

---

## Критические файлы

### Существующие файлы для модификации

| Файл | Модификация | Фаза |
|------|------------|------|
| `src/core/extended_sde.py` | +5-7 переменных (Glucose, Lactate, Epi_M, B, PAMPs); PyTorch-версия drift; Jacobian accessor | 3.1, 3.4, 3.5 |
| `src/core/integration.py` | Strang splitting; полный lifting/restricting для Equation-Free | 3.1, Strang |
| `src/core/parameters.py` | +30 параметров (intracellular, metabolism, microbiome, OC weights) | 3.4 |
| `src/core/therapy_models.py` | AntibioticModel; управляемость u(t) для optimal control | 3.3, 3.4 |
| `src/core/sde_numerics.py` | Параметр dW_sequence для MLMC; QMC wrapper | 3.2 |
| `pyproject.toml` | Новая группа зависимостей [phd] | Все |

### Новые файлы

| Файл | LOC (ожид.) | Фаза |
|------|-------------|------|
| `src/core/bifurcation.py` | ~800 | 3.1 |
| `src/core/large_deviations.py` | ~500 | 3.1 |
| `src/core/mlmc.py` | ~600 | 3.2 |
| `src/core/qmc.py` | ~300 | 3.2 |
| `src/core/intracellular.py` | ~600 | 3.4 |
| `src/core/abm_tau_leaping.py` | ~400 | 3.2 |
| `src/core/convergence_analysis.py` | ~400 | Strang |
| `src/analysis/pce.py` | ~500 | 3.2 |
| `src/analysis/optimal_control.py` | ~700 | 3.3 |
| `src/analysis/abc_calibration.py` | ~500 | 3.3 |
| `src/analysis/model_selection.py` | ~300 | 3.3 |
| `src/analysis/validation.py` | ~400 | 3.0 |
| `src/analysis/benchmarking.py` | ~500 | 3.0 |
| `src/analysis/decision_support.py` | ~400 | 3.6 |
| `src/analysis/virtual_trials.py` | ~500 | 3.6 |
| `src/analysis/tda.py` | ~300 | 3.6 |
| `src/ml/pinn.py` | ~600 | 3.5 |
| `src/ml/neural_sde.py` | ~500 | 3.5 |
| `src/ml/gp_emulator.py` | ~400 | 3.5 |
| `src/ml/gnn_surrogate.py` | ~500 | 3.5 |
| **Итого** | **~9,700** | |

### Новые тесты

| Директория | Файлов | Тестов (ожид.) |
|-----------|--------|---------------|
| `tests/unit/core/` | 7 | ~265 |
| `tests/unit/analysis/` | 7 | ~195 |
| `tests/unit/ml/` | 4 | ~115 |
| `tests/validation/` | 1 | ~15 |
| **Итого** | **19** | **~590** |

---

## Сводка по прогрессу

| Фаза | Название | Статус | Прогресс |
|------|----------|--------|----------|
| 3.0 | Валидация на литературных данных | ✖ Не начато | 0% |
| 3.1 | Стохастический бифуркационный анализ | ✖ Не начато | 0% |
| 3.2 | Продвинутые численные методы | ✖ Не начато | 0% |
| 3.3 | Оптимальное управление и калибровка | ✖ Не начато | 0% |
| 3.4 | Биологическая глубина | ✖ Не начато | 0% |
| 3.5 | Machine Learning интеграция | ✖ Не начато | 0% |
| 3.6 | Клиническая трансляция | ✖ Не начато | 0% |

### Общий прогресс: 0% (от научных расширений PhD+ уровня)

### Ожидаемые итоги

| Метрика | Значение |
|---------|----------|
| Новых файлов кода | 20 (~9,700 LOC) |
| Новых тестовых файлов | 19 (~590 тестов) |
| Публикаций | 4 статьи в top-журналах |
| Timeline | 30 недель (при полной загрузке) |

---

## Верификация

```bash
# Все тесты (существующие + новые)
pytest -v --cov=src --cov-report=term-missing --cov-report=html

# Только научные расширения
pytest tests/unit/core/test_bifurcation.py tests/unit/core/test_large_deviations.py -v
pytest tests/unit/core/test_mlmc.py tests/unit/core/test_qmc.py -v
pytest tests/unit/analysis/ -v
pytest tests/unit/ml/ -v
pytest tests/validation/ -v -m validation

# Бифуркационный анализ (standalone)
python -m src.core.bifurcation --parameter s_TGF_Mf --range 0.001,0.05 --output results/bifurcation/

# MLMC convergence test
python -m src.core.mlmc --levels 5 --epsilon 0.01 --output results/mlmc/

# Оптимальное управление (standalone)
python -m src.analysis.optimal_control --config config/optimal_control.yaml --output results/optimal/

# ABC калибровка
python -m src.analysis.abc_calibration --data data/validation/ --n_particles 1000 --output results/abc/

# PCE surrogate
python -m src.analysis.pce --n_parameters 10 --order 3 --output results/pce/

# PINNs training
python -m src.ml.pinn --data data/validation/ --epochs 10000 --output results/pinn/

# Валидация на литературных данных
python -m src.analysis.validation --model extended --output results/validation/

# Бенчмаркинг
python -m src.analysis.benchmarking --models all --output results/benchmarking/

# Линтинг
ruff check src/
mypy src/
```

---

*Документ создан: 11 марта 2026*
*Версия: 5.0 (PhD+ научные расширения)*
*Основан на: RegenTwin_Update_Implemention_Plan.md v4.4, RegenTwin_Mathematical_Framework.md*
*Предшествует: Фазы 1–2.8 полностью реализованы (1400+ тестов, 8300+ LOC ядра)*
