# Математический фреймворк

Полная математическая постановка модели описана в репозиторном архиве:
[Doks/RegenTwin_Mathematical_Framework.md](https://github.com/RageBober/RegenTwin/blob/master/Doks/RegenTwin_Mathematical_Framework.md).

Ниже — конспект.

## Уровень 1: SDE 20-var

Система стохастических ОДУ Эйлера-Маруямы:

$$
dX_i = \mu_i(X) \, dt + \sigma_i(X) \, dW_i, \quad i = 1, \ldots, 20
$$

Переменные сгруппированы:

| Группа | Переменные | Кол-во |
|---|---|---|
| Клетки | $P, N_e, M_1, M_2, F, M_f, E, S$ | 8 |
| Цитокины | $C_{TNF}, C_{IL10}, C_{PDGF}, C_{VEGF}, C_{TGF\beta}, C_{MCP1}, C_{IL8}$ | 7 |
| ECM | $\rho_\text{collagen}, C_{MMP}, \rho_\text{fibrin}$ | 3 |
| Прочие | $D$ (damage signal), $O_2$ | 2 |

Ключевые механизмы:

- **M1→M2 переключение макрофагов** — фазовый переход, управляемый IL-10/TNF-α
- **TGF-β ↔ Mf бистабильность** — заживление vs фиброз
- **Гипоксия → ангиогенез** — VEGF от низкого $O_2$
- **Эффероцитоз** как драйвер разрешения воспаления

## Уровень 2: Agent-Based Model

Агенты:

- `PlateletAgent` — тромбоциты, дегрануляция → PDGF/TGF-β/VEGF
- Макрофаги M1/M2 — фагоцитоз, секреция цитокинов
- `Fibroblast`, `Myofibroblast` — продукция коллагена
- `EndothelialCell` — ангиогенез
- `StemCell` (CD34+) — пролиферация и дифференцировка

Пространственные процессы:

- Хемотаксис (`ChemotaxisEngine`)
- Контактное ингибирование (`ContactInhibitionEngine`)
- Эффероцитоз (`EfferocytosisEngine`)
- Механотрансдукция (`MechanotransductionEngine`)

## Уровень 3: Monte Carlo

Ансамблевые прогоны для оценки неопределённости. Параметры берутся из `ParameterSet`
с шумом ±10% от номинала; агрегаты — квантили `[0.05, 0.25, 0.5, 0.75, 0.95]`.
Параллелизация через `concurrent.futures.ProcessPoolExecutor`.

## Уровень 4: Sensitivity (Sobol)

Глобальный Sobol-анализ через SALib:

$$
S_i = \frac{V_i}{V}, \quad S_T^i = \frac{E[V(Y|X_{-i})]}{V}
$$

Для оценки $S_i$ и $S_T^i$ нужно $N \cdot (2k + 2)$ симуляций при $k$ параметрах.
