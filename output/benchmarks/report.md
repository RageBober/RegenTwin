# RegenTwin — Benchmark Report

_Сгенерировано из 1 snapshot(ов) в `output/benchmarks/`._

## System Configurations

| Метка | CPU | Cores | RAM | OS | Python |
|-------|-----|-------|-----|----|--------|
| **codex-analysis** | Intel(R) Core(TM) i5-8600 CPU @ 3.10GHz | 6/6 | 15.9 GB | Windows-10 | 3.11.14 |

## Benchmark Results

| Группа | codex-analysis |
|---|---|
| Extended SDE (24h, ~2400 шагов) | 0.1787s ± 0.0259 |
| Extended SDE (72h, ~7200 шагов) | — |
| ABM (100 агентов, 24h) | 1.0099s ± 0.0487 |
| ABM (500 агентов, 24h) | — |
| Monte Carlo serial (4 траектории) | — |
| Monte Carlo parallel (n_jobs=cpu-1) | — |
| Sobol sensitivity (Ishigami, N=64) | 0.0290s ± 0.0207 |

## Графики

![sde-small](figures/sde-small.png)

![abm-small](figures/abm-small.png)

![sensitivity-sobol](figures/sensitivity-sobol.png)
