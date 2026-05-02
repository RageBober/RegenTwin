# Benchmarks

Этот раздел автоматически собирается из `output/benchmarks/report.md`
через [`scripts/sync_benchmarks_to_docs.py`](https://github.com/RageBober/RegenTwin/blob/master/scripts/sync_benchmarks_to_docs.py).

## Как воспроизвести

### Базовые бенчмарки (pytest-benchmark)

```bash
uv run python scripts/benchmark.py --label "my-machine"
```

### С профилированием (py-spy + scalene)

```bash
uv run python scripts/benchmark.py --label "my-machine" --profile
```

### Сравнить две машины

1. На каждой машине: `uv run python scripts/benchmark.py --label "<имя>"`
2. Скопируйте JSON в общий `output/benchmarks/`
3. Локально: `uv run python scripts/generate_benchmark_report.py`
4. Опубликовать: `uv run python scripts/sync_benchmarks_to_docs.py`

## Что измеряется

| Группа | Что | Параметры |
|---|---|---|
| `sde-small` | Extended SDE | t_max=24h, ~2400 шагов |
| `sde-large` | Extended SDE | t_max=72h, ~7200 шагов |
| `abm-small` | ABM | 100 max agents, 24h |
| `abm-medium` | ABM | 500 max agents, 24h |
| `mc-serial` | Monte Carlo | 4 траектории, 1 jobs |
| `mc-parallel` | Monte Carlo | 4 траектории, n_jobs=cpu-1 |
| `sensitivity-sobol` | Sobol (Ishigami) | N=64, k=4 |

!!! info "Когда отчёт ещё не сгенерирован"
    Если `output/benchmarks/report.md` отсутствует — сначала запустите
    `scripts/benchmark.py` хотя бы один раз. После этого `sync_benchmarks_to_docs.py`
    скопирует отчёт сюда.
