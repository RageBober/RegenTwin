# Troubleshooting

## DuckDB файл заблокирован

```
duckdb.IOException: Could not set lock on file
```

DuckDB поддерживает только одного writer'а на файл. Если запущены одновременно
API и Celery worker — оба пишут в `data/regentwin.duckdb` и конфликтуют.

**Решение:** в дипломном сценарии Celery выключен (`use_celery=False`),
все задачи идут через asyncio в том же процессе. Для прод-нагрузки потребуется
переход на PostgreSQL/Postgres-compatible.

## Порт 8000 занят

```bash
uv run python scripts/kill_port.py 8000
```

## PyMC падает на сборке wheel

Pytensor компилируется при первом импорте. Под Windows нужен MSVC build tools
(или `pip install pytensor` с pre-built wheels).

## Tauri build не находит Python

Tauri-launcher (`ui/src-tauri/src/main.rs`) ищет backend в порядке:
1. `uv run uvicorn`
2. project-local `.venv`
3. `python` в PATH

Для production `.msi` бандл `.venv` идёт через `bundle.resources` (Tauri 2).

## `uv sync` падает на `--frozen`

`uv.lock` устарел после изменения `pyproject.toml`. Запустите `uv lock` локально.
