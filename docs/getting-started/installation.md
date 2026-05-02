# Установка

## Требования

- Python `3.11+`
- [uv](https://docs.astral.sh/uv/) `>= 0.5` (PEP 735 dependency groups)
- Node.js `>= 18`
- Rust toolchain (для Tauri desktop сборки)
- (Опционально) Docker `>= 24` — для backend-only контейнера

## Базовая установка

```bash
git clone https://github.com/RageBober/RegenTwin.git
cd RegenTwin

# 1. Python окружение через uv
uv sync                              # ставит main + dev-group

# 2. БД миграции (создаёт data/regentwin.duckdb)
uv run alembic upgrade head

# 3. Pre-commit хуки
uv run pre-commit install

# 4. Frontend
cd ui
npm ci
```

## Запуск

### Раздельный режим (backend + Vite)

```bash
# Терминал 1 — API
uv run uvicorn src.api.main:app --host 127.0.0.1 --port 8000

# Терминал 2 — UI
cd ui
npm run dev          # http://localhost:5173
```

### Tauri desktop

```bash
cd ui
npm run tauri:dev
```

## Документация (опционально)

```bash
uv sync --group docs
uv run mkdocs serve   # http://127.0.0.1:8000
```

## Проверки

```bash
uv run python scripts/diagnose.py    # smoke-тест всего стека
uv run pytest -q                      # юнит/интеграционные тесты
curl http://localhost:8000/api/v1/health/live
```
