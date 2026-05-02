# Installation

## Requirements

- Python `3.11+`
- [uv](https://docs.astral.sh/uv/) `>= 0.5` (PEP 735 groups)
- Node.js `>= 18`
- Rust toolchain (for Tauri desktop build)

## Quick start

```bash
git clone https://github.com/RageBober/RegenTwin.git
cd RegenTwin
uv sync
uv run alembic upgrade head
uv run pre-commit install
cd ui && npm ci
```

## Run

```bash
# Backend API
uv run uvicorn src.api.main:app --host 127.0.0.1 --port 8000

# Frontend (dev)
cd ui && npm run dev
```

## Verify

```bash
uv run python scripts/diagnose.py
curl http://localhost:8000/api/v1/health/live
```
