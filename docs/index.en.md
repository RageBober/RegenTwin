# RegenTwin

**RegenTwin** is a multiscale tissue regeneration simulation platform
with support for PRP (platelet-rich plasma) and PEMF (pulsed electromagnetic
field) therapies.

## What it is

A digital twin of the wound-healing process combining:

- A **20-variable SDE model** of cell–cytokine dynamics (Phase 2 Mathematical Framework)
- An **agent-based model** of spatial cellular behaviour
- **Monte Carlo** ensemble runs (parallelised)
- **Sobol sensitivity analysis**
- **Parameter estimation** via PyMC / emcee
- Import of real flow-cytometry data (`.fcs`) as initial conditions

## Stack

| Layer | Technologies |
|---|---|
| Math core | NumPy, SciPy, PyMC, emcee, SALib |
| Backend  | FastAPI, SQLAlchemy 2.0, Alembic, Celery (optional), DuckDB |
| Frontend | React 19 + TypeScript, Tauri 2.0, Vite, Plotly, Three.js |
| DevOps   | uv, ruff, mypy, pytest, pytest-benchmark, MkDocs Material, GitHub Actions |

## Next steps

- **[Installation](getting-started/installation.en.md)** — local setup
- **[First simulation](getting-started/first-simulation.en.md)** — run SDE via CLI / API
- **[Architecture](architecture/overview.md)** — layers and data flow
- **[API Reference](api-reference/index.md)** — auto-generated module docs
- **[Benchmarks](benchmarks/index.md)** — performance across machines
