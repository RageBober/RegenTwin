# API Reference

Автогенерация из docstrings через
[mkdocstrings](https://mkdocstrings.github.io/python/).

Полные ссылки:

- **[Extended SDE](extended-sde.md)** — `src.core.extended_sde`
- **[ABM](abm.md)** — `src.core.abm_model`, `src.core.abm_spatial`
- **[Monte Carlo](monte-carlo.md)** — `src.core.monte_carlo`

## REST API

REST endpoints документированы в OpenAPI (Swagger UI):

```bash
uv run uvicorn src.api.main:app --port 8000
# открыть http://localhost:8000/docs
```

Краткая сводка — в [README.md на GitHub](https://github.com/RageBober/RegenTwin#rest-api).
