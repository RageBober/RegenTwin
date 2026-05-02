# First simulation

Run an extended-SDE simulation for 24 hours via REST.

```bash
uv run uvicorn src.api.main:app --port 8000
```

```bash
curl -X POST http://localhost:8000/api/v1/simulate \
  -H "Content-Type: application/json" \
  -d '{"mode": "extended", "params": {"t_max": 24.0, "dt": 0.01}}'
```

Poll `/api/v1/simulate/<id>` until `status == "completed"`, then fetch
`/api/v1/results/<id>`.
