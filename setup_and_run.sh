#!/bin/bash
set -e

echo "RegenTwin - Setup and Run"
echo "================================"

if ! command -v uv >/dev/null 2>&1; then
  echo "UV not found. Install UV first: https://docs.astral.sh/uv/"
  exit 1
fi

if ! command -v node >/dev/null 2>&1; then
  echo "Node.js not found. Install Node.js 18+"
  exit 1
fi

echo "Syncing Python environment..."
uv sync --extra dev

if [ ! -d "ui/node_modules" ]; then
  echo "Installing frontend dependencies..."
  (cd ui && npm install)
fi

echo "Releasing port 8000 if needed..."
uv run python scripts/kill_port.py 8000 || true

echo "Starting backend..."
uv run uvicorn src.api.main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

cleanup() {
  kill $BACKEND_PID 2>/dev/null || true
  wait $BACKEND_PID 2>/dev/null || true
}
trap cleanup EXIT INT TERM

sleep 2
cd ui
npm run dev
