"""E2E для WebSocket-стрима прогресса симуляции."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
import websockets

from tests.e2e.conftest import LiveServer


async def _collect_ws_events(url: str, max_wait: float = 60.0) -> list[dict]:
    events: list[dict] = []
    async with websockets.connect(url, max_size=2**20) as ws:
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=max_wait)
                try:
                    events.append(json.loads(raw))
                except json.JSONDecodeError:
                    continue
                if events[-1].get("event") in {"complete", "failed", "cancelled", "not_found"}:
                    break
        except TimeoutError:
            pass
    return events


@pytest.mark.e2e
def test_websocket_streams_progress_until_complete(
    http_client: httpx.Client, live_api_server: LiveServer
) -> None:
    payload = {
        "mode": "extended",
        "t_max_hours": 12.0,
        "dt": 0.5,
        "random_seed": 42,
    }
    response = http_client.post("/api/v1/simulate", json=payload)
    assert response.status_code == 200
    simulation_id = response.json()["simulation_id"]

    events = asyncio.run(_collect_ws_events(live_api_server.ws_url(simulation_id), max_wait=60.0))
    assert events, "expected at least one WebSocket event"

    progress_events = [e for e in events if e.get("event") == "progress"]
    terminal_events = [e for e in events if e.get("event") in {"complete", "failed", "cancelled"}]
    assert progress_events, f"no progress events in {events!r}"
    assert terminal_events, f"no terminal event in {events!r}"
    assert terminal_events[-1]["event"] == "complete"

    percents = [p["data"].get("percent", 0.0) for p in progress_events]
    assert percents[-1] >= 0.0
    assert max(percents) <= 100.0


@pytest.mark.e2e
def test_websocket_rejects_invalid_simulation_id(live_api_server: LiveServer) -> None:
    bad_url = live_api_server.ws_url("not-a-uuid")

    async def _run() -> int | None:
        try:
            async with websockets.connect(bad_url) as ws:
                await ws.recv()
            return None
        except websockets.exceptions.InvalidStatus as exc:
            return exc.response.status_code
        except websockets.exceptions.ConnectionClosed as exc:
            return exc.code

    code = asyncio.run(_run())
    assert code in {1008, 403, 400}, f"unexpected ws close/status: {code}"
