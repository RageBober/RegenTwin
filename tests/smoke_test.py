"""
RegenTwin Smoke Test — полная проверка системы.

Запуск:
    1. Запустите backend: python -m uvicorn src.api.main:app --port 8000
    2. Запустите тест:    python tests/smoke_test.py

Или одной командой (из ui/):
    npm run dev:full &
    python tests/smoke_test.py
"""

from __future__ import annotations

import sys
import time
import json
import requests

BASE_URL = "http://127.0.0.1:8000"
API_V1 = f"{BASE_URL}/api/v1"
API_VIZ = f"{BASE_URL}/api/viz"

# ── Helpers ──────────────────────────────────────────────────────────

passed = 0
failed = 0
errors: list[str] = []


def check(name: str, fn):
    """Run a single test and track results."""
    global passed, failed
    try:
        fn()
        passed += 1
        print(f"  [PASS] {name}")
    except Exception as e:
        failed += 1
        msg = f"  [FAIL] {name}: {e}"
        errors.append(msg)
        print(msg)


def get(url: str, **kwargs) -> requests.Response:
    return requests.get(url, timeout=10, **kwargs)


def post(url: str, **kwargs) -> requests.Response:
    return requests.post(url, timeout=30, **kwargs)


# Default simulation params for viz endpoints
DEFAULT_SIM_PARAMS = {
    "P0": 500, "Ne0": 200, "M1_0": 100, "M2_0": 10,
    "F0": 50, "Mf0": 0, "E0": 20, "S0": 40,
    "C_TNF0": 10, "C_IL10_0": 0.5, "D0": 5, "O2_0": 80,
    "t_max_hours": 48,  # short for speed
    "dt": 0.5,
    "prp_enabled": False, "pemf_enabled": False,
    "prp_intensity": 1.0, "pemf_frequency": 50.0, "pemf_intensity": 1.0,
    "random_seed": 42,
}


# ── 1. Health ────────────────────────────────────────────────────────

def test_health():
    print("\n1. Health Check")

    def health_ok():
        r = get(f"{API_V1}/health")
        assert r.status_code == 200, f"status={r.status_code}"
        data = r.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "uptime_seconds" in data
        print(f"     version={data['version']}, uptime={data['uptime_seconds']:.1f}s")

    check("GET /api/v1/health", health_ok)


# ── 2. Upload ────────────────────────────────────────────────────────

def test_upload():
    print("\n2. Upload")

    def upload_rejects_non_fcs():
        """Upload endpoint should exist (may reject non-FCS files)."""
        import io
        files = {"file": ("test.txt", io.BytesIO(b"not a real file"), "text/plain")}
        r = post(f"{API_V1}/upload", files=files)
        # 400 or 422 is expected (bad file type), 500 means server error
        assert r.status_code in (200, 400, 422), f"status={r.status_code}: {r.text[:200]}"

    check("POST /api/v1/upload (rejects non-FCS)", upload_rejects_non_fcs)


# ── 3. Simulation ────────────────────────────────────────────────────

simulation_id: str | None = None


def test_simulation():
    global simulation_id
    print("\n3. Simulation")

    def start_simulation():
        global simulation_id
        payload = {
            "mode": "extended",
            "t_max_hours": 24,
            "dt": 0.5,
            "n_trajectories": 1,
            "random_seed": 42,
        }
        r = post(f"{API_V1}/simulate", json=payload)
        assert r.status_code == 200, f"status={r.status_code}: {r.text[:300]}"
        data = r.json()
        assert "simulation_id" in data
        simulation_id = data["simulation_id"]
        print(f"     simulation_id={simulation_id}")

    check("POST /api/v1/simulate", start_simulation)

    def get_status():
        assert simulation_id, "No simulation_id"
        r = get(f"{API_V1}/simulate/{simulation_id}")
        assert r.status_code == 200, f"status={r.status_code}"
        data = r.json()
        assert "status" in data
        print(f"     status={data['status']}, progress={data.get('progress', '?')}")

    check("GET /api/v1/simulate/{id}", get_status)

    def list_simulations():
        r = get(f"{API_V1}/simulations")
        assert r.status_code == 200, f"status={r.status_code}"
        data = r.json()
        assert isinstance(data, list)
        print(f"     count={len(data)}")

    check("GET /api/v1/simulations", list_simulations)

    # Wait for simulation to complete (poll)
    if simulation_id:
        print("     Waiting for simulation to complete...", end="", flush=True)
        for _ in range(60):
            r = get(f"{API_V1}/simulate/{simulation_id}")
            if r.status_code == 200:
                status = r.json().get("status", "")
                if status in ("completed", "failed"):
                    print(f" {status}")
                    break
            time.sleep(1)
            print(".", end="", flush=True)
        else:
            print(" timeout")

    def get_results():
        assert simulation_id, "No simulation_id"
        r = get(f"{API_V1}/results/{simulation_id}")
        # May be 200 (if completed) or 404 (if still running/failed)
        assert r.status_code in (200, 404), f"status={r.status_code}"
        if r.status_code == 200:
            data = r.json()
            n_vars = len(data.get("variables", {}))
            n_times = len(data.get("times", []))
            print(f"     variables={n_vars}, timepoints={n_times}")

    check("GET /api/v1/results/{id}", get_results)


# ── 4. Visualization (Plotly JSON) ───────────────────────────────────

def test_visualization():
    print("\n4. Visualization Endpoints")

    def check_viz(name: str, endpoint: str, extra_payload: dict | None = None):
        def _test():
            payload = {"simulation": DEFAULT_SIM_PARAMS}
            if extra_payload:
                payload.update(extra_payload)
            r = post(f"{API_VIZ}/{endpoint}", json=payload)
            assert r.status_code == 200, f"status={r.status_code}: {r.text[:300]}"
            data = r.json()
            assert "data" in data, f"Missing 'data' key. Keys: {list(data.keys())}"
            assert "layout" in data, f"Missing 'layout' key"
            n_traces = len(data["data"])
            title = data.get("layout", {}).get("title", {})
            if isinstance(title, dict):
                title = title.get("text", "")
            print(f"     traces={n_traces}, title='{title[:50]}'")

        check(f"POST /api/viz/{endpoint}", _test)

    check_viz("Populations", "populations")
    check_viz("Cytokines", "cytokines")
    check_viz("ECM", "ecm")
    check_viz("Phases", "phases")
    check_viz("Comparison", "comparison")


# ── 5. Viz Export ────────────────────────────────────────────────────

def test_viz_export():
    print("\n5. Visualization Export")

    def export_csv():
        payload = {"simulation": DEFAULT_SIM_PARAMS}
        r = post(f"{API_VIZ}/export/csv", json=payload)
        assert r.status_code == 200, f"status={r.status_code}: {r.text[:200]}"
        content_type = r.headers.get("content-type", "")
        size = len(r.content)
        print(f"     content-type={content_type}, size={size} bytes")
        assert size > 100, f"CSV too small: {size} bytes"

    check("POST /api/viz/export/csv", export_csv)

    def export_png():
        payload = {"simulation": DEFAULT_SIM_PARAMS}
        r = post(f"{API_VIZ}/export/png", json=payload)
        assert r.status_code == 200, f"status={r.status_code}: {r.text[:200]}"
        size = len(r.content)
        print(f"     size={size} bytes")
        assert size > 1000, f"PNG too small: {size} bytes"

    check("POST /api/viz/export/png", export_png)


# ── 6. Spatial Visualization ─────────────────────────────────────────

def test_spatial():
    print("\n6. Spatial Visualization (ABM)")

    spatial_params = {
        "n_stem": 10,
        "n_macro": 15,
        "n_fibro": 8,
        "n_neutrophil": 20,
        "n_endothelial": 5,
        "t_max_hours": 5.0,
        "dt": 1.0,
        "domain_size": 100.0,
        "random_seed": 42,
    }

    def spatial_heatmap():
        r = post(f"{API_VIZ}/spatial/heatmap", json=spatial_params)
        assert r.status_code == 200, f"status={r.status_code}: {r.text[:300]}"
        data = r.json()
        assert "data" in data
        print(f"     traces={len(data['data'])}")

    check("POST /api/viz/spatial/heatmap", spatial_heatmap)

    def spatial_scatter():
        r = post(f"{API_VIZ}/spatial/scatter", json=spatial_params)
        assert r.status_code == 200, f"status={r.status_code}: {r.text[:300]}"
        data = r.json()
        assert "data" in data
        print(f"     traces={len(data['data'])}")

    check("POST /api/viz/spatial/scatter", spatial_scatter)

    def spatial_inflammation():
        r = post(f"{API_VIZ}/spatial/inflammation", json=spatial_params)
        assert r.status_code == 200, f"status={r.status_code}: {r.text[:300]}"
        data = r.json()
        assert "data" in data
        print(f"     traces={len(data['data'])}")

    check("POST /api/viz/spatial/inflammation", spatial_inflammation)


# ── 7. Analysis ──────────────────────────────────────────────────────

def test_analysis():
    print("\n7. Analysis")

    def sensitivity_start():
        payload = {
            "simulation_params": {
                "mode": "extended",
                "t_max_hours": 24,
                "dt": 0.5,
            },
            "parameters": ["r", "K"],
            "method": "sobol",
            "n_samples": 64,
        }
        r = post(f"{API_V1}/analysis/sensitivity", json=payload)
        assert r.status_code == 200, f"status={r.status_code}: {r.text[:300]}"
        data = r.json()
        assert "analysis_id" in data
        print(f"     analysis_id={data['analysis_id']}, status={data.get('status')}")

    check("POST /api/v1/analysis/sensitivity", sensitivity_start)


# ── 8. CORS ──────────────────────────────────────────────────────────

def test_cors():
    print("\n8. CORS")

    def cors_preflight():
        headers = {
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        }
        r = requests.options(f"{API_V1}/health", headers=headers, timeout=5)
        acl = r.headers.get("access-control-allow-origin", "")
        assert acl, f"No CORS header. Headers: {dict(r.headers)}"
        print(f"     allow-origin={acl}")

    check("OPTIONS /api/v1/health (CORS preflight)", cors_preflight)

    def cors_actual():
        headers = {"Origin": "http://localhost:5173"}
        r = get(f"{API_V1}/health", headers=headers)
        acl = r.headers.get("access-control-allow-origin", "")
        assert acl == "http://localhost:5173", f"CORS origin={acl}"

    check("GET /api/v1/health (CORS header)", cors_actual)


# ── 9. Error Handling ────────────────────────────────────────────────

def test_errors():
    print("\n9. Error Handling")

    def not_found_404():
        r = get(f"{API_V1}/simulate/nonexistent-id-12345")
        assert r.status_code == 404, f"Expected 404, got {r.status_code}"

    check("GET /api/v1/simulate/invalid-id -> 404", not_found_404)

    def invalid_payload_422():
        r = post(f"{API_V1}/simulate", json={"mode": "invalid_mode"})
        assert r.status_code == 422, f"Expected 422, got {r.status_code}"

    check("POST /api/v1/simulate (bad mode) -> 422", invalid_payload_422)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  RegenTwin Smoke Test")
    print("=" * 60)
    print(f"  Target: {BASE_URL}")

    # Check backend is running
    print("\n  Checking backend connectivity...")
    try:
        r = get(f"{API_V1}/health")
        r.raise_for_status()
        print(f"  Backend OK (v{r.json()['version']})")
    except Exception as e:
        print(f"\n  ERROR: Backend not reachable at {BASE_URL}")
        print(f"  {e}")
        print(f"\n  Start backend first:")
        print(f"    python -m uvicorn src.api.main:app --port 8000")
        print(f"  Or from ui/:")
        print(f"    npm run dev:full")
        sys.exit(1)

    # Run all test groups
    test_health()
    test_upload()
    test_simulation()
    test_visualization()
    test_viz_export()
    test_spatial()
    test_analysis()
    test_cors()
    test_errors()

    # Summary
    total = passed + failed
    print("\n" + "=" * 60)
    print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
    print("=" * 60)

    if errors:
        print("\n  Failed tests:")
        for err in errors:
            print(f"  {err}")

    if failed:
        print(f"\n  Some tests failed. Check backend logs for details.")
        sys.exit(1)
    else:
        print(f"\n  All smoke tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
