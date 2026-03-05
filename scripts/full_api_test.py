"""Полный end-to-end тест всех API эндпоинтов."""

import json
import time
import requests

BASE = "http://127.0.0.1:8000"
PASS = 0
FAIL = 0


def test(name, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  PASS  {name}")
        PASS += 1
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        FAIL += 1


def main():
    global PASS, FAIL

    # 1. Health
    print("\n=== Health ===")
    def t_health():
        r = requests.get(f"{BASE}/api/v1/health")
        assert r.status_code == 200, f"status={r.status_code}"
        d = r.json()
        assert d["status"] == "ok"
        assert "uptime_seconds" in d
    test("GET /api/v1/health", t_health)

    # 2. Upload
    print("\n=== Upload ===")
    upload_id = None
    def t_upload():
        nonlocal upload_id
        with open("data/uploads/sample_normal.fcs", "rb") as f:
            r = requests.post(f"{BASE}/api/v1/upload", files={"file": ("sample_normal.fcs", f)})
        assert r.status_code == 200, f"status={r.status_code}: {r.text}"
        d = r.json()
        assert d["status"] == "ready", f"status={d['status']}, meta={d.get('metadata')}"
        assert d["metadata"]["n_events"] == 10000
        upload_id = d["upload_id"]
    test("POST /api/v1/upload (FCS)", t_upload)

    def t_upload_get():
        r = requests.get(f"{BASE}/api/v1/upload/{upload_id}")
        assert r.status_code == 200
        assert r.json()["status"] == "ready"
    test("GET /api/v1/upload/{id}", t_upload_get)

    # 3. Simulate (extended)
    print("\n=== Simulate ===")
    sim_id = None
    def t_simulate():
        nonlocal sim_id
        r = requests.post(f"{BASE}/api/v1/simulate", json={
            "mode": "extended",
            "upload_id": upload_id,
            "t_max_hours": 24,
            "n_trajectories": 1,
        })
        assert r.status_code == 200, f"status={r.status_code}: {r.text}"
        d = r.json()
        assert d["status"] in ("pending", "running", "completed")
        sim_id = d["simulation_id"]
    test("POST /api/v1/simulate (extended)", t_simulate)

    # Wait for completion
    for _ in range(30):
        r = requests.get(f"{BASE}/api/v1/simulate/{sim_id}")
        if r.json()["status"] == "completed":
            break
        time.sleep(0.5)

    def t_sim_status():
        r = requests.get(f"{BASE}/api/v1/simulate/{sim_id}")
        assert r.status_code == 200
        d = r.json()
        assert d["status"] == "completed", f"status={d['status']}, msg={d.get('message')}"
        assert d["progress"] == 100.0
    test("GET /api/v1/simulate/{id} (completed)", t_sim_status)

    # 4. Results
    print("\n=== Results ===")
    def t_results():
        r = requests.get(f"{BASE}/api/v1/results/{sim_id}")
        assert r.status_code == 200, f"status={r.status_code}: {r.text}"
        d = r.json()
        assert len(d["variables"]) == 20, f"got {len(d['variables'])} vars"
        assert len(d["times"]) > 100
        assert "P" in d["variables"]
        assert "C_TNF" in d["variables"]
    test("GET /api/v1/results/{id}", t_results)

    # 5. Simulations list
    def t_list():
        r = requests.get(f"{BASE}/api/v1/simulations")
        assert r.status_code == 200
        assert len(r.json()) >= 1
    test("GET /api/v1/simulations", t_list)

    # 6. Viz endpoints (все POST, принимают JSON с simulation params)
    print("\n=== Visualization ===")
    viz_body = {"simulation": {"t_max_hours": 24, "dt": 0.5}}  # быстрая симуляция

    for endpoint in ["populations", "cytokines", "ecm", "phases"]:
        def t_viz(ep=endpoint):
            r = requests.post(f"{BASE}/api/viz/{ep}", json=viz_body)
            assert r.status_code == 200, f"status={r.status_code}: {r.text[:300]}"
            d = r.json()
            assert "data" in d or "layout" in d, f"unexpected response keys: {list(d.keys())}"
        test(f"POST /api/viz/{endpoint}", t_viz)

    # 7. Export PNG (POST)
    def t_png():
        r = requests.post(f"{BASE}/api/viz/export/png", json=viz_body)
        assert r.status_code == 200, f"status={r.status_code}: {r.text[:300]}"
        assert len(r.content) > 100, f"PNG too small: {len(r.content)} bytes"
    test("POST /api/viz/export/png", t_png)

    # 8. Export PDF (POST)
    def t_pdf():
        r = requests.post(f"{BASE}/api/viz/export/pdf", json=viz_body)
        assert r.status_code == 200, f"status={r.status_code}: {r.text[:300]}"
        assert len(r.content) > 100, f"PDF too small: {len(r.content)} bytes"
    test("POST /api/viz/export/pdf", t_pdf)

    # 9. Spatial endpoints (POST /api/viz/spatial/*)
    print("\n=== Spatial ===")
    spatial_body = {"t_max_hours": 2, "dt": 0.5}  # короткая ABM симуляция

    for ep in ["heatmap", "scatter", "inflammation"]:
        def t_spatial(ep=ep):
            r = requests.post(f"{BASE}/api/viz/spatial/{ep}", json=spatial_body, timeout=120)
            assert r.status_code == 200, f"status={r.status_code}: {r.text[:300]}"
            d = r.json()
            assert "data" in d or "layout" in d, f"unexpected keys: {list(d.keys())}"
        test(f"POST /api/viz/spatial/{ep}", t_spatial)

    # 10. ABM simulate (через /api/v1/simulate)
    print("\n=== ABM Simulation ===")
    def t_abm():
        r = requests.post(f"{BASE}/api/v1/simulate", json={
            "mode": "abm",
            "t_max_hours": 2,
            "dt": 0.5,
            "n_trajectories": 1,
        })
        assert r.status_code == 200, f"status={r.status_code}: {r.text[:300]}"
        abm_id = r.json()["simulation_id"]
        for _ in range(60):
            s = requests.get(f"{BASE}/api/v1/simulate/{abm_id}").json()
            if s["status"] in ("completed", "failed"):
                break
            time.sleep(1)
        assert s["status"] == "completed", f"ABM status={s['status']}, msg={s.get('message')}"
    test("POST /api/v1/simulate (abm, 2h)", t_abm)

    # 11. MVP simulate
    def t_mvp():
        r = requests.post(f"{BASE}/api/v1/simulate", json={
            "mode": "mvp",
            "t_max_hours": 24,
        })
        assert r.status_code == 200, f"status={r.status_code}: {r.text[:300]}"
        mvp_id = r.json()["simulation_id"]
        for _ in range(20):
            s = requests.get(f"{BASE}/api/v1/simulate/{mvp_id}").json()
            if s["status"] in ("completed", "failed"):
                break
            time.sleep(0.5)
        assert s["status"] == "completed", f"MVP status={s['status']}, msg={s.get('message')}"
    test("POST /api/v1/simulate (mvp)", t_mvp)

    # 12. Sensitivity analysis (маленький n_samples для скорости)
    print("\n=== Analysis ===")
    def t_sensitivity():
        r = requests.post(f"{BASE}/api/v1/analysis/sensitivity", json={
            "n_samples": 64,
            "parameters": ["r_F", "K_F"],  # реальные имена из ParameterSet
            "simulation_params": {"t_max_hours": 72, "dt": 0.5},
        })
        assert r.status_code == 200, f"status={r.status_code}: {r.text[:300]}"
        d = r.json()
        assert d["analysis_type"] == "sensitivity"
        anal_id = d["analysis_id"]
        # Ждём до 5 минут
        for _ in range(300):
            s = requests.get(f"{BASE}/api/v1/analysis/{anal_id}").json()
            if s["status"] in ("completed", "failed"):
                break
            time.sleep(1)
        assert s["status"] == "completed", f"Sensitivity status={s['status']}, result={s.get('result')}"
    test("POST /api/v1/analysis/sensitivity", t_sensitivity)

    # 13. Error handling
    print("\n=== Error handling ===")
    def t_404_sim():
        r = requests.get(f"{BASE}/api/v1/simulate/nonexistent")
        assert r.status_code == 404
    test("GET /api/v1/simulate/nonexistent -> 404", t_404_sim)

    def t_404_upload():
        r = requests.get(f"{BASE}/api/v1/upload/nonexistent")
        assert r.status_code == 404
    test("GET /api/v1/upload/nonexistent -> 404", t_404_upload)

    def t_422():
        r = requests.post(f"{BASE}/api/v1/simulate", json={"mode": "invalid"})
        assert r.status_code == 422
    test("POST /api/v1/simulate (invalid mode) -> 422", t_422)

    # Summary
    print(f"\n{'='*50}")
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS+FAIL}")
    if FAIL:
        print("SOME TESTS FAILED!")
    else:
        print("ALL TESTS PASSED!")
    return FAIL


if __name__ == "__main__":
    exit(main())
