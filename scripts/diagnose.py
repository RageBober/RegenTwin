#!/usr/bin/env python3
"""RegenTwin — Полная диагностика проекта.

НЕ юнит-тесты. Это скрипт, который ПЫТАЕТСЯ ИСПОЛЬЗОВАТЬ проект
как реальный пользователь и репортит ВСЕ проблемы.

Запуск: python scripts/diagnose.py
"""

import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

# Устанавливаем рабочую директорию в корень проекта
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class DiagResult:
    name: str
    status: str  # "OK", "FAIL", "WARN", "SKIP"
    message: str = ""
    error: str = ""
    duration_ms: float = 0


@dataclass
class DiagReport:
    results: list[DiagResult] = field(default_factory=list)

    def add(self, result: DiagResult) -> None:
        self.results.append(result)
        icon = {"OK": "\u2705", "FAIL": "\u274c", "WARN": "\u26a0\ufe0f", "SKIP": "\u23ed\ufe0f"}[
            result.status
        ]
        print(f"  {icon} {result.name}: {result.message}")
        if result.error:
            lines = result.error.strip().split("\n")
            for line in lines[:5]:
                print(f"      {line}")
            if len(lines) > 5:
                print(f"      ... ({len(lines) - 5} more lines)")

    def summary(self) -> str:
        ok = sum(1 for r in self.results if r.status == "OK")
        fail = sum(1 for r in self.results if r.status == "FAIL")
        warn = sum(1 for r in self.results if r.status == "WARN")
        skip = sum(1 for r in self.results if r.status == "SKIP")
        total = len(self.results)
        return f"\n{'=' * 60}\nRESULT: {ok}/{total} OK, {fail} FAIL, {warn} WARN, {skip} SKIP\n{'=' * 60}"


def timed_test(report: DiagReport, name: str):
    """Context manager for timed tests that catches exceptions."""

    class TimedContext:
        def __enter__(self):
            self.start = time.time()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            duration = (time.time() - self.start) * 1000
            if exc_type:
                report.add(
                    DiagResult(
                        name=name,
                        status="FAIL",
                        message=f"{exc_type.__name__}: {exc_val}",
                        error=traceback.format_exc(),
                        duration_ms=duration,
                    )
                )
                return True  # suppress exception
            return False

    return TimedContext()


report = DiagReport()

print("=" * 60)
print("RegenTwin Full Diagnostics")
print(f"Python: {sys.version}")
print(f"CWD: {os.getcwd()}")
print(f"Project: {PROJECT_ROOT}")
print("=" * 60)

# ================================================================
print("\n[1/14] DEPENDENCIES")
# ================================================================

critical_deps = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("pandas", "pandas"),
    ("plotly", "plotly"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("pydantic", "pydantic"),
    ("sqlalchemy", "sqlalchemy"),
    ("loguru", "loguru"),
    ("SALib", "SALib"),
    ("matplotlib", "matplotlib"),
    ("fpdf2", "fpdf"),
    ("pydantic_settings", "pydantic_settings"),
    ("aiofiles", "aiofiles"),
    ("python-multipart", "multipart"),
]
optional_deps = [
    ("flowkit", "flowkit"),
    ("pymc", "pymc"),
    ("emcee", "emcee"),
    ("arviz", "arviz"),
    ("kaleido", "kaleido"),
    ("celery", "celery"),
]

for pip_name, import_name in critical_deps:
    with timed_test(report, f"dep:{pip_name}"):
        __import__(import_name)
        report.add(DiagResult(name=f"dep:{pip_name}", status="OK", message="ok"))

for pip_name, import_name in optional_deps:
    try:
        __import__(import_name)
        report.add(DiagResult(name=f"dep:{pip_name}", status="OK", message="ok"))
    except ImportError as e:
        report.add(
            DiagResult(name=f"dep:{pip_name}", status="WARN", message=f"optional missing: {e}")
        )

# ================================================================
print("\n[2/14] PROJECT MODULE IMPORTS")
# ================================================================

project_modules = [
    "src.core.sde_model",
    "src.core.extended_sde",
    "src.core.abm_model",
    "src.core.abm_spatial",
    "src.core.integration",
    "src.core.monte_carlo",
    "src.core.therapy_models",
    "src.core.parameters",
    "src.core.wound_phases",
    "src.core.sde_numerics",
    "src.core.robustness",
    "src.core.numerical_utils",
    "src.core.equation_free",
    "src.data.fcs_parser",
    "src.data.gating",
    "src.data.parameter_extraction",
    "src.data.validation",
    "src.data.dataset_loader",
    "src.visualization.plots",
    "src.visualization.spatial",
    "src.visualization.export",
    "src.visualization.theme",
    "src.api.main",
    "src.api.config",
    "src.api.routes.health",
    "src.api.routes.upload",
    "src.api.routes.simulate",
    "src.api.routes.results",
    "src.api.routes.analysis",
    "src.api.routes.visualization",
    "src.api.routes.spatial",
    "src.api.services.simulation_service",
    "src.api.services.analysis_service",
    "src.api.services.file_service",
    "src.api.models.schemas",
    "src.db.models",
    "src.db.session",
]

# Also check optional analysis modules
for extra in ["src.core.sensitivity_analysis", "src.core.parameter_estimation"]:
    mod_path = extra.replace(".", "/") + ".py"
    if os.path.exists(mod_path):
        project_modules.append(extra)

for mod in project_modules:
    with timed_test(report, f"mod:{mod}"):
        __import__(mod)
        report.add(DiagResult(name=f"mod:{mod}", status="OK", message="ok"))

# ================================================================
print("\n[3/14] SDE MVP (2-variable)")
# ================================================================

with timed_test(report, "sde_mvp:create"):
    from src.core.sde_model import SDEConfig, SDEModel, TherapyProtocol
    from src.data.parameter_extraction import ModelParameters

    config = SDEConfig(t_max=10.0, dt=0.1)
    therapy = TherapyProtocol(
        prp_enabled=True, prp_initial_concentration=10.0, pemf_frequency=50.0, pemf_intensity=1.0
    )
    model = SDEModel(config=config, therapy=therapy)
    report.add(DiagResult(name="sde_mvp:create", status="OK", message="created"))

with timed_test(report, "sde_mvp:simulate"):
    mvp_params = ModelParameters(
        n0=500,
        c0=10.0,
        stem_cell_fraction=0.1,
        macrophage_fraction=0.2,
        apoptotic_fraction=0.05,
        inflammation_level=0.5,
    )
    traj = model.simulate(initial_params=mvp_params)
    n = len(traj.times)
    report.add(DiagResult(name="sde_mvp:simulate", status="OK", message=f"steps={n}"))

# ================================================================
print("\n[4/14] EXTENDED SDE (20-variable)")
# ================================================================

with timed_test(report, "ext_sde:create"):
    from src.core.extended_sde import ExtendedSDEModel
    from src.core.parameters import ParameterSet

    params = ParameterSet()
    ext_model = ExtendedSDEModel(params)
    state0 = ext_model.get_default_initial_state()
    report.add(
        DiagResult(name="ext_sde:create", status="OK", message=f"dim={len(state0.to_array())}")
    )

with timed_test(report, "ext_sde:simulate_720h"):
    import numpy as np

    ext_traj = ext_model.simulate(state0, t_span=(0.0, 720.0))
    n_pts = len(ext_traj.times)
    # Check for NaN and negatives
    problems = []
    for vname in ["P", "Ne", "M1", "M2", "F", "Mf", "E", "S"]:
        try:
            arr = ext_traj.get_variable(vname)
        except KeyError:
            arr = None
        if arr is not None and len(arr) > 0:
            if np.any(np.isnan(arr)):
                problems.append(f"{vname}:NaN")
            if np.any(arr < -1e-6):
                problems.append(f"{vname}:neg")
    status = "FAIL" if any("NaN" in p for p in problems) else ("WARN" if problems else "OK")
    report.add(
        DiagResult(
            name="ext_sde:simulate_720h",
            status=status,
            message=f"pts={n_pts}" + (f", issues={problems}" if problems else ""),
        )
    )

with timed_test(report, "ext_sde:m1m2_biology"):
    m1 = ext_traj.get_variable("M1")
    m2 = ext_traj.get_variable("M2")
    times = np.asarray(ext_traj.times)
    if len(m1) > 0 and len(m2) > 0:
        early = (times >= 24) & (times <= 72)
        late = (times >= 200) & (times <= 500)
        if np.any(early) and np.any(late):
            early_r = np.mean(m1[early] / (m2[early] + 1e-10))
            late_r = np.mean(m1[late] / (m2[late] + 1e-10))
            ok = early_r > 1.0 and late_r < 1.0
            report.add(
                DiagResult(
                    name="ext_sde:m1m2_biology",
                    status="OK" if ok else "WARN",
                    message=f"early={early_r:.2f}(want>1), late={late_r:.2f}(want<1)",
                )
            )
        else:
            report.add(
                DiagResult(
                    name="ext_sde:m1m2_biology", status="WARN", message="not enough timepoints"
                )
            )
    else:
        report.add(DiagResult(name="ext_sde:m1m2_biology", status="FAIL", message="M1/M2 missing"))

# ================================================================
print("\n[5/14] ABM")
# ================================================================

with timed_test(report, "abm:simulate"):
    from src.core.abm_model import ABMConfig, simulate_abm

    abm_config = ABMConfig(t_max=25.0, dt=0.5)
    abm_params = ModelParameters(
        n0=500,
        c0=10.0,
        stem_cell_fraction=0.1,
        macrophage_fraction=0.2,
        apoptotic_fraction=0.05,
        inflammation_level=0.5,
    )
    abm_traj = simulate_abm(initial_params=abm_params, config=abm_config)
    n_snap = len(abm_traj.snapshots)
    n_agents = len(abm_traj.snapshots[-1].agents) if abm_traj.snapshots else 0
    report.add(
        DiagResult(
            name="abm:simulate",
            status="OK" if n_agents > 0 else "WARN",
            message=f"snapshots={n_snap}, final_agents={n_agents}",
        )
    )

# ================================================================
print("\n[6/14] INTEGRATION SDE+ABM")
# ================================================================

with timed_test(report, "integration:run"):
    from src.core.integration import create_default_integration_config, simulate_integrated

    int_config = create_default_integration_config(t_max_days=0.5)
    int_params = ModelParameters(
        n0=500,
        c0=10.0,
        stem_cell_fraction=0.1,
        macrophage_fraction=0.2,
        apoptotic_fraction=0.05,
        inflammation_level=0.5,
    )
    int_traj = simulate_integrated(initial_params=int_params, integration_config=int_config)
    report.add(
        DiagResult(name="integration:run", status="OK", message=f"steps={len(int_traj.times)}")
    )

# ================================================================
print("\n[7/14] MONTE CARLO")
# ================================================================

with timed_test(report, "mc:5_runs"):
    from src.core.monte_carlo import MonteCarloConfig, run_monte_carlo
    from src.core.sde_model import SDEConfig as MCSDEConfig

    mc_sde_config = MCSDEConfig(t_max=10.0, dt=0.5)
    mc_config = MonteCarloConfig(n_trajectories=5, sde_config=mc_sde_config)
    mc_params = ModelParameters(
        n0=500,
        c0=10.0,
        stem_cell_fraction=0.1,
        macrophage_fraction=0.2,
        apoptotic_fraction=0.05,
        inflammation_level=0.5,
    )
    mc_result = run_monte_carlo(initial_params=mc_params, config=mc_config)
    report.add(
        DiagResult(name="mc:5_runs", status="OK", message=f"n={len(mc_result.trajectories)}")
    )

# ================================================================
print("\n[8/14] THERAPY MODELS")
# ================================================================

with timed_test(report, "prp:compute"):
    from src.core.therapy_models import PRPConfig, PRPModel

    prp = PRPModel(PRPConfig())
    rel = prp.compute_release(t=5.0)
    report.add(DiagResult(name="prp:compute", status="OK", message=f"pdgf={rel.theta_pdgf:.4f}"))

with timed_test(report, "pemf:compute"):
    from src.core.therapy_models import PEMFConfig, PEMFModel

    pemf = PEMFModel(PEMFConfig())
    eff = pemf.compute_effects(t=5.0)
    report.add(
        DiagResult(
            name="pemf:compute", status="OK", message=f"anti_inflam={eff.anti_inflammatory:.4f}"
        )
    )

# ================================================================
print("\n[9/14] VISUALIZATION (direct)")
# ================================================================

with timed_test(report, "viz:populations"):
    from src.visualization.plots import plot_populations

    fig = plot_populations(ext_traj)
    report.add(DiagResult(name="viz:populations", status="OK", message=f"traces={len(fig.data)}"))

with timed_test(report, "viz:cytokines"):
    from src.visualization.plots import plot_cytokines

    fig2 = plot_cytokines(ext_traj)
    report.add(DiagResult(name="viz:cytokines", status="OK", message=f"traces={len(fig2.data)}"))

with timed_test(report, "viz:ecm"):
    from src.visualization.plots import plot_ecm

    fig3 = plot_ecm(ext_traj)
    report.add(DiagResult(name="viz:ecm", status="OK", message=f"traces={len(fig3.data)}"))

with timed_test(report, "viz:phases"):
    from src.visualization.plots import plot_phases

    fig4 = plot_phases(ext_traj)
    report.add(DiagResult(name="viz:phases", status="OK", message=f"traces={len(fig4.data)}"))

with timed_test(report, "viz:csv_export"):
    from src.visualization.export import ReportExporter

    exporter = ReportExporter()
    exporter.add_trajectory_data("diag_run", ext_traj)
    csv_dir = PROJECT_ROOT / "output"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_files = exporter.to_csv(output_dir=csv_dir)
    ok = len(csv_files) > 0 and all(p.exists() and p.stat().st_size > 100 for p in csv_files)
    report.add(
        DiagResult(
            name="viz:csv_export", status="OK" if ok else "FAIL", message=f"files={len(csv_files)}"
        )
    )

# ================================================================
print("\n[10/14] DATABASE")
# ================================================================

with timed_test(report, "db:create_tables"):
    from src.db.session import create_tables, engine

    create_tables()
    from sqlalchemy import inspect

    tables = inspect(engine).get_table_names()
    report.add(DiagResult(name="db:create_tables", status="OK", message=f"tables={tables}"))

with timed_test(report, "db:crud"):
    import uuid

    from src.db.models import SimulationRecord
    from src.db.session import SessionLocal

    db = SessionLocal()
    try:
        rec = SimulationRecord(
            id=str(uuid.uuid4()), mode="extended", status="completed", params_json={"test": True}
        )
        db.add(rec)
        db.commit()
        found = db.query(SimulationRecord).filter_by(id=rec.id).first()
        ok = found is not None
        if found:
            db.delete(found)
            db.commit()
        report.add(
            DiagResult(
                name="db:crud",
                status="OK" if ok else "FAIL",
                message="read_back=ok" if ok else "read_back=failed",
            )
        )
    finally:
        db.close()

# ================================================================
print("\n[11/14] FASTAPI ENDPOINTS (TestClient)")
# ================================================================

with timed_test(report, "api:create_app"):
    from src.api.main import app

    report.add(DiagResult(name="api:create_app", status="OK", message=f"routes={len(app.routes)}"))

from fastapi.testclient import TestClient

client = TestClient(app)
client.__enter__()

with timed_test(report, "api:health"):
    r = client.get("/api/v1/health")
    report.add(
        DiagResult(
            name="api:health",
            status="OK" if r.status_code == 200 else "FAIL",
            message=f"{r.status_code}: {r.text[:100]}",
        )
    )

sim_payload = {
    "mode": "extended",
    "P0": 500,
    "Ne0": 200,
    "M1_0": 100,
    "M2_0": 10,
    "F0": 50,
    "Mf0": 0,
    "E0": 20,
    "S0": 40,
    "C_TNF0": 10.0,
    "C_IL10_0": 0.5,
    "D0": 5.0,
    "O2_0": 80.0,
    "t_max_hours": 50,
    "dt": 0.5,
    "prp_enabled": False,
    "pemf_enabled": False,
    "prp_intensity": 1.0,
    "pemf_frequency": 50.0,
    "pemf_intensity": 1.0,
    "random_seed": 42,
    "n_trajectories": 1,
    "upload_id": None,
}

# Simulate endpoints — async polling чтобы event loop выполнял
# фоновые asyncio.create_task() между запросами.
# Синхронный TestClient останавливает event loop после ответа,
# поэтому фоновые задачи не выполняются и статус навсегда "pending".
import asyncio

import httpx


async def _run_simulate_tests(app, sim_payload, report):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        for mode in ["extended", "mvp", "abm", "integrated"]:
            with timed_test(report, f"api:simulate_{mode}"):
                p = dict(sim_payload, mode=mode)
                r = await ac.post("/api/v1/simulate", json=p)
                report.add(
                    DiagResult(
                        name=f"api:simulate_{mode}",
                        status="OK" if r.status_code < 400 else "FAIL",
                        message=f"{r.status_code}: {r.text[:200]}",
                    )
                )

                if r.status_code < 400:
                    sid = r.json().get("simulation_id")
                    if sid:
                        final = {}
                        for _ in range(240):
                            await asyncio.sleep(0.5)
                            sr = await ac.get(f"/api/v1/simulate/{sid}")
                            if sr.status_code == 200 and sr.json().get("status") in (
                                "completed",
                                "failed",
                                "cancelled",
                            ):
                                final = sr.json()
                                break
                        else:
                            final = sr.json() if sr.status_code == 200 else {}
                        report.add(
                            DiagResult(
                                name=f"api:sim_{mode}_result",
                                status="OK" if final.get("status") == "completed" else "FAIL",
                                message=f"status={final.get('status')}, msg={(final.get('message') or '')[:100]}",
                            )
                        )


asyncio.run(_run_simulate_tests(app, sim_payload, report))

with timed_test(report, "api:simulations_list"):
    r = client.get("/api/v1/simulations")
    report.add(
        DiagResult(
            name="api:simulations_list",
            status="OK" if r.status_code == 200 else "FAIL",
            message=f"{r.status_code}, count={len(r.json()) if r.status_code == 200 else 'N/A'}",
        )
    )

# ================================================================
print("\n[12/14] VISUALIZATION API")
# ================================================================

viz_params = {
    "P0": 500,
    "Ne0": 200,
    "M1_0": 100,
    "M2_0": 10,
    "F0": 50,
    "Mf0": 0,
    "E0": 20,
    "S0": 40,
    "C_TNF0": 10.0,
    "C_IL10_0": 0.5,
    "D0": 5.0,
    "O2_0": 80.0,
    "t_max_hours": 50,
    "dt": 0.5,
    "prp_enabled": False,
    "pemf_enabled": False,
    "prp_intensity": 1.0,
    "pemf_frequency": 50.0,
    "pemf_intensity": 1.0,
    "random_seed": 42,
}

for ep in ["populations", "cytokines", "ecm", "phases", "comparison"]:
    with timed_test(report, f"viz_api:{ep}"):
        r = client.post(f"/api/viz/{ep}", json={"simulation": viz_params})
        if r.status_code == 200:
            n_traces = len(r.json().get("data", []))
            report.add(DiagResult(name=f"viz_api:{ep}", status="OK", message=f"traces={n_traces}"))
        else:
            report.add(
                DiagResult(
                    name=f"viz_api:{ep}", status="FAIL", message=f"{r.status_code}: {r.text[:200]}"
                )
            )

# ================================================================
print("\n[13/14] SENSITIVITY ANALYSIS API")
# ================================================================

with timed_test(report, "api:sensitivity"):
    sa_payload = {
        "simulation_params": sim_payload,
        "parameters": ["r", "K"],
        "method": "sobol",
        "n_samples": 64,
    }
    r = client.post("/api/v1/analysis/sensitivity", json=sa_payload)
    report.add(
        DiagResult(
            name="api:sensitivity",
            status="OK" if r.status_code < 400 else "FAIL",
            message=f"{r.status_code}: {r.text[:200]}",
        )
    )

try:
    client.__exit__(None, None, None)
except Exception:
    pass

# ================================================================
print("\n[14/14] FILE STRUCTURE")
# ================================================================

for d in ["data/uploads", "data/results", "data/mock"]:
    p = PROJECT_ROOT / d
    report.add(
        DiagResult(
            name=f"dir:{d}",
            status="OK" if p.exists() else "WARN",
            message="exists" if p.exists() else "MISSING — creating",
        )
    )
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)

# ================================================================
print(report.summary())

# Save JSON report
rpt_path = PROJECT_ROOT / "output" / "diagnosis_report.json"
rpt_path.parent.mkdir(parents=True, exist_ok=True)
with open(rpt_path, "w", encoding="utf-8") as f:
    json.dump(
        [
            {
                "name": r.name,
                "status": r.status,
                "message": r.message,
                "error": r.error[:500] if r.error else "",
            }
            for r in report.results
        ],
        f,
        indent=2,
        ensure_ascii=False,
    )
print(f"\nReport saved: {rpt_path}")

fails = sum(1 for r in report.results if r.status == "FAIL")
print(f"\nFailed checks: {fails}")
if fails > 0:
    print("\nFAILED:")
    for r in report.results:
        if r.status == "FAIL":
            print(f"  - {r.name}: {r.message}")

sys.exit(1 if fails > 0 else 0)
