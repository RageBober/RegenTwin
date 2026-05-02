"""Сборка Markdown-отчёта со сравнением машин по бенчмаркам.

Читает все файлы `output/benchmarks/*.json` (per-machine snapshots),
дедуплицирует по `machine_id` (берёт самый свежий timestamp),
строит matplotlib-графики и пишет `output/benchmarks/report.md`.

Usage:
    uv run python scripts/generate_benchmark_report.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = PROJECT_ROOT / "output" / "benchmarks"
FIGURES_DIR = BENCH_DIR / "figures"

# Группы бенчмарков → human readable names и тип графика.
GROUP_LABELS: dict[str, str] = {
    "sde-small": "Extended SDE (24h, ~2400 шагов)",
    "sde-large": "Extended SDE (72h, ~7200 шагов)",
    "abm-small": "ABM (100 агентов, 24h)",
    "abm-medium": "ABM (500 агентов, 24h)",
    "mc-serial": "Monte Carlo serial (4 траектории)",
    "mc-parallel": "Monte Carlo parallel (n_jobs=cpu-1)",
    "sensitivity-sobol": "Sobol sensitivity (Ishigami, N=64)",
}

GROUP_ORDER = list(GROUP_LABELS.keys())


def _load_snapshots() -> list[dict[str, Any]]:
    """Загрузить все JSON, оставить latest per machine_id."""
    snapshots: dict[str, dict[str, Any]] = {}
    for path in sorted(BENCH_DIR.glob("*.json")):
        if path.name.startswith("_raw_"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        machine_id = data.get("machine_id")
        if not machine_id:
            continue
        existing = snapshots.get(machine_id)
        if existing is None or data["timestamp"] > existing["timestamp"]:
            snapshots[machine_id] = data
    return list(snapshots.values())


def _plot_group_comparison(snapshots: list[dict[str, Any]], group: str, title: str) -> Path | None:
    """Bar chart: одна группа, по машинам."""
    labels: list[str] = []
    means: list[float] = []
    stds: list[float] = []
    for snap in snapshots:
        bench = snap.get("benchmarks", {}).get(group)
        if bench is None:
            continue
        labels.append(snap["label"])
        means.append(bench["mean"])
        stds.append(bench["stddev"])

    if not labels:
        return None

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    x = range(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=6, color="#3182bd", edgecolor="#08519c")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Время выполнения, секунды")
    ax.set_title(title)
    for bar, mean in zip(bars, means, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(stds) * 0.05 if stds else bar.get_height() * 1.02,
            f"{mean:.3g}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()

    out = FIGURES_DIR / f"{group}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def _plot_speedup(snapshots: list[dict[str, Any]]) -> Path | None:
    """Speedup serial → parallel per machine."""
    labels: list[str] = []
    speedups: list[float] = []
    for snap in snapshots:
        serial = snap.get("benchmarks", {}).get("mc-serial")
        parallel = snap.get("benchmarks", {}).get("mc-parallel")
        if not serial or not parallel:
            continue
        speedups.append(serial["mean"] / parallel["mean"] if parallel["mean"] > 0 else 0.0)
        labels.append(snap["label"])

    if not labels:
        return None

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    ax.bar(labels, speedups, color="#fd8d3c", edgecolor="#a63603")
    ax.set_ylabel("Speedup (serial / parallel)")
    ax.set_title("Monte Carlo speedup от мультипроцессинга")
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8)
    for i, val in enumerate(speedups):
        ax.text(i, val + 0.05, f"×{val:.2f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()

    out = FIGURES_DIR / "mc-speedup.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def _format_system_table(snapshots: list[dict[str, Any]]) -> str:
    rows = [
        "| Метка | CPU | Cores | RAM | OS | Python |",
        "|-------|-----|-------|-----|----|--------|",
    ]
    for snap in snapshots:
        s = snap["system"]
        rows.append(
            f"| **{snap['label']}** | {s['cpu_brand']} | "
            f"{s['physical_cores']}/{s['logical_cores']} | {s['ram_gb']} GB | "
            f"{s['os']} | {s['python']} |"
        )
    return "\n".join(rows)


def _format_results_table(snapshots: list[dict[str, Any]]) -> str:
    headers = ["Группа"] + [snap["label"] for snap in snapshots]
    rows = ["| " + " | ".join(headers) + " |"]
    rows.append("|" + "|".join(["---"] * len(headers)) + "|")
    for group in GROUP_ORDER:
        cells = [GROUP_LABELS[group]]
        for snap in snapshots:
            bench = snap.get("benchmarks", {}).get(group)
            if bench is None:
                cells.append("—")
            else:
                cells.append(f"{bench['mean']:.4f}s ± {bench['stddev']:.4f}")
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main() -> int:
    snapshots = _load_snapshots()
    if not snapshots:
        print(f"No benchmark snapshots found in {BENCH_DIR}")
        return 1

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    snapshots.sort(key=lambda s: s["label"])

    print(f"Loaded {len(snapshots)} machine snapshots: {[s['label'] for s in snapshots]}")

    # Generate figures
    figures: list[Path] = []
    for group, title in GROUP_LABELS.items():
        fig = _plot_group_comparison(snapshots, group, title)
        if fig:
            figures.append(fig)
    speedup_fig = _plot_speedup(snapshots)
    if speedup_fig:
        figures.append(speedup_fig)

    # Compose Markdown
    lines: list[str] = [
        "# RegenTwin — Benchmark Report",
        "",
        f"_Сгенерировано из {len(snapshots)} snapshot(ов) в `output/benchmarks/`._",
        "",
        "## System Configurations",
        "",
        _format_system_table(snapshots),
        "",
        "## Benchmark Results",
        "",
        _format_results_table(snapshots),
        "",
        "## Графики",
        "",
    ]
    for fig in figures:
        rel = fig.relative_to(BENCH_DIR).as_posix()
        lines.append(f"![{fig.stem}]({rel})")
        lines.append("")

    if any((BENCH_DIR / "profiles").glob("*.svg")):
        lines += [
            "## Hot-spot Profiles (py-spy flamegraphs)",
            "",
        ]
        for svg in sorted((BENCH_DIR / "profiles").glob("*.svg")):
            rel = svg.relative_to(BENCH_DIR).as_posix()
            lines.append(f"- [{svg.stem}]({rel})")
        lines.append("")

    report = BENCH_DIR / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report: {report}")
    print(f"Figures: {len(figures)} in {FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
