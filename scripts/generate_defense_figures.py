"""Генерация графиков для презентации защиты RegenTwin.

Использует существующие plotly-функции из `src/visualization/`,
читает любой `.npz` из `data/results/` и экспортирует PNG в `output/defense_figures/`
через kaleido (уже в зависимостях проекта).

Usage:
    # 1. Сначала посмотреть, что есть в data/results/ — какие режимы, t_max, и т.д.
    uv run python scripts/generate_defense_figures.py list

    # 2. Сгенерировать конкретный график по UUID симуляции
    uv run python scripts/generate_defense_figures.py populations --sim <uuid>
    uv run python scripts/generate_defense_figures.py cytokines   --sim <uuid>
    uv run python scripts/generate_defense_figures.py phases      --sim <uuid>
    uv run python scripts/generate_defense_figures.py spatial     --sim <abm-uuid>

    # 3. Сравнение baseline vs PRP (нужно два прогона extended-режима)
    uv run python scripts/generate_defense_figures.py compare \\
        --baseline <baseline-uuid> --prp <prp-uuid>

    # 4. Сгенерировать всё, что определится автоматически
    uv run python scripts/generate_defense_figures.py auto

Достаточно полные UUID; принимаются и первые 8 символов, если они уникальны.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Делаем src/ импортируемым, когда скрипт запускается напрямую
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.services.result_bundle import (  # noqa: E402
    build_abm_snapshot,
    build_extended_trajectory,
    load_result_bundle,
)
from src.visualization.plots import (  # noqa: E402
    plot_comparison,
    plot_cytokines,
    plot_phases,
    plot_populations,
)
from src.visualization.spatial import scatter_agents  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "data" / "results"
OUT_DIR = PROJECT_ROOT / "output" / "defense_figures"

# Стандартный размер для слайдов 16:9
FIG_WIDTH = 1600
FIG_HEIGHT = 900
FIG_SCALE = 2  # retina-quality PNG


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_npz(sim_id: str) -> Path:
    """Найти .npz по полному UUID или префиксу."""
    candidates = list(RESULTS_DIR.glob(f"{sim_id}*.npz"))
    if len(candidates) == 0:
        raise SystemExit(f"Не найден .npz для '{sim_id}' в {RESULTS_DIR}")
    if len(candidates) > 1:
        names = ", ".join(c.name for c in candidates)
        raise SystemExit(f"Префикс '{sim_id}' неоднозначен ({names}) — укажи больше символов")
    return candidates[0]


def _save_png(fig, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{name}.png"
    fig.write_image(str(out), width=FIG_WIDTH, height=FIG_HEIGHT, scale=FIG_SCALE)
    print(f"  ✓ {out.relative_to(PROJECT_ROOT)}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  list — что лежит в data/results/
# ─────────────────────────────────────────────────────────────────────────────


def cmd_list(_args: argparse.Namespace) -> None:
    files = sorted(RESULTS_DIR.glob("*.npz"))
    if not files:
        print(f"Нет .npz в {RESULTS_DIR}")
        return

    print(f"{'uuid (8)':<12} {'mode':<12} {'t_max, ч':>10} {'snapshots':>10} {'series':>8}")
    print("-" * 62)
    for f in files:
        try:
            bundle = load_result_bundle(f, include_snapshots=False)
        except Exception as exc:  # noqa: BLE001
            print(f"{f.stem[:8]:<12} ОШИБКА: {exc}")
            continue
        mode = bundle.get("mode", "?")
        times = bundle.get("times", [])
        t_max = times[-1] if times else 0
        meta = bundle.get("metadata", {})
        snaps = meta.get("snapshot_count", "—")
        n_series = len(bundle.get("variables", {}))
        print(f"{f.stem[:8]:<12} {mode:<12} {t_max:>10.1f} {snaps!s:>10} {n_series:>8}")


# ─────────────────────────────────────────────────────────────────────────────
#  Каждый отдельный график
# ─────────────────────────────────────────────────────────────────────────────


def cmd_populations(args: argparse.Namespace) -> Path:
    npz = _resolve_npz(args.sim)
    bundle = load_result_bundle(npz, default_mode="extended")
    if bundle["mode"] not in {"extended", "integrated"}:
        raise SystemExit(
            f"populations требует режим extended/integrated, у этой симуляции: {bundle['mode']}"
        )
    traj = build_extended_trajectory(bundle)
    fig = plot_populations(traj, height=FIG_HEIGHT)
    return _save_png(fig, f"01_populations_{npz.stem[:8]}")


def cmd_cytokines(args: argparse.Namespace) -> Path:
    npz = _resolve_npz(args.sim)
    bundle = load_result_bundle(npz, default_mode="extended")
    if bundle["mode"] not in {"extended", "integrated"}:
        raise SystemExit(
            f"cytokines требует режим extended/integrated, у этой симуляции: {bundle['mode']}"
        )
    traj = build_extended_trajectory(bundle)
    fig = plot_cytokines(traj, layout="overlay", height=FIG_HEIGHT)
    return _save_png(fig, f"02_cytokines_{npz.stem[:8]}")


def cmd_phases(args: argparse.Namespace) -> Path:
    npz = _resolve_npz(args.sim)
    bundle = load_result_bundle(npz, default_mode="extended")
    if bundle["mode"] not in {"extended", "integrated"}:
        raise SystemExit(
            f"phases требует режим extended/integrated, у этой симуляции: {bundle['mode']}"
        )
    traj = build_extended_trajectory(bundle)
    fig = plot_phases(traj, height=FIG_HEIGHT)
    return _save_png(fig, f"03_phases_{npz.stem[:8]}")


def cmd_compare(args: argparse.Namespace) -> Path:
    """Side-by-side baseline vs PRP — главный плот доклада."""
    base_npz = _resolve_npz(args.baseline)
    prp_npz = _resolve_npz(args.prp)

    base = build_extended_trajectory(load_result_bundle(base_npz, default_mode="extended"))
    prp = build_extended_trajectory(load_result_bundle(prp_npz, default_mode="extended"))

    # Сравниваем по фибробластам — главный таргет PRP в литературе (PDGF-driven).
    # Если хочешь сравнить по другой переменной — поменяй variable.
    fig = plot_comparison(
        results={"Контроль": base, "PRP": prp},
        variable=args.variable,
        show_all_populations=args.all_populations,
        height=FIG_HEIGHT,
    )
    return _save_png(fig, f"04_compare_{args.variable}_{base_npz.stem[:6]}_vs_{prp_npz.stem[:6]}")


def cmd_spatial(args: argparse.Namespace) -> list[Path]:
    """Snapshot ABM в нескольких моментах времени — наглядность пространственной части."""
    npz = _resolve_npz(args.sim)
    bundle = load_result_bundle(npz, default_mode="abm", include_snapshots=True)
    if bundle["mode"] != "abm":
        raise SystemExit(f"spatial требует режим abm, у этой симуляции: {bundle['mode']}")

    snapshots = bundle.get("snapshots") or []
    if not snapshots:
        raise SystemExit(
            "В этой ABM-симуляции не сохранены snapshots — перезапусти ABM-режим, чтобы получить пространственную динамику."
        )

    n_snaps = len(snapshots)
    # Берём indices равномерно: начало, середина, конец — или то, что задал пользователь
    if args.timesteps:
        indices = [int(t) for t in args.timesteps.split(",")]
    else:
        # Пытаемся попасть в early / mid / late фазы
        indices = [0, n_snaps // 2, n_snaps - 1]

    out_paths: list[Path] = []
    for i in indices:
        if not 0 <= i < n_snaps:
            print(f"  ⚠ snapshot[{i}] вне диапазона (0..{n_snaps - 1}), пропускаю")
            continue
        snap = build_abm_snapshot(bundle, timestep=i)
        fig = scatter_agents(snap, color_by="type", height=FIG_HEIGHT)
        out = _save_png(fig, f"05_spatial_t{int(snap.t):04d}h_{npz.stem[:8]}")
        out_paths.append(out)
    return out_paths


# ─────────────────────────────────────────────────────────────────────────────
#  auto — попытаться сделать всё, что возможно из самых свежих симуляций
# ─────────────────────────────────────────────────────────────────────────────


def cmd_auto(_args: argparse.Namespace) -> None:
    """Авто-выбор: последний extended → populations/cytokines/phases, последний abm → spatial."""
    files = sorted(RESULTS_DIR.glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise SystemExit(f"Нет .npz в {RESULTS_DIR}")

    extended_id: str | None = None
    abm_id: str | None = None

    for f in files:
        try:
            mode = load_result_bundle(f, include_snapshots=False)["mode"]
        except Exception:  # noqa: BLE001
            continue
        if extended_id is None and mode in {"extended", "integrated"}:
            extended_id = f.stem
        if abm_id is None and mode == "abm":
            abm_id = f.stem
        if extended_id and abm_id:
            break

    if extended_id:
        print(f"\n[populations] sim={extended_id[:8]}")
        cmd_populations(argparse.Namespace(sim=extended_id))
        print(f"\n[cytokines] sim={extended_id[:8]}")
        cmd_cytokines(argparse.Namespace(sim=extended_id))
        print(f"\n[phases] sim={extended_id[:8]}")
        cmd_phases(argparse.Namespace(sim=extended_id))
    else:
        print("⚠ Не нашёл extended/integrated .npz — пропускаю populations/cytokines/phases")

    if abm_id:
        print(f"\n[spatial] sim={abm_id[:8]}")
        try:
            cmd_spatial(argparse.Namespace(sim=abm_id, timesteps=None))
        except SystemExit as exc:
            print(f"  ⚠ {exc}")
    else:
        print("⚠ Не нашёл abm .npz — пропускаю spatial")

    print(
        "\n[compare] для side-by-side baseline vs PRP запусти:\n"
        "  uv run python scripts/generate_defense_figures.py compare "
        "--baseline <id1> --prp <id2>\n"
        "  Это требует двух заранее запущенных extended-симуляций "
        "(одна с prp_enabled=False, другая с True)."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="Показать .npz в data/results/ с режимами и t_max")

    p_pop = sub.add_parser("populations", help="График всех 8 клеточных популяций")
    p_pop.add_argument("--sim", required=True, help="UUID или его префикс")

    p_cyt = sub.add_parser("cytokines", help="График всех 7 цитокинов overlay")
    p_cyt.add_argument("--sim", required=True)

    p_ph = sub.add_parser("phases", help="Фазы заживления + ключевые популяции")
    p_ph.add_argument("--sim", required=True)

    p_cmp = sub.add_parser("compare", help="Side-by-side baseline vs PRP")
    p_cmp.add_argument("--baseline", required=True, help="UUID симуляции без терапии")
    p_cmp.add_argument("--prp", required=True, help="UUID симуляции с PRP")
    p_cmp.add_argument(
        "--variable", default="F", help="Какую переменную сравнивать (F, M2, rho_collagen, ...)"
    )
    p_cmp.add_argument("--all-populations", action="store_true", help="Все 8 популяций в subplots")

    p_sp = sub.add_parser("spatial", help="ABM snapshots в нескольких моментах времени")
    p_sp.add_argument("--sim", required=True)
    p_sp.add_argument(
        "--timesteps",
        default=None,
        help="Индексы snapshots через запятую (по умолчанию: начало, середина, конец)",
    )

    sub.add_parser("auto", help="Сделать всё, что найдётся автоматически")

    args = parser.parse_args()

    handlers = {
        "list": cmd_list,
        "populations": cmd_populations,
        "cytokines": cmd_cytokines,
        "phases": cmd_phases,
        "compare": cmd_compare,
        "spatial": cmd_spatial,
        "auto": cmd_auto,
    }
    handlers[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
