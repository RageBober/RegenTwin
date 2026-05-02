"""Копирует бенчмарк-отчёт и фигуры в docs/benchmarks/ для MkDocs.

Запускается:
- локально перед `mkdocs serve`
- в `.github/workflows/docs.yml` перед `mkdocs gh-deploy`

Источник: output/benchmarks/{report.md,figures/}
Цель:    docs/benchmarks/{index.md,figures/}

Если отчёт отсутствует — оставляет существующий `docs/benchmarks/index.md` нетронутым.
"""

from __future__ import annotations

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_REPORT = PROJECT_ROOT / "output" / "benchmarks" / "report.md"
SRC_FIGURES = PROJECT_ROOT / "output" / "benchmarks" / "figures"
DST_DIR = PROJECT_ROOT / "docs" / "benchmarks"
DST_REPORT = DST_DIR / "report.md"
DST_FIGURES = DST_DIR / "figures"


def main() -> int:
    DST_DIR.mkdir(parents=True, exist_ok=True)

    if SRC_REPORT.exists():
        DST_REPORT.write_text(SRC_REPORT.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Copied report → {DST_REPORT}")
    else:
        print(f"No report at {SRC_REPORT}, skipping (docs/benchmarks/index.md untouched)")

    if SRC_FIGURES.exists():
        if DST_FIGURES.exists():
            shutil.rmtree(DST_FIGURES)
        shutil.copytree(SRC_FIGURES, DST_FIGURES)
        print(f"Copied figures → {DST_FIGURES}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
