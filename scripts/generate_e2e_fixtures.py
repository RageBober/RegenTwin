"""Генерирует единую FCS-фикстуру для E2E тестов (pytest + Playwright).

Создаёт один файл `sample.fcs` одновременно в двух местах:
    - tests/e2e/fixtures/sample.fcs  (для backend pytest)
    - ui/e2e/fixtures/sample.fcs     (для Playwright)

Идемпотентен: пересоздаёт файл, если он отсутствует или старше генератора.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.generate_test_fcs import write_fcs
from tests.fixtures.mock_data import generate_normal_fcs_data

FIXTURE_TARGETS = [
    ROOT / "tests" / "e2e" / "fixtures" / "sample.fcs",
    ROOT / "ui" / "e2e" / "fixtures" / "sample.fcs",
]

SOURCE_SCRIPTS = [
    Path(__file__),
    ROOT / "scripts" / "generate_test_fcs.py",
    ROOT / "tests" / "fixtures" / "mock_data.py",
]


def _needs_regenerate(target: Path) -> bool:
    if not target.exists():
        return True
    target_mtime = target.stat().st_mtime
    return any(src.stat().st_mtime > target_mtime for src in SOURCE_SCRIPTS if src.exists())


def main() -> int:
    targets = [t for t in FIXTURE_TARGETS if _needs_regenerate(t)]
    if not targets:
        print("E2E FCS fixtures are up to date.")
        return 0

    df = generate_normal_fcs_data(n_events=5000, seed=42)
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        write_fcs(target, df)

    print(f"Generated {len(targets)} E2E FCS fixture(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
