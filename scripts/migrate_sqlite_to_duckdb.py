"""One-shot ETL: переносит данные из старой SQLite БД в новую DuckDB.

Usage:
    uv run python scripts/migrate_sqlite_to_duckdb.py \\
        --src data/regentwin.db \\
        --dst data/regentwin.duckdb \\
        [--dry-run]

Перед запуском убедитесь, что схема в DuckDB создана:
    uv run alembic upgrade head

Скрипт читает все три таблицы (`simulations`, `uploads`, `analyses`),
декодирует JSON-колонки и переписывает их в DuckDB через SQLAlchemy.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Allow running as `python scripts/migrate_sqlite_to_duckdb.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from src.db.models import AnalysisRecord, SimulationRecord, UploadRecord  # noqa: E402

TABLES = {
    "simulations": SimulationRecord,
    "uploads": UploadRecord,
    "analyses": AnalysisRecord,
}

JSON_COLUMNS = {
    "simulations": {"params_json"},
    "uploads": {"metadata_json"},
    "analyses": {"params_json", "result_json"},
}

DATETIME_COLUMNS = {
    "simulations": {"created_at", "completed_at"},
    "uploads": {"created_at"},
    "analyses": {"created_at", "completed_at"},
}


def _decode_value(table: str, column: str, raw: Any) -> Any:
    """Convert SQLite raw value to a Python object suitable for the ORM."""
    if raw is None:
        return None
    if column in JSON_COLUMNS.get(table, set()):
        if isinstance(raw, str):
            return json.loads(raw)
        return raw
    if column in DATETIME_COLUMNS.get(table, set()):
        if isinstance(raw, str):
            return datetime.fromisoformat(raw)
        return raw
    return raw


def _read_sqlite(src: Path) -> dict[str, list[dict[str, Any]]]:
    """Считать все три таблицы из SQLite в Python dicts."""
    if not src.exists():
        raise FileNotFoundError(f"Source SQLite file not found: {src}")

    conn = sqlite3.connect(str(src))
    conn.row_factory = sqlite3.Row
    rows_by_table: dict[str, list[dict[str, Any]]] = {}

    try:
        for table in TABLES:
            cur = conn.execute(f"SELECT * FROM {table}")
            rows = [dict(row) for row in cur.fetchall()]
            rows_by_table[table] = rows
    finally:
        conn.close()

    return rows_by_table


def _write_duckdb(dst: Path, rows_by_table: dict[str, list[dict[str, Any]]]) -> None:
    """Insert decoded rows into DuckDB via SQLAlchemy ORM."""
    url = f"duckdb:///{dst.as_posix()}"
    engine = create_engine(url)
    Session = sessionmaker(bind=engine)

    with Session() as session:
        for table, rows in rows_by_table.items():
            model = TABLES[table]
            for row in rows:
                decoded = {col: _decode_value(table, col, val) for col, val in row.items()}
                session.add(model(**decoded))
        session.commit()

    engine.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path("data/regentwin.db"))
    parser.add_argument("--dst", type=Path, default=Path("data/regentwin.duckdb"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Только прочитать SQLite и вывести количество строк, ничего не писать.",
    )
    args = parser.parse_args()

    print(f"Reading SQLite: {args.src}")
    rows_by_table = _read_sqlite(args.src)
    for table, rows in rows_by_table.items():
        print(f"  {table}: {len(rows)} rows")

    if args.dry_run:
        print("\nDry-run complete. No data was written.")
        return 0

    if not args.dst.parent.exists():
        args.dst.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nWriting DuckDB: {args.dst}")
    _write_duckdb(args.dst, rows_by_table)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
