"""
Генерация тестовых FCS 3.1 файлов для RegenTwin.

Запуск:
    python scripts/generate_test_fcs.py

Создаёт 3 файла в data/uploads/:
    - sample_normal.fcs      — нормальная ткань
    - sample_inflamed.fcs    — воспалённая ткань
    - sample_regenerating.fcs — регенерирующая ткань
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

# Добавляем корень проекта в path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.fixtures.mock_data import (
    generate_normal_fcs_data,
    generate_inflamed_fcs_data,
    generate_regenerating_fcs_data,
)

# Каналы с полными именами для FCS TEXT segment
CHANNEL_META = [
    ("FSC-A", "Forward Scatter-A"),
    ("FSC-H", "Forward Scatter-H"),
    ("SSC-A", "Side Scatter-A"),
    ("CD34-APC", "CD34 APC"),
    ("CD14-PE", "CD14 PE"),
    ("CD68-FITC", "CD68 FITC"),
    ("Annexin-V-Pacific Blue", "Annexin V Pacific Blue"),
]


def write_fcs(filepath: Path, df: "pd.DataFrame") -> None:
    """Записать DataFrame в формате FCS 3.1 (float32, little-endian)."""
    n_events = len(df)
    n_params = len(df.columns)

    # ── TEXT segment ──────────────────────────────────────────
    text_pairs: dict[str, str] = {
        "$BEGINANALYSIS": "0",
        "$ENDANALYSIS": "0",
        "$BEGINSTEXT": "0",
        "$ENDSTEXT": "0",
        "$MODE": "L",  # list mode
        "$DATATYPE": "F",  # float32
        "$BYTEORD": "1,2,3,4",  # little-endian
        "$PAR": str(n_params),
        "$TOT": str(n_events),
        "$CYT": "RegenTwin Mock Cytometer",
        "$DATE": "02-MAR-2026",
        "$FIL": filepath.name,
    }

    for i, (short_name, long_name) in enumerate(CHANNEL_META, start=1):
        text_pairs[f"$P{i}N"] = short_name
        text_pairs[f"$P{i}S"] = long_name
        text_pairs[f"$P{i}B"] = "32"  # bits
        text_pairs[f"$P{i}E"] = "0,0"  # no log
        text_pairs[f"$P{i}R"] = "262144"  # range

    # Сериализуем TEXT
    delimiter = "/"
    text_body = delimiter
    for k, v in text_pairs.items():
        text_body += f"{k}/{v}/"

    text_bytes = text_body.encode("ascii")

    # ── DATA segment ──────────────────────────────────────────
    # float32 little-endian
    values = df.values.astype(np.float32)
    data_bytes = values.tobytes()

    # ── HEADER (58 bytes, version + offsets) ──────────────────
    # FCS3.1 header: positions 0-57
    #   0-5:   "FCS3.1"
    #   6-9:   spaces
    #  10-17:  TEXT start offset (ascii, right-justified)
    #  18-25:  TEXT end offset
    #  26-33:  DATA start offset
    #  34-41:  DATA end offset
    #  42-49:  ANALYSIS start (0)
    #  50-57:  ANALYSIS end (0)

    header_size = 58
    text_start = header_size
    text_end = text_start + len(text_bytes) - 1
    data_start = text_end + 1
    data_end = data_start + len(data_bytes) - 1

    # Обновляем TEXT с правильными DATA offsets
    text_pairs["$BEGINDATA"] = str(data_start)
    text_pairs["$ENDDATA"] = str(data_end)

    # Пересериализуем TEXT
    text_body = delimiter
    for k, v in text_pairs.items():
        text_body += f"{k}/{v}/"
    text_bytes = text_body.encode("ascii")

    # Пересчитываем offsets
    text_end = text_start + len(text_bytes) - 1
    data_start = text_end + 1
    data_end = data_start + len(data_bytes) - 1

    # Ещё раз обновляем DATA offsets (они могли сместиться)
    text_pairs["$BEGINDATA"] = str(data_start)
    text_pairs["$ENDDATA"] = str(data_end)
    text_body = delimiter
    for k, v in text_pairs.items():
        text_body += f"{k}/{v}/"
    text_bytes = text_body.encode("ascii")

    # Финальный пересчёт
    text_end = text_start + len(text_bytes) - 1
    data_start = text_end + 1
    data_end = data_start + len(data_bytes) - 1

    header = "FCS3.1"
    header += " " * 4  # spaces
    header += f"{text_start:>8d}"
    header += f"{text_end:>8d}"
    header += f"{data_start:>8d}"
    header += f"{data_end:>8d}"
    header += f"{'0':>8s}"
    header += f"{'0':>8s}"

    header_bytes = header.encode("ascii")
    assert len(header_bytes) == 58, f"Header must be 58 bytes, got {len(header_bytes)}"

    # ── Записываем файл ──────────────────────────────────────
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        f.write(header_bytes)
        f.write(text_bytes)
        f.write(data_bytes)

    print(f"  {filepath.name}: {n_events} events, {n_params} channels, {filepath.stat().st_size:,} bytes")


def main() -> None:
    output_dir = ROOT / "data" / "uploads"

    scenarios = [
        ("sample_normal.fcs", generate_normal_fcs_data, 10000),
        ("sample_inflamed.fcs", generate_inflamed_fcs_data, 10000),
        ("sample_regenerating.fcs", generate_regenerating_fcs_data, 10000),
    ]

    print("Генерация тестовых FCS файлов...")
    for name, generator, n_events in scenarios:
        df = generator(n_events=n_events)
        write_fcs(output_dir / name, df)

    print(f"\nФайлы сохранены в {output_dir}/")
    print("Можно загрузить через UI или API: POST /api/v1/upload")


if __name__ == "__main__":
    main()
