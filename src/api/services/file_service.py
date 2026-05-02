"""Service for handling uploaded files."""

from __future__ import annotations

import re
import shutil
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import UploadFile
from loguru import logger
from sqlalchemy.orm import Session

from src.api.config import settings
from src.db.models import UploadRecord


class FileSizeExceededError(Exception):
    """Raised when the uploaded file exceeds the configured size limit."""


class FileService:
    """Save uploaded files and extract FCS-derived metadata."""

    def __init__(self, db: Session) -> None:
        self._db = db
        self._upload_dir = Path(settings.upload_dir)

    def save_upload(self, file: UploadFile, description: str | None = None) -> UploadRecord:
        """Save an upload atomically and create its DB record."""
        upload_id = str(uuid.uuid4())

        original_name = file.filename or "unknown"
        original_name = re.sub(r"[/\\]", "_", original_name)
        original_name = original_name.lstrip(".")
        if not original_name:
            original_name = "unknown"
        safe_suffix = Path(original_name).suffix.lower()
        safe_filename = f"{upload_id}{safe_suffix}"

        max_bytes = settings.max_upload_bytes
        total_written = 0

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / safe_filename
            with open(tmp_path, "wb") as handle:
                while chunk := file.file.read(8192):
                    total_written += len(chunk)
                    if total_written > max_bytes:
                        raise FileSizeExceededError(
                            f"File exceeds maximum size of {max_bytes // (1024 * 1024)} MB"
                        )
                    handle.write(chunk)

            dest_dir = self._upload_dir / upload_id
            dest_path = dest_dir / safe_filename
            record = UploadRecord(
                id=upload_id,
                filename=original_name,
                file_path=str(dest_path),
                description=description,
            )
            self._db.add(record)

            try:
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(tmp_path), str(dest_path))
                self._populate_upload_metadata(record)
                self._db.commit()
                self._db.refresh(record)
            except Exception:
                self._db.rollback()
                if dest_path.exists():
                    dest_path.unlink(missing_ok=True)
                if dest_dir.exists() and not any(dest_dir.iterdir()):
                    dest_dir.rmdir()
                raise

        logger.info(f"File saved: {original_name} -> {dest_path} ({total_written} bytes)")
        return record

    def _populate_upload_metadata(self, record: UploadRecord) -> None:
        """Populate upload metadata without committing the session."""
        suffix = Path(record.filename).suffix.lower()
        if suffix == ".fcs":
            self._populate_fcs_metadata(record)
        elif suffix == ".csv":
            self._populate_csv_metadata(record)
        else:
            record.status = "ready"

    def _populate_fcs_metadata(self, record: UploadRecord) -> None:
        try:
            from src.data.fcs_parser import FCSLoader

            loader = FCSLoader()
            loader.load(record.file_path)
            meta = loader.get_metadata()
            extraction = self._extract_initial_conditions(loader, meta.n_events, record.filename)

            record.metadata_json = {
                "kind": "fcs",
                "n_events": meta.n_events,
                "n_channels": meta.n_channels,
                "channels": meta.channels,
                "cytometer": meta.cytometer,
                "fcs_version": meta.fcs_version,
                **extraction,
            }
            record.status = "ready"
        except (OSError, ValueError, RuntimeError, KeyError, AttributeError) as exc:
            logger.warning(f"FCS parsing failed for {record.filename}: {exc}")
            record.status = "failed"
            record.metadata_json = {"error": str(exc)}

    def _populate_csv_metadata(self, record: UploadRecord) -> None:
        """Validate a time-series CSV and capture its schema in metadata."""
        import pandas as pd

        try:
            df = pd.read_csv(record.file_path)
        except (OSError, ValueError, UnicodeDecodeError, pd.errors.ParserError) as exc:
            logger.warning(f"CSV parsing failed for {record.filename}: {exc}")
            record.status = "failed"
            record.metadata_json = {"error": f"Cannot read CSV: {exc}"}
            return

        if "time" not in df.columns:
            record.status = "failed"
            record.metadata_json = {
                "error": "CSV must contain a 'time' column (hours).",
                "columns": list(df.columns),
            }
            return

        variable_columns = [c for c in df.columns if c != "time"]
        if not variable_columns:
            record.status = "failed"
            record.metadata_json = {
                "error": "CSV must contain at least one variable column besides 'time'.",
            }
            return

        record.metadata_json = {
            "kind": "csv",
            "n_events": int(len(df)),
            "n_channels": len(variable_columns),
            "channels": variable_columns,
            "time_min": float(df["time"].min()),
            "time_max": float(df["time"].max()),
        }
        record.status = "ready"

    def _extract_initial_conditions(
        self,
        loader,
        n_events: int,
        filename: str,
    ) -> dict:
        """Derive simulation-ready initial conditions from the uploaded FCS file."""
        from src.data.gating import GatingStrategy
        from src.data.parameter_extraction import (
            ExtendedModelParameters,
            ModelParameters,
            ParameterExtractor,
        )

        strategy = GatingStrategy()
        extractor = ParameterExtractor()

        required_standard = [
            strategy.DEFAULT_CHANNELS[key]
            for key in ["fsc_area", "fsc_height", "ssc_area", "cd34", "cd14", "cd68", "annexin"]
        ]
        required_extended = required_standard + [
            strategy.DEFAULT_CHANNELS["cd66b"],
            strategy.DEFAULT_CHANNELS["cd31"],
        ]

        def has_channels(required: list[str]) -> bool:
            try:
                loader.validate_required_channels(required)
                return True
            except ValueError:
                return False

        gating_summary: dict | None = None
        if has_channels(required_extended):
            df = loader.to_dataframe()
            gating_results = strategy.apply_extended(df)
            extended = extractor.extract_extended(gating_results, source_file=filename)
            basic = extended.to_basic_parameters()
            parameter_source = "extended_gating"
            gating_summary = gating_results.get_statistics()
        elif has_channels(required_standard):
            df = loader.to_dataframe()
            gating_results = strategy.apply(df)
            basic = extractor.extract(gating_results, source_file=filename)
            extended = ExtendedModelParameters.from_basic_parameters(basic)
            parameter_source = "basic_gating"
            gating_summary = gating_results.get_statistics()
        else:
            basic = ModelParameters(
                n0=max(n_events / 50.0, 1.0),
                stem_cell_fraction=0.05,
                macrophage_fraction=0.08,
                apoptotic_fraction=0.02,
                c0=10.0,
                inflammation_level=0.35,
                source_file=filename,
                total_events=n_events,
            )
            extended = ExtendedModelParameters.from_basic_parameters(basic)
            parameter_source = "metadata_heuristic"

        return {
            "parameter_source": parameter_source,
            "initial_conditions": self._extended_to_request_initial_conditions(extended),
            "basic_parameters": basic.to_dict(),
            "extended_parameters": extended.to_dict(),
            "gating_summary": gating_summary,
        }

    @staticmethod
    def _extended_to_request_initial_conditions(extended) -> dict[str, float]:
        """Convert extracted extended parameters to SimulationRequest field names."""
        return {
            "P0": float(extended.P0),
            "Ne0": float(extended.Ne0),
            "M1_0": float(extended.M1_0),
            "M2_0": float(extended.M2_0),
            "F0": float(extended.F0),
            "Mf0": float(extended.Mf0),
            "E0": float(extended.E0),
            "S0": float(extended.S0),
            "C_TNF0": float(extended.C_TNF),
            "C_IL10_0": float(extended.C_IL10),
            "D0": float(extended.D),
            "O2_0": float(extended.O2),
        }

    def get_upload(self, upload_id: str) -> UploadRecord | None:
        """Get upload record by id."""
        return self._db.get(UploadRecord, upload_id)
