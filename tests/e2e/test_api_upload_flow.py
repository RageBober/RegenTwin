"""E2E для POST /api/v1/upload и GET /api/v1/upload/{id}."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest


@pytest.mark.e2e
def test_upload_valid_fcs_returns_upload_id(
    http_client: httpx.Client, sample_fcs_path: Path
) -> None:
    with sample_fcs_path.open("rb") as handle:
        response = http_client.post(
            "/api/v1/upload",
            files={"file": ("sample.fcs", handle, "application/octet-stream")},
        )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert "upload_id" in payload
    assert payload["filename"] == "sample.fcs"
    assert payload["status"] in {"ready", "failed"}
    assert "created_at" in payload


@pytest.mark.e2e
def test_upload_get_status_after_upload(http_client: httpx.Client, sample_fcs_path: Path) -> None:
    with sample_fcs_path.open("rb") as handle:
        created = http_client.post(
            "/api/v1/upload",
            files={"file": ("sample.fcs", handle, "application/octet-stream")},
        )
    assert created.status_code == 200
    upload_id = created.json()["upload_id"]

    response = http_client.get(f"/api/v1/upload/{upload_id}")
    assert response.status_code == 200
    payload = response.json()
    assert payload["upload_id"] == upload_id
    assert payload["filename"] == "sample.fcs"


@pytest.mark.e2e
def test_upload_nonexistent_upload_id_returns_404(http_client: httpx.Client) -> None:
    response = http_client.get("/api/v1/upload/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


@pytest.mark.e2e
def test_upload_invalid_upload_id_format_returns_400(http_client: httpx.Client) -> None:
    response = http_client.get("/api/v1/upload/not-a-uuid")
    assert response.status_code == 400


@pytest.mark.e2e
def test_upload_plain_text_file_is_accepted_but_fails_parsing(
    http_client: httpx.Client,
) -> None:
    response = http_client.post(
        "/api/v1/upload",
        files={"file": ("notes.fcs", b"this is not a valid FCS payload", "text/plain")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "failed"
    assert payload.get("metadata", {}).get("error")
