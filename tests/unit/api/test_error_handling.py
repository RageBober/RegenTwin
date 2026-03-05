"""Тесты для structured error handling в main app."""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.main import create_app
from src.db.models import Base
from src.db.session import get_db

# Валидный UUID, которого гарантированно нет в БД
_FAKE_UUID = "00000000-0000-0000-0000-000000000000"


def _setup():  # type: ignore[no-untyped-def]
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine)

    def override_get_db():  # type: ignore[no-untyped-def]
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app = create_app()
    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app, raise_server_exceptions=False)


class TestErrorHandling:
    def test_health_ok(self) -> None:
        client = _setup()
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_404_simulation(self) -> None:
        client = _setup()
        resp = client.get(f"/api/v1/simulate/{_FAKE_UUID}")
        assert resp.status_code == 404

    def test_404_upload(self) -> None:
        client = _setup()
        resp = client.get(f"/api/v1/upload/{_FAKE_UUID}")
        assert resp.status_code == 404

    def test_404_analysis(self) -> None:
        client = _setup()
        resp = client.get(f"/api/v1/analysis/{_FAKE_UUID}")
        assert resp.status_code == 404

    def test_422_invalid_body(self) -> None:
        client = _setup()
        resp = client.post("/api/v1/simulate", json={"dt": -1})
        assert resp.status_code == 422

    def test_404_results(self) -> None:
        client = _setup()
        resp = client.get(f"/api/v1/results/{_FAKE_UUID}")
        assert resp.status_code == 404

    def test_400_invalid_uuid(self) -> None:
        """Невалидный UUID возвращает 400."""
        client = _setup()
        assert client.get("/api/v1/simulate/not-a-uuid").status_code == 400
        assert client.get("/api/v1/upload/not-a-uuid").status_code == 400
        assert client.get("/api/v1/analysis/not-a-uuid").status_code == 400
        assert client.get("/api/v1/results/not-a-uuid").status_code == 400

    def test_all_routers_mounted(self) -> None:
        """Проверить что все роутеры подключены."""
        client = _setup()
        # health
        assert client.get("/api/v1/health").status_code == 200
        # upload (GET non-existent — 404, not 405)
        assert client.get(f"/api/v1/upload/{_FAKE_UUID}").status_code == 404
        # simulate (GET non-existent — 404)
        assert client.get(f"/api/v1/simulate/{_FAKE_UUID}").status_code == 404
        # results (GET non-existent — 404)
        assert client.get(f"/api/v1/results/{_FAKE_UUID}").status_code == 404
        # analysis (GET non-existent — 404)
        assert client.get(f"/api/v1/analysis/{_FAKE_UUID}").status_code == 404
