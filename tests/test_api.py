"""FastAPI contract tests."""
from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.api.services import get_service


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


@pytest.fixture(scope="module")
def csv_bytes(raw_frame):
    buffer = io.StringIO()
    raw_frame.head(4000).to_csv(buffer, index=False)
    return buffer.getvalue().encode()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "version" in body


def test_root_points_at_docs(client):
    assert client.get("/").json()["docs"] == "/docs"


def test_schema_endpoint(client):
    body = client.get("/schema").json()
    assert body["required"] == ["date", "store", "product", "sales"]
    assert "favorita" in body["presets"]


def test_presets_endpoint(client):
    assert "canonical" in client.get("/data/presets").json()["presets"]


def test_upload_returns_validation_report(client, csv_bytes):
    response = client.post(
        "/data/upload?preset=favorita",
        files={"file": ("sample.csv", csv_bytes, "text/csv")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["rows"] > 0
    assert body["passed"] is True
    assert "date" in body["columns"]


def test_upload_rejects_unsupported_type(client):
    response = client.post("/data/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 415


def test_upload_rejects_unknown_preset(client, csv_bytes):
    response = client.post("/data/upload?preset=bogus", files={"file": ("s.csv", csv_bytes, "text/csv")})
    assert response.status_code == 400


def test_history_records_requests(client, csv_bytes):
    client.post("/data/upload?preset=favorita", files={"file": ("sample.csv", csv_bytes, "text/csv")})
    history = client.get("/predictions/history").json()
    assert history and history[0]["endpoint"] == "upload"


def test_forecast_requires_a_trained_model(client):
    service = get_service()
    if service.bundle is None:
        assert client.post("/forecast", json={"horizon": 7}).status_code == 503
    else:
        body = client.post("/forecast", json={"horizon": 7, "aggregate": True}).json()
        assert len(body["records"]) == 7
        assert body["horizon"] == 7


def test_forecast_validates_horizon_bounds(client):
    assert client.post("/forecast", json={"horizon": 0}).status_code == 422
    assert client.post("/forecast", json={"horizon": 5000}).status_code == 422


def test_unknown_dataset_id_returns_404(client):
    response = client.post("/forecast", json={"horizon": 7, "dataset_id": "does-not-exist"})
    assert response.status_code in (404, 503)
