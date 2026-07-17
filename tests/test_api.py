"""Tests for the FastAPI service (model mocked — no trained artifact needed)."""

from fastapi.testclient import TestClient

from api.main import _state, app

client = TestClient(app)

_VALID_PAYLOAD = {
    "mark": "opel", "model": "combo", "fuel": "Diesel", "province": "Mazowieckie",
    "year": 2015, "mileage": 139568, "vol_engine": 1248,
}


class _FakeModel:
    def predict(self, df):
        return [50000.0]


def test_health_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_rejects_invalid_input():
    response = client.post("/predict", json={"mark": "opel"})  # missing required fields
    assert response.status_code == 422


def test_predict_rejects_out_of_range_year():
    bad = {**_VALID_PAYLOAD, "year": 1900}  # below the allowed range
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_returns_price(monkeypatch):
    monkeypatch.setitem(_state, "model", _FakeModel())
    response = client.post("/predict", json=_VALID_PAYLOAD)
    assert response.status_code == 200
    assert response.json()["predicted_price_pln"] == 50000.0


def test_predict_503_without_model(monkeypatch):
    monkeypatch.setitem(_state, "model", None)
    assert client.post("/predict", json=_VALID_PAYLOAD).status_code == 503
