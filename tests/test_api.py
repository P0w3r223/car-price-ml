"""Tests for the FastAPI service (model mocked — no trained artifact needed).

The lifespan never runs here, so `_state` is populated by hand. See
`test_api_integration.py` for the tests that exercise the real artifact through it.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import _state, app
from car_price_ml import config

client = TestClient(app)

_VALID_PAYLOAD = {
    "mark": "opel", "model": "combo", "fuel": "Diesel", "province": "Mazowieckie",
    "year": 2015, "mileage": 139568, "vol_engine": 1248,
}


class _FakeModel:
    def predict(self, df):
        return [50000.0]


@pytest.fixture(autouse=True)
def _loaded_vocabulary(monkeypatch):
    """A served model always carries a vocabulary — without one the service returns 503.

    Applied to every test so each one starts from a serviceable state; the tests that care
    about a missing vocabulary override it themselves.
    """
    monkeypatch.setitem(_state, "vocabulary", _VOCABULARY)


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


def test_predict_rejects_unknown_fuel():
    bad = {**_VALID_PAYLOAD, "fuel": "banana"}
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_rejects_unknown_province():
    # A province the model never saw must fail loudly: the one-hot encoder would otherwise
    # answer with an all-zero row and price the car as if it had no location.
    bad = {**_VALID_PAYLOAD, "province": "Berlin"}
    assert client.post("/predict", json=bad).status_code == 422


class _SpyModel:
    def __init__(self):
        self.seen = None

    def predict(self, df):
        self.seen = df
        return [42000.0]


_VOCABULARY = {
    "mark": ["opel", "bmw"],
    "model": ["combo", "x5"],
    "fuel": list(config.KNOWN_FUELS),
    "province": list(config.PROVINCES),
}


def test_predict_rejects_zero_displacement_on_a_combustion_car(monkeypatch):
    """Cleaning dropped these rows, so the model has never seen one.

    Asked anyway, it answered from what zero displacement meant in the data it did see —
    an EV — and priced a diesel 11 % above the same car with a real engine.
    """
    monkeypatch.setitem(_state, "model", _FakeModel())

    assert client.post(
        "/predict", json={**_VALID_PAYLOAD, "fuel": "Diesel", "vol_engine": 0}
    ).status_code == 422
    assert client.post(
        "/predict", json={**_VALID_PAYLOAD, "fuel": "Electric", "vol_engine": 0}
    ).status_code == 200


def test_predict_refuses_to_serve_when_the_vocabulary_is_missing(monkeypatch):
    """Failing open would resurrect the unknown-make bug invisibly."""
    monkeypatch.setitem(_state, "model", _FakeModel())
    monkeypatch.setitem(_state, "vocabulary", {})

    response = client.post("/predict", json={**_VALID_PAYLOAD, "mark": "ferrari"})
    assert response.status_code == 503
    assert "vocabulary" in response.json()["detail"]


def test_prediction_is_dated_to_the_training_vintage(monkeypatch):
    # The price level belongs to the data's vintage; an undated number invites the reader to
    # assume today's market, which moved in the opposite direction to general inflation.
    monkeypatch.setitem(_state, "model", _FakeModel())

    body = client.post("/predict", json=_VALID_PAYLOAD).json()
    assert body["valuation_as_of"] == config.REFERENCE_YEAR


def test_predict_rejects_a_make_the_model_never_saw(monkeypatch):
    """Unseen make/model must 422 rather than be priced at the dataset mean.

    `TargetEncoder` substitutes the global target mean for an unknown category, so this
    path returned a confident, entirely fictional number: measured against the served
    artifact, "ferrari"/"f40" and "zzzz"/"qqqq" both came back as 33,282 PLN.
    """
    monkeypatch.setitem(_state, "model", _FakeModel())

    response = client.post("/predict", json={**_VALID_PAYLOAD, "mark": "ferrari"})
    assert response.status_code == 422
    assert "ferrari" in response.json()["detail"]

    assert client.post(
        "/predict", json={**_VALID_PAYLOAD, "model": "qqqq"}
    ).status_code == 422


def test_predict_accepts_a_capitalised_make(monkeypatch):
    # "Opel" is how a human writes it; it used to be an unknown category worth +33 %.
    spy = _SpyModel()
    monkeypatch.setitem(_state, "model", spy)

    assert client.post("/predict", json={**_VALID_PAYLOAD, "mark": "Opel"}).status_code == 200
    assert spy.seen.iloc[0]["mark"] == "opel"


def test_vocabulary_is_served_in_the_artifact_s_own_order():
    # Not re-sorted: Python's collation puts Ł, Ś and Ż after Z, which would scramble the
    # province list the form renders.
    body = client.get("/vocabulary").json()
    assert body["mark"] == _VOCABULARY["mark"]
    assert body["fuel"] == list(config.KNOWN_FUELS)
    assert body["province"] == list(config.PROVINCES)


def test_vocabulary_says_nothing_about_makes_without_a_model(monkeypatch):
    # The declared domains stay (the encoder is built from them); the learned ones do not.
    monkeypatch.setitem(_state, "vocabulary", {})
    body = client.get("/vocabulary").json()
    assert body["mark"] == [] and body["model"] == []
    assert body["province"] == list(config.PROVINCES)


def test_predict_normalizes_fuel_case(monkeypatch):
    # Symmetric with the province validator: casing is absorbed, vocabulary is not.
    spy = _SpyModel()
    monkeypatch.setitem(_state, "model", spy)

    assert client.post("/predict", json={**_VALID_PAYLOAD, "fuel": "diesel"}).status_code == 200
    assert spy.seen.iloc[0]["fuel"] == "Diesel"


def test_predict_normalizes_province_spelling(monkeypatch):
    # Older clients (and humans) send "Kujawsko-Pomorskie"; the model was trained on
    # "Kujawsko-pomorskie". The boundary normalises instead of refusing.
    spy = _SpyModel()
    monkeypatch.setitem(_state, "model", spy)
    payload = {**_VALID_PAYLOAD, "province": "Kujawsko-Pomorskie"}

    assert client.post("/predict", json=payload).status_code == 200
    assert spy.seen.iloc[0]["province"] == "Kujawsko-pomorskie"


def test_predict_builds_correct_feature_row(monkeypatch):
    # Protects the inference contract: age is derived and columns match training.
    spy = _SpyModel()
    monkeypatch.setitem(_state, "model", spy)
    assert client.post("/predict", json=_VALID_PAYLOAD).status_code == 200
    row = spy.seen.iloc[0]
    assert row["age"] == config.REFERENCE_YEAR - _VALID_PAYLOAD["year"]
    assert set(spy.seen.columns) == {
        "mark", "model", "fuel", "province", "age", "mileage", "vol_engine"
    }
