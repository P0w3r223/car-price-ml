"""End-to-end tests against the real artifact, with the app's lifespan actually running.

Every other API test injects a fake model and never enters the `TestClient` context manager,
so `lifespan` never runs — which means nothing was checking that the vocabulary `save_model`
stamps is the vocabulary `_require_known` reads. That is the seam the unknown-make guard
depends on, and a fail-open there is invisible.

Skipped when no artifact is present (CI does not train), so the suite stays green without
the 14 MB model.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from car_price_ml import config, model

_MODEL_PATH = config.MODELS_DIR / model.MODEL_FILENAME

pytestmark = pytest.mark.skipif(
    not _MODEL_PATH.exists(), reason="no trained artifact — run python -m car_price_ml.train"
)

_VALID_PAYLOAD = {
    "mark": "opel", "model": "combo", "fuel": "Diesel", "province": "Mazowieckie",
    "year": 2015, "mileage": 139568, "vol_engine": 1248,
}


@pytest.fixture
def live_client():
    with TestClient(app) as client:  # entering the context manager is what runs lifespan
        yield client


def test_lifespan_loads_the_artifact(live_client):
    assert live_client.get("/health").json()["model_loaded"] is True


def test_vocabulary_comes_from_the_artifact(live_client):
    body = live_client.get("/vocabulary").json()
    # The dataset's own domains, not the code's guesses: 23 makes, 328 models.
    assert len(body["mark"]) > 1 and len(body["model"]) > 1
    assert body["fuel"] == list(config.KNOWN_FUELS)
    assert body["province"] == list(config.PROVINCES)


def test_unknown_make_is_refused_by_the_real_vocabulary(live_client):
    response = live_client.post("/predict", json={**_VALID_PAYLOAD, "mark": "ferrari"})
    assert response.status_code == 422
    assert "ferrari" in response.json()["detail"]


def test_spelling_variants_reach_the_same_valuation(live_client):
    """Case and separator differences must not change the price.

    "Opel" once returned 33 % above "opel", because the target encoder answered an unseen
    category with the dataset mean.
    """
    plain = live_client.post("/predict", json=_VALID_PAYLOAD).json()
    variants = [
        {**_VALID_PAYLOAD, "mark": "Opel"},
        {**_VALID_PAYLOAD, "fuel": "diesel"},
        {**_VALID_PAYLOAD, "province": "mazowieckie"},
    ]
    for payload in variants:
        assert live_client.post("/predict", json=payload).json() == plain
