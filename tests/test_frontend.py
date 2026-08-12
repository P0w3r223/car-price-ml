"""The API serves the static valuation form (docs/app) at the site root, same-origin."""

import re

from fastapi.testclient import TestClient

from api.main import app
from car_price_ml import config

client = TestClient(app)


def test_root_serves_the_form():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Used-car valuation" in response.text
    assert 'id="valuation"' in response.text


def test_static_assets_are_served():
    js = client.get("/app.js")
    assert js.status_code == 200
    assert "predictViaApi" in js.text  # the fetch helper is present

    css = client.get("/styles.css")
    assert css.status_code == 200


def _js_config_list(source: str, key: str) -> list[str]:
    """Extract a string array from the CONFIG object in app.js."""
    block = re.search(rf"{key}:\s*\[(.*?)\]", source, re.DOTALL)
    assert block, f"no {key} list found in app.js"
    return re.findall(r'"([^"]*)"', block.group(1))


def _js_config_number(source: str, key: str) -> int:
    """Extract a numeric CONFIG value from app.js (underscore separators allowed)."""
    match = re.search(rf"{key}:\s*([\d_]+)", source)
    assert match, f"no {key} value found in app.js"
    return int(match.group(1).replace("_", ""))


def test_form_vocabularies_match_the_python_config():
    """The form's dropdowns and the trained vocabulary must be the same strings.

    They were not: the form offered "Kujawsko-Pomorskie" while the model knew
    "Kujawsko-pomorskie", and one-hot encoding turns a near-miss into an all-zero row
    rather than an error — those two provinces were priced with no location at all. The
    form is static, so nothing but a test keeps the two lists in step.
    """
    source = client.get("/app.js").text

    assert _js_config_list(source, "provinces") == list(config.PROVINCES)
    assert _js_config_list(source, "fuels") == list(config.KNOWN_FUELS)


def test_form_bounds_match_the_python_config():
    """Same guarantee for the numeric bounds — the form duplicates five of them.

    The province bug was one symptom of a general problem: `docs/app/` restates values
    `config.py` owns, and a copy drifts the moment either side moves. Until the config is
    emitted as data the form reads, each duplicated value needs pinning.
    """
    source = client.get("/app.js").text

    assert _js_config_number(source, "referenceYear") == config.REFERENCE_YEAR
    assert _js_config_number(source, "yearMax") == config.REFERENCE_YEAR
    assert _js_config_number(source, "yearMin") == config.REFERENCE_YEAR - config.AGE_MAX
    assert _js_config_number(source, "mileageMax") == int(config.MILEAGE_MAX)
    assert _js_config_number(source, "volEngineMax") == int(config.VOL_ENGINE_MAX)


def test_api_routes_take_precedence_over_static_mount():
    # /health must still resolve to the API route, not the catch-all static mount.
    assert client.get("/health").json()["status"] == "ok"
