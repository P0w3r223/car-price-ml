"""One fixture, three implementations of the same model — held to each other.

The browser now runs the served model instead of a heuristic, which means the project has
three paths from a car to a price: the Python pipeline, the exported payload, and
``docs/app/predict.js``. Ports drift silently — a wrong column order does not raise, it prices
a different car — so all three answer the same golden cases and are compared here.

The fixture is written by ``site.browser_model`` from the pipeline, which is the definition of
correct; the other two are verified against something they did not produce. It carries the
cases a sample never contains: both ends of every bound, one car per province and per fuel, an
EV with no displacement, values sitting exactly on a split threshold, and the four inputs that
must be refused rather than priced.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from car_price_ml import config, features
from car_price_ml import model as model_module
from car_price_ml.site import browser_model

FIXTURE_PATH = browser_model.PARITY_FIXTURE
EXPORT_PATH = config.SITE_APP_DIR / browser_model.MODEL_FILENAME
RUNNER = config.PROJECT_ROOT / "tests" / "browser_parity_runner.js"

needs_fixture = pytest.mark.skipif(
    not FIXTURE_PATH.is_file() or not EXPORT_PATH.is_file(),
    reason="no browser model exported — run `python -m car_price_ml.site.browser_model`",
)
needs_artifact = pytest.mark.skipif(
    not (config.MODELS_DIR / model_module.MODEL_FILENAME).is_file(),
    reason="no trained artifact",
)
needs_node = pytest.mark.skipif(
    shutil.which("node") is None,
    reason="needs node to run docs/app/predict.js — installed in CI",
)


@pytest.fixture(scope="module")
def fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def payload() -> dict:
    return json.loads(EXPORT_PATH.read_text(encoding="utf-8"))


def _priced(fixture: dict) -> list[dict]:
    return [case for case in fixture["cases"] if "expected_pln" in case]


def _priced_via_api(fixture: dict) -> list[dict]:
    """The API takes whole kilometres and whole years, so a car sitting exactly on a split
    threshold cannot be expressed as a request. Those cases carry ``api: false``; their integer
    neighbours, which straddle the same branch, are asked of every surface."""
    return [case for case in _priced(fixture) if case.get("api") is not False]


def _refused(fixture: dict) -> list[dict]:
    return [case for case in fixture["cases"] if "refused" in case]


# --- the fixture itself -------------------------------------------------------------------

@needs_fixture
def test_the_fixture_covers_the_cases_a_sample_cannot(fixture):
    """Sampled adverts alone would leave the interesting failures untested."""
    names = " ".join(case["case"] for case in fixture["cases"])

    for field in config.NUMERIC_FEATURES:
        # Per field, not as one substring: the fixture could otherwise lose every vol_engine
        # boundary and still read as complete.
        assert f"{field} exactly on a split threshold" in names  # `<` instead of `<=`
    assert "EV" in names                            # the one fuel allowed zero displacement
    for province in config.PROVINCES:
        assert province in names                    # every one-hot slot exercised
    for fuel in config.KNOWN_FUELS:
        assert f"fuel {fuel}" in names
    assert len(_refused(fixture)) == 4


@needs_fixture
def test_every_case_has_its_own_name(fixture):
    """The two runtime comparisons below key results by case name; a duplicate would hide one."""
    names = [case["case"] for case in fixture["cases"]]

    assert len(set(names)) == len(names)


@needs_fixture
def test_the_page_and_the_form_describe_the_same_model(fixture, payload):
    """The published page's numbers and the browser's model must come from one artifact.

    Both files are committed and neither can be regenerated in CI, so nothing else notices if
    one is republished without the other — and the failure is a site whose headline describes
    one model while the form beside it announces another. Comparing the two committed files
    needs neither artifact nor dataset, so this check runs everywhere.
    """
    metrics = json.loads((config.SITE_DATA_DIR / "metrics.json").read_text(encoding="utf-8"))

    assert payload["served_model"] == metrics["served_model"]
    assert payload["trained_at"] == metrics["trained_on"]
    assert payload["n_train"] == metrics["n_train"]
    assert fixture["served_model"] == metrics["served_model"]


@needs_fixture
def test_the_fixture_was_written_for_the_model_that_is_exported(fixture, payload):
    """Both files come out of one command; comparing them catches a half-run publish."""
    assert fixture["served_model"] == payload["served_model"]
    assert fixture["trained_at"] == payload["trained_at"]
    assert fixture["schema"] == payload["schema"] == browser_model.BROWSER_MODEL_SCHEMA


# --- leg 1: the pipeline still answers what the fixture recorded ---------------------------

@needs_fixture
@needs_artifact
def test_the_pipeline_still_prices_the_fixture_as_recorded(fixture):
    """If a retrain moved these, the export and the fixture are stale together — regenerate.

    This is the leg that gives the other two their meaning: without it they could agree
    perfectly with each other about a model nobody serves.
    """
    fitted = model_module.load_model()["model"]
    cases = _priced(fixture)
    frame = pd.DataFrame([case["car"] for case in cases])[list(features.FEATURE_COLUMNS)]
    predicted = fitted.predict(frame)

    worst = max(abs(float(price) - case["expected_pln"])
                for price, case in zip(predicted, cases))
    assert worst <= fixture["tolerance_pln"], (
        f"the pipeline has moved {worst:,.4f} PLN away from the fixture — re-run "
        f"`python -m car_price_ml.site.browser_model`"
    )


# --- leg 2: the exported payload ----------------------------------------------------------

@needs_fixture
def test_the_exported_model_prices_the_fixture_identically(fixture, payload):
    """Reads only the committed export — no artifact, no dataset, the way a browser sees it."""
    worst, worst_case = 0.0, None
    for case in _priced(fixture):
        difference = abs(browser_model.predict(payload, case["car"]) - case["expected_pln"])
        if difference > worst:
            worst, worst_case = difference, case["case"]

    assert worst <= fixture["tolerance_pln"], (
        f"the exported model disagrees with the pipeline by {worst:,.4f} PLN on {worst_case!r}"
    )


@needs_fixture
@pytest.mark.parametrize("index", range(4))
def test_the_exported_model_refuses_what_it_cannot_price(fixture, payload, index):
    case = _refused(fixture)[index]
    with pytest.raises(browser_model.BrowserExportError, match=case["refused"]):
        browser_model.predict(payload, case["car"])


# --- leg 3: the JavaScript runtime, under Node --------------------------------------------

@pytest.fixture(scope="module")
def javascript(fixture) -> dict:
    """Run the fixture through docs/app/predict.js exactly as the browser loads it."""
    finished = subprocess.run(
        ["node", str(RUNNER)], cwd=config.PROJECT_ROOT,
        capture_output=True, text=True, timeout=300, check=False,
    )
    assert finished.returncode == 0, f"the runner failed:\n{finished.stderr}"
    return json.loads(finished.stdout)


@needs_fixture
@needs_node
def test_the_javascript_runtime_prices_the_fixture_identically(fixture, javascript):
    """The claim the whole export exists to support: the page answers what the API answers."""
    answered = {result["case"]: result for result in javascript["results"]}
    worst, worst_case = 0.0, None
    for case in _priced(fixture):
        result = answered[case["case"]]
        assert "price" in result, f"{case['case']!r} did not price: {result}"
        difference = abs(result["price"] - case["expected_pln"])
        if difference > worst:
            worst, worst_case = difference, case["case"]

    assert worst <= fixture["tolerance_pln"], (
        f"predict.js disagrees with the pipeline by {worst:,.4f} PLN on {worst_case!r}"
    )


@needs_fixture
@needs_node
def test_the_javascript_runtime_refuses_what_it_cannot_price(fixture, javascript):
    answered = {result["case"]: result for result in javascript["results"]}
    for case in _refused(fixture):
        assert answered[case["case"]].get("refused") == case["refused"], (
            f"{case['case']!r} was not refused by predict.js: {answered[case['case']]}"
        )


@needs_fixture
@needs_node
def test_the_javascript_runtime_loaded_the_model_that_was_exported(javascript, payload):
    assert javascript["servedModel"] == payload["served_model"]
    assert javascript["treeCount"] == len(payload["trees"]["roots"])


# --- leg 4: the API, which is the other thing a reader can reach ---------------------------

@needs_fixture
@needs_artifact
def test_the_api_prices_the_fixture_identically(fixture):
    """Same cars, through the service. Its rounding is to the grosz, so compare at that scale."""
    client = TestClient(browser_model_api())
    with client:
        for case in _priced_via_api(fixture):
            car = dict(case["car"])
            request = {**car, "year": config.REFERENCE_YEAR - int(car.pop("age"))}
            response = client.post("/predict", json=request)
            assert response.status_code == 200, f"{case['case']}: {response.text}"
            served = response.json()["predicted_price_pln"]
            assert abs(served - case["expected_pln"]) <= 0.01, case["case"]


@needs_fixture
@needs_artifact
def test_the_api_refuses_what_the_browser_refuses(fixture):
    """The two surfaces must draw the same boundary, or one of them is guessing."""
    client = TestClient(browser_model_api())
    with client:
        for case in _refused(fixture):
            car = dict(case["car"])
            request = {**car, "year": config.REFERENCE_YEAR - int(car.pop("age"))}
            response = client.post("/predict", json=request)
            assert response.status_code == 422, (
                f"{case['case']}: the API answered {response.status_code}, "
                f"the browser refuses it"
            )


def browser_model_api():
    """Imported lazily so this module stays usable without the API's dependencies."""
    from api.main import app

    return app
