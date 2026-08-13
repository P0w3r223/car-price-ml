"""The static valuation form: what it validates against, and what it refuses to guess.

The API serves ``docs/app`` at the site root, same-origin, so these tests read the committed
files through the service exactly as a browser would. That matters for the parity tests
below: what they check is that the *committed* ``config.json`` still matches ``config.py`` —
a regeneration nobody ran fails here rather than shipping a form validating against last
month's vocabulary.
"""

from __future__ import annotations

import json
import re

from fastapi.testclient import TestClient

from api.main import app
from car_price_ml import config
from car_price_ml.site import form

client = TestClient(app)

APP_JS = (config.SITE_APP_DIR / "app.js").read_text(encoding="utf-8")


def _code_only(source: str) -> str:
    """``source`` with its ``//`` comments removed.

    The comments are where the history lives — including the misspelling that caused the bug
    these tests guard against — so a test asking "is this value spelled here" has to ask about
    the code. Line comments only, which is all this file has; a ``//`` inside a string would
    fool this, and there is none.
    """
    kept = [re.sub(r"\s+//.*$", "", line) for line in source.splitlines()
            if not line.lstrip().startswith("//")]
    return "\n".join(kept)


APP_JS_CODE = _code_only(APP_JS)


def test_root_serves_the_form():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Value a used car" in response.text
    assert 'id="valuation"' in response.text


def test_static_assets_are_served():
    js = client.get("/app.js")
    assert js.status_code == 200
    assert "predictViaApi" in js.text  # the fetch helper is present

    assert client.get("/styles.css").status_code == 200
    # Fetched by the form at load from this same origin — if the mount stopped serving it the
    # form would refuse to start, which is loud but only in a browser.
    assert client.get("/config.json").status_code == 200


def test_the_committed_config_still_matches_the_python():
    """The generated file is the form's only source of bounds, so it must be current.

    They were not in step once: the form offered "Kujawsko-Pomorskie" while the model was
    trained on "Kujawsko-pomorskie", and one-hot encoding turns a near-miss into an all-zero
    row rather than an error — those provinces were priced with no location at all.
    """
    served = json.loads(client.get("/config.json").text)

    assert served == form.config_payload(), (
        "docs/app/config.json is stale — run `python -m car_price_ml.site.build`"
    )


def test_the_form_validates_against_the_trained_vocabularies():
    """Stated explicitly as well, so the guarantee survives a change to the generator."""
    served = json.loads(client.get("/config.json").text)

    assert served["provinces"] == list(config.PROVINCES)
    assert served["fuels"] == list(config.KNOWN_FUELS)
    assert served["reference_year"] == config.REFERENCE_YEAR
    assert served["year_max"] == config.REFERENCE_YEAR
    assert served["year_min"] == config.REFERENCE_YEAR - config.AGE_MAX
    assert served["mileage_max"] == int(config.MILEAGE_MAX)
    assert served["vol_engine_max"] == int(config.VOL_ENGINE_MAX)
    assert served["electric_fuel"] == config.ELECTRIC_FUEL


def test_the_form_keeps_no_copy_of_the_vocabulary():
    """A second copy is what drifted, so the fix is only real if no second copy exists.

    Fuel *names* are exempt: the offline heuristic carries its own per-fuel factor table,
    which is its own invention rather than a mirror of the domain — and it is not what
    validation consults.
    """
    for province in config.PROVINCES:
        assert province not in APP_JS_CODE, f"app.js still spells {province!r} for itself"


def test_the_form_keeps_no_copy_of_the_numeric_bounds():
    """Compared as numbers, not substrings: 40 is a substring of the heuristic's 400000."""
    literals = {int(match.replace("_", ""))
                for match in re.findall(r"\b\d[\d_]*\b", APP_JS_CODE)}
    bounds = {
        config.REFERENCE_YEAR,
        config.REFERENCE_YEAR - config.AGE_MAX,
        int(config.MILEAGE_MAX),
        int(config.VOL_ENGINE_MAX),
        config.MARK_MAX_LENGTH,
        config.MODEL_MAX_LENGTH,
    }
    assert not literals & bounds, (
        f"app.js restates bounds config.py owns: {sorted(literals & bounds)}"
    )


def test_the_form_reads_the_schema_the_generator_writes():
    """The last constant spelled in both languages, and the one no other guard catches.

    Bumped on the Python side alone, everything stays green — the generated file and the
    payload it is compared against both move — while the deployed form refuses to run for
    every visitor because it is still reading the previous schema.
    """
    declared = re.search(r"const CONFIG_SCHEMA = (\d+);", APP_JS_CODE)
    assert declared, "app.js no longer declares CONFIG_SCHEMA"
    assert int(declared.group(1)) == form.FORM_CONFIG_SCHEMA


def test_the_offline_heuristic_knows_every_fuel_the_form_offers():
    """Its factor table is the one fuel-name copy left, and its miss would be silent.

    ``fuelFactor[car.fuel] ?? 1.0`` prices an unlisted fuel as petrol. The number is labelled
    an estimate wherever it appears, so the blast radius is small — but a domain value
    falling through to a default is the shape of bug this project keeps finding, and here it
    costs one test to make the fallback unreachable instead of load-bearing.
    """
    table = re.search(r"const fuelFactor = \{(.*?)\};", APP_JS_CODE, re.DOTALL)
    assert table, "app.js no longer declares the heuristic's fuel factors"
    priced = set(re.findall(r"(\w+):", table.group(1)))

    assert priced == set(config.KNOWN_FUELS)


def test_the_form_ships_disabled():
    """Fail-closed, and closed from the first byte.

    The submit button is enabled by the same code path that loaded the bounds. Shipped
    enabled, a form whose config fetch failed — or whose JavaScript never ran — would look
    exactly like a working one and validate against nothing.
    """
    page = client.get("/").text
    assert re.search(r'<button[^>]*id="submit"[^>]*disabled', page), (
        "the submit button must ship disabled; only a loaded config may enable it"
    )
    assert "renderUnavailable" in APP_JS  # the refusal path exists
    assert "CONFIG = await loadConfig()" in APP_JS  # and nothing else assigns the bounds


def test_the_form_offers_no_option_the_config_did_not_put_there():
    """The dropdowns are filled from config.json; markup listing them would be a third copy."""
    page = client.get("/").text
    assert "<option" not in page


def test_api_routes_take_precedence_over_static_mount():
    # /health must still resolve to the API route, not the catch-all static mount.
    assert client.get("/health").json()["status"] == "ok"
