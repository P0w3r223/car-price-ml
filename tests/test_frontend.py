"""The API serves the static valuation form (docs/app) at the site root, same-origin."""

from fastapi.testclient import TestClient

from api.main import app

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


def test_api_routes_take_precedence_over_static_mount():
    # /health must still resolve to the API route, not the catch-all static mount.
    assert client.get("/health").json()["status"] == "ok"
