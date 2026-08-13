"""The published page: what it must say, and what it must refuse to publish.

These tests run against the committed aggregates in ``docs/data/`` — the same inputs CI
renders from — because the guarantee being checked is that the page cannot disagree with the
model. A test against a synthetic fixture would verify the templating and miss that entirely.
"""

from __future__ import annotations

import json
import re

import pandas as pd
import pytest
from markupsafe import escape

from car_price_ml import config
from car_price_ml import model as model_module
from car_price_ml.site import build, charts, export, form, stylesheet


@pytest.fixture(scope="module")
def page() -> str:
    return build.render()


@pytest.fixture(scope="module")
def metrics() -> dict:
    return json.loads((config.SITE_DATA_DIR / "metrics.json").read_text(encoding="utf-8"))


def _metrics(served: str, bakeoff: dict, sizes: dict) -> dict:
    return {
        "served_model": served, "bakeoff": bakeoff, "artifact_bytes": sizes,
        "cv_folds": 5, "n_train": 100_000,
        "vocabulary_sizes": {"mark": 23, "model": 328, "fuel": 6, "province": 16},
    }


# --- the headline is derived, not written ------------------------------------------------

def test_headline_states_the_size_gap_when_the_small_model_wins():
    head = build._headline(_metrics(
        "LightGBM",
        {"LightGBM": {"mae": 8_612.2}, "RandomForest": {"mae": 8_798.3}},
        {"LightGBM": 13_874_240, "RandomForest": 590_000_000},
    ))
    assert "13.9 MB" in head["claim"] and "590 MB" in head["claim"]


def test_headline_changes_when_the_measurement_changes():
    """If the heavy model ever wins, the page must stop claiming the opposite.

    The failure this guards against is a headline that was true when it was written and
    quietly false afterwards — the same class of error the page is about.
    """
    head = build._headline(_metrics(
        "RandomForest",
        {"LightGBM": {"mae": 9_500.0}, "RandomForest": {"mae": 8_798.3}},
        {"LightGBM": 13_874_240, "RandomForest": 590_000_000},
    ))
    assert "MB model prices this market better" not in head["claim"]
    assert "RandomForest" in head["claim"]


def test_headline_makes_no_size_claim_about_an_unweighed_model():
    """Ridge has no recorded artifact size, so it cannot be the rival in a size claim."""
    head = build._headline(_metrics(
        "LightGBM",
        {"LightGBM": {"mae": 8_612.2}, "Ridge": {"mae": 14_889.2}},
        {"LightGBM": 13_874_240, "Ridge": None},
    ))
    assert "—" not in head["claim"]
    assert "Ridge" not in head["claim"]


# --- the verdict about the ranking is decided from the numbers ---------------------------

def _scored(mae: float, spread: float) -> dict:
    return {"mae": mae, "mae_fold_std": spread}


def test_a_lead_inside_the_combined_spread_is_called_a_tie():
    verdict = build._accuracy_verdict(_metrics(
        "LightGBM",
        {"LightGBM": _scored(8_612.2, 71.5), "RandomForest": _scored(8_650.0, 80.7)},
        {},
    ))
    assert "tie" in verdict


def test_a_lead_clearing_the_combined_spread_is_not_called_a_tie():
    """The committed numbers land here, and the paragraph this replaced said the opposite:
    a 186 PLN gap against spreads of 72 and 81 clears their quadrature sum of 108."""
    verdict = build._accuracy_verdict(_metrics(
        "LightGBM",
        {"LightGBM": _scored(8_612.2, 71.5), "RandomForest": _scored(8_798.3, 80.7)},
        {},
    ))
    assert "tie" not in verdict
    assert "186" in verdict and "108" in verdict


def test_a_single_scored_model_is_not_reported_as_losing_to_itself():
    """Folded into the "did not win" branch this read "X is served although it did not win
    — the winner was X", which is a confident false statement about the served model."""
    verdict = build._accuracy_verdict(_metrics(
        "LightGBM", {"LightGBM": _scored(8_612.2, 71.5)}, {},
    ))
    assert "did not win" not in verdict
    assert "no comparison" in verdict


def test_the_verdict_says_so_when_the_served_model_lost_its_own_bakeoff():
    verdict = build._accuracy_verdict(_metrics(
        "RandomForest",
        {"LightGBM": _scored(8_612.2, 71.5), "RandomForest": _scored(8_798.3, 80.7)},
        {},
    ))
    assert "did not win" in verdict


# --- a number without its uncertainty is not publishable ---------------------------------

def test_a_comparison_with_nothing_to_compare_fails_the_build():
    """No "no data" fallback for this one: a page without the comparison it is built around
    is a different claim, not a smaller one."""
    with pytest.raises(charts.IncompleteFigure):
        charts.bakeoff_chart([])


def test_a_mae_without_its_fold_spread_fails_the_build():
    with pytest.raises(charts.IncompleteFigure):
        charts.bakeoff_chart([
            charts.SpreadBar(label="LightGBM", value=8_612.2, spread=None, note="13.9 MB"),
        ])


def test_the_comparison_chart_must_carry_the_spread():
    """The structural guard, not the wording: a chart may be rephrased, not stripped."""
    complete = charts.bakeoff_chart([
        charts.SpreadBar(label="LightGBM", value=8_612.2, spread=71.5, note="13.9 MB"),
    ])
    with pytest.raises(charts.IncompleteFigure):
        build._assert_figures_are_complete({"bakeoff": complete.replace("spread", "whisker")})


def test_a_chart_without_value_labels_fails_the_build():
    with pytest.raises(charts.IncompleteFigure):
        build._assert_figures_are_complete({
            "bakeoff": '<svg><line class="spread"></line></svg>',
            "drivers": "<svg><rect></rect></svg>",
        })


# --- the depreciation curve is readable in the middle, not only at its ends --------------

def _curve() -> list[charts.Point]:
    return [charts.Point(x=age, y=160_000 - age * 6_000, n=1_000 + age) for age in range(26)]


def test_the_curve_labels_every_fifth_year_with_its_median():
    """Labelling only the two ends left the whole middle unreadable: a reader could see that
    a car loses value quickly and could not tell what a ten-year-old one costs."""
    svg = charts.curve_chart(_curve(), "Median advert price by age", "years old", "PLN")
    labels = re.findall(r'class="bar-value"[^>]*>([^<]*)</text>', svg)
    assert len(labels) == 6  # ages 0, 5, 10, 15, 20, 25
    assert all("PLN" in label for label in labels)


def test_the_curve_does_not_print_bucket_sizes_on_the_axis():
    """The size rule lives in the caption. What guarantees no point rests on too few adverts
    is the export refusing a dropped bucket inside the range, not a number under each tick.
    """
    svg = charts.curve_chart(_curve(), "Median advert price by age", "years old", "PLN")
    assert "n=" not in svg


def test_the_curve_carries_a_scale_on_both_axes():
    svg = charts.curve_chart(_curve(), "Median advert price by age", "years old", "PLN")
    assert svg.count('class="grid"') >= 3
    ticks = re.findall(r'class="axis"[^>]*text-anchor="end">([^<]*)</text>', svg)
    assert "0" in ticks
    assert any(tick.replace(" ", "").replace(" ", "").isdigit() and tick != "0"
               for tick in ticks)


def test_axis_ticks_are_round_numbers_a_reader_recognises():
    """Dividing the range evenly puts gridlines at 41 574 PLN, which is worse than none."""
    assert charts._nice_step(166_295, 4) == 50_000
    assert charts._nice_step(9.5, 4) == 2.5
    assert charts._nice_step(0, 4) == 1.0


# --- the page against the code it describes ----------------------------------------------

def test_declared_domains_on_the_page_match_config(metrics):
    """The vocabulary sizes are read off the artifact; these two must not drift apart."""
    assert metrics["vocabulary_sizes"]["province"] == len(config.PROVINCES)
    assert metrics["vocabulary_sizes"]["fuel"] == len(config.KNOWN_FUELS)


def test_rules_are_quoted_rather_than_described(page):
    """The verbatim section must contain real current source, not a paraphrase of it.

    Escaped before comparing, because the quotation is rendered as page text: a bare
    ``handle_unknown="error"`` would never appear, and a test looking for it would pass only
    if the source leaked into the page unescaped.
    """
    assert str(escape('handle_unknown="error"')) in page
    assert "def drop_duplicate_adverts" in page
    assert "def load_model" in page


def test_the_served_model_named_on_the_page_is_the_one_in_the_artifact(page, metrics):
    assert metrics["served_model"] in page


def test_the_page_dates_its_valuations(page, metrics):
    """Prices are a property of the snapshot's vintage, so the page has to say which one."""
    assert str(metrics["reference_year"]) in page
    assert "snapshot" in page


def test_the_page_refuses_to_call_shap_values_zloty(page):
    assert "log-price contribution, not złoty" in page


# --- the page is a function of committed inputs ------------------------------------------

def test_rendering_twice_gives_the_same_page():
    """No wall clock anywhere: CI rebuilds the page and compares it with the committed one,
    which only works if nothing in it depends on when the build ran."""
    assert build.render() == build.render()


def test_the_written_page_uses_lf(tmp_path):
    written = build.build(out_dir=tmp_path)
    assert b"\r\n" not in written.read_bytes()


def test_building_without_the_aggregates_names_the_command_that_makes_them(tmp_path):
    with pytest.raises(build.MissingAggregate, match="site.export"):
        build.render(data_dir=tmp_path)


def test_an_aggregate_from_another_schema_is_refused_rather_than_partly_rendered(tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps({"schema": 99}), encoding="utf-8")
    with pytest.raises(build.StaleAggregate, match="site.export"):
        build.render(data_dir=tmp_path)


# --- the page states facts about the served model, so it must read them from it -----------

def test_export_refuses_an_artifact_that_cannot_date_itself():
    with pytest.raises(export.ExportError, match="trained_at"):
        export._require_stamps({"n_train": 111_018})


def test_export_refuses_an_artifact_that_does_not_know_its_row_count():
    with pytest.raises(export.ExportError, match="n_train"):
        export._require_stamps({"trained_at": "2026-08-13"})


def test_the_depreciation_curve_refuses_to_bridge_a_dropped_bucket():
    """A line through a bucket that was too thin to plot would look like measured data."""
    frame = pd.DataFrame({
        "age": [1] * 200 + [2] * 5 + [3] * 200,
        config.TARGET: [50_000.0] * 200 + [40_000.0] * 5 + [30_000.0] * 200,
    })
    with pytest.raises(export.ExportError, match=r"\[2\]"):
        export._depreciation(frame)


# --- the export refuses to publish a stale size claim -------------------------------------
# These two weigh the artifact on disk, which CI does not have (models/ is not committed).
# Skipped rather than mocked: the whole point is that a real file is measured.

needs_artifact = pytest.mark.skipif(
    not (config.MODELS_DIR / model_module.MODEL_FILENAME).is_file(),
    reason="no trained artifact — run `python -m car_price_ml.train`",
)


@needs_artifact
def test_the_export_explains_the_served_artifact_rather_than_a_fresh_fit(monkeypatch, tmp_path):
    """The bug this guards against published behaviour measured on a model nobody serves,
    while the size and date columns beside it vouched for the file on disk — and the page
    looked entirely normal. Re-inserting the refit left every other test green, so the only
    way to hold this is to make training unreachable and require the export to succeed.
    """
    def refuse(*args, **kwargs):
        raise AssertionError("the export refitted the model instead of using the artifact")

    monkeypatch.setattr(export.model_module, "train", refuse)
    written = export.export(out_dir=tmp_path)
    assert {path.name for path in written} == {
        "metrics.json", "drivers.json", "depreciation.json", "refusals.json",
    }


@needs_artifact
def test_the_export_refuses_when_the_data_no_longer_matches_the_artifact(monkeypatch, tmp_path):
    """The artifact is dated and sized from its own metadata while behaviour is measured on
    today's data. Those describe two models the moment either moves."""
    monkeypatch.setattr(export.data, "load_clean", lambda *args, **kwargs: _one_row_frame())
    with pytest.raises(export.ExportError, match="cleans to"):
        export.export(out_dir=tmp_path)


def test_an_unrelated_crash_is_not_published_as_proof_the_domain_is_closed():
    """The refusal table's right-hand column is evidence. A typo or a dtype error must not
    render there as "the encoder raises TypeError" beside a claim that the guard works.
    """
    class OnlyBreaksOnPetrol:
        def predict(self, frame):
            if frame["fuel"].iloc[0] == "Petrol":
                raise ValueError("something else went wrong entirely")
            return [35_000.0]

    with pytest.raises(export.ExportError, match="does not name the value"):
        export._refusals(OnlyBreaksOnPetrol(), {"mark": ["opel"], "fuel": [], "province": []})


def _one_row_frame() -> pd.DataFrame:
    return pd.DataFrame([{
        "mark": "opel", "model": "combo", "fuel": "Diesel", "province": "Mazowieckie",
        "age": 7, "mileage": 139_568, "vol_engine": 1_248, config.TARGET: 35_900.0,
    }])


@needs_artifact
def test_export_rejects_an_artifact_that_no_longer_matches_the_published_curve(monkeypatch):
    """The page quotes a 590 MB rival it cannot weigh on every run, so the one size it *can*
    weigh is checked. Drift there means the curve was re-measured and the page must be too."""
    monkeypatch.setitem(export.RECORDED_ARTIFACT_BYTES, "LightGBM", 1_000)
    with pytest.raises(export.ExportError, match="re-measure"):
        export._artifact_sizes("LightGBM", config.MODELS_DIR)


@needs_artifact
def test_export_refuses_a_served_model_with_no_recorded_size(monkeypatch):
    monkeypatch.setitem(export.RECORDED_ARTIFACT_BYTES, "LightGBM", None)
    with pytest.raises(export.ExportError, match="no recorded artifact size"):
        export._artifact_sizes("LightGBM", config.MODELS_DIR)


# --- the valuation form's generated inputs -----------------------------------------------

def test_the_committed_form_assets_match_the_source():
    """Locally what CI checks by rebuilding: the committed files are the generated ones.

    Without this, the guard only fires in CI — after a page and a form built from two
    different commits of the same constants have already been pushed.
    """
    committed_config = (config.SITE_APP_DIR / "config.json").read_text(encoding="utf-8")
    committed_styles = (config.SITE_APP_DIR / "styles.css").read_text(encoding="utf-8")

    assert json.loads(committed_config) == form.config_payload()
    assert committed_styles == stylesheet(*form.FORM_STYLESHEET_PARTS)


def test_the_form_assets_are_written_with_lf(tmp_path):
    for path in form.write(out_dir=tmp_path):
        assert b"\r\n" not in path.read_bytes()


def test_both_pages_are_built_from_the_same_palette():
    """One visual system, not two.

    The form used to carry its own stylesheet with its own colours and no dark mode at all,
    so following the report's "Try the valuation form" link landed the reader on what looked
    like a different site.
    """
    tokens = stylesheet("tokens.css")
    report_styles = stylesheet(*build.REPORT_STYLESHEET_PARTS)
    form_styles = stylesheet(*form.FORM_STYLESHEET_PARTS)

    assert tokens.strip() in report_styles
    assert tokens.strip() in form_styles
    assert "prefers-color-scheme: dark" in tokens
