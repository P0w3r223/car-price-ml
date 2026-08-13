"""Assemble the published page from the exported aggregates.

    python -m car_price_ml.site.build

The page leads with a finding rather than with a description of itself, and the finding is
derived from the measurements: if the bake-off stops supporting it, the headline changes with
it instead of quietly becoming false.

Two constraints shape everything here. Nothing is read except ``docs/data/*.json`` and the
project's own source, so the page is a pure function of committed inputs and CI can rebuild
it and compare — which is the only way "the page cannot disagree with the model" is a fact
rather than a habit. And no figure may publish a central number without the uncertainty it
rests on; the build fails instead of the reader finding out later.
"""

from __future__ import annotations

import inspect
import json
import math
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from markupsafe import Markup

from car_price_ml import config, data, features
from car_price_ml import model as model_module
from car_price_ml.site import AGGREGATE_SCHEMA, charts

TEMPLATE_DIR = Path(__file__).parent / "templates"
ASSET_DIR = Path(__file__).parent / "assets"

# How many hyper-parameter configurations the size↔quality sweep scored (ADR 0001). Named
# here rather than written into the sentence that quotes it: it is the last number on this
# page that a re-run could leave stale without anything noticing.
SEARCHED_CONFIGURATIONS = 16

# The rules the page shows verbatim instead of describing. This is the project's answer to
# "trust me": each one is read out of the module it lives in at build time, so the page
# cannot drift from the behaviour it claims — a paraphrase can go stale, a quotation cannot.
PUBLISHED_RULES = (
    ("data.drop_duplicate_adverts", data.drop_duplicate_adverts),
    ("data.canonical_province", data.canonical_province),
    ("data.has_plausible_displacement", data.has_plausible_displacement),
    ("features.build_preprocessor", features.build_preprocessor),
    ("model.load_model", model_module.load_model),
)


class MissingAggregate(FileNotFoundError):
    """Raised when the page is built before the aggregates it renders were exported."""


class StaleAggregate(ValueError):
    """Raised when an aggregate was written against a different shape than this build reads."""


@dataclass(frozen=True)
class Kpi:
    label: str
    value: str
    note: str


def _megabytes(size: int | None) -> str:
    if not size:
        return "—"
    return f"{size / 1e6:,.0f} MB" if size >= 100e6 else f"{size / 1e6:.1f} MB"


def _thousands(value: float) -> str:
    return f"{value:,.0f}".replace(",", " ")


def _load(name: str, data_dir: Path) -> dict:
    path = data_dir / name
    if not path.is_file():
        raise MissingAggregate(
            f"no {name} at {path} — run `python -m car_price_ml.site.export` first "
            f"(it needs the trained artifact and the dataset)"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    # The same contract `model.load_model` enforces on the artifact, one layer out. Without
    # it a shape change renders whatever fields still happen to match — or dies on a bare
    # KeyError that says nothing about why.
    if payload.get("schema") != AGGREGATE_SCHEMA:
        raise StaleAggregate(
            f"{path} is schema {payload.get('schema')!r}, this build reads "
            f"{AGGREGATE_SCHEMA} — re-run `python -m car_price_ml.site.export`"
        )
    return payload


def _runner_up(metrics: dict) -> tuple[str, int] | None:
    """The heaviest rival the served model was measured against, if any was weighed."""
    sizes = metrics["artifact_bytes"]
    weighed = [
        (name, sizes[name]) for name in metrics["bakeoff"]
        if name != metrics["served_model"] and sizes.get(name)
    ]
    return max(weighed, key=lambda pair: pair[1]) if weighed else None


def _headline(metrics: dict) -> dict:
    """The claim the page leads with, derived from the bake-off rather than asserted."""
    served = metrics["served_model"]
    served_mae = metrics["bakeoff"][served]["mae"]
    served_size = metrics["artifact_bytes"].get(served)
    rival = _runner_up(metrics)

    if rival and served_size:
        name, size = rival
        rival_mae = metrics["bakeoff"][name]["mae"]
        if size > served_size and rival_mae > served_mae:
            return {
                "claim": f"A {_megabytes(served_size)} model prices this market better "
                         f"than a {_megabytes(size)} one",
                "detail": f"{served} reaches {_thousands(served_mae)} PLN mean absolute "
                          f"error against {name}'s {_thousands(rival_mae)}, in an artifact "
                          f"{size / served_size:.0f}× smaller — so the size↔quality "
                          f"trade-off this project was built around turned out not to exist.",
            }
    return {
        "claim": f"{served} is the model this comparison chose",
        "detail": f"{_thousands(served_mae)} PLN mean absolute error over "
                  f"{metrics['cv_folds']}-fold cross-validation, in an artifact of "
                  f"{_megabytes(served_size)}.",
    }


def _accuracy_verdict(metrics: dict) -> str:
    """Does the winner's lead survive the folds — decided from the numbers, not written.

    This paragraph used to be prose, and the prose was wrong: it called a 186 PLN gap
    "inside the noise" while the chart above it drew two whiskers that do not overlap. Worse,
    the margin was typed in, so a retrain would have moved the headline, the KPI row and the
    chart while this sentence went on asserting a tie. Both halves are derived now.

    Two spreads are combined in quadrature rather than compared one at a time: each fold mean
    carries its own wobble, and the question is whether the difference between them clears
    both.
    """
    served = metrics["served_model"]
    ranked = sorted(metrics["bakeoff"].items(), key=lambda kv: kv[1]["mae"])
    if len(ranked) < 2:
        # Folded into the "did not win" branch, this rendered "X is served although it did
        # not win — the winner was X": a confident false statement about the served model,
        # which is the worst outcome available on this page.
        return (
            f"Only {served} was scored, so there is no comparison to read here — the "
            f"figure above is one measurement, not a ranking."
        )
    if ranked[0][0] != served:
        # A pinned model that lost its own bake-off. `train.py` logs that loudly; the page
        # says it plainly rather than dressing it up as a result.
        return (
            f"{served} is served although it did not win the comparison — the winner was "
            f"{ranked[0][0]} at {_thousands(ranked[0][1]['mae'])} PLN."
        )

    (_, best), (rival, runner_up) = ranked[0], ranked[1]
    gap = runner_up["mae"] - best["mae"]
    combined = math.hypot(best["mae_fold_std"], runner_up["mae_fold_std"])
    if gap <= combined:
        return (
            f"Read the whiskers before the ranking: the {_thousands(gap)} PLN lead over "
            f"{rival} is inside the folds' combined spread ({_thousands(combined)} PLN), so "
            f"on accuracy alone this is a tie and the decision rests on the size column."
        )
    return (
        f"The {_thousands(gap)} PLN lead over {rival} clears the folds' combined spread "
        f"({_thousands(combined)} PLN), so the ranking is not fold noise. It is still the "
        f"best of {SEARCHED_CONFIGURATIONS} configurations scored on the same "
        f"cross-validation, which makes the winner's own score a little optimistic — so the "
        f"decision to serve it rests on the size difference, which selection noise cannot "
        f"touch."
    )


def _kpis(metrics: dict) -> list[Kpi]:
    served = metrics["served_model"]
    scores = metrics["bakeoff"][served]
    sizes = metrics["vocabulary_sizes"]
    rival = _runner_up(metrics)
    served_size = metrics["artifact_bytes"].get(served)

    if rival and served_size and rival[1] > served_size:
        size_kpi = Kpi(
            "Smaller than the runner-up", f"{rival[1] / served_size:.0f}×",
            f"{_megabytes(served_size)} against {rival[0]}'s {_megabytes(rival[1])}",
        )
    else:
        size_kpi = Kpi("Served artifact", _megabytes(served_size), f"{served}, joblib")

    return [
        Kpi("Adverts trained on", _thousands(metrics["n_train"]),
            "after de-duplication and the documented cleaning rules"),
        Kpi("Mean absolute error", f"{_thousands(scores['mae'])} PLN",
            f"± {_thousands(scores['mae_fold_std'])} between folds, out-of-fold, in PLN"),
        size_kpi,
        Kpi("Closed input domains", str(len(sizes)),
            " · ".join(f"{field} ({count})" for field, count in sizes.items())
            + " — anything else is refused"),
    ]


def _bakeoff_bars(metrics: dict) -> list[charts.SpreadBar]:
    served = metrics["served_model"]
    ordered = sorted(metrics["bakeoff"].items(), key=lambda kv: kv[1]["mae"])
    return [
        charts.SpreadBar(
            label=name,
            value=scores["mae"],
            spread=scores.get("mae_fold_std"),
            note=_megabytes(metrics["artifact_bytes"].get(name)),
            served=name == served,
        )
        for name, scores in ordered
    ]


def _assert_figures_are_complete(figures: dict[str, str]) -> None:
    """A chart without its numbers is not publishable — fail the build, not the reader.

    Checked structurally, on the rendered markup: matching on wording instead would pass the
    day someone rephrases a label and silently ships an unlabelled chart. The comparison
    chart carries the extra condition, because it is the one a reader draws a conclusion from.
    """
    for name, markup in figures.items():
        if 'class="empty"' in markup:
            continue  # nothing to plot, and the page says so
        if 'class="bar-value"' not in markup:
            raise charts.IncompleteFigure(f"chart {name!r} would publish values without labels")
    # Unconditional: `bakeoff_chart` has no empty fallback to exempt, because a page without
    # the comparison is a different claim rather than a smaller one.
    if 'class="spread"' not in figures["bakeoff"]:
        raise charts.IncompleteFigure("the model comparison would publish MAE without its spread")


def gather(data_dir: Path | None = None) -> dict:
    """Collect everything the template needs. Reads only committed inputs."""
    data_dir = Path(data_dir or config.SITE_DATA_DIR)
    metrics = _load("metrics.json", data_dir)
    drivers = _load("drivers.json", data_dir)
    depreciation = _load("depreciation.json", data_dir)
    refusals = _load("refusals.json", data_dir)

    buckets = depreciation["buckets"]
    figures = {
        "bakeoff": charts.bakeoff_chart(_bakeoff_bars(metrics)),
        "drivers": charts.driver_chart(
            [charts.Bar(label=row["feature"], value=row["mean_abs_shap"])
             for row in drivers["features"]],
            "What moves a valuation",
        ),
        "depreciation": charts.curve_chart(
            [charts.Point(x=row["age"], y=row["median_price"], n=row["n"]) for row in buckets],
            "Median advert price by age",
            x_label="years old",
            y_label="PLN",
        ),
    }
    _assert_figures_are_complete(figures)

    served = metrics["served_model"]
    return {
        "metrics": metrics,
        "headline": _headline(metrics),
        "accuracy_verdict": _accuracy_verdict(metrics),
        "kpis": _kpis(metrics),
        "served": served,
        "served_scores": metrics["bakeoff"][served],
        "charts": {name: Markup(markup) for name, markup in figures.items()},
        "drivers": drivers,
        "depreciation": depreciation,
        "oldest_bucket": max((row["age"] for row in buckets), default=0),
        "refusals": refusals,
        "provinces": list(config.PROVINCES),
        "fuels": list(config.KNOWN_FUELS),
        "rules": [(name, inspect.getsource(target)) for name, target in PUBLISHED_RULES],
        "megabytes": _megabytes,
        "thousands": _thousands,
    }


def render(data_dir: Path | None = None) -> str:
    """Render the page to a string."""
    page = gather(data_dir)
    environment = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=True,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = environment.get_template("index.html.j2")
    return template.render(
        styles=Markup((ASSET_DIR / "styles.css").read_text(encoding="utf-8")), **page
    )


def build(out_dir: Path | None = None, data_dir: Path | None = None) -> Path:
    """Render the page and write it where GitHub Pages serves it from."""
    out_dir = Path(out_dir or config.DOCS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "index.html"
    # Explicit LF, not the platform's line ending: this file is committed and CI rebuilds it
    # on another OS, so "the page is a function of the aggregates" has to hold across both.
    with open(target, "w", encoding="utf-8", newline="\n") as page:
        page.write(render(data_dir))
    return target


if __name__ == "__main__":
    print("wrote", build())
