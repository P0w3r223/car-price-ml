"""Export the small aggregates the published page is rendered from.

    python -m car_price_ml.site.export

Needs the trained artifact and the dataset, neither of which is in the repository — so this
runs locally, and its output (a few kilobytes of JSON under ``docs/data/``) is committed.
The page build then reads only those files, which is what lets CI rebuild the page from
inputs it actually has and compare the result with the committed one.

Nothing here is asserted. Every number the page prints is measured at export time, including
the ones about what the model gets wrong when its guards are removed: the refusal table is
produced by running the unguarded pipeline and recording what it answers.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from car_price_ml import config, data, features
from car_price_ml import model as model_module
from car_price_ml.site import AGGREGATE_SCHEMA, MIN_BUCKET_N

SHAP_SAMPLE_SIZE = 1_000

# Artifact sizes from the size↔quality curve measured for ADR 0001 on 2026-08-12: each model
# fit on the full cleaned dataset and dumped with joblib, sizes in decimal MB. They are
# recorded rather than re-measured on every export because weighing RandomForest means
# fitting it on all rows and writing 590 MB. The served model's size *is* measured live, and
# the export refuses to publish if the measurement disagrees with the record — so the
# recorded column cannot quietly go stale while the page keeps quoting it.
RECORDED_ARTIFACT_BYTES: dict[str, int | None] = {
    "Ridge": None,  # never measured; the page prints "—" rather than a guess
    "RandomForest": 590_000_000,
    "LightGBM": 13_900_000,
}
_SIZE_TOLERANCE = 0.05

# One car that exists in the data, used as the control in every probe below. A 2015 Opel
# Combo 1.2 diesel with 139 568 km — a real advert, listed at 35 900 PLN.
_CONTROL = {
    "mark": "opel", "model": "combo", "fuel": "Diesel", "province": "Mazowieckie",
    "age": config.REFERENCE_YEAR - 2015, "mileage": 139_568, "vol_engine": 1_248,
}


class ExportError(RuntimeError):
    """Raised when the aggregates cannot be produced from measurements alone."""


@dataclass(frozen=True)
class Probe:
    """One input the model cannot honestly price, and what each path does with it.

    ``rule`` is the verdict of the closed-domain rule the API delegates to — the same
    function, called directly. ``unguarded`` is what the fitted pipeline answers when that
    rule is removed, which is the number the service used to return.
    """

    case: str
    input: str
    rule: str
    unguarded: str
    lesson: str


def _git_commit() -> str | None:
    """The commit the aggregates were exported from, or None outside a checkout.

    Recorded here rather than read at render time on purpose: the page must be a pure
    function of the committed inputs, or CI cannot rebuild it and compare. The SHA therefore
    names the commit the data came from, which is the one before the commit that publishes it.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            # This repository, not whatever directory the export happened to be launched
            # from: run from another checkout, the page would stamp that repository's SHA.
            cwd=config.PROJECT_ROOT,
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _artifact_sizes(served: str, models_dir: Path) -> dict[str, int | None]:
    """The recorded curve, with the served model's entry verified against the file on disk."""
    path = models_dir / model_module.MODEL_FILENAME
    measured = path.stat().st_size
    recorded = RECORDED_ARTIFACT_BYTES.get(served)
    if recorded is None:
        raise ExportError(
            f"no recorded artifact size for the served model {served!r} — measure it and add "
            f"it to RECORDED_ARTIFACT_BYTES (the file on disk is {measured:,} bytes)"
        )
    if abs(measured - recorded) / recorded > _SIZE_TOLERANCE:
        raise ExportError(
            f"{served} artifact measures {measured:,} bytes but the published curve records "
            f"{recorded:,} — re-measure the curve before publishing a page that quotes it"
        )
    return {**RECORDED_ARTIFACT_BYTES, served: measured}


def _depreciation(df: pd.DataFrame) -> list[dict]:
    """Median advert price per year of age, over the cleaned data."""
    grouped = df.groupby("age")[config.TARGET].agg(["median", "size"]).reset_index()
    kept = grouped[grouped["size"] >= MIN_BUCKET_N]
    ages = [int(age) for age in kept["age"]]
    # The chart joins these points with a line, which asserts something about the interval
    # between them. Dropping a thin bucket in the middle would draw a straight segment across
    # ages nobody measured, and it would look exactly like measured data.
    gaps = sorted(set(range(min(ages), max(ages) + 1)) - set(ages)) if ages else []
    if gaps:
        raise ExportError(
            f"the depreciation curve would bridge unmeasured ages {gaps} — buckets under "
            f"n={MIN_BUCKET_N} now fall inside the range, so a line through them would "
            f"invent data. Decide how to show the gap before publishing."
        )
    return [
        {"age": int(row["age"]), "median_price": round(float(row["median"]), 2),
         "n": int(row["size"])}
        for _, row in kept.iterrows()
    ]


def _drivers(fitted, x: pd.DataFrame) -> list[dict]:
    """Mean |SHAP| per model input, in log-price units.

    Reported per *input* column rather than per transformed slot: one-hot turns `province`
    into sixteen columns, and a chart of sixteen provinces beside one `age` bar would say
    location matters least when it has merely been divided sixteen ways.
    """
    sample = x.sample(min(SHAP_SAMPLE_SIZE, len(x)), random_state=config.RANDOM_STATE)
    shap_values, _, names = model_module.shap_explanation(fitted, sample)
    per_slot = np.abs(np.asarray(shap_values)).mean(axis=0)
    # Zip would silently truncate to the shorter of the two, quietly shifting every
    # contribution onto the wrong feature — some SHAP versions append the base value as an
    # extra column, and this page would then attribute it to whatever sorted last.
    if per_slot.shape[0] != len(names):
        raise ExportError(
            f"SHAP returned {per_slot.shape[0]} columns for {len(names)} feature names — "
            f"refusing to guess which value belongs to which feature"
        )

    totals: dict[str, float] = {column: 0.0 for column in features.FEATURE_COLUMNS}
    slots_per_column: dict[str, int] = {column: 0 for column in features.FEATURE_COLUMNS}
    for name, value in zip(names, per_slot):
        # One-hot slots are named "<column>_<category>"; the encoder is fit on the declared
        # domains, so the prefix match is against a closed set rather than a guess.
        owner = next(
            (c for c in features.FEATURE_COLUMNS if name == c or name.startswith(f"{c}_")),
            None,
        )
        if owner is None:
            raise ExportError(f"SHAP slot {name!r} belongs to no known feature column")
        totals[owner] += float(value)
        slots_per_column[owner] += 1
    # A column that produced no slot would publish a measured-looking 0.0 — "this feature
    # does not matter" rather than "this feature was not explained".
    unexplained = [column for column, count in slots_per_column.items() if not count]
    if unexplained:
        raise ExportError(f"no SHAP slot matched feature columns {unexplained}")
    ordered = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    return [{"feature": name, "mean_abs_shap": round(value, 4)} for name, value in ordered]


def _predict(fitted, **overrides) -> float:
    return float(fitted.predict(pd.DataFrame([{**_CONTROL, **overrides}]))[0])


def _pln(value: float) -> str:
    """Thousands separated by a space, as everywhere else on the page and in Polish usage."""
    return f"{value:,.0f}".replace(",", " ") + " PLN"


def _refusals(fitted, vocabulary: dict[str, list[str]]) -> list[dict]:
    """Run every input the model cannot price down both paths, and record what happens.

    The point of the table this feeds is that the second column is not a hypothetical. Each
    "unguarded" cell is a number this pipeline returns right now when the closed-domain rule
    is taken out of the way — which is what the service returned before the rules existed.
    """
    control = _predict(fitted)
    probes: list[Probe] = []

    # Unknown make/model. TargetEncoder answers an unseen category with the global target
    # mean, so two entirely different fictions come back as one confident price.
    ferrari = _predict(fitted, mark="ferrari", model="f40")
    nonsense = _predict(fitted, mark="zzzz", model="qqqq")
    probes.append(Probe(
        case="A make the model was never trained on",
        input="mark=ferrari, model=f40",
        rule="refused — not in the artifact's vocabulary",
        unguarded=f"{_pln(ferrari)} — and zzzz/qqqq returns {_pln(nonsense)}",
        lesson="Both encode to the global target mean, so every unknown car is the same car.",
    ))

    # The same make spelled the way a human writes it. Casefolding is the fix; without it
    # "Opel" is simply an unknown category, priced at that same global mean.
    capitalised = _predict(fitted, mark="Opel")
    # The rule the API applies, called rather than re-implemented. A hand-copied
    # `.casefold()` here would keep publishing "normalised and priced" the day the service's
    # own normalisation changed — which is the drift this whole page exists to prevent.
    normalised = data.canonical_mark_or_model("Opel")
    if normalised is None or normalised not in vocabulary["mark"]:
        raise ExportError(
            f"'Opel' no longer normalises onto a known make ({normalised!r} is not in the "
            f"artifact's vocabulary) — the page claims it does"
        )
    probes.append(Probe(
        case="A known make, capitalised",
        input="mark=Opel (the dataset spells it opel)",
        rule=f"normalised to '{normalised}' and priced: {_pln(control)}",
        unguarded=f"{_pln(capitalised)} "
                  f"({(capitalised / control - 1):+.0%} against the same car)",
        lesson="A spelling variant is normalised; a different word is refused. Not the same thing.",
    ))

    # Combustion car with no displacement. Cleaning drops these, so the model has only ever
    # seen zero displacement on an EV — and prices the diesel as if it were one.
    no_engine = _predict(fitted, vol_engine=0)
    probes.append(Probe(
        case="A diesel with no engine",
        input="fuel=Diesel, vol_engine=0",
        rule="refused — zero displacement is only meaningful for an EV",
        unguarded=f"{_pln(no_engine)} "
                  f"({(no_engine / control - 1):+.0%} against the same car with an engine)",
        lesson="In this dataset zero displacement means 'missing', except on an EV where it is a fact.",
    ))

    # Out-of-domain one-hot values. These no longer reach a prediction at all: the encoder is
    # built on declared domains with handle_unknown="error", so the pipeline itself raises.
    for field, value, spelled in (
        ("fuel", "Petrol", "a different word for a known fuel"),
        ("province", "Berlin", "an advert from outside Poland"),
    ):
        try:
            _predict(fitted, **{field: value})
        except ValueError as error:
            # Narrow on purpose. Catching everything would publish an unrelated crash — a
            # typo in the override key, a dtype error — in the column that exists to prove
            # the guard works. The message has to name the offending value, or this is not
            # the encoder refusing an unknown category.
            if value not in str(error):
                raise ExportError(
                    f"{field}={value!r} raised {error!r}, which does not name the value — "
                    f"that is not the closed domain refusing it, and the page must not say "
                    f"it is"
                ) from error
            unguarded = f"the encoder raises {type(error).__name__}"
        else:
            raise ExportError(
                f"{field}={value!r} was priced instead of raising — the declared one-hot "
                f"domain is no longer closed, and the page must not claim it is"
            )
        probes.append(Probe(
            case=f"An unknown {field}",
            input=f"{field}={value} ({spelled})",
            rule=f"refused — not one of the {len(vocabulary[field])} declared values",
            unguarded=unguarded,
            lesson="Under handle_unknown='ignore' this would have been an all-zero block "
                   "and a confident price.",
        ))

    # The variant that started all of this: correct province, wrong capitalisation.
    variant = "Kujawsko-Pomorskie"
    canonical = data.canonical_province(variant)
    if canonical is None:
        raise ExportError(f"{variant!r} no longer normalises — the page claims it does")
    probes.append(Probe(
        case="A province spelled the common way",
        input=f"province={variant}",
        rule=f"normalised to '{canonical}' and priced",
        unguarded="an all-zero location block, priced as if the car had no province",
        lesson="This shipped: the web form sent this spelling, and ~7 % of the market was "
               "valued with no location at all.",
    ))
    return [asdict(probe) for probe in probes]


def _require_stamps(metadata: dict) -> None:
    """The page states these as facts about the served model, so it must read them from it.

    Both used to have a plausible substitute available — today's row count for ``n_train``,
    the artifact file's mtime for the training date — and both would have been wrong in
    exactly the case that matters: an artifact older than the data beside it. An mtime is
    the date the file was last copied, not fit.
    """
    for field in ("n_train", "trained_at"):
        if not metadata.get(field):
            raise ExportError(
                f"the artifact carries no {field}, so the page cannot state it — this bundle "
                f"predates the stamp; retrain with `python -m car_price_ml.train`"
            )


def export(out_dir: Path | None = None) -> list[Path]:
    """Measure everything the page prints and write it as JSON. Returns the paths written."""
    out_dir = Path(out_dir or config.SITE_DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = model_module.load_model()
    metadata = bundle["metadata"]
    served = metadata["model"]
    bakeoff = metadata.get("cv_all")
    if not bakeoff:
        raise ExportError(
            "the artifact carries no cross-validation metadata, so the comparison the page "
            "is built around cannot be measured — retrain with `python -m car_price_ml.train`"
        )

    # The served artifact itself, not a fresh fit of the same configuration. Refitting here
    # published behaviour — every refusal price, every SHAP value — measured on a model
    # nobody is running, while the size and date columns beside it vouched for the file on
    # disk. `load_model` checks the feature spec, the age anchor and the vocabularies, but
    # not which rows a model saw, so the two could differ with nothing looking wrong.
    fitted = bundle["model"]
    _require_stamps(metadata)

    df = data.load_clean()
    x, _ = features.prepare(df)
    vocabulary = metadata["vocabulary"]

    # The page dates and sizes the artifact while measuring its behaviour against the dataset
    # as it cleans *today*. Those are two different things the moment either moves, and the
    # result would be one page describing two datasets with nothing looking wrong. The row
    # count is the cheapest thing that notices.
    if metadata["n_train"] != len(x):
        raise ExportError(
            f"the artifact was fit on {metadata['n_train']:,} rows but the dataset now "
            f"cleans to {len(x):,} — the page would date and size one model while measuring "
            f"another. Retrain with `python -m car_price_ml.train`."
        )

    metrics = {
        "schema": AGGREGATE_SCHEMA,
        "served_model": served,
        "reference_year": metadata["reference_year"],
        "n_train": metadata["n_train"],
        "cv_folds": config.CV_FOLDS,
        "bakeoff": bakeoff,
        "artifact_bytes": _artifact_sizes(served, config.MODELS_DIR),
        "vocabulary_sizes": {field: len(values) for field, values in vocabulary.items()},
        "trained_on": metadata["trained_at"],
        "commit": _git_commit(),
    }

    written = []
    for name, payload in (
        ("metrics.json", metrics),
        ("drivers.json", {"schema": AGGREGATE_SCHEMA, "served_model": served,
                          "sample_size": min(SHAP_SAMPLE_SIZE, len(x)),
                          "unit": "log-price contribution",
                          "features": _drivers(fitted, x)}),
        ("depreciation.json", {"schema": AGGREGATE_SCHEMA, "min_bucket_n": MIN_BUCKET_N,
                               "buckets": _depreciation(df)}),
        ("refusals.json", {"schema": AGGREGATE_SCHEMA, "control": _CONTROL,
                           "probes": _refusals(fitted, vocabulary)}),
    ):
        path = out_dir / name
        # Explicit LF and a trailing newline: these files are committed and CI reads them on
        # another OS, so their bytes have to be the same there.
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        written.append(path)
    return written


if __name__ == "__main__":
    for path in export():
        print("wrote", path)
    # The browser's copy of the model goes out with the page's numbers: both are measured from
    # the same artifact, and publishing one without the other would put a page describing one
    # model beside a form running another.
    from car_price_ml.site import browser_model

    for path in browser_model.export():
        print("wrote", path)
