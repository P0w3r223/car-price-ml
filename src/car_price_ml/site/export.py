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
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from car_price_ml import config, data, features
from car_price_ml import model as model_module

SCHEMA_VERSION = 1
SHAP_SAMPLE_SIZE = 1_000
# Age buckets thinner than this are dropped from the depreciation curve rather than drawn:
# the oldest bucket in the data holds a handful of adverts, and a median over them wobbles
# by thousands of złoty — a shape the reader would take for a market effect.
MIN_BUCKET_N = 100

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
    probes.append(Probe(
        case="A known make, capitalised",
        input="mark=Opel (the dataset spells it opel)",
        rule=f"normalised to 'opel' and priced: {_pln(control)}",
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
        except Exception as error:  # noqa: BLE001 — the class is the encoder's, the fact is that it raises
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

    df = data.load_clean()
    x, y = features.prepare(df)
    fitted = model_module.train(x, y, name=served)
    vocabulary = metadata["vocabulary"]
    model_path = config.MODELS_DIR / model_module.MODEL_FILENAME

    metrics = {
        "schema": SCHEMA_VERSION,
        "served_model": served,
        "reference_year": metadata["reference_year"],
        "n_train": metadata.get("n_train", len(x)),
        "cv_folds": config.CV_FOLDS,
        "bakeoff": bakeoff,
        "artifact_bytes": _artifact_sizes(served, config.MODELS_DIR),
        "vocabulary_sizes": {field: len(values) for field, values in vocabulary.items()},
        "trained_on": date.fromtimestamp(model_path.stat().st_mtime).isoformat(),
        "commit": _git_commit(),
    }

    written = []
    for name, payload in (
        ("metrics.json", metrics),
        ("drivers.json", {"schema": SCHEMA_VERSION, "served_model": served,
                          "sample_size": min(SHAP_SAMPLE_SIZE, len(x)),
                          "unit": "log-price contribution",
                          "features": _drivers(fitted, x)}),
        ("depreciation.json", {"schema": SCHEMA_VERSION, "min_bucket_n": MIN_BUCKET_N,
                               "buckets": _depreciation(df)}),
        ("refusals.json", {"schema": SCHEMA_VERSION, "control": _CONTROL,
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
