"""Export the served model so a browser can run it — and refuse to export one it would run
differently.

    python -m car_price_ml.site.browser_model

The published form has always had to answer without an API behind it, and until now it did so
with a heuristic formula that was not the model and said so. This ships the model itself:
1 200 trees as parallel arrays, plus the preprocessing expressed as *data* rather than as
JavaScript.

That second half is the point. Hand-porting a `ColumnTransformer` means hand-porting column
order, `TargetEncoder`'s lookup and one-hot's category order, and every one of those fails
silently — a wrong column order does not raise, it prices a different car. So Python emits an
ordered plan of steps, the runtime walks it, and the ordering is something the export can
check rather than something a reader has to trust. `_assert_plan_matches` compares the columns
the plan produces against the fitted transformer's own `get_feature_names_out()`, and
`_assert_agrees_with_pipeline` re-prices a sample of real adverts through the exported payload
and refuses to write if any answer differs from the pipeline's by more than a grosz.

Two things the export deliberately does not carry:

- **No unseen-category fallback.** `TargetEncoder` answers an unknown make with the global
  target mean, which is where "ferrari/f40 and zzzz/qqqq return the same price" came from. The
  exported maps are the vocabulary; the runtime raises on a miss, as the API answers 422.
- **No rounding a reader would notice.** Leaf values are log-price contributions summed over
  1 200 trees and then inverted with `expm1`, so a per-node rounding error is multiplied by
  the tree count and then by the price. Measured on 200 adverts: at 6 decimals the browser and
  the API disagree by up to 3.96 PLN, at 12 decimals by 4e-06 PLN. The extra 240 KB is the
  price of the parity claim being unqualified.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from car_price_ml import config, data, features
from car_price_ml import model as model_module

# Bump when the payload's shape changes. The runtime refuses anything else rather than
# interpreting the fields it recognises — the same contract as the artifact and config.json.
BROWSER_MODEL_SCHEMA = 1

MODEL_FILENAME = "model.json"

# The golden fixture the three implementations are held to: the Python pipeline, this export
# and `docs/app/predict.js` under Node. It lives with the tests rather than under `docs/`
# because nothing serves it — but it is a measurement, so it is written by the same command
# that needs the artifact, not maintained by hand.
PARITY_FIXTURE = config.PROJECT_ROOT / "tests" / "fixtures" / "browser_parity.json"

# See the module docstring: thresholds are compared against feature values, leaves are summed.
THRESHOLD_DECIMALS = 6
LEAF_DECIMALS = 12

# The self-check before writing: how many real adverts are re-priced through the payload, and
# how far the answer may differ from the pipeline's. A grosz is far above the measured 4e-06
# PLN and far below anything a reader could see.
PARITY_SAMPLE = 200
PARITY_TOLERANCE_PLN = 0.01

_LEAF = -1  # `feature[i] == _LEAF` marks a leaf, whose answer is in `value[i]`


class BrowserExportError(RuntimeError):
    """Raised when the model cannot be exported as something the runtime would run identically."""


def _plan(fitted) -> list[dict]:
    """The preprocessing, as an ordered list of steps that each append columns.

    Read off the *fitted* transformers rather than assembled from `config`: what the browser
    must reproduce is the pipeline that exists, not the one the configuration describes.
    """
    prep = fitted.regressor_.named_steps["prep"]
    transformers = prep.named_transformers_
    target_encoder = transformers["target_enc"]
    one_hot = transformers["onehot"]

    steps: list[dict] = []
    for field, categories, encodings in zip(
        config.HIGH_CARD_CATEGORICAL, target_encoder.categories_, target_encoder.encodings_
    ):
        steps.append({
            "kind": "target_encode",
            "field": field,
            # An object, not two parallel lists: a lookup that cannot be misaligned.
            "map": {str(category): float(encoding)
                    for category, encoding in zip(categories, encodings)},
        })
    for field, categories in zip(config.LOW_CARD_CATEGORICAL, one_hot.categories_):
        steps.append({"kind": "one_hot", "field": field,
                      "categories": [str(category) for category in categories]})
    for field in config.NUMERIC_FEATURES:
        steps.append({"kind": "numeric", "field": field})
    return steps


def _plan_columns(plan: list[dict]) -> list[str]:
    """The column names the plan produces, in order — in the transformer's own spelling."""
    names: list[str] = []
    for step in plan:
        if step["kind"] == "one_hot":
            names.extend(f"{step['field']}_{category}" for category in step["categories"])
        else:
            names.append(step["field"])
    return names


def _assert_plan_matches(plan: list[dict], fitted) -> None:
    """The plan must produce exactly the columns the fitted transformer does, in its order.

    A mismatch here is the failure mode that has no symptom: the trees would be walked with
    `province` where they expect `mileage`, and every prediction would be a confident number
    about a different car.
    """
    expected = model_module.transformed_feature_names(fitted)
    produced = _plan_columns(plan)
    if produced != expected:
        raise BrowserExportError(
            f"the exported plan produces {len(produced)} columns and the fitted preprocessor "
            f"{len(expected)}; first divergence at index "
            f"{next((i for i, (a, b) in enumerate(zip(produced, expected)) if a != b), 'end')} "
            f"— refusing to publish a model the runtime would feed in a different order"
        )


def _flatten(booster) -> dict:
    """The trees as parallel arrays, refusing any structure the runtime does not implement."""
    dumped = booster.dump_model()
    feature: list[int] = []
    threshold: list[float] = []
    left: list[int] = []
    right: list[int] = []
    value: list[float] = []
    roots: list[int] = []

    def flatten(node: dict) -> int:
        index = len(feature)
        feature.append(_LEAF), threshold.append(0.0), left.append(-1), right.append(-1)
        value.append(0.0)
        if "leaf_value" in node:
            value[index] = round(float(node["leaf_value"]), LEAF_DECIMALS)
            return index
        # The runtime implements `value <= threshold`, and nothing else. A model containing a
        # categorical split or a missing-value branch would still export cleanly and then be
        # walked wrongly, so it is refused here instead.
        if node["decision_type"] != "<=":
            raise BrowserExportError(
                f"split uses decision_type {node['decision_type']!r}; the runtime implements "
                f"'<=' only"
            )
        if node.get("missing_type") not in (None, "None"):
            raise BrowserExportError(
                f"split declares missing_type {node['missing_type']!r}; the runtime has no "
                f"missing-value branch because cleaning leaves no missing values"
            )
        feature[index] = int(node["split_feature"])
        threshold[index] = round(float(node["threshold"]), THRESHOLD_DECIMALS)
        left[index] = flatten(node["left_child"])
        right[index] = flatten(node["right_child"])
        return index

    for tree in dumped["tree_info"]:
        roots.append(flatten(tree["tree_structure"]))

    if dumped.get("average_output"):
        raise BrowserExportError(
            "the booster averages its trees; the runtime sums them, as this objective requires"
        )
    return {"feature": feature, "threshold": threshold, "left": left, "right": right,
            "value": value, "roots": roots}


def encode_car(plan: list[dict], car: dict) -> list[float]:
    """One car as the model's feature row, following the plan the browser follows.

    The reference implementation of what `docs/app/predict.js` does, kept in Python so the
    export can check itself against the real pipeline before publishing anything.
    """
    row: list[float] = []
    for step in plan:
        field = step["field"]
        if field not in car:
            raise BrowserExportError(f"car has no {field!r}")
        value = car[field]
        if step["kind"] == "target_encode":
            encoded = step["map"].get(str(value))
            if encoded is None:
                # No fallback, deliberately: the global target mean here is what priced an
                # unknown make as a real car.
                raise BrowserExportError(f"unknown {field}: {value!r}")
            row.append(encoded)
        elif step["kind"] == "one_hot":
            if str(value) not in step["categories"]:
                raise BrowserExportError(f"unknown {field}: {value!r}")
            row.extend(1.0 if str(value) == category else 0.0
                       for category in step["categories"])
        else:
            row.append(float(value))
    return row


def predict(payload: dict, car: dict) -> float:
    """Price one car from the exported payload alone — the runtime, in Python."""
    row = encode_car(payload["plan"], car)
    trees = payload["trees"]
    total = 0.0
    for root in trees["roots"]:
        node = root
        while trees["feature"][node] != _LEAF:
            node = (trees["left"][node] if row[trees["feature"][node]] <= trees["threshold"][node]
                    else trees["right"][node])
        total += trees["value"][node]
    return math.expm1(total)


def _assert_agrees_with_pipeline(payload: dict, fitted) -> float:
    """Re-price real adverts through the payload and refuse to publish a disagreement.

    This is the whole reason the export is allowed to exist: a second implementation of a
    pipeline is a second answer to the same question unless something forces them together.
    Returns the worst difference seen, so the caller can report it.
    """
    frame = data.load_clean().sample(PARITY_SAMPLE, random_state=config.RANDOM_STATE)
    x, _ = features.prepare(frame)
    expected = fitted.predict(x)

    worst = 0.0
    for position, (_, car) in enumerate(x.iterrows()):
        difference = abs(predict(payload, car.to_dict()) - float(expected[position]))
        worst = max(worst, difference)
    if worst > PARITY_TOLERANCE_PLN:
        raise BrowserExportError(
            f"the exported model prices a sampled advert {worst:,.4f} PLN away from the "
            f"pipeline (tolerance {PARITY_TOLERANCE_PLN}) — publishing it would put two "
            f"different answers to the same question on one site"
        )
    return worst


def build(models_dir: Path | None = None) -> tuple[dict, float]:
    """Assemble the payload from the served artifact. Returns it with its worst parity gap."""
    bundle = model_module.load_model(**({"models_dir": models_dir} if models_dir else {}))
    fitted = bundle["model"]
    metadata = bundle["metadata"]
    regressor = fitted.regressor_.named_steps["reg"]
    booster = getattr(regressor, "booster_", None)
    if booster is None:
        raise BrowserExportError(
            f"the served model is {metadata['model']}, which has no LightGBM booster to "
            f"export — only a gradient-boosted tree model can run in the browser this way"
        )

    plan = _plan(fitted)
    _assert_plan_matches(plan, fitted)
    payload = {
        "schema": BROWSER_MODEL_SCHEMA,
        # Provenance, so the page can say which model it is running rather than implying it is
        # the same one by omission.
        "served_model": metadata["model"],
        "trained_at": metadata["trained_at"],
        "n_train": metadata["n_train"],
        "reference_year": metadata["reference_year"],
        "inverse": "expm1",
        "plan": plan,
        "trees": _flatten(booster),
    }
    return payload, _assert_agrees_with_pipeline(payload, fitted)


def _on_threshold_values(payload: dict) -> dict[str, float]:
    """One real split threshold per numeric feature, to be priced exactly on the boundary.

    The runtime branches on `value <= threshold`; a port that writes `<` instead is correct
    for every car except one sitting exactly on a split, which is the car no sampled fixture
    will ever contain.
    """
    columns = _plan_columns(payload["plan"])
    trees = payload["trees"]
    chosen: dict[str, float] = {}
    for field in config.NUMERIC_FEATURES:
        index = columns.index(field)
        thresholds = [threshold for feature, threshold
                      in zip(trees["feature"], trees["threshold"]) if feature == index]
        if thresholds:
            # The median split rather than an extreme one: a threshold at the edge of the
            # feature's range is likelier to be a boundary no real car reaches anyway.
            chosen[field] = sorted(thresholds)[len(thresholds) // 2]
    return chosen


def parity_cases(payload: dict) -> list[dict]:
    """The cars every implementation of this model must agree about.

    Sampled adverts plus the cases a sample cannot contain: both ends of every bound, one car
    per province and per fuel, an EV with no displacement, values sitting exactly on a split,
    and the four inputs that must be refused rather than priced.
    """
    frame = data.load_clean()
    sampled = frame.sample(12, random_state=config.RANDOM_STATE + 1)
    x, _ = features.prepare(sampled)
    control = x.iloc[0].to_dict()

    def variant(**overrides) -> dict:
        return {**control, **overrides}

    cases: list[dict] = [
        {"case": f"sampled advert {position + 1}", "car": car.to_dict()}
        for position, (_, car) in enumerate(x.iterrows())
    ]
    cases += [
        {"case": "age at the young bound", "car": variant(age=0)},
        {"case": "age at the old bound", "car": variant(age=config.AGE_MAX)},
        {"case": "no mileage", "car": variant(mileage=0)},
        {"case": "mileage at the ceiling", "car": variant(mileage=int(config.MILEAGE_MAX))},
        {"case": "an EV, for which zero displacement is a fact",
         "car": variant(fuel=config.ELECTRIC_FUEL, vol_engine=0)},
        {"case": "displacement at the ceiling",
         "car": variant(vol_engine=int(config.VOL_ENGINE_MAX))},
    ]
    cases += [{"case": f"province {province}", "car": variant(province=province)}
              for province in config.PROVINCES]
    cases += [{"case": f"fuel {fuel}",
               "car": variant(fuel=fuel,
                              vol_engine=0 if fuel == config.ELECTRIC_FUEL
                              else control["vol_engine"])}
              for fuel in config.KNOWN_FUELS]
    for field, threshold in _on_threshold_values(payload).items():
        # Three cars around one split. The exact one is the only test of `<=` against `<`, and
        # it is unreachable through the API: LightGBM splits at midpoints (…172000.5) while a
        # request takes whole kilometres, so that case is marked as one the service cannot be
        # asked. Its integer neighbours straddle the same branch and every surface can price
        # them.
        cases.append({"case": f"{field} exactly on a split threshold",
                      "car": variant(**{field: threshold}), "api": False})
        cases.append({"case": f"{field} just below a split threshold",
                      "car": variant(**{field: int(math.floor(threshold))})})
        cases.append({"case": f"{field} just above a split threshold",
                      "car": variant(**{field: int(math.floor(threshold)) + 1})})

    # Refusals. Not measured against the pipeline: it answers an unknown make with the global
    # target mean rather than raising, which is the bug this vocabulary work exists to close.
    # What must agree is the three surfaces that face a user.
    cases += [
        {"case": "a make the model never saw", "car": variant(mark="ferrari", model="f40"),
         "refused": "mark"},
        {"case": "a model the model never saw", "car": variant(model="qqqq"),
         "refused": "model"},
        {"case": "a fuel outside the declared domain", "car": variant(fuel="Petrol"),
         "refused": "fuel"},
        {"case": "a province outside Poland", "car": variant(province="Berlin"),
         "refused": "province"},
    ]
    return cases


def write_parity_fixture(payload: dict, fitted, path: Path | None = None) -> Path:
    """Record what the pipeline answers for every parity case, for the other two to be held to.

    Written from the Python pipeline because that is the definition of correct here; the
    exported model and the JavaScript runtime are ports of it, and a port is only verifiable
    against something it did not produce itself.
    """
    path = Path(path or PARITY_FIXTURE)
    if not path.parent.is_dir():
        raise BrowserExportError(
            f"no {path.parent} to write the fixture into — this runs from a checkout, not "
            f"from an installed package"
        )
    cases = parity_cases(payload)
    priced = [case for case in cases if "refused" not in case]
    frame = pd.DataFrame([case["car"] for case in priced])[list(features.FEATURE_COLUMNS)]
    expected = fitted.predict(frame)

    prices = iter(expected)
    recorded = [
        {"case": case["case"], "car": case["car"],
         **({"api": False} if case.get("api") is False else {}),
         **({"refused": case["refused"]} if "refused" in case
            else {"expected_pln": round(float(next(prices)), 6)})}
        for case in cases
    ]

    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump({"schema": BROWSER_MODEL_SCHEMA, "served_model": payload["served_model"],
                   "trained_at": payload["trained_at"], "tolerance_pln": PARITY_TOLERANCE_PLN,
                   "cases": recorded}, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return path


def export(out_dir: Path | None = None, models_dir: Path | None = None,
           fixture_path: Path | None = None) -> list[Path]:
    """Write the browser model where the form is served from, and the fixture that pins it."""
    out_dir = Path(out_dir or config.SITE_APP_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload, worst = build(models_dir)

    path = out_dir / MODEL_FILENAME
    # Compact separators: this is the largest committed file in the repository, and the
    # whitespace of an indented dump would be a third of it. LF for the same reason as every
    # other generated file here.
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, separators=(",", ":"), ensure_ascii=False)
        handle.write("\n")
    print(f"parity against the pipeline: worst {worst:.2e} PLN over {PARITY_SAMPLE} adverts")

    fitted = model_module.load_model(**({"models_dir": models_dir} if models_dir else {}))["model"]
    fixture = write_parity_fixture(payload, fitted, fixture_path)
    return [path, fixture]


if __name__ == "__main__":
    for written in export():
        print("wrote", written)
