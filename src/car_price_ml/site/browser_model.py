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
from bisect import bisect_left, bisect_right
from pathlib import Path

import pandas as pd

from car_price_ml import config, data, features
from car_price_ml import model as model_module
from car_price_ml.site import MIN_BUCKET_N

# Bump when the payload's shape changes. The runtime refuses anything else rather than
# interpreting the fields it recognises — the same contract as the artifact and config.json.
# 2: carries the out-of-fold error bands, so a valuation can be shown with the spread it
# actually has at that price rather than with the model's average error.
# 3: carries the oldest age the data supports, so the what-if curve stops where the report's
# depreciation curve stops instead of drawing ages nobody measured.
BROWSER_MODEL_SCHEMA = 3

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

# Objectives whose prediction is the plain sum of the leaves. Everything else — poisson,
# gamma, tweedie, the classifiers — puts a link function in between, which this export has no
# way to describe and the runtime has no way to apply.
IDENTITY_LINK_OBJECTIVES = frozenset({
    "regression", "regression_l1", "huber", "fair", "quantile", "mape",
})


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
        config.HIGH_CARD_CATEGORICAL, target_encoder.categories_, target_encoder.encodings_,
        strict=True,
    ):
        steps.append({
            "kind": "target_encode",
            "field": field,
            # An object, not two parallel lists: a lookup that cannot be misaligned.
            # strict: a truncated pairing would drop makes off the end of the vocabulary,
            # and the runtime refuses what the vocabulary lacks — so cars the model can price
            # would come back refused, with nothing anywhere saying why.
            "map": {str(category): float(encoding)
                    for category, encoding in zip(categories, encodings, strict=True)},
        })
    for field, categories in zip(config.LOW_CARD_CATEGORICAL, one_hot.categories_,
                                 strict=True):
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
    # The objective decides whether the summed leaves *are* the prediction. Under poisson,
    # gamma or tweedie LightGBM applies a log link on top of the sum, which this runtime does
    # not — and the payload has no field for it, because `inverse` names the target transform
    # rather than the objective's link. Refused by contract for the same reason as the three
    # checks above: the sample check would catch it, and the sample is what a contract exists
    # so as not to depend on.
    if dumped.get("objective", "").split()[0] not in IDENTITY_LINK_OBJECTIVES:
        raise BrowserExportError(
            f"the booster's objective is {dumped.get('objective')!r}, which puts a link "
            f"function between the summed leaves and the prediction; the runtime sums and "
            f"inverts the target transform only"
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


def error_band(payload: dict, price: float) -> dict:
    """The measured error band a valuation falls in — the reference for `predict.js`.

    Outside the range the out-of-fold predictions covered, the nearest band is returned with
    ``measured=False`` rather than an invented one: clamping is a fallback, so it is labelled
    as one instead of being presented as a measurement of that price.
    """
    bands = payload["error_bands"]
    for index, band in enumerate(bands):
        # Half-open, so a price sitting exactly on a shared edge belongs to exactly one band;
        # the last band closes at its top edge because there is nothing above it to hand on to.
        last = index == len(bands) - 1
        if band["from_pln"] <= price and (price <= band["to_pln"] if last
                                          else price < band["to_pln"]):
            return {**band, "measured": True}
    nearest = bands[0] if price < bands[0]["from_pln"] else bands[-1]
    return {**nearest, "measured": False}


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


def _age_support() -> dict:
    """The oldest age the data actually supports, by the rule the report's curve already obeys.

    The form's what-if curve prices the same car at every age, and the model will answer for
    any of them — at 33 years and beyond it returns one flat number, because the trees have run
    out of splits, and between 29 and 33 it has the car *gaining* value. Drawn with the same
    class names as the report's chart, on the same site, that reads as a measurement. So the
    curve stops where the report's stops: at the last age with a bucket the site is willing to
    publish.
    """
    frame = data.load_clean()
    sizes = frame.groupby("age").size()
    supported = sorted(int(age) for age, count in sizes.items() if count >= MIN_BUCKET_N)
    if not supported or supported[0] != 0:
        raise BrowserExportError(
            f"no contiguous run of age buckets from 0 with n >= {MIN_BUCKET_N} — the what-if "
            f"curve would have to start somewhere the reader did not ask about"
        )
    oldest = 0
    for age in supported:
        if age != oldest:
            break
        oldest = age + 1
    return {"max_age": oldest - 1, "min_bucket_n": MIN_BUCKET_N}


def _error_bands(metadata: dict) -> list[dict]:
    """The artifact's out-of-fold error bands, checked for the shape the runtime looks up in.

    Refused rather than defaulted if absent: a form that quoted a price with no spread, or
    with a spread it invented, is the same failure as one that priced an unknown make.
    """
    bands = metadata.get("oof_error_bands")
    if not bands:
        raise BrowserExportError(
            "the artifact carries no out-of-fold error bands, so a valuation could not be "
            "shown with its spread — retrain with `python -m car_price_ml.train`"
        )
    edges = [(band["from_pln"], band["to_pln"]) for band in bands]
    if any(low >= high for low, high in edges) or edges != sorted(edges):
        raise BrowserExportError(f"the error bands are not in ascending price order: {edges}")
    # Contiguous, not merely ordered: a gap between two bands is a price the lookup cannot
    # place, and both lookups answer an unplaceable price with the nearest band — which for
    # anything above the first gap means the most expensive one.
    gaps = [(lower["to_pln"], upper["from_pln"]) for lower, upper in zip(bands, bands[1:])
            if lower["to_pln"] != upper["from_pln"]]
    if gaps:
        raise BrowserExportError(
            f"the error bands leave {len(gaps)} gap(s) a valuation could fall into ({gaps[:3]}"
            f"...) — a price in one would be shown the nearest band's spread and told it was "
            f"outside the measured range"
        )
    return bands


def _reachable_values(payload: dict) -> dict[int, list[float] | tuple[float, float]]:
    """Every value each model column can take, per column index.

    Discrete columns get their values; the three numeric ones get their integer bounds, since
    the API and the form both take whole years, kilometres and cm³. Used to check the
    threshold rounding exactly rather than statistically.
    """
    reachable: dict[int, list[float] | tuple[float, float]] = {}
    index = 0
    bounds = {
        "age": (0.0, float(config.AGE_MAX)),
        "mileage": (0.0, config.MILEAGE_MAX),
        "vol_engine": (0.0, config.VOL_ENGINE_MAX),
    }
    for step in payload["plan"]:
        if step["kind"] == "target_encode":
            reachable[index] = sorted(step["map"].values())
            index += 1
        elif step["kind"] == "one_hot":
            for _ in step["categories"]:
                reachable[index] = [0.0, 1.0]
                index += 1
        else:
            reachable[index] = bounds[step["field"]]
            index += 1
    return reachable


def _assert_rounding_flips_no_branch(payload: dict, booster) -> tuple[int, float]:
    """No reachable feature value may fall between a threshold and its rounded copy.

    A rounded *leaf* costs a little accuracy, which sampling can bound. A rounded *threshold*
    is different in kind: its error is either nothing at all or an entire branch, so a sample
    can only report that it did not happen to hit one. The reachable domain here is finite —
    351 encoded values, {0, 1} per one-hot slot, integers between the declared bounds — so the
    check can be exact, and it stays exact as the artifact changes.

    Returns how many thresholds were checked and the largest distance any of them moved, which
    is the perturbation the guarantee had to survive. (The distance to the nearest reachable
    value is not reported: one-hot columns split just above zero, so it is ~1e-35 there and
    would read as a near miss when the rounding at that scale is itself ~1e-35.)
    """
    exact: list[tuple[int, float]] = []

    def collect(node: dict) -> None:
        if "leaf_value" in node:
            return
        exact.append((int(node["split_feature"]), float(node["threshold"])))
        collect(node["left_child"]), collect(node["right_child"])

    for tree in booster.dump_model()["tree_info"]:
        collect(tree["tree_structure"])

    reachable = _reachable_values(payload)
    shift = 0.0
    for column, threshold in exact:
        rounded = round(threshold, THRESHOLD_DECIMALS)
        low, high = min(threshold, rounded), max(threshold, rounded)
        values = reachable[column]
        if isinstance(values, tuple):  # an integer range: only whole numbers are reachable
            floor, ceiling = values
            candidates = [value for value in (math.floor(low), math.floor(low) + 1,
                                              math.ceil(high))
                          if floor <= value <= ceiling]
        else:
            # Every discrete value inside the interval, not the first couple of them: the
            # interval is tiny, so more than one is implausible — but "implausible" is the
            # word that precedes the bugs this check exists to catch.
            candidates = values[bisect_left(values, low):bisect_right(values, high)]
        for value in candidates:
            # The comparison the runtime makes, before and after rounding. Equal is all that
            # is asked: the branch must not move.
            if (value <= threshold) != (value <= rounded):
                raise BrowserExportError(
                    f"rounding a threshold on column {column} to {THRESHOLD_DECIMALS} decimals "
                    f"moves a reachable value ({value}) to the other branch — the browser and "
                    f"the pipeline would price it differently"
                )
        shift = max(shift, abs(threshold - rounded))
    return len(exact), shift


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


def build(models_dir: Path | None = None) -> tuple[dict, dict, float]:
    """Assemble the payload from the served artifact.

    Returns the payload, the fitted pipeline it came from (so the caller does not load the
    13.9 MB artifact a second time) and the worst disagreement the self-check found.
    """
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
        # Measured at training time from the winner's own out-of-fold predictions. Shipped
        # with the model because a price without its spread is a precision the model does not
        # have — and the spread is not one number: the median absolute error runs from about
        # 1 500 PLN in the cheapest tenth of the market to 21 600 in the dearest.
        "error_bands": _error_bands(metadata),
        "age_support": _age_support(),
        "plan": plan,
        "trees": _flatten(booster),
    }
    # Two checks of two different things: that no branch moved when the thresholds were
    # rounded (exact, over the whole reachable domain) and that the assembled payload prices
    # real adverts as the pipeline does (sampled, and the only thing that can catch a mistake
    # in the plan itself).
    checked, shift = _assert_rounding_flips_no_branch(payload, booster)
    print(f"threshold rounding: no branch moves across {checked:,} splits "
          f"(largest shift {shift:.2e})")
    return payload, fitted, _assert_agrees_with_pipeline(payload, fitted)


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
                      in zip(trees["feature"], trees["threshold"], strict=True)
                      if feature == index]
        if not thresholds:
            # Silently dropping the field would leave the fixture one boundary case lighter
            # while still reading as complete.
            raise BrowserExportError(
                f"no split in the model uses {field!r}, so the fixture cannot carry a car "
                f"sitting exactly on one — refusing to write a fixture that looks complete"
            )
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
    payload, fitted, worst = build(models_dir)

    path = out_dir / MODEL_FILENAME
    # Compact separators: this is the largest committed file in the repository, and the
    # whitespace of an indented dump would be a third of it. LF for the same reason as every
    # other generated file here.
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, separators=(",", ":"), ensure_ascii=False)
        handle.write("\n")
    print(f"parity against the pipeline: worst {worst:.2e} PLN over {PARITY_SAMPLE} adverts")

    fixture = write_parity_fixture(payload, fitted, fixture_path)
    return [path, fixture]


if __name__ == "__main__":
    for written in export():
        print("wrote", written)
