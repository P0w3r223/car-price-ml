# ADR 0005 — The form answers as you type, and never without its spread

Date: 2026-08-13
Status: accepted
Author: P0w3r223
Related to: [ADR 0001](0001_scope-and-site-expansion.md) (W6, partially implemented here),
[ADR 0004](0004_the-model-runs-in-the-browser.md) (which put the model in the page)

---

## Context

ADR 0004 put the served model in the page. That changed what the form can afford: a valuation
costs about a millisecond and no request, so submit-and-wait stopped being a constraint and
became a habit.

It also left the page quoting a single number with no indication of how wrong it might be.
The report publishes 8 612 ± 72 PLN — a mean absolute error with its fold-to-fold spread —
and a reader could reasonably carry that to the form and read "± 8 612" onto their own
valuation. Measured, that reading is wrong in both directions:

| predicted price | n | median abs. error | 90th percentile |
| --- | ---: | ---: | ---: |
| 2 211 – 12 976 | 11 102 | **1 498** | 4 035 |
| 24 303 – 31 650 | 11 101 | 2 660 | 7 652 |
| 70 168 – 94 442 | 11 102 | 6 831 | 20 315 |
| 150 796 – 898 640 | 11 102 | **21 606** | 72 574 |

A single MAE flattens a **14×** range. On a 12 000 PLN car it overstates the error by more
than five times; on a 200 000 PLN one it understates it by nearly three.

## Decision

### 1. The artifact measures its own error, by band of predicted price

`train.py` already runs a k-fold bake-off and threw the out-of-fold predictions away after
scoring them. It now keeps the winner's, and `model.residual_quantiles` summarises them into
ten bands of predicted price: the range, `n`, the median and 90th-percentile absolute error,
and the median *signed* error. Those go into the artifact's metadata, so they are a property
of the model rather than a figure computed beside it — and no extra fits are paid for.

Keyed by the **prediction**, not the truth: at valuation time the prediction is all a caller
has, so a band keyed by anything else could not be looked up. Bands under 500 out-of-fold
predictions are refused rather than published, for the same reason the depreciation curve
drops thin age buckets.

The signed error is carried because a band can be tight and still be systematically low. It
is small everywhere except the top band (+2 204 PLN), which is worth knowing about the model.

### 2. No valuation is shown without it

The exported payload carries the bands (schema 2), `predict.js` refuses a payload without
them, and every price the form shows — live or submitted, from the page or from the API —
comes with the measured spread for its band. Outside the range the out-of-fold predictions
covered, the nearest band is used **and labelled as not measured for this price**, rather
than being presented as if it were.

### 3. The form answers as you type

Every input change re-values the car after a 120 ms debounce, using the local model. Two
sliders sit under `year` and `mileage` as what-if controls over the same values — not new
fields, so there is only ever one answer to "what car is this". The mileage slider stops at
400 000 km rather than the validation ceiling of 1 000 000: a control whose useful travel is
the first 4 % of its length is one nobody can aim, and the number field still accepts
anything the model was trained on.

Live valuation runs **only** where the model is in the page. Against the API alone it would
be a request per keystroke, so there the form keeps its submit-and-wait shape rather than
quietly becoming chatty.

### 4. The what-if curve is computed, not looked up

The panel under the form prices the car currently in the form at every age the model was
trained on — 41 predictions, redrawn on every change. It is deliberately not the report's
depreciation curve: that one is the market's median advert price by age, a description of the
data, while this one is the model's answer for *this* car with everything else held. Drawn by
`docs/app/curve.js` with the class names `charts.py` uses, so the two halves of the site are
one visual system and the stylesheet is shared rather than restated (`assets/chart.css`).

The y axis starts at zero. A truncated axis exaggerates the fall, and this curve is read by
someone deciding what their car is worth.

## Options considered

1. **One global ± MAE beside every price.** The same number the report's KPI shows, so no new
   measurement. Rejected on the table above: it is the most misleading option available
   precisely because it looks like the honest one.
2. **A prediction interval from a quantile model.** Fit LightGBM at the 5th and 95th
   percentiles and ship three models. Defensible, and three times the payload; the out-of-fold
   residuals answer the same question from measurements this project already makes.
3. **Per-car SHAP in the browser** as the explanation half of W6. Deferred: it is a fourth
   implementation of a non-trivial algorithm and needs its own parity fixture. The
   counterfactual form ("this car one year older: −3 100 PLN") is already reachable with the
   local model and is the next thing to land.
4. **Out-of-fold residual bands, live valuation, computed what-if curve (chosen).**

## Consequences

- **The artifact must be retrained to carry the bands**, and `browser_model` refuses to export
  one that cannot state its own error. Payload schema 1 → 2.
- **`cross_validate_models` gained `return_predictions`.** The predictions stay out of the
  metrics dict, which is stamped into the artifact and rendered as JSON — a 111 018-element
  array belongs in neither.
- **Chart styling moved to `assets/chart.css`**, shared by the report and the form.
- **146 → 153 tests**, including a leg that compares the band `predict.js` picks against the
  band the Python reference picks, for every priced case in the parity fixture. A boundary
  chosen with `<` instead of `<=` would otherwise be invisible.

### Accepted residual risk

The bands describe the model's error over the **training market** — a single January 2022
snapshot, cross-validated. They are not a prediction interval for a car being sold today, and
nothing in the data can make them one; the page says the prices are 2022 prices, but a reader
determined to read the spread as "how wrong this is about my car in 2026" will find nothing
that stops them.

The live valuation is still verified only by Node and by reading. No test in this repository
opens the page, so "typing changes the figure" and "the slider redraws the curve" are claims
about code, not about a browser.
