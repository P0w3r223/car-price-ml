# ADR 0004 — The model runs in the browser, and three implementations are held to one fixture

Date: 2026-08-13
Status: accepted
Author: P0w3r223
Related to: [ADR 0001](0001_scope-and-site-expansion.md) (W5, which this implements),
[ADR 0003](0003_form-generated-and-fail-closed.md) (the form this replaces the answer of)

---

## Context

The published valuation form answered with a formula. `heuristicEstimate` in `app.js` was
about eight lines — a base price, a per-fuel factor, 8 % a year of depreciation, a mileage
taper — and it existed because GitHub Pages has no API behind it. It was labelled everywhere
it appeared: a badge, then (after ADR 0003) a dashed card with a lighter, approximate figure
and a banner above the form saying a heuristic would answer.

Labelling it was the most that could be done, and it was not enough. The one surface most
readers of this project actually touch was the one surface that answered a question it could
not answer, and no amount of hedging changes what a reader takes away from a number. It also
left a hole in the project's own claim: the heuristic knew nothing about makes, so on Pages an
unknown make still produced a figure where the service answers `422` — the README had to carry
a paragraph explaining that its headline did not hold on the demo.

ADR 0001's W5 proposed the fix and was revised twice on measurement. Its two load-bearing
numbers were re-measured here against the served artifact rather than taken from the ADR:
`dump_model()` JSON is **57.8 MB** for 1 200 trees at 127 leaves (303 600 nodes), and a
columnar re-encoding brings that to **8.5 MB raw, 2.3 MB gzipped** — close to the 1.83 MB the
curve predicted, the difference being precision (below).

## Decision

### 1. The trees ship as parallel arrays; the preprocessing ships as a plan

`feature[] threshold[] left[] right[] value[] roots[]`, and the `ColumnTransformer` as an
ordered list of steps — `target_encode(mark)`, `target_encode(model)`, `one_hot(fuel)`,
`one_hot(province)`, `numeric(age|mileage|vol_engine)`. The runtime walks the plan and appends
columns, so the column order is *data*, not JavaScript that has to be kept in step by hand.

This was ADR 0001's revision and it is the right one: a hand-ported column order does not
raise when it is wrong, it prices a different car. The export additionally compares the columns
its plan produces against the fitted transformer's own `get_feature_names_out()` and refuses to
write on a mismatch, so the ordering is checked rather than trusted.

### 2. The export refuses to publish a model that would answer differently

Before writing, `browser_model` re-prices 200 real adverts through the payload it is about to
write — using its own interpreter, not the pipeline's — and raises if any answer differs from
the pipeline by more than a grosz. Measured on the current artifact: **3.97e-06 PLN**.

It also refuses structures the runtime does not implement rather than exporting them: a split
that is not `<=`, a missing-value branch, an averaging booster. Each of those would export
cleanly and then be walked wrongly.

### 3. Precision is set by what it costs in złoty, not by what looks tidy

A leaf value is a log-price contribution, summed over 1 200 trees and then inverted with
`expm1`, so a per-node rounding error is multiplied by the tree count and again by the price.
Measured over 200 adverts:

| rounding | worst disagreement | raw | gzipped |
| --- | ---: | ---: | ---: |
| thresholds 6 dp, leaves 6 dp | **3.96 PLN** | 7.6 MB | 1.78 MB |
| thresholds 6 dp, leaves 9 dp | 0.0033 PLN | 8.1 MB | 2.06 MB |
| **thresholds 6 dp, leaves 12 dp** | **0.000004 PLN** | 8.5 MB | **2.31 MB** |

Six decimals was the obvious choice and would have put a visible złoty of drift between the
page and the API. The extra 240 KB buys a parity claim that needs no qualification.

A review of this ADR pointed out that the two halves of the table fail differently, and that
only one of them can be bounded by sampling. A rounded **leaf** costs a little accuracy, which
200 adverts can measure. A rounded **threshold** costs either nothing or an entire branch, and
a sample can only report that it did not happen to hit one — the reachable domain is finite,
so the check is now exact instead: for all 151 200 splits, against every value each column can
take (351 encoded values, `{0, 1}` per one-hot slot, whole numbers between the declared bounds
for `age`/`mileage`/`vol_engine`), no reachable value falls between a threshold and its rounded
copy. Largest shift any threshold takes: 4.97e-07. The export runs this before writing, so the
guarantee survives a retrain rather than describing one artifact.

### 4. One fixture, three implementations

The Python pipeline, the exported payload and `docs/app/predict.js` are three paths from a car
to a price, and ports drift silently. `tests/fixtures/browser_parity.json` records what the
pipeline answers for 53 cars and the other two are held to it — the pipeline is the definition
of correct, so the ports are verified against something they did not produce.

The cases are chosen for what a sample cannot contain: both ends of every bound, one car per
province and per fuel, an EV with zero displacement, a value sitting **exactly** on a real
split threshold (the only test that distinguishes `<=` from `<`), and four inputs that must be
refused rather than priced. The exact-threshold cars are marked as ones the API cannot be
asked: LightGBM splits at midpoints (…172000.5) while a request takes whole kilometres. Their
integer neighbours straddle the same branch and every surface prices those.

The JavaScript leg runs under Node in CI and skips locally with a stated reason. This is the
repository's first Node dependency — the one ADR 0003 named as the price of closing its
residual risk — and it is a runtime with no `package.json`, because `predict.js` is a plain
script that the browser loads and the runner requires.

### 5. The heuristic is deleted, and the page fails closed instead

There is no fallback pricing left in the project. If `model.json` cannot be loaded and no API
answers, the form refuses to run, exactly as it already does without `config.json`. The form
also now carries the model's own vocabulary, so an unknown make is refused on the static demo
too — which is where the README's caveat went.

## Options considered

1. **Host the API on a free tier.** Rejected in ADR 0001 and still rejected: cold starts, and
   a public demo that depends on a service someone will eventually switch off.
2. **Ship the model as binary typed arrays.** Smallest raw file (5.5 MB) and fastest to load,
   but GitHub Pages does not gzip `application/octet-stream`, so the reader downloads 5.5 MB
   instead of 2.3 MB.
3. **Commit the payload gzipped and decompress with `DecompressionStream`.** Same transfer,
   a quarter of the repository size — rejected because the file stops being readable and a
   browser without `DecompressionStream` lands on the refusal path for no gain.
4. **Trim the model until it fits comfortably.** Rejected by ADR 0001's own curve: quality is
   flat past 1 200 × 127 and the encoding is where the size actually was.
5. **Columnar JSON, plan-driven, with a three-way fixture (chosen).**

## Consequences

- **`docs/app/model.json` is 8.5 MB in the working tree** and roughly 2 MB per retrain in git
  history. It is committed and not regenerable in CI, so — like `docs/data/*.json` — it sits
  outside the page's diff guard; the fixture is what stops it going stale silently.
- **Publishing is unchanged in shape:** `site.export` now writes the browser model and the
  fixture along with the page's aggregates, because all three come from the same artifact.
- **`GET /vocabulary` is no longer fetched by the form** — the vocabulary arrives with the
  model. The endpoint stays for API consumers.
- **131 → 146 tests**, three of which skip without Node.

### Accepted residual risk

The fixture pins the three implementations to each other, but nothing pins `model.json` to the
artifact on disk except the export command: an artifact retrained without re-running the export
leaves a stale model in the page and a stale fixture agreeing with it. The pipeline leg of the
parity test is what notices — it re-prices the fixture through the *current* artifact — so the
failure is loud, but only for someone who has the artifact, which CI does not. What CI *can*
check is that the two published halves describe one model: `model.json` and `docs/data/
metrics.json` must agree on the served model, its training date and its row count, and a test
compares those committed files without needing either artifact or dataset. That closes the
half of the risk where the page's headline would describe one model while the form beside it
announced another. This is the same shape as the
aggregates' residual risk in ADR 0002, and it has the same real fix: running the export
somewhere that has the artifact.

The browser runtime is still verified only by Node, not by a browser. `predict.js` uses typed
arrays, `fetch` and `Math.expm1` and nothing else, so the gap is narrow — but "the page works"
remains a claim no test in this repository makes.
