# ADR 0001 — Scope expansion (v0.2): model the whole of Poland, and run the model in the browser

Date: 2026-08-12
Status: accepted — revised the same day after an independent architecture review and three
research passes; the revisions are material and are marked **[revised]** below
Author: P0w3r223 + Claude
Related to: [`docs/research/data-and-methodology.md`](../research/data-and-methodology.md),
[mlops-car-price](https://github.com/P0w3r223/mlops-car-price) ADR 0001 / ADR 0003

---

## Context

A3 ships a defensible ML cycle: cleaning, out-of-fold target encoding, k-fold CV, SHAP,
a FastAPI service and a static mini site. The limits are no longer methodological — they
are about *reach* (how much of the domain the model sees) and *proof* (what a reader can
verify without cloning the repo).

### What the survey found

| Area | State | Consequence |
| --- | --- | --- |
| Geography | `province` is a model feature; `city` (4 427 distinct) and `generation_name` (364) are in the data and **unused** | "Polish market" is asserted, not modelled |
| Province labels | the web form sent `Kujawsko-Pomorskie` / `Warmińsko-Mazurskie`; training data holds `Kujawsko-pomorskie` / `Warmińsko-mazurskie` | `handle_unknown="ignore"` emits an all-zero row — not a neutral province but a combination occurring nowhere in training, so the trees extrapolate off-manifold |
| Province domain | 23 distinct values: 16 Polish provinces plus 44 foreign/garbage rows | noise inside a closed categorical |
| **Duplicates** | **9.8 % of rows sit in exact-duplicate groups; 6 468 rows are removable; 31 groups hold 22 identical adverts** | **`KFold(shuffle=True)` puts identical adverts in train and test — every published metric is optimistic** |
| Served model | RandomForest, 590 MB artifact | rules out free hosting and client-side inference |
| Public demo | GitHub Pages has no API, so the form falls back to `heuristicEstimate` | the most visible surface shows a number the model did not produce |
| Report generation | one 142-line f-string with base64 PNGs (233 KB HTML), copied by hand out of a gitignored directory | no regeneration path in CI, no staleness signal |
| **Data vintage** | **a single January 2022 snapshot — not the "~2021–2023" claimed in the README, the research doc and `config.py`** | `REFERENCE_YEAR = 2024` sits two years past the data; the model has **never seen a car younger than `age` 2**, yet the API accepts model year 2024 |
| **`vol_engine == 0`** | **873 EVs, but also 374 petrol/diesel/hybrid/LPG rows where 0 means "missing"** | the model learns "displacement 0 → expensive EV" and misprices those 374 |
| **SHAP scale** | **`shap_explanation` explains the inner regressor, which is fit on `log1p(price)`** | verified: `sum(SHAP) + base = log1p(prediction)` exactly. Additivity does **not** survive `expm1` |
| **Feature spec** | **`train.py` writes `metadata["features"]`; nothing ever reads it back** | loading a v0.1 artifact with v0.2 code yields silently wrong predictions, not an error |

### Evidence gathered before deciding

**Duplicates change the numbers but not the ranking** (5-fold CV, pooled OOF, PLN):

| | RandomForest | LightGBM | Ridge |
| --- | ---: | ---: | ---: |
| With duplicates (as published) | **8 659** | 9 227 | 15 610 |
| Deduplicated (n = 111 347) | **8 915** | 9 246 | 15 142 |

The inflation is asymmetric exactly as the mechanism predicts: RandomForest gains 256 PLN
from duplicates (`min_samples_leaf=3` forms a pure leaf on three copies of one advert),
LightGBM gains 19 PLN (`min_child_samples=20` cannot). The hypothesis that RandomForest's
win is *entirely* a duplication artifact is **refuted** — it still wins on clean data, but
its margin falls from 6.6 % to 3.7 %.

**The province mismatch was measured, not assumed.** The misspelled input does produce an
all-zero province vector (the fitted encoder holds 23 categories, none matching), and
re-pricing 400 real `Kujawsko-pomorskie` adverts with the form's spelling moves the
valuation by 1 376 PLN on average (1.4 %), median 136, p95 6 344, worst case 42 533. The
usual effect is small; the tail is not.

**A general-inflation adjustment would have the wrong sign.** Eurostat, Poland:

| Series | Jul 2022 | Latest | Change |
| --- | ---: | ---: | ---: |
| All-items HICP (CP00) | 109.5 | 154.7 | **+41 %** |
| **Second-hand cars (CP07112)** | 81.7 | 62.4 | **−24 %** |
| Indicata PL retail index | ~120 | 108.5 | **≈ −10 %** |

Rescaling 2022 prices by consumer inflation gives +43 % where the used-car market moved
−10 % to −24 %: wrong by 50–65 percentage points, with the sign reversed. Drift is also
strongly segment-specific — petrol ≈ 110.6 against **BEV ≈ 74.0**, a 36 pp spread — so a
single scalar is least defensible precisely for the EVs this project deliberately keeps.

Downstream constraint: **P1 `mlops-car-price` consumes this repo as a pinned package**
(`car-price-ml @ git+…@v0.1.1`, its ADR 0001). Its ADR 0003 measured LightGBM at 3.3 MB /
9 278 PLN against RandomForest at 338 MB / 8 908 PLN.

## Decision

Ship **v0.2** as eight work streams, sequenced so nothing is measured on data that cannot
support the measurement.

### W0 — A baseline worth measuring against ✅ *implemented* **[revised — new, and it gates everything]**

Deduplication as a documented cleaning rule; separation of "EV" from "missing" in
`vol_engine` (374 combustion rows carried 0 displacement); `REFERENCE_YEAR` re-anchored to
the data's own vintage (2022, not 2024) with a training-time guard that refuses a dataset
newer than the anchor; the vintage claim corrected in the README, the research doc and
`config.py`; and **both halves of the inference contract — the feature spec and the age
anchor — stamped by `save_model` and enforced by `load_model`**, so a stale artifact fails
at load instead of at the first valuation. Cleaning now takes 117 927 raw rows to 111 018.

The anchor turned out to need the same protection as the column list: moving
`REFERENCE_YEAR` while an old artifact is on disk shifts every derived `age` by the
difference, silently and in a direction no metric would show.

This stream exists because W2's gate — "a feature ships only if it lowers MAE" — is
unusable on contaminated data: a feature can pass by helping the model memorise duplicates,
and `city`, the highest-cardinality candidate, is the most likely to pass for that wrong
reason. **Nothing downstream is measured until W0 lands.**

### W1 — Geography as a real feature ✅ *implemented*

One canonical province vocabulary in `config.py`, in correct Polish orthography (the
dataset's own spelling). Normalisation in `data.py`, applied on both the training and the
inference path, folding case, diacritics and separator variants onto it. Unknown provinces
become an explicit **422**. Foreign rows dropped by a documented rule. One-hot domains
**declared rather than learned**, so the feature space cannot depend on what a fold happened
to contain. A test asserts the JavaScript and Python vocabularies are identical strings.

### W2 — Features from data already present

`city` via out-of-fold target encoding plus a static city → size-class dictionary;
`generation_name` via hierarchical mark → model → generation coding. Two questions to
settle **before** implementation, not after:

- **Nesting.** `city → province` is nearly deterministic, so target-encoding `city` makes
  the province one-hot largely redundant — which matters, because province is what the
  site's map visualises. Decide explicitly whether province survives as a feature or
  becomes purely presentational.
- **Drift.** `generation_name` values (`gen-w206-2021`, `gen-8p-2003-2012`) encode
  production-year ranges, near-collinear with `age`. This is not leakage — generation is a
  real attribute — but the drift argument that banned raw `year` transfers: a vocabulary
  frozen in 2022 has no entry for a 2027 generation. It is also **25.5 % empty**, so the
  missing-value rule must be stated rather than left to the encoder's default.

### W3 — A second snapshot, for drift rather than volume **[revised]**

The original plan was the 208k dataset (May 2021) for its `Power_HP` and gearbox columns.
Research surfaced a better primary target: an **April 2023 Otomoto snapshot (>200k rows,
CC BY-SA 4.0)** — fifteen months newer than the current source. Adding it converts the
staleness problem into a **measured temporal-drift demonstration** on real data, which is
both a stronger portfolio story than a frozen sample and the natural feed for P1's drift
detector. The 208k set remains valuable for power/gearbox and can follow.

Note the share-alike obligation: derived datasets must carry BY-SA. And one uncomfortable
fact recorded honestly: **a CC0 tag applied by a scraper who is not the database maker
cannot extinguish the portal's sui generis right** — which applies to the dataset this
project already uses. The defensible line is "we do not republish third-party database
extracts", not "the tag makes it clean".

### W4 — Freshness: a dated valuation, not a rescaled one **[revised — reversed]**

The draft called price-index calibration "the honest option". The measurements above show
it is the opposite: a scalar multiplier asserts a currency the model cannot back, and
applied across a segment-asymmetric shock it is wrong by more than the effect it corrects.
Compounding it, advancing `REFERENCE_YEAR` already ages the car, so an index that embeds
ageing double-counts.

Revised decision: **the as-of date becomes a first-class model output**, derived from the
data's vintage. Any index adjustment is a separate, clearly-labelled, toggleable layer that
publishes the *disagreement between two independent sources* (Eurostat HICP is
quality-adjusted and diverges ~7 pp from raw market medians — the two answer different
questions) rather than a spuriously precise factor. It never touches the headline number.

Two clean, free supplements that need no scraping: **CEPiK** (national registration data,
CC BY, 60+ attributes, no prices) to check and reweight the training sample against the
actual Polish fleet — the model currently trains on one portal's convenience sample with no
such check; and **AAA AUTO's monthly voivodeship table** as an independent external check on
the model's province effects.

### W5 — The model in the browser **[revised on both technical claims]**

The preprocessing stack is dictionaries: target encoding is a `mark|model → number` map,
one-hot a category list, `age` a subtraction. But two claims in the draft were wrong:

- **Size.** The 3.3 MB figure is a *binary* artifact size. `dump_model()` JSON for 600
  trees at `num_leaves=63` is ~75 000 nodes at 250–400 bytes each → **10–20 MB**. The fix
  is encoding, not model surgery: a columnar layout (parallel `feature[]`, `threshold[]`,
  `left[]`, `right[]`, `leafValue[]` arrays) costs ~30 chars/node → ~1.6 MB raw, ~400–600 KB
  gzipped, and loads straight into typed arrays. **Encode efficiently first; trim trees
  only if that is still not enough.**
- **"Tens of lines"** is true for tree traversal and false for the ColumnTransformer.
  Hand-porting column ordering, `TargetEncoder`'s asymmetric `fit_transform` / `transform`
  behaviour and its unseen-category fallback fails silently. Instead, Python emits a
  **resolved instruction list** — one entry per output slot, in ColumnTransformer order —
  and JS becomes a ~20-line interpreter that cannot get the ordering wrong, because the
  ordering is data rather than code.

The served model is chosen from a **measured size ↔ quality curve** — 16 configurations over
an `n_estimators × num_leaves` grid, each scored by the same 5-fold CV, on deduplicated data:

| trees × leaves | MAE (PLN) | joblib | `dump_model()` JSON | columnar, gzipped |
| --- | ---: | ---: | ---: | ---: |
| 600 × 63 (the previous default) | 9 098 | 3.5 MB | 13.2 MB | 0.45 MB |
| 1200 × 63 | 8 803 | 7.0 MB | 26.3 MB | 0.90 MB |
| **1200 × 127** | **8 612** | 13.9 MB | 53.2 MB | **1.83 MB** |
| 2400 × 127 | 8 627 | 27.8 MB | 106.6 MB | 3.65 MB |
| 4000 × 63 | 8 636 | 23.2 MB | 87.7 MB | 2.98 MB |
| *RandomForest (served until now)* | *8 798* | *590 MB* | — | — |

**The trade-off this ADR was built around does not exist.** LightGBM was never tuned — the
600 × 63 default was simply undertrained, which made "RandomForest wins, accept the size"
look like a law rather than an artifact of one hyperparameter choice. At the knee, LightGBM
is *more* accurate than RandomForest with a **42× smaller artifact**, and past 1200 × 127
quality stops improving while size doubles twice over.

The size claims in the first draft were also wrong by an order of magnitude, as the review
predicted: `dump_model()` JSON is 13.2 MB at 600 × 63, not 3.3 MB. Columnar re-encoding —
parallel `feature/threshold/left/right/leaf` arrays — cuts the raw JSON by **29×** after
gzip. That, not tree trimming, is what makes the browser export viable.

One caveat published with the curve: choosing the best of 16 configurations on the same CV
makes the winner's score slightly optimistic, so the served-model decision rests on the
**42× size difference**, which is not subject to selection noise, rather than on the
186 PLN accuracy margin, which is.

**Parity is three-way, not two.** The FastAPI service is retained as the local-serving
path, so it is a third implementation that can drift. One golden fixture of N cars with
expected PLN outputs, consumed by pytest against the pipeline, pytest against `/predict`,
and Node against the JS runtime — with adversarial cases, not just sampled rows: unseen
make/model, missing `generation_name`, values exactly on split thresholds, `vol_engine = 0`,
`age` at both bounds, and each of the 16 provinces.

### W6 — An interactive site that demonstrates the model **[revised on two elements]**

Vanilla JS + SVG, no external dependencies: an SVG choropleth of the 16 provinces, live
valuation on input with a debounce, what-if sliders redrawing the depreciation curve of the
car currently in the form, a per-car explanation, and an uncertainty band from out-of-fold
residuals. Motion respects `prefers-reduced-motion`.

- **The explanation cannot be a PLN waterfall.** SHAP here is log-space (verified above);
  under `expm1` the contributions become multiplicative. Render it multiplicatively
  (×1.15, ×0.82 …) or on an explicit log axis. A waterfall reading "+8 400 PLN for age"
  would be the exact error class this project's own rules single out, on its most-viewed
  page.
- **The map cannot show median price.** Median advert price per province is confounded by
  fleet mix — Mazowieckie's higher median is mostly premium marques, not location — so the
  map would visually assert a premium the model does not believe in. Show the model's
  marginal province effect, with the raw median available but labelled descriptive.

Structurally, `report.py` splits three ways: hand-written app under `docs/app/`, generated
`docs/data/*.json` (metrics, aggregates, model, config contract), and figures as **files**
rather than base64 — which alone takes the committed HTML from 233 KB to ~10 KB. The small
aggregates (16 rows for the map, 3 for the bake-off) are committed, so CI can build and
validate the page from committed inputs; only retraining needs a local run.

**The invariant that prevents the province bug recurring:** nothing under `docs/app/` may
hardcode a value `config.py` owns. It currently duplicates six. Emitting the config
contract as JSON in W0 costs ~30 lines and kills the whole class.

### W7 — Portfolio index (`current_projects`)

A GitHub Pages index with project cards, generated together with the README from a single
`projects.json`.

### Vehicle segments beyond passenger cars — **conditional, and narrower than hoped** [revised]

Research returns a firm negative for **motorcycles, trucks and agricultural machinery**:
no per-unit price data exists for Poland under any acceptable licence. CEPiK covers every
category but carries no price; bailiff and tax-office auctions publish forced-sale figures
50–70 % below market. The only feasible extension is **light commercial vehicles**, via one
European dataset carrying `vehicle_type ∈ {car, van, truck}` — but its MIT tag sits on a
scraped corpus, so it inherits W3's unresolved licence question and stays conditional on it.

These would also need **separate models, not a `segment` flag**: Polish LCV listings quote
net and gross prices interchangeably (a 23 % gap inside one target column), the attribute
sets barely overlap (Euro class and axles; engine *hours* and PTO power; A2 compliance),
and a shared `mark` encoder would blend Mercedes-the-car with Mercedes-the-Sprinter.

### Sequence

W0 → W1 ✅ → W5 (curve + export + three-way parity) → W2 → W6 → W3 → W4 → segments
(conditional) · W7 is independent and can land at any point.

## Options considered

1. **Cosmetic site work only.** Decorates a fake number — it makes the weakest part of the
   project more prominent.
2. **Host the API on a free tier.** Removes the fake number without touching the model, but
   still needs the 590 MB artifact gone, adds cold starts, and makes the public demo depend
   on a service that will eventually be switched off. Note this is *not* fully rejected —
   the API is retained for local serving, which is why parity is three-way.
3. **Model in the browser** (chosen). No runtime infrastructure, works offline and
   indefinitely, and the parity fixture turns the port into a verifiable claim.

For data scope, the alternative to "exploit what is already there first" was to jump
straight to a second dataset. Rejected as sequencing: merging sources before exploiting the
present ones makes it impossible to attribute an accuracy change to either move.

## Consequences

- **Breaking change for P1, in two independent ways.** The package contract
  (`features.FEATURE_COLUMNS`) ships as **`v0.2.0`**, and P1 stays pinned to `v0.1.1` until
  a deliberate bump. Separately — and **not backward-compatible, contrary to this ADR's
  first draft** — W1's 422-on-unknown-province is a behavioural break in the *API* contract,
  which P1 was built against when the service accepted any string. It needs its own version
  bump on `api/main.py` and a note in P1's migration.
- **The feature spec becomes enforced, not decorative:** `load_model` validates the
  persisted spec against the code's, so a stale artifact fails loudly.
- **Published metrics are optimistic and must be regenerated** after W0's deduplication and
  a retrain. The site currently shows the contaminated numbers.
- **One model, not two.** Serving one model locally while shipping another to the browser
  would mean two answers to the same question.
- Licences of every new dataset are verified before publication, and the provenance chain —
  not just the tag — is recorded.
- The legal rationale in `docs/research/data-and-methodology.md` needs updating: Poland's
  September 2024 TDM amendment to the Database Protection Act changed the picture. The
  defensible line is that we do not **republish** third-party database extracts, not that
  scraping is categorically illegal.
