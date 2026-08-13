# car-price-ml

[![CI](https://github.com/P0w3r223/car-price-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/P0w3r223/car-price-ml/actions/workflows/ci.yml)

**Used-car price prediction for the Polish market** — a model that refuses the cars it cannot
price, rather than answering with a plausible number.

> Portfolio project A3: the full ML cycle end to end — EDA, features, a measured model
> bake-off, SHAP, a FastAPI service and a published page. What the project is actually
> *about* is the paragraph below.

## The problem it is built around

A price model's worst output is not a crash — it is a confident wrong number. Every bug this
project has shipped had one shape: an input the model could not handle was quietly
substituted with something plausible.

- An unseen make reaches `TargetEncoder`, which answers with the **global target mean**. So
  `ferrari`/`f40` and `zzzz`/`qqqq` came back as the same price — 34 093 PLN, with a 200.
- An unseen province reaches a one-hot encoder, which emits an **all-zero row**. The web form
  sent `Kujawsko-Pomorskie` where the data spells it `Kujawsko-pomorskie`, so ~7 % of the
  market was valued as if the car had no location (measured: 1 376 PLN mean deviation, p95
  6 344, max 42 533).
- A diesel with `vol_engine = 0` was priced **11 % above** the same car with an engine,
  because zero displacement in this dataset means "missing" — except on an EV, where it is a
  fact.

So the domains are closed and declared in `config.py`, the encoders raise instead of ignoring,
the make/model vocabulary is **stamped into the artifact** and enforced at the API boundary,
and the artifact carries its own contract — feature spec, age anchor, vocabularies — which
`load_model` refuses if it disagrees with the code. A spelling variant is normalised; a
different word is refused. Those are not the same thing.

The [published page](https://p0w3r223.github.io/car-price-ml/) leads with that table, and
every "without the guard" number in it is produced by **running the pipeline with the guard
removed** at export time — so the table cannot outlive the guards it documents.

## What it does

1. **Data** — an open Kaggle dataset of ~118k Polish used-car adverts (Otomoto-sourced,
   **CC0**), cleaned with documented domain rules: 9.8 % exact duplicate adverts removed
   (with shuffled k-fold, the same advert otherwise lands in train *and* test), combustion
   rows with no displacement dropped while EVs are kept, adverts from outside the 16 Polish
   provinces dropped. 117 927 → **111 018 rows**.
2. **Feature engineering** — `age` derived from a fixed anchor (the snapshot's own vintage,
   not "today"), out-of-fold **target encoding** for high-cardinality make/model, one-hot over
   **declared** domains for the rest; **log-price** target, inverted before every metric.
3. **Models** — a bake-off: Ridge baseline vs. RandomForest vs. LightGBM, **5-fold CV** over
   pooled out-of-fold predictions, each MAE reported with its fold-to-fold spread because a
   gap smaller than the spread is not a better model. The winner is selected by the
   measurement rather than hardcoded, and LightGBM's hyper-parameters come from a published
   size↔quality curve: **8 612 ± 72 PLN MAE in a 14 MB artifact**, against RandomForest's
   8 798 ± 81 in 590 MB.
4. **Interpretability** — SHAP (TreeExplainer), reported per input column and labelled as a
   **log-price contribution**: the explainer runs on the regressor inside the
   `TransformedTargetRegressor`, so the values sum to `log1p(prediction)` and are not złoty.
5. **Serve** — a FastAPI `/predict` (+ `GET /vocabulary`, Docker) that answers an
   out-of-domain input with a `422` and an unservable artifact with a `503`, and dates every
   valuation to the market it is a valuation of.

## Data source

[Car Prices Poland](https://www.kaggle.com/datasets/aleksandrglotov/car-prices-poland)
(aleksandrglotov, Kaggle) — license **CC0-1.0**. An open, published dataset (no scraping):
`mark`, `model`, `year`, `mileage`, `vol_engine`, `fuel`, `city`, `province`, `price`.
The source is a **single January 2022 snapshot** (confirmed by Kaggle's metadata and by the
model-year distribution: 2021 is 9 % of rows, 2022 is 1.8 %, nothing follows), so valuations
are historical and dated as such rather than presented as current. See
[`docs/research/data-and-methodology.md`](docs/research/data-and-methodology.md).

Download (needs a Kaggle account/token):

```bash
kaggle datasets download -d aleksandrglotov/car-prices-poland -p data/raw --unzip
```

## Live site

**<https://p0w3r223.github.io/car-price-ml/>** — what the model refuses to price (measured by
running the pipeline with its guards removed), the bake-off with its fold spreads, the SHAP
drivers, and the cleaning and contract rules quoted verbatim from the modules they live in.

Notebook: [`notebooks/01_eda_and_model.ipynb`](notebooks/01_eda_and_model.ipynb).

**Note.** The page is rendered from `docs/data/*.json`, which are committed; producing those
needs the trained model + dataset (both kept out of git/CI), so `python -m
car_price_ml.site.export` runs locally. CI runs the test suite **and** rebuilds the page and
the form's generated files, failing if any of them no longer matches its inputs.

## Project structure

```
src/car_price_ml/       # config, data, features, model, train, figures
src/car_price_ml/site/  # export the page's numbers, then render the page and the form's inputs
notebooks/              # EDA + feature engineering
api/                    # FastAPI service (also serves the web form, same-origin)
tests/                  # pytest — 131 tests
docs/data/              # the committed aggregates the published page renders from
docs/app/               # vanilla-JS valuation form; config.json + styles.css are generated
docs/adr/               # design decisions, each with the measurements behind it
docs/research/          # data sources, methodology, temporal recalibration
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt      # Windows
pytest

# download the dataset (needs a Kaggle account/token), then train the served model
kaggle datasets download -d aleksandrglotov/car-prices-poland -p data/raw --unzip
python -m car_price_ml.train    # bake-off, then train + save whichever model won it

# serve it
uvicorn api.main:app --reload    # POST /predict   (or: docker build -t car-price-ml . && docker run -p 8000:8000 car-price-ml)

# republish the site: measure, then render the page + the form's generated files
# (the `[site]` extra installs Jinja2)
python -m car_price_ml.site.export
python -m car_price_ml.site.build
```

## Web frontend

A dependency-free **vanilla-JavaScript** valuation form (`docs/app/`) that `fetch`es the
`/predict` endpoint and renders the price with client-side validation and error handling.
`fuel` and `province` are **closed vocabularies**: the service normalises spelling and
casing but answers anything outside the domain with a `422`, rather than pricing the car
from an all-zero category block. The form validates against no copy of them — every
vocabulary and bound it checks is generated from `config.py` into `docs/app/config.json`, and
without that file the form refuses to run rather than falling back to constants of its own.
When the API is running it is served at the site root — same origin as `/predict`, so no
CORS — giving **real model predictions** in the browser:

```bash
uvicorn api.main:app        # then open http://localhost:8000/
```

An interactive copy also lives on GitHub Pages:
**<https://p0w3r223.github.io/car-price-ml/app/>**. Because the model API is not publicly
hosted, that online demo answers with an **offline heuristic** — a formula in `app.js`, not
the trained model. The form says so *above* the input fields, before a price exists: it probes
the service at load and reports which of three things is true (a model is answering, the
service is up with no model, or no API is reachable). A heuristic answer is then also shaped
differently from a prediction — dashed border, lighter, approximate figure — rather than
carrying a differently-coloured badge on the same card.

## Methodology notes

Each of these exists because its absence produced a wrong number, not because it is good
practice in the abstract. The full reasoning, with measurements, is in
[`docs/adr/`](docs/adr/).

- **Log-price target**, inverted with `expm1` before every metric. Reporting metrics on the
  log scale is the classic silent error here.
- **`age` from a fixed anchor**, not raw year and not "today": training and inference must
  derive it identically, and the anchor is the snapshot's own vintage (2022). Anchored later,
  the API happily accepts a model year the model has never seen.
- **Out-of-fold target encoding** inside the pipeline with a seeded splitter. Fitting the
  encoder on the full dataset is the primary leakage source in this design.
- **k-fold CV with pooled out-of-fold predictions**, reported with the fold-to-fold spread.
- **Closed, declared domains** with `handle_unknown="error"`; the make/model vocabulary read
  off the fitted encoder, stamped into the artifact, served at `GET /vocabulary` and enforced
  at the boundary.
- **The artifact carries its own contract.** A stale bundle fails at load, not at the first
  valuation.
- **Valuations are dated.** Polish used-car prices moved roughly −10 % to −24 % after this
  snapshot while general inflation ran +41 %, so an undated figure invites an assumption that
  is both large and signed the wrong way —
  [`docs/research/temporal-recalibration.md`](docs/research/temporal-recalibration.md).

## License

MIT (code). Data © the Kaggle dataset author under CC0-1.0.
