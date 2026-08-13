# car-price-ml

[![CI](https://github.com/P0w3r223/car-price-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/P0w3r223/car-price-ml/actions/workflows/ci.yml)

**Used-car price prediction for the Polish market** — from data, through feature
engineering and a model bake-off, to a FastAPI prediction service.

> Portfolio project A3. Demonstrates the full ML cycle asked about in every interview:
> EDA → features → model comparison → evaluation → deployment. Uses **Polish** data, which
> sets it apart from the thousands of Kaggle clones.

## What it does

1. **Data** — an open Kaggle dataset of ~118k Polish used-car adverts (Otomoto-sourced,
   **CC0**); cleaned with documented outlier rules.
2. **Feature engineering** — `age` (not raw year), out-of-fold **target encoding** for
   high-cardinality make/model, one-hot for low-cardinality categoricals; **log-price**
   target (right-skewed).
3. **Models** — a bake-off: linear/Ridge baseline vs. RandomForest vs. LightGBM, with
   **k-fold cross-validation** and metrics reported in PLN — each with its fold-to-fold
   spread, because a gap smaller than the spread is not a better model. The winner is
   selected by the measurement, not hardcoded, and LightGBM's hyperparameters come from a
   published size↔quality curve: 8 612 PLN MAE in a 14 MB artifact, against RandomForest's
   8 798 in 590 MB.
4. **Interpretability** — SHAP (TreeExplainer) to explain what drives a valuation.
5. **Serve** — a FastAPI `/predict` endpoint (+ Docker) loading the saved best model.

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
src/car_price_ml/   # config, data, features, model
src/car_price_ml/site/  # export the page's numbers, then render the page from them
notebooks/          # EDA + feature engineering
api/                # FastAPI service (also serves the web form)
tests/              # pytest
docs/data/          # the committed aggregates the published page renders from
docs/app/           # vanilla-JS valuation form (served by the API + on Pages)
docs/research/      # data + methodology
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
from an all-zero category block. The form holds no copy of them — every vocabulary and bound
it validates against is generated from `config.py` into `docs/app/config.json`, and without
that file the form refuses to run rather than falling back to constants of its own.
When the API is running it is served at the site root — same origin as `/predict`, so no
CORS — giving **real model predictions** in the browser:

```bash
uvicorn api.main:app        # then open http://localhost:8000/
```

An interactive copy also lives on GitHub Pages:
**<https://p0w3r223.github.io/car-price-ml/app/>**. Because the model API is not publicly
hosted, that online demo falls back to a clearly-labelled **offline heuristic** (a rough
estimate, *not* the trained model) — run the API locally for the real prediction.

## Methodology highlights

- **Log-price target** (invert before metrics), **`age`** not raw year, **out-of-fold
  target encoding** (no leakage), **k-fold CV**, metrics in PLN (MAE/RMSE/MAPE/R²), SHAP.

## License

MIT (code). Data © the Kaggle dataset author under CC0-1.0.
