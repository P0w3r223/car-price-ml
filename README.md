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
   **k-fold cross-validation** and metrics reported in PLN.
4. **Interpretability** — SHAP (TreeExplainer) to explain what drives a valuation.
5. **Serve** — a FastAPI `/predict` endpoint (+ Docker) loading the saved best model.

## Data source

[Car Prices Poland](https://www.kaggle.com/datasets/aleksandrglotov/car-prices-poland)
(aleksandrglotov, Kaggle) — license **CC0-1.0**. An open, published dataset (no scraping):
`mark`, `model`, `year`, `mileage`, `vol_engine`, `fuel`, `city`, `province`, `price`.
Prices are ~2021–2023, so the model is historically biased (documented). See
[`docs/research/data-and-methodology.md`](docs/research/data-and-methodology.md).

Download (needs a Kaggle account/token):

```bash
kaggle datasets download -d aleksandrglotov/car-prices-poland -p data/raw --unzip
```

## Live site

Mini report (bake-off, SHAP, example valuation): **<https://p0w3r223.github.io/car-price-ml/>**

Notebook: [`notebooks/01_eda_and_model.ipynb`](notebooks/01_eda_and_model.ipynb).

**Note.** The site is built locally from the trained model + dataset (both kept out of
git/CI); CI runs the test suite. Regenerate the model with the download + training
commands above.

## Project structure

```
src/car_price_ml/   # config, data, features, model
notebooks/          # EDA + feature engineering
api/                # FastAPI service
tests/              # pytest
docs/research/      # data + methodology
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt      # Windows
pytest

# download the dataset (needs a Kaggle account/token), then train the served model
kaggle datasets download -d aleksandrglotov/car-prices-poland -p data/raw --unzip
python -m car_price_ml.train    # bake-off + train RandomForest into models/

# serve it
uvicorn api.main:app --reload    # POST /predict   (or: docker build -t car-price-ml . && docker run -p 8000:8000 car-price-ml)
```

## Methodology highlights

- **Log-price target** (invert before metrics), **`age`** not raw year, **out-of-fold
  target encoding** (no leakage), **k-fold CV**, metrics in PLN (MAE/RMSE/MAPE/R²), SHAP.

## License

MIT (code). Data © the Kaggle dataset author under CC0-1.0.
