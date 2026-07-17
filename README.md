# car-price-ml

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
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
pytest
```

## Methodology highlights

- **Log-price target** (invert before metrics), **`age`** not raw year, **out-of-fold
  target encoding** (no leakage), **k-fold CV**, metrics in PLN (MAE/RMSE/MAPE/R²), SHAP.

## License

MIT (code). Data © the Kaggle dataset author under CC0-1.0.
