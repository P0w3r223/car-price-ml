# CLAUDE.md — car-price-ml

Guidance for Claude Code (and any contributor) working in this repository.

## What this project is

A used-car price model for the Polish market: clean an open Kaggle dataset, engineer
features, compare models, and serve predictions via FastAPI. Portfolio project A3 — the
full ML cycle end-to-end.

## Architecture

```
src/car_price_ml/
  config.py     # dataset, paths, feature groups, outlier rules, model settings
  data.py       # load + clean (pure transforms; only load_raw touches disk)
  features.py   # preprocessing: age, target encoding (OOF), one-hot, log target
  model.py      # bake-off (linear/RF/LightGBM), k-fold CV, metrics, SHAP, persistence
api/            # FastAPI /predict service
notebooks/      # EDA + feature engineering
tests/          # pytest
docs/research/  # data + methodology
```

## Methodology rules (do not violate)

- **Log-price target.** Train on `log1p(price)`; **invert with `expm1` before every
  metric** (reporting metrics on the log scale is a silent, common mistake).
- **`age`, not raw `year`.** Derive `age = REFERENCE_YEAR - year`; raw year leaks/drifts.
- **Out-of-fold target encoding** for make/model (high cardinality). Fitting the encoder
  on the full dataset is the #1 leakage source — do it inside CV folds / a Pipeline.
- **k-fold cross-validation**, not a single split, for model selection (mean ± std).
- **SHAP (TreeExplainer)** for importance, not impurity `feature_importances_` (biased
  toward high-cardinality make/model).
- Filter outliers by **documented domain rules**; keep EVs (`vol_engine == 0`).

## Conventions

- English for code, comments, README, commits. Conventional Commits.
- No hardcoded values — configurable things live in `config.py`.
- Separate I/O from logic; pure functions are unit-tested.
- Interpreter: `.venv/Scripts/python.exe` (Python 3.12).

## How to run

```bash
.venv/Scripts/python -m pip install -r requirements.txt
kaggle datasets download -d aleksandrglotov/car-prices-poland -p data/raw --unzip
pytest
```
