"""Model bake-off, cross-validation, metrics (PLN), SHAP, and persistence.

Every model is a Pipeline (leakage-safe preprocessing → regressor) wrapped in a
``TransformedTargetRegressor`` that trains on ``log1p(price)`` and inverts with ``expm1``,
so predictions and all metrics are in PLN. Model selection uses k-fold cross-validation
with out-of-fold predictions — never a single split.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline

from car_price_ml import config, features

MODEL_FILENAME = "car_price_model.joblib"


def _wrap(regressor) -> TransformedTargetRegressor:
    """Preprocessor + regressor, with a log1p/expm1 target transform (predicts PLN)."""
    pipe = Pipeline([("prep", features.build_preprocessor()), ("reg", regressor)])
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)


def build_models(random_state: int = config.RANDOM_STATE) -> dict[str, TransformedTargetRegressor]:
    """The bake-off: a linear baseline vs. two tree ensembles."""
    return {
        "Ridge": _wrap(Ridge(alpha=1.0)),
        "RandomForest": _wrap(
            RandomForestRegressor(
                n_estimators=200, min_samples_leaf=3, random_state=random_state, n_jobs=-1
            )
        ),
        "LightGBM": _wrap(
            LGBMRegressor(
                n_estimators=600, learning_rate=0.05, num_leaves=63,
                random_state=random_state, n_jobs=-1, verbose=-1,
            )
        ),
    }


def evaluate(y_true, y_pred) -> dict[str, float]:
    """MAE, RMSE (PLN), MAPE (%), R² — computed on the original PLN scale."""
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 1),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 1),
        "mape": round(float(mean_absolute_percentage_error(y_true, y_pred)) * 100, 2),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def cross_validate_models(
    x: pd.DataFrame, y: pd.Series, n_splits: int = config.CV_FOLDS,
    random_state: int = config.RANDOM_STATE,
) -> dict[str, dict[str, float]]:
    """k-fold CV with out-of-fold PLN predictions; returns metrics per model."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results: dict[str, dict[str, float]] = {}
    for name, model in build_models(random_state).items():
        oof_pred = cross_val_predict(model, x, y, cv=kf, n_jobs=-1)
        results[name] = evaluate(y, oof_pred)
    return results


def train(x: pd.DataFrame, y: pd.Series, name: str = "RandomForest",
          random_state: int = config.RANDOM_STATE) -> TransformedTargetRegressor:
    """Fit one model on all data (for serving). Defaults to the bake-off winner."""
    model = build_models(random_state)[name]
    model.fit(x, y)
    return model


def transformed_feature_names(fitted_model: TransformedTargetRegressor) -> list[str]:
    """Feature names after preprocessing (for SHAP / importance)."""
    prep = fitted_model.regressor_.named_steps["prep"]
    return [n.split("__", 1)[-1] for n in prep.get_feature_names_out()]


def shap_explanation(fitted_model: TransformedTargetRegressor, x_sample: pd.DataFrame):
    """Return ``(shap_values, transformed_X, feature_names)`` for a tree-model sample.

    SHAP TreeExplainer is preferred over impurity-based importance, which is biased toward
    high-cardinality make/model. ``tree_path_dependent`` avoids the slow interventional
    perturbation over a background set (much faster on random forests).
    """
    import shap

    prep = fitted_model.regressor_.named_steps["prep"]
    reg = fitted_model.regressor_.named_steps["reg"]
    x_trans = prep.transform(x_sample)
    explainer = shap.TreeExplainer(reg, feature_perturbation="tree_path_dependent")
    shap_values = explainer.shap_values(x_trans)
    return shap_values, x_trans, transformed_feature_names(fitted_model)


def save_model(model, metadata: dict | None = None, models_dir: Path = config.MODELS_DIR) -> Path:
    """Persist the trained model bundle (joblib)."""
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / MODEL_FILENAME
    joblib.dump({"model": model, "metadata": metadata or {}}, path)
    return path


def load_model(models_dir: Path = config.MODELS_DIR) -> dict:
    """Load the persisted model bundle."""
    path = models_dir / MODEL_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"No saved model at {path} — train first")
    return joblib.load(path)
