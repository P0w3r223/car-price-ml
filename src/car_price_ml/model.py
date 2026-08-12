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


def _wrap(regressor, random_state: int = config.RANDOM_STATE) -> TransformedTargetRegressor:
    """Preprocessor + regressor, with a log1p/expm1 target transform (predicts PLN).

    The seed reaches the preprocessor too: the target encoder's cross-fitting is seeded, so
    leaving it on the default would make ``build_models(random_state=7)`` only half-reseeded.
    """
    pipe = Pipeline([("prep", features.build_preprocessor(random_state)), ("reg", regressor)])
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)


def build_models(random_state: int = config.RANDOM_STATE) -> dict[str, TransformedTargetRegressor]:
    """The bake-off: a linear baseline vs. two tree ensembles."""
    return {
        "Ridge": _wrap(Ridge(alpha=1.0), random_state),
        "RandomForest": _wrap(
            RandomForestRegressor(
                n_estimators=200, min_samples_leaf=3, random_state=random_state, n_jobs=-1
            ),
            random_state,
        ),
        # 1200 × 127 is the knee of a measured size↔quality curve (16 configurations, same
        # 5-fold CV): it reaches 8 612 PLN MAE, and every larger configuration tried costs
        # 2–4× the artifact for no improvement (2400 × 127 → 8 627, 4000 × 63 → 8 636). The
        # previous 600 × 63 was simply undertrained at 9 098, which is what made the earlier
        # "RandomForest wins, accept the size" conclusion look inevitable.
        "LightGBM": _wrap(
            LGBMRegressor(
                n_estimators=1200, learning_rate=0.05, num_leaves=127,
                random_state=random_state, n_jobs=-1, verbose=-1,
            ),
            random_state,
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
    """k-fold CV with out-of-fold PLN predictions; returns pooled metrics per model.

    Also reports the fold-to-fold spread of MAE. A pooled number alone cannot say whether
    one model beating another by a few hundred złoty is a real difference or fold noise —
    and this project already treats effects of that size as decisive, so the ranking needs
    an error bar to be worth anything. The spread is computed from the same out-of-fold
    predictions, so it costs no extra fits.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(kf.split(x))
    results: dict[str, dict[str, float]] = {}
    for name, model in build_models(random_state).items():
        # Outer CV is single-process: the estimators already parallelise internally
        # (n_jobs=-1), so nesting n_jobs here would oversubscribe the CPUs.
        oof_pred = cross_val_predict(model, x, y, cv=kf, n_jobs=None)
        per_fold = [
            mean_absolute_error(y.iloc[test_idx], oof_pred[test_idx]) for _, test_idx in folds
        ]
        results[name] = {
            **evaluate(y, oof_pred),
            "mae_fold_std": round(float(np.std(per_fold, ddof=1)), 1),
        }
    return results


def train(x: pd.DataFrame, y: pd.Series, name: str,
          random_state: int = config.RANDOM_STATE) -> TransformedTargetRegressor:
    """Fit one model on all data (for serving).

    ``name`` is required rather than defaulted: which model wins is a measured result, so a
    default here would be a second, silent answer to a question the bake-off already
    settles — and it would go stale the moment the measurement changed.
    """
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


def artifact_vocabulary(fitted_model: TransformedTargetRegressor) -> dict[str, list[str]]:
    """Every categorical value the fitted pipeline can accept, keyed by column.

    Read from the encoder rather than from the training frame, so it is by construction the
    set the artifact can price. Everything outside it encodes to the global target mean —
    a plausible number for a car the model has never seen — so the serving path needs this
    list to refuse instead of guess.
    """
    transformers = fitted_model.regressor_.named_steps["prep"].named_transformers_
    learned = zip(config.HIGH_CARD_CATEGORICAL, transformers["target_enc"].categories_)
    declared = zip(config.LOW_CARD_CATEGORICAL, transformers["onehot"].categories_)
    return {
        column: [str(v) for v in categories]
        for column, categories in [*learned, *declared]
    }


def save_model(model, metadata: dict | None = None, models_dir: Path = config.MODELS_DIR) -> Path:
    """Persist the trained model bundle (joblib), stamped with the feature spec it was fit on.

    The spec is written here rather than by the caller so it cannot be forgotten — it is
    what ``load_model`` checks the code against.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / MODEL_FILENAME
    bundle_metadata = {
        **(metadata or {}),
        "features": list(features.FEATURE_COLUMNS),
        "reference_year": config.REFERENCE_YEAR,
        "vocabulary": artifact_vocabulary(model),
    }
    joblib.dump({"model": model, "metadata": bundle_metadata}, path)
    return path


def load_model(models_dir: Path = config.MODELS_DIR) -> dict:
    """Load the persisted model bundle, refusing one that was fit on a different feature set.

    Uses joblib (pickle) — only load model files you produced/trust; the served artifact
    ships inside the Docker image, so its provenance is controlled.

    The feature check exists because the failure it prevents is silent: the API builds its
    input row from a hardcoded column list, so an artifact trained on a different set would
    keep returning confident numbers computed from the wrong columns. A stale artifact must
    fail at load, not at the first valuation.
    """
    path = models_dir / MODEL_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"No saved model at {path} — train first")

    bundle = joblib.load(path)
    saved = bundle.get("metadata", {}).get("features")
    expected = list(features.FEATURE_COLUMNS)
    if saved is None:
        raise ValueError(
            f"{path} carries no feature spec — it predates the check and cannot be "
            f"verified against the current one ({expected}). Retrain."
        )
    if list(saved) != expected:
        raise ValueError(
            f"{path} was fit on {list(saved)}, but the code expects {expected}. Retrain."
        )

    # The anchor belongs to the inference contract as much as the column list does: the API
    # derives `age = REFERENCE_YEAR - year`, so serving an artifact trained against a
    # different anchor shifts every age by the difference — quietly, and in a direction the
    # metrics never see.
    saved_year = bundle.get("metadata", {}).get("reference_year")
    if saved_year is None:
        raise ValueError(
            f"{path} carries no age anchor, so it cannot be checked against "
            f"REFERENCE_YEAR={config.REFERENCE_YEAR}. Retrain."
        )
    if saved_year != config.REFERENCE_YEAR:
        raise ValueError(
            f"{path} was fit with REFERENCE_YEAR={saved_year}, but the code uses "
            f"{config.REFERENCE_YEAR} — every derived age would be off by "
            f"{abs(saved_year - config.REFERENCE_YEAR)} years. Retrain."
        )

    # The declared one-hot domains are part of the artifact too. Widening `config.PROVINCES`
    # without retraining would leave the API validator accepting a province the pickled
    # encoder then refuses — a 500 at valuation time instead of a clear message at load.
    saved_vocabulary = bundle.get("metadata", {}).get("vocabulary", {})
    declared_domains = {"fuel": list(config.KNOWN_FUELS), "province": list(config.PROVINCES)}
    for column in (*config.HIGH_CARD_CATEGORICAL, *config.LOW_CARD_CATEGORICAL):
        stamped = saved_vocabulary.get(column)
        # Mandatory, like the two checks above. The serving path refuses an unknown make by
        # consulting this list, so an artifact without it would let that guard pass silently
        # — the failure mode being guarded against, reintroduced by an optional check.
        if not stamped:
            raise ValueError(f"{path} carries no {column} vocabulary. Retrain.")
        declared = declared_domains.get(column)
        if declared is not None and sorted(stamped) != sorted(declared):
            raise ValueError(
                f"{path} was fit with {column} domain {sorted(stamped)}, but the code "
                f"declares {sorted(declared)}. Retrain."
            )
    return bundle
