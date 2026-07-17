"""Feature engineering: preprocessing that is leakage-safe by construction.

High-cardinality make/model use scikit-learn's ``TargetEncoder``, which does **internal
cross-fitting** during ``fit_transform`` — so target encoding never leaks, even inside a
plain Pipeline. Low-cardinality categoricals are one-hot encoded; numeric features pass
through. The log transform of the (right-skewed) price target is applied at the model
level via ``TransformedTargetRegressor`` (see model.py), not here.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

from car_price_ml import config

FEATURE_COLUMNS = (
    *config.HIGH_CARD_CATEGORICAL,  # mark, model
    *config.LOW_CARD_CATEGORICAL,   # fuel, province
    *config.NUMERIC_FEATURES,       # age, mileage, vol_engine
)


def build_preprocessor() -> ColumnTransformer:
    """Column transformer: OOF target encoding + one-hot + passthrough numerics.

    ``TargetEncoder(cv=...)`` performs internal cross-fitting; pandas output preserves
    feature names downstream (for SHAP and to satisfy LightGBM).
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "target_enc",
                TargetEncoder(target_type="continuous", cv=config.CV_FOLDS),
                list(config.HIGH_CARD_CATEGORICAL),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(config.LOW_CARD_CATEGORICAL),
            ),
            ("num", "passthrough", list(config.NUMERIC_FEATURES)),
        ]
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned frame into X (model features) and y (price target)."""
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"missing feature columns: {missing} (run data.clean first)")
    return df[list(FEATURE_COLUMNS)].copy(), df[config.TARGET].copy()
