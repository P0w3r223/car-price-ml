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
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

from car_price_ml import config

FEATURE_COLUMNS = (
    *config.HIGH_CARD_CATEGORICAL,  # mark, model
    *config.LOW_CARD_CATEGORICAL,   # fuel, province
    *config.NUMERIC_FEATURES,       # age, mileage, vol_engine
)


def build_preprocessor(random_state: int = config.RANDOM_STATE) -> ColumnTransformer:
    """Column transformer: OOF target encoding + one-hot + passthrough numerics.

    ``TargetEncoder`` performs internal cross-fitting; pandas output preserves feature
    names downstream (for SHAP and to satisfy LightGBM).

    The cross-fitting splitter is passed explicitly **with a seed**. Handing ``cv`` a
    plain integer lets the encoder shuffle the folds from an unseeded RNG, so two fits on
    identical data return different encodings — and every metric downstream inherits that
    wobble even when the estimator itself is fully seeded. Measured on the full dataset,
    the spread was tens of złoty of MAE: small, but the same order as a real model
    improvement, which is exactly the size that misleads a comparison.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "target_enc",
                TargetEncoder(
                    target_type="continuous",
                    cv=KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=random_state),
                ),
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
