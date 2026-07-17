"""Load and clean the used-car dataset.

Only ``load_raw`` touches disk; ``add_age``, ``filter_outliers`` and ``clean`` are pure
transforms on DataFrames, so they are unit-testable without the file.
"""

from __future__ import annotations

import pandas as pd

from car_price_ml import config


def load_raw(path=config.DATASET_CSV) -> pd.DataFrame:
    """Read the raw CSV and drop the unnamed index column."""
    df = pd.read_csv(path)
    return df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")


def add_age(df: pd.DataFrame, reference_year: int = config.REFERENCE_YEAR) -> pd.DataFrame:
    """Derive ``age = reference_year - year`` (causal quantity; avoids year drift/leakage)."""
    out = df.copy()
    out["age"] = reference_year - out["year"]
    return out


def filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop implausible rows by documented domain rules (price/mileage/vol/age)."""
    mask = (
        df["price"].between(config.PRICE_MIN, config.PRICE_MAX)
        & (df["mileage"].between(0, config.MILEAGE_MAX))
        & df["vol_engine"].between(config.VOL_ENGINE_MIN, config.VOL_ENGINE_MAX)
        & df["age"].between(0, config.AGE_MAX)
    )
    return df[mask].reset_index(drop=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning: derive age, then drop outliers."""
    return filter_outliers(add_age(df))


def load_clean(path=config.DATASET_CSV) -> pd.DataFrame:
    """Convenience: load the raw CSV and return the cleaned frame."""
    return clean(load_raw(path))
