"""Tests for feature preparation and the preprocessor."""

import pandas as pd
import pytest

from car_price_ml import config, features


def _df(n=40):
    return pd.DataFrame({
        "mark": ["opel", "bmw"] * (n // 2),
        "model": ["combo", "x5"] * (n // 2),
        "fuel": ["Diesel", "Petrol"] * (n // 2),
        "province": ["Mazowieckie", "Śląskie"] * (n // 2),
        "age": list(range(1, n + 1)),
        "mileage": [100000 + i * 1000 for i in range(n)],
        "vol_engine": [1600] * n,
        "price": [50000 - i * 300 for i in range(n)],
    })


def test_prepare_splits_features_and_target():
    x, y = features.prepare(_df())
    assert list(x.columns) == list(features.FEATURE_COLUMNS)
    assert y.name == config.TARGET
    assert len(x) == len(y)


def test_prepare_missing_columns_raises():
    with pytest.raises(KeyError):
        features.prepare(pd.DataFrame({"mark": ["opel"]}))


def test_preprocessor_returns_pandas_and_keeps_rows():
    x, y = features.prepare(_df())
    out = features.build_preprocessor().fit_transform(x, y)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(x)
    # target encoding collapses make/model to one numeric column each (prefixed names)
    assert any("mark" in c for c in out.columns)
    assert any("model" in c for c in out.columns)
