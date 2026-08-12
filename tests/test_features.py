"""Tests for feature preparation and the preprocessor."""

import pandas as pd
import pytest

from car_price_ml import config, features


def _df(n=40):
    return pd.DataFrame({
        "mark": ["opel", "bmw"] * (n // 2),
        "model": ["combo", "x5"] * (n // 2),
        "fuel": ["Diesel", "Gasoline"] * (n // 2),
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


def test_preprocessor_is_reproducible():
    """Two fits on identical data must agree exactly.

    Regression guard: with an unseeded cross-fitting splitter the target encoder returned
    different encodings run to run, which moved every downstream metric by tens of złoty
    of MAE — enough to fake (or hide) a model improvement.
    """
    x, y = features.prepare(_df())

    first = features.build_preprocessor().fit_transform(x, y)
    second = features.build_preprocessor().fit_transform(x, y)

    pd.testing.assert_frame_equal(first, second)


def test_onehot_columns_cover_the_declared_domain():
    """The feature space is the declared vocabulary, not whatever the sample contained.

    A training frame holding one province must still produce a column per province —
    otherwise a fold (or the final fit) would define a narrower feature space than the API
    later feeds it.
    """
    df = _df()
    df["province"] = "Mazowieckie"
    x, y = features.prepare(df)

    out = features.build_preprocessor().fit_transform(x, y)

    for province in config.PROVINCES:
        assert any(c.endswith(f"province_{province}") for c in out.columns), province
    for fuel in config.KNOWN_FUELS:
        assert any(c.endswith(f"fuel_{fuel}") for c in out.columns), fuel


def test_preprocessor_refuses_an_out_of_domain_category():
    """An unknown category must raise, not become an all-zero block.

    Under `handle_unknown="ignore"` this row would have been priced with no fuel type at
    all, returned as a confident number — the failure mode that motivated declaring the
    domains in the first place.
    """
    df = _df()
    df.loc[0, "fuel"] = "Petrol"  # a real spelling from another dataset
    x, y = features.prepare(df)

    with pytest.raises(ValueError, match="Found unknown categories"):
        features.build_preprocessor().fit_transform(x, y)


def test_preprocessor_returns_pandas_and_keeps_rows():
    x, y = features.prepare(_df())
    out = features.build_preprocessor().fit_transform(x, y)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(x)
    # target encoding collapses make/model to one numeric column each (prefixed names)
    assert any("mark" in c for c in out.columns)
    assert any("model" in c for c in out.columns)
