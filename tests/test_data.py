"""Tests for data loading and cleaning (pure transforms)."""

import pandas as pd

from car_price_ml import data


def test_add_age_is_reference_minus_year():
    df = pd.DataFrame({"year": [2014, 2020]})
    out = data.add_age(df, reference_year=2024)
    assert list(out["age"]) == [10, 4]
    assert "year" in df.columns  # input not mutated (age not added in place)
    assert "age" not in df.columns


def test_filter_outliers_drops_implausible_rows():
    df = pd.DataFrame({
        "price": [50000, 100, 50000, 50000, 50000],       # 100 below PRICE_MIN
        "mileage": [100000, 100000, 5_000_000, 100000, 100000],  # 5M above MILEAGE_MAX
        "vol_engine": [1600, 1600, 1600, 20000, 1600],    # 20000 above VOL_ENGINE_MAX
        "age": [10, 10, 10, 10, 50],                       # 50 above AGE_MAX
    })
    out = data.filter_outliers(df)
    assert len(out) == 1  # only the first row passes every rule


def test_clean_derives_age_and_filters():
    df = pd.DataFrame({
        "price": [50000, 100],
        "mileage": [100000, 100000],
        "vol_engine": [1600, 1600],
        "year": [2015, 2015],
    })
    out = data.clean(df)
    assert "age" in out.columns
    assert len(out) == 1  # the price=100 row is dropped


def test_filter_outliers_allows_ev_zero_engine():
    df = pd.DataFrame({
        "price": [80000], "mileage": [50000], "vol_engine": [0], "age": [3],
    })
    assert len(data.filter_outliers(df)) == 1  # EVs (vol_engine == 0) are kept
