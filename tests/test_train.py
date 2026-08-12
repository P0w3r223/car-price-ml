"""Tests for the training entrypoint.

This module had no coverage at all, which is how a vintage guard that could never fire got
committed: it was checked against the cleaned frame, and cleaning had already dropped every
row that would have tripped it.
"""

import pandas as pd
import pytest

from car_price_ml import config, data, train


def _raw(year: int) -> pd.DataFrame:
    return pd.DataFrame([{
        "mark": "opel", "model": "combo", "fuel": "Diesel", "province": "Mazowieckie",
        "year": year, "mileage": 100000, "vol_engine": 1600, "price": 50000,
    }])


def test_check_vintage_accepts_data_at_the_anchor():
    assert train._check_vintage(_raw(config.REFERENCE_YEAR)) is None


def test_check_vintage_rejects_data_newer_than_the_anchor():
    with pytest.raises(ValueError, match="update the anchor"):
        train._check_vintage(_raw(config.REFERENCE_YEAR + 1))


def test_check_vintage_must_run_before_cleaning():
    """The guard is only reachable on the raw frame — this pins the ordering.

    `clean` keeps `age` in [0, AGE_MAX], i.e. `year <= REFERENCE_YEAR`, so a newer snapshot
    is silently truncated and the check sees nothing to complain about.
    """
    newer = _raw(config.REFERENCE_YEAR + 1)
    cleaned = data.clean(newer)

    assert cleaned.empty  # the evidence is gone
    assert train._check_vintage(cleaned) is None  # ...so the guard cannot fire here
    with pytest.raises(ValueError):
        train._check_vintage(newer)  # ...but it does on the raw frame


def test_select_prefers_the_lowest_cross_validated_mae():
    cv = {"Ridge": {"mae": 15000.0}, "RandomForest": {"mae": 8800.0}, "LightGBM": {"mae": 8803.0}}
    assert train._select(cv, override=None) == "RandomForest"


def test_select_honours_an_explicit_pin_and_says_so(capsys):
    cv = {"RandomForest": {"mae": 8800.0}, "LightGBM": {"mae": 8803.0}}

    assert train._select(cv, override="LightGBM") == "LightGBM"

    # A deliberate choice against the measurement has to be visible in the run log.
    out = capsys.readouterr().out
    assert "pinned LightGBM" in out and "RandomForest" in out
