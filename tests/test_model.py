"""Tests for the model bake-off, evaluation and persistence."""

import numpy as np
import pandas as pd

from car_price_ml import features, model


def _df(n=60):
    return pd.DataFrame({
        "mark": ["opel", "bmw", "audi"] * (n // 3),
        "model": ["combo", "x5", "a4"] * (n // 3),
        "fuel": ["Diesel", "Petrol", "Diesel"] * (n // 3),
        "province": ["Mazowieckie", "Śląskie", "Opolskie"] * (n // 3),
        "age": list(range(1, n + 1)),
        "mileage": [100000 + i * 2000 for i in range(n)],
        "vol_engine": [1600 + (i % 5) * 100 for i in range(n)],
        "price": [80000 - i * 500 for i in range(n)],
    })


def test_build_models_returns_three():
    assert set(model.build_models()) == {"Ridge", "RandomForest", "LightGBM"}


def test_evaluate_returns_all_metrics_in_pln():
    result = model.evaluate(pd.Series([100.0, 200.0, 300.0]), pd.Series([110.0, 190.0, 300.0]))
    assert set(result) == {"mae", "rmse", "mape", "r2"}
    assert result["mae"] > 0


def test_train_predicts_positive_prices():
    x, y = features.prepare(_df())
    fitted = model.train(x, y, name="Ridge")
    pred = fitted.predict(x.iloc[:3])
    assert len(pred) == 3
    assert all(p > 0 for p in pred)  # log-target guarantees positive prices


def test_save_load_roundtrip(tmp_path):
    x, y = features.prepare(_df())
    fitted = model.train(x, y, name="Ridge")
    path = model.save_model(fitted, {"k": 1}, models_dir=tmp_path)
    assert path.exists()

    bundle = model.load_model(models_dir=tmp_path)
    assert bundle["metadata"]["k"] == 1
    assert np.allclose(bundle["model"].predict(x.iloc[:3]), fitted.predict(x.iloc[:3]))
