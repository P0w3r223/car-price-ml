"""Tests for the model bake-off, evaluation and persistence."""

import joblib
import numpy as np
import pandas as pd
import pytest

from car_price_ml import config, features, model


def _df(n=60):
    return pd.DataFrame({
        "mark": ["opel", "bmw", "audi"] * (n // 3),
        "model": ["combo", "x5", "a4"] * (n // 3),
        "fuel": ["Diesel", "Gasoline", "Diesel"] * (n // 3),
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
    assert bundle["metadata"]["features"] == list(features.FEATURE_COLUMNS)  # stamped on save
    assert np.allclose(bundle["model"].predict(x.iloc[:3]), fitted.predict(x.iloc[:3]))


def test_load_model_rejects_a_stale_feature_spec(tmp_path):
    """An artifact fit on other columns must fail at load, not at the first valuation.

    The API builds its input row from a hardcoded column list, so a mismatched artifact
    would go on returning confident numbers computed from the wrong features.
    """
    joblib.dump(
        {"model": object(), "metadata": {"features": ["mark", "mileage"]}},
        tmp_path / model.MODEL_FILENAME,
    )
    with pytest.raises(ValueError, match="was fit on"):
        model.load_model(models_dir=tmp_path)


def test_load_model_rejects_an_unstamped_bundle(tmp_path):
    joblib.dump({"model": object(), "metadata": {}}, tmp_path / model.MODEL_FILENAME)
    with pytest.raises(ValueError, match="no feature spec"):
        model.load_model(models_dir=tmp_path)


def _stamped(**overrides) -> dict:
    """Bundle metadata that passes every contract check unless a field is overridden."""
    return {
        "features": list(features.FEATURE_COLUMNS),
        "reference_year": config.REFERENCE_YEAR,
        "vocabulary": {
            "mark": ["opel"], "model": ["combo"],
            "fuel": list(config.KNOWN_FUELS), "province": list(config.PROVINCES),
        },
        **overrides,
    }


def test_load_model_rejects_a_bundle_without_a_vocabulary(tmp_path):
    """The serving guard consults this list, so an absent one lets unknown makes through."""
    metadata = _stamped()
    del metadata["vocabulary"]["mark"]
    joblib.dump({"model": object(), "metadata": metadata}, tmp_path / model.MODEL_FILENAME)

    with pytest.raises(ValueError, match="no mark vocabulary"):
        model.load_model(models_dir=tmp_path)


def test_load_model_rejects_a_widened_declared_domain(tmp_path):
    # Adding a province to config without retraining would have the API accept a value the
    # pickled encoder then raises on — a 500 at valuation time instead of a message at load.
    metadata = _stamped()
    metadata["vocabulary"]["province"] = list(config.PROVINCES)[:-1]
    joblib.dump({"model": object(), "metadata": metadata}, tmp_path / model.MODEL_FILENAME)

    with pytest.raises(ValueError, match="province domain"):
        model.load_model(models_dir=tmp_path)


def test_load_model_rejects_a_bundle_without_an_age_anchor(tmp_path):
    joblib.dump(
        {"model": object(), "metadata": {"features": list(features.FEATURE_COLUMNS)}},
        tmp_path / model.MODEL_FILENAME,
    )
    with pytest.raises(ValueError, match="no age anchor"):
        model.load_model(models_dir=tmp_path)


def test_load_model_rejects_a_different_age_anchor(tmp_path):
    """`age` is derived at inference, so a mismatched anchor shifts every prediction."""
    joblib.dump(
        {
            "model": object(),
            "metadata": {
                "features": list(features.FEATURE_COLUMNS),
                "reference_year": config.REFERENCE_YEAR - 2,
            },
        },
        tmp_path / model.MODEL_FILENAME,
    )
    with pytest.raises(ValueError, match="off by 2 years"):
        model.load_model(models_dir=tmp_path)
