"""Reproducible training entrypoint for the served model.

    python -m car_price_ml.train

Runs the k-fold bake-off, trains the winner (RandomForest) on all data, and saves it
with the full CV metrics as metadata — so the deployed model is reproducible from the
repo and the report/site can render metrics from the artifact rather than hardcoding them.
"""

from __future__ import annotations

import json

from car_price_ml import config, data, features, model

BEST_MODEL = "RandomForest"


def train_and_save(model_name: str = BEST_MODEL):
    """Bake-off (CV) → train the winner on all data → persist with metrics metadata."""
    df = data.load_clean()
    x, y = features.prepare(df)
    print(f"[train] {len(x):,} rows — running {config.CV_FOLDS}-fold bake-off ...", flush=True)
    cv = model.cross_validate_models(x, y)
    print("[train] CV (PLN):")
    print(json.dumps(cv, indent=2))

    best = model.train(x, y, name=model_name)
    path = model.save_model(
        best,
        metadata={
            "model": model_name,
            "cv_metrics": cv[model_name],
            "cv_all": cv,
            "n_train": len(x),
            "features": list(features.FEATURE_COLUMNS),
        },
    )
    print(f"[train] saved {model_name} -> {path}")
    return path


if __name__ == "__main__":
    train_and_save()
