"""Reproducible training entrypoint for the served model.

    python -m car_price_ml.train

Runs the k-fold bake-off, trains whichever model won it on all data, and saves it with the
full CV metrics as metadata — so the deployed model is reproducible from the repo and the
report/site can render metrics from the artifact rather than hardcoding them.
"""

from __future__ import annotations

import json

from car_price_ml import config, data, features, model

# Optional override. Left as None, the bake-off decides — the site publishes a "model
# comparison" table, so training a model the comparison did not choose would make that table
# decorative. Set a name to pin one deliberately; the mismatch is then logged, not hidden.
BEST_MODEL: str | None = None


def _check_vintage(df) -> None:
    """Refuse to train if the data is newer than the anchor `age` is derived from.

    ``REFERENCE_YEAR`` is pinned to the source snapshot's vintage. Point this at a newer
    dataset without moving the anchor and every derived `age` is silently wrong — negative
    for the newest cars — so the mismatch has to stop the run rather than be discovered in
    the metrics.

    Must run on the **raw** frame. ``clean`` keeps only ``age`` in ``[0, AGE_MAX]``, which is
    exactly ``year <= REFERENCE_YEAR``, so by the time cleaning is done the evidence is gone:
    a newer snapshot would have its newest adverts silently deleted and this check would
    report nothing. (It did, until a review caught it.)
    """
    if df.empty:
        return  # nothing to date; the caller's own emptiness check owns this case
    newest = int(df["year"].max())
    if newest > config.REFERENCE_YEAR:
        raise ValueError(
            f"data contains model year {newest} but REFERENCE_YEAR is "
            f"{config.REFERENCE_YEAR} — update the anchor to this snapshot's vintage"
        )


def _select(cv: dict[str, dict[str, float]], override: str | None) -> str:
    """Lowest cross-validated MAE wins, unless a name is pinned deliberately."""
    measured = min(cv, key=lambda name: cv[name]["mae"])
    if override and override not in cv:
        raise ValueError(f"pinned model {override!r} is not in the bake-off: {sorted(cv)}")
    if override and override != measured:
        print(
            f"[train] NOTE: training pinned {override} "
            f"(MAE {cv[override]['mae']:,.0f}) although {measured} "
            f"won the bake-off (MAE {cv[measured]['mae']:,.0f})"
        )
    return override or measured


def train_and_save(model_name: str | None = BEST_MODEL):
    """Bake-off (CV) → train the winner on all data → persist with metrics metadata."""
    raw = data.load_raw()
    _check_vintage(raw)
    df = data.clean(raw)
    x, y = features.prepare(df)
    print(f"[train] {len(x):,} rows — running {config.CV_FOLDS}-fold bake-off ...", flush=True)
    cv = model.cross_validate_models(x, y)
    print("[train] CV (PLN):")
    print(json.dumps(cv, indent=2))

    chosen = _select(cv, model_name)
    print(f"[train] training {chosen} (MAE {cv[chosen]['mae']:,.0f})", flush=True)
    best = model.train(x, y, name=chosen)
    path = model.save_model(
        best,
        metadata={
            "model": chosen,
            "cv_metrics": cv[chosen],
            "cv_all": cv,
            "n_train": len(x),
        },  # the feature spec is stamped by save_model, so it cannot drift from the code
    )
    print(f"[train] saved {chosen} -> {path}")
    return path


if __name__ == "__main__":
    train_and_save()
